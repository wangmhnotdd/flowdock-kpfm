import glob
import os
import random
import tempfile
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import rootutils
import torch
from beartype.typing import Any, Callable, Dict
from plinder.core.index.system import PlinderSystem
from plinder.core.structure.structure import Structure
from rdkit import Chem
from torch.utils.data import Dataset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.data.components.mol_features import process_molecule
from flowdock.utils import RankedLogger
from flowdock.utils.data_utils import (
    centralize_complex_graph,
    combine_molecules,
    merge_protein_and_ligands,
    pdb_filepath_to_protein,
    process_protein,
)
from flowdock.utils.model_utils import extract_esm_embeddings, sample_inplace_to_torch

log = RankedLogger(__name__, rank_zero_only=False)


def flowdock_structure_featurizer(
    apo_structure: Structure,
    holo_structure: Structure,
    chain_mapping: Dict[str, str],
    esm_model: esm.ESM2,
    esm_alphabet: esm.data.Alphabet,
    esm_batch_converter: esm.data.BatchConverter,
    esm_repr_layer: int,
    n_lig_patches: int,
) -> dict[str, Any]:
    """Featurize Plinder apo and holo `Structure` objects and return a dictionary of features for
    FlowDock.

    :param apo_structure: The apo `Structure` object to featurize.
    :param holo_structure: The holo `Structure` object to featurize.
    :param chain_mapping: A dictionary mapping chain IDs from the holo structure to the apo structure.
    :param esm_model: The ESM model to use for protein sequence featurization.
    :param esm_alphabet: The ESM alphabet to use for protein sequence featurization.
    :param esm_batch_converter: The ESM batch converter to use for protein sequence featurization.
    :param esm_repr_layer: The ESM representation layer to use for protein sequence featurization.
    :param n_lig_patches: The number of ligand patches to use.
    :return: A dictionary of features for FlowDock.
    """
    try:
        # match holo protein to apo protein
        apo_structure.set_chain(holo_structure.protein_chain_ordered[0])
        holo_structure, apo_structure = holo_structure.align_common_sequence(apo_structure)
        apo_structure, apo_structure_raw_rmsd, apo_structure_refined_rmsd = (
            apo_structure.superimpose(holo_structure)
        )

        with (
            tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp_apo_pdb,
            tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp_holo_pdb,
        ):
            apo_protein_pdb_filepath = tmp_apo_pdb.name
            holo_protein_pdb_filepath = tmp_holo_pdb.name

            map_chain_id = np.vectorize(lambda x: chain_mapping.get(x, x.split(".")[-1]))

            apo_structure.protein_atom_array.chain_id = map_chain_id(
                apo_structure.protein_atom_array.chain_id
            )
            holo_structure.protein_atom_array.chain_id = map_chain_id(
                holo_structure.protein_atom_array.chain_id
            )

            apo_structure.save_to_disk(Path(apo_protein_pdb_filepath))
            holo_structure.save_to_disk(Path(holo_protein_pdb_filepath))

        # process ligands
        ligands = [
            Chem.MolFromMolFile(ligand_sdf) for ligand_sdf in holo_structure.ligand_sdfs.values()
        ]
        lig_samples = [
            process_molecule(
                ligand,
                ref_conf_xyz=np.array(ligand.GetConformer().GetPositions()),
                return_as_dict=True,
            )
            for ligand in ligands
        ]
        for lig_sample in lig_samples:
            lig_sample["metadata"]["sample_ID"] = holo_structure.id

        # process holo protein
        holo_af_protein = pdb_filepath_to_protein(holo_protein_pdb_filepath)
        holo_protein_sample = process_protein(
            holo_af_protein,
            sample_name=f"{holo_structure.id}_",
        )
        complex_graph = holo_protein_sample

        if np.isnan(complex_graph["features"]["res_atom_positions"]).any():
            raise ValueError(
                f"NaN values found in holo protein sample for system {holo_structure.id}"
            )

        # process apo protein
        apo_af_protein = pdb_filepath_to_protein(apo_protein_pdb_filepath)
        if not np.array_equal(
            np.int_(holo_af_protein.aatype),
            np.int_(apo_af_protein.aatype),
        ):
            raise ValueError(
                f"Apo and holo protein sequences do not match for system {holo_structure.id}"
            )

        # embed apo protein sequence with ESM2
        sequences = [
            "".join(np.array(list(chain_seq))[chain_mask])
            for (_, chain_seq, chain_mask) in apo_af_protein.letter_sequences
        ]
        esm_embeddings = extract_esm_embeddings(
            esm_model,
            esm_alphabet,
            esm_batch_converter,
            sequences,
            device="cpu",
            esm_repr_layer=esm_repr_layer,
        )
        sequences_to_embeddings = {
            f"{seq}:{i}": esm_embeddings[i].cpu().numpy() for i, seq in enumerate(sequences)
        }

        apo_protein_sample = process_protein(
            apo_af_protein,
            sample_name=f"{holo_structure.id}_",
            sequences_to_embeddings=sequences_to_embeddings,
        )

        # merge holo and apo proteins
        for key in complex_graph.keys():
            for subkey, value in apo_protein_sample[key].items():
                complex_graph[key]["apo_" + subkey] = value

        complex_graph["metadata"]["sample_ID"] = holo_structure.id

        if np.isnan(complex_graph["features"]["apo_res_atom_positions"]).any():
            raise ValueError(
                f"NaN values found in apo protein sample for system {holo_structure.id}"
            )
        if (
            complex_graph["features"]["res_atom_positions"].shape
            != complex_graph["features"]["apo_res_atom_positions"].shape
        ):
            raise ValueError(
                f"Atom positions shape mismatch between holo and apo protein samples for system {holo_structure.id}"
            )

        # merge ligands with holo and apo proteins
        complex_graph = merge_protein_and_ligands(
            lig_samples,
            complex_graph,
            n_lig_patches=n_lig_patches,
        )

        complex_graph["metadata"]["mol"] = combine_molecules(ligands)
        complex_graph = centralize_complex_graph(sample_inplace_to_torch(complex_graph))

    except Exception as e:
        raise e

    return complex_graph


class FlowDockPlinderDataset(Dataset):  # type: ignore
    """Creates a dataset for FlowDock from Plinder systems.

    Parameters
    ----------
    data_dir : str
        The directory where the data is stored
    split : str
        The split to sample from
    featurizer: Callable[
            [Structure, int], dict[str, torch.Tensor]
    ] = structure_featurizer,
        Transformation to turn structure to input tensors
    system_must_have_apo_or_pred : bool
        Whether to filter systems that do not have apo or predicted structures
    system_must_be_deposited_before : str | None
        Filter systems that were deposited on or after a certain date
    system_must_be_deposited_after : str | None
        Filter systems that were deposited before a certain date
    min_protein_length : int | None
        Filter systems with protein sequence length less than this
    max_protein_length : int | None
        Filter systems with protein sequence length greater than this
    **kwargs : Any
        Additional keyword arguments to pass to the dataset
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        featurizer: Callable[
            [
                Structure,
                Structure,
                esm.ESM2,
                esm.data.Alphabet,
                esm.data.BatchConverter,
                int,
                int,
            ],
            torch.Tensor | dict[str, torch.Tensor],
        ],
        system_must_have_apo_or_pred: bool = True,
        system_must_be_deposited_before: str | None = None,
        system_must_be_deposited_after: str | None = None,
        min_protein_length: int | None = None,
        max_protein_length: int | None = None,
        **kwargs: Any,
    ):
        df = pd.read_parquet(
            glob.glob(os.path.join(data_dir, "*", "*", "*", "splits", "split.parquet"))[0]
        )
        split_df = df[
            (df["split"] == split)
            & (df["system_pass_validation_criteria"] == True)  # noqa: E712
            & (df["system_pass_statistics_criteria"] == True)  # noqa: E712
            & (df["system_has_apo_or_pred"] == system_must_have_apo_or_pred)  # noqa: E712
        ]
        assert (
            len(split_df) > 0
        ), f"No systems found for split '{split}' with apo or pred structures."

        self._system_ids = list(set(split_df["system_id"]))
        self._num_examples = len(self._system_ids)

        self._featurizer = featurizer

        self.system_must_be_deposited_before = system_must_be_deposited_before
        self.system_must_be_deposited_after = system_must_be_deposited_after

        self.min_protein_length = min_protein_length
        self.max_protein_length = max_protein_length

    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return self._num_examples

    def __getitem__(
        self, index: int
    ) -> dict[str, int | str | pd.DataFrame | dict[str, str | pd.DataFrame]]:
        """Get a single item from the dataset.

        :param index: The index of the item to retrieve.
        :return: A dictionary containing the features of the complex graph.
        """
        try:
            if not 0 <= index < self._num_examples:
                raise IndexError(index)

            s = PlinderSystem(system_id=self._system_ids[index])

            if self.system_must_be_deposited_before is not None:
                # filter out systems that were deposited on or after a certain date
                if s.entry["release_date"] >= self.system_must_be_deposited_before:
                    log.info(
                        f"System {s.system_id} was deposited on or after {self.system_must_be_deposited_before}. Skipping..."
                    )
                    return self.__getitem__(np.random.randint(0, self._num_examples))

            if self.system_must_be_deposited_after is not None:
                # filter out systems that were deposited before a certain date
                if s.entry["release_date"] <= self.system_must_be_deposited_after:
                    log.info(
                        f"System {s.system_id} was deposited on or before {self.system_must_be_deposited_after}. Skipping..."
                    )
                    return self.__getitem__(np.random.randint(0, self._num_examples))

            if self.min_protein_length is not None:
                # filter out systems with protein sequence length less than `min_protein_length`
                if len(s.holo_structure.protein_sequence_from_structure) < self.min_protein_length:
                    log.info(
                        f"System {s.system_id} has holo protein length {len(s.holo_structure.protein_sequence_from_structure)} < {self.min_protein_length}. Skipping..."
                    )
                    return self.__getitem__(np.random.randint(0, self._num_examples))

            if self.max_protein_length is not None:
                # filter out systems with protein sequence length greater than `max_protein_length`
                if len(s.holo_structure.protein_sequence_from_structure) > self.max_protein_length:
                    log.info(
                        f"System {s.system_id} has holo protein length {len(s.holo_structure.protein_sequence_from_structure)} > {self.max_protein_length}. Skipping..."
                    )
                    return self.__getitem__(np.random.randint(0, self._num_examples))

            # prefer to load (random) crystal apo protein structures over (random) predicted apo protein structures
            if any(struct.structure_type == "apo" for struct in s.alternate_structures.values()):
                apo_structure = random.choice(  # nosec
                    list(
                        struct
                        for struct in s.alternate_structures.values()
                        if struct.structure_type == "apo"
                    )
                )
            else:
                assert s.alternate_structures, f"System {s.system_id} has no alternate structures."
                apo_structure = random.choice(  # nosec
                    list(
                        struct
                        for struct in s.alternate_structures.values()
                        if struct.structure_type == "pred"
                    )
                )

            holo_structure = s.holo_structure
            complex_graph = self._featurizer(apo_structure, holo_structure, s.chain_mapping)

            complex_graph["features"]["affinity"] = torch.tensor(
                # NOTE: until https://github.com/plinder-org/plinder/issues/94 is fixed,
                # PLINDER's binding affinity values cannot be used
                [torch.nan for _ in range(complex_graph["metadata"]["num_molid"])],
                dtype=torch.float32,
            )

        except Exception as e:
            log.error(f"Skipping system at index {index} because of the error: {e}")
            return self.__getitem__(np.random.randint(0, self._num_examples))

        return complex_graph
