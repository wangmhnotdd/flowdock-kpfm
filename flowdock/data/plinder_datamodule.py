import os
from functools import partial

import esm
import numpy as np
import rootutils
import torch
from beartype.typing import Any, Dict, List, Literal, Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.data.components.mol_features import collate_samples
from flowdock.data.components.plinder_dataset import (
    FlowDockPlinderDataset,
    flowdock_structure_featurizer,
)
from flowdock.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

DATA_PHASE = Literal["train", "val", "test"]


class PlinderDataModule(LightningDataModule):
    """`LightningDataModule` for wrapping custom PlinderDatasets.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/PLINDER/",
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        stage: Optional[str] = None,
        plinder_offline: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a `PlinderDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `16`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param plinder_offline: Whether to use the offline version of PLINDER. Defaults to `False`.
        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        super().__init__()

        os.environ["PLINDER_MOUNT"] = os.path.abspath(data_dir)
        os.environ["PLINDER_OFFLINE"] = str(plinder_offline)  # maybe avoid redownloading dataset

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        log.info("Loading pretrained ESM model...")
        esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet_hub(
            self.hparams.esm_version
        )
        self.esm_model = esm_model.eval().float()
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
        self.esm_model.cpu()

        featurizer = partial(
            flowdock_structure_featurizer,
            esm_model=self.esm_model,
            esm_alphabet=self.esm_alphabet,
            esm_batch_converter=self.esm_batch_converter,
            esm_repr_layer=self.hparams.esm_repr_layer,
            n_lig_patches=self.hparams.n_lig_patches,
        )

        # prepare for dataset(s) to be loaded into rank-shared memory (e.g., when setting `trainer.strategy = "ddp_spawn"`)
        self.data_train = FlowDockPlinderDataset(
            data_dir,
            split="train",
            featurizer=featurizer,
            system_must_be_deposited_before="2018-01-01",  # filter out training systems deposited after this date
            min_protein_length=self.hparams.min_protein_length,  # filter out training systems with protein sequence length less than this
            max_protein_length=self.hparams.max_protein_length,  # filter out training systems with protein sequence length greater than this
            use_alternate_structures=True,
        )

        self.data_val = FlowDockPlinderDataset(
            data_dir,
            split="val",
            featurizer=featurizer,
            system_must_be_deposited_after="2017-12-31",  # filter out validation systems deposited on or before this date
            system_must_be_deposited_before="2019-01-01",  # filter out validation systems deposited after this date
            min_protein_length=self.hparams.min_protein_length,  # filter out validation systems with protein sequence length less than this
            max_protein_length=self.hparams.max_protein_length,  # filter out validation systems with protein sequence length greater than this
            use_alternate_structures=True,
        )

        self.data_test = FlowDockPlinderDataset(
            data_dir,
            split="test",
            featurizer=featurizer,
            system_must_be_deposited_after="2018-12-31",  # filter out test systems deposited on or before this date
            min_protein_length=self.hparams.min_protein_length,  # filter out test systems with protein sequence length less than this
            max_protein_length=self.hparams.max_protein_length,  # filter out test systems with protein sequence length greater than this
            use_alternate_structures=True,
        )

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        pass

    @staticmethod
    def dynamic_batching_by_max_edge_count(
        x: Dict[str, Any], max_n_edges: int, max_batch_size: int
    ) -> Any:
        """Dynamically batch by maximum edge count.

        :param x: The input graph data. If `None`, skip the current batch.
        :param max_n_edges: The maximum number of edges.
        :param max_batch_size: The maximum batch size.
        :return: The batched data.
        """
        if "num_u" in x["metadata"].keys():
            num_edges_upperbound = (
                x["metadata"]["num_a"] * 128 + x["metadata"]["num_i"] * 8 + 160**2
            )
        else:
            num_edges_upperbound = x["metadata"]["num_a"] * 128 + 160**2
        batch_size = max(1, min(max_n_edges // num_edges_upperbound, max_batch_size))
        return collate_samples([x] * batch_size)

    def get_dataloader(
        self,
        phase: DATA_PHASE,
        dataset: Dataset,
        **kwargs: Dict[str, Any],
    ) -> DataLoader[Any]:
        """Create a dataloader from a dataset.

        :param phase: The phase of the dataset. Either `"train"`, `"val"`, or `"test"`.
        :param dataset: The dataset.
        :param kwargs: Additional keyword arguments to pass to the dataloader.
        :return: The dataloader.
        """
        if phase == "train":
            batch_size = self.hparams.batch_size
            epoch_frac = self.hparams.epoch_frac
        else:
            batch_size = 1
            epoch_frac = 1
        sampled_indices = np.random.choice(
            len(dataset),
            int(len(dataset) * epoch_frac),
            replace=False,
        )
        if phase == "val":
            sampled_indices = np.repeat(sampled_indices, self.trainer.world_size)
        subdataset = torch.utils.data.Subset(dataset, sampled_indices)
        return DataLoader(
            subdataset,
            batch_size=None,
            collate_fn=lambda x: self.dynamic_batching_by_max_edge_count(
                x, self.hparams.edge_crop_size, batch_size
            ),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self.get_dataloader(
            "train",
            dataset=self.data_train,
            shuffle=True,
        )

    def val_dataloader(self) -> List[DataLoader[Any]]:
        """Create and return the validation dataloaders.

        :return: The validation dataloaders.
        """
        return self.get_dataloader(
            "val",
            dataset=self.data_val,
            shuffle=False,
        )

    def test_dataloader(self) -> List[DataLoader[Any]]:
        """Create and return the test dataloaders.

        :return: The test dataloaders.
        """
        return self.get_dataloader(
            "test",
            dataset=self.data_test,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = PlinderDataModule()
