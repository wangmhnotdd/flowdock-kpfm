#!/usr/bin/env python
"""
KPFM DOF Preprocessing Script

This script precomputes DOF (Degrees of Freedom) cache for training KPFM models.
It processes protein-ligand complex structures and extracts:
- Ligand rotatable bonds and torsion angles
- Sidechain chi angles for binding site residues
- Reference coordinates and kinematic system definitions

Usage:
    python scripts/precompute_dof_cache.py --data_dir data/pdbbind --output_dir data/dof_cache
    
    # For parallel processing:
    python scripts/precompute_dof_cache.py --data_dir data/pdbbind --output_dir data/dof_cache --num_workers 8

Requirements:
    - RDKit for molecular graph analysis
    - BioPython for PDB parsing
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import multiprocessing as mp
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.data.components.dof_utils import (
    build_kinematic_system,
    precompute_dof_cache,
    CachedDOFData,
    TorsionDef,
)
from flowdock.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute DOF cache for KPFM training")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing processed protein-ligand complexes"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save DOF cache files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pdbbind",
        choices=["pdbbind", "moad", "plinder"],
        help="Dataset type"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split file (train/val/test) to process only specific samples"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--binding_site_cutoff",
        type=float,
        default=6.0,
        help="Distance cutoff for binding site definition (Angstroms)"
    )
    parser.add_argument(
        "--max_chi_angles",
        type=int,
        default=4,
        help="Maximum chi angles per residue"
    )
    parser.add_argument(
        "--include_sidechains",
        action="store_true",
        default=True,
        help="Include sidechain chi angles in DOF"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache files"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Process only first 10 samples for debugging"
    )
    return parser.parse_args()


def load_sample_list(data_dir: str, dataset: str, split_file: Optional[str] = None) -> List[str]:
    """Load list of samples to process."""
    samples = []
    
    if split_file and os.path.exists(split_file):
        with open(split_file, 'r') as f:
            samples = [line.strip() for line in f if line.strip()]
        log.info(f"Loaded {len(samples)} samples from split file: {split_file}")
    else:
        # Scan data directory for samples
        data_path = Path(data_dir)
        
        if dataset == "pdbbind":
            # PDBbind structure: data_dir/PDBID/PDBID_protein.pdb, PDBID_ligand.sdf
            for subdir in data_path.iterdir():
                if subdir.is_dir():
                    pdb_id = subdir.name
                    protein_file = subdir / f"{pdb_id}_protein.pdb"
                    ligand_file = subdir / f"{pdb_id}_ligand.sdf"
                    if protein_file.exists() and ligand_file.exists():
                        samples.append(pdb_id)
                        
        elif dataset == "moad":
            # MOAD structure may vary
            for subdir in data_path.iterdir():
                if subdir.is_dir():
                    samples.append(subdir.name)
                    
        elif dataset == "plinder":
            # Plinder processed files
            for pkl_file in data_path.glob("*.pkl"):
                samples.append(pkl_file.stem)
        
        log.info(f"Found {len(samples)} samples in {data_dir}")
    
    return sorted(samples)


def process_single_sample(
    sample_id: str,
    data_dir: str,
    output_dir: str,
    dataset: str,
    binding_site_cutoff: float,
    max_chi_angles: int,
    include_sidechains: bool,
    overwrite: bool,
) -> Dict[str, Any]:
    """Process a single sample and save DOF cache."""
    
    result = {
        "sample_id": sample_id,
        "success": False,
        "error": None,
        "n_ligand_torsions": 0,
        "n_sidechain_dofs": 0,
    }
    
    # Check if cache already exists
    cache_file = Path(output_dir) / f"{sample_id}.pkl"
    if cache_file.exists() and not overwrite:
        result["success"] = True
        result["error"] = "skipped (exists)"
        return result
    
    try:
        # Load complex data
        if dataset == "pdbbind":
            sample_dir = Path(data_dir) / sample_id
            protein_file = sample_dir / f"{sample_id}_protein.pdb"
            ligand_file = sample_dir / f"{sample_id}_ligand.sdf"
            
            protein_coords, protein_atoms, residue_info = load_protein_pdb(protein_file)
            ligand_coords, ligand_atoms, ligand_bonds = load_ligand_sdf(ligand_file)
            
        elif dataset == "plinder":
            # Load from pickle
            pkl_file = Path(data_dir) / f"{sample_id}.pkl"
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            protein_coords = data['protein_coords']
            protein_atoms = data['protein_atoms']
            residue_info = data['residue_info']
            ligand_coords = data['ligand_coords']
            ligand_atoms = data['ligand_atoms']
            ligand_bonds = data['ligand_bonds']
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        # Build kinematic system
        kin_system = build_kinematic_system(
            protein_coords=protein_coords,
            protein_atoms=protein_atoms,
            residue_info=residue_info,
            ligand_coords=ligand_coords,
            ligand_atoms=ligand_atoms,
            ligand_bonds=ligand_bonds,
            binding_site_cutoff=binding_site_cutoff,
            max_chi_angles=max_chi_angles,
            include_sidechains=include_sidechains,
        )
        
        # Compute DOF cache
        dof_cache = precompute_dof_cache(
            coords=np.concatenate([protein_coords, ligand_coords], axis=0),
            kin_system=kin_system,
        )
        
        # Save cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'kin_system': kin_system,
                'dof_cache': dof_cache,
                'metadata': {
                    'sample_id': sample_id,
                    'n_protein_atoms': len(protein_atoms),
                    'n_ligand_atoms': len(ligand_atoms),
                    'n_ligand_torsions': len(kin_system.ligand_torsion_defs),
                    'n_sidechain_dofs': len(kin_system.sidechain_torsion_defs),
                    'binding_site_cutoff': binding_site_cutoff,
                }
            }, f)
        
        result["success"] = True
        result["n_ligand_torsions"] = len(kin_system.ligand_torsion_defs)
        result["n_sidechain_dofs"] = len(kin_system.sidechain_torsion_defs)
        
    except Exception as e:
        result["error"] = str(e)
        log.warning(f"Failed to process {sample_id}: {e}")
    
    return result


def load_protein_pdb(pdb_file: Path) -> Tuple[np.ndarray, List[str], List[Dict]]:
    """Load protein structure from PDB file."""
    try:
        from Bio.PDB import PDBParser
    except ImportError:
        raise ImportError("BioPython required for PDB parsing. Install with: pip install biopython")
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_file))
    
    coords = []
    atoms = []
    residue_info = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                res_info = {
                    'chain': chain.id,
                    'resname': residue.resname,
                    'resid': residue.id[1],
                    'atom_indices': [],
                }
                for atom in residue:
                    coords.append(atom.coord)
                    atoms.append(atom.element if atom.element else atom.name[0])
                    res_info['atom_indices'].append(len(coords) - 1)
                residue_info.append(res_info)
        break  # Only first model
    
    return np.array(coords), atoms, residue_info


def load_ligand_sdf(sdf_file: Path) -> Tuple[np.ndarray, List[str], List[Tuple[int, int, int]]]:
    """Load ligand structure from SDF file."""
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("RDKit required for SDF parsing. Install with: pip install rdkit")
    
    suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
    mol = next(iter(suppl))
    
    if mol is None:
        raise ValueError(f"Failed to parse ligand from {sdf_file}")
    
    conf = mol.GetConformer()
    coords = conf.GetPositions()
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    # Get bonds as (atom1, atom2, bond_order)
    bonds = []
    for bond in mol.GetBonds():
        bonds.append((
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            int(bond.GetBondTypeAsDouble())
        ))
    
    return np.array(coords), atoms, bonds


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sample list
    samples = load_sample_list(args.data_dir, args.dataset, args.split)
    
    if args.debug:
        samples = samples[:10]
        log.info("Debug mode: processing only first 10 samples")
    
    log.info(f"Processing {len(samples)} samples with {args.num_workers} workers")
    
    # Process samples
    process_fn = partial(
        process_single_sample,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
        binding_site_cutoff=args.binding_site_cutoff,
        max_chi_angles=args.max_chi_angles,
        include_sidechains=args.include_sidechains,
        overwrite=args.overwrite,
    )
    
    results = []
    if args.num_workers > 1:
        with mp.Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_fn, samples),
                total=len(samples),
                desc="Processing samples"
            ))
    else:
        for sample_id in tqdm(samples, desc="Processing samples"):
            results.append(process_fn(sample_id))
    
    # Summary statistics
    n_success = sum(1 for r in results if r["success"])
    n_failed = len(results) - n_success
    total_lig_torsions = sum(r["n_ligand_torsions"] for r in results if r["success"])
    total_sc_dofs = sum(r["n_sidechain_dofs"] for r in results if r["success"])
    
    log.info(f"\n{'='*50}")
    log.info(f"DOF Cache Preprocessing Complete")
    log.info(f"{'='*50}")
    log.info(f"Total samples: {len(results)}")
    log.info(f"Successful: {n_success}")
    log.info(f"Failed: {n_failed}")
    log.info(f"Avg ligand torsions: {total_lig_torsions / max(n_success, 1):.1f}")
    log.info(f"Avg sidechain DOFs: {total_sc_dofs / max(n_success, 1):.1f}")
    log.info(f"Output directory: {output_dir}")
    
    # Save summary
    summary_file = output_dir / "preprocessing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "total_samples": len(results),
            "successful": n_success,
            "failed": n_failed,
            "avg_ligand_torsions": total_lig_torsions / max(n_success, 1),
            "avg_sidechain_dofs": total_sc_dofs / max(n_success, 1),
            "config": {
                "binding_site_cutoff": args.binding_site_cutoff,
                "max_chi_angles": args.max_chi_angles,
                "include_sidechains": args.include_sidechains,
            },
            "failed_samples": [r["sample_id"] for r in results if not r["success"]],
        }, f, indent=2)
    
    log.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
