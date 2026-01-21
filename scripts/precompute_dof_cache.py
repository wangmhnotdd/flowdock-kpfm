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
from flowdock.data.components.topology_validator import (
    validate_topology_full,
    TopologyValidationResult,
    ConformationalMetrics,
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
    # Topology validation and KPFM filtering
    parser.add_argument(
        "--enable_topology_validation",
        action="store_true",
        default=True,
        help="Enable topology validation between holo and apo structures"
    )
    parser.add_argument(
        "--kpfm_max_backbone_rmsd",
        type=float,
        default=3.0,
        help="Maximum backbone RMSD for KPFM filtering (Angstroms)"
    )
    parser.add_argument(
        "--kpfm_max_pocket_rmsd",
        type=float,
        default=2.0,
        help="Maximum pocket RMSD for KPFM filtering (Angstroms)"
    )
    parser.add_argument(
        "--kpfm_min_aligned_fraction",
        type=float,
        default=0.9,
        help="Minimum aligned residue fraction for KPFM filtering"
    )
    parser.add_argument(
        "--save_topology_report",
        action="store_true",
        default=True,
        help="Save topology validation report to JSON"
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
    enable_topology_validation: bool = True,
    kpfm_max_backbone_rmsd: float = 3.0,
    kpfm_max_pocket_rmsd: float = 2.0,
    kpfm_min_aligned_fraction: float = 0.9,
) -> Dict[str, Any]:
    """Process a single sample and save DOF cache."""
    
    result = {
        "sample_id": sample_id,
        "success": False,
        "error": None,
        "n_ligand_torsions": 0,
        "n_sidechain_dofs": 0,
        # Topology validation results
        "topology_valid": None,
        "backbone_rmsd": None,
        "pocket_rmsd": None,
        "aligned_fraction": None,
        "passes_kpfm_filter": None,
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
        
        # Topology validation (for KPFM filtering)
        validation_result = None
        if enable_topology_validation:
            # Check if apo structure exists
            apo_protein_file = None
            if dataset == "pdbbind":
                apo_dir = Path(data_dir).parent / "pdbbind_holo_aligned_esmfold_structures"
                apo_protein_file = apo_dir / f"{sample_id}_holo_aligned_esmfold_protein.pdb"
                if not apo_protein_file.exists():
                    # Try alternative path
                    apo_protein_file = sample_dir / f"{sample_id}_apo.pdb"
            
            if apo_protein_file and apo_protein_file.exists():
                try:
                    apo_coords, apo_atoms, apo_residue_info = load_protein_pdb(apo_protein_file)
                    
                    # Extract residue types
                    holo_residue_types = [r['resname'] for r in residue_info]
                    apo_residue_types = [r['resname'] for r in apo_residue_info]
                    
                    # Create simple atom37 masks (simplified - full implementation would 
                    # use actual atom37 format)
                    n_holo_res = len(residue_info)
                    n_apo_res = len(apo_residue_info)
                    holo_atom37_mask = np.ones((n_holo_res, 37), dtype=bool)  # Simplified
                    apo_atom37_mask = np.ones((n_apo_res, 37), dtype=bool)    # Simplified
                    
                    # Run validation
                    validation_result = validate_topology_full(
                        holo_coords=protein_coords.reshape(n_holo_res, -1, 3)[:, :37, :],
                        apo_coords=apo_coords.reshape(n_apo_res, -1, 3)[:, :37, :],
                        holo_residue_types=holo_residue_types,
                        apo_residue_types=apo_residue_types,
                        holo_atom37_mask=holo_atom37_mask,
                        apo_atom37_mask=apo_atom37_mask,
                        ligand_coords=ligand_coords,
                        kpfm_max_backbone_rmsd=kpfm_max_backbone_rmsd,
                        kpfm_max_pocket_rmsd=kpfm_max_pocket_rmsd,
                        kpfm_min_aligned_fraction=kpfm_min_aligned_fraction,
                    )
                    
                    # Record results
                    result["topology_valid"] = validation_result.is_valid
                    if validation_result.conformational_metrics:
                        result["backbone_rmsd"] = validation_result.conformational_metrics.backbone_rmsd
                        result["pocket_rmsd"] = validation_result.conformational_metrics.pocket_backbone_rmsd
                        result["aligned_fraction"] = validation_result.conformational_metrics.aligned_fraction
                        result["passes_kpfm_filter"] = validation_result.conformational_metrics.passes_kpfm_filter(
                            kpfm_max_backbone_rmsd, kpfm_max_pocket_rmsd, kpfm_min_aligned_fraction
                        )
                    
                except Exception as e:
                    log.warning(f"Topology validation failed for {sample_id}: {e}")
                    result["topology_valid"] = False
                    result["error"] = f"Topology validation failed: {e}"
        
        # Save cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'kin_system': kin_system,
                'dof_cache': dof_cache,
                'validation_result': validation_result,
                'metadata': {
                    'sample_id': sample_id,
                    'n_protein_atoms': len(protein_atoms),
                    'n_ligand_atoms': len(ligand_atoms),
                    'n_ligand_torsions': len(kin_system.ligand_torsion_defs),
                    'n_sidechain_dofs': len(kin_system.sidechain_torsion_defs),
                    'binding_site_cutoff': binding_site_cutoff,
                    # KPFM metrics for filtering during training
                    'kpfm_metrics': {
                        'backbone_rmsd': result.get("backbone_rmsd"),
                        'pocket_backbone_rmsd': result.get("pocket_rmsd"),
                        'aligned_fraction': result.get("aligned_fraction"),
                        'passes_filter': result.get("passes_kpfm_filter"),
                    } if enable_topology_validation else None,
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
        enable_topology_validation=args.enable_topology_validation,
        kpfm_max_backbone_rmsd=args.kpfm_max_backbone_rmsd,
        kpfm_max_pocket_rmsd=args.kpfm_max_pocket_rmsd,
        kpfm_min_aligned_fraction=args.kpfm_min_aligned_fraction,
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
    
    # Topology validation statistics
    n_topology_valid = sum(1 for r in results if r.get("topology_valid") == True)
    n_passes_kpfm = sum(1 for r in results if r.get("passes_kpfm_filter") == True)
    backbone_rmsds = [r["backbone_rmsd"] for r in results if r.get("backbone_rmsd") is not None]
    pocket_rmsds = [r["pocket_rmsd"] for r in results if r.get("pocket_rmsd") is not None]
    aligned_fracs = [r["aligned_fraction"] for r in results if r.get("aligned_fraction") is not None]
    
    log.info(f"\n{'='*50}")
    log.info(f"DOF Cache Preprocessing Complete")
    log.info(f"{'='*50}")
    log.info(f"Total samples: {len(results)}")
    log.info(f"Successful: {n_success}")
    log.info(f"Failed: {n_failed}")
    log.info(f"Avg ligand torsions: {total_lig_torsions / max(n_success, 1):.1f}")
    log.info(f"Avg sidechain DOFs: {total_sc_dofs / max(n_success, 1):.1f}")
    
    # Topology validation summary
    if args.enable_topology_validation and backbone_rmsds:
        log.info(f"\n{'='*50}")
        log.info(f"Topology Validation Summary")
        log.info(f"{'='*50}")
        log.info(f"Topology valid: {n_topology_valid}/{len(results)}")
        log.info(f"Passes KPFM filter: {n_passes_kpfm}/{len(results)} ({100*n_passes_kpfm/len(results):.1f}%)")
        log.info(f"Backbone RMSD: mean={np.mean(backbone_rmsds):.2f}Å, median={np.median(backbone_rmsds):.2f}Å")
        log.info(f"Pocket RMSD: mean={np.mean(pocket_rmsds):.2f}Å, median={np.median(pocket_rmsds):.2f}Å")
        log.info(f"Aligned fraction: mean={np.mean(aligned_fracs):.2%}, min={np.min(aligned_fracs):.2%}")
    
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
                "kpfm_max_backbone_rmsd": args.kpfm_max_backbone_rmsd,
                "kpfm_max_pocket_rmsd": args.kpfm_max_pocket_rmsd,
                "kpfm_min_aligned_fraction": args.kpfm_min_aligned_fraction,
            },
            "failed_samples": [r["sample_id"] for r in results if not r["success"]],
        }, f, indent=2)
    
    # Save topology validation report
    if args.save_topology_report and args.enable_topology_validation:
        topology_report_file = output_dir / "topology_report.json"
        topology_report = {
            "summary": {
                "total_samples": len(results),
                "topology_valid": n_topology_valid,
                "passes_kpfm_filter": n_passes_kpfm,
                "kpfm_pass_rate": n_passes_kpfm / max(len(results), 1),
                "mean_backbone_rmsd": float(np.mean(backbone_rmsds)) if backbone_rmsds else None,
                "median_backbone_rmsd": float(np.median(backbone_rmsds)) if backbone_rmsds else None,
                "mean_pocket_rmsd": float(np.mean(pocket_rmsds)) if pocket_rmsds else None,
                "mean_aligned_fraction": float(np.mean(aligned_fracs)) if aligned_fracs else None,
            },
            "thresholds": {
                "kpfm_max_backbone_rmsd": args.kpfm_max_backbone_rmsd,
                "kpfm_max_pocket_rmsd": args.kpfm_max_pocket_rmsd,
                "kpfm_min_aligned_fraction": args.kpfm_min_aligned_fraction,
            },
            "per_sample": [
                {
                    "sample_id": r["sample_id"],
                    "topology_valid": r.get("topology_valid"),
                    "passes_kpfm_filter": r.get("passes_kpfm_filter"),
                    "backbone_rmsd": r.get("backbone_rmsd"),
                    "pocket_rmsd": r.get("pocket_rmsd"),
                    "aligned_fraction": r.get("aligned_fraction"),
                }
                for r in results if r.get("backbone_rmsd") is not None
            ],
            "filtered_out_samples": [
                r["sample_id"] for r in results 
                if r.get("passes_kpfm_filter") == False
            ],
        }
        with open(topology_report_file, 'w') as f:
            json.dump(topology_report, f, indent=2)
        log.info(f"Topology report saved to: {topology_report_file}")
    
    log.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
