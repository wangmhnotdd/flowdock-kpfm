# KPFM: Kinematic-Projected Flow Matching

This document describes the implementation of KPFM (Kinematic-Projected Flow Matching) on top of the FlowDock codebase.

## Overview

KPFM is a novel approach to protein-ligand docking that combines flow matching with kinematic constraints. Unlike standard flow matching methods that operate in Cartesian coordinate space, KPFM:

1. **Parameterizes motion in DOF (Degrees of Freedom) space**: Translation, rotation, and torsion angles
2. **Enforces kinematic constraints**: Bond lengths and angles are preserved via forward kinematics reconstruction
3. **Uses projected velocity targets**: The velocity field lies in the tangent space of the kinematic manifold

## Key Concepts

### DOF Space Parameterization

The state is represented as:
```
q = [translation (3), rotation (3), ligand_torsions (L), sidechain_chi (K)]
```

Where:
- **Translation** (3 DOF): Global ligand translation in Angstroms
- **Rotation** (3 DOF): Axis-angle representation of ligand orientation
- **Ligand Torsions** (L DOF): Rotatable bonds in the ligand
- **Sidechain Chi** (K DOF): Chi angles for binding site sidechains

### Forward Kinematics (FK)

Converts DOF state to Cartesian coordinates:
```
x = FK(q, x_ref)
```

The FK function applies:
1. Ligand translation and rotation (rigid body motion)
2. Sequential torsion rotations for ligand rotatable bonds
3. Sidechain chi angle rotations for protein atoms

### Jacobian Matrix

The Jacobian J relates velocity in DOF space to velocity in atom space:
```
v = J(q) @ dq
```

For KPFM training, the target velocity is:
```
v_target = J(q_t) @ dq_target
```

This ensures the velocity field lies in the kinematic manifold's tangent space.

### Projection Layer

During inference, predicted velocities are projected onto the manifold:
```
dq = J^+ @ v_pred
```

Where J^+ is the damped pseudo-inverse:
```
J^+ = (J^T J + λI)^{-1} J^T
```

## File Structure

```
flowdock/
├── models/
│   └── components/
│       ├── kinematics.py          # Core KPFM module
│       ├── flowdock.py            # Modified with KPFM support
│       ├── noise.py               # KPFM prior sampling
│       ├── losses.py              # KPFM velocity loss
│       └── cuda/
│           └── sparse_jacobian_kernel.py  # CUDA acceleration
├── data/
│   └── components/
│       └── dof_utils.py           # DOF extraction utilities

configs/
├── model/
│   └── flowdock_kpfm.yaml         # KPFM model config
└── experiment/
    └── flowdock_kpfm.yaml         # KPFM experiment config

scripts/
├── precompute_dof_cache.py        # DOF preprocessing
└── kpfm_finetuning.sh             # Training script
```

## Usage

### 1. Precompute DOF Cache

Before training, precompute DOF data for your dataset:

```bash
python scripts/precompute_dof_cache.py \
    --data_dir data/pdbbind \
    --output_dir data/dof_cache \
    --dataset pdbbind \
    --num_workers 8 \
    --binding_site_cutoff 6.0 \
    --include_sidechains
```

### 2. Fine-tune with KPFM

```bash
# Using the provided script (for SLURM clusters with 4xA100):
sbatch scripts/kpfm_finetuning.sh

# Or directly:
python train.py experiment=flowdock_kpfm
```

### 3. Sample with KPFM

The trained model can sample structures using KPFM kinematic sampling:

```python
from flowdock.models.components.flowdock import FlowDock

model = FlowDock.load_from_checkpoint("path/to/checkpoint.ckpt")

# KPFM sampling preserves kinematic constraints
results = model.sample_pl_complex_structures_kpfm(
    batch,
    num_steps=100,
    damping=1e-4,
)
```

## Configuration

Key KPFM parameters in `configs/model/flowdock_kpfm.yaml`:

```yaml
cfg:
  prior_type: kpfm  # Use KPFM prior
  
  task:
    kpfm:
      # Prior sampling
      translation_std: 5.0      # Angstroms
      rotation_std: 0.5         # Radians
      torsion_std: 3.14159      # Uniform over [-π, π]
      
      # Jacobian computation
      damping: 1e-4             # Pseudo-inverse regularization
      use_sparse: true          # Sparse Jacobian for large proteins
      sparse_threshold: 500     # Atom count threshold for sparse
      
      # Sidechain handling
      include_sidechains: true
      max_chi_angles: 4
      binding_site_only: true
      binding_site_cutoff: 6.0
      
      # DOF cache
      use_dof_cache: true
      dof_cache_dir: ${oc.env:PROJECT_ROOT}/data/dof_cache
```

## Memory Optimization

KPFM requires additional memory for Jacobian computation. For large proteins on A100 40G:

1. **Use sparse Jacobian**: Set `use_sparse: true` for proteins > 500 atoms
2. **Reduce batch size**: Default is 8 per GPU
3. **Use mixed precision**: `trainer.precision=16-mixed`
4. **Gradient checkpointing**: Enabled in config

## Training Tips

1. **Initialize from FlowDock pretrained weights**: The KPFM fine-tuning assumes you have FlowDock pretrained weights
2. **Lower learning rate**: Use ~5e-5 for fine-tuning (vs 2e-4 for pretraining)
3. **Freeze most components**: Only fine-tune score head for KPFM objective
4. **Monitor TM-score**: Use `val/tm_lbound` for model selection

## Algorithm Details

### Training Target

During training at time t ∈ [0, 1]:

1. Sample prior DOF state q₁ ~ p(q)
2. Get target DOF state q₀ from holo structure
3. Interpolate: q_t = geodesic_interp(q₀, q₁, t)
4. Compute DOF velocity: dq_target = (q₀ - q_t) / (1 - t)
5. Compute target velocity in atom space: v_target = J(q_t) @ dq_target

The network learns to predict v_target from the noisy structure x_t = FK(q_t).

### Inference

1. Sample initial DOF state q_T ~ p(q)
2. For t = T, T-Δt, ..., 0:
   - Compute coordinates: x_t = FK(q_t)
   - Predict velocity: v_pred = network(x_t, t)
   - Project to DOF space: dq = J^+ @ v_pred
   - Integrate: q_{t-Δt} = q_t + Δt * dq
3. Final structure: x_0 = FK(q_0)

The projection step ensures all generated structures satisfy geometric constraints.

## Citation

If you use KPFM in your research, please cite:

```bibtex
@article{kpfm2024,
  title={Kinematic-Projected Flow Matching for Protein-Ligand Docking},
  author={...},
  journal={...},
  year={2024}
}
```

## References

- FlowDock: Geometric Flow Matching for Generative Protein-Ligand Docking and Affinity Prediction
- Riemannian Flow Matching on General Geometries (Chen et al., 2024)
- NeuralPLexer: Neural Parameterized Complexer for Biomolecular Docking
