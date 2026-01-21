#!/bin/bash
#SBATCH --job-name=kpfm_finetune
#SBATCH --output=logs/kpfm_finetune_%j.out
#SBATCH --error=logs/kpfm_finetune_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G

# =============================================================================
# KPFM Fine-tuning Script for 4xA100 40G
# =============================================================================
# This script fine-tunes FlowDock with KPFM (Kinematic-Projected Flow Matching)
# using DOF-space parameterization for improved docking accuracy.
#
# Prerequisites:
# 1. FlowDock pretrained weights in checkpoints/
# 2. DOF cache precomputed using precompute_dof_cache.py
# 3. Training data prepared (PDBbind/MOAD/Plinder)
# =============================================================================

set -e

# Load environment
source ~/.bashrc
conda activate flowdock

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Memory optimization for A100
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create log directory
mkdir -p logs

echo "=============================================="
echo "KPFM Fine-tuning - Starting at $(date)"
echo "=============================================="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo ""

# Step 1: Check if DOF cache exists
DOF_CACHE_DIR="${PROJECT_ROOT}/data/dof_cache"
if [ ! -d "$DOF_CACHE_DIR" ] || [ -z "$(ls -A $DOF_CACHE_DIR 2>/dev/null)" ]; then
    echo "WARNING: DOF cache not found at $DOF_CACHE_DIR"
    echo "Running DOF preprocessing first..."
    python scripts/precompute_dof_cache.py \
        --data_dir data/pdbbind \
        --output_dir $DOF_CACHE_DIR \
        --dataset pdbbind \
        --num_workers 8 \
        --binding_site_cutoff 6.0 \
        --include_sidechains
    echo "DOF preprocessing complete."
fi

# Step 2: Run KPFM fine-tuning
echo ""
echo "Starting KPFM fine-tuning..."
echo ""

python train.py \
    experiment=flowdock_kpfm \
    trainer.devices=4 \
    trainer.accelerator=gpu \
    trainer.strategy=ddp_find_unused_parameters_true \
    trainer.precision=16-mixed \
    trainer.max_epochs=100 \
    trainer.check_val_every_n_epoch=5 \
    data.batch_size=8 \
    data.num_workers=4 \
    model.optimizer.lr=5e-5 \
    model.cfg.prior_type=kpfm \
    model.cfg.task.kpfm.use_dof_cache=true \
    model.cfg.task.kpfm.dof_cache_dir=$DOF_CACHE_DIR \
    model.cfg.task.kpfm.use_sparse=true \
    model.cfg.task.kpfm.damping=1e-4 \
    model.cfg.task.kpfm.translation_std=5.0 \
    model.cfg.task.kpfm.rotation_std=0.5 \
    model.cfg.task.kpfm.include_sidechains=true \
    model.cfg.task.kpfm.binding_site_only=true \
    model.cfg.task.kpfm_velocity_loss_weight=1.0 \
    model.cfg.task.kpfm_projection_loss_weight=0.1 \
    logger=wandb \
    logger.wandb.project=FlowDock-KPFM \
    logger.wandb.name="kpfm_finetune_$(date +%Y%m%d_%H%M%S)" \
    callbacks.model_checkpoint.monitor="val/tm_lbound" \
    callbacks.model_checkpoint.mode="max" \
    callbacks.model_checkpoint.save_top_k=3 \
    hydra.run.dir="outputs/kpfm_finetune_$(date +%Y%m%d_%H%M%S)" \
    "$@"

echo ""
echo "=============================================="
echo "KPFM Fine-tuning - Completed at $(date)"
echo "=============================================="
