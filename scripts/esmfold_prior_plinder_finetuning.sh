#!/bin/bash -l
######################### Batch Headers #########################
#SBATCH --partition chengji-lab-gpu    # NOTE: use reserved partition `chengji-lab-gpu` to use reserved A100 or H100 GPUs
#SBATCH --account chengji-lab  # NOTE: this must be specified to use the reserved partition above
#SBATCH --nodes=1              # NOTE: this needs to match Lightning's `Trainer(num_nodes=...)`
#SBATCH --gres gpu:1           # request A100/H100 GPU resource(s)
#SBATCH --ntasks-per-node=1    # NOTE: this needs to be `1` on SLURM clusters when using Lightning's `ddp_spawn` strategy`; otherwise, set to match Lightning's quantity of `Trainer(devices=...)`
#SBATCH --mem=59G              # NOTE: use `--mem=0` to request all memory "available" on the assigned node
#SBATCH -t 2-00:00:00          # time limit for the job (up to 2 days: `2-00:00:00`)
#SBATCH -J esmfold_prior_plinder_finetuning    # job name
#SBATCH --output=R-%x.%j.out   # output log file
#SBATCH --error=R-%x.%j.err    # error log file

module purge
module load cuda/11.8.0_gcc_9.5.0

# determine location of the project directory
use_private_project_dir=false # NOTE: customize as needed
if [ "$use_private_project_dir" = true ]; then
    project_dir="/home/acmwhb/data/Repositories/Lab_Repositories/FlowDock"
else
    project_dir="/cluster/pixstor/chengji-lab/acmwhb/Repositories/Lab_Repositories/FlowDock"
fi

# shellcheck source=/dev/null
source /cluster/pixstor/chengji-lab/acmwhb/miniforge3/etc/profile.d/conda.sh
conda activate "$project_dir"/FlowDock/

# Reference Conda system libraries
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "Calling flowdock/train.py!"
cd "$project_dir" || exit
srun python3 flowdock/train.py \
    callbacks.last_model_checkpoint.filename=null \
    callbacks.last_model_checkpoint.every_n_train_steps=200 \
    callbacks.last_model_checkpoint.every_n_epochs=null \
    ckpt_path="$(realpath 'logs/train/runs/2025-03-17_17-39-39/checkpoints/169-562000.ckpt')" \
    data=plinder \
    experiment='flowdock_fm' \
    environment=slurm \
    logger=wandb \
    logger.wandb.entity='bml-lab' \
    logger.wandb.group='FlowDock-FM' \
    +logger.wandb.name='2025-03-17_17:00:00-ESMFold-Prior-PLINDER-Finetuning' \
    +logger.wandb.id='1x2k5a79' \
    model.cfg.prior_type=esmfold \
    model.cfg.task.freeze_score_head=false \
    model.cfg.task.freeze_affinity=true \
    paths.output_dir="$(realpath 'logs/train/runs/2025-03-17_17-39-39')" \
    strategy=ddp \
    trainer=ddp \
    +trainer.accumulate_grad_batches=4 \
    trainer.devices=1 \
    trainer.num_nodes=1
echo "Finished calling flowdock/train.py!"

# NOTE: the following commands must be used to resume training from a checkpoint
# ckpt_path="$(realpath 'logs/train/runs/2025-03-17_17-39-39/checkpoints/169-562000.ckpt')" \
# paths.output_dir="$(realpath 'logs/train/runs/2025-03-17_17-39-39')" \

# NOTE: the following commands may be used to speed up training
# model.compile=false \
# +trainer.precision=bf16-mixed
