#!/bin/bash -l
######################### Batch Headers #########################
#SBATCH --partition general    # NOTE: use reserved partition `chengji-lab-gpu` to use reserved A100 or H100 GPUs
#SBATCH --account chengji-lab  # NOTE: this must be specified to use the reserved partition above
#SBATCH --nodes=1              # NOTE: this needs to match Lightning's `Trainer(num_nodes=...)`
#SBATCH --ntasks-per-node=1    # NOTE: this needs to be `1` on SLURM clusters when using Lightning's `ddp_spawn` strategy`; otherwise, set to match Lightning's quantity of `Trainer(devices=...)`
#SBATCH --mem=59G              # NOTE: use `--mem=0` to request all memory "available" on the assigned node
#SBATCH -t 0-02:00:00          # time limit for the job (up to 2 days: `2-00:00:00`)
#SBATCH -J plinder_download    # job name
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

# determine location of PLINDER dataset
export PLINDER_MOUNT="$project_dir/data/PLINDER" # NOTE: customize as needed
mkdir -p "$PLINDER_MOUNT" # create the directory if it doesn't exist

echo "Downloading PLINDER to $PLINDER_MOUNT!"
cd "$project_dir" || exit
plinder_download -y
echo "Finished downloading PLINDER to $PLINDER_MOUNT!"
