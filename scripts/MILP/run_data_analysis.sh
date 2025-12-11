#!/bin/bash
#SBATCH --account=dilkina_1761
#SBATCH --job-name=shade_la

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --exclude=d14-16

#SBATCH --output=slurm_jobs/%x_%j.out
#SBATCH --error=slurm_jobs/%x_%j.err
# choose from A100, A40, V100, P100, K40

# commands to activate github and check slurm job info
# eval "$(ssh-agent -s)" ssh-add ~/.ssh/id_rsa
# sinfo -t idle -o "%N %G"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# load conda env when necessary
# module purge
source /scratch1/ubhuwani/miniconda3/etc/profile.d/conda.sh
conda activate shade_la_env

python transit_stop_public_services_data_analysis.py