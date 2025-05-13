#!/bin/bash

#SBATCH --job-name=finben
#SBATCH --time=01-00:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --constraint="h100"
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --output=/home/xp83/Documents/project/logs/%j_gpu.out

module load miniconda
conda activate finben

echo '-------------------------------------------------'
echo "Job Name: ${SLURM_JOB_NAME}"
echo "I have ${SLURM_CPUS_ON_NODE} CPUs on node $(hostname -s) on partition ${SLURM_JOB_PARTITION}"
echo ${SLURM_SUBMIT_DIR}
echo Running on host $(hostname)
echo Time is $(date)
echo SLURM_NODES are $(echo ${SLURM_NODELIST})
echo '-------------------------------------------------'
echo -e '\n\n'

export HF_HOME='/gpfs/radev/home/xp83/project/hf_cache'
export HF_MODELS_CACHE='/gpfs/radev/home/xp83/project/hf_cache/saved_models'
export HF_DATASETS_CACHE='/gpfs/radev/home/xp83/project/hf_cache/saved_datasets'

bash run_ml.sh
