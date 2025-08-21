#!/bin/bash

#SBATCH --job-name=finben
#SBATCH --time=01-00:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --qos=qos_bids
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --output=/home/xp83/Documents/project/logs/%j_gpu.out

module load miniconda
conda activate /gpfs/radev/home/xp83/.conda/envs/finben

echo '---------------------------------------------------------------------------'
echo "Job Name              : ${SLURM_JOB_NAME}"
echo "Running on Host       : $(hostname)"
echo "Partition             : ${SLURM_JOB_PARTITION}"
echo "CPUs on Node          : ${SLURM_CPUS_ON_NODE}"
echo "GPUs on Node          : ${SLURM_GPUS_ON_NODE:-0}"
echo "CUDA_VISIBLE_DEVICES  : ${CUDA_VISIBLE_DEVICES:-None}"
echo "Environment Name      : ${CONDA_DEFAULT_ENV:-Not Set}"
echo "Node List             : ${SLURM_NODELIST}"
echo "Submit Directory      : ${SLURM_SUBMIT_DIR}"
echo "Start Time            : $(date)"
echo '---------------------------------------------------------------------------'
echo -e '\n\n'

export HF_HOME=/gpfs/radev/scratch/xu_hua/shared/hf_models

bash run_fl.sh
