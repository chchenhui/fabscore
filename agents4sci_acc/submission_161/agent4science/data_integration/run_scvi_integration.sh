#!/bin/bash
#SBATCH --job-name=scvi_integration
#SBATCH --output=/scratch/rli/project/agent/logs/scvi_integration_%j.out
#SBATCH --error=/scratch/rli/project/agent/logs/scvi_integration_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Initialize conda properly
source /wanglab/rli/miniforge3/etc/profile.d/conda.sh
conda activate data_integration

# Set optimization parameters
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024,expandable_segments:True
export OMP_NUM_THREADS=5
export MKL_NUM_THREADS=5
export NUMEXPR_NUM_THREADS=5
ulimit -v unlimited

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE"

# Check GPU availability
nvidia-smi

# Change to working directory
cd /scratch/rli/project/agent/data_integration

# Run the integration pipeline
python neuroendocrine_integration_scvi_unique.py

# Print completion
echo "Job completed at: $(date)"