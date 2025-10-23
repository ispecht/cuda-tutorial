#!/bin/bash

#SBATCH --job-name=vector_add          # Name of the job
#SBATCH --output=vector_add_%j.out     # Output file (%j will be replaced with job ID)
#SBATCH --error=vector_add_%j.err      # Error file
#SBATCH --time=00:05:00                # Time limit (5 minutes is plenty)
#SBATCH --partition=gpu                # Partition name (adjust for your cluster)
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --ntasks=1                     # Number of tasks (1 for this simple program)
#SBATCH --cpus-per-task=1              # Number of CPU cores per task
#SBATCH --mem=4G                       # Memory per node (4GB is plenty)

# Print some information about the job
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load CUDA module (adjust version for your cluster)
# Check available modules with: module avail cuda
module load cuda/11.8

# Print GPU information
nvidia-smi

# Compile the program
echo "Compiling CUDA program..."
make clean
make

# Run the program
echo "Running CUDA program..."
./vector_add

echo "Job finished at: $(date)"