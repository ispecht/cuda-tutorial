# CUDA Vector Addition Tutorial

This is a simple CUDA program that adds two vectors together on the GPU.

## Files Included

- **vector_add.cu** - The main CUDA program with detailed comments
- **Makefile** - For compiling the program
- **submit.slurm** - SLURM batch script for submitting to a cluster
- **README.md** - This file

## Key CUDA Concepts Demonstrated

1. **Kernel Function** (`__global__`): A function that runs on the GPU
2. **Thread Indexing**: How to calculate which data element each thread processes
3. **Memory Management**: 
   - Allocating memory on GPU with `cudaMalloc()`
   - Copying data between CPU and GPU with `cudaMemcpy()`
   - Freeing GPU memory with `cudaFree()`
4. **Kernel Launch**: Using `<<<blocks, threads>>>` syntax
5. **Synchronization**: Waiting for GPU to finish with `cudaDeviceSynchronize()`

## How to Use on a Cluster

### 1. Log into the cluster

```bash
ssh your_username@sherlock.stanford.edu
```

### 2. Clone the repository

```bash
git clone https://github.com/ispecht/cuda-tutorial.git
cd cuda-tutorial
```

### 3. Adjust the SLURM script for your cluster

Edit `submit.slurm` and modify:
- `#SBATCH --partition=gpu` - Change to your cluster's GPU partition name
- `module load cuda/11.8` - Change to your cluster's CUDA module version

To find available modules: `module avail cuda`
To find available partitions: `sinfo`

### 4. Submit the job

```bash
sbatch submit.slurm
```

### 5. Check job status

```bash
squeue -u $USER
```

### 6. View results

Once the job completes, check the output file:

```bash
cat vector_add_JOBID.out
```

## Compiling and Running Locally (if you have a GPU)

```bash
make
./vector_add
```

## Expected Output

```
Adding two vectors of 1000000 elements
Launching kernel with 3907 blocks and 256 threads per block
Verifying results...
h_result[0] = 3.0 (expected 3.0)
h_result[1] = 3.0 (expected 3.0)
...
SUCCESS! All results are correct.
```

## Common Issues and Solutions

**Problem**: Compilation fails with architecture error
- **Solution**: Change `-arch=sm_70` in Makefile to match your GPU (use `nvidia-smi` to check)

**Problem**: `module load cuda` fails
- **Solution**: Check available CUDA versions with `module avail cuda`

**Problem**: Job stays in queue
- **Solution**: Check if GPU partition is correct: `sinfo -p gpu`