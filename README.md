# CUDA Vector Addition Tutorial

This is a simple CUDA program that adds two vectors together on the GPU.

*This tutorial was generated using Claude (Anthropic) and configured for the Sherlock cluster at Stanford.*

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

## How to Use on Sherlock

### 1. Log into Sherlock

```bash
ssh your_username@sherlock.stanford.edu
```

### 2. Clone the repository

```bash
git clone https://github.com/ispecht/cuda-tutorial.git
cd cuda-tutorial
```

### 3. Submit the job

```bash
sbatch submit.slurm
```

### 4. Check job status

```bash
squeue -u $USER
```

### 5. View results

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
- **Solution**: The Makefile is configured for `-arch=sm_70` which works for most Sherlock GPUs. If you encounter issues, check your GPU with `nvidia-smi` on a GPU node and adjust accordingly.

**Problem**: Job stays in queue
- **Solution**: Check queue status with `squeue` - Sherlock's GPU partition may be busy during peak hours.

**Problem**: Can't find output file
- **Solution**: Look for files matching the pattern `vector_add_*.out` in your directory