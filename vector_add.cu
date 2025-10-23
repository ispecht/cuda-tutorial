#include <stdio.h>
#include <cuda_runtime.h>

// This is the CUDA kernel that runs on the GPU
// __global__ means this function can be called from CPU but runs on GPU
// Each thread will execute this function independently
__global__ void vectorAdd(float *x, float *y, float *result, int n) {
    // Calculate which element this thread should process
    // blockIdx.x = which block this thread belongs to
    // blockDim.x = number of threads per block
    // threadIdx.x = thread's position within its block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't go past the end of our arrays
    if (idx < n) {
        result[idx] = x[idx] + y[idx];
    }
}

int main() {
    // Size of our vectors
    int n = 1000000;  // 1 million elements
    size_t bytes = n * sizeof(float);
    
    printf("Adding two vectors of %d elements\n", n);
    
    // Step 1: Allocate memory on the CPU (host)
    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    float *h_result = (float*)malloc(bytes);
    
    // Step 2: Initialize the input vectors with some values
    for (int i = 0; i < n; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }
    
    // Step 3: Allocate memory on the GPU (device)
    float *d_x, *d_y, *d_result;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_result, bytes);
    
    // Step 4: Copy input data from CPU to GPU
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);
    
    // Step 5: Configure and launch the kernel
    // We need to decide how many threads and blocks to use
    int threadsPerBlock = 256;  // Common choice, GPU can handle many threads
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;  // Ceiling division
    
    printf("Launching kernel with %d blocks and %d threads per block\n", 
           blocksPerGrid, threadsPerBlock);
    
    // Launch the kernel: kernel_name<<<blocks, threads>>>(arguments)
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_result, n);
    
    // Step 6: Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Step 7: Copy result back from GPU to CPU
    cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost);
    
    // Step 8: Verify the result (check first 10 elements)
    printf("Verifying results...\n");
    int errors = 0;
    for (int i = 0; i < 10; i++) {
        printf("h_result[%d] = %.1f (expected 3.0)\n", i, h_result[i]);
        if (h_result[i] != 3.0f) errors++;
    }
    
    if (errors == 0) {
        printf("SUCCESS! All results are correct.\n");
    } else {
        printf("ERROR: Found %d errors\n", errors);
    }
    
    // Step 9: Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    
    // Step 10: Free CPU memory
    free(h_x);
    free(h_y);
    free(h_result);
    
    return 0;
}