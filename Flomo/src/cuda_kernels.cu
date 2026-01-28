#include <iostream>
#include <cuda_runtime.h>

// Simple CUDA kernel that adds two arrays
__global__ void add_arrays(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Wrapper function to call from C++
extern "C" void run_cuda_addition(float* a, float* b, float* c, int n) {
    float* d_a, * d_b, * d_c;

    // Allocate device memory
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    add_arrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Copy result back to host
    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// Helper function to check CUDA errors
extern "C" void cuda_check_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): "
            << cudaGetErrorString(err) << std::endl;
    }
}
