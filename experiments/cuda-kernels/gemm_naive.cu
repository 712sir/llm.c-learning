// GEMM V1: Naive — Global Memory Only
// 每个线程算 C 的一个元素

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void sgemm_naive_v1(
    float* A, float* B, float* C,
    int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Test harness
int main() {
    int M = 1024, N = 1024, K = 1024;
    // TODO: allocate, initialize, launch, verify, cleanup
    printf("GEMM V1 Naive - M=%d N=%d K=%d\n", M, N, K);
    return 0;
}
