# Week 8：第一个 CUDA kernel —— 从 matmul 开始

> 状态：🔴 未开始

---

## 前置材料阅读进度

- [ ] CUDA C++ Programming Guide Chapter 1-3
- [ ] PMPP 第 4-6 章

### 重要概念速查

| 概念 | 一句话解释 | 代码中的体现 |
|------|-----------|-------------|
| Thread | 最小执行单元 | `threadIdx.x` |
| Warp | 32 个线程为一组，同步执行 | `__syncwarp()` |
| Block | 多个 warp，shared memory 的共享范围 | `blockIdx.x`, `blockDim.x` |
| Grid | 多个 block，一个 kernel 的全部线程 | `gridDim.x` |
| Shared Memory | block 内共享，~48-164KB，低延迟 | `__shared__ float tile[]` |
| Global Memory | 所有线程可访问，高延迟(~800 cycles) | `cudaMalloc` 的空间 |
| Register | 每个线程私有，最快 | 局部变量 |
| Memory Coalescing | 相邻线程访问相邻内存 → 合并为一次事务 | — |
| Bank Conflict | shared memory 的 bank 冲突 → 串行化 | 多线程访问同一 bank 的不同地址 |
| Occupancy | active warps / max warps | 受 register 和 shared memory 限制 |
| `__syncthreads()` | block 内所有线程的屏障 | shared memory 写入后、读取前 |
| Roofline Model | 计算密度 vs 带宽上限分析 | 判断 kernel 是 compute-bound 还是 memory-bound |

---

## Day 1：第一个 CUDA 程序 —— Vector Add

### 代码

```cuda
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    vector_add<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, n);
}
```

### 实验结果

- 编译：`nvcc vector_add.cu -o vector_add`
- 运行：`./vector_add`
- 输出：

---

## Day 2：V1 —— Naive GEMM (Global Memory Only)

### 代码

```cuda
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
```

### 性能分析

- 计算量：(M×N×2K) FLOPs
- 访存量：(M×N×2K×4) bytes
- 计算密度：______ → memory-bound
- 实测 GFLOPS：______

---

## Day 3：V2 —— Shared Memory Tiling

### 关键思想

把 A 和 B 分块加载到 shared memory（延迟 20-30 cycles vs global 300-800）

```cuda
#define TILE_SIZE 16

__global__ void sgemm_tiled_v2(
    float* A, float* B, float* C,
    int M, int N, int K) {
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // 协作加载
        As[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();
        
        // 在 shared memory 上计算
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

### Tiling 示意图

> 图见 [diagrams/gemm-tiling.png](../diagrams/gemm-tiling.png)

### 性能对比

| 版本 | GFLOPS | 相对提升 |
|------|--------|---------|
| V1 Naive | | 1x |
| V2 Tiled | | ?x |

---

## Day 4：V3 —— Register Blocking

### 关键改进

每个线程多算几个输出元素，减少 shared memory 重复读取

```cuda
#define BLOCK_ROWS 4
#define BLOCK_COLS 4
// 每个线程算 BLOCK_ROWS × BLOCK_COLS = 16 个输出
```

---

## Day 5：V4 —— Vectorized Memory Access

### 关键改进

用 `float4` 一次读 4 个 float（128-bit），减少内存指令数

### 所有版本性能汇总

| 版本 | 优化手段 | GFLOPS | 利用率 |
|------|---------|--------|--------|
| V1 Naive | 无 | | |
| V2 Tiled | Shared memory | | |
| V3 Reg Block | Tiling + Register reuse | | |
| V4 Vectorized | V3 + float4 | | |
| cuBLAS | 厂商优化 | | 上限 |
