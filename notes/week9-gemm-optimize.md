# Week 9-10：GEMM 深度优化 + Attention CUDA

> 状态：🔴 未开始

---

## GEMM 优化进阶技巧

### Bank Conflict 分析与解决

```cuda
// Shared memory 有 32 个 bank，每个 bank 4 bytes
// 当多个线程同时访问同一个 bank 的不同地址 → 串行化
// 
// 解决：加 padding
__shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 打破 bank conflict
```

### Double Buffering

```cuda
// 用两个 shared memory buffer，ping-pong
// 一边计算当前 tile，一边加载下一个 tile
__shared__ float As[2][TILE_SIZE][TILE_SIZE];
// 目的：隐藏 global memory 加载延迟
```

### Warp-level 优化

```cuda
// 用 __shfl_down_sync 做 warp 内归约，比 shared memory 更快
for (int offset = 16; offset > 0; offset /= 2)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
```

---

## Week 10：llm.c 的 Attention CUDA Kernel

### 找到对应文件

```bash
cd llm.c/dev/cuda/
ls
# attention_forward.cu / attention_backward.cu 等
```

### 阅读清单

| 文件 | 内容 | 阅读状态 |
|------|------|---------|
| `attention_forward.cu` | Attention 前向 | [ ] |
| `attention_backward.cu` | Attention 反向 | [ ] |
| `matmul.cu` | MatMul kernel | [ ] |
| `layernorm.cu` | LayerNorm kernel | [ ] |

### Flash Attention 核心思想

1. Tiling：把 Q,K,V 分块加载到 SRAM
2. Recomputation：反向时重新计算 softmax，不存中间结果
3. Online softmax：流式更新 softmax 的分母

### 与标准 Attention kernel 的对比

- 标准实现：O(N^2) 显存（存整个 attention matrix）
- Flash Attention：O(N) 显存（分块计算，不存完整矩阵）

---

## Nsight Compute 使用笔记

### 常用命令

```bash
# Profile 一个 kernel
ncu --set full -o profile.ncu-rep ./your_program

# 查看 kernel 的 stall 原因
ncu --section SpeedOfLight ./your_program

# 查看 memory 访问模式
ncu --section MemoryWorkloadAnalysis ./your_program
```

### Roofline 分析

```
计算密度 = FLOPs / Bytes

如果计算密度 > GPU 的 FLOPs/Bandwidth 比值 → Compute Bound
否则 → Memory Bound
```

### 典型瓶颈

- Memory Bound：latency stall 高，需要增大计算密度（register reuse）
- Compute Bound：可以换 FP16/BF16 提升吞吐
- Occupancy 不足：register 或 shared memory 用太多
