# 最终性能分析报告

> 基于 nanoGPT + llm.c 的 AI Infra 学习项目

---

## 1. 摘要

- 项目背景：基于 nanoGPT 和 llm.c 的系统性 AI Infra 学习
- 核心工作：从 PyTorch 高层抽象 → 纯 C 实现 → CUDA kernel 优化
- 关键成果：
  - GEMM kernel 从 50 GFLOPS 优化到 ? GFLOPS（?x 提升）
  - 实现 Fused Attention CUDA Kernel
  - 完成 PyTorch C++ Extension 集成

---

## 2. 硬件环境

| 项目 | 详情 |
|------|------|
| GPU 型号 | |
| 显存 | |
| CUDA 版本 | |
| 驱动版本 | |
| nvcc 版本 | |

---

## 3. 技术栈总览

```
┌─────────────────────────────────────┐
│           PyTorch (nanoGPT)          │  ← 高层抽象
├─────────────────────────────────────┤
│           纯 C (llm.c)              │  ← 逐算子手写
├─────────────────────────────────────┤
│         CUDA Kernel (llm.c)          │  ← GPU 优化
├─────────────────────────────────────┤
│     自定义 CUDA Kernel + Extension   │  ← 动手实践
└─────────────────────────────────────┘
```

---

## 4. GEMM 优化历程

### 4.1 各版本对比

| 版本 | 优化手段 | GFLOPS | 利用率 | 相对提升 |
|------|---------|--------|--------|---------|
| V1 Naive | Global memory only | | | 1x |
| V2 Tiled | Shared memory tiling | | | ?x |
| V3 Register Block | Tiling + register reuse | | | ?x |
| V4 Vectorized | float4 coalesced access | | | ?x |
| V5 Double Buffer | Async copy + compute overlap | | | ?x |
| cuBLAS | Vendor optimized | | | 上限 |

### 4.2 Roofline 分析

> 图见 [diagrams/roofline-analysis.png](../diagrams/roofline-analysis.png)

- V1 位置：memory-bound 区域，远低于 bandwidth 上限
- V2 位置：shared memory 降低访存，计算密度提高
- V3-V5：逐步逼近 roofline 的拐点

### 4.3 关键优化技术总结

| 技术 | 原理 | 效果 |
|------|------|------|
| Shared Memory Tiling | 数据复用，减少 global memory 访问 | ~5x |
| Register Blocking | 每个线程多算几个输出，减少 shared 读 | ~2x |
| float4 Vectorization | 128-bit 内存事务，减少指令数 | ~1.5x |
| Double Buffering | 异步加载，隐藏延迟 | ~1.3x |
| Bank Conflict Avoidance | shared memory padding | ~1.1x |

---

## 5. Attention Kernel 分析

### 5.1 标准实现 vs 优化实现

| 指标 | 标准 Attention | Flash Attention | 我的实现 |
|------|---------------|-----------------|---------|
| 显存占用 | O(N^2) | O(N) | |
| 计算时间 | | | |
| 最大支持序列长度 | | | |

### 5.2 Fused Kernel 设计

融合以下操作到一个 kernel：
1. Q @ K^T
2. Scale (1/sqrt(d))
3. Causal Mask
4. Softmax
5. Weighted sum with V

---

## 6. 训练吞吐对比

| 实现 | GPT-2 Small (124M) | GPT-2 Medium (355M) |
|------|-------------------|---------------------|
| nanoGPT (PyTorch) | | |
| llm.c 纯 C (CPU) | | |
| llm.c CUDA | | |
| 自定义 kernel | | |

---

## 7. 关键发现（Top 3 Insights）

1. **____________**：
   
2. **____________**：
   
3. **____________**：

---

## 8. 参考材料

- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [llm.c](https://github.com/karpathy/llm.c)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [PMPP Book](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-323-91231-0)
