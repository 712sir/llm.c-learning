# Week 17-18：整理输出 —— 性能分析报告 + 技术博客

> 状态：🔴 未开始

---

## 最终性能分析报告大纲

### 1. 摘要

- 项目背景：基于 nanoGPT + llm.c 的 AI Infra 学习
- 核心工作：从 PyTorch → 纯 C → CUDA kernel 的完整链路
- 关键成果：GEMM 优化提升 ?x，Fused Attention 实现

### 2. 硬件环境

| 项目 | 详情 |
|------|------|
| GPU 型号 | |
| 显存 | |
| CUDA 版本 | |
| 驱动版本 | |

### 3. Baseline 性能

| 实现层 | 吞吐 (tokens/sec) | 瓶颈 |
|--------|-------------------|------|
| PyTorch (GPU) | | |
| PyTorch (CPU) | | |
| llm.c 纯 C | | |
| llm.c CUDA | | |

### 4. GEMM 优化历程

| 版本 | 优化手段 | GFLOPS | 利用率 | 备注 |
|------|---------|--------|--------|------|
| V1 | Naive global memory | | | |
| V2 | Shared memory tiling | | | |
| V3 | Register blocking | | | |
| V4 | float4 vectorized | | | |
| V5 | Double buffering | | | |
| cuBLAS | 厂商实现 | | | 理论上限 |

### 5. Attention Kernel 分析

- 标准实现 vs Flash Attention 的显存对比
- Causal mask 对 tiling 的影响
- 反向传播的计算/访存比

### 6. 关键发现 (Top 3 Insights)

1. 
2. 
3. 

### 7. 参考材料

---

## 技术博客草稿

### 选题方向

- [ ] "从 nanoGPT 到 llm.c：一个 AI Infra 实习生的 18 周学习记录"
- [ ] "手写 GEMM：从 50 GFLOPS 到 2.8 TFLOPS"
- [ ] "Attention 的反向传播：数学推导 + C 实现"
- [ ] "CUDA Shared Memory Tiling 完全图解"

### 草稿

> 待填写

---

## 面试准备

### 核心知识点（脱稿能讲）

1. Attention 的正向和反向传播
2. Transformer Block 的完整结构
3. CUDA 编程模型（thread/block/grid/shared memory）
4. GEMM 优化技术（tiling/register blocking/vectorization）
5. Flash Attention 的核心思想
6. Roofline Model 分析

### 简历要点

- 18 周系统学习 AI Infra，从 PyTorch 到 CUDA kernel
- 读懂 nanoGPT + llm.c 全部核心代码（~5000 行）
- 手写 GEMM kernel，从 50 GFLOPS 优化到 ? GFLOPS
- 实现 Fused Attention CUDA Kernel + PyTorch Extension
- 发布技术博客 ? 篇，GitHub ? stars
