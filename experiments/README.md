# 实验总览

> 按课程周次整理，标注每个实验的状态和产物位置。

## 已完成的实验

| 周次 | 标题 | 实验目录 | 平台 | 说明 |
|------|------|----------|------|------|
| **Week 1 Day 2** | 超参实验 | [hyperparam-search/](hyperparam-search/) | 本地 GTX 1650 | block_size=32 / n_layer=2 / lr=3e-3 三组对比 |
| **Week 1 Day 4-5** | GPT-2 完整训练 | [gpt2-wikitext/](gpt2-wikitext/) | AutoDL RTX 4090D | 124M 模型 / WikiText-103 / 5000 步 / val loss 3.05 |

## 有框架未执行

| 周次 | 标题 | 实验目录 | 待做事项 |
|------|------|----------|----------|
| **Week 7** | 纯 C 性能测试 | [perf-benchmark/](perf-benchmark/) | 跑 `benchmark.sh`，填 results |
| **Week 8** | 第一个 CUDA kernel | [cuda-kernels/](cuda-kernels/) | 补全 `gemm_naive.cu` 的 host 代码，编译运行 |

## 无需实验的周次

| 周次 | 标题 | 原因 |
|------|------|------|
| Week 2 | nanoGPT 源码精读 | 纯代码阅读 |
| Week 3 | 关键问题自测 | 问答式检验 |
| Week 4 | train_gpt2.c 主循环 | 阅读 C 代码结构 |
| Week 5 | 逐算子 C 实现 | 代码实现（非独立实验） |
| Week 6 | 反向传播 | 纯理论推导 |

## 待补充实验

| 周次 | 标题 | 实验目录 | 计划内容 |
|------|------|----------|----------|
| **Week 9-10** | GEMM 优化 + Attention CUDA | [gemm-optimize/](gemm-optimize/) | Tiled GEMM、shared memory 优化、Attention kernel |
| **Week 13-16** | 自定义 CUDA Kernel | [custom-kernels/](custom-kernels/) | LayerNorm / Softmax / Flash Attention |
| **Week 17-18** | 性能分析报告 | [perf-analysis/](perf-analysis/) | Nsight 剖面、roofline 分析、耗时统计 |

---

> 周次总览见 [notes/](../notes/)，Week 1 详细记录见 [notes/week1-environment.md](../notes/week1-environment.md)
