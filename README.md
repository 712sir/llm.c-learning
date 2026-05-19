# AI Infra 学习笔记 —— 从 PyTorch 到 CUDA Kernel

> 核心原则：以"同一个算子，两层实现"的方式交替进行。学了 Attention 的 PyTorch 版，立刻去 llm.c 找到对应的 C 和 CUDA 版对照着看，建立"高层抽象→底层实现"的映射关系。

---

## 我做了什么

在 18 周内，从 [nanoGPT](https://github.com/karpathy/nanoGPT) 的 PyTorch 实现出发，逐层深入 [llm.c](https://github.com/karpathy/llm.c) 的纯 C 实现，最终到 CUDA kernel 级别：

- 读懂并注释了 nanoGPT 全部核心代码
- 逐算子对照了 llm.c 的 C 实现并手写了反向传播公式
- 优化了 GEMM kernel：从 50 GFLOPS → ? TFLOPS
- 自己实现了 Fused Attention CUDA Kernel

---

## 学习路线图

| 阶段 | 时间 | 核心目标 | 笔记 |
|------|------|---------|------|
| 一：跑通+读懂 | 第 1-3 周 | 跑通两个项目，读懂 nanoGPT 每一行代码 | [笔记 →](notes/week1-environment.md) |
| 二：拆解对照 | 第 4-7 周 | 将 llm.c 的每个算子与 nanoGPT 对照，读懂纯 C 实现 | [笔记 →](notes/week4-llmc-mainloop.md) |
| 三：CUDA 攻坚 | 第 8-12 周 | 读懂所有 CUDA kernel，能 benchmark 并分析瓶颈 | [笔记 →](notes/week8-first-kernel.md) |
| 四：动手改造 | 第 13-16 周 | 自己写/优化 kernel，做 PyTorch 扩展集成 | [笔记 →](notes/week13-custom-kernel.md) |
| 五：整理输出 | 第 17-18 周 | 写性能分析报告 + 技术博客，准备面试 | [笔记 →](notes/week17-final-report.md) |

---

## 仓库导航

```
llm.c-learning/
├── notes/              # 学习笔记（每周一篇）
├── experiments/        # 实验代码 + 数据
│   ├── hyperparam-search/   # 超参实验
│   ├── perf-benchmark/      # 性能测试
│   └── cuda-kernels/        # CUDA kernel 实验
├── diagrams/           # 手绘图（架构图、数据流图）
├── reports/            # 阶段总结 + 最终报告
└── .github/            # CI workflows
```

---

## 关键技术栈

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![C](https://img.shields.io/badge/C-00599C?style=flat&logo=c&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white)
![Nsight Compute](https://img.shields.io/badge/Nsight_Compute-76B900?style=flat&logo=nvidia&logoColor=white)
![cuBLAS](https://img.shields.io/badge/cuBLAS-76B900?style=flat&logo=nvidia&logoColor=white)

---

## 相关仓库

- [nanoGPT-annotated](https://github.com/用户名/nanoGPT) — Fork nanoGPT，完整中文注释版
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — 原始 nanoGPT
- [karpathy/llm.c](https://github.com/karpathy/llm.c) — 原始 llm.c

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/用户名/llm.c-learning.git

# 浏览笔记
cd llm.c-learning/notes/
ls

# 查看实验
cd experiments/
```
