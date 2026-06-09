# 🌙 夜班学习计划 — 2026-06-09（周一）

> 总时长：~2h | 主线：B1 CUDA Ch2 线程模型巩固 + C 线 GPT-2 源码 + D 线 cs224n
>
> 📦 **白天已完成**：B1 CUDA 课程 Ch2 笔记框架 · C 线 llm.c wandb + 5000步训练 · D 线 cs224n Lec1
> 
> ⚠️ 所有笔记都在 `ai-infra-career/` 下，夜班用手机打开看。不写代码。

---

## 📋 总览

| # | 线 | 内容 | 载体 | ⏱ |
|:--:|:--:|------|------|:--:|
| 1 | C | GPT-2 源码 Day1 精读（LayerNorm + CausalSelfAttention） | 📝 笔记 | 35min |
| 2 | B1 | CUDA 线程模型巩固 + Softmax CUDA kernel 结构预览 | 📝 笔记 | 30min |
| 3 | B2 | C++ 模板 + Python GIL 面试自测 | 🧠 闭卷 | 15min |
| 4 | D | cs224n Lec2：Word2Vec / Skip-gram / CBOW | 🎬 视频 | 40min |
| 5 | A | 链表 206/707 回顾 + 24 两两交换思路 | 🧠 脑刷 | 10min |

> **合计：~2h10min** | 📝 笔记阅读 65min · 🎬 视频 40min · 🧠 脑刷/自测 25min

---

## 一、C 线 — GPT-2 源码 Day1 精读（35min）

> 📝 [week2-model-forward.md](llm.c-learning/notes/week2-model-forward.md) 第1-4节

| # | 内容 | ✅ |
|:--:|------|:--:|
| 1.1 | §一 LayerNorm：mean/var/std/affine 四步 + Shape 追踪 | ⬜ |
| 1.2 | §二 CausalSelfAttention `__init__`：Fused QKV projection + register_buffer 作用 | ⬜ |
| 1.3 | §二 CausalSelfAttention `forward`：10 步 Shape 变换（B,T → B,T,C）| ⬜ |
| 1.4 | §三 Attention 数据流 ASCII 图：QKV → scores → softmax → weighted sum | ⬜ |
| 1.5 | §四 Day1 自测：causal mask 为什么是下三角？多头体现在代码哪行？ | ⬜ |

---

## 二、B1 线 — CUDA 线程模型巩固 + Softmax kernel 预览（30min）

> 📝 白天写了 [ch02-thread-memory-model.md](ai-infra-career/fundamentals/cuda/course/ch02-thread-memory-model.md)

| # | 内容 | ✅ |
|:--:|------|:--:|
| 2.1 | 闭卷画 Grid→Block→Thread→Warp 层级图 | ⬜ |
| 2.2 | 手写 1D/2D 全局索引计算公式 | ⬜ |
| 2.3 | 口述：Warp Divergence 什么时候影响性能？同一 warp 内 vs 不同 warp 间 | ⬜ |
| 2.4 | 口述：GPU 怎么隐藏内存延迟？为什么需要几万个线程？ | ⬜ |
| 2.5 | 内存层次表默写：Register→Shared→L1→L2→Global 延迟倍数 | ⬜ |
| 2.6 | 📝 **预览**：[shared-memory-bank-conflict.md](ai-infra-career/fundamentals/cuda/shared-memory-bank-conflict.md) §一~§三 | ⬜ |
| 2.7 | 思考：Softmax 在 GPU 上怎么并行？哪一步需要线程间通信？ | ⬜ |

---

## 三、B2 线 — 面试自测（15min）

> 📝 闭卷！不看笔记，口述以下问题

| # | 内容 | ✅ |
|:--:|------|:--:|
| 3.1 | C++ 模板：全特化和偏特化的区别？各举一个例子 | ⬜ |
| 3.2 | C++ 模板：SFINAE 一句话解释？`enable_if` 解决了什么问题？ | ⬜ |
| 3.3 | Python GIL：CPU 密集 8 线程为什么只有 0.9x？8 进程为什么 3.7x？ | ⬜ |
| 3.4 | Python GIL：IO 密集 20 线程为什么 19.6x？GIL 在 IO 时发生了什么？ | ⬜ |
| 3.5 | 一句话决策：什么时候用 threading / multiprocessing / asyncio？ | ⬜ |

---

## 四、D 线 — cs224n Lec2：Word2Vec（40min）

> 🎬 B站搜「cs224n 2024」第二讲 | 📝 建立笔记

| # | 内容 | ✅ |
|:--:|------|:--:|
| 4.1 | Word2Vec 两种架构：Skip-gram（中心词预测上下文）vs CBOW（上下文预测中心词） | ⬜ |
| 4.2 | 目标函数推导：`J(θ) = -1/T Σ log P(w_{t+j}|w_t)` | ⬜ |
| 4.3 | Softmax 的计算瓶颈：分母要对整个词表求和（|V| 太大） | ⬜ |
| 4.4 | Negative Sampling：把多分类变成二分类（正样本 vs 噪声） | ⬜ |
| 4.5 | Hierarchical Softmax：用 Huffman 树把 O(|V|) 降到 O(log|V|) | ⬜ |
| 4.6 | 面试拷问：Word2Vec 为什么有两个向量矩阵（输入+输出）？用哪个？ | ⬜ |

---

## 五、A 线 — 链表脑刷（10min）

| # | 内容 | ✅ |
|:--:|------|:--:|
| 5.1 | LC206 反转链表：三种方法（双指针/头插/递归）脑中模拟 | ⬜ |
| 5.2 | LC707 双链表：addAtHead 和 addAtTail 的四行代码对照 | ⬜ |
| 5.3 | LC24 两两交换：画图模拟指针操作（dummy→1→2→3 → dummy→2→1→3） | ⬜ |

---

## 📋 检查清单

| # | 线 | 内容 | ⏱ | ✅ |
|:--:|:--:|------|:--:|:--:|
| 1 | C | GPT-2 Day1 精读 | 35min | ⬜ |
| 2 | B1 | CUDA 线程模型 + Softmax 预览 | 30min | ⬜ |
| 3 | B2 | C++/Python 面试自测 | 15min | ⬜ |
| 4 | D | cs224n Lec2 Word2Vec | 40min | ⬜ |
| 5 | A | 链表脑刷 | 10min | ⬜ |

---

## 🔜 明天白天预告

| 线 | 任务 | 对应夜班 |
|:--:|------|:--:|
| B1 | Shared Memory 笔记精读 + 手写 Softmax CUDA kernel v1（单线程块） | #2 |
| B2 | C++ 多线程：mutex/atomic/condition_variable 手撕练习 | #3 |
| C | GPT-2 Day2：MLP + Block + GPT.forward + generate | #1 |
| D | cs224n Lec1-2 笔记整理 | #4 |
| A | 链表继续刷（你自己推进） | #5 |

---

## 💡 问题记录区

| # | 线 | 问题 |
|:--:|:--:|------|
| 1 | | |
| 2 | | |
| 3 | | |
