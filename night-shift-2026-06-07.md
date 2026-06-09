# 🌙 夜班学习计划 — 2026-06-07（周日）

> 总时长：~1h50min | 主线：C 线 GPT-2 源码 + B1 Shared Memory

---

## 📋 总览

| # | 线 | 内容 | 载体 | ⏱ |
|:--:|:--:|------|------|:--:|
| 1 | C | GPT-2 Day1: LayerNorm + CausalSelfAttention | 📝 笔记 | 40min |
| 2 | C | GPT-2 Day2: MLP + Block + GPT + generate | 📝 笔记 | 35min |
| 3 | B1 | CUDA Shared Memory + Bank Conflict | 📝 笔记 | 30min |
| 4 | A | 链表脑刷 | 🧠 LeetCode | 10min |

> **合计：~1h50min** | 📝 阅读 100min · 🧠 脑刷 10min

---

## 一、C 线 — nanoGPT 源码面经级详解 Day1（40min）

> 📝 [week2-model-forward.md](ai-infra-career/../llm.c-learning/notes/week2-model-forward.md) 第1-4节

| # | 内容 | ✅ |
|:--:|------|:--:|
| 1.1 | §一 LayerNorm：逐行源码 + Shape 追踪表 + 4 道面试题 | ⬜ |
| 1.2 | §二 CausalSelfAttention `__init__`：Fused QKV + register_buffer | ⬜ |
| 1.3 | §二 CausalSelfAttention `forward`：10 步 Shape 追踪 + 7 连问 | ⬜ |
| 1.4 | §三 Attention 数据流全图（手撕级 ASCII 图） | ⬜ |
| 1.5 | §四 Day1 自测 12 题（闭卷！） | ⬜ |

---

## 二、C 线 — nanoGPT 源码 Day2（35min）

> 📝 [week2-model-forward.md](ai-infra-career/../llm.c-learning/notes/week2-model-forward.md) 第5-12节

| # | 内容 | ✅ |
|:--:|------|:--:|
| 2.1 | §四 MLP：GELU vs ReLU + 膨胀比 4× 的由来 | ⬜ |
| 2.2 | §五 Block：Pre-Norm vs Post-Norm（面试区分度最高的题） | ⬜ |
| 2.3 | §六 GPT 类：Weight Tying + `_init_weights` | ⬜ |
| 2.4 | §七 forward：全链路 Shape 追踪 + 4 连问 | ⬜ |
| 2.5 | §八 generate：Temperature/Top-K/KV Cache | ⬜ |
| 2.6 | §十二 Day2 自测 12 题（闭卷！） | ⬜ |

---

## 三、B1 线 — CUDA Shared Memory + Bank Conflict（30min）

> 📝 [shared-memory-bank-conflict.md](ai-infra-career/fundamentals/cuda/shared-memory-bank-conflict.md)

| # | 内容 | ✅ |
|:--:|------|:--:|
| 3.1 | §一 内存层次全景 + 性能对比表 | ⬜ |
| 3.2 | §二 Shared Memory 基础 + `__syncthreads()` 死锁风险 | ⬜ |
| 3.3 | §三 Bank Conflict：3 种情况 + Padding 解法 | ⬜ |
| 3.4 | §四 Tiled GEMM 代码走读（协作加载 → 片上计算 → 写回） | ⬜ |
| 3.5 | §六 面试拷问 5 题 | ⬜ |

---

## 四、A 线 — 链表脑刷（10min）

| # | 内容 | ✅ |
|:--:|------|:--:|
| 5.1 | 回顾 LC203 三解法思路（直接/虚拟头/递归） | ⬜ |
| 5.2 | LC206 反转链表：脑中模拟双指针 + 头插法 | ⬜ |

---

## 📋 检查清单

| # | 线 | 内容 | ⏱ | ✅ |
|:--:|:--:|------|:--:|:--:|
| 1 | C | GPT-2 Day1 | 40min | ⬜ |
| 2 | C | GPT-2 Day2 | 35min | ⬜ |
| 3 | B1 | Shared Memory | 30min | ⬜ |
| 4 | A | 链表脑刷 | 10min | ⬜ |

---

## 💡 问题记录区

| # | 主题 | 问题 |
|:--:|------|------|
| 1 | | |
| 2 | | |
| 3 | | |
