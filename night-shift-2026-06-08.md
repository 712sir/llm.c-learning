# 🌙 夜班学习计划 — 2026-06-08（周日）

> 总时长：~2h | 主线：B 线巩固白天产出 + C 线 GPT-2 源码 + D 线 cs224n 启动
>
> 📦 **白天已完成**：B1 Softmax CPU 基线 · B2 C++ 模板练习（Array<T,N>/is_same/SFINAE）· B2 Python GIL benchmark（threading vs multiprocessing vs asyncio）

---

## 📋 总览

| # | 线 | 内容 | 载体 | ⏱ |
|:--:|:--:|------|------|:--:|
| 1 | C | GPT-2 源码 Day1 回顾 + Day2 推进 | 📝 笔记 | 35min |
| 2 | B1 | Softmax 数值稳定性复习 + CUDA 线程模型 | 📝 笔记 | 25min |
| 3 | B2 | C++ 模板回顾 + Python GIL 复习 | 📝 白天代码 | 15min |
| 4 | D | cs224n Lec1：NLP 导论 + Word2Vec 动机 | 🎬 视频 | 35min |
| 5 | A | 链表回顾 | 🧠 LeetCode | 10min |

> **合计：~2h** | 📝 阅读笔记 75min · 🎬 视频 35min · 🧠 脑刷 10min
>
> ⚠️ 夜班定位：白天写代码，晚上读源码/看视频/复习。不写新代码。

---

## 一、C 线 — GPT-2 源码 Day1 回顾 + Day2 推进（35min）

> 📝 [week2-model-forward.md](ai-infra-career/../llm.c-learning/notes/week2-model-forward.md)

| # | 内容 | ✅ |
|:--:|------|:--:|
| 1.1 | §一 LayerNorm 源码回顾：mean/var/std/affine 四步 + Shape 追踪 | ⬜ |
| 1.2 | §二 CausalSelfAttention forward：10 步 Shape 变换默写 | ⬜ |
| 1.3 | §三 Attention 数据流全图（闭卷画一遍） | ⬜ |
| 1.4 | §四 MLP：GELU 公式 + 膨胀比 4× 为什么 | ⬜ |
| 1.5 | §五 Block：Pre-Norm vs Post-Norm（面试区分度最高） | ⬜ |
| 1.6 | §六 GPT.forward：全链路 Shape 追踪 `[B,T]→[B,T,V]` | ⬜ |

---

## 二、B1 线 — Softmax 回顾 + CUDA 线程模型（25min）

> 📝 白天写了 [softmax_cpu.cpp](ai-infra-career/fundamentals/cuda/kernels/03-softmax/softmax_cpu.cpp)

| # | 内容 | ✅ |
|:--:|------|:--:|
| 2.1 | 回顾 Softmax 数值稳定性：为什么减去 max？naive 版在什么输入会炸？ | ⬜ |
| 2.2 | 手写 Softmax 三行公式（naive → stable → log_softmax） | ⬜ |
| 2.3 | 2D Softmax（Attention 场景）：每行独立做，shape [B, T, T] | ⬜ |
| 2.4 | 📝 CUDA 课程 Ch2 笔记阅读：Grid/Block/Thread 层级 + 内存层次 | ⬜ |
| 2.5 | 思考：Softmax 在 GPU 上怎么并行？哪些步骤需要 reduce？ | ⬜ |

---

## 三、B2 线 — 白天产出回顾（15min）

> 📝 白天写了 [template-demo.cpp](ai-infra-career/fundamentals/cpp/code/template-demo.cpp) + [gil-benchmark.py](ai-infra-career/fundamentals/python/code/gil-benchmark.py)

| # | 内容 | ✅ |
|:--:|------|:--:|
| 3.1 | C++ 模板：全特化 vs 偏特化 区别？SFINAE 一句话解释？ | ⬜ |
| 3.2 | C++ 模板：`Array<T, N>` 的 N 为什么必须编译期确定？ | ⬜ |
| 3.3 | Python GIL：CPU 密集 8 线程 = 0.93x，8 进程 = 3.67x 说明了什么？ | ⬜ |
| 3.4 | Python GIL：IO 密集 20 线程 = 19.6x，为什么 GIL 不拖后腿？ | ⬜ |
| 3.5 | 闭卷口述：什么时候用 threading / multiprocessing / asyncio？ | ⬜ |

---

## 四、D 线 — cs224n Lec1 预热（35min）

> 🎬 B站搜「cs224n 2024」第一讲 | 📝 [cs224n-learning/](../cs224n-learning/) 新建笔记

| # | 内容 | ✅ |
|:--:|------|:--:|
| 4.1 | NLP 历史三条线：规则系统 → 统计 NLP → 神经网络 | ⬜ |
| 4.2 | Word2Vec 动机：One-hot 有什么问题？分布式表示好在哪？ | ⬜ |
| 4.3 | 课程概览：Assignment 有哪些？Final Project 做什么？ | ⬜ |
| 4.4 | 建立笔记：`cs224n-learning/notes/lec01-intro.md` | ⬜ |

---

## 五、A 线 — 链表脑刷（10min）

| # | 内容 | ✅ |
|:--:|------|:--:|
| 5.1 | LC206 反转链表：三种方法（双指针/头插/递归）脑中过一遍 | ⬜ |
| 5.2 | LC707 设计链表：get/addAtHead/addAtTail/addAtIndex/deleteAtIndex 5 个操作回顾 | ⬜ |
| 5.3 | 链表技巧速查表默念：虚拟头/快慢指针/双指针换轨/头插法 | ⬜ |

---

## 📋 检查清单

| # | 线 | 内容 | ⏱ | ✅ |
|:--:|:--:|------|:--:|:--:|
| 1 | C | GPT-2 Day1 回顾 + Day2 推进 | 35min | ⬜ |
| 2 | B1 | Softmax 复习 + CUDA 线程模型 | 25min | ⬜ |
| 3 | B2 | C++ 模板 + Python GIL 回顾 | 15min | ⬜ |
| 4 | D | cs224n Lec1 预热 | 35min | ⬜ |
| 5 | A | 链表脑刷 | 10min | ⬜ |

---

## 🔜 明天白天预告

> 夜班学完，明早趁热打铁。A 线你自己刷，以下是其他线的安排。

| 线 | 明早任务 | 对应夜班 |
|:--:|------|:--:|
| C | llm.c W1 Day4-5：Wandb 接入 + 5000 步完整训练 | — |
| B1 | 读 CUDA 课程 Ch2 学线程模型 + 开始看 Softmax CUDA kernel 结构 | #2 |
| D | 完善 cs224n Lec1 笔记 | #4 |
| B2 | 笔记更新：templates.md 和 gil-multiprocessing.md 补充白天实验数据 | #3 |

---

## 💡 问题记录区

| # | 线 | 问题 |
|:--:|:--:|------|
| 1 | | |
| 2 | | |
| 3 | | |
