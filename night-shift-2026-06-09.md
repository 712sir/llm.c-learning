# 🌙 夜班学习计划 — 2026-06-09（周一）

> 总时长：~1.5h | 主线：B1 CUDA 线程模型巩固 + Shared Memory 预习
>
> 📦 **白天已完成**：Softmax CUDA kernel v1-v4 · B2 C++ BoundedBlockingQueue · B2 Python asyncio 实战 · Course Ch2 笔记
>
> ⚠️ **新调整**（见 plan-v2）：D 线 cs224n + E 线 vLLM 暂停，聚焦 B1 CUDA。C 线随 B 线推进，不独立排时间。

---

## 📋 总览

| # | 线 | 内容 | 载体 | ⏱ |
|:--:|:--:|------|------|:--:|
| 1 | B1 | CUDA 课程 Ch2 线程/内存模型精读 | 📝 笔记 | 30min |
| 2 | B1 | Shared Memory + Bank Conflict 预习 | 📝 笔记 | 25min |
| 3 | B2 | 白天 C++ 多线程 + Python asyncio 回顾 | 🧠 闭卷 | 15min |
| 4 | A | 链表回顾 + 数组拓展题思路 | 🧠 脑刷 | 10min |
| 5 | 🎯 | 对齐 plan-v2：本周目标确认 | 📋 5min |

> **合计：~1h25min** | 📝 笔记 55min · 🧠 自测/脑刷 25min · 📋 规划 5min

---

## 一、B1 — CUDA 课程 Ch2 线程/内存模型精读（30min）

> 📝 [ch02-thread-memory-model.md](ai-infra-career/fundamentals/cuda/course/ch02-thread-memory-model.md)

| # | 内容 | ✅ |
|:--:|------|:--:|
| 1.1 | §一 GPU 硬件结构：SM 内部组件 + 各指标数据（A100 vs GTX 1650） | ⬜ |
| 1.2 | §二 三级线程组织：闭卷画 Grid→Block→Thread 层级图 | ⬜ |
| 1.3 | §二 全局索引公式：1D/2D/3D 各写一遍 | ⬜ |
| 1.4 | §三 Warp：最小调度单元 = 32 threads，Divergence 什么时候真正影响性能？ | ⬜ |
| 1.5 | §三 Latency Hiding：GPU 靠什么隐藏内存延迟？为什么需要几万个线程？ | ⬜ |
| 1.6 | §四 内存层次：Register→Shared→L1→L2→Global 延迟倍数默写 | ⬜ |
| 1.7 | §六 自测 6 题（闭卷！） | ⬜ |

---

## 二、B1 — Shared Memory + Bank Conflict 预习（25min）

> 📝 [shared-memory-bank-conflict.md](ai-infra-career/fundamentals/cuda/shared-memory-bank-conflict.md)
>
> ⚠️ 今晚只读 §一~§三，Tiled GEMM（§四）留到明天白天手撕。

| # | 内容 | ✅ |
|:--:|------|:--:|
| 2.1 | §一 内存层次全景图：Shared Memory 在什么位置？为什么比 Global Memory 快 ~100×？ | ⬜ |
| 2.2 | §二 Shared Memory 基础：`__shared__` 声明 + `__syncthreads()` 什么时候必须用？ | ⬜ |
| 2.3 | §二 死锁风险：同一个 warp 内不同线程走不同分支调 `__syncthreads()` = 死锁 | ⬜ |
| 2.4 | §三 Bank Conflict：32 banks / 4B per bank，什么访问模式会产生 conflict？ | ⬜ |
| 2.5 | §三 3 种情况 + Padding 解法：stride=1 / stride=32 / random | ⬜ |
| 2.6 | 思考：今天写的 Softmax v1 kernel，两次 `block_reduce` 有 bank conflict 吗？ | ⬜ |

---

## 三、B2 — 白天产出回顾（15min）

> 📝 闭卷自测，不翻代码

| # | 内容 | ✅ |
|:--:|------|:--:|
| 3.1 | C++：`lock_guard` 和 `unique_lock` 的区别？为什么要用 `while` 而不是 `if` 检查条件？ | ⬜ |
| 3.2 | C++：`memory_order_relaxed` / `acquire` / `release` / `seq_cst` 各自适用场景？ | ⬜ |
| 3.3 | Python：`asyncio.gather` vs `create_task` 的区别？ | ⬜ |
| 3.4 | Python：CPU 密集任务在 asyncio 里怎么办？（关键词：`run_in_executor`） | ⬜ |
| 3.5 | Python：`asyncio.Queue` 为什么不需要 mutex？（关键词：单线程协作式） | ⬜ |

---

## 四、A — 链表回顾 + 数组拓展（10min）

| # | 内容 | ✅ |
|:--:|------|:--:|
| 4.1 | 链表技巧速查：虚拟头 / 快慢指针 / 双指针换轨 / 头插法 | ⬜ |
| 4.2 | LC24 两两交换：脑中画图模拟 dummy→1→2→3 → dummy→2→1→3 | ⬜ |
| 4.3 | 数组拓展题预告（本周 A 线）：LC35 搜索插入位置 / LC34 查找边界 / LC26 删除重复项 | ⬜ |

---

## 五、🎯 对齐 plan-v2（5min）

> 📋 [plan-v2-iterative.md](ai-infra-career/plan-v2-iterative.md)

| # | 检查 | ✅ |
|:--:|------|:--:|
| 5.1 | 本周目标确认：Grid/Block/Thread 模型 + 飞书 CUDA Ch1-2 完成 | ⬜ |
| 5.2 | 今天推进了什么？CUDA 方面比昨天进了一步吗？ | ⬜ |
| 5.3 | 明天最重要的 1 件事：装好 MSVC C++ 工作负载 → 编译 Softmax kernel | ⬜ |

---

## 📋 检查清单

| # | 线 | 内容 | ⏱ | ✅ |
|:--:|:--:|------|:--:|:--:|
| 1 | B1 | CUDA Ch2 线程/内存模型精读 | 30min | ⬜ |
| 2 | B1 | Shared Memory + Bank Conflict 预习 | 25min | ⬜ |
| 3 | B2 | C++ 多线程 + Python asyncio 自测 | 15min | ⬜ |
| 4 | A | 链表回顾 + 数组拓展 | 10min | ⬜ |
| 5 | 🎯 | 对齐 plan-v2 | 5min | ⬜ |

---

## 🔜 明天白天

| 线 | 任务 | 对应夜班 |
|:--:|------|:--:|
| 🛠️ | **装 VS 2022 C++ 桌面开发工作负载** | — |
| B1 | 编译验证 Softmax v1-v4 kernel | #1, #2 |
| B1 | 手写 MatMul naive kernel（白板！） | #2 |
| A | 数组拓展题：LC35 + LC34 + LC26 | #4 |

---

## 💡 问题记录区

| # | 线 | 问题 |
|:--:|:--:|------|
| 1 | B1 | |
| 2 | | |
| 3 | | |
