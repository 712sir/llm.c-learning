# 🌙 夜班学习计划 — 2026-06-09（周一）

> 总时长：~2.5h | 主线：C llm.c Week 2 手搓 Attention + B1 CUDA Ch2 收尾
>
> 📦 **白天已完成**：Softmax CUDA kernel v1-v4 · B2 C++ BoundedBlockingQueue · B2 Python asyncio 实战 · Course Ch2 笔记
>
> ⚠️ **今晚调整**：Week 2 笔记已完成阅读，**今晚重点是手搓代码**——关掉 nanoGPT，白板写出 CausalSelfAttention。读十遍不如写一遍。

---

## 📋 总览

| # | 线 | 内容 | 载体 | ⏱ |
|:--:|:--:|------|------|:--:|
| 1 | 🔨 C | **白板手搓 CausalSelfAttention** | 关掉 nanoGPT，从零写 | 60-90min |
| 2 | B1 | CUDA Ch2 线程/内存模型精读（收尾） | 📝 闭卷画图 | 20min |
| 3 | B1 | Softmax kernel 编译验证 | `nvcc softmax_v1_to_v4.cu` | 10min |
| 4 | A | 数组拓展题 LC35 + LC34（脑刷思路） | 🧠 不写代码 | 10min |
| 5 | 🎯 | 复盘 | 📋 | 5min |

> **合计：~2h15min** | 🔨 手搓 90min · 📝 笔记 20min · 🧠 脑刷 10min · 📋 复盘 5min

---

## 一、🔨 白板手搓 CausalSelfAttention（60-90min）

> 📋 [week2-model-forward.md § 手搓代码](../notes/week2-model-forward.md)
>
> ⚠️ **关掉 nanoGPT/model.py。** 只允许看 Week 2 笔记第 4 章的概念解释和 data flow 图。

**验收标准**：`(B=2, T=8, C=64, n_head=4)` 的随机输入跑通，输出 shape = `(2, 8, 64)`。

| # | 步骤 | ✅ |
|:--:|------|:--:|
| 1.1 | `__init__`：`c_attn`（768→2304）+ `c_proj`（768→768）+ causal mask buffer | ⬜ |
| 1.2 | forward step 1：`c_attn(x)` → split → Q, K, V 各 (B,T,C) | ⬜ |
| 1.3 | forward step 2-3：view+transpose → Q@K^T / √d → (B,nh,T,T) | ⬜ |
| 1.4 | forward step 4-5：causal mask（`masked_fill(-inf)`）+ softmax | ⬜ |
| 1.5 | forward step 6-7：att@V → transpose+contiguous+view → c_proj | ⬜ |
| 1.6 | 跑通测试：`torch.randn(2,8,64)` 输入，输出 shape 正确 | ⬜ |
| 1.7 | 打开 nanoGPT/model.py 对比：哪里写的不一样？为什么？ | ⬜ |

> 产物放在 `d:\study\llm.c-learning\experiments\handwrite-gpt\attention.py`

---

## 二、B1 — CUDA Ch2 收尾（20min）

> 今晚重点是手搓，CUDA 只做最小收尾

| # | 内容 | ✅ |
|:--:|------|:--:|
| 2.1 | 闭卷画 Grid→Block→Thread 层级图（3 级） | ⬜ |
| 2.2 | 写出 1D/2D 全局索引公式 | ⬜ |
| 2.3 | 内存层次延迟默写：Register(0) → Shared(~20) → L1(~100) → L2(~200) → Global(~500-800) | ⬜ |

---

## 三、B1 — Softmax kernel 编译验证（10min）

```bash
cd d:\study\ai-infra-career\fundamentals\cuda\kernels\03-softmax
nvcc softmax_v1_to_v4.cu -o softmax_test
./softmax_test
```

确认 v1-v4 输出与 CPU 基线一致。

---

## 四、A — 数组拓展题脑刷（10min）

| # | 题 | 思路 | ✅ |
|:--:|------|------|:--:|
| 4.1 | LC35 搜索插入位置 | 二分查找，找第一个 ≥ target 的位置 | ⬜ |
| 4.2 | LC34 排序数组查找边界 | 两次二分：找左边界 + 右边界 | ⬜ |

> 今晚只脑刷思路，不写代码。白天补代码。

---

## 五、复盘（5min）

| # | 问题 | 答案 |
|:--:|------|------|
| 5.1 | Attention 写出来了吗？卡在哪个 step？ | |
| 5.2 | CUDA 三层线程能闭卷画出来吗？ | |
| 5.3 | 明天白天最重要的 1 件事 | |

---

## 📋 检查清单

| # | 线 | 内容 | ⏱ | ✅ |
|:--:|:--:|------|:--:|:--:|
| 1 | 🔨 C | 白板手搓 CausalSelfAttention | 60-90min | ⬜ |
| 2 | B1 | CUDA Ch2 收尾（画图+索引+内存层次） | 20min | ⬜ |
| 3 | B1 | Softmax kernel 编译验证 | 10min | ⬜ |
| 4 | A | LC35+LC34 脑刷思路 | 10min | ⬜ |
| 5 | 🎯 | 复盘 | 5min | ⬜ |

---

## 🔜 明天白天

| 线 | 任务 | 说明 |
|:--:|------|------|
| 🔨 C | 继续手搓：GPT Block（白板题2）+ GPT forward（白板题3） | 如果今晚 Attention 写通了 |
| B1 | 手写 MatMul naive kernel | CUDA 路线本月目标 |
| A | 数组拓展题：LC35 + LC34 + LC26（C++ + Python） | 代码落实 |
| 🏃 | 跑步 15min | |
