# 每日学习计划

> 📅 2026-06-12（周五）| 状态：🟡
>
> 昨天：Week 2 精读完成 + 24 问全部回答 | 今天：Week 3 自测启动 + CUDA 线程模型深入

---

## 今日目标

> 主线：初级课程 M004。其他块为辅助，控制在 20-30min。

| # | 优先级 | 内容 | 时间 |
|:--:|:--:|------|:--:|
| 1 | 🥇 | **初级课程 M004 ML 基础 + GPU Fundamentals** | 60min |
| 2 | 🥈 | CUDA：线程层次 + 内存层次 + `__syncthreads()` | 30min |
| 3 | 🥈 | 算法：24. 两两交换链表中的节点 | 30min |
| 4 | 🥉 | C++：手撕 unique_ptr | 20min |
| 5 | 🥉 | Python：手撕 @timer | 20min |
| 6 | 🥉 | D2L：第 2 集 | 20min |
| 7 | ⏸️ | llm.c：Week 3 自测（今天不排，优先主线） | — |

---

## Block 1：初级课程 M004（60min）🥇 主线

> `D:\study\ai-infra-junior-engineer-learning\lessons\mod-004-ml-basics\`
>
> 优先做 GPU Fundamentals 练习。按课程顺序推进，完成当前 lesson 的 exercise。

**验收：** 至少完成 1 个 exercise + 1 个 quiz 题

---

## Block 2：CUDA 线程 + 内存层次（30min）🥈

### 线程层次

```
Grid（网格）         ← 一次 kernel launch
  ├── Block 0        ← 共享 Shared Memory
  │     ├── Warp 0   ← 32 Threads，SM 调度最小单位
  │     └── Warp 1
  ├── Block 1
  └── ...
```

| 术语 | 含义 |
|------|------|
| Thread | 最小执行单元 |
| Warp | 32 线程一组，同时执行同一条指令 |
| Block | 最多 1024 线程，共享 Shared Memory |
| Grid | 所有 Block 的集合 |

### 内存层次

```
Local Memory     ← 线程私有，寄存器溢出时用
Shared Memory    ← Block 内共享，比 Global 快 ~100x
Global Memory    ← 所有线程可访问，最慢（cudaMalloc）
```

### `__syncthreads()`

Block 内所有线程到这行必须等——都到了才继续。防止 race condition（A 没写完 B 就读）。

---

## Block 3：算法（30min）🥈

24. 两两交换链表中的节点。Python + C++，先画图再写代码。

---

## Block 4：C++ 手撕（20min）🥉

unique_ptr 所有权转移练习。

---

## Block 5：Python 手撕（20min）🥉

@timer 装饰器练习。

---

## Block 6：D2L 第 2 集（20min）🥉

记一句话收获。

---

## ✅ 今日检查

- [ ] 🥇 初级课程：完成 1 个 exercise + 1 个 quiz
- [ ] 🥈 CUDA：能画图讲 Thread/Warp/Block/Grid + 三种内存 + `__syncthreads`
- [ ] 🥈 算法：24 双版本 AC
- [ ] 🥉 C++：手撕 unique_ptr 跑通
- [ ] 🥉 Python：手撕 @timer 跑通
- [ ] 🥉 D2L：第 2 集看完

---

## 🔍 今日反馈

```
今天做了什么：
________________________________

什么卡住了：
________________________________

明天最重要的 1 件事：
________________________________
```

---

*脑子跑通才算数。* 🧠
