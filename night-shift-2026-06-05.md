# 🌙 夜班学习计划 — 2026-06-05

> 总时长：~2h | 主线：B 线基础技术栈（对照笔记读原文）
>
> 📦 **背景**：白天 CUDA 环境已搞定（nvcc 12.4 + GTX 1650），今天夜班趁热打铁，把 PMPP Ch1-2 和 C++ 智能指针/move 语义的原文学一遍。
>
> ⚠️ 所有笔记都存在 `fundamentals/` 下了，手机打开看。

---

## 📋 总览

| # | 主题 | 笔记 | 原文 | ⏱ |
|:--:|------|------|------|:--:|
| 1 | CUDA 异构计算 | Ch01 笔记 | PMPP Ch1 | 15min |
| 2 | CUDA 数据并行 | Ch02 笔记 | PMPP Ch2 | 25min |
| 3 | C++ 智能指针 | smart-pointers 笔记 | EMC++ Item 17-25 | 30min |
| 4 | C++ Move 语义 | move-semantics 笔记 | EMC++ Item 23-29 | 25min |
| 5 | Python 装饰器 | decorators 笔记 | Fluent Python Ch7 | 15min |
| 6 | Python 生成器 | generators 笔记 | Fluent Python Ch14 | 10min |

> **合计：~2h** | 先看笔记再读原文，笔记里的「面试高频问题」重点看
>
> ⚠️ **重要**：夜班学理论 → 明早回电脑做对应手撕练习。练习任务在 [handson-exercises.md](ai-infra-career/fundamentals/handson-exercises.md)

---

## 一、🔴 CUDA — PMPP Ch1（15min）

> 📝 [ch01-heterogeneous-computing.md](ai-infra-career/fundamentals/cuda/pmpp-notes/ch01-heterogeneous-computing.md)

| # | 内容 | ✅ |
|:--:|------|:--:|
| 1.1 | 笔记 §1.1 CPU vs GPU 架构差异图 + 对比表 | ⬜ |
| 1.2 | 笔记 §1.3 Roofline Model — compute-bound vs memory-bound | ⬜ |
| 1.3 | 笔记末尾面试要点自测：为什么 GPU 适合 DL？CUDA 程序 5 步骤？ | ⬜ |

---

## 二、🔴 CUDA — PMPP Ch2（25min）

> 📝 [ch02-data-parallel.md](ai-infra-career/fundamentals/cuda/pmpp-notes/ch02-data-parallel.md)

| # | 内容 | ✅ |
|:--:|------|:--:|
| 2.1 | Kernel launch 语法 `<<<grid, block>>>` + 全局索引计算 | ⬜ |
| 2.2 | Grid→Block→Thread 层级图（重点！） | ⬜ |
| 2.3 | Grid-Stride Loop 模式 — 为什么需要 | ⬜ |
| 2.4 | Warp (32 threads) + Divergence 问题 | ⬜ |
| 2.5 | 对照原文 PMPP Ch2 对应段落 | ⬜ |

---

## 三、🔴 C++ — 智能指针（30min）

> 📝 [smart-pointers.md](ai-infra-career/fundamentals/cpp/smart-pointers.md)

| # | 内容 | ✅ |
|:--:|------|:--:|
| 3.1 | unique_ptr — 独占所有权 + 零开销 | ⬜ |
| 3.2 | shared_ptr — 控制块结构（use_count + weak_count）| ⬜ |
| 3.3 | 对照原文 EMC++ Item 18-21 | ⬜ |
| 3.4 | weak_ptr — 打破循环引用 + lock() 原理 | ⬜ |
| 3.5 | 面试速查表自测 | ⬜ |
| 3.6 | 🔨 明早手撕：unique_ptr 所有权转移 + shared_ptr/weak_ptr 循环引用（练习 1-2） | ⬜ |

---

## 四、🔴 C++ — Move 语义（25min）

> 📝 [move-semantics.md](ai-infra-career/fundamentals/cpp/move-semantics.md)

| # | 内容 | ✅ |
|:--:|------|:--:|
| 4.1 | 左值 vs 右值直觉 + 对照表 | ⬜ |
| 4.2 | vector 拷贝构造 vs 移动构造 代码对比 | ⬜ |
| 4.3 | 对照原文 EMC++ Item 23-24 | ⬜ |
| 4.4 | std::forward + 引用折叠规则 | ⬜ |
| 4.5 | 面试 Q1: move vs forward 区别 | ⬜ |

---

## 五、🟡 Python — 装饰器（15min）

> 📝 [decorators.md](ai-infra-career/fundamentals/python/decorators.md)

| # | 内容 | ✅ |
|:--:|------|:--:|
| 5.1 | 装饰器原理 — 语法糖 `@decorator` = `func = decorator(func)` | ⬜ |
| 5.2 | 带参数装饰器三层嵌套 | ⬜ |
| 5.3 | 闭卷手写 @timer（验证学习效果） | ⬜ |
| 5.4 | 🔨 明早手撕：@timer + @retry 装饰器（练习 3-4） | ⬜ |

---

## 六、🟡 Python — 生成器（10min）

> 📝 [generators.md](ai-infra-career/fundamentals/python/generators.md)

| # | 内容 | ✅ |
|:--:|------|:--:|
| 6.1 | yield 暂停/恢复机制 | ⬜ |
| 6.2 | AI Infra 实战：流式 token 生成 + 惰性数据集加载 | ⬜ |
| 6.3 | 🔨 明早手撕：生成器惰性文件读取（练习 5） | ⬜ |

---

## 📋 完整检查清单

| # | 线 | 内容 | 笔记 | ⏱ | ✅ |
|:--:|:--:|------|------|:--:|:--:|
| 1 | B1 | CUDA 异构计算 | ch01 | 15min | ⬜ |
| 2 | B1 | CUDA 线程模型 | ch02 | 25min | ⬜ |
| 3 | B2 | C++ 智能指针 | smart-pointers | 30min | ⬜ |
| 4 | B2 | C++ Move 语义 | move-semantics | 25min | ⬜ |
| 5 | B2 | Python 装饰器 | decorators | 15min | ⬜ |
| 6 | B2 | Python 生成器 | generators | 10min | ⬜ |

---

## 🔨 明早回电脑·手撕练习配对

> 夜班学完理论，明早趁热打铁。每题从空白文件开始写，写完对照答案。

| 序号 | 🥇 白板手撕 | 对应夜班 | 答案位置 |
|:--:|------|:--:|------|
| 1 | unique_ptr 所有权转移 | #3 智能指针 | [smart-ptr-demo.cpp](ai-infra-career/fundamentals/cpp/code/smart-ptr-demo.cpp) |
| 2 | shared_ptr + weak_ptr 打破循环引用 | #3 智能指针 | 同上 |
| 3 | 手写 @timer 装饰器 | #5 装饰器 | [decorators.md](ai-infra-career/fundamentals/python/decorators.md) |
| 4 | 手写 @retry(times, delay) 装饰器 | #5 装饰器 | 同上 |
| 5 | 生成器惰性文件读取 | #6 生成器 | [generators.md](ai-infra-career/fundamentals/python/generators.md) |
| 6 | 手写 Array<T, N> 类模板 | #3/#4 C++ | [templates.md](ai-infra-career/fundamentals/cpp/templates.md) |

> 完整练习清单 → [handson-exercises.md](ai-infra-career/fundamentals/handson-exercises.md)

---

## 💡 问题记录区

| # | 主题 | 问题 |
|:--:|------|------|
| 1 | | |
| 2 | | |
| 3 | | |

---

## ✅ 白天进展

- [x] CUDA 12.4 环境搭建完成（nvcc + GTX 1650 验证通过）
- [x] 环境搭建文档：[environment-setup.md](ai-infra-career/fundamentals/cuda/environment-setup.md)
- [x] 今日 B 线笔记待产出（白天进行中）
