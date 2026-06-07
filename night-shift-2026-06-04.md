# 🌙 夜班学习计划 — 2026-06-04

> 聚焦：**B 线基础技术栈** | 总时长：~2.5h | 方式：笔记对照原文精读
>
> 🎯 **今日核心产出**：建立 CUDA 编程思维模型 + 吃透 C++ 智能指针/move 语义 + 掌握 Python 装饰器/生成器
>
> ⚠️ 已为你准备好 7 份笔记（`fundamentals/cuda/`、`fundamentals/cpp/`、`fundamentals/python/`），下面按优先级排列。**先看笔记再读原文**，笔记里标了面试高频点。

---

## 📋 总览

| # | 主题 | 笔记文件 | 原文 | ⏱ |
|:--:|------|------|------|:--:|
| 1 | CUDA 异构计算导论 | [cuda/pmpp-notes/ch01](ai-infra-career/fundamentals/cuda/pmpp-notes/ch01-heterogeneous-computing.md) | PMPP Ch1 | 25min |
| 2 | CUDA 数据并行 + 线程模型 | [cuda/pmpp-notes/ch02](ai-infra-career/fundamentals/cuda/pmpp-notes/ch02-data-parallel.md) | PMPP Ch2 | 30min |
| 3 | C++ 智能指针 | [cpp/smart-pointers](ai-infra-career/fundamentals/cpp/smart-pointers.md) | Effective Modern C++ Item 17-25 | 35min |
| 4 | C++ Move 语义 | [cpp/move-semantics](ai-infra-career/fundamentals/cpp/move-semantics.md) | Effective Modern C++ Item 23-29 | 30min |
| 5 | Python 装饰器 | [python/decorators](ai-infra-career/fundamentals/python/decorators.md) | Fluent Python Ch7 | 20min |
| 6 | Python 生成器 | [python/generators](ai-infra-career/fundamentals/python/generators.md) | Fluent Python Ch14 | 20min |
| 7 | 回顾 + 问题整理 | — | — | 10min |

---

## 一、🔴 B1 CUDA — PMPP Ch1（25min）

> 📝 笔记：[fundamentals/cuda/pmpp-notes/ch01-heterogeneous-computing.md](ai-infra-career/fundamentals/cuda/pmpp-notes/ch01-heterogeneous-computing.md)
> 📖 原文：PMMP 4th Edition, Chapter 1: Introduction

### 对照学习路径

| 步骤 | 内容 | 时长 |
|:--:|------|:--:|
| 1.1 | **先读笔记** §1.1「CPU vs GPU 架构差异」— 看那张 ASCII 图和对比表，建立直觉 | 5min |
| 1.2 | **对照原文** PMPP Ch1 同段落 — 重点：图 1.x 的晶体管分配图、延迟 vs 吞吐 | 10min |
| 1.3 | **笔记** §1.3「Roofline Model」— 理解 compute-bound vs memory-bound 的区别 | 5min |
| 1.4 | **笔记末尾**「面试向要点」— 4 个问题自问自答，检验理解 | 5min |

### 🎯 学完检查

- [ ] 能用一句话解释「为什么 GPU 适合深度学习」
- [ ] 能画出 Roofline Model 的两条线并标出含义
- [ ] 能说出 CUDA 程序的 5 个步骤

---

## 二、🔴 B1 CUDA — PMPP Ch2（30min）

> 📝 笔记：[fundamentals/cuda/pmpp-notes/ch02-data-parallel.md](ai-infra-career/fundamentals/cuda/pmpp-notes/ch02-data-parallel.md)
> 📖 原文：PMMP 4th Edition, Chapter 2: Heterogeneous Data Parallel Computing

### 对照学习路径

| 步骤 | 内容 | 时长 |
|:--:|------|:--:|
| 2.1 | **先读笔记** §2.1「数据并行的概念」— SIMD/SIMT 的区别 | 5min |
| 2.2 | **笔记** §2.2「Host vs Device 代码」— `__global__`/`__device__` 关键字 + kernel launch 语法 `<<<>>>` | 8min |
| 2.3 | **笔记** §2.3「Grid→Block→Thread 层次」— **这是今晚最重要的一个概念！** 看懂那张 ASCII 层级图 | 10min |
| 2.4 | **笔记** §2.3「Grid-Stride Loop」— 理解为什么需要这个模式，别死记硬背 | 5min |
| 2.5 | **笔记** §2.3「Warp Divergence」— 理解为什么 if-else 在 GPU 上很贵 | 2min |

### 🎯 学完检查

- [ ] 能算出 `blockIdx.x * blockDim.x + threadIdx.x` 的含义
- [ ] 能解释 Grid-Stride Loop 解决什么问题
- [ ] 能说出 Warp 大小（32）和 Divergence 的后果

---

## 三、🔴 B2 C++ — 智能指针（35min）

> 📝 笔记：[fundamentals/cpp/smart-pointers.md](ai-infra-career/fundamentals/cpp/smart-pointers.md)
> 📖 原文：《Effective Modern C++》Item 17–25（有中英文两版 PDF）
> 📂 文件：`resources/books/04-Effective-Modern-C++-zh.pdf` / `-en.pdf`

### 对照学习路径

| 步骤 | 内容 | 时长 |
|:--:|------|:--:|
| 3.1 | **先读笔记**「为什么需要智能指针」— 5 个裸指针的坑 | 3min |
| 3.2 | **笔记**「unique_ptr」— 理解独占所有权 + 零开销 + 自定义 deleter | 8min |
| 3.3 | **对照原文** Item 18（unique_ptr）、Item 21（make_unique vs new） | 10min |
| 3.4 | **笔记**「shared_ptr」— 重点看那张内存布局图（控制块！） | 8min |
| 3.5 | **对照原文** Item 19（shared_ptr）、Item 21（make_shared） | 5min |
| 3.6 | **笔记**「weak_ptr」— 循环引用场景 + lock() 原理 | 5min |

### 🎯 学完检查

- [ ] 能画出 shared_ptr 的控制块结构（use_count + weak_count + deleter）
- [ ] 能解释为什么 `make_shared` 比 `new + shared_ptr` 好
- [ ] 能写出打破循环引用的代码（Node 例子）

---

## 四、🔴 B2 C++ — Move 语义（30min）

> 📝 笔记：[fundamentals/cpp/move-semantics.md](ai-infra-career/fundamentals/cpp/move-semantics.md)
> 📖 原文：《Effective Modern C++》Item 23–29

### 对照学习路径

| 步骤 | 内容 | 时长 |
|:--:|------|:--:|
| 4.1 | **先读笔记**「左值 vs 右值」— 建立基础直觉 | 5min |
| 4.2 | **笔记**「为什么需要 Move」— vector 的拷贝构造 vs 移动构造，看代码对比 | 8min |
| 4.3 | **对照原文** Item 23（std::move）、Item 24（std::forward） | 7min |
| 4.4 | **笔记**「std::forward 与万能引用」— 引用折叠规则是核心，别慌，记住那个表 | 8min |
| 4.5 | **笔记**「面试高频问题」— Q1 move vs forward 的区别（最重要） | 2min |

### 🎯 学完检查

- [ ] 能用一句话区分 move 和 forward（unconditional vs conditional）
- [ ] 知道什么时候必须加 `noexcept`（移动构造！）
- [ ] 理解万能引用 `T&&` 的识别方法（有类型推导才是万能引用）

---

## 五、🟡 B2 Python — 装饰器（20min）

> 📝 笔记：[fundamentals/python/decorators.md](ai-infra-career/fundamentals/python/decorators.md)
> 📖 原文：《Fluent Python》2nd Edition, Chapter 7: Decorators and Closures

### 对照学习路径

| 步骤 | 内容 | 时长 |
|:--:|------|:--:|
| 5.1 | **先读笔记**「闭包」— 理解函数捕获外部变量 | 3min |
| 5.2 | **笔记**「装饰器原理」— 语法糖 `@decorator` = `func = decorator(func)` | 5min |
| 5.3 | **笔记**「带参数的装饰器」— 三层嵌套，一层层拆 | 7min |
| 5.4 | **笔记**「实用装饰器」— @timer / @cache / @retry，面试手写高频 | 5min |

### 🎯 学完检查

- [ ] 能手写出 `@timer` 装饰器的完整代码（闭卷！）
- [ ] 知道 `functools.wraps` 为什么是必须的
- [ ] 理解带参数装饰器的三层嵌套结构

---

## 六、🟡 B2 Python — 生成器（20min）

> 📝 笔记：[fundamentals/python/generators.md](ai-infra-career/fundamentals/python/generators.md)
> 📖 原文：《Fluent Python》2nd Edition, Chapter 14: Iterables, Iterators, and Generators

### 对照学习路径

| 步骤 | 内容 | 时长 |
|:--:|------|:--:|
| 6.1 | **先读笔记**「迭代器协议」— `__iter__` + `__next__` + StopIteration | 5min |
| 6.2 | **笔记**「yield 核心」— 生成器的暂停/恢复魔法 | 5min |
| 6.3 | **笔记**「AI Infra 实际用途」— 流式 token 生成 + 惰性数据集加载，**这两个例子是你的面试弹药** | 8min |
| 6.4 | **笔记**「面试高频问题」— 快速过 | 2min |

### 🎯 学完检查

- [ ] 能解释「生成器用一次就耗尽」的原因
- [ ] 能写出流式 token 生成的伪代码
- [ ] 知道 `yield from` 的作用

---

## 📋 完整检查清单

| # | 线 | 内容 | 笔记 | 原文 | ⏱ | ✅ |
|:--:|:--:|------|------|------|:--:|:--:|
| 1 | B1 | CUDA 异构计算 | ch01 笔记 | PMPP Ch1 | 25min | ⬜ |
| 2 | B1 | CUDA 线程模型 | ch02 笔记 | PMPP Ch2 | 30min | ⬜ |
| 3 | B2 | C++ 智能指针 | smart-pointers 笔记 | EMC++ Item 17-25 | 35min | ⬜ |
| 4 | B2 | C++ Move 语义 | move-semantics 笔记 | EMC++ Item 23-29 | 30min | ⬜ |
| 5 | B2 | Python 装饰器 | decorators 笔记 | Fluent Python Ch7 | 20min | ⬜ |
| 6 | B2 | Python 生成器 | generators 笔记 | Fluent Python Ch14 | 20min | ⬜ |
| 7 | — | 回顾 + 整理问题给 Claude | — | — | 10min | ⬜ |

> **合计：~2h50min** | 🔴 B1 CUDA 55min · 🔴 B2 C++ 65min · 🟡 B2 Python 40min · 📝 回顾 10min
>
> 🎯 **今日核心产出**：理解 GPU 并行编程模型 + 吃透 C++ 智能指针和 move 语义 + 掌握 Python 装饰器和生成器

---

## 💡 问题记录区

> 学习过程中的困惑写在这里，明天统一发给我。

| # | 主题 | 问题 |
|:--:|------|------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

---

## ⚠️ 明日回电脑待办

- [ ] B1 CUDA Toolkit 完整安装（nvcc 编译器 — pip 包不包含，需官方 installer）
- [ ] 编译测试 `vec_add.cu`
- [ ] A 线：链表核心 7 题 C++ + Python 双版本
- [ ] C 线 llm.c：W1 Day4 Wandb + 5000 步训练
