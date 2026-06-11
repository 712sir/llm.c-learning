# 每日学习计划

> 📅 日期：2026-06-11（周四）| Iteration 1 Week 1（6/9–7/6）
> ⏰ 计划学习时长：3-4 小时
> ⚡ CUDA 环境就绪

---

## ⚠️ 原则

- **亲自过目才算完成**。笔记模板是 Claude 建的半成品——只有我读过、理解、能脱稿讲出来，才算真的做完
- **每天五线都碰**。哪怕某条线只投入 20 分钟，也比一天全砸一条线、其他线荒着强

---

## 📊 当前进度快照

| 线 | 真实进度 |
|----|---------|
| 🥇 B1 CUDA | 环境就绪 ✅；线程模型零基础 |
| 🥇 B2 C++ | 智能指针/move 语义/模板 — 未开始 |
| 🥇 B2 Python | 装饰器/生成器 — 未开始 |
| 🥇 D2L 李沐 | 171 集，零基础 |
| 🥈 A 算法 | 代码随想录：链表刷到反转链表 |
| 🥉 C llm.c | Week 1 ✅；Week 2 model.py 读了一部分 |

---

## 📋 需亲自过目的已有材料

> 这些是 Claude 建的骨架，必须自己看一遍才算数

| 优先级 | 文件 | 内容 | 状态 |
|:--:|------|------|:--:|
| 🔴 | [notes/week3-self-test.md](notes/week3-self-test.md) | 10 个核心自测题 | 待看 |
| 🔴 | [notes/week4-llmc-mainloop.md](notes/week4-llmc-mainloop.md) | nanoGPT ↔ llm.c 对照表 | 待看 |
| 🔴 | [fundamentals/cuda/pmpp-notes/ch01](../ai-infra-career/fundamentals/cuda/pmpp-notes/ch01-heterogeneous-computing.md) | PMPP Ch1 笔记 | 待看 |
| 🔴 | [fundamentals/cuda/pmpp-notes/ch02](../ai-infra-career/fundamentals/cuda/pmpp-notes/ch02-data-parallel.md) | PMPP Ch2 笔记 | 待看 |
| 🔴 | [fundamentals/cpp/smart-pointers.md](../ai-infra-career/fundamentals/cpp/smart-pointers.md) | 智能指针笔记 | 待看 |
| 🔴 | [fundamentals/python/decorators.md](../ai-infra-career/fundamentals/python/decorators.md) | 装饰器笔记 | 待看 |
| 🟡 | [handson-exercises.md](../ai-infra-career/fundamentals/handson-exercises.md) | 6 个手撕练习 | 待做 |

---

## 🎯 今日目标

| # | 线 | 目标 | 时间 |
|:--:|----|------|:--:|
| 1 | 🥇 CUDA | 理解 Grid/Block/Thread 三层线程模型 + 临摹 vecAdd kernel | 50min |
| 2 | 🥇 C++ | 读 smart-pointers.md + 手撕练习 1（unique_ptr 所有权转移） | 30min |
| 3 | 🥇 Python | 读 decorators.md + 手撕练习 3（@timer 装饰器） | 25min |
| 4 | 🥇 D2L | 李沐《动手学深度学习 V2》看 1 集（跟当前最相关的） | 25min |
| 5 | 🥈 算法 | 链表 1 题 | 30min |
| 6 | 🥉 llm.c | 精读 model.py：CausalSelfAttention 类 + 写注释 | 40min |

---

## ⏱ 时间块安排

### Block 1：CUDA 线程模型（50 min）🥇

> 📖 文本 + 🎥 视频 | 参考资料：飞书 CUDA 课程 Ch2 + PMPP Ch2

**Step 1：读 PMPP Ch2 笔记（15 min）**
> 打开 [fundamentals/cuda/pmpp-notes/ch02](../ai-infra-career/fundamentals/cuda/pmpp-notes/ch02-data-parallel.md)，**亲自过目**

| # | 读完能回答 | 我的回答 |
|---|-----------|---------|
| 1 | Thread、Block、Grid 分别是什么？层级关系？ | |
| 2 | `blockDim`、`blockIdx`、`threadIdx` 各是什么意思？ | |
| 3 | `<<<gridSize, blockSize>>>` 语法含义？ | |
| 4 | 全局线程索引公式 `i = blockIdx.x * blockDim.x + threadIdx.x` 为什么是这样？ | |
| 5 | Warp 是什么？一个 warp 多少线程？ | |

**Step 2：看飞书 CUDA 课程 Ch2 视频（20 min）**
- 线程模型和显存模型部分
- 边看边在上面表格里填回答

**Step 3：临摹 vecAdd kernel（15 min）**
> 打开 [fundamentals/cuda/pmpp-notes/ch02](../ai-infra-career/fundamentals/cuda/pmpp-notes/ch02-data-parallel.md) 里的 vecAdd 示例，逐行写注释

```cuda
// 【我的理解】
__global__ void vecAdd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // ↑ 这行为什么能定位到正确的数组元素？
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

**验收：** 5 个问题全部用自己的话回答了；vecAdd 代码每行都写了注释

---

### Block 2：C++ 智能指针（30 min）🥇

> 📖 读笔记 + 💻 手撕练习

**Step 1：读笔记（15 min）**
> 打开 [fundamentals/cpp/smart-pointers.md](../ai-infra-career/fundamentals/cpp/smart-pointers.md)，亲自过目

- `unique_ptr`：独占所有权，不能拷贝，只能 move
- `shared_ptr`：共享所有权，引用计数，控制块结构
- `weak_ptr`：不增加引用计数，打破循环引用

**Step 2：手撕练习 1（15 min）**
> 打开 [handson-exercises.md](../ai-infra-career/fundamentals/handson-exercises.md) 练习 1

**需求：** 实现 `unique_ptr` 所有权转移
```
1. 创建 unique_ptr<string> p1 = make_unique<string>("hello")
2. 写 void takeOwnership(unique_ptr<string> p)，打印 *p 和 p==nullptr
3. takeOwnership(std::move(p1))  ← 转移
4. takeOwnership(move(p1))       ← p1 已空，验证
```

**验收：** 代码编译运行通过；能口头解释为什么第二次调用 p1 为 null

---

### Block 3：Python 装饰器（25 min）🥇

> 📖 读笔记 + 💻 手撕练习

**Step 1：读笔记（10 min）**
> 打开 [fundamentals/python/decorators.md](../ai-infra-career/fundamentals/python/decorators.md)，亲自过目

- 装饰器本质是语法糖：`@timer` = `func = timer(func)`
- `functools.wraps` 保留原函数的 `__name__`、`__doc__`
- 带参数的装饰器 = 三层嵌套

**Step 2：手撕练习 3（15 min）**
> 打开 [handson-exercises.md](../ai-infra-career/fundamentals/handson-exercises.md) 练习 3

**需求：** 实现 `@timer` 装饰器
```python
@timer
def slow_add(n):
    return sum(range(n))

result = slow_add(10**7)
# 输出：slow_add took 0.xxxxxxs
# slow_add.__name__ == "slow_add"  ← 关键验收点
```

**验收：** 代码运行通过；`__name__` 没有被改成 `"wrapper"`；能口头解释闭包原理

---

### Block 4：D2L 李沐（25 min）🥇 基石

> 🎥 [动手学深度学习 V2 — 李沐（B站 171 集）](https://b23.tv/IjnkTRm)
>
> 第 1 集已看完 ✅，今天继续第 2 集

**今天看 1 集（选跟当前进度最相关的）：**

| 优先级 | 集数 | 内容 | 为什么选它 |
|:--:|------|------|-----------|
| 🥇 | 第 44 集 | **Transformer + Attention 机制** | 正在读 nanoGPT model.py，直接对照 |
| 🥈 | 继续第 2 集 | 按顺序推进 | 基石积累 |

**看的时候做三件事：**
1. 李沐写了什么代码 → 我能不能自己写一遍？
2. 他讲的跟我读的 nanoGPT model.py 有什么关系？
3. 记一个"今天最大的收获"（一句话）

**验收：** 笔记本里有今天这集的一句话收获 + 一个代码片段

---

### Block 5：算法（30 min）🥈

> **原则：先核心题，再拓展题。** 代码随想录链表篇核心 7 题，当前进度：

| # | 核心 7 题 | 状态 |
|:--:|------|:--:|
| 1 | 203 移除链表元素 | ✅ |
| 2 | 707 设计链表 | ✅ |
| 3 | **206 反转链表** | ✅ ← 上次做到这 |
| 4 | **24 两两交换链表中的节点** | 👈 **今天做** |
| 5 | 19 删除链表的倒数第 N 个节点 | 明天 |
| 6 | 160 链表相交 | |
| 7 | 142 环形链表 II | |

**今天：**[24. 两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/) 🟡

**要求：** Python + C++ 双版本，注释用自己的话写指针移动逻辑；先自己想 10 分钟，再看题解

---

### Block 6：llm.c Week 2 继续（40 min）🥉

> 打开 `D:\study\Project-nanoGPT\model.py`，**自己读，自己写注释**

**今天只读一个类：CausalSelfAttention**

```
阅读顺序：
1. __init__  →  self.c_attn 为什么输出维度是 3 * n_embd？
2. forward  →  Q/K/V 怎么拆出来的？哪一行做了 multi-head reshape？
3. forward  →  scaled_dot_product_attention 传了什么参数？
4. forward  →  c_proj 做什么？残差在哪？
```

**写注释的格式（自己的话）：**
```python
# 【我的理解】c_attn 把输入投影到 Q/K/V 三个空间，拼接成一个向量
# 3 * n_embd = Q(n_embd) + K(n_embd) + V(n_embd)
self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
```

**验收：** CausalSelfAttention 区域有自己写的中文注释；能脱稿说出 forward 的 8 个步骤

---

## ✅ 今日任务清单

### 🥇 CUDA
- [ ] PMPP Ch2 笔记亲自看完，5 个问题全部回答
- [ ] 飞书 CUDA 课程 Ch2 视频看完
- [ ] vecAdd kernel 临摹 + 写注释

### 🥇 C++
- [ ] smart-pointers.md 亲自看完
- [ ] 手撕练习 1：unique_ptr 所有权转移

### 🥇 Python
- [ ] decorators.md 亲自看完
- [ ] 手撕练习 3：@timer 装饰器

### 🥇 D2L
- [x] 第 1 集 ✅
- [ ] 今天继续 1 集
- [ ] 记一句话收获 + 一个代码片段

### 🥈 算法
- [ ] 链表 1 题，Python + C++，自己写注释

### 🥉 llm.c（Week 2 继续，未完成）
- [ ] model.py CausalSelfAttention 读完，自己写中文注释
- [ ] 能脱稿说出 forward 8 步骤

---

## 🔍 今日反馈（睡前 5 分钟）

```
今天实际做了什么（哪条线没碰？）：
________________________________

什么卡住了：
________________________________

今天有什么是我自己理解了的（不是 Claude 替我理解的）：
________________________________

明天最重要的 1 件事：
________________________________
```

---

## 📚 资源附录

### 书籍
| 书名 | 本地位置 | 今天用到 |
|------|---------|---------|
| PMPP（大规模并行处理器编程）Ch1-2 | [ch01](../ai-infra-career/fundamentals/cuda/pmpp-notes/ch01-heterogeneous-computing.md) / [ch02](../ai-infra-career/fundamentals/cuda/pmpp-notes/ch02-data-parallel.md) | CUDA 线程模型 |
| Effective Modern C++ Item 17-25 | — | 智能指针 |
| Fluent Python Ch7 | — | 装饰器 |

### 论文
| 论文 | 链接 | 今天用到 |
|------|------|---------|
| Attention Is All You Need | [arXiv](https://arxiv.org/abs/1706.03762) | llm.c / D2L Transformer 理解 |

### 视频
| 资源 | 链接 | 今天用到 |
|------|------|---------|
| 飞书 CUDA 编程基础（Ch1-2） | [飞书文档](https://tvle9mq8jh.feishu.cn/docx/BnqMdyaJ9oyXb1xwktgc7esMn4c) | CUDA 线程模型+显存模型 |
| 动手学深度学习 V2（李沐） | [B站 171集](https://b23.tv/IjnkTRm) | 第44集 Transformer/Attention |
| Karpathy "Let's build GPT" | [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) | nanoGPT 对照 |

### GitHub 参考项目
| 仓库 | 链接 | 说明 |
|------|------|------|
| nanoGPT | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | 今天精读 model.py |
| llm.c | [karpathy/llm.c](https://github.com/karpathy/llm.c) | 对照 C 实现 |
| CUDA-Learn-Notes | [DefTruth/CUDA-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes) | 200+ kernel，GEMM 参考 |
| how-to-optim-algorithm-in-cuda | [BBuf/how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) | Reduction 优化链 |
| AIInfraGuide | [caomaolufei/AIInfraGuide](https://caomaolufei.github.io/AIInfraGuide/) | 面试题+面经 |

### 本地笔记（今天亲自过目）
| 文件 | 内容 |
|------|------|
| [smart-pointers.md](../ai-infra-career/fundamentals/cpp/smart-pointers.md) | C++ 智能指针 |
| [decorators.md](../ai-infra-career/fundamentals/python/decorators.md) | Python 装饰器 |
| [handson-exercises.md](../ai-infra-career/fundamentals/handson-exercises.md) | 6 个手撕练习 |
| [week3-self-test.md](notes/week3-self-test.md) | 10 个核心自测题 |
| [week4-llmc-mainloop.md](notes/week4-llmc-mainloop.md) | nanoGPT ↔ llm.c 对照表 |

### 计划文件
| 文件 | 内容 |
|------|------|
| [plan.md](../ai-infra-career/plan.md) | 44周总计划 + 周计划 |
| [plan-v2-iterative.md](../ai-infra-career/plan-v2-iterative.md) | 迭代执行模型 |
| [fundamentals/plan.md](../ai-infra-career/fundamentals/plan.md) | B线详细路径 |

---

## 🌟 明日预告

| 线 | 内容 |
|----|------|
| 🥇 CUDA | Shared Memory + Bank Conflict（飞书 CUDA 课程 Ch3） |
| 🥇 C++ | move 语义 + 手撕练习 |
| 🥇 Python | 生成器 + 手撕练习 5 |
| 🥇 D2L | 继续李沐 1 集 |
| 🥈 算法 | 链表 1 题 |
| 🥉 llm.c | model.py：MLP + Block + GPT 类 |

---

*计划创建时间：2026-06-11*
*五条线都碰，每天推进。脑子跑通才算数。* 🧠
