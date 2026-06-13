# 每日学习计划

> 📅 2026-06-12（周五）| 状态：🟡
>
> 昨天：Week 2 精读完成 + 24 问全部回答 | 今天：Week 3 自测启动 + CUDA 线程模型深入

---

## 今日目标

| # | 线 | 内容 | 时间 |
|:--:|----|------|:--:|
| 1 | 🥇 CUDA | 线程层次 + 内存层次 + `__syncthreads()` 搞懂 | 45min |
| 2 | 🥇 C++ | 手撕 unique_ptr 所有权转移 | 25min |
| 3 | 🥇 Python | 手撕 @timer 装饰器 | 25min |
| 4 | 🥇 D2L | 第 2 集 | 25min |
| 5 | 🥈 算法 | 24. 两两交换链表中的节点 | 30min |
| 6 | 🥇 初级课程 | M004 ML 基础 — GPU Fundamentals | 35min |
| 7 | 🥉 llm.c | Week 3 自测题 Q1-Q4（Attention 四问） | 35min |

---

## Block 1：CUDA 线程 + 内存层次（45min）🥇

> 昨天的问题：Thread/Block/Grid、Local/Shared/Global、`__syncthreads()` 没搞懂

### 线程层次

```
Grid（网格）          ← 整个 GPU 上跑一次 kernel 就是一个 Grid
  ├── Block 0         ← 一个 Block 里的线程共享 Shared Memory
  │     ├── Thread 0
  │     ├── Thread 1
  │     └── ...       ← 32 个 Thread 组成一个 Warp（最小调度单位）
  ├── Block 1
  └── Block 2
```

| 术语 | 含义 | 硬件对应 |
|------|------|------|
| Thread | 最小执行单元，一个线程处理一个数据元素 | CUDA Core |
| Warp | 32 个线程一组，同时执行同一条指令 | SM 调度单位 |
| Block | 一组线程（最多 1024 个），共享 Shared Memory | 一个 SM 上运行 |
| Grid | 所有 Block 的集合，一次 kernel launch | 整个 GPU |

### 内存层次

```
Thread 私有    Local Memory（寄存器溢出时用）   最快，容量最小
Block 共享     Shared Memory（__shared__）     比 Global 快 100 倍，Block 内线程共享
Block 内同步   __syncthreads()                  所有线程到这行必须等，都到了才继续
全局           Global Memory（cudaMalloc 分配的） 最慢，所有线程都能访问
```

### `__syncthreads()` 干什么

```cuda
__global__ void example() {
    // ① 每个线程往 shared memory 写数据
    shared[tid] = data[tid];

    __syncthreads();  // ② 屏障：必须等所有线程写完，才能往下走

    // ③ 每个线程读取其他线程写的数据
    int val = shared[blockDim.x - 1 - tid];
}
```

不加 `__syncthreads()`：线程 A 还没写完，线程 B 就去读了 → 读到垃圾数据（race condition）。

**验收：** 能画图讲清 Thread→Warp→Block→Grid 的关系；能解释为什么 `__syncthreads()` 只在 Block 内有效

---

## Block 2：C++ 手撕 unique_ptr（25min）🥇

> 昨天读了 smart-pointers.md，今天动手写

**练习题：**

```cpp
// 创建 unique_ptr<string> p1 = make_unique<string>("hello")
// 写函数 void takeOwnership(unique_ptr<string> p)，打印 *p 和 p==nullptr
// 验证 move 后 p1 变为 nullptr
```

**验收：** 编译运行，输出显示第一次调用有值、第二次为空

---

## Block 3：Python 手撕 @timer（25min）🥇

```python
# 用 functools.wraps 保留 __name__
# slow_add(10**7) 输出 "slow_add took 0.xxxxxxs"
# slow_add.__name__ 仍是 "slow_add" 不是 "wrapper"
```

**验收：** 代码能跑，`__name__` 正确

---

## Block 4：D2L 第 2 集（25min）🥇

继续李沐。记一句话收获。

---

## Block 5：算法 — 24. 两两交换链表中的节点（30min）🥈

> 链表核心第 4 题。练虚拟头节点 + 多指针操作。

```
1 → 2 → 3 → 4    →    2 → 1 → 4 → 3
```

- 先自己想 10 分钟，画指针图
- Python + C++ 双版本
- 注释用自己的话写指针移动逻辑

---

## Block 6：初级课程 M004（35min）🥇

> 打开 `D:\study\ai-infra-junior-engineer-learning\lessons\mod-004-ml-basics\`
>
> 优先做 GPU Fundamentals 练习。

---

## Block 7：Week 3 自测 Q1-Q4（35min）🥉

> 打开 [notes/week3-self-test.md](notes/week3-self-test.md)
>
> 闭卷手写回答 Q1-Q4（Attention 四问）

---

## ✅ 今日检查

- [ ] CUDA：能画图讲清 Thread/Warp/Block/Grid + 三种内存层次 + `__syncthreads`
- [ ] C++：手撕 unique_ptr 跑通
- [ ] Python：手撕 @timer 跑通
- [ ] D2L：第 2 集看完
- [ ] 算法：24 双版本 AC
- [ ] 初级：M004 GPU Fundamentals 练习
- [ ] Week 3：Q1-Q4 手写回答

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
