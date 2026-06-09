# 🌙 夜班学习计划 — 2026-06-10（周二）

> 总时长：~5h（夜班整晚） | 主线：B1 MatMul kernel + B1 Shared Memory + 🔨 手搓 Block
>
> 📦 **白天已完成**：⬜ MatMul · ⬜ 算法 · ⬜ 跑步
>
> ⚠️ 今晚是 Month 1 的 MatMul 攻坚战——本月目标是「能白板写出 GEMM naive kernel」。今晚拿下它。

---

## 📋 总览

| # | 线 | 内容 | 载体 | ⏱ |
|:--:|:--:|------|------|:--:|
| 1 | B1 | **手写 MatMul naive kernel（白板！）** | 🔨 CUDA C++ | 90min |
| 2 | B1 | Shared Memory + Bank Conflict 完整学习 | 📝 笔记 | 45min |
| 3 | 🔨 C | 白板手搓 GPT Block（Week 2 白板题2） | 🐍 PyTorch | 45min |
| 4 | A | LC35 + LC34 + LC26 代码实现 | 💻 C++ + Python | 45min |
| 5 | B1 | Reduction 复习 + Softmax kernel 编译 | 🛠️ | 20min |
| 6 | 🎯 | 复盘 | 📋 | 5min |

> **合计：~4h10min** | 🔨 手搓 135min · 📝 笔记 45min · 💻 算法 45min · 🛠️ 20min · 📋 5min

---

## 一、🔨 手写 MatMul naive kernel（90min）⭐ 今晚最重要

> 📖 参考：[CUDA-Learn-Notes GEMM 基础版](https://github.com/DefTruth/CUDA-Learn-Notes)
>
> ⚠️ 先临摹再白板。白板时不看任何代码，只凭理解写。

### Phase A：CPU 版 MatMul（先理解算法，10min）

```c
// matmul_cpu.c
// A: M×K, B: K×N, C: M×N
// C[i][j] = sum_k(A[i][k] * B[k][j])
void matmul_cpu(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)          // 遍历 C 的每一行
        for (int j = 0; j < N; j++) {     // 遍历 C 的每一列
            float sum = 0.0f;
            for (int k = 0; k < K; k++)   // 内积：A 的第 i 行 × B 的第 j 列
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}
```

| # | 内容 | ✅ |
|:--:|------|:--:|
| 1.1 | 手写 CPU 版 MatMul，验证正确性（小矩阵：M=N=K=4） | ⬜ |
| 1.2 | 理解：每个 `C[i][j]` = A 第 i 行和 B 第 j 列的**内积** | ⬜ |

### Phase B：GPU naive（每个线程算一个元素，30min）

| # | 内容 | ✅ |
|:--:|------|:--:|
| 1.3 | 线程映射：`row = blockIdx.y*blockDim.y + threadIdx.y`，`col = blockIdx.x*blockDim.x + threadIdx.x` | ⬜ |
| 1.4 | 每个线程计算 C[row][col] = A第row行 · B第col列 的内积（K 次乘加） | ⬜ |
| 1.5 | grid 大小：`(N+15)/16 × (M+15)/16` 的 block（每 block 16×16=256 线程） | ⬜ |
| 1.6 | 编译跑通，与 CPU 版本对比验证正确性 | ⬜ |

### Phase C：临摹 → 白板（50min）

| # | 内容 | ✅ |
|:--:|------|:--:|
| 1.7 | 打开 CUDA-Learn-Notes 的 GEMM naive kernel，临摹一遍（理解每行） | ⬜ |
| 1.8 | **关掉所有代码**，白板重写 MatMul naive kernel（从 `#include` 开始） | ⬜ |
| 1.9 | 编译，和 CPU 版本对比。正确 → 过关。错了 → 自己 debug，不看答案 | ⬜ |

> 代码放 `d:\study\ai-infra-career\fundamentals\cuda\kernels\02-matmul\matmul_naive.cu`

---

## 二、B1 — Shared Memory + Bank Conflict（45min）

> 📝 笔记：[shared-memory-bank-conflict.md](../ai-infra-career/fundamentals/cuda/shared-memory-bank-conflict.md)

| # | 内容 | ✅ |
|:--:|------|:--:|
| 2.1 | §一 内存全景：Shared Memory 位置 + 为什么比 Global Memory 快 ~100×？ | ⬜ |
| 2.2 | §二 `__shared__` 声明 + `__syncthreads()` 规则 + 死锁场景 | ⬜ |
| 2.3 | §三 Bank Conflict 原理：32 banks / 4B per bank | ⬜ |
| 2.4 | §三 3 种 stride 对比：stride=1（最优）/ stride=32（最差×32）/ random | ⬜ |
| 2.5 | §三 Padding 解法：`extern __shared__ float s[N][33]` 多出一列消除 conflict | ⬜ |
| 2.6 | 思考：今晚写的 MatMul naive 有 shared memory 吗？没有的话性能瓶颈在哪？ | ⬜ |

---

## 三、🔨 白板手搓 GPT Block（45min）

> 📋 [week2-model-forward.md #手搓代码](../notes/week2-model-forward.md)
>
> ⚠️ 关掉 nanoGPT。只允许看 Week 2 笔记第 5-6 章。

**验收标准**：`(B=4, T=8, C=128)` 输入 → `(4, 8, 128)` 输出，shape 不变。

| # | 步骤 | ✅ |
|:--:|------|:--:|
| 3.1 | `__init__`：`ln_1` + `attn` + `ln_2` + `mlp` | ⬜ |
| 3.2 | MLP 实现：`c_fc(C→4C) → GELU → c_proj(4C→C)` | ⬜ |
| 3.3 | forward：`x = x + self.attn(self.ln_1(x))`（Pre-Norm） | ⬜ |
| 3.4 | forward：`x = x + self.mlp(self.ln_2(x))` | ⬜ |
| 3.5 | 跑测试：`torch.randn(4,8,128)` 输入，输出 shape = `(4,8,128)`，无报错 | ⬜ |

> 代码放 `d:\study\llm.c-learning\experiments\handwrite-gpt\block.py`

---

## 四、A — 算法代码实现（45min）

| # | 题 | 语言 | 验收 | ✅ |
|:--:|------|:--:|------|:--:|
| 4.1 | LC35 搜索插入位置 | C++ + Python | 二分查找，AC | ⬜ |
| 4.2 | LC34 排序数组查找边界 | C++ + Python | 两次二分，AC | ⬜ |
| 4.3 | LC26 删除重复项 | C++ + Python | 双指针，AC | ⬜ |

> 代码放 `algorithm/cpp/01-programmercarl/01-array/` 和 `algorithm/python/01-programmercarl/01-array/`

---

## 五、B1 — Reduction 复习 + Softmax 编译（20min）

| # | 内容 | ✅ |
|:--:|------|:--:|
| 5.1 | Reduction 7 版优化链快速回顾（笔记已有） | ⬜ |
| 5.2 | Softmax v1-v4 编译验证：`nvcc → run → 确认 v1-v4 与 CPU 基线一致` | ⬜ |
| 5.3 | MatMul naive 有无 shared memory → 理解为什么需要 tiled MatMul | ⬜ |

---

## 六、复盘（5min）

| 问 | 答 |
|----|----|
| MatMul 白板写出来了吗？卡在哪个阶段？ | |
| Shared Memory + Bank Conflict 三句话讲清楚？ | |
| GPT Block 写出来了吗？ | |
| 算法 3 道全 AC 了吗？ | |
| 明天最重要的 1 件事 | |

---

## 📋 检查清单

| # | 线 | 内容 | ⏱ | ✅ |
|:--:|:--:|------|:--:|:--:|
| 1 | B1 | MatMul naive kernel（先临摹，再白板） | 90min | ⬜ |
| 2 | B1 | Shared Memory + Bank Conflict | 45min | ⬜ |
| 3 | 🔨 C | 白板手搓 GPT Block | 45min | ⬜ |
| 4 | A | LC35 + LC34 + LC26 双版本 | 45min | ⬜ |
| 5 | B1 | Reduction 回顾 + Softmax 编译 | 20min | ⬜ |
| 6 | 🎯 | 复盘 | 5min | ⬜ |
| — | 🏃 | 跑步 15min（白天） | 15min | ⬜ |

---

## 💡 问题记录区

| # | 线 | 问题 |
|:--:|:--:|------|
| 1 | B1 | |
| 2 | 🔨 | |
| 3 | A | |
