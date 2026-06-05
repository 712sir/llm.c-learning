# Week 2：逐文件精读 nanoGPT 源码 —— 面经级详解

> 状态：🟡 进行中 | 目标：吃透 GPT-2 124M 每一行代码，面试能画图能讲原理
>
> 📖 源码：[karpathy/nanoGPT/model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py)
> 🔗 对照：vLLM 推理引擎 / llm.c CUDA 实现的同一套架构

---

## 先建立全局概念

### GPT-2 124M 的参数配置

```python
GPTConfig:
    block_size = 1024        # 最大上下文长度（不是 token 数，是位置数）
    vocab_size = 50257       # GPT-2 tokenizer 的词汇表大小
    n_layer = 12             # 12 层 Transformer Block
    n_head = 12              # 12 个注意力头
    n_embd = 768             # 隐藏维度（embedding dimension）
    # 推算：
    # 每个 head 的维度：hs = n_embd / n_head = 768/12 = 64
    # 参数量 ≈ 12 * n_layer * n_embd^2 ≈ 12 * 12 * 768^2 ≈ 85M
    # 加上 embedding + lm_head ≈ 85M + 39M ≈ 124M
```

### 模型整体结构（5 层嵌套）

```
GPT
├── transformer.wte   ← Token Embedding (vocab_size → n_embd)
├── transformer.wpe   ← Position Embedding (block_size → n_embd)
├── transformer.h     ← [Block × n_layer]  ← 12 层
│   └── Block
│       ├── ln_1      ← Pre-Attention LayerNorm
│       ├── attn      ← CausalSelfAttention
│       ├── ln_2      ← Pre-MLP LayerNorm
│       └── mlp       ← MLP (n_embd → 4*n_embd → n_embd)
├── transformer.ln_f  ← Final LayerNorm（输出前最后归一化）
└── lm_head           ← 输出投影 (n_embd → vocab_size)
```

---

# Day 1：model.py 上半部分 — LayerNorm + CausalSelfAttention

---

## 一、LayerNorm — 被低估的核心组件

### 源码全文（带行号解读）

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        # ① weight (gamma): 可学习的缩放参数，shape = (ndim,)
        #    初始化为全 1.0，即不做缩放
        self.weight = nn.Parameter(torch.ones(ndim))
        # ② bias (beta): 可学习的平移参数，shape = (ndim,)
        #    初始化为全 0.0，即不做平移
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        # ③ 沿最后一维（C/n_embd 维度）计算均值
        #    input.shape = (B, T, C) → mean.shape = (B, T, 1)
        #    keepdim=True: 保留维度，让广播机制生效
        #    -1: 表示最后一维
        mean = input.mean(dim=-1, keepdim=True)

        # ④ 沿最后一维计算方差（无偏估计用 unbiased=False）
        #    深度学习场景一直用 biased=False（有偏估计）因为样本量够大
        #    var.shape = (B, T, 1)
        var = input.var(dim=-1, keepdim=True, unbiased=False)

        # ⑤ 归一化：减均值，除标准差
        #    eps=1e-5: 防止除零。面试问"为什么是 1e-5 不是 1e-8"→fp16 精度够用
        #    xhat.shape = (B, T, C)
        xhat = (input - mean) / torch.sqrt(var + 1e-5)

        # ⑥ 仿射变换：缩放 + 平移（恢复模型的表达能力）
        #    如果没有这一步，归一化后会丢失信息
        #    weight/bias 都是 (C,) → 广播到 (B, T, C)
        output = self.weight * xhat + self.bias if self.bias is not None \
                 else self.weight * xhat
        return output
```

### 逐行 Shape 追踪（面试必须烂熟于胸）

> 假设输入 `x.shape = (B=2, T=3, C=4)`，即 batch 2，序列长 3，隐藏维 4

| 行 | 操作 | 输入 Shape | 输出 Shape | 解释 |
|:--:|------|:--:|:--:|------|
| ③ | `input.mean(-1, keepdim=True)` | (2, 3, 4) | (2, 3, **1**) | 每行的 4 个数求平均 |
| ④ | `input.var(-1, keepdim=True)` | (2, 3, 4) | (2, 3, **1**) | 每行的 4 个数求方差 |
| ⑤ | `(input - mean) / sqrt(var + eps)` | (2,3,4) | (2, 3, 4) | 广播：mean/var 从 (2,3,1) 扩到 (2,3,4) |
| ⑥ | `weight * xhat + bias` | (2,3,4) | (2, 3, 4) | 广播：weight/bias 从 (4,) 扩到 (2,3,4) |

### 🔥 面试必问

**Q1: 为什么沿最后一维归一化，而不是沿序列维度？**

> LayerNorm 对每个 token 独立归一化。沿 `C` 维度归一化意味着：
> - token_i 的归一化只依赖于 token_i 自己的 embedding，不依赖其他 token
> - 这使得推理时可以逐个 token 处理（KV Cache 场景）
> - 如果沿 `T` 维度归一化，每来一个新 token 都要重新算 → 没法做 KV Cache

**Q2: BatchNorm vs LayerNorm，GPT 为什么用 LayerNorm？**

| | BatchNorm | LayerNorm |
|------|:--:|:--:|
| 归一化维度 | 沿 Batch 维度 | 沿 Feature 维度 |
| 依赖 batch 大小 | 是（小 batch 不稳定） | 否 |
| 训练/推理一致性 | 不一致（推理用 running mean） | 一致（都直接算） |
| 序列长度敏感 | 需要 padding 到固定长度 | 不在意 |
| NLP 为什么选它 | 每个 token 独立，batch 内不等长麻烦 | 一个 token 算一次，干净 |

**Q3: RMSNorm vs LayerNorm，推理引擎为什么用 RMSNorm？**

> RMSNorm 去掉了"减均值"步骤，只保留"除均方根"：
> `RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight`
>
> - 省了一次减均值计算（~10-15% 的归一化开销）
> - Llama / Mistral 系列全用 RMSNorm
> - vLLM 的 CUDA kernel 里有专门的 RMSNorm 实现

**Q4: 为什么 weight/bias 初始化为 1.0/0.0？**

> 这是"恒等初始化"——训练初期 LayerNorm 近乎不生效，模型先学习 attention 和 MLP 的核心变换，LayerNorm 的缩放/平移在训练过程中慢慢调。

### ✏️ 手撕练习

> 关掉参考，写出 LayerNorm forward 的完整实现（含 shape 注释）。

---

## 二、CausalSelfAttention — 面试最高频考点 ⭐⭐⭐

### 2.1 `__init__` 逐行解读

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # n_embd=768, n_head=12 → 每个 head 的维度 hs=64

        # ① c_attn: 将输入 x(C) 映射为 Q+K+V 拼接 → 输出 3*C
        #    为什么是 3×？因为 Q、K、V 各占 C 维，合在一起 3C
        #    面试加分：这叫"fused QKV projection"——一次矩阵乘法算完三个投影
        #    比分开三个 Linear 少了两次 kernel launch 开销
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # ② c_proj: 将多头的输出拼接后映射回 C 维
        #    attention 的输出也是 C 维，再做一次线性变换（output projection）
        #    为什么需要？attention 把各头的信息混在一起了，c_proj 负责整合
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # ③ 注册 causal mask（注册为 buffer 而非 parameter）
        #    buffer: 不参与梯度计算，但会随模型保存/加载
        #    parameter: 参与梯度计算
        #    因果掩码是固定的（下三角矩阵），不需要学习 → 用 buffer
        #    面试：nn.Parameter vs register_buffer 的区别？
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        # ④ bias 的 shape: (1, 1, block_size, block_size)
        #    四维：batch_head 各 1, HW 各 block_size=1024
        #    两个 1 维度是为了广播：匹配 (B, nh, T, T) 的 attention scores

        self.n_head = config.n_head      # 12
        self.n_embd = config.n_embd      # 768
        self.dropout = config.dropout     # 训练时 0.1，推理时 0.0

        # ⑤ attention dropout（正则化用）
        #    注意与 residual dropout、embedding dropout 的区别
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
```

### 🔥 `__init__` 面试必问

**Q: 为什么 Q/K/V 合在一起用一个 Linear，不分开三个？**

> 性能优化——Fused QKV Projection：
> - 分开：`q=Wq@x; k=Wk@x; v=Wv@x` → 三次矩阵乘法 + 三次 kernel launch
> - 合并：`qkv=Wqkv@x` → 一次矩阵乘法 + 一次 kernel launch
> - 省了 kernel launch overhead（每次 ~5-10μs），对短序列推理尤其重要
> - vLLM/TensorRT-LLM 全都用 fused QKV

**Q: `register_buffer` vs `nn.Parameter`？**

> | | nn.Parameter | register_buffer |
> |------|:--:|:--:|
> | 参与梯度 | ✅ | ❌ |
> | 随 model.state_dict() 保存 | ✅ | ✅ |
> | 随 model.to(device) 移动 | ✅ | ✅ |
> | 适用场景 | 权重、bias | 固定常量（mask、位置编码） |

### 2.2 `forward` — 逐行 Shape 追踪（重中之重！）

```python
def forward(self, x):
    # 输入 x.shape = (B, T, C)，例如 (4, 512, 768)
    B, T, C = x.size()

    # ===== Step 1: Fused QKV Projection =====
    # c_attn(x): (B,T,C) → (B,T,3C)
    qkv = self.c_attn(x)                    # (B, T, 3*C)
    # 拆成 Q、K、V 三份，每份 C 维
    q, k, v = qkv.split(self.n_embd, dim=2) # 三个(B, T, C)

    # ===== Step 2: 重塑为多头格式 =====
    # 从 (B, T, C) 变成 (B, nh, T, hs)
    # 例如：nh=12, hs=64, C = nh*hs = 768
    # transpose: 把 head 维提到 batch 后面，方便并行计算
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    # q: (B, T, 768) → view → (B, T, 12, 64) → transpose → (B, 12, T, 64)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    # k: (B, 12, T, 64)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    # v: (B, 12, T, 64)

    # ===== Step 3: 计算 Attention Scores =====
    # Q @ K^T: (B, 12, T, 64) @ (B, 12, 64, T) → (B, 12, T, T)
    # k.transpose(-2, -1): 将 K 的最后两维转置 (B, 12, 64, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    # att.shape = (B, 12, T, T)
    # 每个 (T,T) 是一个方阵：att[t_i][t_j] = token_i 对 token_j 的注意力分数

    # ===== Step 4: Causal Mask =====
    # bias[:,:,:T,:T] → (1, 1, T, T): 下三角全 0，上三角全 -inf
    # masked_fill(bias==0, -inf): 把上三角（未来位置）的分数设为 -inf
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    # softmax(-inf) → 0，所以未来位置的注意力权重变成 0
    # 意味着 token_i 只能看到 token_0...token_i，看不到 token_{i+1} 及以后

    # ===== Step 5: Softmax + Dropout =====
    att = F.softmax(att, dim=-1)            # 沿最后一维（被关注方）归一化
    att = self.attn_dropout(att)            # 随机丢弃一些注意力连接
    # att.shape 仍为 (B, 12, T, T)
    # att[b][h][i][j] = token_i 关注 token_j 的概率（∑_j = 1）

    # ===== Step 6: 加权求和 =====
    # att @ V: (B, 12, T, T) @ (B, 12, T, 64) → (B, 12, T, 64)
    y = att @ v
    # y[b][h][i][:] = sum_j(att[b][h][i][j] * v[b][h][j][:])
    # 即 token_i 的新表示 = 所有之前 token 的 V 表示的加权和

    # ===== Step 7: 合并多头 + 输出投影 =====
    # (B, 12, T, 64) → transpose → (B, T, 12, 64) → view → (B, T, 768)
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    # transpose(1,2): 把 head 维和 T 维换回来
    # contiguous(): 让内存连续（transpose 只是改了 stride，没改实际排布）
    #   → contiguous 会触发一次内存拷贝，重新排列数据
    #   → 面试：为什么 transpose 后要 contiguous？
    #     答：view 要求 tensor 在内存中连续排列，transpose 后 stride 变了，
    #         不 contiguous 的话 view 会报错

    # 输出投影：把多头信息再整合一次
    y = self.c_proj(y)                      # (B, T, 768) → (B, T, 768)
    y = self.resid_dropout(y)               # residual pathway dropout

    return y
```

### Shape 变化全表（面试画图用）

```
Step  |  Operation              | Input Shape    | Output Shape
══════╪══════════════════════════╪════════════════╪═════════════════
  1   | c_attn(x)               | (B, T, C)      | (B, T, 3C)
  2   | split                    | (B, T, 3C)     | 3×(B, T, C)
  3   | view + transpose (Q/K/V)| (B, T, C)      | (B, nh, T, hs)
  4   | Q @ K^T                 | (B,nh,T,hs)    | (B, nh, T, T)
  5   | * (1/sqrt(hs))          | (B, nh, T, T)  | (B, nh, T, T)
  6   | + causal mask           | (B, nh, T, T)  | (B, nh, T, T)
  7   | softmax(dim=-1)         | (B, nh, T, T)  | (B, nh, T, T)
  8   | att @ V                 | (B,nh,T,T)     | (B, nh, T, hs)
  9   | transpose + view        | (B,nh,T,hs)    | (B, T, C)
 10   | c_proj                  | (B, T, C)      | (B, T, C)
```

### 🔥 Attention 面试高频七连问

**Q1: 为什么除以 `sqrt(hs)`？**

> **缩放点积注意力 (Scaled Dot-Product Attention)** 的核心：
>
> 如果 Q 和 K 的每个元素都是均值 0、方差 1 的随机变量，则 `Q @ K^T` 中每个元素是 hs 个独立随机变量的内积 → 方差为 hs → 标准差为 sqrt(hs)。
>
> 不除 sqrt(hs)：
> - scores 方差很大 → softmax 输出会非常"尖锐"（接近 one-hot）
> - 梯度接近 0（softmax 在饱和区梯度消失）
> - 训练不稳定
>
> 除了 sqrt(hs)：
> - scores 方差 ≈ 1 → softmax 输出平滑 → 梯度正常流动
>
> 这就是原始论文叫"Scaled Dot-Product Attention"的原因——多了一个缩放因子。

**Q2: Causal Mask 怎么实现"只能看到过去"？**

```
tril 矩阵 (T=5):
    0  1  2  3  4    ← token_j (被关注方)
0  [1  0  0  0  0]   token_0 只能看自己
1  [1  1  0  0  0]   token_1 能看 0,1
2  [1  1  1  0  0]   token_2 能看 0,1,2
3  [1  1  1  1  0]   token_3 能看 0,1,2,3
4  [1  1  1  1  1]   token_4 能看全部
↑ token_i (关注方)

masked_fill(tril==0, -inf):
位置 [0][1] = -inf → softmax(-inf) = 0 → token_0 看不见 token_1
```

**Q3: `transpose` 后为什么要 `contiguous()`？**

> PyTorch 的 tensor 有 stride（步长）概念。`view` 要求数据在内存中连续排列：
> ```
> transpose 前: data=[a1,a2,a3,b1,b2,b3], stride=(3,1)
> transpose 后: data=[a1,a2,a3,b1,b2,b3], stride=(1,3)  ← 物理没变！
> ```
> `view` 在 stride 不为 1 时无法推断新 shape → 报错。
> `.contiguous()` 按新的 stride 重新排列数据，变成真正连续的内存布局。

**Q4: Multi-Head 为什么有效？**

> 类比：CNN 的多个卷积核提取不同特征。
> - Head 0：关注句法关系（主语-谓语）
> - Head 1：关注近距离依赖（相邻词）
> - Head 2：关注远距离依赖（指代消解）
> - ...
> 每个头在不同的低维子空间（hs=64）独立做 attention，然后拼接。模型自动学会分工。

**Q5: `c_proj` 为什么要做一次输出投影？**

> Multi-head 的输出是各头独立计算的，`c_proj` 做的事情：
> 1. 将各头的信息混合（跨头交互）
> 2. 将维度从 C 映射到 C（看起来没有维度变化，但权重矩阵 W_proj 提供了信息重组能力）
> 3. 与残差连接配合：`x = x + attn(ln(x))`，attn 的输出经过 c_proj 后加到主路径

**Q6: Attention 的计算复杂度？**

> | 步骤 | 复杂度 |
> |------|------|
> | QKV projection | O(B·T·C²) |
> | Q @ K^T | O(B·nh·T²·hs) = O(B·T²·C) |
> | att @ V | O(B·nh·T²·hs) = O(B·T²·C) |
> | Output projection | O(B·T·C²) |
>
> 瓶颈在 **Q@K^T 和 att@V**，都是 O(T²)。这也是 FlashAttention / PagedAttention 的优化目标。

**Q7: Multi-head 换成 Multi-Query / Grouped-Query 能省什么？**

> | 变体 | Key/Value 头数 | 显存 | 速度 | 代表模型 |
> |------|:--:|:--:|:--:|------|
> | Multi-Head (MHA) | = Q 头数 | 基准 | 基准 | GPT-2, BERT |
> | Multi-Query (MQA) | 1 | KV Cache = 1/nh | 更快 | PaLM |
> | Grouped-Query (GQA) | 1<g<nh | KV Cache 按比例缩小 | 折中 | Llama 2 70B, Llama 3 |
>
> 推理引擎（vLLM）之所以快，部分原因就是 GQA 让 KV Cache 小了很多。

---

## 三、Attention 数据流全图（手撕级）

```
                         ╔════════════════════════╗
                         ║    CausalSelfAttention  ║
                         ╚════════════════════════╝
输入: x (B=4, T=512, C=768)  ← 上一层 Block 的输出（或第一层的 embedding）
    │
    ▼
┌─────────────────────────────────────────────────┐
│  c_attn: Linear(768, 2304)                      │
│  W_qkv: (768, 2304)  一次算出 Q+K+V             │
│  output: (4, 512, 2304)                         │
└──────────────┬──────────────────────────────────┘
               │  split(768, dim=2)
        ┌──────┼──────┐
        ▼      ▼      ▼
    Q (4,512,768)  K (4,512,768)  V (4,512,768)
        │      │      │
        │  view + transpose
        ▼      ▼      ▼
    Q (4,12,512,64)  K (4,12,512,64)  V (4,12,512,64)
        │      │
        │   K^T ──→ (4,12,64,512)
        │      │
        └── Q @ K^T ──→ scores (4,12,512,512)
                │
                ├── / sqrt(64) = / 8
                ├── + causal mask (下三角=0, 上三角=-inf)
                ├── softmax(dim=-1)
                ├── dropout(0.1)
                │
                ▼
         att_weights (4,12,512,512)  ← 每行和=1
                │
                └── @ V (4,12,512,64) ─→ y (4,12,512,64)
                                            │
                                   transpose + view
                                            │
                                            ▼
                                    y (4,512,768)
                                            │
                                    c_proj: Linear(768,768)
                                            │
                                    dropout(0.1)
                                            │
                                            ▼
                              output (4,512,768) → 加到残差路径
```

---

## 四、自测题（闭卷）

1. [ ] LayerNorm 的 `eps=1e-5`，写成 `1e-8` 行不行？为什么？
2. [ ] 画出 Q、K、V 的 shape 变化：`(B,T,C) → (B,nh,T,hs)` 中间经历了哪些操作？
3. [ ] `Q @ K^T` 的 shape 是 `(B, nh, T, T)`，`att[0][0][3][5]` 的含义是什么？
4. [ ] 不用 `contiguous()`，transpose 后直接 view 会发生什么？
5. [ ] 如果 `n_head=1`（单头），计算复杂度从 O(T²·C) 变了吗？
6. [ ] causal mask 用 `-inf` 而不是 `-1e9`，有区别吗？（softmax 里 -1e9 的结果是什么？）
7. [ ] 训练时 attention dropout=0.1，推理时呢？代码里怎么控制的？
8. [ ] 能口述 "Scaled Dot-Product Attention" 每个词的含义吗？
9. [ ] `c_attn` 的权重矩阵 W 的 shape 是 `(768, 2304)`，W 里哪 768 列对应 Q？哪 768 列对应 K？哪 768 列对应 V？
10. [ ] flash attention 和标准 attention 的计算结果一样吗？为什么 vLLM 用 flash attention？
11. [ ] 如果 T=2048（超长序列），Q@K^T 的显存峰值是多少（fp16）？
12. [ ] `model.eval()` 和 `model.train()` 对 CausalSelfAttention 的哪一行有影响？

---

# Day 2：model.py 下半部分 — MLP + Block + GPT + 全流程

---

## 四、MLP — Transformer 的"思考"单元

### 源码全文

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ① c_fc: C → 4*C  膨胀 4 倍（GPT-2 是 4×，Llama 是 8/3.5×）
        #    为什么叫 fc？fully-connected 的缩写
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # ② gelu: 激活函数，比 ReLU 更平滑
        #    见下方详解
        self.gelu = nn.GELU()
        # ③ c_proj: 4*C → C  压缩回来
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: (B, T, C) → c_fc → (B, T, 4C) → gelu → (B, T, 4C) → c_proj → (B, T, C)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

### Shape 流水线

```
x (B, T, 768)
    │  c_fc: Linear(768, 3072)
    │  W_fc: (768, 3072), 3072 = 768×4
    ▼
x (B, T, 3072)    ← "膨胀"：让模型有更多维度做非线性变换
    │  GELU
    │  对每个元素独立做激活（element-wise）
    ▼
x (B, T, 3072)
    │  c_proj: Linear(3072, 768)
    │  W_proj: (3072, 768)
    ▼
x (B, T, 768)     ← "压缩"回原始维度，准备加残差
```

### 🔥 GELU vs ReLU — 面试高频

```
ReLU:  f(x) = max(0, x)
GELU: f(x) = x * Φ(x)  ≈ 0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))

          ReLU                    GELU
           |                      |
          /|                     /|
         / |                    / |
        /  |                   /  |
    ───┘   |────        ─────┘    └─────
           0                      0
   折线，x=0 处不可导       光滑，处处可导
   梯度要么 0 要么 1         梯度连续变化
   稀疏激活                  稠密激活（每个神经元都有一点输出）
```

| | ReLU | GELU |
|------|------|------|
| 公式 | max(0, x) | x·Φ(x) |
| x<0 行为 | 全 0（死神经元风险） | 小负值（不死） |
| 可导性 | x=0 不可导 | 处处可导 |
| 谁在用 | CNN, ResNet | GPT, BERT, ViT |
| 为什么 transformer 用它 | 更平滑 → 梯度流更好 | ReLU 在 NLP 效果差 |

**Q: GELU 的近似公式为什么长这样？**

> `GELU(x) = x * Φ(x)`，其中 Φ 是标准正态分布的 CDF。
> 这个公式的直觉：GELU = x * P(X <= x), X ~ N(0,1)
> - 当 x 很大时，Φ(x) ≈ 1 → GELU(x) ≈ x（像个线性激活）
> - 当 x 很小时，Φ(x) ≈ 0 → GELU(x) ≈ 0（抑制噪声）
> - 中间区域有一个平滑的非线性过渡
>
> 精确 Φ 需要误差函数，太慢。近似公式用 tanh 拟合：
> `GELU(x) ≈ 0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))`

**Q: 为什么膨胀比是 4× 而不是 3× 或 8×？**

> 经验值。4× 在计算开销和模型容量的 trade-off 中表现最好。
> - 太小（2×）：非线性变换的空间不够，模型表达能力弱
> - 太大（8×）：参数量暴涨，收益递减
> - Llama 用了 8/3 ≈ 2.67×（SwiGLU 激活），属于后 GPT 时代的改进

---

## 五、Block — 最小可重复单元

### 源码全文

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-Norm 结构：先归一化，再计算，再加残差
        x = x + self.attn(self.ln_1(x))   # Attention 子层
        x = x + self.mlp(self.ln_2(x))    # MLP 子层
        return x
```

### 🔥 Pre-Norm vs Post-Norm — 区分度最高的面试题

```
Post-Norm (原版 Transformer/ViT):       Pre-Norm (GPT-2/LLaMA/vLLM):
                                        x
x                                        │
│                                        ├── ln_1 ──→ attn ──→ (+)
├── attn ──→ (+) ──→ ln ──→              │                      │
│              ↑                         └──────────────────────┘
└──────────────┘                         │
                │                        ├── ln_2 ──→ mlp ──→ (+)
├── mlp ──→ (+) ──→ ln ──→               │                      │
│              ↑                         └──────────────────────┘
└──────────────┘                         │
                │                        ▼
```

| | Post-Norm | Pre-Norm |
|------|:--:|:--:|
| 归一化位置 | 残差之后 | 残差之前 |
| 梯度流 | LayerNorm 在残差路径末端，可能阻碍梯度 | 残差路径直达，梯度如履平地 |
| 训练稳定性 | 需要 warmup（Adam 前几步特别小心） | 不需要 warmup，直接训 |
| 谁在用 | BERT, ViT, 原版 Transformer | GPT-2/3, Llama, 所有现代 LLM |
| NanoGPT 用的 | — | Pre-Norm ✅ |

**为什么 Pre-Norm 更稳定？**

> 残差网络的核心是信息有两条路：
> - **残差路径**：`x` 直接传递（identity mapping）
> - **变换路径**：`attn(ln(x))` 或 `mlp(ln(x))`
>
> Post-Norm 在残差加完之后才归一化 → 归一化会压缩残差信号的幅度 → 浅层梯度变小。
> Pre-Norm 在变换路径之前归一化 → 残差路径是纯 identity → 梯度可以直接从第 12 层传到第 1 层。

### 一个 Block 的 FLOPs 估算（面试加分项）

```
假设 T=1024, C=768, 4C=3072, hs=64, nh=12

Attention 子层:
  QKV proj:   B×T×C×(3C) × 2 = 2×3×1024×768^2 ≈ 3.6B FLOPs
  Q@K^T:      B×nh×T^2×hs × 2  = 2×12×1024^2×64 ≈ 1.6B FLOPs
  att@V:      B×nh×T^2×hs × 2  = 同上 ≈ 1.6B FLOPs
  Out proj:   B×T×C^2 × 2       = 2×1024×768^2 ≈ 1.2B FLOPs
  小计: ~8B FLOPs

MLP 子层:
  c_fc:       B×T×C×(4C) × 2   = 2×1024×768×3072 ≈ 4.8B FLOPs
  c_proj:     B×T×(4C)×C × 2   = 同上 ≈ 4.8B FLOPs
  小计: ~9.6B FLOPs

一层 Block: ~17.6B FLOPs
12 层: ~211B FLOPs / token
```

---

## 六、GPT 类 — 把所有积木拼起来

### `__init__` 逐行解读

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ① wte: Word Token Embedding
        #    参数矩阵 (vocab_size=50257, n_embd=768)
        #    50257×768 ≈ 38.6M 参数 ← 这是参数量的大头！
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),

            # ② wpe: Word Position Embedding
            #    参数矩阵 (block_size=1024, n_embd=768)
            #    1024×768 ≈ 0.79M 参数
            wpe = nn.Embedding(config.block_size, config.n_embd),

            # ③ drop: embedding 级别的 dropout
            #    和 attention 里的 dropout 是两码事
            drop = nn.Dropout(config.dropout),

            # ④ h: 12 个 Block 的序列
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),

            # ⑤ ln_f: 最终 LayerNorm（只在输出前做一次）
            #    为什么叫 ln_f？final LayerNorm
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # ⑥ lm_head: Language Model Head
        #    参数矩阵 (n_embd=768, vocab_size=50257)
        #    把 hidden state 映射回词汇表 → 预测下一个 token 的 logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # ⑦ Weight Tying: wte 和 lm_head 共享权重
        #    wte.weight 和 lm_head.weight 指向同一块内存！
        #    所以实际 lm_head 不占额外参数
        self.transformer.wte.weight = self.lm_head.weight

        # ⑧ 参数初始化（单独函数，见下方）
        self.apply(self._init_weights)

        # ⑨ 对 c_proj 的权重做特殊缩放
        #    "GPT-2 风格的初始化"：残差路径的投影权重缩小 sqrt(2*n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

### 🔥 Weight Tying — 面试必问

**Q: wte 和 lm_head 为什么要共享权重？**

```
wte:   token_id ─→ embedding vector (50257 → 768)
lm_head: embedding vector ─→ logits over vocab (768 → 50257)

这两个矩阵恰好是转置关系：
  wte:    (50257, 768)  查表：token_id → 第 token_id 行的 768 维向量
  lm_head: (768, 50257)  投影：768 维向量 → 50257 维 logits

共享 = lm_head.weight = wte.weight^T
```

> 三个好处：
> 1. **省参数**：省掉 lm_head 的 38.6M 参数（约 1/3 的总参数量！）
> 2. **语义一致性**：token 的 embedding 和解码时的投影用同一个空间
>    → 语义相似的 token 在输出层也会有相似的 logits
> 3. **正则化效果**：权重复用减少了过拟合风险

**Q: 为什么 weight tying 在 GPT 里有效，但 BERT 不常用？**

> BERT 的 MLM 任务需要预测被 mask 的 token，输入和输出语义不一致（输入有 [MASK]，输出是原词），weight tying 反而限制了模型。GPT 是自回归的，输入和输出是同一个词汇表，tying 自然合理。

### `_init_weights` — 有什么讲究

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        # 正态分布初始化，std=0.02
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

**Q: std=0.02 怎么来的？**

> GPT-2 论文的经验值。不是 Xavier/Kaiming 初始化那样的理论推导。
> 特殊之处：`c_proj` 的权重在初始化后额外除以 `sqrt(2*n_layer)`：
> ```python
> std = 0.02 / sqrt(2 * 12) ≈ 0.004
> ```
> 这确保初始时残差路径的贡献很小，模型从"近乎恒等映射"开始训练。训练初期主要由 identity path 传递信息，attention/MLP 的贡献慢慢增大。这让深层网络不需要 learning rate warmup。

---

## 七、forward — 完整前向传播

```python
def forward(self, idx, targets=None):
    # idx: (B, T), 值为 token id
    # targets: (B, T) or None
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size, \
        f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

    pos = torch.arange(0, t, dtype=torch.long, device=device)  # (T,)

    # ===== ① Token Embedding + Position Embedding =====
    tok_emb = self.transformer.wte(idx)  # (B, T) → (B, T, C)
    pos_emb = self.transformer.wpe(pos)  # (T,) → (T, C)
    x = self.transformer.drop(tok_emb + pos_emb)
    # tok_emb + pos_emb: (B, T, C) + (T, C) = (B, T, C)
    # 广播：pos_emb 从 (T, C) 广播到 (B, T, C)

    # ===== ② 逐层过 Transformer Block =====
    for block in self.transformer.h:
        x = block(x)  # 每个 Block: x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))

    # ===== ③ 最终 LayerNorm =====
    x = self.transformer.ln_f(x)  # (B, T, C) → (B, T, C)

    # ===== ④ 输出 logits =====
    if targets is not None:
        # 训练模式：算 loss
        logits = self.lm_head(x)  # (B, T, vocab_size)
        # 交叉熵需要 (N, vocab_size) vs (N,) 的输入
        # N = B*T：把所有 token 展开成一维
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
            targets.view(-1),                   # (B*T,)
            ignore_index=-1                     # -1 位置不算 loss（padding 占位）
        )
        return logits, loss
    else:
        # 推理模式：只返回 logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits, None
```

### Shape 全链路追踪

```
Step | Operation               | Input Shape         | Output Shape
═════╪══════════════════════════╪═════════════════════╪════════════════════
 ①a  | wte(idx)                | (B, T) [int ids]    | (B, T, C=768)
 ①b  | wpe(pos)                | (T,)  [int ids]     | (T, C=768) → 广播到 (B,T,C)
 ①c  | tok_emb + pos_emb       | (B,T,C)+(T,C)       | (B, T, C)
 ①d  | dropout                 | (B, T, C)           | (B, T, C)
 ②   | block × 12              | (B, T, C)           | (B, T, C)  每层不变 shape
 ③   | ln_f                    | (B, T, C)           | (B, T, C)
 ④a  | lm_head (训练)          | (B, T, C)           | (B, T, 50257)
 ④b  | view(-1, 50257)         | (B, T, 50257)       | (B*T, 50257)
 ④c  | cross_entropy           | (B*T,50257)         | scalar (loss)
```

### 🔥 forward 面试拷打

**Q1: 为什么 token embedding 和 position embedding 可以直接相加？**

> 两个 (B,T,C) 的矩阵逐元素相加。直觉：
> - Token embedding 编码了"这个词是什么"
> - Position embedding 编码了"这个词在哪个位置"
> - 相加 = 告诉模型"某个词在某个位置"——模型自己学会把这两个信号分开
>
> 为什么不拼接？拼接会翻倍维度（(B,T,2C)），后续所有层的参数都要翻倍。
> 实践证明加法足够好，省参数。

**Q2: Position Embedding 是学习的还是固定的？**

> nanoGPT 用的是**可学习的绝对位置编码**（Learned Absolute Position Embedding）。
> - 为每个位置 0..1023 随机初始化一个 (768,) 的向量
> - 训练中自动调整
> - 缺点：不能泛化到 block_size 之外的序列长度
>
> 现代替代方案：
> - **RoPE**（Rotary Position Embedding）：Llama/Mistral/千问 都用它
>   位置信息通过旋转 Q 和 K 来编码，不占参数，可外推更长序列
> - **ALiBi**：用 attention bias 编码位置（简单但效果略差于 RoPE）

**Q3: `ignore_index=-1` 是干什么的？**

> 如果 targets 里有 `-1` 的位置，这个位置的 loss 不计入。用于 padding——当 batch 内有不等长的序列时，短序列的在 targets 填 -1。

**Q4: `cross_entropy` 的输入为什么要 view 成 `(B*T, vocab_size)`？**

> PyTorch 的 `F.cross_entropy` 签名为：`(N, C) vs (N,)`
> - N = 样本数，C = 类别数
> - 一个 token 的预测 = 一个分类任务（从 vocab_size 个词里选一个）
> - 把 B×T 个 token 全部展开成 N=B×T 个独立样本
> - 每个样本是一个 C=vocab_size=50257 的分类问题

---

## 八、generate — 自回归生成

```python
@torch.no_grad()  # 推理不开梯度，省显存
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    # idx: (B, T) 初始 prompt
    for _ in range(max_new_tokens):
        # ① 裁剪上下文：只取最后 block_size 个 token
        idx_cond = idx if idx.size(1) <= self.config.block_size \
                   else idx[:, -self.config.block_size:]

        # ② 前向传播，只取最后一个位置的 logits
        logits, _ = self(idx_cond)
        # logits: (B, T, vocab_size)
        logits = logits[:, -1, :]  # (B, vocab_size)  ← 只要最后一个位置！

        # ③ Temperature 缩放
        logits = logits / temperature
        # T=0 → 最确定（贪婪），T=1 → 正常，T>1 → 更随机

        # ④ Top-K 过滤（可选）
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # ⑤ Softmax + 采样
        probs = F.softmax(logits, dim=-1)      # (B, vocab_size)
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # ⑥ 拼接到序列末尾
        idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
    return idx
```

### 🔥 generate 面试拷打

**Q1: 为什么只取 `logits[:, -1, :]`？以前的位置为什么不重新算？**

> GPT 是因果的（causal）——每个 token 只能看到自己的过去。
> 在 KV Cache 场景下（见下方），之前的 token 的 hidden states 已经被缓存了，不需要重新算。
> 即使没有 KV Cache，前向传播时前面的 logits 也算出来了，但我们只需要最后一个来预测下一个 token。
>
> nanoGPT 没有实现 KV Cache——它每次都把整个序列重算一遍（包括之前的 token）。
> 这是 nanoGPT 慢的原因之一，也是 vLLM 的核心优化点。

**Q2: Temperature 的作用机制？**

```
probs = softmax(logits / T)

T → 0:  logits/T 差异被放大 → softmax 接近 one-hot → 贪婪解码
        例：[2.0, 1.0, 0.5] / 0.1 = [20, 10, 5] → softmax → [1.0, 0, 0]
T = 1:  正常分布
T → ∞:  logits/T 差异被抹平 → softmax 接近均匀分布 → 随机输出
        例：[2.0, 1.0, 0.5] / 100 = [0.02, 0.01, 0.005] → softmax → 几乎均匀
```

> 面试场景：
> - "想输出更有创意的文本" → 提高 T (1.2~1.5)
> - "想输出确定性的结果" → 降低 T (0.2~0.5)
> - 典型生产环境 T ≈ 0.7~1.0

**Q3: Top-K sampling 怎么工作的？**

> 只保留概率最高的 K 个 token，其余置为 -inf。然后重新 softmax 并采样。
> - K=50：保留前 50 个，过滤掉低概率的噪声词
> - 比纯 temperature 更可控，减少生成垃圾文本的概率
> - 现代做法更多用 Top-P (nucleus sampling)：保留累计概率达到 P 的最小 token 集合

---

## 九、KV Cache — nanoGPT 没有但你必须懂

### 没有 KV Cache 的问题

```
nanoGPT 每生成一个新 token，整个序列重新过一遍模型：

Step 1: "The"           → model("The")             → 1 次前向
Step 2: "The cat"       → model("The cat")          → 2 次前向（"The" 重复算了）
Step 3: "The cat sat"   → model("The cat sat")       → 3 次前向（"The","cat" 重复算）

生成 N 个 token 的总计算量: O(N²)  ← 平方级！
```

### KV Cache 的解决方案

```
每层的 Attention 中，K 和 V 只依赖于过去的 token（因果性质）。

Step 1: "The"           → 算 Q/K/V, 存 K_1, V_1          → 1 次前向
Step 2: "cat"           → 算 Q（新 token）, 查 K_1, V_1   → 1 次前向（只算新 token！）
Step 3: "sat"           → 算 Q（新 token）, 查 K_1,2 V_1,2 → 1 次前向

生成 N 个 token 的总计算量: O(N)  ← 线性！
```

**KV Cache 的大小计算（面试可能让你算）：**

```
假设 Llama-2 7B, FP16:
  n_layer = 32
  n_head = 32
  hs = 128
  T = 4096 (上下文长度)
  batch = 1

单层 KV Cache = 2 × (32 × 4096 × 128) × 2 bytes = 2 × 16.78M × 2B ≈ 67 MB
总 KV Cache  = 32 层 × 67 MB ≈ 2.1 GB  ← 这就是推理显存的主要消耗！
```

> 这就是 PagedAttention (vLLM) 和 GQA (Llama-2) 的优化动机——让 KV Cache 可管理。

---

## 十、configure_optimizers — 权重衰减的讲究

```python
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    # ① 收集所有需要 grad 的参数
    param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

    # ② 分成两组：做 weight decay 和不做的
    decay_params = [p for n, p in param_dict.items()
                    if p.dim() >= 2]     # 矩阵参数（Linear.weight, Embedding.weight）
    nodecay_params = [p for n, p in param_dict.items()
                      if p.dim() < 2]    # 向量参数（bias, LayerNorm.weight/bias）

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    # ③ 根据设备选 fused AdamW（更快）或标准 AdamW
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate,
                                  betas=betas, fused=use_fused)
    return optimizer
```

### 🔥 为什么 bias 和 LayerNorm 不做 weight decay？

> Weight decay = L2 正则化 → 把权重往 0 推。
> - Linear.weight / Embedding.weight：应该防止过拟合 → 做 weight decay ✅
> - bias：只是一个偏移量，推到 0 没有意义 → 不做 weight decay
> - LayerNorm.weight：初始为 1，代表"不做缩放"。推到 0 意味着"把输出全归零"→ 显然不对
>
> 面试口诀：**权重矩阵做 decay，偏置和归一化不做。**

---

## 十一、参数精确计算（对标面试）

| 组件 | 计算 | 参数量 |
|------|------|:--:|
| wte | 50257 × 768 | 38,597,376 |
| wpe | 1024 × 768 | 786,432 |
| 每层 Attn c_attn | 768 × 2304 + 2304(bias) | 1,771,776 |
| 每层 Attn c_proj | 768 × 768 + 768 | 590,592 |
| 每层 MLP c_fc | 768 × 3072 + 3072 | 2,362,368 |
| 每层 MLP c_proj | 3072 × 768 + 768 | 2,360,064 |
| 每层 LayerNorm ×2 | 2 × 768(bias) + 2 × 768(weight) | 3,072 |
| 12 层合计 | 12 × 7,085,568 | 85,026,816 |
| ln_f | 768 + 768 | 1,536 |
| **总参数** | | **~124.4M** |

> 面试现场心算技巧：`C=768, C²≈590K, 12×C²≈7M, ×12层≈84M, +embedding≈124M`

---

## 十二、闭卷总自测

1. [ ] GELU 的两种近似公式（tanh 版和 sigmoid 版），写出一种
2. [ ] Pre-Norm vs Post-Norm：画图 + 说差别 + 为什么现代 LLM 全用 Pre-Norm
3. [ ] "Weight Tying" 是什么意思？为什么省 38M 参数？
4. [ ] c_proj 为什么初始化特别小（除以 sqrt(2*n_layer)）？
5. [ ] forward 中 `logits.view(-1, vocab_size)` 的 -1 是多少？（B×T = ?）
6. [ ] generate 里为什么只取 `logits[:, -1, :]`？
7. [ ] KV Cache 让推理从 O(?) 变 O(?)？如果没有 KV Cache，生成 100 个 token 要做多少次前向？
8. [ ] 哪些参数不做 weight decay？为什么？
9. [ ] "Scaled Dot-Product Attention" 的 Scaled 指除以 sqrt(d_k)，为什么要除？
10. [ ] CausalSelfAttention 的 `bias` 是 buffer 还是 parameter？为什么？
11. [ ] 写出 GPT-2 124M 的总参数计算公式（不用精确数字，推导过程即可）
12. [ ] 如果 `block_size=2048, n_head=16, n_embd=1024`，每层 attention 的 Q@K^T 矩阵多大（MB）？
