# Week 2：精读 nanoGPT 源码 —— GPT-2 从零到面试

> 状态：🟡 进行中 | 目标：看懂每一行代码，面试能画图讲原理
>
> 源码：[karpathy/nanoGPT/model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py)
> 对照：vLLM / TensorRT-LLM / llm.c 用的都是同一套 Transformer 架构

---

## 第 0 章：开工前必须懂的三个概念

### 0.1 Token 和 Tokenizer

GPT 不认识文字，只认识数字。**Tokenizer** 就是"翻译官"：把文字变成数字序列（tokenize），需要输出时再变回来（detokenize）。

```
"Hello world!"   →   tokenizer   →   [15496, 995, 0]
                                        ↑      ↑   ↑
                                      Hello  world  !

反过来：
[15496, 995, 0]  →   tokenizer   →   "Hello world!"
```

GPT-2 的 tokenizer 有 50257 个"词"。一个 token 不一定是完整的英文单词——BPE 算法会把低频词切得更细，比如 `"unfortunately"` 可能变成 `["un", "fortunate", "ly"]` 三个 token。

### 0.2 Token Embedding vs Position Embedding

一个句子经过 tokenizer 后变成一个整数序列。但整数不能直接做矩阵运算——"42" 不一定比 "100" 小——所以需要把每个整数映射成一个固定长度的浮点向量。这个映射表就是 **Embedding**。

GPT-2 有两套 Embedding：

| | Token Embedding (`wte`) | Position Embedding (`wpe`) |
|------|------|------|
| 回答什么 | "这个 token **是**哪个词？" | "这个 token **在第几个**位置？" |
| 查表方式 | 输入 token ID → 查出对应行 | 输入位置编号 0,1,2... → 查出对应行 |
| 参数规模 | (50257, 768) ≈ 38.6M 个参数 | (1024, 768) ≈ 0.79M 个参数 |
| 为什么需要 | 把离散的整数变成连续的向量 | Attention 本身不知道谁先谁后——"狗咬人"和"人咬狗"对 Attention 来说只是三个一样的 token、排列不同。Position Embedding 告诉模型顺序信息。 |

两个 embedding 直接逐元素相加：`x = tok_emb + pos_emb`。为什么不拼接？（拼接会让维度翻倍，后面所有层都要为此多付参数。）

### 0.3 参数是怎么凑到 124M 的

```
wte:      50257 × 768 = 38.6M
lm_head:  768 × 50257 = 38.6M （但跟 wte 共享，不计入总数）
wpe:      1024 × 768  =  0.79M
12层Block: 每层 ~7.09M × 12  = 85.0M
ln_f:     768 + 768   =  0.0015M
──────────────────────────────
总计                    ≈ 124.4M
```

"共享"是 GPT-2 的一个设计技巧——`lm_head.weight` 和 `wte.weight` 指向同一块内存。省了 38.6M 参数，相当于白送了一个大矩阵。

### 0.4 把 GPT-2 想象成一条流水线

不要把它当程序读——没有 if-else 分支，没有业务逻辑。它是一条**数据变换流水线**：

```
输入：Token IDs (B=4, T=512)  ← 4句话，每句512个token编号
  │
  ▼  wte + wpe（查表，得到向量）
Embedding (4, 512, 768)        ← 每个token变成768维的向量
  │
  ▼  Block x12（思考）
  │  每个Block做两件事：
  │    ① Attention：每个token去问前面的token"谁跟我有关？"→ 融合上下文
  │    ② MLP：每个token独立消化信息 → 非线性变换
  │    ③ 残差：不管算了什么，原始信息都保留一份备份
  │  12层下来，信息越来越"理解"了
  │
  ▼  ln_f（最后归一化）
  │
  ▼  lm_head（译回词汇表）
输出：Logits (4, 512, 50257)   ← 每个位置输出"下一个token是哪个词"的概率
```

阅读源码时只需要追踪两件事：(1) 每一步数据进去是什么 shape，出来是什么 shape；(2) 这一步在整个流水线中扮演什么角色。

---

## 第 1 章：PyTorch 语法速成

> 只讲 nanoGPT 源码里实际出现的。每一条都标注了在代码哪里用到。

### 1.1 类和继承

```python
class CausalSelfAttention(nn.Module):  # 继承 nn.Module
    def __init__(self, config):
        super().__init__()             # 必须第一行！否则参数追踪失效
        self.c_attn = nn.Linear(768, 2304)  # 注册为子模块，自动加入参数列表
```

`super().__init__()`：先让父类 `nn.Module` 初始化内部的参数注册表。没这行，后面 `model.to(device)`、`model.state_dict()`、梯度计算全部静默失败。**这一行没有商量余地，必须写。**

### 1.2 `nn.Linear` —— 全连接层

```python
layer = nn.Linear(768, 2304)
# 数学上：y = x @ W^T + b
# x (B,T,768) → W (2304,768) → y (B,T,2304)
# 2304 = 3×768（Q,K,V各768维，拼在一起）
```

### 1.3 `nn.Embedding` —— 查表

```python
emb = nn.Embedding(50257, 768)
# 内部：一个 (50257, 768) 的可学习矩阵
# 输入 token ID = 42 → 返回第 42 行的 768 维向量
```

### 1.4 `model.state_dict()` 和 `model.to(device)`

```python
model.state_dict()  # 把所有参数打包成字典 → 保存/加载用
torch.save(model.state_dict(), 'checkpoint.pt')
model.load_state_dict(torch.load('checkpoint.pt'))

model.to('cuda')    # 把模型搬到 GPU。所有参数和 buffer 一起搬
model.to('cpu')     # 搬回 CPU
```

### 1.5 矩阵运算三件套

```python
C = A @ B           # 矩阵乘法。(M,K) @ (K,N) → (M,N)
# 源码：att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))

x.transpose(1, 2)   # 交换维度。shape (B,nh,T,hs) → (B,T,nh,hs)
# 注意：transpose 不真正移动数据，只是改了"读取顺序"（stride）

x.view(B, T, C)     # 改变形状。要求数据在内存中连续
# 如果之前 transpose 过，必须先 .contiguous() 重新排列

x.split(768, dim=2) # 沿 dim=2 切分 → q,k,v = qkv.split(768, dim=2)
```

### 1.6 `register_buffer` —— 跟着模型走但不需要学的东西

```python
self.register_buffer("bias", tril_tensor)
# 特性：不参与梯度（不是参数）| 随 state_dict 保存/加载 | 随 model.to(device) 迁移
# 适用：causal mask——模型一辈子都不变的常量
```

---

## 第 2 章：Q、K、V 和 Attention 是什么

### 2.1 用"查资料"来理解 Attention

把 Self-Attention 理解成一次**精准检索**。你是一个 token（比如句子中的 "sat"），现在想去前面的 token 那里查点有用的上下文信息：

```
Q (Query):  我发出的"检索需求"——"我需要什么样的上下文？"
K (Key):    每个前面 token 的"索引标签"——"我是什么类型的词？"
V (Value):  每个前面 token 的"实际资料"——"我能告诉你什么？"

Attention = 用 Q 跟每个 K 做匹配打分 → softmax 变成百分比 → 按百分比加权取 V 的内容
```

**具体演示**：句子 "The cat sat on the mat"，处理 token_2（"sat"）时：

```
① sat 发出 Q₂："我是动词，需要主语"
② 用 Q₂ 去匹配前面的 K：
   · K₀(The)  → 不相关，得分 0.1
   · K₁(cat)  → 很相关（猫可以做主语），得分 0.7
   · K₂(sat)  → 自己跟自己，得分 0.2
③ softmax → 概率: The=12%, cat=65%, sat=23%
④ 加权融合：12%×V_The + 65%×V_cat + 23%×V_sat
   → "sat" 的新表示里融入了 "cat" 的语义信息
```

### 2.2 数学公式（翻译成人话）

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

Q·K^T  → 算出一个 (T,T) 的"互相相关程度"矩阵
/√d_k  → 除以一个数，防止内积结果太大 → softmax 死机
softmax → 把每一行变成百分比（行求和 = 1）
·V     → 按百分比把 V 的内容加权混合
```

### 2.3 为什么叫 Causal（因果）Self-Attention？

普通的 Self-Attention 每个 token 能看所有 token（包括还没生成的未来 token）。但 GPT 是**一个一个生成**的——生成第 5 个词时，第 6 个词还不存在，不能偷看。

Causal 就是用一个**下三角掩码**：把"未来位置"的得分强行设成 -inf，这样 softmax(-inf) = 0，未来 token 的 V 就参与不进来。

### 2.4 `Q @ K^T` 的形状变化

```
Q: (B, nh, T, hs)   ← B组数据，nh个头，T个token，每个头hs维
K: (B, nh, T, hs)

① K 转置：K.transpose(-2, -1) → (B, nh, hs, T)  （最后两维交换）
② Q @ K^T               → (B, nh,  T,  T)

结果矩阵 [b][h][i][j] = token_i 对 token_j 的注意力原始分数
```

---

## 第 3 章：LayerNorm —— 第一道工序

### 3.1 源码逐行走读

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        # ① weight (gamma): 可学习的缩放参数，shape = (ndim,)
        #    初始化为全 1.0 —— 即不做缩放，训练中慢慢调
        self.weight = nn.Parameter(torch.ones(ndim))
        # ② bias (beta): 可学习的平移参数，shape = (ndim,)
        #    初始化为全 0.0 —— 即不做平移
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        # ③ 沿最后一维（C / n_embd 维度）计算均值
        #    input.shape = (B, T, C) → mean.shape = (B, T, 1)
        #    keepdim=True: 保留维度，让广播机制能自动对齐
        #    -1: 表示最后一维
        mean = input.mean(dim=-1, keepdim=True)

        # ④ 沿最后一维计算方差
        #    unbiased=False: 用有偏估计（深度学习场景样本量够大，不需无偏修正）
        #    var.shape = (B, T, 1)
        var = input.var(dim=-1, keepdim=True, unbiased=False)

        # ⑤ 归一化：(x - 均值) / 标准差
        #    eps=1e-5: 防止除零 — FP16 下 1e-8 会出 subnormal 精度问题
        #    xhat.shape = (B, T, C)
        xhat = (input - mean) / torch.sqrt(var + 1e-5)

        # ⑥ 仿射变换：y = γ·x̂ + β（恢复模型的表达能力）
        #    归一化会强制分布→会丢失信息，γ 和 β 让模型"恢复"需要的信息
        #    weight/bias 都是 (C,) → 广播到 (B, T, C)
        output = self.weight * xhat + self.bias if self.bias is not None \
                 else self.weight * xhat
        return output
```

每一步在做什么：

| 操作 | 输入 (B,T,C) | 作用 |
|------|:--:|------|
| 减均值 | 使每个 token 的 768 个特征以 0 为中心 | 消除不同 token 的向量范数差异 |
| 除标准差 | 使每个 token 的 768 个特征的方差统一为 1 | 防止某些维度过大、在后续 Attention 内积中取得不正当优势 |
| 乘 γ 加 β | 可学习的缩放和平移 | 归一化会丢失信息，γ 和 β 负责"恢复"模型觉得有用的信息 |

### 3.2 为什么沿 C 维度而不是 T 维度归一化？

```
沿 C 维度（LayerNorm 的做法）：每个 token 独立归一化，token_i 的统计量跟其他 token 无关
沿 T 维度（为什么不这样做）：跨 token 归一化 → 每来一个新 token，所有旧 token 都要重新算 → KV Cache 无法工作
```

LayerNorm 让推理时可以逐个 token 处理——这是 vLLM 能做 KV Cache 的前提之一。

### 3.3 RMSNorm：去掉减均值

LayerNorm 做了两步归一化：减均值 + 除标准差。实验发现减均值对最终效果贡献很小，但消耗约 10-15% 的计算。LLaMA、Mistral 等现代 LLM 全用 RMSNorm：

```
LayerNorm:  y = (x - mean) / std * γ + β   （两步）
RMSNorm:    y =  x         / rms * γ        （一步，只除均方根）
```

### 3.4 面试要点

- 为什么 eps 用 1e-5 不用 1e-8？FP16 半精度的最小有效数字约 6e-5，1e-8 在 FP16 下精度不够
- 为什么 γ 初始化为 1、β 初始化为 0？恒等初始化——刚开始 LayerNorm 不做任何事，让模型先学会 Attention/MLP 的核心计算
- BatchNorm vs LayerNorm？BN 沿 batch 维度归一化，小 batch 不稳定；LN 沿 feature 维度，NLP 天然适配

---

## 第 4 章：CausalSelfAttention —— 核心中的核心

### 4.1 `__init__` —— 逐行注释

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # n_embd=768, n_head=12 → 每个 head 的维度 hs=64

        # ① c_attn: Fused QKV Projection
        #    输入维度 C=768，输出维度 3C=2304
        #    为什么是 3×？因为 Q、K、V 各占 C 维，合在一起 3C
        #    一次矩阵乘法同时算出三个投影 → 比分开三次省了两次 kernel launch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # ② c_proj: Output Projection
        #    attention 的输出也是 C 维，再做一次线性变换
        #    为什么需要？attention 把各头的信息混在一起了，c_proj 负责跨头整合
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # ③ causal mask: 下三角矩阵，注册为 buffer 而不是 parameter
        #    buffer: 不参与梯度计算，但会随模型保存/加载和自动设备迁移
        #    parameter: 参与梯度计算
        #    因果掩码是固定的——下三角全1，上三角全0——不需要学习 → 用 buffer
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        # ④ bias 的 shape: (1, 1, block_size=1024, block_size=1024)
        #    四维：batch维=1, head维=1, 高=1024, 宽=1024
        #    前两个维度为 1 是为了广播：匹配 (B, nh, T, T) 的 attention scores

        self.n_head = config.n_head      # 12
        self.n_embd = config.n_embd      # 768
        self.dropout = config.dropout    # 训练时 0.1，推理时 0.0（model.eval() 自动切换）

        # ⑤ 两种 dropout——施加位置和目标不同（区别见附录A）
        self.attn_dropout = nn.Dropout(config.dropout)   # Attention 权重上的 dropout
        self.resid_dropout = nn.Dropout(config.dropout)   # 残差路径上的 dropout
```

关键设计决策：

**2304 = 3 × 768**：Q、K、V 各 768 维，拼在一次矩阵乘法里算完——叫 **Fused QKV Projection**。分开三次 Linear 要多花两次 kernel launch 的时间（每次几微秒，但短序列场景这个开销很可观）。vLLM、TensorRT-LLM 全用 Fused QKV。

**causal mask 为什么用 buffer 而不是 parameter？** causal mask 是固定的下三角矩阵，不需要学。buffer 的特性：不参与梯度、随 state_dict 保存、随 model.to(device) 自动迁移——刚好满足所有需求。

### 4.2 `forward` —— 逐行 Shape 追踪（整个模型最核心的代码）

```python
def forward(self, x):
    # 输入 x.shape = (B, T, C)，例如 (B=4, T=512, C=768)
    B, T, C = x.size()

    # ===== Step 1: Fused QKV Projection =====
    # c_attn(x): (B, T, C=768) → (B, T, 3C=2304)
    # 一次矩阵乘法同时算出 Q、K、V（分开三次要多花 kernel launch 开销）
    qkv = self.c_attn(x)                    # (B, T, 3*C)
    # 沿 dim=2 切成三份，每份 768 维
    q, k, v = qkv.split(self.n_embd, dim=2) # 三个 (B, T, C)

    # ===== Step 2: 重塑为多头格式 =====
    # 从 (B, T, C=768) 变成 (B, nh=12, T, hs=64)
    # view( B, T, 12, 64) → 每个头拿到 64 维的子空间
    # transpose(1, 2)       → 把 n_head 维提到 batch 后面，方便并行计算
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    # q: (B, T, 768) → view → (B, T, 12, 64) → transpose → (B, 12, T, 64)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    # k: (B, 12, T, 64)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    # v: (B, 12, T, 64)

    # ===== Step 3: 计算 Attention Scores =====
    # Q @ K^T: (B, 12, T, 64) @ (B, 12, 64, T) → (B, 12, T, T)
    # k.transpose(-2, -1): 将 K 的最后两维交换 (B,12,T,64)→(B,12,64,T)
    # 结果矩阵 [b][h][i][j] = token_i 对 token_j 的"原始注意力分数"
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    # att.shape = (B, 12, T, T)

    # ===== Step 4: Causal Mask —— 屏蔽未来 =====
    # bias[:,:,:T,:T] → (1, 1, T, T): 下三角全 0，上三角全 -inf
    # masked_fill(bias==0, -inf): 把上三角（未来位置）的分数设为 -inf
    # softmax(-inf) → 0，所以未来位置的注意力权重变成 0
    # → token_i 只能看到 token_0...token_i，看不到 token_{i+1} 及以后
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

    # ===== Step 5: Softmax + Dropout =====
    att = F.softmax(att, dim=-1)            # 沿最后一维（被关注方）归一化，每行和=1
    att = self.attn_dropout(att)            # 训练时随机切断一些注意力连接，防止过拟合
    # att.shape 仍为 (B, 12, T, T)
    # att[b][h][i][j] = token_i 关注 token_j 的概率

    # ===== Step 6: 加权求和 =====
    # att @ V: (B, 12, T, T) @ (B, 12, T, 64) → (B, 12, T, 64)
    y = att @ v
    # y[b][h][i][:] = Σ_j att[b][h][i][j] × v[b][h][j][:]  ← 加权混合
    # token_i 的新表示 = 所有之前 token 的 V 的加权和

    # ===== Step 7: 合并多头 + 输出投影 =====
    # transpose(1,2): 把 head 维和 T 维换回来 → (B, T, 12, 64)
    # contiguous(): 必须！transpose 只改了 stride，view 要求连续内存
    # view(B, T, C): 合并 12 个头 → (B, T, 768)
    y = y.transpose(1, 2).contiguous().view(B, T, C)

    # c_proj: 把各头信息混在一起（跨头交互）
    y = self.c_proj(y)                      # (B, T, 768) → (B, T, 768)
    y = self.resid_dropout(y)               # 残差路径上的 dropout
    return y
```

### 4.3 10 步 Shape 变化速查表

```
Step  |  操作                     |  输入 Shape     |  输出 Shape
══════╪════════════════════════════╪═════════════════╪═════════════════
  1   |  c_attn (Fused QKV)       |  (B, T,  768)   |  (B, T, 2304)
  2   |  split → Q, K, V          |  (B, T, 2304)   |  3×(B, T, 768)
  3   |  view + transpose         |  (B, T,  768)   |  (B,12, T, 64)
  4   |  Q @ K^T                  |  (B,12,T, 64)   |  (B,12, T, T)
  5   |  / sqrt(64)               |  (B,12, T, T)   |  (B,12, T, T)
  6   |  + causal mask            |  (B,12, T, T)   |  (B,12, T, T)
  7   |  softmax + dropout        |  (B,12, T, T)   |  (B,12, T, T)
  8   |  att @ V                  |  (B,12, T, T)   |  (B,12, T, 64)
  9   |  transpose + view         |  (B,12,T, 64)   |  (B, T, 768)
 10   |  c_proj (output proj)     |  (B, T,  768)   |  (B, T, 768)
```

### 4.4 Attention 数据流图

```
x (B=4, T=512, C=768)
  │
  ▼  c_attn: Linear(768, 2304)
qkv (4, 512, 2304)
  │  split(768, dim=2)
  ├──── Q (4,512,768) ──→ view+transpose ──→ Q (4,12,512,64)
  ├──── K (4,512,768) ──→ view+transpose ──→ K (4,12,512,64)
  └──── V (4,512,768) ──→ view+transpose ──→ V (4,12,512,64)
                              │
                         Q @ K^T ──→ scores (4,12,512,512)
                              │
                         /8 + mask + softmax + dropout
                              │
                         att (4,12,512,512)
                              │
                         att @ V ──→ y (4,12,512,64)
                              │
                         transpose + view ──→ y (4,512,768)
                              │
                         c_proj ──→ output (4,512,768)
```

### 4.5 Attention 面试七连问

**Q1：为什么除以 `√hs`？**
> Q@K^T 中每个元素是 hs 个独立随机变量的内积，方差为 hs。不除 → softmax 过尖锐 → 梯度消失。除了 → 方差归一化 → 训练稳定。

**Q2：causal mask 怎么工作的？**
> 下三角全 1（允许关注），上三角全 0（设为 -inf）。softmax(-inf)=0，未来 token 参与不进来。

**Q3：transpose 后为什么必须 contiguous()？**
> transpose 改的是 stride，不是实际内存排布。view 要求连续内存，不 contiguous 会报错。

**Q4：Multi-Head 为什么有效？**
> 每个头在不同的低维空间做 Attention，自动学会不同的关注模式（句法/语义/指代）。

**Q5：c_proj 做什么？**
> 各头独立计算的结果需要跨头混合。c_proj 的权重矩阵提供了这种信息重组能力。

**Q6：Attention 的复杂度？**
> O(B·T²·C)，瓶颈在 Q@K^T 和 att@V。T=2048 时，一个 attention 矩阵约 34MB（fp16）。这就是 FlashAttention 要优化的问题。

**Q7：MHA vs MQA vs GQA？**
> MHA：每个头独立 K、V（GPT-2）。MQA：所有头共享 K、V。GQA：分组共享（LLaMA 用）。GQA 让 KV Cache 缩小 2-8 倍。

---

## 第 5 章：MLP —— 每个 token 独立的思考单元

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ① c_fc: C → 4*C  膨胀 4 倍（GPT-2 是 4×，Llama 用 SwiGLU 是 ~2.67×）
        #    为什么叫 fc？fully-connected 的缩写
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # ② gelu: 激活函数——比 ReLU 更平滑、处处可导（见下方对比图）
        self.gelu = nn.GELU()
        # ③ c_proj: 4*C → C  压缩回来，维度恢复
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: (B, T, C=768)
        #   → c_fc: Linear(768, 3072) → (B, T, 3072)
        #   → gelu: 逐元素激活 → (B, T, 3072)
        #   → c_proj: Linear(3072, 768) → (B, T, 768)
        #   → dropout
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

MLP 的作用：膨胀到 4 倍维度 → 让模型有足够空间做非线性变换 → GELU 激活 → 压回原维度。每个 token 独立处理，不看其他 token。

**GELU vs ReLU**：ReLU 在 x<0 时直接归零（死神经元风险），GELU 处处光滑、x<0 也有小梯度。Transformer 这种深层网络对梯度流敏感，用 GELU 比 ReLU 效果好。

---

## 第 6 章：Block —— 组装最小可重复单元

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)   # Attention 前的归一化
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)   # MLP 前的归一化
        self.mlp  = MLP(config)

    def forward(self, x):
        # Pre-Norm 结构：先归一化 → 再计算 → 最后残差加回
        # 两条路：
        #   残差路径 (identity): x 原封不动往前传 → 梯度直接反向传播
        #   变换路径: ln → attn/mlp → 输出加到残差上
        x = x + self.attn(self.ln_1(x))   # Attention 子层
        x = x + self.mlp(self.ln_2(x))    # MLP 子层
        return x
```

这是 **Pre-Norm** 结构：先归一化，再计算，最后残差加上去。对比原版 Transformer 的 Post-Norm（先计算出结果、加残差、再归一化），Pre-Norm 让梯度能通过残差路径直接反向传播到最浅层，训练不需要 lrmup。

现代 LLM（GPT-2/3、LLaMA、vLLM）全用 Pre-Norm。

---

## 第 7 章：GPT 类 —— 把积木拼起来

### 7.1 组件清单 —— 逐行注释

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ① wte: Word Token Embedding
        #    参数矩阵 (vocab_size=50257, n_embd=768)
        #    50257×768 ≈ 38.6M 参数 ← 这是参数量的大头！
        #    输入 token ID → 返回第 token_id 行的 768 维向量
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),

            # ② wpe: Word Position Embedding
            #    参数矩阵 (block_size=1024, n_embd=768)
            #    1024×768 ≈ 0.79M 参数
            #    输入位置编号 0..1023 → 返回对应行的 768 维向量
            wpe = nn.Embedding(config.block_size, config.n_embd),

            # ③ drop: Embedding 级别的 dropout
            #    施加在 tok_emb + pos_emb 的逐元素结果上
            #    与 attention dropout、residual dropout 是三个不同的 dropout
            drop = nn.Dropout(config.dropout),

            # ④ h: 12 个 Block 的序列
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),

            # ⑤ ln_f: Final LayerNorm（只在输出前做一次）
            #    为什么叫 ln_f？final LayerNorm
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # ⑥ lm_head: Language Model Head
        #    参数矩阵 (n_embd=768, vocab_size=50257)
        #    把 768 维 hidden state 映射回 50257 维 → 每个位置输出"下一个 token 的概率"
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # ⑦ Weight Tying: wte 和 lm_head 共享权重！
        #    wte.weight 和 lm_head.weight 指向同一块内存
        #    → 实际 lm_head 不占额外参数，省了 38.6M
        self.transformer.wte.weight = self.lm_head.weight

        # ⑧ 对所有 Linear 和 Embedding 做标准初始化（正态分布，std=0.02）
        self.apply(self._init_weights)

        # ⑨ 对 c_proj 的权重做特殊缩放 —— "GPT-2 风格的初始化"
        #    残差路径的投影权重额外除以 sqrt(2*n_layer)
        #    让初始时残差贡献很小，模型从"近乎恒等映射"起步 → 不需要 lr warmup
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

### 7.2 Weight Tying —— 省了 38M 参数

`wte` 做的事：token ID → 768 维向量 `lm_head` 做的事：768 维向量 → token ID 的 logits

这两个矩阵互为转置。GPT-2 让它们共享权重——`wte.weight` 和 `lm_head.weight` 指向同一块内存。省了 38.6M 参数（约 1/3 的总大小），对训练还有正则化效果。现代 LLM 基本都这么做。

### 7.3 forward —— 完整前向传播

```python
def forward(self, idx, targets=None):
    # idx: (B, T), 值为 token ID（整数）
    # targets: (B, T) or None（训练时有、推理时没有）
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size  # 序列不能超过最大上下文长度

    pos = torch.arange(0, t, dtype=torch.long, device=device)  # (T,) 位置序号 0,1,2...

    # ===== ① Token Embedding + Position Embedding =====
    tok_emb = self.transformer.wte(idx)  # (B, T) → (B, T, C=768)  查词表
    pos_emb = self.transformer.wpe(pos)  # (T,) → (T, C=768)  查位置表
    x = self.transformer.drop(tok_emb + pos_emb)
    # tok_emb + pos_emb: (B, T, C) + (T, C) → (B, T, C)
    # 广播：pos_emb 从 (T, C) 自动扩展到 (B, T, C)，每 batch 共享
    # ← 为什么相加而不是拼接？拼接会让维度翻倍 → 后续所有层参数翻倍 → 浪费

    # ===== ② 逐层过 Transformer Block =====
    for block in self.transformer.h:
        x = block(x)  # 每层：x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))
    # (B, T, C) 进，(B, T, C) 出，形状始终不变

    # ===== ③ Final LayerNorm =====
    x = self.transformer.ln_f(x)  # (B, T, C) 最后统一归一化

    # ===== ④ 输出 logits =====
    if targets is not None:
        # 训练模式：算 logits + loss
        logits = self.lm_head(x)  # (B, T, C) → (B, T, vocab_size)
        # 交叉熵要求 (N, C) vs (N,) 的输入 → 把 B*T 个 token 展平
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # (B*T, vocab_size) 预测值
            targets.view(-1),                   # (B*T,)          真实值
            ignore_index=-1                     # -1 位置不参与 loss（padding 占位）
        )
        return logits, loss
    else:
        # 推理模式：只返回 logits
        logits = self.lm_head(x)
        return logits, None
```

### 7.4 generate —— 自回归生成

```python
@torch.no_grad()  # 关闭梯度计算——推理不需要反向传播，省显存
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    # idx: (B, T) 初始 prompt 的 token IDs
    for _ in range(max_new_tokens):
        # ① 裁剪上下文：如果超长，只保留最后 block_size=1024 个 token
        idx_cond = idx if idx.size(1) <= self.config.block_size \
                   else idx[:, -self.config.block_size:]

        # ② 前向传播，logits.shape = (B, T, vocab_size)
        logits, _ = self(idx_cond)
        # ③ 只取最后一个位置的 logits！
        #    GPT 是因果的——知道前面所有 token 后，只要最后一个位置
        #    就可以预测下一个 token。前面的 logits 不需要
        logits = logits[:, -1, :]  # (B, vocab_size)

        # ④ Temperature 缩放
        #    T→0: 差异放大→接近 one-hot→贪婪解码（最确定）
        #    T=1: 正常分布
        #    T→∞: 差异抹平→接近均匀→随机输出
        logits = logits / temperature

        # ⑤ Top-K 过滤（可选）：只保留得分最高的 K 个候选
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')  # 低分 token 置 -inf

        # ⑥ Softmax → 概率 → 按概率采样
        probs = F.softmax(logits, dim=-1)      # (B, vocab_size) 每行和=1
        idx_next = torch.multinomial(probs, num_samples=1)  # 按概率随机抽一个 token

        # ⑦ 拼接到序列末尾
        idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
    return idx
```

**为什么只取最后一个位置？** GPT 是因果的——知道前面所有 token 后，只需要最后一个位置的 logits 来预测下一个 token。在有 KV Cache 的场景下，前面的 hidden states 已被缓存，不需要重算。

**Temperature 的作用**：`probs = softmax(logits / T)`。T→0：选最确定的词（贪婪），T=1：正常采样，T→∞：纯随机。生产环境一般 T ≈ 0.7-1.0。

---

## 第 8 章：KV Cache —— nanoGPT 没有但你必须懂

### 推理加速：从 O(N²) 到 O(N)

nanoGPT 每生成一个新 token，整个序列重新过一遍模型。生成 N 个 token 需要 1+2+3+...+N 次前向 = O(N²)。

KV Cache 利用了一个事实：每层的 K 和 V 只依赖于过去 token。把已经算过的 K、V 存下来，新 token 进来时，只算新 token 的 Q，拿缓存的 K、V 做 Attention。

```
nanoGPT (无 KV Cache): 计算量 O(N²)
有 KV Cache:           计算量 O(N)
```

KV Cache 是推理显存的主要消耗——Llama-2 7B、上下文长度 4096、batch=1 时，KV Cache 约 2.1 GB。

这是 vLLM（PagedAttention）和 LLaMA（GQA）的优化动机——让 KV Cache 可以高效管理。

---

## 第 9 章：configure_optimizers —— 权重衰减的讲究

```python
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    # ① 收集所有需要梯度的参数
    param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

    # ② 分成两组：做 weight decay 的和不做的一─
    #    p.dim() >= 2: 矩阵参数（Linear.weight、Embedding.weight）→ 做 weight decay
    #    p.dim() < 2:  向量参数（bias、LayerNorm.weight/bias）→ 不做 weight decay
    decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params,   'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    # ③ 根据设备选 fused AdamW（GPU 上用 CUDA 后端，更快）
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate,
                                  betas=betas, fused=use_fused)
    return optimizer
```

### 为什么 bias 和 LayerNorm 不做 weight decay？

```
Weight decay = L2 正则化 → 把权重往 0 方向推

Linear.weight / Embedding.weight:
  权重矩阵应该防止过拟合 → 做 weight decay ✓

bias:
  只是一个偏移量，推到 0 没有意义 → 不做 weight decay

LayerNorm.weight:
  初始为 1，代表"不做任何缩放"。
  如果推到 0 → 意味着"把整层输出归零" → 模型废了 → 不做 weight decay
```

> 面试口诀：**权重矩阵做 decay，偏置和归一化不做。**

### 参数精确计算（面试验证用）

| 组件 | 计算 | 参数量 |
|------|------|:--:|
| wte | 50257 × 768 | 38,597,376 |
| wpe | 1024 × 768 | 786,432 |
| 每层 Attn c_attn | 768 × 2304 + 2304(bias) | 1,771,776 |
| 每层 Attn c_proj | 768 × 768 + 768 | 590,592 |
| 每层 MLP c_fc | 768 × 3072 + 3072 | 2,362,368 |
| 每层 MLP c_proj | 3072 × 768 + 768 | 2,360,064 |
| 每层 LayerNorm ×2 | 2 × (768+768) | 3,072 |
| 12 层合计 | 12 × 7,085,568 | 85,026,816 |
| ln_f | 768 + 768 | 1,536 |
| **总参数** | | **~124.4M** |

> 面试心算：`C=768, C²≈590K, 12×C²≈7M, ×12层≈84M, +embedding≈124M`

---

## 第 10 章：自测题

> 闭卷做完再对答案。

**LayerNorm 部分**
1. LayerNorm 的 eps=1e-5，写成 1e-8 为什么不行？
2. 减均值和除标准差各自解决什么问题？
3. RMSNorm 去掉了哪一步？凭什么敢去掉？

**CausalSelfAttention 部分**
4. 默写 Q、K、V 从 `(B,T,C)` 到 `(B,nh,T,hs)` 的 view+transpose 过程
5. `Q@K^T` 的 shape 是什么？`att[0][0][3][5]` 的含义是什么？
6. 为什么除以 √hs？不除会怎样？
7. causal mask 怎么工作的？为什么用 -inf 而不是 -1e9？
8. transpose 后为什么必须 `contiguous()` 才能 `view()`？
9. Fused QKV 比独立三个 Linear 好在哪？
10. `c_proj` 维度是 (768,768)，看起来没有变化，为什么还要这一步？
11. 训练时 attention dropout=0.1，推理时呢？谁控制的？
12. flash attention 和标准 attention 的计算结果一样吗？
13. T=2048 时，`Q@K^T` 矩阵多大（fp16，MB）？
14. MHA vs MQA vs GQA 的核心区别？为什么推理引擎喜欢 GQA？

**MLP + Block 部分**
15. MLP 为什么膨胀 4 倍？
16. GELU vs ReLU 的核心区别？
17. Pre-Norm vs Post-Norm：画图 + 说明为什么现代 LLM 全用 Pre-Norm。

**GPT + generate 部分**
18. "Weight Tying" 省了多少参数？为什么可以省？
19. forward 中 `logits.view(-1, vocab_size)` 的 -1 是多少？
20. generate 为什么只取 `logits[:, -1, :]`？
21. KV Cache 让推理从 O(?) 变成 O(?)？
22. 哪些参数不做 weight decay？为什么？
23. `c_proj` 的初始化为什么特别小（除以 sqrt(2*n_layer)）？
24. 写出 GPT-2 124M 的总参数估算过程。

---

## 附录 A：Dropout 三种类型

| 类型 | 代码位置 | 施加在 | 效果 |
|------|------|------|------|
| Embedding dropout | `self.transformer.drop(x)` | tok_emb + pos_emb | 随机把某些特征维度置零 |
| Attention dropout | `self.attn_dropout(att)` | softmax 之后的注意力权重 | 随机切断某些 token 间的注意力连接 |
| Residual dropout | `self.resid_dropout(y)` | 每个子层输出 | 随机丢弃子层部分贡献，增大残差路径权重 |

训练时生效，`model.eval()` 后全部自动关闭。

---

## 附录 B：FlashAttention 与 PagedAttention

**FlashAttention**：标准 Attention 的 (T,T) 中间矩阵要先写回 HBM（显存）再读出来。FlashAttention 把矩阵切成小块，在片上 SRAM 里算完直接出结果，HBM 读写量降 4-8 倍。vLLM、PyTorch 2.0+ 已内置。

**PagedAttention**：KV Cache 按 token 连续分配时产生碎片问题。PagedAttention 借鉴 OS 虚拟内存，按固定大小 Block 管理 KV Cache——消除碎片，支持写时复制（beam search 场景共享前缀）。vLLM 的核心技术，论文发表于 SOSP 2023。

---

## 附录 C：参数速查表

| 缩写 | 全称 | shape (GPT-2 124M) | 作用 |
|------|------|------|------|
| `wte` | Word Token Embedding | (50257, 768) | token ID → 768维向量 |
| `wpe` | Word Position Embedding | (1024, 768) | 位置编号 → 768维向量 |
| `c_attn` | Combined Attention (Fused QKV) | (768, 2304) | 输入 → QKV拼接 (3×768) |
| `c_proj` | Combined Projection (Attn输出) | (768, 768) | 各头信息跨头混合 |
| `c_fc` | Combined Fully-Connected (MLP第一层) | (768, 3072) | 膨胀到4倍 (768→3072) |
| `c_proj`(mlp) | Combined Projection (MLP第二层) | (3072, 768) | 压缩回来 (3072→768) |
| `ln_1` | LayerNorm (Attn前) | (768,) | 归一化 |
| `ln_2` | LayerNorm (MLP前) | (768,) | 归一化 |
| `ln_f` | LayerNorm Final | (768,) | 最后一层归一化 |
| `lm_head` | Language Model Head | (768, 50257) | hidden → logits (与wte共享权重) |

## 附录 D：源码函数速查

| 函数 | 文件 | 行范围 | 做什么 |
|------|------|--------|--------|
| `LayerNorm.__init__` | model.py | ~30 | 初始化 weight(γ)、bias(β) |
| `LayerNorm.forward` | model.py | ~40 | 减均值→除标准差→乘γ加β |
| `CausalSelfAttention.__init__` | model.py | ~55 | Fused QKV、c_proj、causal mask(buffer)、两种dropout |
| `CausalSelfAttention.forward` | model.py | ~70 | 10步：QKV→多头→scores→mask→softmax→加权→合并→c_proj |
| `MLP.__init__` | model.py | ~105 | c_fc(768→3072)、GELU、c_proj(3072→768) |
| `MLP.forward` | model.py | ~112 | c_fc→gelu→c_proj→dropout |
| `Block.__init__` | model.py | ~120 | ln_1、attn、ln_2、mlp (Pre-Norm) |
| `Block.forward` | model.py | ~125 | x = x + attn(ln_1(x)); x = x + mlp(ln_2(x)) |
| `GPT.__init__` | model.py | ~135 | wte、wpe、drop、h(12×Block)、ln_f、lm_head、Weight Tying、初始化 |
| `GPT.forward` | model.py | ~170 | embedding相加→12层Block→ln_f→lm_head→loss or logits |
| `GPT.generate` | model.py | ~220 | 逐token生成：裁剪→前向→取最后→temperature→topk→采样→拼接 |
| `GPT.configure_optimizers` | model.py | ~260 | 分两组：矩阵参数做weight decay，bias/LN不做 |
| `F.cross_entropy` | PyTorch | — | 交叉熵：`-log(softmax(logits)[target])` |
| `F.softmax` | PyTorch | — | `exp(x)/sum(exp(x))`，把任意向量变成概率分布 |
| `torch.multinomial` | PyTorch | — | 按概率采样，返回token索引 |
| `torch.tril` | PyTorch | — | 下三角矩阵，用于causal mask |

## 附录 E：学完有什么用

### 写在简历上

```
- 精读 GPT-2 124M 完整源码，理解从 Embedding 到 lm_head 的全链路数据流
- 能画出 CausalSelfAttention 的 10 步 Shape 变化全图
- 掌握 Pre-Norm vs Post-Norm、Fused QKV Projection、Weight Tying、KV Cache 等核心设计
```

### 面试时可以说的

> "我完整阅读过 nanoGPT 的源码实现，能讲清楚 CausalSelfAttention 从 QKV Projection 到 output projection 的 10 步形状变化全链路，包括 Fused QKV 的设计动机、transpose 后必须 contiguous() 的原因、以及 causal mask 的 -inf 填充与 softmax 的交互行为。"
>
> "基于对 Pre-Norm 残差路径的理解，我能解释为什么现代 LLM 全面采用 Pre-Norm——核心在于梯度流经过 identity path 直接反向传播，消除了对 learning rate warmup 的依赖。"

### 对后续学习的价值

- **MiniInfer 项目**：模型加载和前向推理 100% 依赖本笔记的架构理解
- **vLLM 源码**：vLLM 的 ModelRunner、BlockManager 与 GPT-2 架构一一对应
- **CUDA kernel**：理解 Attention 计算模式后，手写 GEMM/FlashAttention 才有优化目标
- **面试复习**：24 道自测题 + 各章面试要点 = 系统回顾清单

---

## 🔨 手搓代码（白板 · 关掉 nanoGPT 源码）

> ⚠️ 手搓规则：**关掉 nanoGPT/model.py，不参考任何代码。** 只允许看本笔记的概念解释和 data flow 图。每道题限时 60-90 分钟，超时才能看参考答案。

### 🥇 白板题 1：手写 CausalSelfAttention（90min）

**验收标准**：写出来的代码能跑通 `(B=2, T=8, C=64, n_head=4)` 的随机输入，输出 shape 正确。

核心 checklist：
- [ ] `__init__`：`c_attn`（Fused QKV，in=C, out=3C）、`c_proj`（out projection）、`register_buffer("bias", ...)` 下三角 causal mask，shape: `(1, 1, block_size, block_size)`
- [ ] `forward`：10 步 shape 变化全链路（见第 4 章速查表）
- [ ] `qkv.split` + `view + transpose` → 多头格式
- [ ] `Q @ K^T / sqrt(d)` + causal mask（`masked_fill` with `-inf`）+ softmax
- [ ] `att @ V` → `transpose + contiguous + view` 合并多头 → `c_proj`
- [ ] 写完后对照 nanoGPT 源码：看哪里不一样？为什么？

### 🥇 白板题 2：手写 GPT Block（60min）

**验收标准**：输入 `(B, T, C)`，输出 `(B, T, C)`，shape 不变。

核心 checklist：
- [ ] `ln_1 → attn → residual add`
- [ ] `ln_2 → mlp → residual add`
- [ ] Pre-Norm 结构：先在每层输入做 LayerNorm，再进子层，最后加残差
- [ ] MLP：`c_fc(C→4C) → GELU → c_proj(4C→C)`

### 🥇 白板题 3：手写完整 GPT 前向传播（90min）

**验收标准**：喂入 `(B=2, T=16)` 的 token IDs，跑完 forward 得到 `(B, T, vocab_size)` 的 logits。

核心 checklist：
- [ ] `wte` + `wpe` 相加（不是拼接）
- [ ] Embedding dropout
- [ ] 逐层过 blocks
- [ ] `ln_f`
- [ ] `lm_head` 输出 logits
- [ ] **Weight Tying**：`lm_head.weight = wte.weight`（共享权重，省参数）
- [ ] 用 `torch.ones(2, 16, dtype=torch.long)` 作为假输入跑一遍，确认不报错

### 🥈 临摹题 1：手写 KV Cache 推理（60min）

> 允许参考本笔记第 8 章的概念解释，但不允许看 nanoGPT 的 `generate` 方法。

- [ ] 实现一个简化版 `generate_with_kv_cache`：维护 `cache_k` / `cache_v` 两个 list
- [ ] 每步只算新 token 的 Q，拿缓存里的 K、V 做 Attention
- [ ] 验证：有 KV Cache 和无 KV Cache 生成的文本完全一致

### 代码位置

手搓产物放在 `d:\study\llm.c-learning\experiments\handwrite-gpt\`：

```
handwrite-gpt/
├── attention.py      # 白板题1：CausalSelfAttention
├── block.py          # 白板题2：GPT Block
├── gpt.py            # 白板题3：完整 GPT forward
├── kv_cache.py       # 临摹题1：KV Cache 推理
└── test_all.py       # 跑通以上所有模块的正确性验证
```

> 面试官问的不是 "你看过 nanoGPT 吗"，而是 **"你能在白板上写出 Attention 吗"**。这 4 道题做完，面试时直接默写。
