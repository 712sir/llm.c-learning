# Week 2：精读 nanoGPT 源码

> 状态：✅ 已完成 | 源码：[karpathy/nanoGPT/model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py)

---

## 1. 前置概念

### 1.1 Token 和 Tokenizer

GPT 不认识文字，只认识数字。Tokenizer 把文字变成数字序列。

```
"Hello world!" → tokenizer → [15496, 995, 0]
[15496, 995, 0] → tokenizer → "Hello world!"
```

GPT-2 词表有 50257 个 token。一个 token 不一定是完整单词——BPE 会把低频词切碎，`"unfortunately"` → `["un", "fortunate", "ly"]`。

### 1.2 Token Embedding vs Position Embedding

整数不能直接做矩阵运算，需要映射成固定长度的浮点向量 = **Embedding**。

| | Token Embedding (`wte`) | Position Embedding (`wpe`) |
|------|------|------|
| 回答什么 | "这个 token **是**哪个词？" | "这个 token **在第几个**位置？" |
| shape | (50257, 768) | (1024, 768) |
| 为什么需要 | 把离散整数变成连续向量 | Attention 不知道顺序——"狗咬人"和"人咬狗"对它只是三个一样的 token。wpe 告诉模型位置 |

两个 embedding **逐元素相加**：`x = tok_emb + pos_emb`。不拼接是因为拼接会让维度翻倍 → 所有后续层参数翻倍。

### 1.3 参数怎么凑到 124M

```
wte:      50257 × 768 = 38.6M
wpe:      1024 × 768  =  0.79M
12层Block: 每层 ~7.09M × 12  = 85.0M
ln_f:     768 + 768   =  0.0015M
──────────────────────────────
总计                    ≈ 124.4M
```

`lm_head.weight` 和 `wte.weight` 共享同一块内存，省了 38.6M。

### 1.4 数据流全景

GPT-2 是一条数据变换流水线，没有 if-else 分支：

```
输入 Token IDs (B=4, T=512)
  → wte + wpe（查表得向量）
  → Embedding (4, 512, 768)
  → Block ×12：每层做 Attention（融合上下文）+ MLP（独立消化）+ 残差
  → ln_f（归一化）
  → lm_head（译回词表）
  → Logits (4, 512, 50257)
```

**阅读源码只需要追踪两件事**：每一步数据进去是什么 shape、出来是什么 shape；这一步在流水线中扮演什么角色。

### 1.5 维度速查

| 符号 | 含义 | 默认值 |
|------|------|:--:|
| B | Batch，一次训练塞几句 | 64 |
| T | Time/Tokens，序列长度 | 256 |
| C | Channels，嵌入维度 | 384 |
| nh | number of Heads，注意力头数 | 6 |
| hs | Head Size = C/nh | 64 |
| V | Vocab，词表大小 | 50304 |

---

## 2. PyTorch 语法速成

> 只讲 nanoGPT 源码里实际出现的。

### 2.1 类和继承

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()  # 必须第一行！否则参数追踪失效
        self.c_attn = nn.Linear(768, 2304)
```

`super().__init__()` 让父类 `nn.Module` 初始化参数注册表，没这行 → `model.to(device)`、梯度计算全部静默失败。

### 2.2 `nn.Linear` — 全连接层

```python
layer = nn.Linear(768, 2304)
# 数学：y = x @ W^T + b
# x(B,T,768) → W(2304,768) → y(B,T,2304)
# 2304 = 3×768（Q,K,V各768维，拼在一起）
```

### 2.3 `nn.Embedding` — 查表

```python
emb = nn.Embedding(50257, 768)
# 内部：(50257, 768) 可学习矩阵
# token ID=42 → 返回第42行的768维向量
```

### 2.4 矩阵运算三件套

```python
C = A @ B           # (M,K) @ (K,N) → (M,N)
x.transpose(1, 2)   # 交换维度，只改 stride 不拷贝数据
x.view(B, T, C)     # 改变形状，要求连续内存
x.split(768, dim=2) # 沿 dim=2 切分
```

**stride 举例**：`shape(2,3) stride(3,1)` — 内存里 6 个数挨着排 `[a,b,c,d,e,f]`，逻辑 2×3：

```
行0: a b c    行1: d e f
```

第0维（行）跨 3 个元素，第1维（列）跨 1 个。取 `[1][2]` = 起始 1×3+2×1 = 位置5 = f。

**view 举例**：`x.view(2,3,4)` — 假设原来 24 个数扁的，view 后读成 2×3×4 三维。不改内存，只改解读方式。如果之前 transpose 过，物理顺序是旧的，stride 已改，view 就报错——需要先 `.contiguous()`。

### 2.5 `register_buffer` — 不学但跟着模型走

```python
self.register_buffer("bias", tril_tensor)
# 不参与梯度 | 随 state_dict 保存 | 随 model.to(device) 迁移
# 适用：causal mask——固定常量，不需要学
```

### 2.6 `state_dict()` 和 `model.to(device)`

```python
model.state_dict()  # 所有参数打包成字典 → 保存/加载
model.to('cuda')    # 搬到 GPU，所有参数和 buffer 一起搬
```

---

## 3. Attention：Q、K、V 是什么

### 3.1 用"查资料"来理解

把 Self-Attention 理解成一次精准检索。你是句子中的 "sat"，想去前面的 token 那里查上下文：

```
Q (Query):  你的检索需求——"我需要什么样的上下文？"
K (Key):    每个前面 token 的索引标签——"我是什么类型的词？"
V (Value):  每个前面 token 的实际资料——"我能告诉你什么？"

Attention = Q 跟每个 K 匹配打分 → softmax 变百分比 → 按百分比加权取 V
```

**演示**：句子 "The cat sat on the mat"，处理 token_2（"sat"）时：

```
① sat 发出 Q₂："我是动词，需要主语"
② Q₂ 去匹配前面的 K：
   K₀(The) → 不相关，得分 0.1
   K₁(cat) → 很相关（猫能做主语），得分 0.7
   K₂(sat) → 自己跟自己，得分 0.2
③ softmax → 概率：The=12%, cat=65%, sat=23%
④ 加权融合：12%×V_The + 65%×V_cat + 23%×V_sat
   → "sat" 的新表示里融入了 "cat" 的语义信息
```

### 3.2 数学公式

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

Q·K^T  → 算出一个 (T,T) 的"互相相关程度"矩阵
/√d_k  → 防止内积太大 → softmax 死机
softmax → 每行变成百分比（行求和 = 1）
·V     → 按百分比加权混合
```

> softmax 公式：`softmax(z_i) = e^{z_i} / Σe^{z_j}`。三件事：① 任意实数变正数；② 所有输出和为 1（概率分布）；③ 大值放大、小值压小 → "只关注大的"。

### 3.3 为什么叫 Causal Self-Attention？

GPT 是一个一个生成的——生成第 5 个词时，第 6 个词还不存在，不能偷看。Causal 用**下三角掩码**：把未来位置的得分强行设成 -inf → softmax(-inf)=0 → 未来 token 参与不进来。

### 3.4 `Q @ K^T` 的形状变化

```
Q: (B, nh, T, hs)
K: (B, nh, T, hs)
K.transpose(-2, -1) → (B, nh, hs, T)
Q @ K^T             → (B, nh,  T,  T)

结果 [b][h][i][j] = token_i 对 token_j 的注意力原始分数
```

### 3.5 FAQ

**Q: "头"是啥？**
一个"头"= 一套独立的 Q/K/V 投影。nh=6 时，384 维切成 6 个 64 维子空间，各自独立做 Attention。6 个头可能分别关注"动词-宾语""形容词-名词""位置邻近"。最后拼回来。

**Q: 加权融合那步是干啥？**
让每个 token 融入上下文。"sat" 做完 Attention 后不再是孤立的词向量，而是融入了 "cat" 的语义——不加这步，每个词只知道自己。

**Q: embedding 相加后信息还在吗？**
不是"抹掉"，是"融合"。token_emb("cat")=[0.1,0.2,-0.3] + pos_emb(3)=[0.0,-0.1,0.2] → [0.1,0.1,-0.1]。分不开了，但 Transformer 不在乎分开，只在乎能从结果里提取有用模式。实验证明模型自动学会区分。

---

## 4. LayerNorm

### 4.1 源码

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))   # γ，初始全1
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # β，初始全0

    def forward(self, input):
        # input: (B, T, C)
        mean = input.mean(dim=-1, keepdim=True)       # (B, T, 1)
        var = input.var(dim=-1, keepdim=True, unbiased=False)  # (B, T, 1)
        xhat = (input - mean) / torch.sqrt(var + 1e-5) # 归一化：eps=1e-5 防除零
        return self.weight * xhat + self.bias if self.bias is not None \
               else self.weight * xhat                  # γ·x̂+β 恢复表达力
```

| 步骤 | 作用 |
|------|------|
| 减均值 | 中心化，消除不同 token 向量范数差异 |
| 除标准差 | 统一尺度，防止某些维度过大在 Attention 内积中获不正当优势 |
| 乘 γ 加 β | 归一化会丢信息，γ 和 β 负责"恢复"模型觉得有用的信息 |

### 4.2 关键设计

**为什么沿 C 维度归一化（不是 T 维度）？** 沿 C：每个 token 独立归一化。沿 T：跨 token 归一化 → 每来新 token 所有旧 token 都要重算 → KV Cache 无法工作。

**RMSNorm（LLaMA/Mistral 用）**：去掉了减均值。实验发现减均值贡献小但消耗 10-15% 计算 → 只除均方根，更快。

**eps=1e-5 不是 1e-8？** FP16 最小有效数字约 6e-5，1e-8 精度不够会被吞掉。

**γ=1、β=0 初始化？** 恒等初始化——刚开始 LayerNorm 不做任何事，让模型先学会 Attention/MLP。

---

## 5. CausalSelfAttention

### 5.1 `__init__`

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # ① Fused QKV：一次矩阵乘法同时算 Q/K/V → 比分开三次省两次 kernel launch
        self.c_attn   = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # ② Output Projection：跨头整合各头信息
        self.c_proj   = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # ③ Causal Mask：下三角矩阵，buffer 不参与梯度但随模型保存/迁移
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        # bias shape: (1, 1, 1024, 1024) → 前两维为1为了广播匹配 (B, nh, T, T)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.attn_dropout = nn.Dropout(config.dropout)   # Attention 权重上
        self.resid_dropout = nn.Dropout(config.dropout)   # 残差路径上
```

### 5.2 `forward` — 10 步 Shape 追踪

```python
def forward(self, x):
    B, T, C = x.size()                        # x: (B, T, 768)

    # Step 1-2: Fused QKV → 切分成 Q/K/V
    qkv = self.c_attn(x)                      # (B, T, 2304)
    q, k, v = qkv.split(self.n_embd, dim=2)   # 三个 (B, T, 768)

    # Step 3: 重塑为多头 (B, nh, T, hs)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, 12, T, 64)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    # Step 4-5: 注意力分数
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, 12, T, T)

    # Step 6: Causal Mask — 下三角保留，上三角置 -inf
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

    # Step 7: Softmax + Dropout
    att = F.softmax(att, dim=-1)              # 每行和=1
    att = self.attn_dropout(att)

    # Step 8: 加权求和
    y = att @ v                                # (B, 12, T, 64)

    # Step 9: 合并多头
    y = y.transpose(1, 2).contiguous().view(B, T, C)  # contiguous() 必须！transpose 后内存不连续

    # Step 10: 输出投影
    y = self.c_proj(y)                        # (B, T, 768)
    return self.resid_dropout(y)
```

### 5.3 Shape 速查表

| Step | 操作 | 输入 | 输出 |
|:--:|------|------|------|
| 1 | c_attn (Fused QKV) | (B, T, 768) | (B, T, 2304) |
| 2 | split → Q, K, V | (B, T, 2304) | 3×(B, T, 768) |
| 3 | view + transpose | (B, T, 768) | (B, 12, T, 64) |
| 4 | Q @ K^T | (B, 12, T, 64) | (B, 12, T, T) |
| 5 | / √64 | (B, 12, T, T) | (B, 12, T, T) |
| 6 | causal mask | (B, 12, T, T) | (B, 12, T, T) |
| 7 | softmax + dropout | (B, 12, T, T) | (B, 12, T, T) |
| 8 | att @ V | (B, 12, T, T) | (B, 12, T, 64) |
| 9 | transpose + view | (B, 12, T, 64) | (B, T, 768) |
| 10 | c_proj | (B, T, 768) | (B, T, 768) |

### 5.4 FAQ

**Q: 为什么除以 √hs？**
Q@K^T 中每个元素是 hs 个独立随机变量的内积，方差=hs。不除 → softmax 过尖锐 → 梯度消失。

**Q: causal mask 为什么用 -inf 不用 -1e9？**
大 T 时 -1e9 不够小，softmax 可能残留非零概率。-inf 保证 softmax(-inf)=0。

**Q: transpose 后为什么必须 `contiguous()`？**
transpose 只改 stride 不改内存 → 不连续。view 要求连续内存。不 contiguous 就报错。

**Q: c_proj 维度 (768,768) 没变化，为什么还要？**
各头独立计算的结果需要跨头混合。6 个头 = 6 份报告，c_proj = 主编整合。

**Q: 训练时 attention dropout=0.1，推理时呢？**
推理时自动关闭。`model.eval()` 控制——nn.Dropout 在 eval 模式自动 bypass。

**Q: MHA vs MQA vs GQA？**
MHA：每头独立 KV（GPT-2）；MQA：全共享 KV；GQA：分组共享（LLaMA）。GQA 让 KV Cache 缩小 2-8 倍。

**Q: 三种 dropout 有什么区别？**

| 类型 | 位置 | 效果 |
|------|------|------|
| Embedding dropout | tok_emb+pos_emb 后 | 随机置零某些特征维度 |
| Attention dropout | softmax 后的 attention weights | 随机切断某些注意力连接 |
| Residual dropout | 每个子层输出 | 随机丢弃残差贡献 |

训练时生效，`model.eval()` 后全部关闭。

**Q: FlashAttention 和标准 Attention 结果一样吗？**
数学上等价（近似在浮点精度内），显存 O(N) vs O(N²)。标准 Attention 的 (T,T) 矩阵先写 HBM 再读 → FlashAttention 切成小块在 SRAM 里算完 → HBM 读写降 4-8 倍。PyTorch 2.0+ 已内置。

**Q: PagedAttention 是什么？**
KV Cache 按固定大小 Block 管理 → 消除碎片 + 写时复制（beam search 共享前缀）。vLLM 核心技术。

## 6. MLP

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)  # 768→3072
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)  # 3072→768
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)       # (B,T,768) → (B,T,3072)
        x = self.gelu(x)       # 逐元素激活
        x = self.c_proj(x)     # (B,T,3072) → (B,T,768)
        return self.dropout(x)
```

膨胀 4 倍给模型足够空间做非线性变换，再压缩回来对接下一层。**GELU vs ReLU**：ReLU 在 x<0 时归零（死神经元风险），GELU 处处光滑、x<0 也有小梯度，深层网络用 GELU。

---

## 7. Block

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # x + Attn(Norm(x))
        x = x + self.mlp(self.ln_2(x))    # x + MLP(Norm(x))
        return x
```

**Pre-Norm** 结构：先归一化 → 再计算 → 最后残差加回。梯度通过残差路径直传浅层 → 训练不需要 lr warmup。GPT-2/3、LLaMA 全用 Pre-Norm。

**残差的作用**：`out = x + Attention(LayerNorm(x))`。不加残差：网络深了梯度消失。加了：梯度通过 x 这条高速路直达浅层。Attention/MLP 只需学"和输入不一样的部分"。

---

## 8. GPT 类

### 8.1 组件清单

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),   # (50257, 768)
            wpe  = nn.Embedding(config.block_size, config.n_embd),   # (1024, 768)
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # 12层
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight Tying: wte 和 lm_head 共享权重 → 省 38.6M 参数
        self.transformer.wte.weight = self.lm_head.weight
        # 初始化
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

### 8.2 forward

```python
def forward(self, idx, targets=None):
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)  # (T,) 位置 0,1,2...

    # ① Embedding
    tok_emb = self.transformer.wte(idx)     # (B, T) → (B, T, C)
    pos_emb = self.transformer.wpe(pos)     # (T,) → (T, C)
    x = self.transformer.drop(tok_emb + pos_emb)  # 相加（不是拼接！）

    # ② 12 层 Block
    for block in self.transformer.h:
        x = block(x)                        # (B, T, C) → (B, T, C)

    # ③ 输出
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)                # (B, T, C) → (B, T, vocab_size)

    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),  # (B*T, V)
                               targets.view(-1),                  # (B*T,)
                               ignore_index=-1)
        return logits, loss
    return logits, None
```

### 8.3 generate（自回归生成）

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # ① 裁剪到 block_size
        idx_cond = idx[:, -self.config.block_size:] if idx.size(1) > self.config.block_size else idx
        # ② 前向 → 只取最后一个位置
        logits = self(idx_cond)[0][:, -1, :]        # (B, vocab_size)
        # ③ Temperature → 概率 → 采样
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        # ④ 拼接
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

**为什么只取最后一个位置？** GPT 是因果的——只需最后一个位置的 logits 预测下一个 token。有 KV Cache 时前面的 hidden states 已缓存不重算。

**Temperature**：`probs = softmax(logits / T)`。T→0 保守（贪婪），T=1 正常，T→∞ 随机。

### 8.4 FAQ

**Q: FP16 是啥？**
16 位浮点数。省显存、算得快，但精度低。训练时 AMP 混合精度——大部分 FP16，关键处 FP32。

**Q: train() 和 eval() 区别？**
train()：dropout 生效、算梯度；eval()：dropout 关闭、不计算梯度（torch.no_grad）。

**Q: logits 和 loss 分别是什么？**
logits = 最后一层原始分数，shape=(B,T,V)，可正可负；loss = logits 和 targets 的交叉熵，一个标量，越小越好。

---

## 9. KV Cache

nanoGPT 每生成一个新 token 重算整个序列 → O(N²)。KV Cache 把算过的 K、V 存下来，新 token 只算新的 Q → O(N)。

```
无 KV Cache: 计算量 O(N²)
有 KV Cache: 计算量 O(N)
```

Llama-2 7B、T=4096、batch=1 → KV Cache 约 2.1GB。这是 vLLM（PagedAttention）和 LLaMA（GQA）的优化动机。

---

## 10. configure_optimizers — Weight Decay

```python
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    # 分组：矩阵参数（p.dim()>=2）做 weight decay；向量参数（bias/LN）不做
    decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params,   'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
```

**为什么 bias 和 LN 不做 weight decay？** weight decay = 把参数往 0 推。bias 推到 0 没意义；LN.weight 初始为 1（恒等），推到 0 会废掉整层。口诀：**矩阵做 decay，偏置和归一化不做。**

### 参数精确计算

| 组件 | 参数量 |
|------|--:|
| wte | 38,597,376 |
| wpe | 786,432 |
| 每层 (Attn c_attn + c_proj + MLP c_fc + c_proj + 2×LN) | 7,085,568 |
| 12 层 | 85,026,816 |
| ln_f | 1,536 |
| **总计** | **~124.4M** |

> 面试心算：`C=768, C²≈590K, 12×C²≈7M, ×12层≈84M, +embedding≈124M`

---

## 11. 自测题

> 闭卷做完再对答案。

### 11.1 LayerNorm

| # | 问题 | 答案 |
|:--:|------|------|
| 1 | eps=1e-5，写成 1e-8 为什么不行？ | FP16 最小有效数字 ~6e-5，1e-8 精度不够 |
| 2 | 减均值和除标准差各自解决什么？ | 减均值→中心化消除范数差异；除标准差→统一尺度防止内积不正当优势 |
| 3 | RMSNorm 去掉了什么？凭什么？ | 去掉减均值。贡献小但消耗 10-15% 计算。LLaMA/Mistral 已采用 |

### 11.2 CausalSelfAttention

| # | 问题 | 答案 |
|:--:|------|------|
| 4 | Q/K/V 从 (B,T,C) 到 (B,nh,T,hs) 的变换？ | `view(B,T,nh,hs).transpose(1,2)` |
| 5 | `Q@K^T` shape？`att[0][0][3][5]` 含义？ | (B,nh,T,T)。batch0 head0 中 token3 对 token5 的注意力原始分数 |
| 6 | 为什么除以 √hs？ | Q@K^T 方差=hs。不除→softmax 尖锐→梯度消失 |
| 7 | causal mask 为什么用 -inf？ | 大 T 时 -1e9 不够小，softmax 可能残留非零。softmax(-inf)=0 绝对 |
| 8 | transpose 后为什么必须 contiguous()？ | transpose 只改 stride → 内存不连续 → view 报错 |
| 9 | Fused QKV 比三个独立 Linear 好在哪？ | 一次矩阵乘法 = 省两次 kernel launch |
| 10 | c_proj 维度没变，为什么还要？ | 6 个头各自算完，需要跨头混合 = 主编整合 |
| 11 | 推理时 dropout？谁控制？ | 自动关闭。model.eval() → nn.Dropout bypass |
| 12 | FlashAttention vs 标准 Attention？ | 数学等价，显存 O(N) vs O(N²) |
| 13 | T=2048 时 `Q@K^T` 矩阵多大？ | 2048²×2B=8MB。12层×12头=144个≈1.15GB |
| 14 | MHA/MQA/GQA？ | MHA：每头独立KV；MQA：共享KV；GQA：分组共享→KV Cache 缩小 2-8x |

### 11.3 MLP + Block

| # | 问题 | 答案 |
|:--:|------|------|
| 15 | MLP 为什么膨胀 4 倍？ | 给足够空间做非线性变换。LLaMA SwiGLU 约 2.67× |
| 16 | GELU vs ReLU？ | ReLU：x<0 归零（死神经元）；GELU：处处光滑，深层网络首选 |
| 17 | Pre-Norm vs Post-Norm？ | Pre-Norm：先 Norm 再计算 → 梯度直传浅层 → 不需 lr warmup |

### 11.4 GPT + generate

| # | 问题 | 答案 |
|:--:|------|------|
| 18 | Weight Tying 省了多少？ | ~38.6M（1/3 总参数）。wte 和 lm_head 互为转置，共享权重 |
| 19 | `logits.view(-1, V)` 的 -1 是多少？ | B×T。展平所有 token 逐 token 算 loss |
| 20 | generate 为什么只取 `[:, -1, :]`？ | 因果模型只需最后一个位置预测下一个 token |
| 21 | KV Cache 从 O(?) 到 O(?)？ | 无：O(N²)；有：O(N) |
| 22 | 哪些参数不做 weight decay？ | bias 和 LN.weight。推到 0 无意义或有害 |
| 23 | c_proj 初始化为什么 /sqrt(2×n_layer)？ | GPT-2 风格：残差初始很小 → 近乎恒等起步 → 不需 lr warmup |
| 24 | GPT-2 124M 估算？ | wte(~38.6M)+wpe(~0.79M)+12×~7.09M≈124M |

---

## 附录 A：参数速查

| 缩写 | 全称 | shape | 作用 |
|------|------|------|------|
| `wte` | Word Token Embedding | (50257, 768) | token ID → 向量 |
| `wpe` | Word Position Embedding | (1024, 768) | 位置 → 向量 |
| `c_attn` | Combined Attention (Fused QKV) | (768, 2304) | 输入 → QKV 拼接 |
| `c_proj`(attn) | Combined Projection | (768, 768) | 跨头混合 |
| `c_fc` | Combined Fully-Connected | (768, 3072) | MLP 膨胀 4× |
| `c_proj`(mlp) | Combined Projection | (3072, 768) | MLP 压缩 |
| `ln_1/ln_2` | LayerNorm | (768,) | Block 内归一化 |
| `ln_f` | LayerNorm Final | (768,) | 最终归一化 |
| `lm_head` | Language Model Head | (768, 50257) | hidden → logits |

## 附录 B：函数速查

| 函数 | 位置 | 做什么 |
|------|------|--------|
| `LayerNorm.__init__` | model.py ~30 | 初始化 γ(全1)、β(全0) |
| `LayerNorm.forward` | model.py ~40 | 减均值→除标准差→γx̂+β |
| `CausalSelfAttention.__init__` | model.py ~55 | Fused QKV、c_proj、causal mask(buffer)、dropout |
| `CausalSelfAttention.forward` | model.py ~70 | 10 步：QKV→多头→scores→mask→softmax→加权→合并→c_proj |
| `MLP.__init__` | model.py ~105 | c_fc(768→3072)、GELU、c_proj(3072→768) |
| `MLP.forward` | model.py ~112 | c_fc→gelu→c_proj→dropout |
| `Block.__init__` | model.py ~120 | ln_1、attn、ln_2、mlp |
| `Block.forward` | model.py ~125 | x+attn(ln_1(x)); x+mlp(ln_2(x)) |
| `GPT.__init__` | model.py ~135 | wte、wpe、h(12×Block)、ln_f、lm_head、Weight Tying、初始化 |
| `GPT.forward` | model.py ~170 | emb→12层Block→ln_f→lm_head→loss |
| `GPT.generate` | model.py ~220 | 裁剪→前向→取最后→temperature→topk→采样→拼接 |
| `GPT.configure_optimizers` | model.py ~260 | 矩阵参数 decay / bias+LN 不 decay |

## 附录 C：面试话术

> "我完整阅读过 nanoGPT 源码，能讲清楚 CausalSelfAttention 从 QKV Projection 到 output projection 的 10 步形状变化全链路，包括 Fused QKV 的设计动机、transpose 后 contiguous() 的原因、以及 causal mask 的 -inf 与 softmax 的交互。"
>
> "基于对 Pre-Norm 残差路径的理解，我能解释为什么现代 LLM 全面采用 Pre-Norm——梯度通过 identity path 直接反向传播，消除了对 lr warmup 的依赖。"

## 附录 D：24 问索引

| # | 问题 | 位置 |
|:--:|------|------|
| 1 | 离散整数→连续向量 | §1.2 |
| 2 | embedding 相加后信息还在吗 | §3.5 |
| 3 | wte、wpe 参数说明 | §8.1 + 附录 A |
| 4 | shape 是啥 | §1.4-1.5 |
| 5 | 函数说明 | 附录 B |
| 6 | 源码注释应有步骤 | 全书按源码顺序 |
| 7 | softmax 数学原理 | §3.2 |
| 8 | 加权融合 | §3.1 |
| 9 | nh 个头是啥 | §3.5 |
| 10-11 | 源码步骤 | §5.2-5.3 |
| 12 | C/T 维度 | §1.5 |
| 13 | FP16 | §8.4 |
| 14 | 随机切断注意力 | §5.4 FAQ |
| 15 | transpose/stride/contiguous | §2.4 + §5.4 FAQ |
| 16 | c_proj 信息重组 | §5.4 FAQ |
| 17 | 先膨胀再压缩 | §6 |
| 18 | stride vs view | §2.4 |
| 19 | 残差 | §7 |
| 20 | dropout 三种 | §5.4 FAQ |
| 21 | loss/logits | §8.2 + §8.4 |
| 22 | train/eval 区别 | §8.4 |
| 23 | Temperature | §8.3 |
| 24 | weight decay | §10 |

---

## 🔨 手搓代码

> 关掉 nanoGPT/model.py，从零写。写完对照源码。

| # | 题目 | 时限 | 验收 |
|:--:|------|:--:|------|
| 1 | 手写 CausalSelfAttention | 90min | (B=2,T=8,C=64,nh=4) 跑通 |
| 2 | 手写 GPT Block | 60min | (B,T,C) in → (B,T,C) out |
| 3 | 手写完整 GPT forward | 90min | (B=2,T=16) token IDs → logits |
| 4 | 手写 KV Cache 推理 | 60min | 有/无 KV Cache 输出一致 |

代码位置：`d:\study\llm.c-learning\experiments\handwrite-gpt\`
