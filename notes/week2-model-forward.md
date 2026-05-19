# Week 2：逐文件精读 nanoGPT 源码

> 状态：🔴 未开始

---

## Day 1：model.py 上半部分 —— 模型结构定义

### 1. LayerNorm 类

**代码位置**：[model.py](../Project-nanoGPT/model.py) 开头

**阅读要点**：
- `__init__`: weight 和 bias 分别是什么？
- `forward`: 均值、方差、归一化、仿射变换，每一步的 shape 是什么？
- 为什么 PyTorch 自带 `nn.LayerNorm` 但这里要重新实现？

**手算验证**（输入 `x.shape = (B, T, C)`，B=2, T=3, C=4）：

| 步骤 | Shape | 说明 |
|------|-------|------|
| mean | (B, T, 1) | 沿 C 维度求均值 |
| var | (B, T, 1) | 沿 C 维度求方差 |
| xhat | (B, T, C) | 归一化 |
| output | (B, T, C) | 仿射变换 |

---

### 2. CausalSelfAttention 类

**代码位置**：[model.py](../Project-nanoGPT/model.py) `CausalSelfAttention`

#### `__init__` 阅读

- `c_attn`：输入维度 ______，输出维度 ______（为什么是 3×？）
- `c_proj`：做什么用？
- `attention bias`：tril 矩阵长什么样？

```python
# 跑一下看看：
import torch
print(torch.tril(torch.ones(5, 5)))
```

- 为什么 attention bias 要注册为 `buffer` 而不是 `parameter`？
  - 答：____________

#### `forward` 阅读

- 输入 `x.shape = (B, T, C)`，第一步做了什么？
- Q, K, V 拆分后的 shape 分别是什么？`(B, nh, T, hs)` 其中 `C = nh * hs`
- `att = (Q @ K^T) * (1.0 / sqrt(hs))` — 为什么除以 `sqrt(hs)`？
  - 答：____________
- causal mask 怎么加上的？`masked_fill` 把什么位置填成了 `-inf`？
- softmax 之后，`att @ V` 的 shape 是什么？
- 最后 `y = c_proj(y)`，为什么还需要一次线性变换？

---

### Attention 数据流（手绘）

```
输入 x (B, T, C)
    │
    ▼
c_attn: Linear(C, 3C)
    │
    ▼
Q (B,nh,T,hs)    K (B,nh,T,hs)    V (B,nh,T,hs)
    │                  │                 │
    └──── Q @ K^T ─────┘                 │
            │                            │
            ▼                            │
    scores (B,nh,T,T)                    │
            │                            │
            ▼                            │
    / sqrt(hs) + mask + softmax          │
            │                            │
            ▼                            │
    att_weights (B,nh,T,T)               │
            │                            │
            └──── att @ V ───────────────┘
                        │
                        ▼
                y (B,nh,T,hs) → reshape → (B,T,C)
                        │
                        ▼
                c_proj: Linear(C, C)
                        │
                        ▼
                output (B, T, C)
```

---

## Day 2：model.py 下半部分 —— MLP + Block + GPT

### 3. MLP 类

- `c_fc`：输入 C，输出 4×C（为什么是 4 倍？）
- GELU vs ReLU 曲线对比：[手绘图]
- `c_proj`：和 c_fc 对称

### 4. Block 类

- 结构：`LayerNorm → Attention → Residual → LayerNorm → MLP → Residual`
- 这里用的是 Pre-Norm 还是 Post-Norm？
  - 答：____________
- 残差连接为什么能缓解梯度消失？
  - 答：____________

### 5. GPT 类

| 组件 | Shape | 说明 |
|------|-------|------|
| `wte` | (vocab_size, n_embd) | Token Embedding |
| `wpe` | (block_size, n_embd) | Position Embedding |
| `h` | n_layer × Block | Transformer Blocks |
| `ln_f` | LayerNorm | 最终 LayerNorm |
| `lm_head` | (n_embd, vocab_size) | 输出投影 |

- 为什么 `wte` 和 `lm_head` 共享权重？

### 6. forward 函数

- 输入 `idx (B, T)`，如果有 targets 则返回 loss
- `logits.shape = (B, T, vocab_size)`
- loss 怎么算？为什么要把第一二维合并？

### 7. configure_optimizers

- 哪些参数不做 weight decay？为什么？

---

## Day 5：手绘完整 GPT 模型结构图

> 图见 [diagrams/gpt-model-arch.png](../diagrams/gpt-model-arch.png)

---

## 阶段检查清单

- [ ] 能默画 Attention 数据流图（含 shape 标注）
- [ ] 能默画完整 GPT 模型结构图
- [ ] 每读完一个类，在代码旁写了中文注释
