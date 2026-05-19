# Week 3：关键问题自测 + 代码注释归档

> 状态：🔴 未开始

---

## 10 个核心问题逐题回答

### Attention 相关

#### 1. CausalSelfAttention 的 causal mask 为什么是下三角矩阵？不 mask 会怎样？

**回答**：
- 
- 
- 如果在训练时不加 causal mask，模型会____________
- 推理时可以不加吗？____________，因为____________

#### 2. `scaled_dot_product_attention` 的 `is_causal=True` 底层做了什么优化？

**回答**：
- 
- FlashAttention 利用了 causal 的特性来____________

#### 3. 多头注意力的"多头"体现在代码的哪一行？

**回答**：
- 代码行：`q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)`
- 为什么能实现多头？因为____________

#### 4. 为什么除以 `sqrt(head_size)`？不除会怎样？

**回答**：
- Q 和 K 的每个元素是____________
- Q @ K^T 是 hs 个独立随机变量的内积 → 方差为______
- 如果不除 sqrt(hs)，大的 hs 会导致 softmax 过于______

---

### 训练相关

#### 5. AMP 的 GradScaler 的 scale/unscale 在什么时候调用？为什么 backward 之后才 unscale？

**回答**：
```
with ctx:                      → 前向用 FP16
    logits, loss = model()
loss = scaler.scale(loss)      → ① 放大 loss
loss.backward()                → ② 反向（梯度也被放大了）
scaler.unscale_(optimizer)     → ③ 缩回原始尺度 ← 关键！
clip_grad_norm_()              → ④ 在真实尺度上做 clip
scaler.step(optimizer)         → ⑤ 优化器更新
scaler.update()                → ⑥ 调整 scale factor
```

为什么这个顺序？
- 

#### 6. Gradient Accumulation 和增大 batch_size 等价吗？什么时候不等价？

**回答**：
- 等价的条件：____________
- 不等价的条件：____________（如 BatchNorm 层）

#### 7. 学习率 warmup 为什么必要？

**回答**：
- 

#### 8. Cosine LR Scheduler 比固定 lr 好在哪？

**回答**：
- 初期：____________
- 后期：____________

---

### 推理相关

#### 9. `model.generate()` 里为什么要 `idx_cond = idx[:, -max_new_tokens:]`？

**回答**：
- 因为模型只能处理 block_size 个 token
- 截断后问题：____________
- 这就是 KV Cache 要解决的问题

#### 10. Temperature 和 Top-K 采样有什么区别？什么场景用哪个？

**回答**：

| 方法 | 原理 | 适用场景 |
|------|------|---------|
| Temperature | 调整分布的尖锐程度 | — |
| Top-K | 只保留 K 个最可能的 token | — |

---

## 3 分钟口头回答：Attention 正向传播

> 录音自测：能否脱稿讲清以下 8 步？

1. 输入 x 经过 c_attn 线性层，输出 3C 维（QKV 拼接）
2. 拆成 Q、K、V，各自 reshape 成 multi-head 形状
3. Q @ K^T 得到 attention scores，除以 sqrt(head_size)
4. 加上 causal mask（下三角保留，上三角置 -inf）
5. softmax 得到 attention weights
6. attention weights @ V 得到加权输出
7. 经过 c_proj 投影回 C 维
8. 加上残差连接

**自评**：卡壳的地方 → ____________

---

## 代码注释归档

- nanoGPT fork 地址：https://github.com/____________/nanoGPT
- 已注释文件：
  - [ ] model.py
  - [ ] train.py
  - [ ] sample.py
  - [ ] configurator.py

---

## 阶段检查清单

- [ ] 10 个核心问题全部手写回答了
- [ ] nanoGPT fork 中 model.py 和 train.py 有完整中文注释
- [ ] 能脱稿讲清 Attention forward 的 8 个步骤
- [ ] 能脱稿讲清训练循环的每一步
- [ ] 3 分钟口头回答录音（自己听一遍，看哪里卡壳）
