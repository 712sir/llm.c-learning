# Week 3：关键问题自测 —— 能脱稿讲才算真懂

> 状态：⬜ 待进行 | 前置：Week 2 精读完成
>
> 闭上笔记、合上代码，手写回答下面每一道题。写不出来 = 回去翻 Week 2 对应章节。

---

## 1. Attention 四问

### Q1 — CausalSelfAttention 的 causal mask 为什么是下三角矩阵？不 mask 会怎样？

训练时不 mask：每个 token 能看到未来的 token → 模型学会"作弊"（直接从未来抄答案）→ 推理时未来还不存在 → 模型报废。

mask 用下三角：token_i 只能看到 token_0...token_i，看不到 token_{i+1} 及以后。实现是上三角设 -inf → softmax(-inf)=0。

### Q2 — `scaled_dot_product_attention` 的 `is_causal=True` 底层做了什么优化？

PyTorch 2.0+ 调用 FlashAttention 内核：不生成完整的 (T,T) attention 矩阵写回 HBM，而是在 SRAM 里分块算完直接出结果，显存 O(N)，速度更快。等价于手写的 causal mask + softmax + dropout，只是省掉了中间矩阵。

### Q3 — 多头注意力的"多头"体现在代码的哪一行？

```python
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
```

把 768 维切成 12 个 64 维子空间，transpose 把 n_head 维提到 batch 后面 → 每个头独立做 Attention。最后拼回来：

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)
```

### Q4 — 为什么除以 `sqrt(head_size)`？不除会怎样？

Q@K^T 每个元素是 hs 个独立随机变量的内积，方差=hs。不除 → softmax 输入值过大 → softmax 过于尖锐（接近 one-hot）→ 梯度接近 0 → 训不动。除了 → 方差归一化到 1 → softmax 平滑 → 梯度正常。

---

## 2. 训练四问

### Q5 — AMP 的 GradScaler 的 scale/unscale 在什么时候调用？为什么 backward 之后才 unscale？

```
with autocast():            → 前向用 FP16
    logits, loss = model()

scaler.scale(loss)          → ① 放大 loss（防止小梯度在 FP16 下变 0）
loss.backward()             → ② 反向（梯度也被放大了）
scaler.unscale_(optimizer)  → ③ 缩回原始尺度 ← 关键！
clip_grad_norm_()           → ④ 在真实尺度上做梯度裁剪
scaler.step(optimizer)      → ⑤ 更新参数
scaler.update()             → ⑥ 动态调整 scale factor
```

**为什么这个顺序？** backward 时梯度被 scale 放大了 → FP16 下的小梯度不会下溢变 0。unscale 必须在 clipping 之前——你要在真实的梯度尺度上判断"有没有炸"，不能在放大后的尺度上剪。

### Q6 — Gradient Accumulation 和增大 batch_size 等价吗？什么时候不等价？

等价条件：所有操作都是"求和再除以 N"的线性操作（如 Linear、Attention 的加权求和）。

不等价条件：有 BatchNorm 层（每个 micro-batch 的统计量不同，都按各自统计量归一化后再平均，跟大 batch 的统计量不一样）。GPT-2 用 LayerNorm 不用 BatchNorm → nanoGPT 中 gradient accumulation 完全等价于大 batch。

### Q7 — 学习率 warmup 为什么必要？

训练初期，参数是随机的，梯度方向不稳定。如果一开始就用大 lr，可能一步走太远冲到奇怪的地方回不来。warmup 从很小的 lr 开始，让优化器先"摸清地形"，再加速。Pre-Norm 结构（GPT-2）对 warmup 需求较小，Post-Norm（原版 Transformer）没有 warmup 基本训不动。

### Q8 — Cosine LR Scheduler 比固定 lr 好在哪？

初期：lr 大 → 快速收敛。后期：lr 逐步减小 → 精细调参，不会在最优解附近震荡跳过去。Cosine 退火是平滑的衰减曲线，比 step decay（台阶式）更稳定。

---

## 3. 推理两问

### Q9 — `model.generate()` 里为什么要 `idx_cond = idx[:, -max_new_tokens:]`？

模型有最大上下文限制（block_size=1024）。如果 prompt + 已生成的 token 超过 1024，只保留最后 1024 个。问题是：前面 token 的信息丢失了。KV Cache 可以解决——所有历史 K、V 都缓存着，不重算也不丢失。

### Q10 — Temperature 和 Top-K 采样有什么区别？什么场景用哪个？

| 方法 | 原理 | 场景 |
|------|------|------|
| Temperature | 缩放 logits 再 softmax：T↓→分布更尖→更保守 | 控制"创造性"的整体基调 |
| Top-K | 只保留 K 个最高分候选，其余置 -inf | 过滤掉明显不合理的 token |

通常一起用：先 Temperature 调分布 → 再 Top-K 剪枝 → 最后 multinomial 采样。

---

## 4. 口头自测：Attention forward 8 步

> 录音或对镜子讲，卡壳的地方 = 没真懂。

| 步 | 内容 |
|:--:|------|
| 1 | x 经过 c_attn 线性层 → 输出 3C 维（Q/K/V 拼在一起） |
| 2 | split 拆成 Q、K、V，各 C 维 |
| 3 | view+transpose → 多头格式 (B, nh, T, hs) |
| 4 | Q @ K^T → 注意力分数矩阵 (B, nh, T, T) |
| 5 | 除以 √hs |
| 6 | causal mask：上三角置 -inf |
| 7 | softmax → 概率分布 |
| 8 | att @ V → transpose+view 合并多头 → c_proj → 加残差 |

**卡壳的地方**：__________

---

## 5. 检查清单

- [ ] 10 道题全部手写回答，然后对照 Week 2 笔记
- [ ] 能脱稿讲清 Attention forward 8 步
- [ ] 能脱稿讲清训练循环每一步（data → forward → backward → clip → update）
- [ ] 能解释 AMP scale/unscale 顺序
- [ ] nanoGPT model.py 里有自己写的中文注释

---

## 附录 A：10 题答案索引

| # | 问题 | Week 2 位置 |
|:--:|------|------|
| 1 | causal mask 下三角 | §3.3, §5.4 FAQ |
| 2 | is_causal 底层优化 | §5.4 FAQ（FlashAttention） |
| 3 | 多头体现在哪一行 | §5.2 |
| 4 | 为什么除以 √hs | §5.4 FAQ |
| 5 | AMP scale/unscale 顺序 | §A（本笔记） |
| 6 | Gradient Accumulation | §A（本笔记） |
| 7 | lr warmup | §A（本笔记） |
| 8 | Cosine LR Scheduler | §A（本笔记） |
| 9 | generate 裁剪 | §8.3 |
| 10 | Temperature vs Top-K | §8.3 + §A（本笔记） |
