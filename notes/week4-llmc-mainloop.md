# Week 4：从 train_gpt2.c 主循环入手

> 状态：🔴 未开始

---

## Day 1：找到所有关键文件

### llm.c 目录结构

```
llm.c/
├── train_gpt2.c        # 训练主程序（纯C，CPU版本）
├── train_gpt2.py       # 训练参考代码（PyTorch，用于验证C版本正确性）
├── llmc/               # 核心库目录
│   ├── dataloader.h    # 数据加载器
│   ├── tokenizer.h     # BPE Tokenizer
│   ├── matmul.h/c      # 矩阵乘法（纯C实现）
│   ├── attention.h/c   # Attention 前向和反向
│   ├── layernorm.h/c   # LayerNorm
│   ├── gelu.h/c        # GELU 激活函数
│   ├── encoders.h/c    # Transformer Block + GPT 主结构
│   └── adamw.h/c       # AdamW 优化器
├── dev/                # CUDA kernel 开发目录
│   └── cuda/           # CUDA kernel 实现
└── Makefile
```

### 验证环境

```bash
cd llm.c
python train_gpt2.py
# 输出：
```

---

## Day 2：精读 train_gpt2.c 的 main 函数

### 第 1 段：内存分配

- `num_parameters` 是怎么算出来的？
- 函数位置：`llmc/encoders.c` → `gpt2_num_parameters()`

### 第 2 段：数据加载

- `dataloader_open()` — 底层是 mmap
- `dataloader_next_batch()` — 返回 B*T 个 int（token ID）
- 这和 nanoGPT 的 `get_batch()` 对应关系：

### 第 3 段：训练循环

```c
for (int step = 0; step <= max_steps; step++) {
    // 3a. 等数据加载
    // 3b. 前向传播：gpt2_forward(model, inputs, targets, B, T)
    // 3c. 反向传播：gpt2_backward(model)
    // 3d. 优化器更新：adamw_update(model, lr, beta1, beta2, ...)
    // 3e. 日志输出
}
```

---

## Day 3：nanoGPT 与 llm.c 完整对照表

| 功能 | nanoGPT (Python) | llm.c (C) | 文件位置 |
|------|-----------------|-----------|---------|
| 模型参数定义 | `GPT.__init__()` | `GPT2.__init__()` | encoders.c |
| 前向传播 | `GPT.forward()` | `gpt2_forward()` | encoders.c |
| 反向传播 | `loss.backward()` | `gpt2_backward()` | encoders.c |
| Attention 前向 | `CausalSelfAttention.forward()` | `attention_forward()` | attention.c |
| Attention 反向 | Autograd 自动 | `attention_backward()` | attention.c |
| LayerNorm 前向 | `LayerNorm.forward()` | `layernorm_forward()` | layernorm.c |
| LayerNorm 反向 | Autograd 自动 | `layernorm_backward()` | layernorm.c |
| GELU 前向 | `F.gelu(x)` | `gelu_forward()` | gelu.c |
| GELU 反向 | Autograd 自动 | `gelu_backward()` | gelu.c |
| 矩阵乘法 | `@` / `torch.matmul` | `matmul_forward()` | matmul.c |
| CrossEntropy | `F.cross_entropy()` | `crossentropy_forward()` | encoders.c |
| 数据加载 | `get_batch()` | `dataloader_next_batch()` | dataloader.h |
| 优化器 | `optimizer.step()` | `adamw_update()` | adamw.c |

**关键区别**：
- nanoGPT 的 backward 是 PyTorch autograd 自动计算的
- llm.c 的每个 backward 函数都是**手写的**——这是理解反向传播本质的最好机会

---

## Day 4：调用链地图

> 图见 [diagrams/call-chain-map.png](../diagrams/call-chain-map.png)

```
main()
  │
  ├─ malloc(参数/梯度/AdamW状态)
  ├─ dataloader_open("train.bin")
  │
  └─ for step in range(max_steps):
       │
       ├─ dataloader_next_batch()  →  x, y
       ├─ gpt2_forward(model, x, y, B, T)
       │    ├─ encoder_forward(x, params, ...)
       │    │    ├─ layernorm_forward()
       │    │    ├─ matmul_forward(QKV)
       │    │    ├─ attention_forward(Q, K, V)
       │    │    │    ├─ matmul_forward(Q @ K^T)
       │    │    │    ├─ softmax
       │    │    │    └─ matmul_forward(att @ V)
       │    │    ├─ matmul_forward(output proj)
       │    │    ├─ layernorm_forward()
       │    │    ├─ matmul_forward(MLP fc)
       │    │    ├─ gelu_forward()
       │    │    └─ matmul_forward(MLP proj)
       │    ├─ layernorm_forward(final)
       │    ├─ matmul_forward(lm_head)
       │    └─ crossentropy_forward()
       │
       ├─ gpt2_backward(model)
       │    ├─ crossentropy_backward()
       │    └─ encoder_backward()
       │         ├─ matmul_backward(lm_head)
       │         ├─ layernorm_backward()
       │         ├─ ... (每个算子的 backward)
       │         ├─ attention_backward()
       │         └─ layernorm_backward(first)
       │
       └─ adamw_update(model)
```

---

## Day 5：两个训练循环逐行对比

| 步骤 | nanoGPT | llm.c 纯C | 理解 |
|------|---------|-----------|------|
| 数据加载 | `X, Y = get_batch('train')` | `dataloader_next_batch(batch_loader)` | C 版需要异步加载 |
| 前向 | `logits, loss = model(X, Y)` | `gpt2_forward(model, inputs, targets, B, T)` | C 版需要传 B 和 T |
| 梯度归零 | `optimizer.zero_grad()` | 手动 memset 为 0 | 在 adamw_update 之后做 |
| 反向 | `loss.backward()` | `gpt2_backward(model)` | C 版必须手写每个 backward |
| 梯度裁剪 | `clip_grad_norm_()` | 手动遍历梯度，计算 norm，clip | 同样是手写 |
| 参数更新 | `optimizer.step()` | `adamw_update(model, ...)` | 需要传所有超参数 |
| 学习率调度 | Cosine + Warmup | 相同的公式，手写 | 在循环里手动计算 lr |
| AMP | `torch.amp.autocast()` | ❌ 纯 C 版不支持 FP16 | 只有 CUDA 版才支持 |
