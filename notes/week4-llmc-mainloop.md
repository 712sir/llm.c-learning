# Week 4：llm.c 主循环精读 —— 从 PyTorch 到纯 C

> 状态：⬜ 待进行 | 前置：Week 2 精读完成
>
> 源码：[karpathy/llm.c/train_gpt2.c](https://github.com/karpathy/llm.c/blob/master/train_gpt2.c)
> 对照：nanoGPT `train.py` — 同一个训练流程，一个 Python，一个纯 C

---

## 1. llm.c 是什么

llm.c 是 Karpathy 用纯 C 写的 GPT-2 训练代码——没有 PyTorch，没有 autograd，所有 forward、backward、optimizer 全部手写。目标是让你看到深度学习框架的"底层到底在干什么"。

**nanoGPT（Python）** → PyTorch 帮你做了 backward、GPU 调度、混合精度
**llm.c（纯 C）** → 你看到每一个 matmul、每一个 softmax、每一个梯度是怎么算的

---

## 2. 项目结构速览

```
llm.c/
├── train_gpt2.c         # 训练主程序（纯C，CPU版本，~1200行）
├── train_gpt2.py         # 训练参考代码（PyTorch，用来验证 C 版的正确性）
├── train_gpt2.cu         # CUDA 版本
├── train_gpt2_fp32.cu    # CUDA FP32 版本
├── test_gpt2.c           # 单元测试
├── llmc/                 # 核心库
│   ├── dataloader.h      # mmap 数据加载
│   ├── tokenizer.h       # BPE Tokenizer
│   ├── matmul.h          # 矩阵乘法（手写 forward/backward）
│   ├── attention.h       # Attention forward/backward
│   ├── layernorm.h       # LayerNorm
│   ├── gelu.h            # GELU 激活
│   ├── encoders.h        # Transformer Block + GPT 主结构
│   └── adamw.h           # AdamW 优化器
├── dev/
│   ├── cuda/             # CUDA kernel 开发
│   ├── data/             # 数据预处理
│   └── eval/             # 评估脚本
└── Makefile
```

对比 nanoGPT：nanoGPT 的 `model.py` = llm.c 的 `encoders.h + attention.h + layernorm.h + gelu.h`。nanoGPT 的 `train.py` = llm.c 的 `train_gpt2.c`。

---

## 3. nanoGPT ↔ llm.c 完整对照表

| 功能 | nanoGPT (Python) | llm.c (C) | 文件 |
|------|-----------------|-----------|------|
| 模型参数定义 | `GPT.__init__()` | `gpt2_init()` | encoders.h |
| 前向传播 | `GPT.forward()` | `gpt2_forward()` | encoders.h |
| 反向传播 | `loss.backward()` (autograd) | `gpt2_backward()` (手写) | encoders.h |
| Attention 前向 | `CausalSelfAttention.forward()` | `attention_forward()` | attention.h |
| Attention 反向 | autograd 自动 | `attention_backward()` (手写) | attention.h |
| LayerNorm 前向 | `LayerNorm.forward()` | `layernorm_forward()` | layernorm.h |
| LayerNorm 反向 | autograd 自动 | `layernorm_backward()` (手写) | layernorm.h |
| GELU | `F.gelu(x)` | `gelu_forward()` / `gelu_backward()` | gelu.h |
| 矩阵乘法 | `@` / `torch.matmul` | `matmul_forward()` / `matmul_backward()` | matmul.h |
| CrossEntropy | `F.cross_entropy()` | `crossentropy_forward()` / `crossentropy_backward()` | encoders.h |
| 数据加载 | `get_batch()` | `dataloader_next_batch()` | dataloader.h |
| 优化器 | `optimizer.step()` | `adamw_update()` | adamw.h |

**核心区别**：nanoGPT 的 backward 是 PyTorch autograd 自动算的；llm.c 每个 backward 函数都是**手写的**——这是理解反向传播本质的最好机会。

---

## 4. 调用链地图

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
       │    ├─ encoder_forward()
       │    │    ├─ layernorm_forward()
       │    │    ├─ matmul_forward(QKV)
       │    │    ├─ attention_forward(Q, K, V)
       │    │    │    ├─ matmul_forward(Q @ K^T)
       │    │    │    ├─ softmax（手写循环）
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
       │         ├─ attention_backward()
       │         │    ├─ matmul_backward(att @ V)
       │         │    ├─ softmax_backward
       │         │    └─ matmul_backward(Q @ K^T)
       │         └─ layernorm_backward(first)
       │
       └─ adamw_update(model)
```

---

## 5. 两个训练循环逐行对比

| 步骤 | nanoGPT (Python) | llm.c (C) |
|------|---------|------|
| 数据加载 | `X, Y = get_batch('train')` | `dataloader_next_batch(batch_loader)` — C 版需要异步加载 |
| 前向 | `logits, loss = model(X, Y)` | `gpt2_forward(model, inputs, targets, B, T)` — C 版需要手动传 B 和 T |
| 梯度归零 | `optimizer.zero_grad()` | 手动 `memset(grads, 0)` — 在 adamw_update 之后做 |
| 反向 | `loss.backward()` | `gpt2_backward(model)` — **每个 backward 都是手写的** |
| 梯度裁剪 | `clip_grad_norm_()` | 手动遍历梯度、算 norm、clip |
| 参数更新 | `optimizer.step()` | `adamw_update(model, lr, beta1, beta2, ...)` — 手动传所有超参数 |
| 学习率调度 | Cosine + Warmup（PyTorch 内置） | 相同的公式，手写循环计算 lr |
| AMP | `torch.amp.autocast()` | ❌ 纯 C 版不支持 FP16 |
| 日志 | `tqdm` + `print` | `printf` — 自己算 tokens/sec |

---

## 6. 验证环境

```bash
cd llm.c
make train_gpt2          # 编译纯 C 版本
./train_gpt2             # 跑一次训练
python train_gpt2.py     # PyTorch 参考版本，用于验证 C 版 loss 是否正确
```

---

## 7. 自测题

| # | 问题 | 答案 |
|:--:|------|------|
| 1 | llm.c 的 backward 和 nanoGPT 的 backward 本质区别？ | nanoGPT 用 autograd 自动；llm.c 每个 backward 函数手写 |
| 2 | 为什么要手写 backward？ | 理解反向传播本质；CUDA kernel 优化需要手写反向 |
| 3 | `gpt2_forward` 和 `GPT.forward` 参数上有什么区别？ | C 版需要显式传 B、T；Python 版从 tensor shape 自动推导 |
| 4 | 纯 C 版为什么不支持 FP16？ | FP16 运算需要 GPU 硬件支持，纯 CPU 代码只能 FP32 |
| 5 | 数据加载用的是什么？ | mmap 内存映射，零拷贝读取预处理好的 `.bin` 文件 |
| 6 | 纯 C 版梯度怎么归零？ | 手动 `memset(grads, 0, num_params * sizeof(float))` |
| 7 | llm.c 的 attention_forward 里 softmax 怎么实现的？ | 手写循环，先 exp 再归一化，没有 PyTorch 的数值稳定技巧需要自己加 |
