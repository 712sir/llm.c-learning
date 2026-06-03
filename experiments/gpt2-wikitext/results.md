# GPT-2 (124M) on WikiText-103

> Week 1 Day 4-5 · AutoDL RTX 4090D 24GB

## 最终结果

| 指标 | 值 |
|------|-----|
| iter_num | 5000 |
| best_val_loss | **3.045815** |
| perplexity | ~21.0 |
| 总耗时 | ~4h 27min |
| MFU | ~27% |
| Checkpoint | 1.4 GB |

## 完整配置

```python
# config/train_gpt2_wikitext.py
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 40  # effective batch = 327,680 tokens
max_iters = 5000                   # ~1.6B tokens, ~13 epochs
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 2000
lr_decay_iters = 5000
weight_decay = 1e-1
grad_clip = 1.0
dtype = bfloat16
compile = True
dataset = wikitext
wandb_log = False
```

## 模型架构 (GPT-2 124M)

```python
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
vocab_size = 50304
bias = False
dropout = 0.0
```

## Loss 曲线

```
iter   0: loss 11.01    (随机初始)
iter 100: loss  7.66
iter 200: loss  6.40  val 6.32
iter 300: loss  5.92
...
iter 5000:              val 3.0458
```

## 关键踩坑

1. **磁盘不足**：系统盘 30G → 数据迁移到 50G 数据盘 + 符号链接
2. **OpenWebText 不可用**：HF 被墙 + 镜像不支持 XetHub → 换 WikiText-103
3. **SSH 不稳**：本地直连频繁断开 → VS Code Remote-SSH 更稳定
4. **镜像可用性差异**：WikiText-103（旧存储格式）hf-mirror 可用，OpenWebText（XetHub）不可用
5. **训练日志丢失**：nanoGPT 默认只输出到终端，不写文件。以后启动训练时加 `2>&1 | tee training_log.txt` 保存日志

## 生成样本

| Temperature | 效果 | 文件 |
|-------------|------|------|
| **0.6** | 偏保守，结构清晰，但容易重复（"O'Brien" 循环） | [samples_t06.txt](samples_t06.txt) |
| **0.8** | 最佳平衡——句子通顺 + 多样性好 | [samples_t08.txt](samples_t08.txt) |
| **1.0** | 创意更多，偶尔跑偏（"make love"） | [samples_t10.txt](samples_t10.txt) |

共同特征：学会了 WikiText 的结构（`== Section ==`）、日期/人名/数字格式、段落衔接。推荐 T=0.8。
