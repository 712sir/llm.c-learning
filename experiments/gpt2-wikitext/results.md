# GPT-2 (124M) on WikiText-103

> Week 1 Day 4-5 · AutoDL RTX 4090D

## 训练结果

| 指标 | 值 |
|------|-----|
| 最终 Val Loss | **3.0458** |
| Perplexity | ~21.0 |
| 总步数 | 5000 |
| 总耗时 | ~4h 27min |
| MFU | ~27% |
| Checkpoint | 1.4 GB |

## Loss 曲线

```
iter   0: loss 11.01  (初始随机)
iter 100: loss  7.66
iter 200: loss  6.40  val 6.32
iter 300: loss  5.92
...
iter 5000:        val 3.0458
```

## 关键踩坑

1. **磁盘不足**：系统盘 30G → 数据迁移到 50G 数据盘 + 符号链接
2. **OpenWebText 不可用**：HF 被墙 + 镜像不支持 XetHub → 换 WikiText-103
3. **SSH 不稳**：本地直连频繁断开 → VS Code Remote-SSH 更稳定
4. **镜像可用性差异**：WikiText-103（旧存储格式）hf-mirror 可用，OpenWebText（XetHub）不可用
