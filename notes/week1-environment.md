# Week 1：环境搭建 + 首次训练

> 状态：🟡 进行中

## Day 1：环境搭建

### 硬件环境
- GPU：____________
- 显存：____________
- CUDA 版本：____________
- 驱动版本：____________

### 软件安装记录

```bash
# 1. CUDA Toolkit
nvcc --version
# 输出：

# 2. 克隆项目
git clone https://github.com/karpathy/nanoGPT.git
git clone https://github.com/karpathy/llm.c.git

# 3. 安装依赖
cd nanoGPT
pip install torch numpy transformers datasets tiktoken wandb

# 4. 编译 llm.c
cd ../llm.c
make train_gpt2        # 纯 CPU 版
# make train_gpt2cu    # CUDA 版（有 GPU 时）
```

### 遇到的问题

1. 
2. 
3. 

---

## Day 1-2：首次训练

### Shakespeare 数据集训练

```bash
cd nanoGPT
python data/shakespeare_char/prepare.py

python train.py config/train_shakespeare_char.py \
    --max_iters=100 \
    --batch_size=4 \
    --block_size=64 \
    --eval_interval=50
```

### 训练结果
- 最终 train loss：________
- 最终 val loss：________
- 100 步耗时：________

---

## Day 2：超参实验

| 实验 | 配置变更 | train loss | val loss | 观察 |
|------|---------|-----------|----------|------|
| Baseline | block_size=64 | | | |
| 实验1 | block_size=32 | | | |
| 实验2 | n_layer=2 | | | |
| 实验3 | lr=3e-3 | | | |

---

## Day 3：换数据集 + 生成文本

### OpenWebText 训练

```bash
python data/openwebtext/prepare.py
python train.py config/train_gpt2.py --max_iters=1000 --eval_interval=200
```

### 文本生成效果

| temperature | 生成效果 | 观察 |
|-------------|---------|------|
| 0.8 | | |
| 1.0 | | |
| 1.5 | | |

---

## Day 4-5：Wandb 可视化 + 完整训练

- Wandb 项目链接：____________
- 5000 步训练最终 loss：____________
- 截图保存位置：[diagrams/](../diagrams/)

---

## 阶段检查清单

- [ ] Shakespeare 数据集训练成功，loss 正常下降
- [ ] OpenWebText 数据集训练成功
- [ ] `sample.py` 能正常生成文本
- [ ] 调整过至少 3 个超参数，记录了对 loss 的影响
- [ ] Wandb 可视化的截图保存
