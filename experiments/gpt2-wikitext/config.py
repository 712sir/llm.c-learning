# GPT-2 (124M) 训练配置
# 对应课程：Week 1 Day 4-5
# 平台：AutoDL RTX 4090D 24GB

wandb_log = False
wandb_project = 'wikitext'
wandb_run_name = 'gpt2-124M-wikitext'

# 有效批量 = 8 × 1024 × 40 = 327,680 token/step
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 40

# 训练步数：~1.6B token, WikiText-103 约 13 epoch
max_iters = 5000
lr_decay_iters = 5000

# 评估
eval_interval = 200
eval_iters = 50
log_interval = 10

# 正则化
weight_decay = 1e-1

# 学习率
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 2000
decay_lr = True

# 模型（在 train.py 中默认值）
# n_layer = 12
# n_head = 12
# n_embd = 768
# vocab_size = 50304
# bias = False
# dropout = 0.0

dataset = 'wikitext'
compile = True
dtype = 'bfloat16'
