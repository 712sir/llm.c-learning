#!/bin/bash
# 超参搜索实验脚本
# 运行方式：bash run_configs.sh

cd ../../Project-nanoGPT

echo "===== Baseline: block_size=64 ====="
python train.py config/train_shakespeare_char.py \
    --max_iters=200 --eval_interval=100

echo "===== Experiment 1: block_size=32 ====="
python train.py config/train_shakespeare_char.py \
    --max_iters=200 --block_size=32 --eval_interval=100

echo "===== Experiment 2: n_layer=2 ====="
python train.py config/train_shakespeare_char.py \
    --max_iters=200 --n_layer=2 --eval_interval=100

echo "===== Experiment 3: lr=3e-3 ====="
python train.py config/train_shakespeare_char.py \
    --max_iters=200 --learning_rate=3e-3 --eval_interval=100

echo "All experiments done!"
