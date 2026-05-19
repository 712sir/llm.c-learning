#!/bin/bash
# 性能 benchmark 脚本
# 对比各算子的耗时分布

echo "=== Performance Benchmark ==="
echo "Hardware: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU only')"
echo ""

# 编译并运行 llm.c 纯 C 版
cd ../../Project-llm.c

echo "=== Building train_gpt2 ==="
make train_gpt2

echo ""
echo "=== Running benchmark ==="
# 运行并收集输出
./train_gpt2 2>&1 | tee ../llm.c-learning/experiments/perf-benchmark/results/raw_output.txt

echo ""
echo "Results saved to experiments/perf-benchmark/results/"
