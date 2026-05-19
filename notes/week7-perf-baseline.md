# Week 7：纯 C 版性能测试 + 第二阶段总结

> 状态：🔴 未开始

---

## Day 1：插入计时代码

### 训练主循环计时

```c
#include <time.h>

clock_t start, end;
double forward_time = 0, backward_time = 0, update_time = 0;

for (int step = 0; step < num_steps; step++) {
    start = clock();
    gpt2_forward(&model, inputs, targets, B, T);
    end = clock();
    forward_time += (double)(end - start) / CLOCKS_PER_SEC;
    
    start = clock();
    gpt2_backward(&model);
    end = clock();
    backward_time += (double)(end - start) / CLOCKS_PER_SEC;
    
    start = clock();
    adamw_update(&model, ...);
    end = clock();
    update_time += (double)(end - start) / CLOCKS_PER_SEC;
}
```

### 实测数据

| 阶段 | 耗时 (s) | 占比 |
|------|---------|------|
| Forward | | % |
| Backward | | % |
| Update | | % |

---

## Day 2：各算子耗时细分

### Forward 细分

| 算子 | 耗时 (s) | 占比 |
|------|---------|------|
| QKV matmul | | |
| Attention | | |
| Out proj | | |
| LayerNorm | | |
| MLP fc | | |
| GELU | | |
| MLP proj | | |

### 最大瓶颈

→ matmul 占 50%+ 的时间（符合预期）

---

## Day 3-5：性能对比 + 笔记整理

### 纯 C vs PyTorch 吞吐对比

| 实现 | tokens/sec | 相对速度 |
|------|-----------|---------|
| PyTorch (GPU) | | 1x |
| PyTorch (CPU) | | |
| llm.c 纯 C | | |

### 关键发现

1. 
2. 
3. 

---

## 第二阶段检查清单

- [ ] 调用链地图完成
- [ ] 所有算子的 forward 精读完成
- [ ] 所有算子的 backward 精读完成
- [ ] 手写 attention_backward 通过 autograd 验证
- [ ] 性能 baseline 数据采集完成
