# Week 5：逐算子的 C 实现精读

> 状态：🔴 未开始

---

## Day 1：matmul_forward —— C 中的矩阵乘法

### 数学公式

`C[i][j] = sum_{k=0}^{K-1} A[i][k] * B[k][j]`

### llm.c 实现（简化版）

```c
void matmul_forward(float* out, float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    // inp:  (B, T, C)    输入特征
    // weight: (OC, C)    权重矩阵（注意：已经是转置形式）
    // bias: (OC,)        偏置
    // out:  (B, T, OC)   输出
    
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++) {
                    val += inp[b * T * C + t * C + i] * weight[o * C + i];
                }
                out[b * T * OC + t * OC + o] = val;
            }
        }
    }
}
```

### 内存访问模式分析

- `weight[o * C + i]`：连续访问 → 对 cache 友好（行优先）
- `inp[b * T * C + t * C + i]`：连续访问 → 也友好
- 最内层循环 C ≈ 768 → 每次 matmul 约 72 亿次浮点运算

### 思考题

1. 为什么 C 代码比 PyTorch 的 `@` 慢 10-100 倍？
   - 答：
2. 如何优化？（预告：Tiling、SIMD、多线程）

---

## Day 2：attention_forward —— C 中的 Attention

### 步骤分解

```c
void attention_forward(float* out, float* preatt, float* att,
                       float* inp, int B, int T, int C, int nh) {
    int hs = C / nh;  // head size
    
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < nh; h++) {
            // 1. Q @ K^T → preatt
            for (int t = 0; t < T; t++)
                for (int t2 = 0; t2 < T; t2++) {
                    float val = 0.0f;
                    for (int k = 0; k < hs; k++)
                        val += Q[b,h,t,k] * K[b,h,t2,k];
                    preatt[...] = val / sqrtf((float)hs);
                }
            
            // 2. Softmax（causal：t2 <= t）
            for (int t = 0; t < T; t++) {
                // 2a. 找 max（数值稳定）
                float max_val = find_max(preatt[t, 0..t]);
                // 2b. exp + 求和
                float sum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++)
                    sum += (att[t,t2] = expf(preatt[t,t2] - max_val));
                // 2c. 归一化
                for (int t2 = 0; t2 <= t; t2++)
                    att[t,t2] /= sum;
            }
            
            // 3. att @ V → out
            for (int t = 0; t < T; t++)
                for (int k = 0; k < hs; k++) {
                    float val = 0.0f;
                    for (int t2 = 0; t2 <= t; t2++)
                        val += att[t,t2] * V[b,h,t2,k];
                    out[...] = val;
                }
        }
    }
}
```

### 与 nanoGPT 的对应

| nanoGPT | llm.c |
|---------|-------|
| `att = (Q @ K^T) * (1/sqrt(hs))` | 三重循环手动计算 |
| `att = att.masked_fill(..., -inf)` | 循环 `t2 <= t` 隐式实现 |
| `att = F.softmax(att, dim=-1)` | 手写 softmax（减 max + exp + 归一化） |
| `y = att @ V` | 三重循环手动计算 |

---

## Day 3：softmax 和 mask —— 数值细节

### 为什么 softmax 要减 max？

```c
// 不安全：
exp_val = expf(x[i]);  // x[i] = 100 → overflow!

// 安全（llm.c 采用）：
float max_val = find_max(x);
exp_val = expf(x[i] - max_val);  // x[i]-max ≤ 0 → ≤ 1 → 绝对安全
```

数学上等价：`softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))`

### causal mask 实现

- llm.c：softmax 循环时直接跳过 `t2 > t`，然后在 softmax 之后显式将上三角置 0
- 为什么需要显式置 0？→ backward 时这些值会被用到

---

## Day 4：layernorm_forward 和 gelu_forward

### LayerNorm

```c
void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    // 公式：y = (x - mean) / sqrt(var + eps) * weight + bias
    // mean 和 rstd 存储是为了 backward 复用
}
```

### GELU

```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

---

## Day 5：crossentropy_forward

### 数值稳定的 CE Loss

```c
// 数学：CE = -log(softmax(logits)[target])
// 数值稳定版：
// log(softmax(logits_t)) = (logits_t - max) - log(sum(exp(logits_j - max)))
```

**思考题**：
- 为什么 `(logit - max) - log(sum)` 比 `log(exp(logit-max)/sum)` 更好？
  - 答：
