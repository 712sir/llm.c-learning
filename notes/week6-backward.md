# Week 6：反向传播精读 —— 最硬核的一周

> 状态：🔴 未开始

---

## Day 1：crossentropy_backward

### 核心公式

```
dloss/dlogits_i = (softmax(logits_i) - one_hot(target)_i) / (B * T)
```

### 推导过程

令 `p_i = softmax(logits_i) = exp(logits_i) / sum(exp(logits_j))`

- 对于 `i == target`：`d(log(p_i))/d(logits_i) = 1 - p_i`
- 对于 `i != target`：`d(log(p_target))/d(logits_i) = -p_i`
- 合并：`d(log(p_target))/d(logits_i) = (p_i - 1_{i==target})`
- 所以：`dloss/dlogits = (probs - one_hot_target) / N`

这是整个 AI Infra 领域最核心的公式之一。

---

## Day 2：matmul_backward

### 数学推导

Forward: `C = A @ B`，其中 A(M,K), B(K,N), C(M,N)

Backward（给定 dC，求 dA 和 dB）：

```
dA = dC @ B^T     → shape (M, K) ← (M, N) @ (N, K)
dB = A^T @ dC     → shape (K, N) ← (K, M) @ (M, N)
```

### llm.c 实现

```c
void matmul_backward(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    
    // 1. dinp = dout @ weight（形状变换，注意维度）
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++) {
                float val = 0.0f;
                for (int oc = 0; oc < OC; oc++)
                    val += dout[b,t,oc] * weight[oc, c];
                dinp[b,t,c] = val;
            }
    
    // 2. dweight = sum_{b,t} dout[b,t,oc] * inp[b,t,c]
    for (int oc = 0; oc < OC; oc++)
        for (int c = 0; c < C; c++) {
            float val = 0.0f;
            for (int b = 0; b < B; b++)
                for (int t = 0; t < T; t++)
                    val += dout[b,t,oc] * inp[b,t,c];
            dweight[oc, c] = val;
        }
    
    // 3. dbias = sum_{b,t} dout[b,t,oc]
    if (dbias != NULL)
        for (int oc = 0; oc < OC; oc++) {
            float val = 0.0f;
            for (int b = 0; b < B; b++)
                for (int t = 0; t < T; t++)
                    val += dout[b,t,oc];
            dbias[oc] += val;  // 累加！
        }
}
```

### 思考题

1. 为什么 dweight 需要 B × T 两层循环？计算量是多少？
2. dweight 是否需要除以 B×T？
3. 三种梯度的用途：
   - dinp：继续往后传的梯度
   - dweight：用于更新参数
   - dbias：用于更新参数

---

## Day 3-4：attention_backward

### 正向回顾

```
Forward:  Q ─┐
              ├─→ scores ─→ weights ─→ output
            K ─┘                │
            V ──────────────────┘
```

### 反向推导

```
Backward: dQ ←─ dscores ←─ dweights ←─┬── dout (given)
            dK ←──┘          │          │
            dV ←─────────────┘──────────┘
```

### 五个步骤

```python
# 给定 dout [B, nh, T, hs]

# 1. dV = weights^T @ dout
dV[b,h,t2,k] = sum_t(weights[b,h,t,t2] * dout[b,h,t,k])

# 2. dweights = dout @ V^T
dweights[b,h,t,t2] = sum_k(dout[b,h,t,k] * V[b,h,t2,k])

# 3. dscores = softmax_backward(dweights, weights)
sum_wd = sum_j(weights[j] * dweights[j])
dscores[t] = weights[t] * (dweights[t] - sum_wd)

# 4. dQ = dscores @ K / sqrt(hs)
dQ[b,h,t1,k] = sum_t2(dscores[b,h,t1,t2] * K[b,h,t2,k]) * scale

# 5. dK = dscores^T @ Q / sqrt(hs)
dK[b,h,t2,k] = sum_t1(dscores[b,h,t1,t2] * Q[b,h,t1,k]) * scale
```

### PyTorch 验证代码

```python
# 用 PyTorch autograd 验证手写 backward
Q = torch.randn(2, 4, 8, 16, requires_grad=True)
K = torch.randn(2, 4, 8, 16, requires_grad=True)
V = torch.randn(2, 4, 8, 16, requires_grad=True)

# PyTorch 版本
out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
out.sum().backward()
# 对比 Q.grad, K.grad, V.grad 与手写版本
```

---

## Day 5：layernorm_backward + gelu_backward

### LayerNorm 反向推导

```
forward: y = (x - mean) * rstd * gamma + beta
其中 xhat = (x - mean) * rstd
      rstd = 1 / sqrt(var + eps)

核心公式（面试常考）：
dl/dx = rstd * (C * dy*gamma - sum(dy*gamma) - xhat * sum(dy*gamma * xhat)) / C
```

### GELU 反向

```
GELU'(x) ≈ 0.5 * (1 + tanh(...)) + 0.5 * x * (1 - tanh^2(...)) * deriv
其中 deriv = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
```
