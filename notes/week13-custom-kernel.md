# Week 13-16：动手改造 —— 自己写 CUDA Kernel

> 状态：🔴 未开始

---

## 目标

- 自己写一个 Fused Attention CUDA Kernel
- 做 PyTorch C++ Extension 集成
- Benchmark 对比 cuBLAS / Flash Attention

---

## 项目 1：Fused Attention Kernel

### 设计文档

```
输入：Q, K, V [B, nh, T, hs]
输出：output [B, nh, T, hs]

融合的操作：
1. Q @ K^T
2. Scale (1/sqrt(hs))
3. Causal Mask
4. Softmax
5. att @ V

全部在一个 kernel 里完成，避免多次 global memory 读写
```

### 代码

```cuda
// fused_attention.cu
// 位置：experiments/cuda-kernels/fused_attention.cu

__global__ void fused_attention_kernel(
    float* output,
    const float* Q,
    const float* K,
    const float* V,
    int B, int nh, int T, int hs) {
    
    // TODO: 实现
}
```

---

## 项目 2：PyTorch C++ Extension

### 代码

```cpp
// pytorch_extension/
// ├── setup.py
// ├── fused_attention.cpp      # PyTorch binding
// └── fused_attention_kernel.cu # CUDA kernel

#include <torch/extension.h>

torch::Tensor fused_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V) {
    
    // Launch CUDA kernel
    // ...
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_attention_forward, "Fused Attention Forward");
}
```

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_attention',
    ext_modules=[
        CUDAExtension('fused_attention', [
            'fused_attention.cpp',
            'fused_attention_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension},
)
```

---

## Benchmark 记录

### 测试环境

- GPU：____________
- CUDA：____________
- 测试 shape：B=?, nh=?, T=?, hs=?

### 结果

| 实现 | 耗时 (ms) | GFLOPS | 显存 |
|------|----------|--------|------|
| PyTorch SDPA | | | |
| cuBLAS | | | |
| 我的 Fused Kernel | | | |
| Flash Attention (参考) | | | |

---

## 踩坑记录

1. 
2. 
3. 
