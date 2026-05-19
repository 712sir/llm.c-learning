# Week 1：环境搭建 + 首次训练

> 状态：🟡 进行中

## Day 1：环境搭建

### 硬件环境
- GPU：NVIDIA GeForce GTX 1650
- 显存：4 GB
- CUDA 版本：12.4（驱动支持），**CUDA Toolkit 未安装**
- 驱动版本：552.12

### 软件环境

| 工具 | 版本 | 路径/备注 |
|------|------|-----------|
| Python | 3.9.13 | `/d/ananconda3/python.exe`（conda） |
| PyTorch | 2.0.1+cpu | **CPU 版，需升级为 CUDA 版** |
| nvcc | ❌ 未安装 | 需安装 CUDA Toolkit |
| gcc | ❌ 未安装 | 需安装 MinGW-w64 或 VS Build Tools |
| make | ❌ 未安装 | 需安装 |
| wandb | 0.26.1 | 需 `wandb login` |

### 软件安装记录

```bash
# ===== 1. 网络环境说明 =====
# GitHub、PyPI 在国内直连超时，全程使用镜像/代理
# GitHub 镜像：kkgithub.com（仅读），ssh.github.com:443（推送）
# PyPI 镜像：pypi.tuna.tsinghua.edu.cn

# ===== 2. 克隆项目（走 kkgithub 镜像）=====
git clone --depth 1 https://kkgithub.com/karpathy/nanoGPT.git
git clone --depth 1 https://kkgithub.com/karpathy/llm.c.git
# 注意：kkgithub 是只读镜像，push 需走 SSH over 443

# ===== 3. 安装依赖（走清华 PyPI 镜像）=====
cd nanoGPT
/d/ananconda3/python.exe -m pip install torch numpy transformers \
    datasets tiktoken wandb \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

# 结果：PyTorch 安装的是 2.0.1+cpu（非 CUDA 版）
# 解决方案：后续需卸载重装 CUDA 版 PyTorch

# ===== 4. 编译 llm.c =====
# ❌ 当前无法编译——缺少 gcc 和 make
# 待办：安装 MinGW-w64 或 Visual Studio Build Tools

# ===== 5. 配置 Git 推送（SSH over 443）=====
# HTTPS 和 SSH 22 端口均被封
# 解决方案：走 ssh.github.com:443
git remote add origin ssh://git@ssh.github.com:443/712sir/llm.c-learning.git
git push -u origin master
# 成功推送首次 commit
```

---

## 踩坑全记录（13 个问题）

### 问题 1：GitHub HTTPS 克隆超时
- **现象**：`Failed to connect to github.com port 443: Timed out`
- **原因**：国内网络封锁 GitHub
- **解决**：使用镜像 `kkgithub.com` 替代 `github.com`
  ```bash
  git clone --depth 1 https://kkgithub.com/karpathy/nanoGPT.git
  ```

### 问题 2：Gitee 镜像需要认证
- **现象**：`could not read Username for 'https://gitee.com'`
- **原因**：Gitee 公开镜像也需要登录
- **解决**：放弃 Gitee，改用 kkgithub

### 问题 3：ghproxy / ghproxy.com / ghp.ci 均不可用
- **现象**：连接超时或 DNS 解析失败
- **原因**：这些第三方代理服务不稳定或已失效
- **解决**：放弃代理方式，改用 kkgithub 域名替换

### 问题 4：gitclone.com 返回 502
- **现象**：`The requested URL returned error: 502`
- **原因**：服务端故障
- **解决**：放弃

### 问题 5：Python 路径缺失
- **现象**：`No Python at 'C:\Users\21716\AppData\Local\Programs\Python\Python38\python.exe'`
- **原因**：Python 目录存在但 exe 被移除/卸载
- **解决**：使用 conda 自带的 Python `/d/ananconda3/python.exe`

### 问题 6：pip 安装卡住无响应
- **现象**：`pip install torch numpy ...` 运行数分钟无输出
- **原因**：PyPI 官方源国内不可达
- **解决**：换清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`

### 问题 7：PyTorch 安装为 CPU 版本
- **现象**：`torch.cuda.is_available() → False`
- **原因**：pip 默认安装 CPU-only 版本
- **解决（待办）**：
  ```bash
  pip uninstall torch -y
  pip install torch --index-url https://download.pytorch.org/whl/cu121 \
      -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

### 问题 8：CUDA Toolkit 未安装
- **现象**：`nvcc: command not found`
- **原因**：NVIDIA 驱动 552.12 只提供 CUDA 12.4 运行时，不含编译工具链
- **解决（待办）**：下载安装 [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)

### 问题 9：gcc 和 make 未安装
- **现象**：无法编译 llm.c 的纯 C 版本
- **原因**：Windows 环境没有 C 编译器
- **解决（待办）**：
  - 方案 A：安装 [MinGW-w64](https://www.mingw-w64.org/)
  - 方案 B：安装 Visual Studio Build Tools + 用 `cl.exe`

### 问题 10：Git Push 走 HTTPS 被重置
- **现象**：`Recv failure: Connection was reset`
- **原因**：HTTPS 443 端口到 github.com 被 GFW 阻断
- **解决**：改用 SSH over 443 端口走 `ssh.github.com`

### 问题 11：SSH 默认端口 22 不通
- **现象**：`ssh -T git@github.com` 超时
- **原因**：SSH 22 端口也被封锁
- **解决**：使用 GitHub 的 SSH-over-HTTPS 端点 `ssh.github.com:443`

### 问题 12：SSH Host Key 验证失败
- **现象**：`Host key verification failed`
- **原因**：`ssh.github.com` 不在 `~/.ssh/known_hosts` 中
- **解决**：
  ```bash
  ssh-keyscan -p 443 ssh.github.com >> ~/.ssh/known_hosts
  ```

### 问题 13：SSH Key 未添加到 GitHub
- **现象**：`git@ssh.github.com: Permission denied (publickey)`
- **原因**：本地 SSH 公钥未上传 GitHub
- **解决**：在 https://github.com/settings/keys 添加 `~/.ssh/id_rsa.pub`
- **注意**：添加后需在 GitHub 上创建仓库再 push

---

## 环境遗留待办

- [ ] 安装 CUDA Toolkit（让 nvcc 可用）
- [ ] 安装 gcc + make（编译 llm.c）
- [ ] PyTorch CPU 版替换为 CUDA 版
- [ ] `wandb login` 登录 Weights & Biases
- [ ] 确认 `python data/shakespeare_char/prepare.py` 可正常运行 

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
