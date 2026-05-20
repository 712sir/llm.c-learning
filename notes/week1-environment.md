# Week 1：环境搭建 + 首次训练

> 状态：🟡 进行中

## Day 1：环境搭建

### 硬件环境
- GPU：NVIDIA GeForce GTX 1650
- 显存：4 GB
- CUDA 版本：12.4（驱动支持），CUDA Toolkit 暂缓至 Week 8
- 驱动版本：552.12

### 软件环境

| 工具 | 版本 | 路径/备注 |
|------|------|-----------|
| Python | 3.9.13 | `/d/ananconda3/python.exe`（conda） |
| PyTorch | 2.5.1+cu121 | CUDA 可用，GPU：GTX 1650 |
| nvcc | ⏸️ 暂缓 | Week 8 安装 CUDA Toolkit 12.x |
| gcc | 16.1.0 | `C:\mingw64\bin\gcc.exe`（MinGW-W64） |
| make | 4.4.1 | `C:\mingw64\bin\mingw32-make.exe` |
| wandb | 0.26.1 | 已登录 |

### 软件安装记录

```bash
# ===== 1. 网络环境说明 =====
# GitHub、PyPI 在国内直连超时，全程使用镜像
# GitHub 拉取：kkgithub.com（只读镜像）
# GitHub 推送：ssh.github.com:443（SSH over HTTPS）
# PyPI 镜像：pypi.tuna.tsinghua.edu.cn
# PyTorch CUDA wheel：mirrors.aliyun.com/pytorch-wheels/cu121（国内唯一可用）

# ===== 2. 克隆项目 =====
git clone --depth 1 https://kkgithub.com/karpathy/nanoGPT.git
git clone --depth 1 https://kkgithub.com/karpathy/llm.c.git

# ===== 3. 安装 Python 依赖 =====
pip install torch numpy transformers datasets tiktoken wandb \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
# ⚠️ 清华源安装的是 CPU 版 PyTorch，需后续替换

# ===== 4. 替换 PyTorch CUDA 版 =====
# 国内 PyPI 镜像（清华/阿里）都没有 CUDA 版 PyTorch
# 阿里云有专用 PyTorch wheel 镜像，必须用 -f 参数（非 -i）
pip uninstall torch -y
pip install torch==2.5.1 -f https://mirrors.aliyun.com/pytorch-wheels/cu121
# 版本选择：torch 2.5.1 是最后一个支持 Python 3.9 + CUDA 的版本

# ===== 5. 安装 gcc + make =====
winget install --id BrechtSanders.WinLibs.POSIX.UCRT --location "C:\mingw64"
# winget 只下载了 ZIP，需手动解压：
unzip -o winlibs-*.zip -d /c/mingw64/
mv /c/mingw64/mingw64/* /c/mingw64/ && rmdir /c/mingw64/mingw64
echo 'export PATH="/c/mingw64/bin:$PATH"' >> ~/.bashrc
# ⚠️ Windows 下 make 命令是 mingw32-make

# ===== 6. wandb 登录 =====
wandb login <api-key>

# ===== 7. Git 推送配置 =====
# HTTPS 和 SSH 22 端口均被封，走 ssh.github.com:443
git remote add origin ssh://git@ssh.github.com:443/712sir/llm.c-learning.git
ssh-keyscan -p 443 ssh.github.com >> ~/.ssh/known_hosts
git push -u origin master

# ===== 8. 验证环境 =====
python -c "import torch; print(torch.cuda.is_available())"  # True
gcc --version    # 16.1.0
mingw32-make -v  # 4.4.1
```

---

## 踩坑记录

### 问题 1：GitHub HTTPS 克隆超时
- **现象**：`Failed to connect to github.com port 443: Timed out`
- **原因**：国内网络封锁 GitHub
- **解决**：使用镜像 `kkgithub.com` 替代 `github.com`

### 问题 2：镜像站大面积失效
- **尝试过**：Gitee（需登录）、ghproxy.com（超时）、ghp.ci（DNS 失败）、gitclone.com（502）
- **唯一可用**：`kkgithub.com`（域名替换），但只读不写

### 问题 3：Python 路径缺失
- **现象**：`No Python at ...\Python38\python.exe`
- **原因**：Python 目录存在但 exe 被卸载
- **解决**：换用 conda 自带的 Python `/d/ananconda3/python.exe`

### 问题 4：pip 安装卡住无响应
- **现象**：`pip install` 运行数分钟无输出
- **原因**：PyPI 官方源不可达
- **解决**：换清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`

### 问题 5：PyTorch 始终安装 CPU 版
- **现象**：`torch.cuda.is_available() → False`，反复重装无效
- **原因**：国内 PyPI 镜像（清华/阿里）只有 CPU 版，不含 `+cu121` 后缀的 CUDA 版
- **解决**：用阿里云 PyTorch wheel 专用镜像 + `-f` 参数（非 `-i`）
  ```bash
  pip install torch==2.5.1 -f https://mirrors.aliyun.com/pytorch-wheels/cu121
  ```
- **额外坑**：torch 2.6+ 不再支持 Python 3.9，需锁定 2.5.1

### 问题 6：Git Push 完全不通
- **现象**：HTTPS 被 reset，SSH 22 端口超时
- **解决**：SSH over 443 → `ssh.github.com:443`
- **附带问题**：Host key 验证失败 → `ssh-keyscan` 添加；公钥未上传 → GitHub Settings 添加 SSH Key

### 问题 7：conda 安装工具链全部超时
- **现象**：`conda install` 一直 `Solving environment` 后失败
- **原因**：conda 残留 `defaults` 源指向 `repo.anaconda.com`（被封）
- **解决**：`conda config --remove channels defaults`，全部切清华源
- **后记**：gcc/make 最终还是走 winget 装的，conda 的 m2w64 太慢

### 问题 8：winget 装了 MinGW 但不生效
- **现象**：`winget install WinLibs` 提示成功，但 `gcc` 找不到
- **原因**：winget 只下载了 ZIP 到 Temp 目录，没有解压安装
- **解决**：手动 unzip 到 `C:\mingw64`，注意 ZIP 内有两层 `mingw64/` 目录需 flatten

---

## Day 1：数据准备 + 首次训练

### Shakespeare 数据集

```bash
cd Project-nanoGPT
python data/shakespeare_char/prepare.py
```

结果：
- 总字符数：1,115,394
- 词表大小：65（字符级）
- 训练集：1,003,854 tokens
- 验证集：111,540 tokens

### 首次训练

```bash
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

```bash
python data/openwebtext/prepare.py
python train.py config/train_gpt2.py --max_iters=1000 --eval_interval=200
```

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
