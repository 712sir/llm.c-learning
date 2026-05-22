# Week 1：环境搭建 + 首次训练

> 状态：🟡 进行中（Day 1-2 完成，Day 3 待开始）

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
    --eval_interval=50 \
    --compile=False
```

### 训练结果
- 最终 train loss：2.7108
- 最终 val loss：2.7131
- 100 步耗时：~14s（每步约 20ms，eval 步约 4s × 3 次）
- 模型参数量：10.65M
- 每步 tokens：256

### 踩坑记录

**问题：torch.compile 需要 Triton，Windows 不支持**
- **现象**：`RuntimeError: Cannot find a working triton installation`
- **原因**：nanoGPT train.py 默认 `compile=True`，torch.compile 在 GPU 上调用 inductor 后端需要 Triton，但 Triton 官方仅支持 Linux
- **解决**：添加 `--compile=False` 禁用编译，GTX 1650 仅 4GB 显存，后续大模型也无暇使用 compile 加速，影响可忽略

---

## Day 1：训练指令详解

### 命令逐条拆解

```bash
python train.py config/train_shakespeare_char.py \
    --max_iters=100 \
    --batch_size=4 \
    --block_size=64 \
    --eval_interval=50 \
    --compile=False
```

| 参数 | 值 | 含义 |
|------|-----|------|
| `config/train_shakespeare_char.py` | 配置文件 | 基础配置：6层 transformer，6头注意力，384维嵌入，原始设定 batch_size=64、block_size=256、max_iters=5000，本次全部用命令行覆盖 |
| `--max_iters=100` | 100 步 | 只跑 100 步（完整训练是 5000），仅验证流程能走通 |
| `--batch_size=4` | 4 | 每次拿 4 条样本并行。原配置是 64，降到 4 是因为 GTX 1650 只有 4GB 显存 |
| `--block_size=64` | 64 | 每条样本的上下文窗口 = 64 个字符。原配置 256，同样为省显存 |
| `--eval_interval=50` | 50 | 每 50 步做一次验证集评估，本次在 step 0/50/100 各评估一次 |
| `--compile=False` | 关闭 | Windows 无 Triton，禁用 torch.compile |

### 每步发生了什么

每一步训练的数据量 = `batch_size × block_size = 4 × 64 = 256 tokens`。模型用这 256 个 token 做一次**前向传播**（计算 logits → 算 loss）→ **反向传播**（算梯度）→ **参数更新**（AdamW 优化器更新 10.65M 参数）。

---

## Day 1：结果评估

### Loss 曲线

```
step 0:   train 4.3241 → val 4.3153    初始随机权重，loss ≈ ln(65) ≈ 4.17
step 10:  loss 3.3288                   快速下降
step 20:  loss 3.0446
step 50:  train 2.7608 → val 2.7698
step 100: train 2.7108 → val 2.7131
```

### 解读

**1. 初始 loss 正常。** 词表大小 65（65 种字符），随机猜测的 loss 理论上限是 `ln(65) ≈ 4.17`。step 0 的 4.32 接近这个值，说明模型初始化正确，没有数值异常。

**2. Loss 在持续下降。** 从 4.29 → 2.71，下降了约 38%，且没有反弹或震荡。说明学习率设置合理，梯度正常回传，没有梯度爆炸/消失。

**3. 没有过拟合。** train loss (2.7108) 和 val loss (2.7131) 几乎一致。如果 train loss 远低于 val loss（比如 0.5 vs 3.0），说明模型在"背"训练集而非学习规律。目前两者非常接近，是健康信号。

**4. 下降速度在放缓。** 前 20 步下降很快（4.29 → 3.04），后 80 步变慢（3.04 → 2.71）。这是正常现象——模型先学到高频模式（空格、常见字母 e/t/a/o），后学的规律越来越"贵"（罕见组合、长程依赖）。

**5. 10.65M 参数跑 100 步远未收敛。** 这个模型设计是跑 5000 步的，100 步只是验证管线，loss 还有大量下降空间。

### 结论

> 环境搭建正确，模型能正常收敛。loss 下降曲线健康，train/val 一致无过拟合。可以进入 Day 2 的超参实验。

---

## Day 2：超参实验

### 实验流程说明

naoGPT 通过 `configurator.py` 支持命令行覆盖配置文件中的任意参数，格式为 `--key=value`。这意味着不需要修改配置文件，一条命令就能完成一次实验。

**实验设计**：控制变量法。每次只改一个参数，其他保持不变，这样才能确定 loss 变化是由哪个参数引起的。

**实验模板**（以 baseline 为例）：

```bash
cd Project-nanoGPT
python train.py config/train_shakespeare_char.py \
    --max_iters=100 \        # 只跑 100 步快速验证
    --batch_size=4 \         # GTX 1650 4GB 显存限制
    --block_size=64 \        # 上下文窗口
    --eval_interval=50 \     # 每 50 步评估一次
    --compile=False          # Windows 无 Triton
```

**改一个参数的实验**（以 block_size 为例）：

```bash
# 只改 --block_size=32，其余参数与 baseline 完全一致
python train.py config/train_shakespeare_char.py \
    --max_iters=100 --batch_size=4 --block_size=32 \
    --eval_interval=50 --compile=False
```

**每次实验关注哪些输出**：

| 指标 | 位置 | 含义 |
|------|------|------|
| train loss (step 100) | 输出倒数第二行 `step 100: train loss X` | 模型在训练集上的最终表现 |
| val loss (step 100) | 同上 `val loss X` | 模型在验证集上的表现，判断过拟合 |
| 收敛速度 | 观察 iter 0→100 的 loss 下降趋势 | 判断学习率是否合适 |
| 震荡程度 | 相邻 iter 之间 loss 的波动幅度 | 判断学习率是否过大 |
| 每步耗时 | `time XXms` | 判断模型规模对速度的影响 |
| 参数量 | `number of parameters: X` | 确认模型结构变更生效 |

**关键经验**：

1. **一次只改一个变量**。如果同时改 block_size 和 lr，loss 变了你分不清是谁造成的。
2. **固定随机种子不关键**（100 步太短影响不大），但必须保证 baseline 和实验用的是**同一个数据集**（Shakespeare）。
3. **eval_interval 设密一点**（50 步一次）。100 步一共才 3 个评估点，已经够看了。
4. **GTX 1650 4GB 显存是硬上限**。batch_size 最高到 4-6，再大就 OOM。

| 实验 | 配置变更 | train loss | val loss | 每步耗时 | 参数量 | 观察 |
|------|---------|-----------|----------|---------|--------|------|
| Baseline | block_size=64, n_layer=6, lr=1e-3 | 2.7108 | 2.7131 | ~17ms | 10.65M | loss 平稳下降，train/val 一致 |
| 实验1 | block_size=32 | 2.8735 | 2.8799 | ~16ms | 10.65M | loss 变差 0.16，震荡更大，上下文短导致预测更难 |
| 实验2 | n_layer=2 | 2.7240 | 2.7220 | ~9ms | 3.57M | loss 略差 0.01，但速度快 2x，参数少 67% |
| 实验3 | lr=3e-3 | 2.8119 | 2.8157 | ~17ms | 10.65M | loss 变差 0.10，震荡明显（2.67~2.93 跳动），学习率过大 |

### 解读

**实验1 (block_size=32)**：上下文窗口减半后，模型每次只看到 32 个字符，比 baseline 的 64 少了一半。Loss 从 2.71 升到 2.88，说明上下文信息对字符级语言模型很重要——需要足够的历史字符来预测下一个字符。另注意到 loss 波动更大（iter 70 的 2.83 vs iter 80 的 3.10），小上下文导致每个 batch 的梯度更不稳定。

**实验2 (n_layer=2)**：从 6 层减到 2 层，参数量从 10.65M 降到 3.57M。Loss 只差了 0.01，几乎和 baseline 持平，但训练速度快了一倍（9ms vs 17ms）。这说明在 Shakespeare 这种小数据集上，深层的模型容量是过剩的——浅层模型也能学到差不多的表示。

**实验3 (lr=3e-3)**：学习率提高 3 倍后，loss 震荡加剧（在 2.67 和 2.93 之间反复跳动），最终 loss 也比 baseline 差。说明 1e-3 的学习率对当前模型规模来说已经接近上限，再大会导致梯度更新过冲，模型难以收敛到好的局部最优。

### 结论

> Baseline (block_size=64, n_layer=6, lr=1e-3) 在当前配置下是最优的。降低 block_size 和升高 lr 都会损害收敛质量。降低 n_layer 用 67% 的参数换来了几乎相同的 loss，如果追求推理速度是一个好的取舍方向。

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

- [x] Shakespeare 数据集训练成功，loss 正常下降
- [x] 完成 3 组超参实验（block_size / n_layer / lr），记录分析
- [ ] OpenWebText 数据集训练成功
- [ ] `sample.py` 能正常生成文本
- [ ] 调整过至少 3 个超参数，记录了对 loss 的影响
- [ ] Wandb 可视化的截图保存
