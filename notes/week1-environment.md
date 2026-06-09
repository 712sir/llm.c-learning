# Week 1：环境搭建 + 首次训练

> 状态：🟢 全部完成

## 实验与 Week 对应关系

> 本阶段实验按时间线分布在 Week 1 的不同 Day，产物存放在 `experiments/` 目录。

| 周次 | 实验 | 平台 | 产物位置 |
|------|------|------|----------|
| **Week 1 Day 1** | 环境搭建 + Shakespeare 100 步训练 | 本地 GTX 1650 | —（验证流程） |
| **Week 1 Day 2** | 超参实验（block_size / n_layer / lr） | 本地 GTX 1650 | [experiments/hyperparam-search/](../experiments/hyperparam-search/) |
| **Week 1 Day 3** | Temperature 生成对比 + OpenWebText 下载踩坑 | 本地 GTX 1650 | —（定性实验） |
| **Week 1 Day 4-5** | GPT-2 (124M) 完整训练 | AutoDL RTX 4090D | [experiments/gpt2-wikitext/](../experiments/gpt2-wikitext/) |

### Week 1 Day 1 · 内容速览

- 本地环境搭建（Python/PyTorch/gcc/make/wandb）
- GPU 验证 + 网络镜像配置
- Shakespeare 字符级数据集分词
- 100 步训练验证（loss 从 4.32 → 2.71）

### Week 1 Day 2 · 内容速览

- 3 组控制变量超参实验（block_size=32 / n_layer=2 / lr=3e-3）
- 结论：baseline（block_size=64, n_layer=6, lr=1e-3）最优
- 产物：[experiments/hyperparam-search/](../experiments/hyperparam-search/)（results.csv + run_configs.sh）

### Week 1 Day 3 · 内容速览

- 1000 步续训（loss 从 2.71 → 2.50）
- Temperature 生成对比（0.8 / 1.0 / 1.5），理解 softmax + temperature 原理
- OpenWebText 下载 6 种方案全部失败，记录根本原因

### Week 1 Day 4-5 · 内容速览

- AutoDL 云端环境搭建（RTX 4090D 24GB）
- 磁盘管理（数据迁移到 50GB 数据盘）
- WikiText-103 替代 OpenWebText（HF 镜像可用性差异）
- GPT-2 124M 完整训练：5000 步，val loss 3.05，4.5h
- 产物：[experiments/gpt2-wikitext/](../experiments/gpt2-wikitext/)（config.py + results.md）

---

## 基础概念补充：3Blue1Brown ML 全家桶

> 通过 B 站 3Blue1Brown 频道系统学习，补齐 ML 零基础短板。视频直观展示了神经网络内部运作机制，比文字效率高 10 倍。

| 知识点 | 核心理解 |
|--------|---------|
| **神经网络** | 输入 → 隐藏层（权重矩阵 + 激活函数）→ 输出；本质是逼近任意函数的万有拟合器 |
| **梯度下降** | 计算损失对每个参数的偏导数，沿负梯度方向调整；学习率控制步长 |
| **反向传播** | 链式法则从输出往回逐层算梯度；PyTorch 的 `loss.backward()` 在 C 里要手写 |
| **CNN** | 卷积核滑动扫描图像，局部感受野 + 参数共享 |
| **Transformer** | 抛弃 RNN 的序列依赖，全靠 Attention 让每个位置直接看所有其他位置 |
| **Softmax** | 把任意实数向量压成概率分布（和=1），温度控制分布尖锐/平坦 |
| **贝叶斯** | 先验概率 + 新证据 → 后验概率；不确定性思维框架 |

### 关键收获

- **看懂模型结构图不再全是黑盒**：知道每个矩形是"权重矩阵"，箭头是"矩阵乘法 + 激活函数"
- **理解训练循环的本质**：前向（make prediction）→ 算 loss（measure error）→ 反向（compute gradients）→ 更新参数（gradient descent）
- **`B, T, C` 不再抽象**：就是数组三个维度，每一层都在对这个三维数组做变换
- **"GPT 只看左边"（causal）** 和 Attention 的视觉直觉建立

---

## Day 1：环境搭建 —— 本地 GTX 1650

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

## Day 3：OpenWebText 下载失败 + Temperature 生成对比

### OpenWebText 下载踩坑（完整追踪）

**环境**：Python 3.9.13, datasets 2.x, huggingface_hub 0.36.2, VPN: 星云（代理端口 7897, TUN 模式对 Git Bash 无效）

**数据集规模**：80 个 parquet 文件 × 288MB = 23GB，tokenize 后 train.bin ≈ 17GB

| 方案 | 命令/配置 | 结果 | 原因 |
|------|----------|------|------|
| ① 直连 | `python prepare.py` | `ConnectTimeout` | huggingface.co 被墙 |
| ② HF 镜像 | `HF_ENDPOINT=https://hf-mirror.com` | `FileNotFoundError` | 镜像只代理 API，文件实际在 S3 (`cas-bridge.xethub.hf.co`)，镜像未代理 |
| ③ 镜像 + 代理 | `HF_ENDPOINT=... HTTP_PROXY=...` | SSL EOF error | 代理对 S3 的 HTTPS CONNECT 隧道不稳定 |
| ④ 直连 + 代理（TUN 模式） | 星云 TUN 模式 | `ConnectTimeout` | Git Bash/MSYS2 终端不走 TUN 虚拟网卡 |
| ⑤ 直连 + 代理（HTTP_PROXY） | `HTTP_PROXY=127.0.0.1:7897` | API 返回 200，文件下载 7MB 后断连 | 代理连接不稳定，大文件传输频繁中断 |
| ⑥ curl + 代理 | `curl -x http://127.0.0.1:7897` | 127MB 后超时 | 代理速度 ~1MB/s，单文件 288MB 需 5 分钟，中途常断 |

**根本原因**：

1. HuggingFace 的文件存储从自建 CDN 迁到了 AWS S3 (`cas-bridge.xethub.hf.co`)，国内直连不可达
2. `hf-mirror.com` 只代理了 Hub API / 元数据，不代理底层 S3 文件传输
3. 星云 VPN 是 IEPL 中转方案，带宽 3000Mbps 但连接稳定性不足以支撑 23GB 连续下载
4. 每月流量配额 100GB，即使下载成功也会消耗 23%

**结论**：国内环境下载 OpenWebText 不可行。改用 Shakespeare 数据集替代，学习目标等价。

### 从 100 步继续训练到 1000 步

```bash
python train.py config/train_shakespeare_char.py \
    --init_from=resume \
    --out_dir=out-shakespeare-char \
    --max_iters=1000 --batch_size=4 --block_size=64 \
    --eval_interval=50 --compile=False
```

| 指标 | 100 步（起点） | 1000 步（终点） |
|------|---------------|-----------------|
| train loss | 2.7108 | 2.4983 |
| val loss | 2.7131 | 2.5081 |
| 每步耗时 | ~17ms | ~17ms |

Loss 从 2.71 → 2.50，平稳下降。900 步额外训练约 15 秒。

### Temperature 生成对比（1000 步模型）

```bash
python sample.py --out_dir=out-shakespeare-char --temperature=X \
    --max_new_tokens=300 --num_samples=2 --compile=False
```

| temperature | 生成效果（截选） | 观察 |
|-------------|-----------------|------|
| **0.8** | `And thif brid owind t s s, be mad...` / `HOLINY:` / `PIF d las ate ar ce...` | 较保守，出现重复模式（`llll`），角色名格式稳定 |
| **1.0** | `And thik brid owinen O la, bth...` / `KENOBY...` / `SWanous l lind...` | 多样性增加，出现更多新"角色名"，偶尔跳出空格 |
| **1.5** | `And; bef bridcowi,fakis n, bte...` / `GUETHerabousel...` / `ENaW-CIAn...` | 最随机，大小写混乱，标点异常，几乎不可读 |

#### 解读

**1. Temperature 原理**：模型输出的是 logits（每个 token 的原始分数），经过 softmax 变成概率分布。Temperature 在 softmax 之前对 logits 做除法：

```
p(token_i) = softmax(logits / T)
```

- **T < 1**（如 0.8）：logits 被放大，高概率 token 更突出，低概率 token 被压制 → 输出更确定、更保守
- **T = 1**：原始概率分布，不做任何干预
- **T > 1**（如 1.5）：logits 被缩小，概率分布趋于均匀，低概率 token 获得更多机会 → 输出更多样、更随机

**2. 为什么 100 步时看不出差异？** 当模型未经充分训练时，所有 token 的 logits 几乎相等（≈ 1/65 均匀分布）。此时无论 T=0.8 还是 T=1.5，概率分布没有本质区别，都是随机采样。**Temperature 只有在模型已经学到有意义的概率分布时才有调节作用。**

**3. 1000 步模型的问题**：虽然能产生类单词片段和莎士比亚格式（角色名 + 冒号），但整体仍不通顺。字符级模型的特点——学会了局部拼写规律（如 `th`、`ou`、`ing`），但没有学会有意义的词语和语法。需要继续训练。

**4. Temperature 选择的工程权衡**：
- T=0.7~0.9：适合需要稳定输出的场景（如代码生成、翻译）
- T=1.0：默认值，平衡多样性和质量
- T=1.2~1.5：适合需要创意的场景（如故事创作、头脑风暴）
- T>2.0：通常不可用，退化为随机采样 |

---

## Day 4-5：AutoDL GPT-2 (124M) 完整训练

> Week 1 Day 4-5，AutoDL C 线学习任务

### 环境

| 项目 | 详情 |
|------|------|
| **平台** | AutoDL (SeetaCloud) |
| **GPU** | RTX 4090 D，24 GB 显存 |
| **系统盘** | 30 GB（overlay） |
| **数据盘** | 50 GB（/root/autodl-tmp） |
| **Python** | 3.8.10 (miniconda3) |
| **PyTorch** | 2.0.1+cu118 |
| **代码** | nanoGPT (karpathy/nanoGPT) |
| **SSH** | `ssh -p 11802 root@connect.westc.seetacloud.com` |

### 数据集：WikiText-103（替代 OpenWebText）

OpenWebText 在 AutoDL 上同样无法下载（与 Day 3 踩坑原因一致：HF 被墙 + 镜像不代理 S3/XetHub），改用 **WikiText-103**：

| 文件 | 大小 | Token 数 |
|------|------|----------|
| `train.bin` | 228 MB | 119,721,490（~1.2 亿） |
| `validation.bin` | 490 KB | 251,049 |

预处理命令：
```bash
cd /root/autodl-tmp/nanoGPT_data/wikitext
HF_ENDPOINT=https://hf-mirror.com python prepare.py
# 注：WikiText-103 使用旧存储格式，hf-mirror.com 可用；OpenWebText 用 XetHub 则不行
```

### 模型与训练配置

**GPT-2 Small (124M)**：12 层 / 12 头 / 768 维 / block_size=1024

```python
# config/train_gpt2_wikitext.py
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 40   # 有效批量 = 8×1024×40 = 327,680 token/step
max_iters = 5000                   # 约 1.6B token，~13 epoch
learning_rate = 6e-4
min_lr = 6e-5                      # 1/10
warmup_iters = 2000
lr_decay_iters = 5000              # Cosine 衰减
weight_decay = 0.1
grad_clip = 1.0
dtype = bfloat16
compile = True                     # Linux + CUDA，torch.compile 正常工作
```

### 训练过程

启动：
```bash
cd /root/code/nanoGPT-master
python train.py config/train_gpt2_wikitext.py
```

Loss 下降曲线（关键节点）：

| Iter | Train Loss | Val Loss | 说明 |
|------|------------|----------|------|
| 0 | 11.01 | - | 初始随机权重 |
| 100 | 7.66 | - | 快速下降期 |
| 200 | 6.40 | 6.32 | 第一次评估 |
| 300 | 5.92 | - | 持续下降 |
| ... | ... | ... | |
| 5000 | - | **3.0458** | 最终 |

训练速度：
- 每步 ~3.2 秒（含 DataLoader + forward + backward + 40 步梯度累积）
- MFU（Model FLOPs Utilization）：~27%
- **总耗时：约 4 小时 27 分钟**

### 结果

| 指标 | 值 |
|------|-----|
| Final Val Loss | **3.0458** |
| Perplexity | ~21.0 |
| Checkpoint | 1.4 GB（含模型 + AdamW 状态） |

### 踩坑记录

**问题 1：磁盘空间不足**
- 系统盘仅 30GB，训练数据+HF 缓存+checkpoint 会撑爆
- 解决：将 `data/` 目录迁移到 50GB 数据盘并建立符号链接

**问题 2：SSH 连接不稳定**
- 本地 paramiko/原生 SSH 频繁断开（"Error reading SSH protocol banner"）
- VS Code Remote-SSH 相对稳定

**问题 3：gradient_accumulation 的作用**
- 单卡显存有限，`batch_size=8` 无法再大
- 通过 `gradient_accumulation_steps=40` 等效增大批量，保持训练稳定
- 代价：每步 3.2s 中约 100ms 是梯度累积的同步开销

### 与 Day 1 本地训练的对比

| 维度 | Day 1 (GTX 1650) | Day 4-5 (RTX 4090D) |
|------|------------------|---------------------|
| 模型 | 10.65M（6层/384维） | 124M（12层/768维） |
| 数据集 | Shakespeare (1MB) | WikiText-103 (228MB) |
| 步数 | 100 / 1000 | 5000 |
| 耗时 | 14s / 15s | 4.5h |
| Val Loss | 2.71 / 2.50 | 3.05 |
| torch.compile | 不可用（Windows 无 Triton） | 可用（Linux） |
| 训练目标 | 验证流程 | 完整训练 |

---

## Day 4-5：AutoDL Shakespeare 5000 步完整训练

> 2026-06-10 完成 | 产物：[experiments/shakespeare-5000/](../experiments/shakespeare-5000/)

### 环境

| 项目 | 详情 |
|------|------|
| **平台** | AutoDL (SeetaCloud) |
| **GPU** | RTX 4090，24 GB 显存 |
| **磁盘** | 系统盘 30 GB（overlay） |
| **Python** | 3.8.10 (miniconda3) |
| **PyTorch** | 2.0.0+cu118 |
| **SSH** | `ssh -p 53322 root@connect.westb.seetacloud.com` |
| **Wandb** | offline 模式（无 API key，本地保存日志） |

### 环境搭建

```bash
# 1. conda 激活
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base

# 2. 克隆 + 装依赖
cd /root && git clone https://github.com/karpathy/nanoGPT.git
pip install tiktoken wandb -q

# 3. 准备莎士比亚数据集
cd /root/nanoGPT/data/shakespeare_char && python prepare.py
# train 1,003,854 tokens / val 111,540 tokens / vocab 65 chars
```

### 训练配置

Baby GPT ~30M 参数，与 Day 1 同一 config，区别是 RTX 4090 显存充足，用满默认配置：

| 参数 | Day 1 (GTX 1650) | Day 4-5 (RTX 4090) |
|------|:--:|:--:|
| batch_size | 4 | 64 |
| block_size | 64 | 256 |
| 每步 tokens | 256 | 16,384 |
| max_iters | 100 | 5000 |
| 每步耗时 | ~17ms | ~27ms |

启动：
```bash
cd /root/nanoGPT
WANDB_MODE=offline nohup python train.py config/train_shakespeare_char.py \
    --wandb_log=True --wandb_run_name=shakespeare-5000 \
    > train_5000.log 2>&1 &
```

### 训练结果

完整 loss 曲线（每 250 步 eval）：

```
step     0: train 4.2874  val 4.2823    ← 初始，≈ ln(65)
step   250: train 1.9725  val 2.0774
step   500: train 1.5317  val 1.7292
step   750: train 1.3651  val 1.5908
step  1000: train 1.2814  val 1.5352
step  1250: train 1.2070  val 1.5103
step  1500: train 1.1529  val 1.4841
step  1750: train 1.1043  val 1.4704    ← 🏆 最佳 val loss
step  2000: train 1.0555  val 1.4791    ← 开始过拟合
step  2500: train 0.9609  val 1.4944
step  3000: train 0.8653  val 1.5320
step  3500: train 0.7826  val 1.5831
step  4000: train 0.7086  val 1.6378
step  4500: train 0.6527  val 1.6880
step  5000: train 0.6234  val 1.7133
```

| 指标 | 值 |
|------|-----|
| Final Train Loss | **0.6234** |
| Final Val Loss | 1.7133 |
| Best Val Loss | **1.4704**（step 1750） |
| 总耗时 | ~3 分钟 |
| MFU | ~14% |
| Checkpoint | [ckpt.pt](../experiments/shakespeare-5000/ckpt.pt)（129 MB） |
| 训练日志 | [train_5000.log](../experiments/shakespeare-5000/train_5000.log)（597 行） |

### 解读

**1. 过拟合出现在 step 1750。** train loss 一直下降到 0.62，但 val loss 从 1.47 反弹到 1.71。Shakespeare 只有 1MB / 100 万 tokens，30M 模型跑 5000 步 × 16,384 token/step ≈ 8200 万 token ≈ 82 个 epoch，严重过拟合。最优模型在 step 1750。

**2. 与 Day 1 1000 步对比。** Day 1 的 val loss 最低 2.50（batch_size=4, block_size=64），这次用满默认配置（64/256）后 val loss 降到 1.47，每步处理 64 倍数据，收敛质量大幅提升。

**3. MFU 14%。** 30M 小模型计算量小，GPU 大量时间在 kernel launch 开销上。大模型 MFU 能到 27%+。

### 踩坑

| 问题 | 解决 |
|------|------|
| 数据集放错目录（`shakespeare/` vs `shakespeare_char/`） | 在正确目录重新 prepare |
| Wandb 无 API key | `WANDB_MODE=offline`，本地保存 |
| 本地 paramiko 通过密码连 SSH 可用 | — |

---

## Wandb 可视化

> 此前训练均使用 `wandb_log=False`，未记录可视化数据。本次补跑 500 步 Shakespeare 短训练，启用 wandb 日志，生成本地可视化图表。

### 训练配置

```bash
python train.py config/train_shakespeare_char.py \
    --wandb_log=True --wandb_project='shakespeare-char' --wandb_run_name='week1-demo' \
    --max_iters=500 --batch_size=4 --block_size=64 \
    --eval_interval=50 --compile=False
```

### 产物

| 文件 | 说明 |
|------|------|
| `01_loss_curves.png` | train/val loss 曲线（4.32 → 2.54，11 个评估点） |
| `02_iter_loss.png` | 每步迭代 loss + 滑动平均平滑曲线 |
| `03_lr_schedule.png` | Cosine 学习率衰减 + 停止位置标记 |
| `04_iter_time.png` | 迭代耗时（平均 16.5ms/步，排除 eval 的 ~3.4s） |
| `05_mfu.png` | Model FLOPs Utilization（平均 0.25%，GTX 1650 小模型利用率低） |
| `06_combined_dashboard.png` | 综合仪表盘（四合一视图） |
| `00_run_info.txt` | 运行摘要 + 最终指标 |

产物目录：[experiments/gpt2-wikitext/wandb-screenshots/](../experiments/gpt2-wikitext/wandb-screenshots/)
Wandb 在线：https://wandb.ai/models-hefei-university-of-technology/shakespeare-char/runs/1ugkht12

### 解读

**Train loss 4.32 → 2.54**：500 步持续下降，无反弹，学习率健康。

**Val loss 与 train loss 始终接近**：step 500 时 train 2.54 vs val 2.55，差距仅 0.01，未过拟合。

**MFU 仅 0.25%**：GTX 1650 算力 2.9 TFLOPS，10.65M 小模型 forward/backward 计算量小，大部分时间消耗在 Python 开销和 kernel launch 上。大模型（GPT-2 124M）MFU 可达 27%。

**Iter time 稳定在 16-17ms**：eval 步骤因跑 200 步验证需要 ~3.4s。正常迭代时间非常一致，说明系统没有间歇性干扰。

---

## 阶段检查清单

- [√] Shakespeare 数据集训练成功，loss 正常下降
- [√] 完成 3 组超参实验（block_size / n_layer / lr），记录分析
- [√] Temperature 生成对比实验（0.8 / 1.0 / 1.5），理解了 softmax + temperature 原理
- [√] OpenWebText 数据集：国内网络无法下载（HF 直连+镜像均失败），改用 WikiText-103 在 AutoDL 上完成
- [√] AutoDL GPT-2 (124M) 完整训练：5000 步，val loss 3.05，4.5h，checkpoint 1.4GB
- [√] `sample.py` 在充分训练的模型上生成可读文本（T=0.6/0.8/1.0 三组对比，T=0.8 最佳）
- [√] Wandb 可视化的截图保存
  - 跑 500 步 Shakespeare 短训练，`wandb_log=True`，生成 6 张可视化图表
  - 产物：[experiments/gpt2-wikitext/wandb-screenshots/](../experiments/gpt2-wikitext/wandb-screenshots/)（6 张 PNG + 1 个 run_info.txt）
  - Wandb 在线地址：https://wandb.ai/models-hefei-university-of-technology/shakespeare-char/runs/1ugkht12
  - 图表清单：loss 曲线、迭代 loss（平滑）、LR schedule、迭代时间、MFU、综合仪表盘
