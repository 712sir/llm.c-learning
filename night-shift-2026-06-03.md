# 🌙 夜班学习计划 — 2026-06-03

> 总时长：3h40min | 原则：视频为主 + 论文起步 + 碎片脑刷 | 五线全覆盖
>
> ⚠️ 出发前手机缓存好：B站视频（D2L + cs224n）+ PDF 论文/书籍

---

## 总览

| 线 | 方向 | 今日占比 | 载体 |
|:--:|------|:--:|------|
| D/C | 深度学习入门 | 75min | 🎬 李沐 D2L 视频 |
| B | CUDA 基础 | 40min | 📖 PMPP 书籍 |
| D | NLP 理论 | 40min | 🎬 cs224n 视频 |
| E | vLLM 推理引擎 | 25min | 📄 PagedAttention 论文 |
| A | 算法刷题 | 20min | 🧠 LeetCode 脑刷 |
| C | llm.c 训练 | 20min | 📝 笔记回顾 |

---

## 一、🎬 李沐「动手学深度学习 V2」P1 + P2（75 min）

> 链接：[B站 171集](https://b23.tv/IjnkTRm)
> 为什么选：计划里「cs224n 理论 + llm.c 工程 + D2L 上手」三位一体，D2L 是唯一能脱离电脑推进的视频入口

| # | 集 | 内容 | 时长 | ✅ |
|---|------|------|:--:|:--:|
| 1 | P1 | 课程介绍 + 深度学习基础框架 | 45min | ⬜ |
| 2 | P2 | 数据操作 + 数据预处理（Tensor 操作入门） | 30min | ⬜ |

---

## 二、📖 B 线 — PMPP Ch1 + Ch2（40 min）

> 文件：`resources/books/01-PMMP-4th.pdf` | B1 CUDA 轨道主教材

| # | 章 | 内容 | 时长 | ✅ |
|---|------|------|:--:|:--:|
| 3 | Ch1 | Introduction — GPU 架构 + 为什么 GPU 适合并行计算 | 20min | ⬜ |
| 4 | Ch2 | Heterogeneous Computing — Host/Device 模型 + 内存管理 | 20min | ⬜ |

> Ch3（Grid/Block/Thread）明天回电脑对着代码学更高效，今晚两章打底够用。

---

## 三、🧠 A 线 — 算法脑刷（20 min）

> LeetCode App / 代码随想录网页版

| # | 内容 | 方式 | 时长 | ✅ |
|---|------|------|:--:|:--:|
| 5 | Day A1-A2 已完成的 5 道核心题 | 脑中过思路：704 二分查找 / 27 移除元素 / 977 有序数组的平方 / 209 长度最小的子数组 / 59 螺旋矩阵 II | 8min | ⬜ |
| 6 | Day A3 拓展题 35 + 34 | 看题，脑中模拟二分查找的左右边界处理，不写码 | 7min | ⬜ |
| 7 | LeetCode 随机 1 题 | App 热榜点开 → 读题 → 想思路 → 不看题解 | 5min | ⬜ |

> 算法一天不碰手感就掉。不写代码没关系，想清楚二分边界就是收获。

---

## 四、🎬 D 线 — cs224n Lec1（40 min）

> B站搜「cs224n 2024」| 你 W4 才正式启动，今晚先预热第一讲

| # | 内容 | 时长 | ✅ |
|---|------|:--:|:--:|
| 8 | Lecture 1：NLP 导论 + word2vec 动机 + 课程概览 | 40min | ⬜ |

> Lec2 讲 Word2Vec 数学推导，没电脑推公式效率低，留到 W4 配合 Assignment 1 一起搞。

---

## 五、📄 E 线 — vLLM PagedAttention 论文起步（25 min）

> 文件：`resources/papers/11-Kwon等-vLLM-PagedAttention-2023.pdf` | E 线 W8 正式启动

| # | 内容 | 方式 | 时长 | ✅ |
|---|------|------|:--:|:--:|
| 9 | Abstract + Section 1 + 图 1/2/3 | 带着问题读：① KV Cache 为什么是推理的内存瓶颈？② PagedAttention 借鉴了 OS 虚拟内存的什么思想？ | 25min | ⬜ |

> E 线是你「简历杠杆最高」的线。今晚花 25 分钟了解 PagedAttention 是什么，不算早。

---

## 六、📝 C 线 — llm.c 笔记回顾（20 min）

> 文件：`llm.c-learning/notes/week1-environment.md`

| # | 内容 | 方式 | 时长 | ✅ |
|---|------|------|:--:|:--:|
| 10 | 回顾 W1 Day1-3 笔记 | ① 环境搭建踩了哪些坑 ② 超参对照实验（block_size / n_layer / lr）各组结果 ③ Temperature 对生成质量的影响 | 15min | ⬜ |
| 11 | Day4-5 规划 | 想清楚：明天回电脑，5000 步完整训练的目标是什么？Wandb 可视化要记录哪些指标？ | 5min | ⬜ |

---

## 📋 完整检查清单

| # | 线 | 类型 | 内容 | ⏱ | ✅ |
|:--:|:--:|:--:|------|:--:|:--:|
| 1 | D/C | 🎬 | 李沐 D2L P1：深度学习基础框架 | 45min | ⬜ |
| 2 | D/C | 🎬 | 李沐 D2L P2：数据操作 + 预处理 | 30min | ⬜ |
| 3 | B | 📖 | PMPP Ch1：GPU 并行计算模型 | 20min | ⬜ |
| 4 | B | 📖 | PMPP Ch2：Host/Device 模型 | 20min | ⬜ |
| 5 | A | 🧠 | 回顾 Day A1-A2 五道核心题 | 8min | ⬜ |
| 6 | A | 🧠 | 脑刷 35 + 34（二分边界） | 7min | ⬜ |
| 7 | A | 🧠 | LeetCode 随机 1 题想思路 | 5min | ⬜ |
| 8 | D | 🎬 | cs224n Lec1：NLP 导论 | 40min | ⬜ |
| 9 | E | 📄 | vLLM 论文 Abstract + §1 + 图表 | 25min | ⬜ |
| 10 | C | 📝 | llm.c W1 Day1-3 笔记回顾 | 15min | ⬜ |
| 11 | C | 📝 | 确认 Day4 回电脑的任务 | 5min | ⬜ |

> **合计：3h40min** | 视频 115min · 阅读 85min · 脑练 20min
>
> 🎯 **今日核心产出**：理解 FlashAttention/PagedAttention 解决什么问题 + 建立 GPU 编程思维模型 + 深度学习框架全局认知 + 五线手感全保持

---

## 💡 备忘

- [ ] 出发前：B站 D2L P1/P2 + cs224n Lec1 全部离线缓存
- [ ] 出发前：PMMP PDF + vLLM 论文 PDF + llm.c 笔记传到手机
- [ ] 明天回电脑：B1 CUDA Toolkit 安装 + 第一个 kernel → llm.c W1 Day4 5000 步训练 → A 线 Day A3 写码
