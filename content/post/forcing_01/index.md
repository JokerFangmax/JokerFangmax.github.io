---
title: "Forcing: 01. Scheduled Sampling"
date: 2026-10-14
math: true
tags:
  - "Forcing"
  - "Sequence Modeling"
  - "Scheduled Sampling"
categories:
  - Forcing
series: forcing
weight: 1
description: "Scheduled Sampling（Bengio et al., NeurIPS 2015）精读：训练看真实前缀、推理只能吃自己的预测，早期小错沿序列放大。处方是课程学习式的逐 token 抛硬币——以概率 ε_i 用真实历史、1−ε_i 用模型自己的预测，ε_i 按 linear/exponential/inverse sigmoid 三种调度从 1 退火。MSCOCO 描述 CIDEr 89.5→92.1（集成 5 模型 98.7 超 10 模型基线的 95.7，拿下 2015 竞赛第一）；成分句法分析 F1 86.54→88.08；TIMIT 解码 FER 46.0→34.5。注意：混合历史后的损失已不再是原最大似然目标。"
paper_id: "1506.03099v3"
---

# 从 Teacher Forcing 到模型自己的历史

> **Forcing 系列 · 第 01 篇 · 训练与推理的差异**
> [← 总览](../forcing/) · [下一篇：Scheduled Sampling for Transformers →](../forcing_02/)
> 论文：*Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*（Bengio, Vinyals, Jaitly, Shazeer；Google Research；NeurIPS 2015）。
>
> 📂 **阅读方式**：正文只留主线；长推导、超参标本与实验全表收折叠块。

## 论文信息

- **原题**：Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks
- **阅读版本**：arXiv:1506.03099v3，对应本地收集的 PDF 版本。
- **原文**：[arXiv 摘要页](https://arxiv.org/abs/1506.03099v3) · [PDF](https://arxiv.org/pdf/1506.03099v3)

## 1. 这篇论文解决什么问题

自回归序列生成（autoregressive sequence generation）的标准训练法是教师强制（Teacher Forcing）：预测第 $t$ 个 token 时，把**数据里的真实前一个 token** $y_{t-1}$ 喂给模型。推理时真值不存在，只能把**模型自己的预测** $\hat{y}_{t-1}$ 喂回去。论文开篇的诊断是：一旦某步出错，模型进入训练时从未见过的状态空间区域，后续错误被快速放大——这就是后来被称作暴露偏差（exposure bias）的问题的原始表述。

| | 条件前缀 | 一步错误的后果 |
|---|---|---|
| 训练（Teacher Forcing） | 真实 $y_{1:t-1}$ | 每步条件都正确，错误**不**跨步传播 |
| 推理（自回归解码） | 模型 $\hat{y}_{1:t-1}$ | 一步错 → 状态偏离训练分布 → 步步错 |

前人方案各有缺陷（论文 §1、§3）：

| 方案 | 思路 | 缺陷（论文给出的） |
|---|---|---|
| 束搜索（beam search） | 推理时维护多个候选序列缓解错配 | 连续状态空间无动态规划，实际能保留的候选极少 |
| SEARN（Daumé et al., 2009） | 批式元算法：每轮训新模型、在新策略上修正 | 批式而非在线，训练慢；面向强化学习设定 |
| 训练-推理都用 beam（Collins & Roark, 2004） | 训练时也跑 beam，用 ranking loss 对齐 | RNN 的多维连续状态无法分解，训练时跑不动 beam |
| 动态 oracle（Goldberg & Nivre, 2012） | 在线调整目标 | perceptron、非状态模型；选真值概率固定不可调 |

本文的处方是**课程学习（curriculum learning）**：单一模型在线训练，训练条件从「历史全真」（$\epsilon_i=1$，即教师强制）平滑过渡到「历史几乎全是自己的」（$\epsilon_i=0$，即推理工况），让模型在训练中就学会应付自己的错误。

## 2. 最小背景

本篇是系列的概念起点，前置只需三件：

- **自回归分解与教师强制**：序列似然按链式法则逐 token 分解，训练时条件在真实前缀上——这是 [总览](../forcing/) §1 问题地图里「历史来源」维度的原点；
- **LSTM**：论文实验全部用 LSTM 做循环单元；理解「状态 $h_t$ 汇总了输入与已输出 token」即可；
- **课程学习**（Bengio et al., 2009）：从易到难安排训练样本/目标的范式；SS 把「难」定义为「条件历史不再可靠」。

顺带一个定量锚点：[WM 03](../world_memory/WM_03_memory_forcing/) §6.3 的 toy 把教师强制的错配算成了数字——训练条件误差恒 0、推理时误差随机游走无界增长，**训练分布对推理 drift 的覆盖率为精确 0%**。这与本篇「模型可能身处训练时从未见过的状态空间」的动机同源，只是那里换成了相机位姿 drift。

## 3. 方法总览

用一段话走完整条流水线（论文 Figure 1）：输入 $X$ 先被编码（图像过预训练 CNN 得到嵌入、当作「第一个词」；源句用词嵌入；语音用逐帧声学特征），LSTM 从初始状态出发逐步递推；**新增的历史选择器**在预测每个 token $y_t$ 时抛一枚硬币——正面喂真实前一个 token $y_{t-1}$，反面喂模型自己的估计 $\hat{y}_{t-1}$（按模型分布采样或取 argmax）；状态 $h_t$ 经线性投影 + softmax 得到字典上的分布，对真实 $y_t$ 累积对数似然；**新增的调度器**在训练外层按 mini-batch 计数 $i$ 计算 $\epsilon_i$，控制「用真值」的概率随训练推进从 1 退火下降。

| 模块 | 做什么 | 相对标准训练的变化 |
|---|---|---|
| 输入编码 | 图像 CNN 嵌入 / 词嵌入 / 声学特征 | 无 |
| 循环状态 | $h_t = f(h_{t-1},\ y_{t-1})$ | 上一步 token 的**来源**可切换 |
| 历史选择器 | 逐 token 抛硬币：$y_{t-1}$（概率 $\epsilon_i$）或 $\hat{y}_{t-1}$（概率 $1-\epsilon_i$） | **新增** |
| 输出头 | $h_t$ 线性投影 + softmax | 无 |
| 调度器 | $\epsilon_i$ 随 mini-batch 计数 $i$ 按曲线退火 | **新增** |

监督目标**不变**：仍是对真实 $y_t$ 的对数似然。SS 替换的是**条件输入的来源**，而不是把监督标签也换成模型预测——「在自己的历史上学习」不等于「给自己制造标签」，这个区分贯穿整个系列。

## 4. 核心原理完整推导

### 4.1 从链式法则到 RNN 参数化

对单条样本 $(X, Y)$，$Y=(y_1,\dots,y_T)$，序列对数概率按链式法则分解（论文 Eq. 1）：

$$
\log P(Y\mid X) = \log P(y_1^T\mid X) = \sum_{t=1}^{T} \log P\big(y_t \mid y_1^{t-1},\ X\big)
\tag{Eq.1}
$$

直觉：生成一个序列的概率，等于每一步「看了前面所有内容后，猜对下一个 token」的对数概率之和。RNN 把无限长的条件历史压缩进状态向量（论文 Eq. 2）：

$$
\log P\big(y_t \mid y_1^{t-1},\ X;\ \theta\big) = \log P(y_t \mid h_t;\ \theta)
\tag{Eq.2}
$$

状态由循环函数递推（论文 Eq. 3）：

$$
h_t =
\begin{cases}
f(X;\ \theta) & t = 1,\\[2pt]
f(h_{t-1},\ y_{t-1};\ \theta) & \text{otherwise},
\end{cases}
\tag{Eq.3}
$$

$P(y_t\mid h_t;\theta)$ 实现为 $h_t$ 到字典分数的线性投影 + softmax；变长序列靠字典里加一个 `<EOS>` 终止符处理。训练目标是对数似然最大化（论文 Eq. 4）：

$$
\theta^{*} = \arg\max_{\theta} \sum_{(X^i,\, Y^i)} \log P\big(Y^i \mid X^i;\ \theta\big)
\tag{Eq.4}
$$

### 4.2 错配到底发生在哪一步

对比 Eq. 3 在两种场合的实例化：

| 场合 | 第 $t$ 步的递推 | 条件分布的支撑集 |
|---|---|---|
| 训练 | $h_t = f(h_{t-1},\ y_{t-1};\theta)$ | 由**真实**前缀诱导的状态分布 |
| 推理 | $h_t = f(h_{t-1},\ \hat{y}_{t-1};\theta)$ | 由**模型 rollout** 诱导的状态分布 |

关键不是某一步预测错，而是**错的 token 会成为下一步的输入**：模型被带进训练分布支撑集之外的状态区域，此后每一步都-conditioning 在「没见过的世界」上。束搜索只能缓解——连续状态无法合并等价路径，能保留的候选数极少。

### 4.3 Scheduled Sampling 与三种调度

**机制**：用 mini-batch SGD 训练时，对第 $i$ 个 mini-batch 里每个待预测 token $y_t \in Y$，抛一枚硬币——以概率 $\epsilon_i$ 用真实前一个 token $y_{t-1}$，以概率 $1-\epsilon_i$ 用模型自己的估计 $\hat{y}_{t-1}$；估计可按 $P(y_{t-1}\mid h_{t-1})$ 采样，也可取 argmax。$\epsilon_i=1$ 退化为教师强制，$\epsilon_i=0$ 恰是推理工况。

三种调度曲线（论文 Figure 2，公式在论文中未编号）：

| 调度 | 公式 | 形状直觉 |
|---|---|---|
| 线性衰减 | $\epsilon_i = \max(\epsilon,\ k - ci)$ | 恒定斜率下降；$\epsilon$ 是真值概率下限，$k$/$c$ 定起点与坡度 |
| 指数衰减 | $\epsilon_i = k^{i}$（$k<1$） | 一开始就猛降——每个 mini-batch 乘一次 $k$ |
| 反 S 形衰减 | $\epsilon_i = k / \big(k + \exp(i/k)\big)$（$k \ge 1$） | 前期贴着 1 不动，后期才衰减：$e^{i/k}$ 要追上 $k$ 需 $i \approx k\ln k$ |

课程直觉（论文原话的转述）：训练初期模型很差，从它采样基本等于随机 token，会导致收敛极慢，所以前期该多用真值；训练后期模型已经够好，多用自身预测才对齐真实推理工况。注意 $k$ 在三种调度里含义各不相同（起点/衰减底数/尺度），这是符号复用。

### 4.4 损失函数到底变成了什么

<details>
<summary><b>展开：混合历史下的目标——为什么它不再是 Eq. 4 的最大似然</b></summary>

把抛硬币的随机性显式写出来。定义混合前缀 $\tilde{y}_t$：各 token 独立地以概率 $\epsilon_i$ 取 $y_t$、以概率 $1-\epsilon_i$ 取模型采样 $\hat{y}_t$。SS 实际优化的是（我的形式化，论文未给出此式）：

$$
\mathcal{L}_{\text{SS}}(\theta) = -\,\mathbb{E}_{\tilde{y}_{1:T} \sim q_{\theta,\,\epsilon}}\left[\sum_{t=1}^{T} \log p_\theta\big(y_t \mid \tilde{y}_1^{t-1},\ X\big)\right]
$$

与 Eq. 4 有两个本质差别：

1. **前缀分布 $q_{\theta,\epsilon}$ 依赖 $\theta$**：模型变，混合历史的分布也变。若对采样决策回传梯度，目标里会多出 $\nabla_\theta q$ 项（结构上接近强化学习）。论文明确说**没有**这样做——对 $\hat{y}_{t-1}$ 实施了事实上的梯度截断（stop-gradient），把它当常数条件用，留作未来工作；
2. **截断后得到的是有偏代理**：梯度只流经「以固定混合历史为条件」的那部分，不对应任何单一明确目标的精确梯度。

再用 Jensen 不等式看一眼期望目标的形状（$-\log$ 是凸函数）：

$$
\mathbb{E}_{\tilde{y}}\big[-\log p_\theta(y_t \mid \tilde{y}_{1:t-1})\big] \;\ge\; -\log\, \mathbb{E}_{\tilde{y}}\big[p_\theta(y_t \mid \tilde{y}_{1:t-1})\big]
$$

右边的期望项是把条件按混合分布**平滑后的边缘似然**——SS 在优化一个被 $q$ 平滑过的目标，而不是精确的联合对数似然。这与系列后续（尤其 Goyal et al., 2017 的可微化分析，见 [02](../forcing_02/)）直接相关。

一个论文自己做的旁证（§4.1）：把「从模型采样」换成「从均匀分布采样」（Uniform Scheduled Sampling），MSCOCO CIDEr 90.9——好于基线 89.5 但差于真 SS 的 92.1。如果 SS 只是某种正则化，两者应该一样好；差距说明增益确实来自「建立在当前模型之上」。

</details>

**正文结论**：SS 之后的训练目标是对数似然的一个**课程式有偏代理**——不要把混合历史训练直接视为 Eq. 4 的等价优化。这个偏差是特性不是 bug：它换来的，是训练分布的支撑集从「gold 前缀诱导的状态」扩展到「模型自己会走到的状态」。

## 5. 训练与推理细节

训练侧要点：$\epsilon_i$ 以 **mini-batch 计数 $i$** 为自变量（不是 epoch）；硬币**逐 token** 抛（脚注 2：逐序列抛一次的尝试结果差得多，接近 Always Sampling——因为训练前期连续错误被整段放大）；脚注 3 说明与基线相比训练速度无实质差异（RNN 训练本来就是逐步算的，采样不增加额外的前向次数）。推理侧完全不变（beam search 或采样解码），SS 只动训练协议。

<details>
<summary><b>展开：三个任务的超参标本（论文 §4.1–4.3）</b></summary>

| 项 | MSCOCO 图像描述 | 成分句法分析 | TIMIT 语音识别 |
|---|---|---|---|
| 输入 | 图像，预训练 CNN（去分类头）嵌入当「第一个词」 | 源句，词嵌入；带注意力（Bahdanau et al.） | 逐帧声学：40 维 log Mel 滤波器组 + 一阶/二阶时间差分 |
| 目标 | 5 个描述任选其一 | 128 个树操作符号 | 180 维 GMM-HMM 强制对齐状态标签 |
| 模型 | 1 层 LSTM，512 隐单元 | 1 层 LSTM，512 隐单元 | 2 层 LSTM，每层 250 单元 |
| 词/符号表 | 8857 词 | 输入约 90k 词 | 180 个 HMM 状态 + 起始符 $S$ |
| 嵌入维度 | 512 | 512 | — |
| 数据 | 训练 75k 图 / 验证 5k 图，每图 5 句 | 40k 训练实例，WSJ 22 开发集 | 训练/验证/测试 3696/400/192 条，平均长 304 帧 |
| $\epsilon_i$ 调度 | inverse sigmoid | inverse sigmoid | 线性：10 个 epoch 内从 $\epsilon_s$ 线性降到 $\epsilon_e$ 后保持 |
| 其他 | 每配置 1 个模型 | — | 每配置训 3 个模型取平均；解码 beam 宽 10 |

语音任务的模型结构与另两个不同（论文 Eq. 5/6）：声学帧 $x_t$ 与上一步标签一起进循环，

$$
\log P(Y\mid X;\theta) = \sum_{t=1}^{T}\log P\big(y_t \mid y_1^{t-1},\ x_1^t;\ \theta\big) = \sum_{t=1}^{T}\log P(y_t\mid h_t;\theta)
\tag{Eq.5}
$$

$$
h_t =
\begin{cases}
f(o_h,\ S,\ x_1;\ \theta) & t=1,\\[2pt]
f(h_{t-1},\ y_{t-1},\ x_t;\ \theta) & \text{otherwise},
\end{cases}
\tag{Eq.6}
$$

其中 $o_h$ 是与 $h_t$ 同维的零向量、$S$ 是起始符。注意这个设定里训练与推理都额外看声学输入 $x_t$，错配只来自标签历史 $y_{1:t-1}$。

</details>

## 6. 实验与消融

### 6.1 MSCOCO 图像描述（论文 Table 1）

| 方法 | BLEU-4 | METEOR | CIDEr |
|---|---|---|---|
| Baseline | 28.8 | 24.2 | 89.5 |
| Baseline + Dropout | 28.1 | 23.9 | 87.0 |
| Always Sampling | 11.2 | 15.7 | 49.7 |
| **Scheduled Sampling** | **30.6** | 24.3 | **92.1** |
| Uniform Scheduled Sampling | 29.2 | 24.2 | 90.9 |
| Baseline 集成 ×10 | 30.7 | 25.1 | 95.7 |
| **Scheduled Sampling 集成 ×5** | **32.3** | **25.4** | **98.7** |

读表要点：

1. **SS 单模型全面提升**（CIDEr 89.5→92.1），而 Dropout 反而伤真实指标（只帮对数似然）——两者修的不是同一个东西；
2. **Always Sampling 崩溃**（BLEU-4 11.2）：不退火、一开始就全用自身预测，模型根本学不动——课程调度的必要性直接可见；
3. **Uniform SS 介于两者之间**（90.9）：增益不是单纯正则化（§4.4 的旁证）；
4. **集成效率**：5 个 SS 模型（98.7）超过 10 个基线模型（95.7）——一半的模型数、更高的分；
5. 逐序列抛硬币的结果与 Always Sampling 一样差（§4.1 文字叙述，未进表）。

这套方法拿下 **2015 年 MSCOCO 图像描述竞赛第一名**。

### 6.2 成分句法分析（论文 Table 2）

| 方法 | F1 |
|---|---|
| Baseline LSTM | 86.54 |
| Baseline + Dropout | 87.0 |
| Always Sampling | —（生成的常不是合法树） |
| **Scheduled Sampling** | **88.08** |
| **SS + Dropout** | **88.68** |

这个任务与描述生成处于两个极端：绝大多数句子只有一棵句法树，函数近乎确定——训练/测试困惑度 1.1（描述任务约 7），模型本不该犯多少错。即便如此 SS 仍有正效应，且**与 Dropout 可加**（86.54 → 88.08 → 88.68）。论文诚实地留了一个开口：只有 40k 训练实例、过拟合占比很大，SS 帮的到底是过拟合还是错配，说不清。

### 6.3 TIMIT 语音识别（论文 Table 3）

| 方法 | $\epsilon_s$ | $\epsilon_e$ | 逐步预测 FER（验证集） | 解码 FER（测试集） |
|---|---|---|---|---|
| Always Sampling | 0 | 0 | 34.6 | **35.8** |
| Scheduled Sampling 1 | 0.25 | 0 | 34.3 | **34.5** |
| Scheduled Sampling 2 | 0.5 | 0 | 34.1 | 35.0 |
| Scheduled Sampling 3 | 0.9 | 0.5 | **19.8** | 42.0 |
| Baseline LSTM | 1 | 1 | **15.0** | 46.0 |

<details>
<summary><b>展开：这张表为什么值得细读（论文 §4.3 的机制分析）</b></summary>

表面现象很反直觉：**基线的逐步预测最好（15.0）、解码最差（46.0）**，差了 31 个点；SS 1（$\epsilon$ 从 0.25 线性降到 0）解码最好（34.5）。

论文的解释链：

1. HMM 对齐的状态标签高度冗余——同一状态连续占据多帧，且状态转移受限于固定的子集。教师强制下的模型发现「抄上一步标签」就能把逐步预测做得很好，于是**过度依赖标签历史、没有充分利用声学输入** $x_1^t$；
2. 推理解码时标签历史换成自己的预测，模型立即露馅：测试条件离训练条件太远，解码 FER 46.0；
3. Always Sampling 的模型反过来**只信声学、忽略自身预测**，训练与测试条件一致，解码 35.8 反而不差；
4. SS 1 两头兼顾：解码 34.5，全场最好。

另一处细节：采样类模型在约 9 个 epoch 达到验证最优后开始退化，基线则要约 14 个 epoch——论文归因于训练不精确（没对采样概率回传梯度，§4.4 的偏差在后期显形）。SS 3（终值 $\epsilon_e=0.5$ 仍偏高真值）逐步预测 19.8 但解码 42.0，说明调度终点选错时代价很大。

</details>

## 7. 与系列其他篇目的关系

| 连接 | 关系 |
|---|---|
| [总览](../forcing/) 五维地图 | 本篇是「**历史来源**」维度的原点：条件来自真实数据、混合历史、还是模型 rollout，从这里开始成为一条独立的设计轴 |
| [02. Scheduled Sampling for Transformers](../forcing_02/) | SS 逐 token 抛硬币在 RNN 上顺手，在并行训练的 Transformer 上不成立——02 用两遍解码重构它 |
| [03. Parallel Scheduled Sampling](../forcing_03/) | 指出 SS 的串行性在加速器上是致命伤（0.13 步/秒 vs 教师强制 47），用多轮并行采样-混合救回，并证明 $K\ge T$、$p=1$ 时收敛到真采样解码 |
| [04. Diffusion Forcing](../forcing_04/)（下一篇的下一篇） | 系列从「离散 token 的历史替换」跳到「**连续噪声等级**的历史混合」——$\epsilon_i$ 的课程思想被推广到每 token 独立噪声级；读它之前可补 [VG 01 扩散基础](../video_generation/VG_01_diffusion_basics/) |
| [WM 03: Memory Forcing](../world_memory/WM_03_memory_forcing/) | 同族问题的视频版定量：TF 的 drift 覆盖率 0%；其 Chained Forward Training（链入自身预测）就是 SS 思想在视频扩散里的直系后代 |
| Goyal et al., 2017 | 可微化 SS：把 argmax 换成按分数加权的嵌入混合并回传梯度——[02](../forcing_02/) 的 Embedding Mix 直接继承它 |

## 8. 局限与个人评价

- **目标不再是 MLE**：§4.4 的偏差是结构性的——对采样决策截断梯度后，优化的对象连论文自己都没有明确定义。语音实验「9 epoch 后退化」就是这个偏差的显形；「回传采样决策的梯度」被列为未来工作，后来由 Goyal et al.（可微 SS）与序列级训练（Ranzato et al., 2015）两个方向分头接走；
- **调度超参全靠手调**：$k$、$c$、退火终点都依赖任务与收敛速度的先验；语音任务 SS 3 选错 $\epsilon_e$ 直接把解码 FER 推到 42.0，调度的脆弱性没有系统研究；
- **增益解释不干净**：句法分析实验里论文自己承认无法区分 SS 帮的是过拟合还是错配（40k 实例的小数据）；
- **串行性被论文的脚注掩盖**：「训练不比基线慢」只在 RNN 成立——RNN 训练本来就是逐步的。到了并行训练的 Transformer，逐 token 采样的串行代价变成两个数量级（[03](../forcing_03/) 实测 0.13 vs 47 步/秒）。这是本篇方法最大的时代局限，也是 02/03 存在的理由；
- **个人评价**：作为概念起点它定义了此后十年「训练-推理错配」问题的标准叙事（课程式混合历史 + 监督不变），MSCOCO 竞赛第一与三任务的全面正效应让这个叙事站稳了。但读它最该带走的是 §4.4 那个没有写进论文的警告：**换历史来源的同时，你优化的是什么目标，必须重新回答**——这个问题在系列后段（Self Forcing、SGF）会反复出现。

---

*下一篇 [02. Scheduled Sampling for Transformers](../forcing_02/)：RNN 的逐 token 抛硬币为什么搬不进并行训练的 Transformer——两遍共享参数解码的适配方案。*

{{< forcing_nav >}}
