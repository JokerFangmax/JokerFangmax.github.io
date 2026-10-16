---
title: "Forcing: 04. Diffusion Forcing"
date: 2026-10-14
math: true
tags:
  - "Forcing"
  - "Video Generation"
  - "Diffusion Forcing"
categories:
  - Forcing
series: forcing
weight: 4
description: "Diffusion Forcing（NeurIPS 2024，MIT CSAIL 等）精读：给序列中每个 token 独立采样噪声等级，把 next-token 预测的时间轴 masking 与全序列扩散的噪声轴 masking 统一成一个训练目标，并被证明优化所有子序列似然的 ELBO 下界。采样由 M×T 噪声调度矩阵驱动：Minecraft 上 72 帧训练长度稳定 rollout 2000 帧不发散；Maze2D Large 规划平均回报 159.0 vs Diffuser 123.0；真机水果交换成功率 80%，遮挡干扰下仅降到 76%（逐帧扩散基线 48%）。后续 CausVid / Self Forcing / Rolling Forcing 与 WM 系列全部以此为地基"
paper_id: "2407.01392v4"
---

# 每个 token 都有自己的噪声等级

> **Forcing 系列 · 第 04 篇 · 系列枢纽论文**
> 从这一篇起系列进入扩散分支：[03](../forcing_03/) 的 scheduled sampling 用「替换历史」修补训练-推理差异，本篇用「给历史加噪」把同一件事做成了统一的概率模型；[05](../forcing_05/) 把它推广到非因果 Transformer，[06](../forcing_06/)（CausVid）再把它推向实时。
> 📂 **阅读方式**：正文只留主线；长推导、超参标本、消融全表收折叠块。

## 论文信息

- **原题**：Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion
- **作者**：Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, Vincent Sitzmann（MIT CSAIL / TUM）
- **发表**：NeurIPS 2024
- **阅读版本**：arXiv:2407.01392v4，对应本地收集的 PDF 版本。
- **原文**：[arXiv 摘要页](https://arxiv.org/abs/2407.01392v4) · [PDF](https://arxiv.org/pdf/2407.01392v4)

## 1. 这篇论文解决什么问题

序列生成模型长期分裂成两个阵营，各有对方做不到的能力（论文 Fig. 1）：

| 能力 | next-token 预测（teacher forcing） | 全序列扩散（full-sequence diffusion） |
|------|------|------|
| 可变长度生成 / 无限 rollout | ✅ 天然支持 | ❌ 只能生成定长序列 |
| 连续 token（视频等）长 rollout 稳定性 | ❌ 误差累积、发散 | ✅ 联合细化 |
| 采样期 guidance（引导到高奖励轨迹） | ❌ 前帧必须先定死 | ✅ 支持 |
| 因果不确定性（近未来确定、远未来不确定） | ❌ 无法表达 | ❌ 所有 token 同一噪声级 |

自然的想法是「用因果架构训练全序列扩散」——论文指出这会失败，直觉原因：**它不建模「早 token 的微小不确定性必然迫使晚 token 有高不确定性」**。全序列扩散把所有 token 放在同一个噪声级上，这个约束结构就丢了。

Diffusion Forcing（DF）的答案：**训练时给每个 token 独立采样噪声等级** $k_t \in \{0,1,\dots,K\}$。加噪被理解为一种**连续的 masking**：$k=0$ 是完全可见，$k=K$ 是完全遮蔽，中间值是部分遮蔽。于是「时间轴上的 masking」（teacher forcing 遮当前 token）与「噪声轴上的 masking」（扩散遮全部 token）统一成同一张二维表上的不同取法。

## 2. 最小背景

- **DDPM**（详见 [VG 01：扩散模型基础](../video_generation/VG_01_diffusion_basics/)）：前向过程逐步加高斯噪声

$$
q(x^k \mid x^{k-1}) = \mathcal{N}\!\big(x^k;\ \sqrt{1-\beta_k}\, x^{k-1},\ \beta_k I\big),
\tag{Eq. 2.1}
$$

反向过程由网络参数化 $p_\theta(x^{k-1} \mid x^k)$（Eq. 2.2），训练目标是噪声预测的最小二乘损失

$$
L(\theta) = \mathbb{E}_{k, x_0, \epsilon}\left[ \big\| \epsilon^k - \epsilon_\theta(x^k, k) \big\|^2 \right],
\qquad x^k = \sqrt{\bar\alpha}\, x_0 + \sqrt{1-\bar\alpha}\, \epsilon^k,
\tag{Eq. 2.3}
$$

其中 $\bar\alpha_k = \prod_{j\le k}(1-\beta_j)$。

- **分类器引导 (classifier guidance)**：把 $\epsilon_\theta(x^k, k)$ 替换为 $\epsilon_\theta(x^k, k) - \sqrt{1-\bar\alpha_k}\, \nabla_{x^k} \log c(y \mid x^k)$，即可在不训练条件模型的前提下偏向期望属性 $y$。
- **teacher forcing**：用真值历史预测下一 token 的训练法（[03](../forcing_03/) 的主角）。
- **贝叶斯滤波 (Bayes filtering)**：隐状态 $z_t$ 随观测递归更新——先验 $p(z_{t+1}\mid z_t)$、观测模型 $p(x_t \mid z_t)$、后验 $p(z_{t+1} \mid z_t, x_{t+1})$。DF 的隐状态更新恰好是它的连续插值（见 §3）。

## 3. 方法总览：一张 $M \times T$ 的二维噪声表

**核心记号**：$x_t^k$ 表示时间步 $t$ 的 token 加噪到等级 $k$；序列 $(x_t^{k_t})_{1\le t\le T}$ 里每个 token 的 $k_t$ **互相独立**。

论文的因果实例化 **CDF（Causal Diffusion Forcing）** 用一个 RNN 单元实现，每个单元由两个模型组成：

| 模块 | 形式 | 角色 |
|------|------|------|
| 动力学模型 | $z_t \sim p_\theta(z_t \mid z_{t-1}, x_t^{k_t}, k_t)$ | 读入带噪观测与噪声级，马尔可夫式更新隐状态 |
| 观测模型 | $\epsilon_\theta(z_{t-1}, x_t^{k_t}, k_t)$ | 从隐状态 + 带噪 token 预测噪声（等价于预测干净 $x_t^0$） |

一个关键的统一视角（论文 §3.2 与附录 B.7）：**隐状态更新在先验与后验之间连续滑动**——$k_t=0$ 时 $x_t^{k_t}$ 是干净观测，更新就是贝叶斯滤波的后验更新；$k_t=K$ 时 $x_t^{k_t}$ 是纯噪声、不含任何信息，更新退化为先验 $p_\theta(z_t \mid z_{t-1})$。同一个网络通过 $k_t$ 的取值覆盖了整条谱。

**采样**则由一张 $M \times T$ 的**噪声调度矩阵 (scheduling matrix)** $\mathcal{K} \in [K]^{M\times T}$ 规定：列是时间步 $t$，行 $m$ 是一轮去噪的进度；$\mathcal{K}_{m,t}$ 给出第 $m$ 轮中 token $t$ 应处于的噪声级。采样从全是白噪声（$k=K$）开始，逐行向下、行内从左到右推进，最后一行全为 0。**换一张 $\mathcal{K}$，同一个模型就能表演完全不同的生成协议**（附录 B.8 的统一视角）：

| 采样协议 | 噪声表上的样子（3 个 token、3 步 DDIM 为例） |
|------|------|
| 自回归（teacher forcing 式） | 每次只去噪当前 token，历史保持 $k=0$：$[x_1^0, x_2^0, x_3^K] \to \cdots \to [x_1^0, x_2^0, x_3^0]$，然后滑动窗口 |
| 全序列扩散 | 所有 token 同步去噪：$[x_1^K, x_2^K, x_3^K] \to [x_1^{K/2}, x_2^{K/2}, x_3^{K/2}] \to [x_1^0, x_2^0, x_3^0]$ |
| 因果不确定性（zig-zag） | 近未来先去噪、远未来保持高噪声：$[x_1^{K/2}, x_2^K, x_3^K] \to [x_1^0, x_2^{K/2}, x_3^K] \to [x_1^0, x_2^0, x_3^{K/2}] \to [x_1^0, x_2^0, x_3^0]$ |
| 稳定化长 rollout | 历史以 $k=1$ 的轻微噪声入模：$[x_1^1, x_2^K, x_3^K] \to [x_1^1, x_2^0, x_3^K] \to \cdots$ |

注意表中最后两行是**全序列扩散做不到**的：它的架构里所有 token 必须同噪声级。

## 4. 核心原理完整推导

### 4.1 训练目标：为什么独立噪声可以直接套 DDPM 损失

固定 $z_{t-1}$ 时，CDF 的 RNN 单元**输入输出行为与一个标准条件扩散模型完全相同**：条件变量 $z_{t-1}$ + 带噪输入 $x_t^{k_t}$ + 噪声级 $k_t$ → 预测噪声。所以训练目标就是把每个 token 当独立的条件扩散来训（Algorithm 1）：

$$
\mathbb{E}_{k_{1:T},\, x_{1:T},\, \epsilon_{1:T}}
\left[
\sum_{t=1}^{T}
\big\| \epsilon_t - \epsilon_\theta\!\big(z_{t-1}, x_t^{k_t}, k_t\big) \big\|^2
\right],
\tag{Eq. 3.1}
$$

其中 $k_{1:T}$ 从 $[K]^T$ 均匀独立采样，$\epsilon_t \sim \mathcal{N}(0, \sigma_{k_t}^2 I)$。逐行看 Algorithm 1：采样轨迹 → 每个 $t$ 独立采 $k_t$ 并前向加噪 → 递归更新 $z_t$ → 预测 $\hat\epsilon_t$ → 对全序列做一次 MSE。隐状态取**确定性**实现（$z_t$ 是信念分布的表示而非采样），这样损失可以穿过隐状态动力学反传。

**直觉**：每个训练样本相当于随机抽了一张「噪声表」的一行——有的 token 干净、有的全噪、有的半噪。模型被迫学会在**任意噪声组合**下恢复每个 token，这既是数据增广，也是后面所有灵活采样协议的分布覆盖保证。

### 4.2 理论保证：训练目标优化所有子序列似然的 ELBO

论文的定理把 Eq. 3.1 接到了变分下界上（正文 Theorem 3.1，正式版 Theorem A.1）：

$$
\mathbb{E}_{\text{forward}}\Big[\ln p_\theta\big((x_t^{k_t})_{1\le t\le T}\big)\Big] \ge C\big(x^0_{1:T}\big) + \mathbb{E}_{\text{forward}}\, \mathbb{E}_{p, z_{1:T}}
\left[
\sum_{t=1}^{T}
\left(
\frac{1}{K+1} \ln p_\theta\big(x_t^0 \mid x_t^1, z_{t-1}\big)
+ \sum_{j=2}^{K} \frac{j}{K+1}\,
D_{KL}\Big( q\big(x_t^{j-1} \mid x_t^j, x_t^0\big) \,\Big\|\, p_\theta\big(x_t^j \mid x_t^{j-1}, z_{t-1}\big) \Big)
\right)
\right],
$$

其中 $\mathbb{E}_{\text{forward}}$ 是对 $k_{1:T}$ 均匀采样 + 前向加噪取期望。**每个 token 的 KL 项只通过 $z_{t-1}$ 依赖历史**——这正是马尔可夫性质带来的解耦。

<details>
<summary><b>展开：证明骨架（附录 A.2，只用了两次不等式）</b></summary>

1. **ELBO 技巧**：$\ln p_\theta = \ln \mathbb{E}_{q}\!\left[p_\theta/q\right] \ge \mathbb{E}_q \ln\!\left[p_\theta/q\right]$（Jensen）；
2. **展开联合分布**：把 $\ln(p_\theta/q)$ 沿 $z_{1:T}$ 积分拆开，利用马尔可夫性质 $p_\theta(z_t, x_t^{k_t} \mid z_{1:t-1}, \text{历史}) = p_\theta(z_t, x_t^{k_t} \mid z_{t-1})$，联合对数似然分解为逐 token 的 $\ln p_\theta(x_t^{k_t} \mid z_{t-1})$ 之和；
3. **对噪声级取期望**：每个 $\ln p_\theta(x_t^{k_t} \mid z_{t-1})$ 再对 $k_t$ 均匀采样展开，得到「重建项 + 逐级 KL」的 DDPM 标准形式，系数 $\frac{1}{K+1}$ 与 $\frac{j}{K+1}$ 即均匀采样的重加权。

论文强调整个证明**只用两次不等式**：第一次（Jensen）在确定性隐状态下取等；第二次（变分近似）当且仅当 $q \equiv p_\theta$ 时取等——ELBO 相当紧，是较强的代理目标。

</details>

**特化到高斯扩散**（Corollary A.2）：把逐级 KL 用 DDPM 的标准计算换成加权的 $x_0$ 预测损失，得到

$$
\mathbb{E}_{\text{forward}}\big[\ln p_\theta\big] + C\big(x^0_{1:T}\big) \ge
\mathbb{E}\,\mathbb{E}
\left[
\sum_{t=1}^{T} \mathbb{1}\{k_t \ge 1\} \cdot k_t\, c_{k_t}\,
\big\| \hat x_\theta^0\!\big(x_t^{k_t}, z_{t-1}, k_t\big) - x_t^0 \big\|^2
\right],
$$

其中 $c_j$ 是只依赖噪声级 $j$ 的固定系数（由高斯 KL 计算给出）。到这一步，Eq. 3.1 就是这个加权和**丢掉权重**的版本。

### 4.3 关键一步：为什么权重可以丢

这是全篇理论最漂亮的一步（Corollary A.2 之后的 remark）。设想一个**理想隐状态** $\bar z_{t-1}$：包含全部历史带噪 token $(x_s^{k_s})_{s<t}$ 及其噪声级 $k_{1:t-1}$——它一定是真实 $z_{t-1}$ 的充分统计量。对两边取 $\inf$ over 函数逼近器：

$$
\inf_{\hat x_\theta^0}\ \mathbb{E}\Big[ \big\| \hat x_\theta^0\big(x_t^{k_t}, \bar z_{t-1}, k_t\big) - x_t^0 \big\|^2 \Big]
= \mathrm{Var}_q\Big[ x_t^0 \,\Big|\, (x_s^{k_s})_{s\le t},\ k_{1:t} \Big].
$$

**直觉**：最优预测器的损失就是条件方差——一个与权重系数无关的量。各时间步的最优解互不耦合，所以只要隐状态与网络足够 expressive，各项可以**分别**优化到最优，权重怎么摆都不影响最优点。这就严格 justify 了「均匀采 $k_t$ + 无权重 MSE」的训练协议。

### 4.4 覆盖所有子序列（Theorem A.3）

把 $k_{1:T}$ 的分布从均匀换成任意分布 $\mathcal{D}$（可以时间相关），同样的下界成立，只是各项系数变成 $\mathcal{D}$ 的条件概率 $P_t(j \mid k_{1:t-1})$ 的重加权。特别地，取 $\mathcal{D}$ 为**单点分布**（固定某条噪声序列 $k_1,\dots,k_T$）：

- 训练目标虽然不再是这条特定序列似然的直接下界，但下界里的每一项都是训练目标中项的子集（至多差一个重加权）；
- 由 §4.3，expressive 网络可以同时把所有项优化到位。

**结论**：一次训练，得到的是**所有可能噪声序列似然的有效代理**。最实用的特例：$k_t \in \{0, K\}$ 的二值序列对应「任意遮蔽某些 token」——DF 因此建模了训练集**所有子序列**的联合分布，这是 §6.5 组合生成的来源。

### 4.5 采样：任意窗口去噪（Algorithm 2）

给定调度矩阵 $\mathcal{K}$，第 $m$ 轮对每个 $t$ 执行：

1. 用上一轮噪声级 $\mathcal{K}_{m+1,t}$ 更新隐状态：$z_t^{new} \sim p_\theta(z_t \mid z_{t-1}, x_t, \mathcal{K}_{m+1,t})$；
2. 按目标噪声级 $k \leftarrow \mathcal{K}_{m,t}$ 做 DDIM 式更新：

$$
x_t^{new} \leftarrow \frac{1}{\sqrt{\alpha_k}}
\left( x_t - \frac{1-\alpha_k}{\sqrt{1-\bar\alpha_k}}\, \epsilon_\theta\!\big(z_t^{new}, x_t, k\big) \right) + \sigma_k w,
\qquad w \sim \mathcal{N}(0, I);
$$

3. 每轮结束后可对部分去噪的轨迹加 guidance（见 §4.7）。

规划实验用的**金字塔调度 (pyramid scheduling)**（附录 D.1，Eq. D.1）：近未来的列从 $K$ 单调降到 $0$ 更快，远未来的列始终保持在更高的噪声级——即「近未来确定、远未来不确定」的显式实现。

<details>
<summary><b>展开：调度的边角情形与隐状态复用（附录 D.5 / D.6）</b></summary>

- **噪声级不变的 token 怎么办**：$\mathcal{K}_{m+1,t} = \mathcal{K}_{m,t}$ 时既可以原样复制，也可以「先反向去噪再前向加噪回来」重采样。论文选择后者，并把它用于 Monte Carlo Guidance 的多样本生成；$k=K$ 时保持白噪声无需采样，$k=0$ 时两种做法等价、取复制即可。
- **$z_t$ 直接赋值而非重算后验**（Algorithm 2 第 8 行）：$z_t^{new}$ 已经在噪声级 $k_t$ 下估计过、包含 $x_t^{new}$ 的信息，直接 `zt ← z_new` 省一半计算，且恰好实现了稳定化想要的行为（隐状态比观测「更噪」一档）。

</details>

### 4.6 稳定化长 rollout：噪声当正则（附录 B.4）

自回归视频生成的死穴是**复合误差**：把预测帧当干净观测回灌，模型从未见过这种输入。DF 的解法天然内嵌于训练分布：

1. 生成 token $t$ 时，维护上一时间步在**小噪声级** $0 < k_{small} \ll K$ 下的隐状态；
2. 对 token $t$ 做单 token 扩散（Algorithm 2 取 $T=1$），产生各噪声级的隐状态链 $z_t^K, \dots, z_t^0$；
3. 取 $z_t^{k_{small}}$ 继续下一时间步——等价于**把生成帧加轻噪后当条件**。

为什么合法：训练时模型见过的历史本来就是任意噪声级的带噪 token，所以「轻微噪声的历史」**在分布内**。论文把这个联系精确到理论层面：这是行为克隆中 DART 噪声注入的特例，且因训练时同样使用带噪 token，继承了 HINT 算法的理论保证。这与 [03](../forcing_03/) 的 scheduled sampling 殊途同归——都是让训练分布覆盖推理时会遇到的「不完美历史」，只是 DF 的覆盖是连续的（按噪声等级插值），替换式的覆盖是离散的。

### 4.7 Guidance 与 Monte Carlo Guidance

**长程引导**：由于未来 token 可以在历史保持低噪声的状态下部分去噪，guidance 梯度能从未来 token **反向传播到过去 token**——这是因果架构 + 独立噪声独有的能力，全序列扩散做不到（它必须全部加噪才能引导）。可用的能量函数（附录 B.3）：cost-to-go $c(x_t^k) = \mathbb{E}[\sum_{t' > t} r'(x_{t'}^{k_{t'}}) \mid x_t^k]$（用累计奖励做无偏估计）或目标距离 $c = -\|x_T^K - g\|^2$。

**Monte Carlo Guidance (MCG)**：对当前 token $x_t^k$ 的引导不必只看一条未来轨迹——**采多条未来、平均引导梯度**，即按未来分布的期望奖励引导（MPPI 一类的 shooting 思想）。MCG 与 zig-zag 调度天然互补：远未来保持高噪声 → 未来样本更多样 → 期望奖励估计更准。附录 B.5 论证全序列扩散无法做 MCG：给定噪声级下没有现成的随机源做蒙特卡洛。

## 5. 训练与推理细节

主配置一览（视频实验）：

| 项 | 值 |
|----|-----|
| 骨干 | 扩散 U-Net（帧内）+ GRU（时间递归）+ 1 层 ResNet 观测模型 |
| 参数量 | Minecraft 36M（隐状态 32 通道）/ DMLab 24M（16 通道）/ 迷宫规划 4.33M |
| 预测参数化 | 视频 v-参数化（对收敛与质量「必需」）；规划与模仿学习 $x_0$-参数化 |
| 噪声日程 | 视频 sigmoid、迷宫线性、其余 cosine |
| 扩散步数 | $K = 1000$；采样 DDIM：视频 100 步、非视频 50 步 |
| 帧堆叠 | DMLab 4 帧 / Minecraft 8 帧 / 迷宫 10 帧为一个 token |
| 训练长度 | Minecraft 72 帧（128px）/ DMLab 36 帧（64px） |
| 算力 | 视频 8×A100、50K 步、batch 8×16、约 12 小时于 40K 步收敛；非视频单卡 2080Ti（11GB）即可，4–8 小时 |

<details>
<summary><b>展开：Fused SNR 重加权（附录 D.1）——DF 版 min-SNR</b></summary>

SNR 重加权按信噪比缩放各噪声级的损失。DF 里条件变量 $z_{t-1}$ 本身也含 $x_t$ 的信息（比如确定性系统里 $k_{t-1}=0$ 时 $z_{t-1}$ 足以完全预测 $x_t^0$），原版 SNR 低估了这种「历史贡献的信号」。Fused SNR 的修正：

1. 对当前 token 照常算归一化 SNR 因子 $S_t \in [0,1]$（论文用 min-SNR 策略）；
2. 用指数衰减模型累积历史信号：$\bar S_t = \gamma \bar S_{t-1} + (1-\gamma) S_t$，信号衰减因子 $0 < \gamma < 1$；
3. 独立事件的联合概率组合：$S_t' = 1 - (1 - S_t)(1 - \bar S_{t-1})$，以 $S_t'$ 作为重加权因子。

论文报告该技巧对视频预测收敛「极其有用」，非图像域无增益故不用。

</details>

## 6. 实验与消融

### 6.1 视频预测：无限 rollout 不发散

Minecraft / DMLab（TECO 数据集，仅用约 10% 数据）上与两个**同架构**基线对比：teacher forcing 逐帧扩散、因果全序列扩散。定性结果（Fig. 3）：从 496 到 1000 帧，只有 DF 保持环境一致的 ego-motion；两个基线早已发散，且全序列基线在训练长度内还有帧间跳变。

附录 E.3 的定量补充：**不做滑窗、隐状态从不重置**地连续 rollout——训练长度 72 帧（Minecraft）/ 36 帧（DMLab）的模型稳定 rollout 180 帧（2–5 倍外推），尝试 2000 帧也未观察到崩溃。

### 6.2 规划：D4RL Maze2D（Table 1）

六个环境上 DF 全面领先。核心对比列（Large 环境最说明问题）：

| 方法 | Maze2D Large | 说明 |
|------|------|------|
| Diffuser（PD 控制器执行） | 123.0 ± 6.4 | 生成动作与状态不因果一致，必须丢弃生成动作 |
| Diffuser（直接执行生成动作） | 6.3 ± 2.1 | 直接执行即崩溃 |
| Ours（去掉 MCG） | 142.8 ± 5.6 | 即使无 MCG 仍领先 |
| **Ours（完整）** | **159.0 ± 2.7** | 因果一致 + 期望奖励引导 |

单任务平均 141.7 vs Diffuser 119.5，多任务平均 146.2 vs 129.4。两个消融结论：**MCG 有效**（去掉后下降但保持竞争力）、**因果性有效**（Diffuser 的状态-动作不一致只能靠手工控制器兜底，DF 的动作生成自洽）。

### 6.3 真机模仿学习：记忆 + 抗扰（Fig. 4）

水果交换任务（苹果橙子随机放两个槽、用第三个空中槽交换，须记住初始配置才能决定终态——非马尔可夫）。150 段 Franka 遥操作演示，每段 500–600 帧降采样到 40 帧、每帧捆绑 15 个动作：

| 设定 | 成功率 |
|------|------|
| DF | **80%** |
| DF + 相机遮挡 / 视觉干扰（观测标记为 $k > 0$） | **76%**（仅降 4%） |
| 逐帧扩散基线（无记忆，须把扰动观测当真值） | 48% |

遮挡实验是「噪声即 masking」的直接应用：把可疑观测标成带噪，模型自动转靠先验 $p(z_t \mid z_{t-1})$ 行动。

### 6.4 通用序列模型：时间序列预测

附录 E.1 / Table 2，GluonTS 六数据集 CRPSsum：DF 优于除 ScoreGrad（sub-VP SDE）外的全部先前方法（与 ScoreGrad 总体打平；Wikipedia 上 DF 第四）。要点不在刷榜——而是证明**新训练目标没有牺牲作为通用序列模型的能力**。

<details>
<summary><b>展开：时间序列全表（Table 2，CRPSsum ↓）</b></summary>

| 方法 | Exchange | Solar | Electricity | Traffic | Taxi | Wikipedia |
|------|------|------|------|------|------|------|
| DeepAR | - | 0.336 | 0.023 | 0.055 | - | 0.127 |
| GP-Copula | 0.007 | 0.337 | 0.025 | 0.078 | 0.208 | 0.086 |
| TimeGrad | 0.006 | 0.287 | 0.021 | 0.044 | 0.114 | 0.049 |
| ScoreGrad sub-VP | 0.006 | 0.256 | 0.019 | 0.041 | 0.101 | 0.043 |
| **DF（Ours）** | **0.003** | 0.289 | 0.023 | **0.040** | **0.075** | 0.085 |

主基线 TimeGrad 恰是「teacher forcing 训练的 next-token 扩散模型」——CRPSsum 上两者互有胜负（DF 在 Exchange / Traffic / Taxi 更优，Solar / Electricity / Wikipedia 略逊），统一目标没有付出通用性代价。全部数据集用同一套架构与超参。

</details>

### 6.5 组合生成

十字形轨迹数据集（全部从一角到对角）：开满记忆采样 → 复现十字分布；用无记忆 MPC 短视野采样 → **拼贴出 V 形轨迹**（附录 E.2 / Fig. 7）。「建模所有子序列」的直接演示。

## 7. 与系列其他篇目的关系

| 篇目 | 关系 |
|------|------|
| [03 · Parallel Scheduled Sampling](../forcing_03/) | 同一个「训练-推理历史差异」问题的两种修补：PSS 用离散的历史替换，DF 用连续的噪声插值；DF 的稳定化 rollout（§4.6）因此天然覆盖任意「不完美历史」 |
| [05 · History-Guided Video Diffusion](../forcing_05/) | 把逐帧独立噪声从因果 RNN 推广到**非因果 Transformer**（DFoT），并首次把噪声级变成采样期可调的 guidance 旋钮 |
| [06 · CausVid](../forcing_06/) | DF 的因果 Transformer 实例化 + 蒸馏提速（附录 B.1 的预言成真）；DF 训练目标正是其教师-学生的公共底座 |
| [VG 01 · 扩散基础](../video_generation/VG_01_diffusion_basics/) | Eq. 2.1–2.3 的 DDPM 推导细节 |
| [VG 06 · VideoScene](../video_generation/VG_06_videoscene/) | 蒸馏提采样速度的谱系——DF 用 DDIM 100 步、后续工作沿此线压到少步 |
| [WM 03 · Memory Forcing](../world_memory/WM_03_memory_forcing/) | 其 Chained Forward Training 直接运行在 DF 框架上（逐窗独立噪声级 $t \sim U(0, T_{noise})$）——「把模型自己的预测当带噪条件」正是 §4.6 稳定化的训练期版本 |
| [WM 04 · RELIC](../world_memory/WM_04_relic/) | teacher forcing / diffusion forcing / self-forcing 三者构成的世界模型训练协议谱系，RELIC 把第三者做到实时 |
| [WM 16 · FloWM](../world_memory/WM_16_flowm/) | DFoT（本篇思想的 Transformer 版）是其主基线：MNIST World 150 帧预测 MSE 0.2111 vs FloWM 0.0018（约 100 倍）——DF 系「窗口外即遗忘」的记忆短板在此被定量暴露 |

## 8. 局限与个人评价

- **骨干与规模**：主文实现是 RNN（迷宫 4.33M、视频 36M 参数），作者自认这是最大局限——高分辨率视频需要大 Transformer（附录 B.1 只给了路线图）。没有在互联网规模数据上验证 scaling 行为。这个空白由 [05](../forcing_05/) 和 [06](../forcing_06/) 补上。
- **理论的前提**：「权重不重要」依赖充分 expressive 的隐状态与网络（§4.3 的 $\bar z$ 论证）；实际 RNN 压缩有损，各时间步的项仍有耦合，均匀采样未必是最优的 $k$ 分布（附录 B.2 只做了定性辩护）。
- **评测指标的坑**：附录 D.8 自曝 Maze2D 的 episode reward 奖励「慢走到终点」的策略，Diffuser 的对比可能部分被指标缺陷扭曲；作者建议改用首次到达时间。
- **个人评价**：这篇的持久贡献是**记号层面的**——$x_t^k$ 这个双下标记号把「时间」和「噪声」两根轴正交化之后，teacher forcing、全序列扩散、AR-Diffusion、Rolling Diffusion 全部变成了同一张噪声表上的不同路径（附录 B.8 的统一视角）。后面整个 Forcing 系列的论文（DFoT、CausVid、Self Forcing、Rolling Forcing）都在这张表上做文章。理论部分（所有子序列的 ELBO）在工程上常被引用但很少被认真使用，实际上它给出的「为什么可以丢权重」论证，是这些后续工作敢直接用无权重 MSE 的许可证。

{{< forcing_nav >}}
