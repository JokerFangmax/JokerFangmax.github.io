---
title: "Forcing: 05. History-Guided Video Diffusion"
date: 2026-10-14
math: true
tags:
  - "Forcing"
  - "Video Generation"
  - "History-Guided Video Diffusion"
categories:
  - Forcing
series: forcing
weight: 5
description: "History-Guided Video Diffusion（ICML 2025，MIT 等）精读：DFoT 把逐帧独立噪声推广到非因果 Transformer，历史帧以任意长度、任意子集、任意噪声等级进入注意力窗口，历史因此成为采样期可调节的 guidance——HG-v / HG-t / HG-f 三变体沿时间与频率两轴组合。Kinetics-600 上从全序列模型微调仅花 12.5% 训练成本即得 FVD 4.7（从头训 4.3）；64 帧 rollout 上 HG-f 把 FVD 从 208.0 压到 170.4（FS 基线 1040、SD 247.5）；单图扩出 862 帧长视频，真机抗扰任务成功率 83%"
paper_id: "2502.06764v2"
---

# 把历史帧作为可调节的 guidance

> **Forcing 系列 · 第 05 篇 · Diffusion 与 Self Forcing**
> [04](../forcing_04/) 证明了逐 token 独立噪声的因果版；本篇把它搬进非因果 Transformer，并第一次把「噪声等级」本身变成采样期的 guidance 旋钮。下一篇 [06](../forcing_06/)（CausVid）沿因果方向做实时化。
> 📂 **阅读方式**：正文只留主线；长推导、超参标本、消融全表收折叠块。

## 论文信息

- **原题**：History-Guided Video Diffusion
- **作者**：Kiwhan Song*, Boyuan Chen*, Max Simchowitz, Yilun Du, Russ Tedrake, Vincent Sitzmann（MIT / CMU / Harvard）
- **发表**：ICML 2025
- **阅读版本**：arXiv:2502.06764v2，对应本地收集的 PDF 版本。
- **原文**：[arXiv 摘要页](https://arxiv.org/abs/2502.06764v2) · [PDF](https://arxiv.org/pdf/2502.06764v2)

## 1. 这篇论文解决什么问题

视频扩散模型的条件不只是文本或首帧——**任意长度、任意子集的历史帧**本身是最自然的条件。想对历史做无分类器引导 (classifier-free guidance, CFG)，会遇到两个障碍（§3）：

| 障碍 | 表现 |
|------|------|
| **架构只支持定长条件** | DiT / U-Net 用 AdaLN 或通道拼接注入条件，历史必须编码成定长向量；现有模型只能引导固定且短的历史（首帧、或固定的几帧） |
| **逐帧二值 dropout 训练效果差** | 直觉的修补是训练时随机丢弃历史帧子集；但注意力处理全部 $\lvert T \rvert$ 帧却只有随机子集 $\lvert G \rvert$ 帧贡献损失——token 利用率低，视频越长越糟（消融里 BD 基线 FVD 6.4 vs DFoT 4.3） |

论文的两步走：**DFoT**（Diffusion Forcing Transformer）解决训练侧——历史与生成帧不区分身份，统一为「噪声等级不同的同类帧」，任意历史长度因此天然在训练分布内；**History Guidance（HG）**解决采样侧——把 [04](../forcing_04/) 的噪声等级当作连续的 masking 旋钮，沿**时间轴**（用哪些历史子集）与**频率轴**（历史加多强的噪声）组合出一系列引导方法。

一句话：**历史不再是输入接口，而是一个可以在采样时随意调节的引导项。**

## 2. 最小背景

- **CFG**（详见 [VG 01](../video_generation/VG_01_diffusion_basics/)）：联合训练条件/无条件分数，采样时用加权分数

$$
\nabla \log p^k(x) + \omega \big( \nabla \log p^k(x \mid c) - \nabla \log p^k(x) \big), \qquad \omega \ge 1,
\tag{Eq. 1}
$$

- **噪声即 masking**（[04](../forcing_04/) §3）：前向扩散把第 $t$ 帧变成 $x_t^{k_t}$，$k_t \in [0,1]$——$k=0$ 干净可见，$k=1$ 完全遮蔽，中间值保留带噪快照。本篇把这个思想从因果 RNN 移植到非因果 Transformer。
- **视频扩散模型 (VDM)** 的常规条件注入：AdaLN / 通道拼接 / 独立编码器——这正是定长约束的来源。

## 3. 方法总览

**DFoT 的统一输入**：记 $T = \{1,\dots,T\}$ 为帧索引，历史帧索引 $H \subset T$，生成帧索引 $G = T \setminus H$。条件生成 $p(x_G \mid x_H)$ 被统一为「对整段序列去噪、历史帧噪声级取 0」：

$$
k_t = \begin{cases} 0 & t \in H \\ k & t \in G \end{cases}
\tag{Eq. 3}
$$

历史与生成帧**同一套 token、同一个注意力窗口**——不需要独立的条件编码器，任何全序列 Transformer（DiT、U-ViT）都能直接微调得到。训练时则更进一步，每个帧独立采 $k_t \sim [0,1]$（不分过去未来），最小化噪声预测损失：

$$
\mathbb{E}_{k_T,\, x_T,\, \epsilon_T}\left[ \big\| \epsilon_T - \epsilon_\theta\big(x_T^{k_T}, k_T\big) \big\|^2 \right]
\tag{Eq. 4}
$$

两个直接收益（相对常规 VDM 训练）：**所有帧都算损失**（token 利用率），以及**任意历史长度都在训练分布内**（每个 $k_t$ 独立采，$k \in \{0,1\}$ 的二值特例覆盖任意子集）。论文 Theorem 4.1 + 附录 A.1 证明该目标优化期望对数似然的一个（重加权）ELBO——推导见 §4.1。

**History Guidance 家族**：把 CFG 的「条件分数 − 无条件分数」推广为多个条件分数的加权和，每个条件可以 用不同的历史子集 $H_i$（时间轴）且 该子集带不同的噪声级 $k_{H_i}$（频率轴）：

| 变体 | 条件构造 | 解决什么 |
|------|------|------|
| **HG-v**（vanilla） | 任意长度历史 $H$，$k_H = 0$，直接套 CFG（Eq. 2） | 单一旋钮同时提升质量与一致性 |
| **HG-t**（temporal） | 组合多个历史子窗 $\{H_i\}$ 的分数（$k_{H_i} = 0$） | 长历史易 out-of-distribution（OOD）——改用多个分布内短窗的分数之和近似长窗 |
| **HG-f**（fractional） | 同一历史但加噪到 $k_H \in (0,1)$ 再作条件（Eq. 6） | HG-v 在高引导强度下生成「贴着历史不动」的静态视频——部分遮蔽只保留低频信息，高频细节与快速运动不被约束 |
| **HG-tf** | 时间 × 频率两轴同时组合 | 论文的完整形态，用于超长视频 |

## 4. 核心原理完整推导

### 4.1 训练目标为什么合法：路径视角的 ELBO

[04](../forcing_04/) 的 ELBO 是序列视角、带因果隐状态的；DFoT 换成更一般的**路径 (path)** 视角（附录 A.1），把因果性从证明里剥离：

- **Definition A.1（路径）**：一条路径 $\rho = (k^j)_{0\le j\le N}$ 是从零噪声 $k^0 = (0,\dots,0)$ 到全噪声 $k^N = (K,\dots,K)$ 的噪声级序列。自回归扩散、全序列扩散、DDIM 跳步……都是特定路径。
- **Definition A.2（因式分解）**：前向 $q$ 与模型 $p_\theta$ 沿路径逐级分解：$q(x^{k_{1:N}} \mid x^{k_0}) = \prod_j q(x^{k_j} \mid x^{k_{j-1}})$，$p_\theta(x^{k_{0:N}}) = \prod_j p_\theta(x^{k_{j-1}} \mid x^{k_j})\, p_\theta(x^{k_N})$。DFoT 对所有单调路径满足此性质。

**Theorem A.3**：只要 $(p_\theta, q)$ 沿路径因式分解，

$$
\ln p\big(x^{k_0}\big) \ge C + \mathbb{E}_{x^{k_{1:N}} \sim q}
\left[
\ln p_\theta\big(x^{k_0} \mid x^{k_1}\big)
+ \sum_{j=1}^{N-1} D_{KL}\Big( p_\theta\big(x^{k_j} \mid x^{k_{j+1}}\big) \,\Big\|\, q\big(x^{k_j} \mid x^{k_{j+1}}, x^{k_0}\big) \Big)
\right],
\tag{Eq. 7}
$$

再对路径分布 $\mathcal{D}_\rho$ 取期望即覆盖「每帧独立采噪声级」的训练目标。

<details>
<summary><b>展开：证明骨架与 Corollary A.4</b></summary>

1. **ELBO 技巧**：$\ln p(x^{k_0}) = \ln \mathbb{E}_{q}\!\left[p(x^{k_{0:N}})/q(x^{k_{1:N}} \mid x^{k_0})\right] \ge \mathbb{E}_q \ln\!\left[\cdot\right]$（Jensen）；
2. **因式分解展开**：把对数里的联合 $p_\theta$、$q$ 按路径逐级拆开，逐 token 的 KL 项涌现——注意 DFoT 没有 $z_t$，条件信息全部经由「其他帧的当前噪声级」进入，分解因此比 [04] 更干净；
3. **特化到高斯（Corollary A.4）**：对逐级加一档噪声的路径，KL 项用 DDPM 标准计算换成加权的 $x_0$ 预测损失：$\sum_j \sum_{t:\, k_t^j < k_t^{j+1}} c_{k_t^j} \big\| \hat x_\theta^0(\cdot)_t - x_t^{k_0} \big\|^2$，其中 $c_i$ 为只依赖噪声级的系数。约定「噪声级没变的帧不重新预测」。
4. **Remark A.5 的诚实声明**：前向过程天然沿所有路径分解，但 $p_\theta$ 在不同路径上的因式分解**未必彼此一致**——跨路径的一致性约束是开放问题。这与 [04] §4.3 的「expressive 网络下权重无关」论证同一层面：理论许可证，不是工程保证。

</details>

### 4.2 History Guidance 的三种变体：从 CFG 到分数组合

起点是把 CFG（Eq. 1）搬到历史条件上（Eq. 2）：

$$
\nabla \log p^k\big(x_G^k\big) + \omega \Big( \nabla \log p^k\big(x_G^k \mid x_H\big) - \nabla \log p^k\big(x_G^k\big) \Big)
\tag{Eq. 2}
$$

关键观察：**无条件分数就是 $H = \emptyset$ 的条件分数**——把历史帧全部换成白噪声即可，无需 dropout 训练。HG-v 就是 Eq. 2 配任意长度 $H$。

推广为多个条件的加权和（Eq. 5）：

$$
\nabla \log p^k\big(x_G^k\big) + \sum_i \omega_i \Big( \nabla \log p^k\big(x_G^k \mid x_{H_i}^{k_{H_i}}\big) - \nabla \log p^k\big(x_G^k\big) \Big)
\tag{Eq. 5}
$$

- **HG-t**：取 $k_{H_i} = 0$、$\{H_i\}$ 为多个短窗。动机是维度诅咒——保证恒定数据支撑所需的样本量随条件长度指数增长，长历史必然 OOD；每个短窗分数在自己的受限上下文上更接近分布内（组合生成文献的思路）。
- **HG-f**：取同一 $H$、但条件帧加噪到 $k_H \in (0,1)$：

$$
\nabla \log p^k\big(x_G^k \mid x_H\big) + \omega \Big( \nabla \log p^k\big(x_G^k \mid x_H^{k_H}\big) - \nabla \log p^k\big(x_G^k\big) \Big)
\tag{Eq. 6}
$$

直觉：强引导下的平凡解是「复制最近历史帧」；把条件帧部分遮蔽，只约束其低频（结构、布局），把高频（纹理、快速运动）留给生成先验自由发挥。

**采样算法（Algorithm 1）有一个极容易做错的实现细节**：估计任一条件分数时，必须 (a) 把该分数对应的噪声级向量 $k_T$（生成帧 $n/N$、被遮蔽历史为 1、部分遮蔽历史为 $k_{H_i}$）逐 token 传给网络，并且 (b) **把干净历史帧真的前向加噪到对应等级**（$x_{H_i} \leftarrow \alpha_{k_{H_i}} x_{H_i} + \sigma_{k_{H_i}} \epsilon$、其余历史换白噪声）——保证模型输入与训练时「任意噪声组合」的分布一致。这一步就是条件分数估计的稳定化：跳过它，历史帧的「干净程度」与训练分布错配，分数估计直接失效。无条件分数、各条件分数沿 batch 维并行计算后按 Eq. 5 组合，交给任意采样器（论文用 DDIM 50 步）。

<details>
<summary><b>展开：Algorithm 1 的四种实例化（附录 A.4）</b></summary>

| 任务 | 引导配置 $\{(H_i, k_{H_i}, \omega_i)\}_i$ |
|------|------|
| 无引导条件采样 | $\{(H, 0, 1)\}$ |
| HG-v（强度 $\omega > 1$） | $\{(H, 0, \omega)\}$ |
| HG-t（$I$ 个子窗） | $\{(H_i, 0, \omega_i)\}_{i=1}^I$ |
| HG-f（强度 $\omega$、遮蔽级 $k_H$） | $\{(H, 0, 1),\ (H, k_H, \omega - 1)\}$ |

外推：$H = \{1,\dots,\tau\}$、$G = \{\tau+1,\dots,T\}$；插值：$H = \{1, T\}$、$G = \{2,\dots,T-1\}$——同一模型同一次前向协议，只改索引集合。

</details>

### 4.3 频率轴的理论：加噪即低通滤波（Prop A.7）

HG-f 的「保留低频」说法需要证明。论文在回归设定下算闭式解：特征 $x \sim \mathcal{N}(0, \Sigma_x^2)$、目标 $y \mid x \sim \mathcal{N}(Ax, \Sigma_y^2)$、观测加噪 $x_\sigma = x + \sigma z$。在傅里叶域（$\mathcal{F}_d$ 为 DFT，Parseval 下是等距映射），条件分布为

$$
\mathcal{F}_m(y) \mid x_\sigma \sim \mathcal{N}\big( \hat A \hat S(\sigma)\, \mathcal{F}_d(x_\sigma),\ \hat\Sigma_y + d\sigma^2 \hat A \hat S(\sigma) \hat A^\top \big),
\qquad \hat S(\sigma) := \hat\Sigma_x \big(\hat\Sigma_x + d\sigma^2 I\big)^{-1},
\tag{Eq. 11}
$$

其中 $\hat A = \mathcal{F}_m A \mathcal{F}_d^{-1}$ 等为频域算子。$\hat S(\sigma)$ 是**收缩算子**：若自然信号在频域呈幂律衰减（$\hat\Sigma_x = C\,\mathrm{Diag}(\{i^{-\alpha}\})$），则

$$
\hat S(\sigma)_{ii} \sim \begin{cases} 1 & i \lesssim (C / d\sigma^2)^{1/\alpha} \\ i^{-\alpha} & i \gtrsim (C / d\sigma^2)^{1/\alpha} \end{cases}
$$

即**加噪把弱频率正则化掉**，$\sigma$ 越大衰减越激进。对自然图像弱频率恰好是高频，所以「加噪 = 低通」；论文强调这不是高斯噪声偏爱低频，而是**滤掉信噪比低的频段**——换个域，被滤的可能是别的频段。

<details>
<summary><b>展开：高斯条件闭式解（Lemma A.8）</b></summary>

$(x_\sigma, y)$ 联合高斯：$\Sigma_{22} = \sigma^2 I + \Sigma_x$，$\Sigma_{12} = A\Sigma_x$。标准条件分布公式给出

$$
y \mid x_\sigma \sim \mathcal{N}\big( A S(\sigma)\, x_\sigma,\ \Sigma_y + \sigma^2 A S(\sigma) A^\top \big), \qquad S(\sigma) := \Sigma_x (\Sigma_x + \sigma^2 I)^{-1},
\tag{Eq. 12}
$$

协方差项用恒等式 $\Sigma_x - \Sigma_x(\Sigma_x + \sigma^2 I)^{-1}\Sigma_x = \sigma^2 \Sigma_x(\Sigma_x + \sigma^2 I)^{-1}$ 化简。转到频域时注意 $\tfrac{1}{\sqrt d}\mathcal{F}_d$ 正交 ⇒ $\sigma \mathcal{F}_d(z) \stackrel{d}{=} \sigma\sqrt d\, \hat z$，这解释了 $d\sigma^2$ 的出现。$S(\sigma)$ 正是维纳滤波 / 贝叶斯收缩的形式：噪声越大，观测被收缩得越狠。

</details>

### 4.4 分数相加的极大似然解释（附录 A.3）

Eq. 5 的分数线性组合看起来纯启发式。论文给了一个概率解释：设真分数 $s^\star(x) = \nabla_x \ln q(x)$，有 $n$ 个估计器 $\hat s_i(x)$，并**假设**（Assumption A.9）估计误差在给定 $x$ 时服从零均值高斯、协方差 $\Sigma(x)$。则 $\hat s_{1:n}(x) \mid x \sim \mathcal{N}(\mathbb{I}\, s^\star(x), \Sigma(x))$（$\mathbb{I}$ 为 $n$ 个单位阵堆叠），极大似然估计为广义最小二乘：

$$
\hat s_{\mathrm{MLE}}(x) = \big( \mathbb{I}^\top \Sigma(x)^{-1} \mathbb{I} \big)^{-1} \mathbb{I}^\top \Sigma(x)^{-1}\, \hat s_{1:n}(x),
\tag{Eq. 15}
$$

即**真分数的 MLE 是各估计器的（$x$ 依赖的）线性组合**。误差独立、协方差为缩放单位阵时退化为凸组合 $\hat s_{\mathrm{MLE}} = \sum_i \frac{\alpha_i}{\sum_j \alpha_j} \hat s_i$。history guidance 因此可以读作：把 $n$ 段历史证据当作 $n$ 个分数估计器，用 MAP 方式融合。

<details>
<summary><b>展开：为什么平均比挑最好的强（论文的反例构造）</b></summary>

设 $n$ 个估计器 $\hat s_i: \mathcal{X} \to [0,1]$，各自在状态空间的 $1/n$ 分区 $\mathcal{X}_i$ 上犯错（$\hat s_i = \mathbb{1}[x \in \mathcal{X}_i]$，真值为 0）。任一估计器的期望平方误差 $\mathbb{E}[\hat s_i^2] = 1/n$；而平均 $\tfrac1n\sum_i \hat s_i \equiv \tfrac1n$，误差 $1/n^2$——**误差在互补区域上相互抵消**。论文猜测 HG-t 的收益同源：不同历史上下文的分数各自在不同实现上犯错，求和摊薄总误差。此为直觉论证（高斯误差假设在实践不真），但与估计器最优聚合的一般理论（凸包上优化优于单点选择）一致。

</details>

### 4.5 稳定化与超长 rollout

长 rollout 用到三层技术（附录 C.9）：

1. **HG-f 为默认引导**（$\omega = 4$、$k_H = 0.4$）——部分遮蔽的历史让 rollout 极稳；
2. **遇硬场景切 HG-v**（$\omega = 4$）——需要「外推到新区域」时 HG-v 更强但稳定性差，故仅当导航方向变化超 30° 或移动距离超阈值时切换；
3. **[04](../forcing_04/) 的稳定化技巧**——已生成帧标记为轻微噪声 $k = 0.02$ 后再入窗，防误差累积（DF 训练分布内操作，见 [04](../forcing_04/) §4.6）。

插值阶段（对相邻生成帧之间插 6 帧、共 ×7）用小强度 HG-v（$\omega = 1.5$）保平滑。另外附录 A.8 的**扩展 HG-t**：把生成本身也拆成子序列 $\{G_j\}$，组合 $\prod_j \sum_i \nabla \log p^k(x_{G_j}^k \mid x_{H_i})$（$L$ 为逐帧平均）——OOD 不只来自历史，整段输入序列都可能 OOD。

## 5. 训练与推理细节

| 项 | Kinetics-600 | RealEstate10K | Minecraft | 机器人模仿 |
|----|------|------|------|------|
| 分辨率 | 128²（VideoVAE → 16²×16ch，每 4 帧一 chunk） | 256²（**像素空间**） | 256²（ImageVAE → 32²×4ch） | 21×32² |
| 序列长度 | 17 帧（5 latent） | 8 帧 | 50 帧 | 21 帧（每帧捆 15 动作） |
| 骨干 | DiT，28 层，hidden 1152，16 头 | U-ViT [3,3,6,20]，hidden 至 1152 | DiT，12 层，hidden 768 | Attention UNet，8 层 |
| 参数化 / 日程 | v / cosine | v / cosine + shift 0.125 | v / cosine + shift 0.125 | $x_0$ / cosine |
| 采样 | DDIM 50 步 | 同 | 同 | 同 |

共性：3D 注意力（不分时空）、3D RoPE、所有条件（**逐 token 的噪声级**、动作 one-hot、相机位姿的 180 维 NeRF 式嵌入）经 AdaLN 注入——DFoT 的每个 token 携带自己的噪声级嵌入。损失重加权：Kinetics 与机器人用 min-SNR，RE10K/Minecraft 用 sigmoid。训练 12×H100、每模型约 5 天；机器人模型 4×RTX4090、4 小时。DFoT Kinetics 模型 673M 参数（DiT/XL 量级）。

<details>
<summary><b>展开：微调现有模型成 DFoT（§6.2 Q4 / 附录 D.1）</b></summary>

从预训练的全序列模型（FS）微调成 DFoT 只花 **12.5% 的从头训练成本**：

- 80k 步微调的训练损失已低于 640k 步从头训练；40k 步时与从头训 405k 步相当（约 **10× 提速**）；FVD 上微调版 4.7 略逊于从头 4.3——论文归因于 EMA（0.9999）让采样权重受最近数万步影响，属暂态；
- EMA 不可省：80k 步时无 EMA 的 FVD 为 7.3 vs 有 EMA 的 4.7。

这条路线意味着**现有视频基础模型可以低成本升级出 history guidance 能力**——论文明确把「微调大模型 + HG」列为未来方向（[06](../forcing_06/) CausVid 的蒸馏路线是另一条，见 [VG 06](../video_generation/VG_06_videoscene/) 的蒸馏谱系）。

</details>

<details>
<summary><b>展开：数据集与基线设定</b></summary>

- **Kinetics-600**：约 400K 视频、600 类；基准任务为给前 5 帧预测后 11 帧（模型生成 3 个 latent token = 12 帧、去尾 1 帧），50K 条 16 帧视频、3 个种子算 FVD；
- **RealEstate10K**：室内 walkthrough、带相机位姿；latent 扩散跟不住位姿条件，故用像素空间；**Minecraft**：200K 条 × 300 帧、3 个动作（前进/左/右），设计上**只有长上下文才能拿到好 FVD**（含大量折返重访轨迹）；
- **三个基线**：**SD**（按测试历史长度定制的标准条件扩散）、**BD**（逐帧二值 dropout——即「最直觉的历史 CFG 训练法」，需 DFoT 架构才支持任意长度，实为消融）、**FS**（无条件全序列 + replacement/reconstruction guidance，Ho et al. 2022b 的做法）。

</details>

## 6. 实验与消融

### 6.1 基础模型：DFoT 本身就是强通用模型（Table 1）

Kinetics-600 基准（FVD，同架构管线内对比）：

| 方法 | 灵活历史 | FVD ↓ |
|------|------|------|
| W.A.L.T（工业基线，batch 512） | ✗ | 3.3 ± 0.1 |
| **DFoT（scratch）** | ✔ | **4.3 ± 0.1** |
| MAGVIT-v2（工业基线） | ✗ | 4.3 ± 0.1 |
| **DFoT（从 FS 微调，12.5% 成本）** | ✔ | 4.7 ± 0.0 |
| SD（专为该历史长度训练） | ✗ | 4.8 ± 0.0 |
| BD（二值 dropout） | ✔ | 6.4 ± 0.1 |
| Rolling Diffusion（1.2B 参数） | ✗ | 5.2 |
| FS | ▲ | 95.5 ± 0.4 |
| Video Diffusion（1.1B 参数） | ▲ | 16.2 ± 0.3 |

两个结论：(1) DFoT **不牺牲**任务专项性能还白送灵活性（打平 MAGVIT-v2、超过为其定制的 SD）；(2) **BD 的失败坐实了 §1 的判断**——同为「支持任意历史」的训练法，二值 dropout 比 DFoT 差 2 个点，且 rollout 下差距更大（下节）。工业基线里 W.A.L.T / Rolling Diffusion 的参数量（1.1B/1.2B）与 batch（512）都显著大于本篇 673M、batch 192 的设定。

### 6.2 History Guidance：质量-动态的三角（Fig. 5/6）

64 帧滑窗 rollout（条件最近 2 个 latent token、生成 3 个，1,024 条视频）：

| 方法 | 最优 FVD ↓ |
|------|------|
| FS | 1040 |
| SD | 247.5 |
| BD + HG-v | 196.0 |
| DFoT（无引导，$\omega = 1$） | 208.0 |
| DFoT + HG-v（$\omega = 1.5$） | 181.6 |
| **DFoT + HG-f（$\omega = 1.5$ 档 + $k_H = 0.8$）** | **170.4** |

- **HG-v**：引导强度升高，帧质量与一致性（VBench）单调改善，但 FVD 在 $\omega > 1.5$ 后回升（多样性损失）；$\omega \ge 3$ 时出现静态视频——模型学会复制最近历史帧；
- **HG-f**：同样的强度下动态程度随 $k_H$ 升高而恢复，把最优 FVD 再压到 170.4——**质量、一致性、动态三者首次同时守住**；BD 配 HG-v 全程劣于 DFoT（最优 196.0 vs 181.6；BD 无引导 270.8），HG-f 只对 DFoT 可用（BD 的训练方式给不出「部分遮蔽的历史」分数）。

### 6.3 HG-t 的新能力（§6.4 三个任务）

**任务 1：OOD 历史（RE10K 大转角插值）**。按历史内相机转角分箱：训练场景极少超过 100°，100–140° 为轻微 OOD（< 500 场景）、≥ 140° 为 OOD（< 100 场景）。基线（FS/SD/DFoT/DFoT+HG-v）的 LPIPS 从分布内到 OOD 急剧恶化、生成不可辨认；**DFoT + HG-t 几乎无衰减**——配置：$H_1 = \{0,1,2\}$、$H_2 = \{1,2,3\}$ 各 $\omega = 2$，扩展版再组合生成子序列 $G_1, G_2$。把 OOD 长窗分解为分布内短窗的分数之和，正是 §4.4「估计器互补」的用武之地。

**任务 2：长上下文生成（Minecraft）**。50 帧模型（按附录 A.5 简化目标只训到 25 帧上下文 = 原视频 100 帧），自回归 rollout 5 次共 125 帧：全上下文条件模型 FVD 97.63，长（25）+ 短（4）上下文分数各 50% 组合后 **79.19**。

**任务 3：长程且反应式的模仿学习（真机水果交换）**。数据集只有「换水果」与「重新抓取」两类各 300 段演示、从不并存——纯记忆模型过拟合（被抓走的水果还往原位置抓），纯反应模型没有记忆。DFoT 组合全历史（权重 0.2）+ 1 帧、4 帧（各 0.45）三个分数：**成功率 83%**（100 次随机化），Markov / 2 帧 / 全历史基线全部失败；对抗性更强的「打掉手里的水果」扰动下仍有 28%。

### 6.4 超长视频：862 帧（Fig. 1）

单图起步、用户经 UI 逐窗给导航指令，rollout（条件 4 帧 → 生成 4 帧）+ ×7 插值两阶段，在 RE10K 上稳定生成 **862 帧**导航视频——训练集视频普遍不足 300 帧、且只覆盖单房间。对照：同样任务下 SD 在约 30 帧附近失守，DFoT + HG 稳过 72 帧；文献里同数据集最长的 4DiM 也只到 32 帧。§4.5 的三层稳定化技术全部启用。

## 7. 与系列其他篇目的关系

| 篇目 | 关系 |
|------|------|
| [04 · Diffusion Forcing](../forcing_04/) | 直接前驱：噪声即 masking、逐帧独立噪声、ELBO 与稳定化技巧全部继承；本篇的贡献是**非因果、无状态**版本 + 把噪声级变成 guidance 旋钮。注意「用了 Diffusion Forcing ≠ 用了因果注意力」——比较架构时必须保留这个区分 |
| [06 · CausVid](../forcing_06/) | 因果方向的姐妹篇（论文原文点名）：DF 思想 + 因果 Transformer + KV cache 实时化；本篇附录 A.6 指出两者不冲突，且非因果 DFoT 也能靠「未来全噪」实现因果采样 |
| [03 · Parallel Scheduled Sampling](../forcing_03/) | 历史条件的「不完美化」谱系：PSS 离散替换、DF 连续加噪、本篇把加噪再升级为可调引导 |
| [WM 03 · Memory Forcing](../world_memory/WM_03_memory_forcing/) | 训练协议同族：Hybrid Training / Chained Forward Training 都运行在 DF 式逐窗独立噪声上（「把模型预测当带噪条件」）；本篇则把同类操作全部搬到**采样期**——一个改训练分布、一个改采样分数，互补而非替代 |
| [WM 04 · RELIC](../world_memory/WM_04_relic/) | teacher / diffusion / self-forcing 谱系中的蒸馏端；本篇的「FS → DFoT 12.5% 成本微调」是同一「复用预训练全序列模型」思想的非蒸馏版 |
| [WM 16 · FloWM](../world_memory/WM_16_flowm/) | DFoT 是其主基线：MNIST World 150 帧预测 MSE 0.2111 vs FloWM 0.0018（约 100 倍）——本篇的「窗口内历史」在出窗即遗忘的记忆任务上的短板被定量放大 |
| [VG 01 · 扩散基础](../video_generation/VG_01_diffusion_basics/) | CFG 与分数匹配的基础 |
| [VG 06 · VideoScene](../video_generation/VG_06_videoscene/) | 蒸馏与采样加速谱系；本篇微调路线与 CausVid 蒸馏路线是「让大模型获得 DF 能力」的两条平行路径 |

## 8. 局限与个人评价

- **理论是许可证不是保证**：分数相加的 MLE 解释依赖高斯误差假设（论文自认不真）；ELBO 的跨路径一致性（Remark A.5）开放；HG 的权重 $\omega_i, k_H$ 全靠逐任务手调——OOD 任务 $\omega = 2$、Minecraft 50/50、机器人 0.2/0.45/0.45、长视频 $\omega = 4, k_H = 0.4$，没有统一的自动选择机制。
- **每步多次前向**：HG 每个采样步要算无条件 + 各条件分数（沿 batch 并行），引导配置越复杂计算越贵；论文未报引导开销的分解。
- **评测口径要小心**：Table 1 的 FVD（3–16 量级，16 帧单窗、50K 样本）与 rollout 实验（170–1040，64 帧滑窗、1,024 样本）是两套协议，直接横比数字会得出荒谬结论。
- **个人评价**：本篇把 [04](../forcing_04/) 留下的最重要的未竟之事做完了——**证明这套训练目标在主流 Transformer 架构与真实 benchmark 上不仅不亏、还是免费增益**。「历史帧以不同噪声等级进入注意力窗口」这个视角的价值超出视频生成：它示范了「条件的强度」可以是一个连续、可组合、采样期可调的维度，而不是「给/不给」的二值开关。后续 Rolling Forcing 等工作把「历史怎么噪」进一步变成训练期课程，是对这条轴的再次开采。真机实验（83% vs 全部基线失败）是全系列里「分数组合解决 OOD」论证最干净的证据。

{{< forcing_nav >}}
