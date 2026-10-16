---
title: "Forcing: 15. Causal Forcing++"
date: 2026-10-14
math: true
tags:
  - "Forcing"
  - "Video Generation"
  - "Causal Forcing++"
categories:
  - Forcing
series: forcing
weight: 15
description: "帧级自回归 + 1–2 步采样把少步学生的初始化推成关键瓶颈：双向教师 ODE 初始化目标错配，直接用多步 AR 模型缺少步能力，因果 ODE 初始化又要 11,600 A800 GPU·时和 1,900 GiB 轨迹存储。Causal Forcing++ 用因果一致性蒸馏（causal CD）的在线相邻时间步单步监督替换：Stage 2 成本降到 2,900 GPU·时、零额外存储（约 4×），帧级 2 步 VBench Total 84.14 超过 chunk 级 4 步 Causal Forcing 的 84.04，首帧延迟 0.27 秒（降低 50%）"
paper_id: "2605.15141v4"
---

# 用在线一致性蒸馏初始化因果学生

> **Forcing 系列 · 第 15 篇 · 少步蒸馏与初始化**
> 论文：*Causal Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation for Real-Time Interactive Video Generation*（arXiv:2605.15141v4，技术报告）。
> 前置：[13 Causal Forcing](../forcing_13/)（因果 ODE 初始化——本篇要替换的对象）、[06 CausVid](../forcing_06/)（两阶段蒸馏管线）、[07 Self Forcing](../forcing_07/)（自 rollout 训练）；扩散与一致性蒸馏基础见 [VG 01](../video_generation/VG_01_diffusion_basics/)、[VG 06 VideoScene](../video_generation/VG_06_videoscene/)。
> 本篇所有数字按论文 v4 正文与表格逐条核对。
>
> 📂 **阅读方式**：正文只保留主线——结论、关键公式与总结表；长推导、超参标本与消融全表收进折叠块（点击展开）。

---

## 论文信息

- **原题**：Causal Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation for Real-Time Interactive Video Generation
- **阅读版本**：arXiv:2605.15141v4（技术报告），对应本地收集的 PDF 版本。
- **原文**：[arXiv 摘要页](https://arxiv.org/abs/2605.15141v4) · [PDF](https://arxiv.org/pdf/2605.15141v4)

---

## 1. 这篇论文解决什么问题

### 1.1 更激进的设定，卡在初始化上

现有 AR 视频扩散蒸馏（CausVid、Self Forcing、Causal Forcing）都停在 **chunk 级（3 个 latent 帧）+ 4 步采样**这个舒适区。本篇推向更激进的低延迟设定：**帧级自回归 + 1–2 步采样**。帧级意味着自回归调用次数翻 3 倍、每次调用的近似误差被自 rollout 复合放大——这时非对称 DMD（asymmetric DMD，教师/评论家保持双向、学生因果）对学生初始化变得高度敏感，而三种现有初始化恰好以三种方式失败：

| 初始化策略 | 代表方法 | 失败模式 |
|-----------|---------|---------|
| 双向教师 ODE 蒸馏 | CausVid、Self Forcing | **架构错配**：违反帧级单射性（frame-level injectivity），同一噪声帧在不同未来上下文下对应多个干净帧，回归目标坍缩为条件期望 $\mathbb{E}[x_0^i\mid x_t^i, x_t^{<i}, t]$，得到模糊初始化；帧级 1 步设定下灾难性崩溃 |
| 直接用多步 AR 扩散模型 | LiveAvatar、WorldPlay | **缺少步能力**：每帧近似误差在自 rollout 中被严重放大，帧级 1 步设定下近乎崩溃（Dynamic Degree = 0） |
| 因果 ODE 蒸馏（AR 教师） | [Causal Forcing](../forcing_13/) | **正确但太贵**：每个训练样本都要 AR 教师生成完整多步 PF-ODE 轨迹（如 48 步），离线成对存储，教师/数据/配置一变就要重新生成 |

一句话：帧级少步设定需要的初始化必须**同时**是 AR 的、少步的、可扩展的——现有三者各缺一角。

### 1.2 答案与达到的数字

Causal Forcing++ 的答案是把 Stage 2 换成**因果一致性蒸馏（causal consistency distillation, causal CD）**：它与因果 ODE 蒸馏学习**同一个目标**（AR 条件 flow map），但监督来自真实视频上相邻时间步之间的**单次在线教师 ODE 步**，完全不需要预计算与存储轨迹。数字上：Stage 2 成本从 ~11,600 降到 ~2,900 A800 GPU·时（约 4×），辅助存储从 ~1,900 GiB 降到零；帧级 2 步生成 VBench Total **84.14**，超过 chunk 级 4 步 Causal Forcing 的 84.04，首帧延迟 0.27 s（降低 50%）。

---

## 2. 最小背景

- **三阶段管线术语**（本篇沿用）：Stage 1 = 教师强迫（teacher forcing）训练多步 AR 扩散模型；Stage 2 = 少步学生初始化；Stage 3 = 自 rollout 的非对称 DMD。三者分别来自 CausVid 的两阶段框架与 [Self Forcing](../forcing_07/)、[Causal Forcing](../forcing_13/) 的修正（见 §4.1）；
- **流匹配（flow matching）**：噪声日程取 $\alpha(t)=1-t$、$\sigma(t)=t$，模型预测速度 $v_\theta$，目标 $v_t=\epsilon-x_0$；解 PF-ODE $ \mathrm{d}x_t = v_\theta(x_t,t)\,\mathrm{d}t$（$t:1\to0$）采样。基础见 [VG 01](../video_generation/VG_01_diffusion_basics/)；
- **一致性模型（consistency model）**：训练一个端点预测器，使教师轨迹上不同状态的输出一致，从而支持少步/单步生成。本篇的核心操作是把它**提升（lift）到 AR 条件设定**；
- 无需世界模型背景——动作条件扩展（§6.5）只是管线复用。

---

## 3. 方法总览

管线继承 [Causal Forcing](../forcing_13/)，只换 Stage 2 的发动机：

| 阶段 | 做什么 | 本篇改动 |
|------|--------|---------|
| Stage 1 | 双向 Wan2.1-1.3B 微调成多步 AR 扩散教师（教师强迫） | 无（继承） |
| Stage 2 | 少步 AR 学生初始化 | **因果 ODE 蒸馏 → 因果 CD**：真实视频上在线相邻时间步一致性，教师只需单次 ODE 步 |
| Stage 3 | 非对称 DMD + 学生自 rollout（教师/评论家双向，real score 用 Wan2.1-14B） | 无（继承） |

Stage 2 内部一个训练步的文字流程（严禁 ASCII 图）：

| 步骤 | 操作 | 关键点 |
|------|------|--------|
| ① | 从真实视频取帧 $x_{\mathrm{gt}}^i$，前向加噪得 $x_t^i$ | 数据是**真值**，不是教师生成 |
| ② | AR 教师条件于真值前缀 $x_{\mathrm{gt}}^{<i}$，从 $x_t^i$ 做**一次** ODE 步得 $\hat{x}_{t-\Delta t}^i$ | 教师只出一步，在线计算 |
| ③ | 学生（在线）与 EMA 学生（stop-gradient）分别预测两端的干净估计 | 相邻时间步一致性约束 |
| ④ | 距离 $d(\cdot,\cdot)$ 加权求损失，反传更新学生 | 见 Eq.3 |

---

## 4. 核心原理完整推导

### 4.1 三阶段管线从哪来：三篇论文的接力

CausVid 确立「ODE 初始化 + 非对称 DMD」两阶段；Self Forcing 指出 CausVid 的 Stage 3 在真值上下文（diffusion forcing）下训练学生，拼出来的帧不是合法生成视频，与推理时自 rollout 有大间隙——改为学生自 rollout；[Causal Forcing](../forcing_13/) 再指出 Self Forcing 的 ODE 初始化有理论缺陷：ODE 蒸馏意在**匹配生成轨迹**，而 AR 学生原则上无法拟合双向教师诱导的 ODE 轨迹（未来帧对 AR 学生不可见），于是先把双向模型微调成 AR 教师、再用它生成轨迹做初始化。本篇完全接受这套分析，只攻击剩下的问题：**因果 ODE 蒸馏太贵**。

### 4.2 因果 ODE 蒸馏在学什么

Causal Forcing 的 Stage 2 收集 AR 教师 PF-ODE 轨迹上的中间态 $x_t^i$ 与对应干净输出 $x_0^i$，教师强迫下做 MSE 回归：

$$
\theta^* = \arg\min_\theta\ \mathbb{E}_{x_{\mathrm{gt}}^{<i},\,t,\,i,\,x_t^i}\Big[\big\|G_\theta\big(x_t^i,\ x_{\mathrm{gt}}^{<i},\ t\big) - x_0^i\big\|^2\Big].
\tag{Eq.1}
$$

直觉：把「沿教师轨迹任意时刻 $t$ 一步跳到轨迹终点 $x_0$」的映射教给学生。这个目标的最小化解就是教师的 **AR 条件 flow map**：

$$
f_\phi:\ \big(x_t^i,\ x_{\mathrm{gt}}^{<i},\ t\big)\ \mapsto\ x_0^i,
\tag{Eq.2}
$$

即把 $x_t^i$ 映到教师扩散模型 $\phi$ 在 $t=0$ 的 PF-ODE 样本。**关键观察**：$f_\phi$ 正是一致性模型文献里一致性函数（consistency function）的 AR 条件版本——ODE 蒸馏和 CD 想学的是**同一个东西**，差别只在监督怎么拿。

### 4.3 因果一致性蒸馏：把 CD 提升到 AR 设定

标准 CD 目标经教师强迫提升到 AR 条件设定：

$$
\theta^* = \arg\min_\theta\ \mathbb{E}_{x_{\mathrm{gt}},\,\epsilon,\,t,\,i}\Big[w(t)\,d\Big(G_\theta\big(x_t^i,\ x_{\mathrm{gt}}^{<i},\ t\big),\ G_{\theta^-}\big(\hat{x}_{t-\Delta t}^i,\ x_{\mathrm{gt}}^{<i},\ t-\Delta t\big)\Big)\Big],
\tag{Eq.3}
$$

各项来源：$x_t^i$ 由真值 $x_{\mathrm{gt}}^i$ 前向加噪得到；$\hat{x}_{t-\Delta t}^i$ 由 **AR 教师条件于真值前缀**从 $x_t^i$ 做单次 ODE 步得到；$\theta^-$ 是 $\theta$ 的 EMA 且带 stop-gradient；$w(t)$ 是时间步权重、$d$ 是给定范数下的距离（实现取平方范数，见 §5）。

流匹配参数化下，学生的干净预测由速度场读出：

$$
G_\theta\big(x_t^i,\ x_{\mathrm{gt}}^{<i},\ t\big) = x_t^i - t\,v_\theta\big(x_t^i,\ x_{\mathrm{gt}}^{<i},\ t\big),
\tag{Eq.4}
$$

直觉：$t=1$ 时 $G_\theta=x_t-v_\theta$ 恰是从纯噪声出发的单步预测；$t\to0$ 时 $G_\theta\to x_t$。一致性约束要求 $G_\theta$ 沿轨迹近似不变——学生从任何时刻都能一步读出终点。

**为什么学到的就是 $f_\phi$**：沿用标准 CD 分析，最优模型与目标 flow map 的误差受 ODE 求解器数值误差控制：

$$
\sup\big\|f_\phi\big(x_t^i,\ x_{\mathrm{gt}}^{<i},\ t\big) - G_{\theta^*}\big(x_t^i,\ x_{\mathrm{gt}}^{<i},\ t\big)\big\|_2 = \mathcal{O}\big((\Delta t)^p\big),
\tag{Eq.5}
$$

$\Delta t$ 是相邻时间步的最大间隔、求解器 $p$ 阶精度。离散越细（本篇用 48 个离散时间步），上界越小——**因果 CD 与因果 ODE 蒸馏共享学习目标**，只是逼近方式不同：ODE 蒸馏在预生成轨迹上做 $t\to0$ 的大跨度回归；CD 在真实数据上做相邻时间步的局部一致性。

### 4.4 替换的两重红利

| 维度 | 因果 ODE 蒸馏 | 因果 CD |
|------|--------------|---------|
| 监督获取 | 教师离线生成完整多步轨迹（如 48 步/样本），成对存储，教师或数据一变全部重算 | 每次迭代在线一次教师 ODE 步，真实视频直接用 |
| 优化难度 | $x_t^i$ 直接回归 $x_0^i$，一步跨整个 $t\to0$ 区间，gap 大 | 配对 $(x_t^i,\ \hat{x}_{t-\Delta t}^i)$ 只隔 $\Delta t$，每步 gap 小，易优化（与双向蒸馏中 InstaFlow 等的观察一致） |

在 80K 视频规模上，第一项的量化：数据整理 + 训练 ~11,600 → ~2,900 A800 GPU·时（约 4×），辅助存储 ~1,900 GiB → **0**。

### 4.5 反面对照：因果 score 蒸馏为什么不行

既然 CD 行，它的对偶——score 蒸馏（DMD）——行不行？先把它改造成教师强迫的因果形式。双向 DMD 的梯度（学生先采样 $\tilde{x}\sim p_\theta$，加噪到 $\tilde{x}_t$，冻结的 $s_{\mathrm{real}}$ 与在线的 $s_{\mathrm{fake}}$ 之差提供方向）：

$$
\nabla_\theta\mathbb{E}_t\big[D_{\mathrm{KL}}\big(p_{\theta,t}(\tilde{x}_t)\,\big\|\,p_{\mathrm{data},t}(\tilde{x}_t)\big)\big]
= -\mathbb{E}_{\tilde{x},t,\tilde{x}_t}\Big[\big(s_{\mathrm{real}}(\tilde{x}_t,t)-s_{\mathrm{fake}}(\tilde{x}_t,t)\big)\,\partial_\theta \tilde{x}\Big],
\tag{Eq.6}
$$

AR 设定下教师、评论家、学生全部自回归，目标必须改写为条件版、以教师强迫训练：

$$
\nabla_\theta\mathbb{E}_t\big[D_{\mathrm{KL}}\big(p_{\theta,t}(\tilde{x}_t^i\mid x_{\mathrm{gt}}^{<i})\,\big\|\,p_{\mathrm{data},t}(\tilde{x}_t^i\mid x_{\mathrm{gt}}^{<i})\big)\big]
= -\mathbb{E}\Big[\big(s_{\mathrm{real}}(\tilde{x}_t^i, x_{\mathrm{gt}}^{<i}, t)-s_{\mathrm{fake}}(\tilde{x}_t^i, x_{\mathrm{gt}}^{<i}, t)\big)\,\partial_\theta \tilde{x}^i\Big].
\tag{Eq.7}
$$

注意这是**纯因果蒸馏**（师生皆 AR），与 Stage 3 的非对称 DMD（教师双向）角色不同——它只想当初始化。实验结论：**不行**。因果 DMD 的前几帧甚至比因果 CD 更锐利，但随着自回归推进迅速漂移、出现严重相机漂移，后期崩到不可用。

机制解释（论文 Fig. 5b 的直觉图）：DMD 优化反向 KL，是 **mode-seeking**——概率质量集中在少数高密度模态附近，分布尖、早期帧好；CD 优化前向 KL，是 **mode-covering**——分布更宽。自回归的条件分布会因历史漂移（自生成前缀偏离真值历史）而整体平移：mode-seeking 的分布一旦平移进低质量区，**大部分样本跟着错误模态走**；mode-covering 的分布平移后仍有相当质量留在好质量区。所以对历史误差的敏感度：DMD ≫ CD，exposure bias 在 DMD 初始化里被放大。这不是证明，是论文给出的直觉模型，但它精确预测了「前几帧更好、后面崩掉」的观测。

---

## 5. 训练与推理细节

| 项 | 设置 |
|----|------|
| 基座 | Wan2.1-1.3B（师生同底座；Stage 3 的 real score 例外，用 Wan2.1-14B） |
| 生成分辨率 | 480×832、81 帧、**帧级**自回归 |
| 三阶段步数 | Stage 1/2/3 = 20K / 5K / 1K 步，batch 64，其余超参同 Causal Forcing |
| Stage 2（causal CD） | 平方范数、48 个离散时间步、Euler 求解器 |
| Stage 3 步数设定 | 4 步（$t=1,0.9375,0.8333,0.625$）、2 步（$t=1,0.8333$）、1 步；**ASD 技巧**：首个 latent 帧保持 4 步，其后 20 个 latent 帧用 2/1 步（仅 Stage 3 使用） |
| 数据 | Stage 1/2：80K 视频（OpenVid 采样）；Stage 3：VidProM |
| 效率测量 | 单张 A800，**不含 VAE 时间**（Self Forcing / Causal Forcing 原文用 H100——口径不同，不可直接比） |

<details>
<summary><b>展开：消融设定与成本口径标本</b></summary>

- **消融的公平性细节**（论文 §4.2）：步数差异只作用于 Stage 3；Stage 2 在各设定下完全相同；特别地，因果 ODE 初始化统一按 **4 步设定**训练（不是每个步数设定重训一份）；
- **成本口径**：Table 2 的 Time（Stage 2）以 A800 GPU·时计、四舍五入到百位，ODE 一栏**包含 80K 数据整理时间**；Extra Storage 是 ODE 成对数据的 GiB 数；
- **首帧延迟注记**：因 ASD 技巧，1/2/4 步的首帧延迟相同（都由首帧的 4 步生成决定）——「0.27 s」是三个设定共享的数字，吞吐才随步数变化（20.7 / 14.1 / 8.69 FPS）；
- **动作条件世界模型扩展**（§3.3）：Genie3 式相机位姿作动作信号——WorldPlay 构造位姿标注数据 → PRoPE 注入位姿、把 Wan2.1-1.3B 微调成双向位姿条件模型 → 用 CF++ 管线蒸馏成可交互 AR 世界模型。**chunk 级 4 步**设定，帧级 2 步留作 future work。

</details>

---

## 6. 实验与消融

### 6.1 主表：帧级 2 步整体登顶

论文 Table 1（VBench 等指标 ×100；吞吐/延迟在 A800 上测，延迟单位秒）：

| 模型 | 吞吐 ↑ | 延迟 ↓ | Total ↑ | Quality ↑ | Semantic ↑ | Dynamic ↑ | VisionReward ↑ | Instruct ↑ |
|------|-------|--------|---------|----------|-----------|----------|----------------|------------|
| CausVid（chunk 4 步） | 10.4 | 0.60 | 81.33 | 83.98 | 70.72 | 62 | 5.741 | 12 |
| Self Forcing（chunk 4 步） | 10.4 | 0.60 | 83.74 | 84.48 | 80.77 | 57 | 5.820 | 48 |
| Causal Forcing（chunk 4 步） | 10.4 | 0.60 | 84.04 | 84.59 | 81.84 | 68 | 6.326 | 56 |
| CF++（帧级 1 步） | **20.7** | 0.27 | 83.35 | 84.50 | 78.75 | 66 | 5.412 | 38 |
| **CF++（帧级 2 步）** | 14.1 | 0.27 | **84.14** | 84.89 | 81.13 | 64 | **6.661** | 51 |
| CF++（帧级 4 步） | 8.69 | 0.27 | 84.10 | **84.94** | 80.75 | **71** | 6.798 | 47 |

三个读法：**帧级 2 步的 Total 84.14 与 VisionReward 6.661 全面超过 chunk 级 4 步的 Causal Forcing**（84.04 / 6.326）——更细的粒度、更少的步数、更好的质量；首帧延迟 0.27 s（前代 0.60 s，降 50%），吞吐约 1.4×（2 步 14.1 vs 10.4）；4 步设定再涨 Quality 与 Dynamic（84.94 / 71），但吞吐掉到 8.69——帧级 2 步是质量-延迟的甜点。

### 6.2 初始化消融：三败一胜的全景

论文 Table 2 的核心列（帧级 2 步设定，完整三设定全表见折叠块）：

| 初始化 | Total ↑ | Quality ↑ | Dynamic ↑ | VisionReward ↑ | Stage 2 时间 ↓ | 额外存储 ↓ |
|--------|---------|----------|----------|----------------|---------------|------------|
| Self Forcing ODE（双向教师） | 79.44 | 80.43 | 0 | 2.826 | 5,000 GPU·h | 1,500 GiB |
| 多步 AR 扩散直接初始化 | 82.43 | 83.04 | 8 | 3.645 | — | 0 |
| 因果 ODE | 83.77 | 84.42 | 57 | 6.224 | 11,600 GPU·h | 1,900 GiB |
| 因果 DMD | 83.73 | 84.56 | 69 | 6.108 | 2,900 GPU·h | 0 |
| **因果 CD（本篇）** | **84.14** | **84.89** | 64 | **6.661** | **2,900 GPU·h** | **0** |

结论与 §3.1 的预测一一对应：Self Forcing 式初始化帧级下 Total 掉到 80 以下、Dynamic 最高 2（错配被帧级放大）；多步 AR 初始化随步数减少急剧劣化（1 步设定 Dynamic 0、VisionReward 1.101、指令跟随 **−14**，近乎崩溃）；因果 ODE 正确但 11,600 GPU·时 + 1,900 GiB；因果 CD 以 1/4 成本匹配或超过它。因果 DMD 比多步 AR 好得多，但 VisionReward 在所有设定下比因果 CD 低约 0.5——不是理想初始化，且印证 §4.5 的 exposure bias 分析。

<details>
<summary><b>展开：Table 2 全表（1/2/4 步 × 五种初始化）</b></summary>

| 设定 | 初始化 | Total | Quality | Semantic | Dynamic | VisionReward | Instruct | 时间 | 存储 |
|------|--------|-------|---------|----------|---------|--------------|----------|------|------|
| 1 步 | Self Forcing | 78.87 | 79.85 | 74.95 | 0 | 1.992 | −12 | 5000 | 1500 |
| 1 步 | AR 扩散 | 80.54 | 80.97 | 78.84 | 0 | 1.101 | −14 | — | 0 |
| 1 步 | 因果 ODE | 83.06 | 83.88 | 79.77 | 46 | 5.464 | 40 | 11600 | 1900 |
| 1 步 | 因果 DMD | 82.34 | 83.50 | 77.71 | 62 | 4.868 | 20 | 2900 | 0 |
| 1 步 | **因果 CD** | 83.35 | 84.50 | 78.75 | 66 | 5.412 | 38 | 2900 | 0 |
| 2 步 | Self Forcing | 79.44 | 80.43 | 75.47 | 0 | 2.826 | −14 | 5000 | 1500 |
| 2 步 | AR 扩散 | 82.43 | 83.04 | 80.00 | 8 | 3.645 | 16 | — | 0 |
| 2 步 | 因果 ODE | 83.77 | 84.42 | 81.19 | 57 | 6.224 | 46 | 11600 | 1900 |
| 2 步 | 因果 DMD | 83.73 | 84.56 | 80.39 | 69 | 6.108 | 38 | 2900 | 0 |
| 2 步 | **因果 CD** | 84.14 | 84.89 | 81.13 | 64 | 6.661 | 51 | 2900 | 0 |
| 4 步 | Self Forcing | 79.82 | 80.58 | 76.76 | 2 | 3.675 | −2 | 5000 | 1500 |
| 4 步 | AR 扩散 | 83.41 | 84.42 | 79.38 | 62 | 5.166 | 30 | — | 0 |
| 4 步 | 因果 ODE | 83.78 | 84.90 | 79.28 | 75 | 6.435 | 42 | 11600 | 1900 |
| 4 步 | 因果 DMD | 83.49 | 84.84 | 78.11 | 67 | 6.309 | 36 | 2900 | 0 |
| 4 步 | **因果 CD** | 84.10 | 84.94 | 80.75 | 71 | 6.798 | 47 | 2900 | 0 |

单位同前（时间 A800 GPU·时、存储 GiB）。几个正文未展开但值得注意的点：因果 ODE 在 4 步设定 Semantic 只有 79.28（因果 CD 80.75）；Self Forcing 初始化的 Instruct 在三个设定里两个为负；因果 DMD 的 Dynamic 反而常是最高档（1 步 62、2 步 69）——锐利与漂移并存，与 §4.5 的 mode-seeking 画像一致。

</details>

### 6.3 Stage 2 直接对比与世界模型扩展

- 论文 Fig. 3(a)：Stage 2 结束时与作为 Stage 3 初始化两个口径下，因果 CD 的 VBench 都高于因果 ODE——替换不是「省钱降质」，是**省钱提质**；
- 论文 Fig. 3(b)：时间成本 2,900 vs 11,600 GPU·时、存储 0 vs ~1,900 GiB 的柱状对照；
- 世界模型：相机位姿条件（前进、先前进再俯仰等动作）的交互式生成定性成立（论文 Fig. 4），无量化指标。

---

## 7. 与系列其他篇目的关系

| 篇目 | 连接点 |
|------|--------|
| [13 Causal Forcing](../forcing_13/) | 直接前作：13 证明初始化必须因果对齐（本篇 §4.2 的 Eq.1/Eq.2 即其 Stage 2），本篇把它的成本瓶颈拆掉——「正确」之后的下一问是「能不能便宜且好」 |
| [06 CausVid](../forcing_06/) | 三阶段管线的源头（ODE 初始化 + 非对称 DMD）；CausVid 的双向教师 ODE 初始化正是本篇 §1.1 判死刑的第一种 |
| [07 Self Forcing](../forcing_07/) | Stage 3 自 rollout 直接继承；Self Forcing 式初始化在帧级设定下的崩溃是本篇动机实验的对照组 |
| [14 Diagonal Distillation](../forcing_14/) | 同属少步蒸馏，切法不同：14 沿时间×噪声二维网格分配步数（chunk 级、前重后轻），本篇把粒度推到帧级、步数压到 1–2——两者一个改日程、一个改粒度，理论上可叠 |
| [16 One-Forcing](../forcing_16/) | 下一篇：16 的生成器恰恰从 Causal Forcing 的 ODE 初始化检查点出发，且其实验发现 CD 初始化语义分更高但动态大幅落后（52.76 vs 23.61）——与本篇「CD 全面替 ODE」的结论放在一起读很有意思：初始化的选择依赖后续目标 |
| [17 Causal-rCM](../forcing_17/) | 再下一篇：把一致性初始化与分布匹配细化统一进一个框架，可视为本篇「CD 与 DMD 各司其职」观点的延伸 |
| [VG 06 VideoScene](../video_generation/VG_06_videoscene/) | 少步蒸馏的另一战场（场景生成），DMD 用法可对照 |

**跨篇排榜警告**：本篇效率数字在 A800 上测且不含 VAE，[14](../forcing_14/) 在 H100 + Tiny VAE 上测，[16](../forcing_16/) 推理在 H100/H200 上测——同指标不同口径，不可直接排榜。

---

## 8. 局限与个人评价

1. **效率口径自成一体**：A800、不含 VAE，且 Self Forcing / Causal Forcing 原文用 H100——Table 1 里前代的 0.60 s / 10.4 FPS 是本篇协议下的重测值，与原论文数字不能混用；
2. **首帧延迟的公平性**：ASD 技巧让 1/2/4 步共享首帧 4 步生成，0.27 s 的「50% 降延迟」部分来自帧级化而非步数本身；真正的单步首帧未测；
3. **1 步设定的语义掉档**：Total 83.35 尚可，Semantic 78.75、VisionReward 5.412 明显低于 2 步——「帧级 1 步」还没有完全站住，论文主打的是 2 步；
4. **世界模型只有定性**：动作条件扩展是管线复用的展示，帧级 2 步交互版留作 future work；
5. **mode-covering 解释是直觉模型**：§4.5 的分析靠示意图与前几帧/后期帧的观测支撑，没有理论证明；「CD 初始化更优」在 [16 One-Forcing](../forcing_16/) 的设定下出现反例（动态维度），说明结论有边界；
6. **对 causal ODE 的比较有一处不完全公平**：消融里因果 ODE 统一按 4 步设定训练，没有为 1/2 步重训——本篇自己的论点恰恰是「初始化要匹配目标设定」。

**个人评价**：这篇的技术内核其实很薄——「把 CD 提升到 AR 条件 + 教师强迫」一个公式的事——但论证链完整：先证明现有三种初始化在激进设定下各死一种死法，再给出与因果 ODE 共享目标、只换监督获取方式的替代，最后用 4× 成本削减和全面质量数字收尾。真正的方法论贡献是把「初始化必须是 AR + 少步 + 可扩展」三条件变成这个子领域的显式检查表。**面试可能被问的三个问题**：causal CD 与 causal ODE 为什么等价？（同一 AR 条件 flow map / 一致性函数，Eq.5 误差界只差求解器阶数）；DMD 初始化为什么前几帧好后面崩？（反向 KL 的 mode-seeking 对历史漂移敏感）；为什么不直接用非对称 DMD 做初始化？（它是精修器不是训练器，初始化缺少步能力时 DMD 接不住优化负担）。

---

*下一篇 [Forcing 16：One-Forcing](../forcing_16/)——单步生成的清晰度与动态性：教师轨迹的曲率几何如何卡死一致性蒸馏，GAN 信号如何补上 DMD 的细节盲区。全系列目录见[总览](../forcing/)。*

{{< forcing_nav >}}
