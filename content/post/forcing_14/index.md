---
title: "Forcing: 14. Diagonal Distillation"
date: 2026-10-14
math: true
tags:
  - "Forcing"
  - "Video Generation"
  - "Diagonal Distillation"
categories:
  - Forcing
series: forcing
weight: 14
description: "把图像少步蒸馏直接搬到视频，会丢掉 chunk 之间的条件依赖。DiagDistill 在「视频时间 × 去噪步数」二维网格上重新分配计算：前三个 chunk 用 5/4/3 步、之后一律 2 步，KV 缓存改存加噪中间态（对角 Forcing），再加流分布匹配保运动幅度。5 秒视频 2.61 秒生成（31 FPS，对未蒸馏 Wan2.1 加速 277.3×），VBench Total 84.48 追平教师的 84.26，首帧延迟 0.37 秒（Self-Forcing 为 0.69 秒）"
paper_id: "2603.09488v2"
---

# 沿时间与噪声两条轴分配计算

> **Forcing 系列 · 第 14 篇 · 少步蒸馏与初始化**
> 论文：*Streaming Autoregressive Video Generation via Diagonal Distillation*（arXiv:2603.09488v2，ICLR 2026）。前置：[13 Causal Forcing](../forcing_13/)（师生因果条件对齐）、[06 CausVid](../forcing_06/)（双向→因果蒸馏）、[07 Self Forcing](../forcing_07/)（滚动 KV 缓存）；扩散与少步蒸馏基础见 [VG 01](../video_generation/VG_01_diffusion_basics/) 与 [VG 06 VideoScene](../video_generation/VG_06_videoscene/)。本篇所有数字按论文 v2 正文与附录逐条核对。
>
> 📂 **阅读方式**：正文只保留主线——结论、关键公式与总结表；长推导、超参标本与消融全表收进折叠块（点击展开）。

---

## 论文信息

- **原题**：Streaming Autoregressive Video Generation via Diagonal Distillation
- **阅读版本**：arXiv:2603.09488v2（ICLR 2026 会议论文），对应本地收集的 PDF 版本。
- **原文**：[arXiv 摘要页](https://arxiv.org/abs/2603.09488v2) · [PDF](https://arxiv.org/pdf/2603.09488v2)

---

## 1. 这篇论文解决什么问题

### 1.1 图像蒸馏直接搬视频，漏掉了两条轴

[CausVid](../forcing_06/) 与 [Self Forcing](../forcing_07/) 把双向视频扩散蒸馏成因果少步学生之后，每个 chunk 仍要 4 步左右的去噪，且所有 chunk 用**同样的步数**。论文把这种「图像式搬法」的结构性缺陷归为两条：

| 被忽略的事实 | 后果 |
|------|------|
| 预测下一个 chunk 时，模型隐式地也在**预测下一个噪声等级** | 噪声等级预测的误差随自回归逐步累积——长序列过饱和、质量单调衰减（exposure bias 的一种形态） |
| 早期 chunk 已经建立起外观与结构先验，后续 chunk 本可以少步生成 | 步数均匀分配 → 要么整体多步（慢），要么整体少步（运动幅度衰减、细节丢失） |

两条轴——**视频时间**（第几个 chunk）与**去噪时间**（第几步去噪）——在图像蒸馏里根本不存在，在视频里却是计算分配的天然坐标。论文的答案是对角日程：**早期 chunk 多步、后期 chunk 少步**，让后期 chunk 从前面充分去噪的 chunk 继承先验。

### 1.2 目标、数字与系列位置

流式自回归 + 少步蒸馏 + 长视频稳定三者同时要：5 秒视频 **2.61 秒**生成（最高 31 FPS），对未蒸馏 Wan2.1 加速 **277.3×**，VBench Total **84.48** 追平未蒸馏模型的 84.26（论文 Table 1）；45 秒长视频质量平稳，用户研究 93 人对 CausVid 偏好率 66.1%。

本篇是第四部分「少步蒸馏与初始化」的第一篇，可读作**「噪声与时间」维度的蒸馏版**：[04 Diffusion Forcing](../forcing_04/) 与 [08 Rolling Forcing](../forcing_08/) 在训练/推理的二维（帧 × 噪声级）日程上做文章，本篇问同一张网格上**哪些 (帧, 噪声级) 组合值得花算力**。它与 [13 Causal Forcing](../forcing_13/) 正交：13 问「教师与学生的条件结构是否对齐」，本篇问「对齐之后计算怎么摆」。

---

## 2. 最小背景

- **分布匹配蒸馏（Distribution Matching Distillation, DMD）**：教师扩散模型 + 在线训练的 fake score 网络，用两个 score 之差给学生提供分布匹配梯度；外加一条对教师输出的回归损失做正则。推导见 [VG 06](../video_generation/VG_06_videoscene/)（VideoScene 同样是蒸馏压步数，可对照）与本篇 §4.1；
- **自回归 chunk 生成 + 滚动 KV 缓存**：因果 DiT 逐 chunk 生成，历史以 KV 缓存形式滚动保留（[07 Self Forcing](../forcing_07/)，chunk 大小 3 个 latent 帧）；
- **Wan2.1-T2V-1.3B 的流匹配参数化**：模型预测速度场 $v_\theta$，前向过程为线性插值加噪（细节收在 §5 折叠块）；
- **VBench 指标聚合（本文口径）**：Temporal Quality = 5 项时序子分平均；Frame Quality = Aesthetic + Imaging 平均；Text Alignment = 9 项语义子分平均（与 CausVid、Self Forcing 同口径以便公平比较）。

---

## 3. 方法总览

一个 5 秒视频 = 21 个 latent 帧 = 7 个 chunk（每 chunk 3 帧）。整套系统四个模块：

| 模块 | 做什么 | 对应论文概念 |
|------|--------|------------|
| ① 对角去噪 | 步数日程 $s=[5,4,3,2,2,2,2]$：chunk 1 用 5 步、chunk 2 用 4 步、chunk 3 用 3 步、chunk 4 起一律 2 步 | Diagonal Denoising |
| ② 对角 Forcing | KV 缓存不存干净帧，存「去噪到倒数第二步再加噪」的中间 latent——训练时显式模拟这条对角路径 | Diagonal Forcing |
| ③ 流分布匹配 | 在 DMD 的空间分布匹配之外，对齐教师与学生的**运动流场分布**，防止少步导致的运动幅度衰减 | Flow Distribution Matching |
| ④ 流式系统 | 滚动 KV 缓存收窄到 4 个 chunk（12 帧）、Tiny VAE 解码、支持动态提示（任意时刻换 prompt） | 工程贡献 |

推理按 chunk 顺序推进：chunk $k\le 3$ 从纯噪声多步生成，建立外观与运动先验；每个 chunk 去噪到**倒数第二步**时，把中间 latent 加噪写入 KV 缓存；chunk $k\ge 4$ 条件于上一 chunk 的噪声态缓存两步生成——先验已建立，2 步够用。

---

## 4. 核心原理完整推导

### 4.1 起点：DMD 的梯度与它的缺口

DMD 最小化加噪后的真实分布 $p_{\mathrm{real},t}$ 与生成分布 $p_{\mathrm{fake},t}$ 的 KL 散度，其梯度可以用两个分数（score）之差写出——$s_{\mathrm{real}}(x_t,t)=-\big(x_t-\alpha_t\mu_{\mathrm{real}}(x_t,t)\big)/\sigma_t^2$（论文 Eq.1）来自冻结教师（$\mu_{\mathrm{real}}$ 是数据侧去噪估计，$\alpha_t,\sigma_t$ 由噪声日程决定，score 基础见 [VG 01](../video_generation/VG_01_diffusion_basics/)）：

$$
\nabla_\theta \mathcal{L}_{\mathrm{DMD}}
= \mathbb{E}_t\big[\nabla_\theta\,\mathrm{KL}(p_{\mathrm{fake},t}\,\|\,p_{\mathrm{real},t})\big]
= -\mathbb{E}_t\left[\int \big(s_{\mathrm{real}}(x_t,t)-s_{\mathrm{fake}}(x_t,t)\big)\,\frac{\mathrm{d}G_\theta(z)}{\mathrm{d}\theta}\,\mathrm{d}z\right],
\tag{Eq.2}
$$

其中 $z\sim\mathcal{N}(0,I)$，$x_t = F(G_\theta(z), t)$ 是对学生输出加噪，$s_{\mathrm{real}}$ 来自冻结教师、$s_{\mathrm{fake}}$ 来自在生成数据上在线训练的 fake score 网络。直觉：**两个 score 的差就是「把生成分布往真实分布推」的方向场**，反传到生成器参数上。配套的回归损失

$$
\mathcal{L}_{\mathrm{reg}} = \mathbb{E}_{(z,y)}\, d\big(G_\theta(z),\, y\big)
\tag{Eq.3}
$$

（$y$ 是教师从 $z$ 生成的样本）负责锚定优化、防止坍塌。

**缺口**：$\mathcal{L}_{\mathrm{reg}}$ 与 $\mathcal{L}_{\mathrm{DMD}}$ 都在**单帧（或单 chunk）**的空间上定义——没有任何一项显式约束**帧间变化**的分布。视频压到极少步时，最先坏的就是这个没人管的维度：运动幅度衰减。

### 4.2 对角去噪：步数沿时间递减

前三个 chunk（$k=1,2,3$）用步数递减的蒸馏模型：

$$
X_k = D_{s_k}\big(Z_k \,\big|\, \tilde{X}_{<k}\big),\qquad s_k = 5,4,3,
\tag{Eq.4}
$$

$Z_k\sim\mathcal{N}(0,I)$ 是该 chunk 的初始噪声，$\tilde{X}_{<k}$ 是**之前各 chunk 的加噪版本**（见 §4.3），$D_{s_k}$ 是蒸馏到 $s_k$ 步的学生。从 $k\ge 4$ 起切换为固定两步：

$$
C_k = \mathcal{T}(\tilde{X}_{k-1}),\qquad
X_k = D_2\big(D_1(Z_k \mid C_k)\,\big|\, C_k\big),
\tag{Eq.5}
$$

$C_k$ 是由条件模块 $\mathcal{T}$ 从上一 chunk 的噪声态导出的条件信号，$D_1,D_2$ 是两步去噪的两次前向。

**直觉**：为什么后期 2 步就够？因为条件 $C_k$ 里已经装着「被充分去噪过的先验」——外观、结构、运动模式都在。这解释了为什么步数日程应该**单调递减**而非均匀或非单调（附录 E.2 的探索证实非单调日程无一致收益，还会破坏运动一致性：见 §6.3）。

### 4.3 对角 Forcing：KV 缓存里存噪声态

条件输入从上一 chunk 的**干净输出** $X_{k-1}$ 经一次噪声注入得到：

$$
\tilde{X}_{k-1} = \sqrt{\alpha_{k-1}}\cdot X_{k-1} + \sqrt{1-\alpha_{k-1}}\cdot\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I),
\tag{Eq.6}
$$

$\alpha_{k-1}$ 控制对角路径上的噪声日程。$\tilde{X}_{k-1}$ 直接作为 chunk $k$ 的 KV 缓存输入——论文称这维护了「对角去噪轨迹」，即每个 chunk 的条件不再是干净的过去，而是**与推理时一致的、带噪声的过去**。

训练侧（附录 C）：条件是**混合上下文**——对给定目标帧，近期帧用模型自己的预测、远期帧用真值，混合模式随目标帧位置**对角移动**。三个好处：逐步暴露自身误差（稳健性）、远处真值锚定（防语义漂移）、训练条件分布与推理时的对角去噪对齐。

**KV 噪声等级是超参**（Wan 调度共 1000 个离散步，0 = 干净帧、1000 = 完全噪声）：消融（论文 Fig. 5a）显示 **100 步最优**——噪声过高会把上下文里的结构先验抹掉、运动幅度下降；噪声过低则回到「隐式预测下一噪声等级」的老问题，导致后续 chunk 过度去噪、输出过饱和。这正好把 §1.1 的两个失败模式钉在了同一根轴的两端。

### 4.4 流分布匹配：给运动分布一条 DMD 梯度

先诊断运动衰减：少步去噪 = **截断的噪声估计路径**。多步采样时运动信息沿轨迹逐步修正；两步生成要求运动在一次前向里全部到位，而 §4.1 的两个损失都不监督它。

论文用流场散度量化时序分布失配：

$$
\mathcal{E}_{\mathrm{motion}} = D_{\mathrm{KL}}\big(p_{\mathrm{teacher}}(F(x)\mid x_t)\,\big\|\,p_{\mathrm{student}}(F(x)\mid x_t)\big),
\tag{Eq.7}
$$

其中 $F(x)$ 是从视频序列 $x$ 提取的**运动流场**（这里不是外部光流估计器，而是一个直接作用在 latent 上的轻量可学习模块，见下）。把 DMD 的反向 KL 整体搬到流场空间：

$$
\nabla_\phi \mathcal{L}^{\mathrm{flow}}_{\mathrm{DMD}}
\triangleq \mathbb{E}_t\big[\nabla_\phi\,\mathrm{KL}\big(p_{\mathrm{gen,flow},t}\,\big\|\,p_{\mathrm{data,flow},t}\big)\big],
\tag{Eq.8}
$$

$p_{\mathrm{data,flow},t}=p(F(x)\mid \Psi(x,t))$ 与 $p_{\mathrm{gen,flow},t}=p(F(x)\mid \Psi(G_\phi(\epsilon),t))$ 分别是真实数据与学生输出**加噪后**的流分布（$\Psi$ 为前向加噪）。与 Eq.2 完全同构的梯度近似：

$$
\nabla_\phi \mathcal{L}^{\mathrm{flow}}_{\mathrm{DMD}}
\approx -\mathbb{E}_t\left[\int\Big(s^{\mathrm{flow}}_{\mathrm{data}}\big(\Psi(G_\phi(\epsilon),t),t\big) - s^{\mathrm{flow}}_{\mathrm{gen},\xi}\big(\Psi(G_\phi(\epsilon),t),t\big)\Big)\frac{\mathrm{d}G_\phi(\epsilon)}{\mathrm{d}\phi}\,\mathrm{d}\epsilon\right],
\tag{Eq.9}
$$

其中流分数（flow score）定义为

$$
s^{\mathrm{flow}}(x_t, t) = \nabla_{x_t}\log p\big(F(x)\mid x_t\big).
\tag{Eq.10}
$$

再加一条流回归损失做锚定（结构对应 Eq.3 的 $\mathcal{L}_{\mathrm{reg}}$）：

$$
\mathcal{L}^{\mathrm{flow}}_{\mathrm{reg}} = \mathbb{E}_{t,\epsilon}\left[\big\|F\big(G^{\mathrm{teacher}}_\phi(\epsilon, t)\big) - F\big(G^{\mathrm{student}}_\phi(\epsilon, t)\big)\big\|_2^2\right].
\tag{Eq.11}
$$

**直觉**：Eq.9 告诉学生的不再是「每帧要像」，而是「**帧间变化的分布**也要像」——两个流分数的差把运动空间的分布差异反传回生成器。总目标把空间与时间两项拼起来：

$$
\mathcal{L}_{\mathrm{Total}} = \lambda_{\mathrm{spatial}}\mathcal{L}_{\mathrm{DMD}} + \mathcal{L}_{\mathrm{reg}} + \gamma\big(\lambda_{\mathrm{flow}}\mathcal{L}^{\mathrm{flow}}_{\mathrm{DMD}} + \mathcal{L}^{\mathrm{flow}}_{\mathrm{reg}}\big),
\tag{Eq.12}
$$

$\lambda_{\mathrm{spatial}}=4$、$\lambda_{\mathrm{flow}}=4$；流项总权重 $\gamma$ 的八组消融显示 **1.0 最优**（论文 Fig. 5b）——过强的流约束反而有害，回归锚定丢了训练就极易坍塌（附录 G）。

**$F(\cdot)$ 的实现**：相邻 latent 帧做差 → 卷积提局部运动模式 → MLP 适配特征；学生端可训练、教师端 EMA 更新。两层卷积的 Conv-MLP 开销几乎可忽略；刻意**不用** RAFT 这类外部光流估计器，避免外部误差偏置（七种实现的对比见 §6.3 折叠块）。

---

## 5. 训练与推理细节

| 项 | 设置 |
|----|------|
| 基座模型 | Wan2.1-T2V-1.3B（流匹配，832×480，16 FPS，5 秒 = 7 个 chunk × 3 帧） |
| 训练数据 | 过滤 + LLM 扩写版 VidProM 提示（ODE 初始化与对角蒸馏训练都用它） |
| 蒸馏目标 | DMD（对角去噪模式，时间步列表 $[1000, 100]$）+ 流分布匹配（Eq.12） |
| 推理硬件 | 单张 H100 + Tiny VAE；滚动 KV 缓存 4 个 chunk（12 帧），显存 17.5 GB |
| 步数日程 | $4322222$（5 秒）；更长视频循环延长，如 10 秒用 $43222224322222$ |
| 长视频 | 45 秒无显著衰减；O(TL) 复杂度；支持动态提示（任意时刻插入新 prompt） |

<details>
<summary><b>展开：噪声日程、算法与加速来源标本</b></summary>

**时间步偏移**（Wan2.1 设计，偏移因子 $k=5.0$）：

$$
t'(k,t) = \frac{k\,(t/1000)}{1+(k-1)(t/1000)}\times 1000,\qquad
x_t = \frac{t'}{1000}\,x + \Big(1-\frac{t'}{1000}\Big)\,\epsilon,
$$

数据预测参数化 $G_\theta(x,t,c) = c_{\mathrm{skip}}\cdot\epsilon - c_{\mathrm{out}}\cdot v_\theta(c_{\mathrm{in}}\cdot x_t,\ c_{\mathrm{noise}}(t'),\ c)$，保持基座配置 $c_{\mathrm{skip}}=c_{\mathrm{in}}=c_{\mathrm{out}}=1$、$c_{\mathrm{noise}}(t)=t$。

**算法 1（对角去噪 + 噪声 KV 缓存）骨架**：输入步数表 $s=[5,4,3,2,2,2,2]$。基础段 $k\le 4$：从纯噪声起做 $s[k]$ 步去噪，**在倒数第二步**执行 CACHENOISYRESULT——把中间 latent 按 Eq.6 加噪成 $\tilde{X}_{\mathrm{interim}}$ 并缓存其 KV。扩展段 $k>4$：条件于 $\tilde{X}_{k-1}$ 两步生成，第一步结果同样缓存。每个 chunk 的最终输出与噪声 Mix 后进入下一 chunk 的条件流。首帧延迟口径：包含初始 chunk 的完整多步去噪；后续 chunk 借滚动缓存，prefill 成本基本消除。

**四项加速来源（附录 E.1）**：① **去噪步数减少**（总 NFE 34，日程 4322222）；② **KV 缓存机制**——Self-Forcing 要在干净帧上单独算一遍 KV，本篇直接在噪声 latent 上缓存，一次前向同时供条件与推进去噪；③ **注意力窗口收窄**（6 chunk/18 帧 → 4 chunk/12 帧，显存 19.2 GB → 17.5 GB）；④ **Tiny VAE**（解码参数 9.84M vs 73.3M，解码 0.12 s vs 1.67 s，快 10 倍以上——计入总加速比，换回 Full VAE 数字会明显变化）。

</details>

---

## 6. 实验与消融

### 6.1 主表：追平教师，快一个档次

论文 Table 1（VBench，5 秒视频）：

| 模型 | 吞吐 ↑ | 首帧延迟 ↓ | 加速 | Total ↑ | Quality ↑ | Semantic ↑ |
|------|-------|-----------|------|--------|----------|-----------|
| Wan2.1（未蒸馏） | 0.78 | 103 s | 1.0× | 84.26 | 85.30 | 80.09 |
| SkyReels-V2 | 0.49 | 112 s | 0.91× | 82.67 | 84.70 | 74.53 |
| MAGI-1 | 0.19 | 282 s | 0.36× | 79.18 | 82.04 | 67.74 |
| CausVid | 17.0 | 0.69 s | 149.3× | 81.20 | 84.05 | 69.80 |
| Self-Forcing | 17.0 | 0.69 s | 149.3× | 84.31 | 85.07 | 81.28 |
| **DiagDistill** | **31.0** | **0.37 s** | **277.3×** | **84.48** | **85.26** | **81.73** |

Total 84.48 **超过**未蒸馏教师（84.26）；吞吐 31 FPS 约为 Self-Forcing 的 1.8 倍；正文称相对之前最快方法延迟改善 1.53×，Fig. 1 则展示 5 秒生成端到端 4.91 s → 2.61 s（×1.88）——两个口径都来自论文，引用时注明出处即可。

### 6.2 组件消融：每块都不可少，但幅度不同

论文 Table 2：

| 变体 | Temporal ↑ | Frame ↑ | Text ↑ | Total ↑ |
|------|-----------|---------|--------|---------|
| 去掉 Diagonal Forcing（KV 存干净帧） | 92.1 | 60.1 | 26.9 | 83.58 |
| 去掉流损失 | 92.5 | 60.8 | 27.8 | 84.18 |
| 去掉 Diagonal Denoising（步数均匀） | 95.1 | 63.2 | 28.6 | 84.46 |
| **完整方法** | 94.9 | 63.4 | 28.9 | **84.48** |

两个诚实的观察：去掉 Diagonal Denoising 的 Total 只掉 0.02——论文自己指出这一档与完整方法相当，**它省的是推理成本**（对应 Self-Forcing 在 Table 1 的开销），质量收益主要体现在同延迟的对比里；流损失则**主要在少步区间起作用**，把少步配置拉回多步基线水平，多步设定里收益有限。

### 6.3 步数日程：单调递减就是对的

论文 Table 3（7 位数字 = 7 个 chunk 各自步数）：

| 日程 | Temporal ↑ | Frame ↑ | Text ↑ | NFEs | 在途延迟 ↓ | 吞吐 ↑ |
|------|-----------|---------|--------|------|-----------|--------|
| **4322222（选定）** | 94.9 | 63.4 | 28.9 | **34** | **0.23 s** | **31.0** |
| 5433333 | 95.1 | 63.2 | 29.3 | 48 | 0.34 s | 23.3 |
| 5333333 | 95.0 | **63.9** | 29.1 | 46 | 0.34 s | 22.5 |
| 4333333 | 95.0 | 63.7 | 28.5 | 44 | 0.34 s | 23.5 |
| 5432222 | 94.8 | 63.1 | 29.0 | 40 | 0.23 s | 29.7 |
| 4222222 | 93.4 | 62.3 | 27.8 | 32 | 0.23 s | 32.0 |

选定 4322222：NFE 第二低、质量与高成本配置只有边际差距。附录 E.2 的扩展探索（16 种日程）确认：**单调递减最优**，非单调日程（如 4343232）无一致收益且伤害运动质量——后续 chunk 的运动条件建立在早期 chunk 的运动之上，中途加步反而破坏一致性。

<details>
<summary><b>展开：同 NFE 对照与其余消融全表</b></summary>

**同 NFE 预算下 Diagonal vs Self（附录 Table 5，五个配置全部胜出）**：NFE 34（日程 4322222）下 Diagonal 94.9/63.4/28.9、0.23 s、31.0 FPS，Self 93.5/62.1/27.5、0.43 s、25.9 FPS；NFE 56（4444444）下 Diagonal 95.1/63.2/28.6、0.46 s、21.5 FPS，Self 94.3/63.1/29.3、0.69 s、17.0 FPS。这隔离了「KV 缓存直接建在噪声 latent 上」的纯机制收益——步数相同，延迟近乎减半。

**其余消融**：KV 窗口扫描（附录 Table 4，VBench-Long）——窗口 3→27 帧，Total 80.9→84.5、延迟 0.37→0.68 s、显存 14.9→21.8 GB，12 帧后收益递减，选定 12 帧（84.3、0.46 s、17.5 GB）。流表示七选一（附录 Table 9）——Raw Latent 92.5/60.8/27.8 一路升到**可学习 + 帧差卷积 94.9/63.4/28.9**（Temporal/Frame/Text）。chunk 级 vs 帧级（附录 Table 7）——chunk=3：31.0 FPS、0.37 s、Total 84.48；chunk=1：16.5 FPS、0.25 s、Total 84.29（帧级延迟更低、吞吐更低、质量略降）。日程探索（附录 Table 6）——强调早期帧的 5422222（95.1/64.1/29.2，38 NFE）、5522222（95.3/63.8/29.4，40 NFE）质量略升但有延迟代价；非单调组质量不稳，3333533 掉到 93.5/60.0/27.7。

</details>

### 6.4 长视频与用户研究

| 项 | 结果 |
|----|------|
| 45 秒长视频 | 质量随时间平稳（各时间点偏好 >50%）；CausVid、Self-Forcing 等基线饱和失真、质量衰减（论文 Fig. 7/8） |
| 用户研究 | 93 人、每模型对 150 次比较、MovieGenBench 前 50 提示、双盲：vs CausVid **66.1%**、vs Wan2.1 62.7%、vs SkyReels-V2 57.9%、vs MAGI-1 54.2%、vs Self-Forcing **59.3%**，全部 $p<0.01$，Cohen's $\kappa=0.78$ |
| 动态提示 | 时间轴任意点插入新 prompt，生成叙事演化长视频（论文 Fig. 9，无量化指标） |

---

## 7. 与系列其他篇目的关系

| 篇目 | 连接点 |
|------|--------|
| [13 Causal Forcing](../forcing_13/) | 13 证明 ODE 初始化的师生**条件结构对齐**是正确性问题；本篇在同一管线内改的是**计算分配**（步数日程 + 缓存内容），两轴正交 |
| [06 CausVid](../forcing_06/) | CausVid 的双向→因果蒸馏是本篇的管线底座之一；本篇 Table 1 里对 CausVid 的优势主要来自时间信息利用与 KV 机制 |
| [07 Self Forcing](../forcing_07/) | 滚动 KV 缓存机制直接继承自 Self-Forcing，三处改造：缓存建在噪声 latent（省一次前向）、窗口 6→4 chunk、暴露偏差经对角 Forcing 对齐 |
| [08 Rolling Forcing](../forcing_08/) | 共享「时间 × 噪声」二维日程视角：08 用滚动联合去噪抑制误差传播，本篇在蒸馏设定下问同一张网格上步数怎么摆 |
| [15 Causal Forcing++](../forcing_15/) | 15 攻初始化的成本与扩展性（帧级 2 步），本篇攻推理日程——少步日程的两种切法，见下篇 |
| [16 One-Forcing](../forcing_16/) | 16 把单步的清晰度问题交给 GAN 信号；本篇用流分布匹配解决同类的运动衰减，两条互补路线（VideoScene 等[图像/场景侧少步蒸馏](../video_generation/VG_06_videoscene/)可互参 DMD 用法） |

**跨篇排榜警告**：本篇（H100 + Tiny VAE）、15（A800、不含 VAE 时间）、16（H100 推理/H200 计时）硬件与计时口径都不同，步数/延迟/质量数字**不能**直接跨论文比较。

---

## 8. 局限与个人评价

1. **加速比的构成**：277.3× 里含 Tiny VAE 的 10 倍以上解码加速与 KV 机制收益，不只是对角去噪本身；消融也显示去掉对角去噪后质量几乎不降（84.46 vs 84.48）——「对角」的价值主要是**同质量下省算力**，论文的表述有时让两者显得混同；
2. **运动幅度靠流损失权重精细平衡**：$\gamma=1.0$ 是八组扫描选出的，回归锚定一丢训练就易坍塌（附录 G 自陈）——这个模块的稳健性存疑；
3. **评测口径为自定义聚合**：Temporal/Frame/Text 的子项平均是本篇（及 CausVid/SF 一系）的约定口径，与其他论文的官方 VBench Total 对比时要小心；
4. **规模单一**：只有 Wan2.1-1.3B、832×480；对角日程是否随模型规模、分辨率、chunk 大小稳定，论文未验证；
5. **动态提示无量化；KV 窗口被预算截断**：叙事切换只有定性展示；窗口 27 帧还有 84.5（vs 12 帧的 84.3），质量上限是拿延迟换的。

**个人评价**：这篇的价值不在某个单点技术，而在把「视频蒸馏 ≠ 图像蒸馏 + 时间维」这件事用二维日程讲清楚了——哪些 (帧, 噪声级) 组合值得算，答案是「早期高噪声值得，后期低噪声不值得」。噪声态 KV 缓存是个漂亮的工程-理论双重收益（省一次前向 + 对齐训练推理分布）。**面试可能被问的三个问题**：为什么步数日程单调递减？（先验传递：运动的条件建立在早期 chunk 的运动上）；KV 缓存存噪声态为什么反而好？（双重职责：条件 + 去噪推进，且与推理一致）；流分布匹配与普通 DMD 的区别？（score 差搬进流场空间——监督对象从帧分布变成帧间变化的分布）。

---

*下一篇 [Forcing 15：Causal Forcing++](../forcing_15/)——初始化的成本问题：用在线一致性蒸馏替换离线 ODE 轨迹，把因果学生的 Stage 2 成本砍掉 4 倍。全系列目录见[总览](../forcing/)。*

{{< forcing_nav >}}
