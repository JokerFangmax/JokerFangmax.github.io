---
title: "Forcing: 00. 系列总览与阅读路线"
date: 2026-10-14
math: true
tags:
  - "Forcing"
  - "Diffusion Model"
  - "Video Generation"
categories:
  - Forcing
series: forcing
weight: 0
description: "从 Teacher Forcing、Scheduled Sampling 到 Diffusion Forcing、Self Forcing、长视频蒸馏与上下文梯度：21 篇论文的学习路线。"
---

# Forcing 到底在改变什么？

序列生成有一个贯穿文本、视频与世界模型的问题：**训练时模型看见什么历史，生成时又依赖什么历史？** 从 Teacher Forcing 到 Self Forcing，这个问题逐渐延伸到噪声日程、教师监督范围、缓存记忆和梯度流。

这个系列以收集的 **21 篇论文**为基础，按学习依赖组织，而不只按发表时间排序。包含本篇总览与 21 篇论文精读：每篇提供问题背景、完整推导（训练目标/损失/梯度）、训练与采样流程、核对过的实验数字，以及与系列内其他篇目和 [WM 记忆系列](../world_memory/WM_01_why_memory/)、[VG 视频生成系列](../video_generation/VG_01_diffusion_basics/)的连接。

这里的 forcing 指机器学习中的序列生成技术，重点是自回归视频扩散。名称相近的方法可能改变完全不同的环节，阅读时需要分别辨认。

## 1. 先建立一张问题地图

设序列为 $x_{1:T}$，条件为 $c$。自回归模型通常写成：

$$
p_\theta(x_{1:T}\mid c)=\prod_{t=1}^{T}p_\theta(x_t\mid x_{1:t-1},c).
$$

Teacher Forcing 在训练时提供数据中的真实前缀 $x_{1:t-1}$；推理时，模型只能使用自己生成的 $\hat{x}_{1:t-1}$。模型会遇到训练时较少出现的历史，这是 exposure bias 的核心背景。它不意味着 Teacher Forcing 本身无效，也不意味着所有生成错误都来自这一差异。

对于视频扩散，还存在第二条轴：**视频时间与去噪时间**。同一个视频位置可以经历多次去噪，不同位置也可以处于不同噪声等级。再加上 KV cache 和蒸馏教师，就形成了本系列持续追踪的五个问题：

| 观察维度 | 每读一篇都要问的问题 | 代表工作 |
| --- | --- | --- |
| 历史来源 | 条件来自真实数据、混合历史，还是模型自己的 rollout？ | Scheduled Sampling、Self Forcing |
| 噪声与时间 | 每帧噪声是否独立？哪些帧在一起去噪？ | Diffusion Forcing、Rolling Forcing |
| 教师与目标 | 师生看见的信息是否一致？初始化与细化如何配合？ | CausVid、Causal Forcing、Causal-rCM |
| 历史记忆 | 保留哪些 KV？如何响应提示切换？ | LongLive、Deep Forcing、Context Forcing、Sparse Forcing |
| 梯度职责 | 未来损失能否训练历史表示？记忆写入和去噪是否冲突？ | Self Gradient Forcing、SGF+ |

这些是组织阅读的视角，并不是所有论文共享同一个目标函数。尤其要区分：**独立噪声等级、因果注意力、自生成历史训练是三个不同的设计选择。**

## 2. 阅读前需要哪些基础？

- **自回归与 Transformer**：条件概率分解、causal mask、Teacher Forcing、KV cache。
- **扩散与 Flow Matching**：加噪、去噪、score、采样轨迹，以及视频 latent 表示。
- **蒸馏与优化**：ODE 轨迹蒸馏、DMD、一致性模型、stop-gradient 和梯度截断；可在进入相应章节时再补充。

已有博客中的 [生成模型入门]({{< relref "/post/MITS6184/MITS6184_1" >}}) 可以作为扩散分支的前置阅读。[Latent Diffusion]({{< relref "/post/video_generation/LatentDiffusion" >}})、[CogVideoX]({{< relref "/post/video_generation/CogVideoX" >}}) 和 [WAN-Video]({{< relref "/post/video_generation/WAN-Video" >}}) 可作为相关模型的资料入口。

## 3. 完整目录：五个部分，21 篇论文

### 第 1 部分：训练与推理的差异

先理解真实历史与生成历史的差异，再看历史替换如何适应 Transformer 的并行训练。

| 编号 | 论文导读 | 核心问题 | 本地论文版本 |
| --- | --- | --- | --- |
| 01 | [Scheduled Sampling]({{< relref "/post/forcing_01" >}}) | 从 Teacher Forcing 到模型自己的历史 | [1506.03099v3](https://arxiv.org/abs/1506.03099v3) |
| 02 | [Scheduled Sampling for Transformers]({{< relref "/post/forcing_02" >}}) | 为什么 Transformer 需要两遍解码 | [1906.07651v1](https://arxiv.org/abs/1906.07651v1) |
| 03 | [Parallel Scheduled Sampling]({{< relref "/post/forcing_03" >}}) | 让历史替换适应并行训练 | [1906.04331v2](https://arxiv.org/abs/1906.04331v2) |

### 第 2 部分：Diffusion 与 Self Forcing

从逐 token 噪声出发，进入历史 guidance、双向教师蒸馏和模型自身 rollout。

| 编号 | 论文导读 | 核心问题 | 本地论文版本 |
| --- | --- | --- | --- |
| 04 | [Diffusion Forcing]({{< relref "/post/forcing_04" >}}) | 每个 token 都有自己的噪声等级 | [2407.01392v4](https://arxiv.org/abs/2407.01392v4) |
| 05 | [History-Guided Video Diffusion]({{< relref "/post/forcing_05" >}}) | 把历史帧作为可调节的 guidance | [2502.06764v2](https://arxiv.org/abs/2502.06764v2) |
| 06 | [CausVid]({{< relref "/post/forcing_06" >}}) | 从双向教师走向流式因果学生 | [2412.07772v4](https://arxiv.org/abs/2412.07772v4) |
| 07 | [Self Forcing]({{< relref "/post/forcing_07" >}}) | 在自己的生成历史上训练 | [2506.08009v2](https://arxiv.org/abs/2506.08009v2) |

### 第 3 部分：长视频与上下文

分别研究去噪窗口、长视频训练、短片段监督、缓存管理与教师上下文。

| 编号 | 论文导读 | 核心问题 | 本地论文版本 |
| --- | --- | --- | --- |
| 08 | [Rolling Forcing]({{< relref "/post/forcing_08" >}}) | 滚动联合去噪如何抑制误差传播 | [2509.25161v1](https://arxiv.org/abs/2509.25161v1) |
| 09 | [LongLive]({{< relref "/post/forcing_09" >}}) | 长视频训练与实时提示切换 | [2509.22622v2](https://arxiv.org/abs/2509.22622v2) |
| 10 | [Self-Forcing++]({{< relref "/post/forcing_10" >}}) | 用短片段监督长 rollout | [2510.02283v1](https://arxiv.org/abs/2510.02283v1) |
| 11 | [Deep Forcing]({{< relref "/post/forcing_11" >}}) | 无需微调的历史缓存管理 | [2512.05081v1](https://arxiv.org/abs/2512.05081v1) |
| 12 | [Context Forcing]({{< relref "/post/forcing_12" >}}) | 让教师也看见长历史 | [2602.06028v1](https://arxiv.org/abs/2602.06028v1) |

### 第 4 部分：少步蒸馏与初始化

先对齐师生因果条件，再比较少步日程、一致性初始化、单步对抗目标与统一蒸馏流程。

| 编号 | 论文导读 | 核心问题 | 本地论文版本 |
| --- | --- | --- | --- |
| 13 | [Causal Forcing]({{< relref "/post/forcing_13" >}}) | 师生因果结构为何影响 ODE 初始化 | [2602.02214v5](https://arxiv.org/abs/2602.02214v5) |
| 14 | [Diagonal Distillation]({{< relref "/post/forcing_14" >}}) | 沿时间与噪声两条轴分配计算 | [2603.09488v2](https://arxiv.org/abs/2603.09488v2) |
| 15 | [Causal Forcing++]({{< relref "/post/forcing_15" >}}) | 用在线一致性蒸馏初始化因果学生 | [2605.15141v4](https://arxiv.org/abs/2605.15141v4) |
| 16 | [One-Forcing]({{< relref "/post/forcing_16" >}}) | 单步生成的清晰度与动态性 | [2605.23458v2](https://arxiv.org/abs/2605.23458v2) |
| 17 | [Causal-rCM]({{< relref "/post/forcing_17" >}}) | 统一一致性初始化与分布匹配细化 | [2606.25473v1](https://arxiv.org/abs/2606.25473v1) |

### 第 5 部分：效率与上下文梯度

最后从架构、稀疏记忆和梯度三个角度，追问模型如何高效使用并学习历史。

| 编号 | 论文导读 | 核心问题 | 本地论文版本 |
| --- | --- | --- | --- |
| 18 | [Separable Causal Diffusion]({{< relref "/post/forcing_18" >}}) | 把时间推理与逐帧去噪拆开 | [2602.10095v1](https://arxiv.org/abs/2602.10095v1) |
| 19 | [Sparse Forcing]({{< relref "/post/forcing_19" >}}) | 让持久记忆与局部稀疏注意力一起训练 | [2604.21221v1](https://arxiv.org/abs/2604.21221v1) |
| 20 | [Self Gradient Forcing]({{< relref "/post/forcing_20" >}}) | 未来损失能否监督历史记忆写入 | [2607.20368v2](https://arxiv.org/abs/2607.20368v2) |
| 21 | [SGF+]({{< relref "/post/forcing_21" >}}) | 分离上下文写入与去噪的梯度职责 | [2610.10429v2](https://arxiv.org/abs/2610.10429v2) |

## 4. 按目标选择阅读路线

### 第一次接触 forcing

按 **01 → 02 → 03 → 04 → 05 → 06 → 07** 阅读。先搞清楚历史来源、独立噪声、双向与因果架构，再读长视频和蒸馏分支。

### 重点研究实时视频蒸馏

按 **04 → 06 → 07 → 13 → 14 → 15 → 16 → 17** 阅读，比较各方法的师生结构、初始化目标、采样步数、首帧延迟与质量。

### 重点研究长视频与记忆

以 **07** 为起点，阅读 **08–12**，再读 **19 → 20 → 21**。其中 08–12 是不同的长视频路线，不应默认它们之间存在直接继承关系。

### 重点研究梯度流

按 **07 → 20 → 21** 阅读：先理解 Self Forcing 的 rollout 与梯度截断，再理解 SGF 的上下文梯度重建，最后看 SGF+ 如何处理角色之间的梯度冲突。**18** 提供相关的架构职责拆分视角。

## 5. 每篇精读的统一骨架

21 篇正文沿用同一结构，便于交叉对照：

1. **解决什么问题**：动机与前人方法的缺陷。
2. **最小背景**：只补该篇特有前置，扩散/蒸馏基础见 [VG 01](../video_generation/VG_01_diffusion_basics/)。
3. **方法总览**：pipeline 逐模块描述。
4. **核心原理完整推导**：目标函数、损失、梯度从零推，论文公式保留原编号。
5. **训练与推理细节**：数据、条件注入、采样流程、超参（标本收折叠块）。
6. **实验与消融**：只写核对过的数字。
7. **与其他篇目的关系**：系列内互链 + WM/VG 系列连接。
8. **局限与个人评价**。

比较不同论文时记住：FPS、首帧延迟和生成时长对应不同评价维度；不同设置下的数字不能直接排成一个排行榜。长时生成演示也需要与广泛的长期质量评价分开理解。

## 6. 文献范围与版本

本系列对应 2026 年 10 月 14 日整理的 21 份本地 PDF。所有原文链接均保留所收集的 arXiv 版本号；后续修订解读时，可以显式记录版本变化。

目录中的序号是建议学习顺序，不代表发表顺序。每篇论文导读都能通过文末导航返回总览，或继续阅读前后篇；文章也会统一收录在博客的 **Forcing** 分类下。

{{< forcing_nav >}}
