---
title: "Forcing: 12. Context Forcing"
date: 2026-10-14
math: true
tags:
  - "Forcing"
  - "Video Generation"
  - "Context Forcing"
categories:
  - Forcing
series: forcing
weight: 12
description: "Context Forcing（arXiv:2602.06028v1）精读：流式长视频调优的师生错配——无记忆教师只看 5 秒窗口，学生上下文被教师视野封顶，陷入 forgetting–drifting 两难。解法：长上下文教师蒸馏长上下文学生，全局 KL 链式分解（Eq. 2）导出两阶段课程与 CDMD 目标；Slow-Fast Memory（sink 3 + slow 12 + fast 6，共 21 latent 帧）加有界位置编码把有效上下文做到 20 秒以上，比 LongLive（3.0s）/FramePack-F1（9.2s）长 2–10×；60 秒 VBench 背景一致性 95.34、主体一致性 94.88 均第一（Table 1/2）。"
paper_id: "2602.06028v1"
---

# Forcing 12：Context Forcing——让教师也看见长历史

> **Forcing 系列 · 第 12 篇 · 第三部分「长视频与上下文」收尾（教师与目标维度）**
> 上一篇：[Deep Forcing（11）](../forcing_11/) ｜ 下一篇：[Causal Forcing（13）](../forcing_13/)
> 阅读前提：扩散与 DMD 蒸馏基础见 [VG 01](../video_generation/VG_01_diffusion_basics/)、[VG 06](../video_generation/VG_06_videoscene/)。正文只留主线，长推导、超参标本与消融全表收进折叠块（点击展开）。

## 论文信息

- **原题**：Context Forcing: Consistent Autoregressive Video Generation with Long Context
- **阅读版本**：arXiv:2602.06028v1，对应本地收集的 PDF 版本。
- **原文**：[arXiv 摘要页](https://arxiv.org/abs/2602.06028v1) · [PDF](https://arxiv.org/pdf/2602.06028v1)

---

## 1. 这篇论文解决什么问题

实时长视频生成的流行配方是「流式调优」：学生做长 rollout，教师却是**无记忆的短窗口模型**（典型 5 秒）。论文把这条配方的病灶点名为**师生错配（student–teacher mismatch）**：教师看不到长期历史，就无法在全局时间依赖上指导学生——学生的可学习上下文长度被教师的视野**封顶**。仅扩大学生自己的上下文窗口解决不了问题，反而落入两难：

- **Forgetting（遗忘）**：记忆窗口短 → 误差累积小，但长 rollout 中丢主体、丢场景；
- **Drifting（漂移）**：上下文长 → 身份保得住，但模型暴露在自己的累积误差前，没有教师能纠正长期偏差，分布逐渐漂离真实流形。

论文的实证锚点（Fig. 1）：SOTA 方法的有效上下文普遍只有 3.0–9.2 秒（LongLive 3.0s、Rolling Forcing 6.0s、FramePack-F1 9.2s、Infinity-RoPE 1.5s）；而 LongLive 在推理时把窗口从 3.0s 拉到 5.25s，得到的是误差累积与分布偏移——**窗口不是想拉就能拉的**。

解法一句话：**用长上下文教师蒸馏长上下文学生**，教师与学生共享同一套上下文记忆机制，再靠慢-快记忆架构把 20 秒以上的历史塞进有界计算预算。结果：有效上下文 20+ 秒（2–10× 于 SOTA），60 秒生成的一致性全面领先。

---

## 2. 最小背景

**因果自回归视频扩散**。长视频 $X_{1:N}$ 分解为 $\prod_t p(X_t \mid X_{<t})$（chunk 级块状因果注意力，chunk 内双向）；CausVid / Self Forcing 用 DMD 把双向教师蒸馏成因果学生（[VG 06](../video_generation/VG_06_videoscene/)）。

**DMD 一瞥**。学生分布 $p_\theta$ 对数据分布 $p_{\mathrm{data}}$ 做前向 KL 匹配，梯度由「真分数 − 假分数」之差驱动（详见第 4 节推导）。关键结构性事实：**KL 的条件分布必须两边同源**，否则匹配没有共同基准。

**三种训练范式的坐标系**（Fig. 2，本文的骨架）：

| 范式 | 教师 | 学生 | 问题 |
|------|------|------|------|
| (a) Self-Forcing | 短（5s） | 短（5s rollout） | 天生短程 |
| (b) 记忆无关长调优（LongLive） | 短且**无记忆** | 长 rollout | **本文攻击的错配**：教师在随机 chunk 上监督，看不到全局 |
| (c) Context 长调优（本文） | 长（带记忆） | 长（带记忆） | 教师知晓完整生成历史 |

---

## 3. 方法总览

系统由四个模块构成，师生**两侧同装**：

1. **Context Teacher（上下文教师）**：在「视频续写」任务上预训练的长上下文自回归模型——输入长前缀、续写后续片段。这不是原版双向底座，而是专门造出来的「看得见历史的教师」。再经 **ERFT（Error-Recycling Fine-Tuning）** 增强：向教师上下文注入真实漂移误差，训练它从被污染的历史中恢复正确速度场，保证学生漂移时教师的续写分布仍然可靠。
2. **两阶段课程**：阶段 1 局部分布匹配（短窗口 DMD，学局部动力学、产出高质量短上下文）；阶段 2 上下文分布匹配 **CDMD**（学生在**自己 rollout 出的长上下文**上续写，教师以同一上下文为条件提供真分数）。
3. **Slow-Fast Memory（慢-快记忆）**：KV cache 划分为 attention sink + 慢记忆 + 快记忆三段，用「惊讶度」策略决定哪些 token 晋级慢记忆，把线性增长的上下文压成固定预算。
4. **有界位置编码（Bounded Positional Indexing）**：所有 token 的时间 RoPE 位置被限制在固定区间内，与生成步数无关——长序列下注意力的稳定器。

两条配套训练策略：**长自 rollout 课程**（rollout 长度随训练步数线性增长，避免早期长 rollout 的分布崩塌）与**干净上下文策略**（上下文帧完整去噪、目标帧保留随机时间步退出，兼顾教师视野与梯度覆盖）。

---

## 4. 核心原理完整推导

### 4.1 目标：全局 KL 及其链式分解

目标是让因果模型 $p_\theta$ 的长视频分布匹配真实分布，$N$ 为几十到几百秒：

$$
\mathcal{L}_{\mathrm{global}} = \min_\theta\; \mathrm{KL}\big(p_\theta(X_{1:N}) \,\|\, p_{\mathrm{data}}(X_{1:N})\big)
\tag{Eq. 1}
$$

直接优化不可行（$N$ 太大）。用 KL 的链式法则在任意切点 $k$ 分解：

$$
\mathcal{L}_{\mathrm{global}} = \underbrace{\mathrm{KL}\big(p_\theta(X_{1:k}) \,\|\, p_{\mathrm{data}}(X_{1:k})\big)}_{\mathcal{L}_{\mathrm{local}}:\ \text{局部动力学}} + \underbrace{\mathbb{E}_{X_{1:k} \sim p_\theta}\Big[\mathrm{KL}\big(p_\theta(X_{k+1:N} \mid X_{1:k}) \,\|\, p_{\mathrm{data}}(X_{k+1:N} \mid X_{1:k})\big)\Big]}_{\mathcal{L}_{\mathrm{context}}:\ \text{全局续写动力学}}
\tag{Eq. 2}
$$

**推导直觉**：联合分布的 KL 等于「前缀分布的 KL」加上「在前缀条件下续写分布 KL 的期望」——概率链式法则逐项取对数比即可得，这一步把「整段视频像不像真的」拆成两个可分别优化的问题：前 $k$ 帧生成得好不好（局部），以及**在任意（学生自己造的）历史下续得对不对**（全局）。第二项正是 LongLive 范式 (b) 丢掉的东西：它的教师只会算 5 秒窗口内的条件分布，$\mathcal{L}_{\mathrm{context}}$ 里跨长时间的条件项没人监督。两阶段课程由此自然导出：阶段 1 优化 $\mathcal{L}_{\mathrm{local}}$，阶段 2 优化 $\mathcal{L}_{\mathrm{context}}$。

### 4.2 阶段 1：局部分布匹配

$k$ 对应 1–5 秒窗口，教师分布 $p_T(X_{1:k})$ 近似真实数据：

$$
\mathcal{L}_{\mathrm{local}} = \mathrm{KL}\big(p_\theta(X_{1:k}) \,\|\, p_T(X_{1:k})\big)
\tag{Eq. 3}
$$

DMD 式梯度估计。设 $x = G_\theta(z)$（$z$ 为噪声），$x_t$ 是 $x$ 的加噪版本，$\alpha_t$ 为噪声调度的信号系数，$w_t$ 为权重函数：

$$
\nabla_\theta \mathcal{L}_{\mathrm{local}} \approx \mathbb{E}_{z, t, x_t}\Big[ w_t \alpha_t \big(s_\theta(x_t, t) - s_T(x_t, t)\big) \frac{\partial G_\theta(z)}{\partial \theta} \Big]
\tag{Eq. 4}
$$

**从零推这一步**：把 KL 提升到每个噪声级 $t$ 上的加噪分布之间（$p_{\theta,t}$ 与 $p_{\mathrm{data},t}$），则 $\nabla_\theta \mathrm{KL} = -\mathbb{E}[(\nabla_x \log p_{\theta,t}(x_t) - \nabla_x \log p_{\mathrm{data},t}(x_t))\, \partial x / \partial \theta]$——两个对数概率的梯度就是两个分数函数，而 $p_{\mathrm{data},t}$ 的分数由冻结教师 $s_T$ 近似、$p_{\theta,t}$ 的分数由在线训练的假分数模型 $s_\theta$ 给出；$\alpha_t w_t$ 是把分数差换算回数据空间梯度的加权。**直觉**：教师与学生对同一张噪声图「指的方向不同」，把差向量沿着学生生成器的雅可比回传，就把学生的输出分布往教师（≈数据）方向推。此阶段保证 $p_\theta(X_{1:k}) \approx p_{\mathrm{data}}(X_{1:k})$——为阶段 2 产出可靠的上下文来源。

### 4.3 阶段 2：CDMD（Contextual Distribution Matching）

$\mathcal{L}_{\mathrm{context}}$ 里 $p_{\mathrm{data}}(X_{k+1:N} \mid X_{1:k})$ 对学生造出的任意上下文不可得。两个假设把它接到教师上：

- **假设 1（教师在学生上下文附近可靠）**：只要学生上下文 $X_{1:k}$ 贴近真实流形，教师的续写 $p_T(X_{k+1:N} \mid X_{1:k})$ 就是准的——教师在真实视频前缀上训练充分时成立；
- **假设 2（前缀近似真实）**：阶段 1 已把 $p_\theta(X_{1:k})$ 对齐到 $p_{\mathrm{data}}(X_{1:k})$，保证阶段 2 训练时学生的 rollout 仍落在教师的可靠区。

于是 $p_{\mathrm{data}} \approx p_T$，得到 CDMD 目标：

$$
\mathcal{L}_{\mathrm{CDMD}} = \mathbb{E}_{X_{1:k} \sim p_\theta(X_{1:k})}\Big[\mathrm{KL}\big(p_\theta(X_{k+1:N} \mid X_{1:k}) \,\|\, p_T(X_{k+1:N} \mid X_{1:k})\big)\Big]
\tag{Eq. 5}
$$

**注意期望的取法**：$X_{1:k}$ 从**学生自己的分布**采——学生在自己造的历史上被监督，曝光偏差被显式压进目标里（Self Forcing 思想从 5 秒窗口推广到任意长上下文）。梯度是条件版 DMD：设 $x_{\mathrm{cont}} = G_\theta(z_{\mathrm{cont}} \mid X_{1:k})$，真假分数模型跑在**同一个学生生成的上下文**上：

$$
\nabla_\theta \mathcal{L}_{\mathrm{CDMD}} \approx \mathbb{E}_{X_{1:k} \sim p_\theta}\mathbb{E}_{z_{\mathrm{cont}}, t}\Big[ w_t \alpha_t \big(s_\theta(x_{t,\mathrm{cont}}, t \mid X_{1:k}) - s_T(x_{t,\mathrm{cont}}, t \mid X_{1:k})\big) \frac{\partial G_\theta(z_{\mathrm{cont}} \mid X_{1:k})}{\partial \theta} \Big]
\tag{Eq. 6}
$$

与 Eq. 4 的唯一区别是所有分数都**以同一长上下文为条件**——这一条就是「师生信息一致性」的数学落点：教师打分时看到的世界必须与学生生成时看到的世界相同，否则 Eq. 5 的 KL 对的是两个不同条件下的分布。

<details>
<summary><b>展开：ERFT——教师为什么还要再训一遍</b></summary>

标准训练以真值历史为条件，推理却跑在自生成历史上（曝光偏差）。Context Teacher 用 **Error-Recycling Fine-Tuning** 显式处理：构造被污染的上下文

$$
\tilde{X}_{1:k} = X_{1:k} + \mathbb{I} \cdot e_{\mathrm{drift}}
$$

其中 $e_{\mathrm{drift}}$ 从**历史模型残差库**中采样（真实的漂移误差样本），$\mathbb{I}$ 是伯努利指示变量；教师被优化为从 $\tilde{X}_{1:k}$ 恢复正确速度 $v_{\mathrm{target}}$。这保证学生漂移时 $p_T(\cdot \mid X_{1:k})$ 仍是 $p_{\mathrm{data}}$ 的可靠代理——假设 1 的「附近」被人为撑大了。消融（Fig. 7）：以 5 秒真值视频起自回归 rollout 30 秒，有 ERFT 的教师伪影明显更少，为后续蒸馏提供更好的分布。

</details>

### 4.4 Slow-Fast Memory：惊讶度晋级 + 有界位置

**三段划分**。总 cache $M = S \cup C_{\mathrm{slow}} \cup L_{\mathrm{fast}}$（互斥并集）：attention sink $S$（前 $N_s$ 个 token，稳定注意力，沿 StreamingLLM）；慢记忆 $C_{\mathrm{slow}}$（上限 $N_c$，存高熵关键帧，只在有显著新信息时更新）；快记忆 $L_{\mathrm{fast}}$（大小 $N_l$ 的滚动 FIFO，保即时局部上下文）。

**惊讶度晋级（Surprisal-Based Consolidation）**。新 token $x_t$ 入队快记忆时，比较它与前一 token 的 key 相似度：

$$
\pi(x_t) = \begin{cases} \text{Consolidate（晋级慢记忆）} & \text{if } \mathrm{sim}(k_t, k_{t-1}) < \tau \\ \text{Discard（丢弃）} & \text{otherwise} \end{cases}
\tag{Eq. 7}
$$

**直觉**：与前一 token 高度相似 → 信息冗余（低惊讶度）；不相似 → 状态跳变/视觉变化（高惊讶度），值得进长期记忆。慢记忆只存「时间梯度」与标志性事件，不存静态冗余；满了按 FIFO 驱逐最旧项。这等价于把「均匀采样历史」换成「按信息量采样历史」。

**有界位置编码**。标准 AR 视频模型的位置索引无界增长（$p_t = t \to \infty$），长序列必撞分布偏移。本文把所有 token 的时间 RoPE 位置锁进固定区间 $\Phi = [0, N_s + N_c + N_l - 1]$，与生成步数 $t$ 无关：

$$
\phi(x) = \begin{cases} i \in [0, N_s - 1] & \text{if } x \in S \\ j \in [N_s, N_c - 1] & \text{if } x \in C_{\mathrm{slow}} \\ k \in [N_c, N_c + N_l - 1] & \text{if } x \in L_{\mathrm{fast}} \end{cases}
\tag{Eq. 8}
$$

效果是一个**静态注意力窗口**：新鲜历史（fast）在高索引区滑动，显著历史（slow）压缩进低索引区——注意力格局永远不超出训练见过的范围。消融（第 6 节）显示这是所有组件里权重最大的一根柱子。

---

## 5. 训练与推理细节

<details>
<summary><b>展开：实现与超参标本（师生两侧同装记忆系统）</b></summary>

| 项 | 值 |
|----|-----|
| 底座 | Wan2.1-T2V-1.3B（教师与学生同底座） |
| 教师数据 | Sekai + Ultravideo 过滤出 >10s 高质量视频，共 40k clips；上下文帧从视频 5–20s 区间均匀采样 |
| 教师训练 | 8k steps，batch size 8 |
| 阶段 1 | VidProM 81 帧短片段，600 iterations，batch 64 |
| 阶段 2 | rollout 视野 10–30 秒，VidProM，500 iterations，batch 64 |
| 记忆预算 | KV cache 21 latent 帧：$N_s = 3$、$N_c = 12$、$N_l = 6$ |
| 晋级阈值 | $\tau = 0.95$；每 2 个 chunk 做一次晋级判断，chunk 晋级时只保留**首个 latent**——等效上下文因此突破 20 秒 |

**阶段 2 的算法骨架（Algorithm 1，Contextual DMD）**：每次迭代采 prompt 与 rollout 长度 $L$（随训练进度增长），再采随机退出步 $r$。逐 chunk 生成 $L$ 个 chunk；位于目标窗口前 $r$ 长度内的 chunk 用「完整 $T$ 步去噪」（保证该段每个扩散步都有监督），其余 chunk 只去噪到第 $r$ 步；**仅退出步那一次前向开启梯度**，之后的 KV 重建全部无梯度。最后取上下文窗口 $v_c = X[L - r - l : L - l]$、目标 $v_t = \mathrm{addnoise}(X[L-l:], t)$，用 $s_{\mathrm{fake}}(v_t, t, v_c)$ 与 $s_{\mathrm{real}}(v_t, t, v_c)$ 算 CDMD 损失。两个细节：上下文帧**完整去噪**（干净上下文策略——上下文要「可被教师理解」，目标帧保留随机时间步——梯度要覆盖所有扩散级）；rollout 长度上限 $N^{(t)}_{\max}$ 随训练步线性增长、每次 $k \sim \mathcal{U}(k_{\min}, N^{(t)}_{\max})$（长自 rollout 课程——早期避开漂移最重的长 rollout）。

</details>

---

## 6. 实验与消融

**60 秒一致性细粒度评测**（Table 1，MovieGenBench 随机 100 prompt；DINOv2 结构身份、CLIP-F 语义上下文、CLIP-T 提示对齐；以首帧为锚、对任意时刻取 $[t-0.5s, t+0.5s]$ 窗口内平均余弦相似度，5 个随机种子取平均）。DINO 分数随时间衰减是主战场：

| 模型 | 有效上下文 | DINO @10s | @20s | @30s | @40s | @50s | @60s |
|------|-----------|-----------|------|------|------|------|------|
| FramePack-F1 | 9.2s | 81.86 | 77.95 | 78.84 | 73.10 | 69.52 | 68.50 |
| LongLive | 3.0s | 91.25 | 89.55 | 89.12 | 86.51 | 87.83 | 86.26 |
| Infinity-RoPE | 1.5s | 91.18 | 88.17 | 85.37 | 79.80 | 81.10 | 83.72 |
| **Ours（student）** | **20+s** | **91.45** | **89.25** | **89.66** | **87.45** | **88.33** | **87.89** |

ours 的 DINO 曲线几乎平坦（91.45 → 87.89），60 秒时领先 LongLive 1.63 分；CLIP-T（提示对齐）在 60s 还升到 37.66。背景/主体一致性 95.95 / 94.68，均为全场第一。

**VBench 总表**（Table 2，60 秒设置摘录）：

| 模型 | FPS | Total | Quality | Semantic | BG Cons. | Subj Cons. |
|------|-----|-------|---------|----------|----------|------------|
| Self Forcing（chunk） | 17.0 | 71.86 | 77.20 | 50.51 | 87.84 | 83.60 |
| Rolling Forcing | 15.8 | 79.31 | 81.87 | 67.69 | 94.12 | 93.10 |
| Infinity-RoPE | 17.0 | 79.99 | 80.81 | 74.30 | 94.21 | 93.05 |
| LongLive | 20.7 | 83.64 | 84.53 | 74.97 | 94.62 | 93.88 |
| **Ours（student）** | 17.0 | 82.45 | 83.55 | 76.10 | **95.34** | **94.88** |

要点：LongLive 的 Total/Quality 略高，但论文指出其高分背后是病态模式——频繁的**突然场景重置**与循环运动（附录 Fig. 8：第 524 帧后视频突然闪回初始帧，时间连续性被硬切断）；语义分（76.10）与一致性上 ours 全面领先。5 秒短视频设置上两者相当（83.44 vs 84.87 Total），长上下文没有牺牲短程质量。

<details>
<summary><b>展开：消融全表（Table 3，60 秒）</b></summary>

| 变体 | Total | Quality | Semantic | BG Cons. | Subj Cons. | Dyn. Degree |
|------|-------|---------|----------|----------|------------|-------------|
| 慢记忆均匀采样，间隔 1 chunk | 80.82 | 82.20 | 75.32 | 92.45 | 92.10 | 52.15 |
| 慢记忆均匀采样，间隔 2 chunks | 81.11 | 82.61 | 75.12 | 93.12 | 92.85 | 55.30 |
| w/o Contextual Distillation（只训练忆系统） | 80.36 | 82.28 | 72.70 | 93.55 | 93.20 | 48.12 |
| w/o Bounded Positional Encoding | 73.52 | 75.44 | 65.82 | 84.68 | 79.24 | 27.45 |
| **Ours（完整）** | **82.45** | **83.55** | **76.10** | **95.34** | **94.88** | **58.26** |

三条读法：

1. **有界位置编码是最大的单点**：去掉后 Total 掉 8.93、Dynamic Degree 从 58.26 跌到 27.45（腰斩还多）、主体一致性 94.88→79.24——位置索引无界增长对长程注意力的破坏是全局性的；
2. **记忆系统与蒸馏互补分工**：只装记忆系统不做 CDMD，背景/主体一致性已到 93.55/93.20（记忆结构本身就能稳住外观），但语义分停在 72.70、动态度 48.12——**内容「对不对、动不动」要靠长上下文教师监督**；
3. **相似度采样 > 均匀采样**：慢记忆按 key 相似度（选与当前窗口**低相似**的历史 chunk，保语义增量）优于两种固定间隔均匀采样。

</details>

---

## 7. 与系列其他篇目的关系

| 对照 | 本文 | 对方 | 分歧点 |
|------|------|------|--------|
| [Deep Forcing（11，上一篇）](../forcing_11/) | 监督侧：教师的**视野** | 缓存侧：历史的**存取** | 长视频的两条正交轴——11 说明零训练的缓存管理能走多远，本篇说明剩下的一段（教师看不到历史）必须靠训练补 |
| [Self-Forcing++（10）](../forcing_10/) | 长上下文教师**直接监督**长 rollout | 短片段教师在长 rollout 的**片段**上监督 | 片段监督仍丢跨片段全局约束；本篇把教师的上下文窗口本身做成研究对象 |
| Self Forcing / CausVid | CDMD = 把「自生成历史」从 5 秒推广到 20 秒+ | 自 rollout 限 5 秒 | Eq. 5 的期望取法与 Self Forcing 同构，只是 $k$ 不再受教师窗口限制 |
| [WM 05（WorldPlay）](../world_memory/WM_05_worldplay/) | 独立预训练的长上下文续写教师 | 蒸馏组件同样命名为 Context Forcing：教师条件 = 学生记忆上下文 − 生成段 | **同期撞名、思想同族**：都要求蒸馏时教师复现学生的记忆条件。构造不同——WorldPlay 让双向教师带着检索出的记忆条件打分（无需长视频训练），本文训练一个专门的长上下文教师。WorldPlay 的结果（蒸馏后 4 NFE 长程 PSNR 18.94 反超蒸馏前 100 NFE 的 16.27）与本文消融互相印证：**对齐上下文后，蒸馏不仅不伤长程一致性反而增益** |
| [VG 06](../video_generation/VG_06_videoscene/) | CDMD 是 DMD 的条件版推广 | DMD 家族全景 | 本篇给 DMD 加的约束：条件分布必须师生同源 |
| [Causal Forcing（13，下一篇）](../forcing_13/) | 教师与学生的**上下文长度**要一致 | 教师与学生的**因果结构**要一致 | 同一个「师生信息一致性」命题的两个投影；13 进入第四部分「少步蒸馏与初始化」 |

系列视角：第三部分到这里收束。11 与 12 合起来给出长视频的完整命题——**历史既要存得进（缓存管理），也要被监督者看见（教师视野）**；二者缺一，学生要么漂移要么被教师的短视封顶。

---

## 8. 局限与个人评价

**论文自认**：记忆压缩策略在**信息密度**上仍有优化空间，未来应做可学习的上下文压缩与自适应记忆机制。

**顺着系列主线的四条评价**：

1. **「教师的视野封顶学生的上限」是本篇最可迁移的论点**。它把 DMD 的隐含前提（条件分布同源）第一次摆到长上下文场景下显式处理：Eq. 5 里若教师的条件是 5 秒窗口、学生的条件是 20 秒历史，KL 对齐的就是两个不同的条件分布——这在形式上与 [WM 05](../world_memory/WM_05_worldplay/) 的记忆条件对齐、下一篇的因果结构对齐构成一个「师生信息一致性」三连。
2. **两个假设有轻微的循环性风险**。假设 2（阶段 1 保证前缀近似真实）支撑假设 1（教师在前缀附近可靠），而阶段 2 又会把学生的分布往教师拉——若教师本身在长上下文续写上有系统偏差，这套循环没有外部纠错机制。论文用 ERFT 部分对冲（教师对漂移上下文鲁棒），但没有检验教师对「20 秒以上的学生上下文」的续写质量随时间的衰减曲线；Table 1 里教师自己的 DINO@10s（87.61）其实低于学生（91.45），教师的可靠性边界值得单独刻画。
3. **Bounded Positional Encoding 的消融数字（Dyn. Degree 58.26→27.45）是全文最硬的证据，但也是最「便宜」的组件**。它不涉及任何训练目标创新，纯粹是推理/输入侧的位置索引重排——与 [Deep Forcing（11）](../forcing_11/) 的 Temporal RoPE Adjustment、WorldPlay 的 Temporal Reframing 构成三连击：**三篇论文独立发现长程失败的最大单点都在时间位置编码**。系列应该把「RoPE 时间外推」记为长视频问题的第一堵墙。
4. **评测上的诚实与不足并存**。论文主动指出 LongLive 高分背后的闪回伪影（Fig. 8），这是好品味；但自身的 60 秒评测依赖 DINO/CLIP 的首帧锚定相似度，对「内容正确演化」不敏感（静态画面也能拿高分）——[Deep Forcing（11）](../forcing_11/) 对 Dynamic Degree 的强调恰好是反面提醒。两篇合读才得到长视频评测的完整拼图。

下一篇 [Causal Forcing（13）](../forcing_13/) 换一个维度重启战线：不再问「教师看多长的历史」，而问「教师是否与学生同样因果」——并给出第四部分「少步蒸馏与初始化」的理论地基。

{{< forcing_nav >}}
