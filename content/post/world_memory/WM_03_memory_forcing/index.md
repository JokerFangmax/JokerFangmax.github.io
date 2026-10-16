---
title: "WM 03: Memory Forcing——时空记忆的两难与训练协议"
date: 2026-10-24
math: true
tags:
  - Notes
  - World Models
  - Memory
  - Video Generation
  - Minecraft
categories:
  - World Models
description: "Memory Forcing（arXiv:2510.03198，CUHK-SZ 等）精读：temporal-only 记忆利于新场景探索、加 spatial memory 重访才一致——两难的根源是模型不分场合地依赖单一记忆。解法三件套：Hybrid Training 用两种数据分布教模型分场合（VPT 探索 temporal / MineDojo 重访 spatial）；Chained Forward Training 把模型自己的预测链进后续窗口当条件，制造更大位姿变化（teacher forcing 训练时条件误差恒 0、推理时随机游走无界增长——训练分布对推理 drift 覆盖率 0%）；Geometry-indexed Spatial Memory 用 point-to-frame 检索 + 增量 3D 重建（VGGT + 跨窗尺度对齐最小二乘闭解），7.3× 检索加速、98.2% 存储削减。全配置 FVD 165.9 vs 微调基线 366.1。附 toy：CFT 链长 4/16 的 drift 覆盖率 43.8%/74.3%，temporal 条件误差与 spatial 对齐后误差之比随 rollout 从 1.2× 涨到 18.5×——drift 越大越该信 spatial"
---

# WM 03：Memory Forcing——时空记忆的两难

> 📖 系列第三篇，[WM 02](../WM_02_vmem/) 的直接续篇。VMem 证明了「按几何相关性检索」比「按时间近邻」好；本篇补上被它跳过的一环：**检索到的记忆再好，模型不会用也白搭——「什么时候信哪种记忆」本身要进训练目标**。
> 论文：*Memory Forcing: Spatio-Temporal Memory for Consistent Scene Generation on Minecraft*（arXiv:2510.03198，CUHK-Shenzhen / HKU / 微软亚研 等）。
>
> 📂 **阅读方式**：正文只留主线；推导与数字走查收折叠块。

---

## 1. 这一篇解决什么问题

Minecraft 游玩式世界模型：给定过去帧 + 玩家动作（25 维：移动、转向、跳跃……），自回归生成下一帧。上下文窗口有限（16 帧），于是**记忆怎么分配**成为核心问题。论文 Fig. 1 给出两个对称的失败模式：

| 范式 | 重访已探索区域 | 新场景探索 | 失败原因 |
|------|--------------|-----------|---------|
| **长程 spatial memory**（WorldMem 等） | ✅ 一致 | ❌ 内容劣化、崩回训练集地形 | 模型**过度依赖**检索到的空间记忆——新场景里检索结果是空/不相关的，模型仍硬套 |
| **temporal-only**（Oasis / NFD 等） | ❌ 几何漂移 | ✅ 生成质量好 | 上下文只有最近帧，重访时相关视图早被挤出窗口（[WM 02](../WM_02_vmem/) §6.2 的机制版） |

这是**记忆的探索-利用两难**：探索要的是生成先验的自由，重访要的是记忆的约束。已有工作把它当架构问题（加什么记忆模块）；本文的诊断是**训练问题**：

1. **单一数据分布教不出分场合**：只在「频繁重访」的数据上训练，模型学不会「新场景时忽略 spatial 记忆」；
2. **teacher forcing 系统性低估 drift**：训练时条件帧全是 GT（位姿误差恒 0），推理时 AR rollout 的位姿误差是随机游走（无界增长）——模型从未见过「条件帧位姿有偏差」的世界，测试时遇到 drift 就乱。

一句话：**Memory Forcing = 教模型分场合用记忆（Hybrid Training）+ 把推理时的 drift 提前喂进训练（Chained Forward Training）**，外配一个高效几何记忆当弹药库。

---

## 2. 所需背景最小集

- **[WM 02](../WM_02_vmem/)**：几何索引检索的范式（surfel 投票）；本篇的 point-to-frame 检索是它的近亲；
- **Diffusion Forcing**：视频序列的每帧赋**独立噪声级** $k_t \in [0,1]$（而非整段同一级）——干净帧和噪声帧可以混在一个序列里训练。这是 CFT 能「把预测帧当条件」的框架基础；
- **AR 视频 DiT**：DiT 骨干 + 时空 self-attention + adaLN-zero 注入动作 + 3D 位置编码（沿用 NFD 架构）；
- **[WM 01](../WM_01_why_memory/) §1.3**：训练时看不见的演化在损失里没有梯度——本篇是「训练目标决定记忆行为」这一论点的第一个正面案例。

---

## 3. 方法总览：三件套

**一句话主线**：DiT 生成器带两个记忆入口——temporal（最近帧，窗口内 self-attention）与 spatial（检索到的历史帧，memory cross-attention）；Hybrid Training 用两种数据分布分别模拟探索与重访两种玩法状态，Chained Forward Training 在训练中链入模型自己的预测以放大位姿漂移；底层用流式 3D 重建维护一个 point-to-frame 可检索的几何记忆。

| 组件 | 做什么（白话） | 解决什么 |
|------|---------------|---------|
| **Memory Cross-Attention** | 每个 DiT 块加一层 cross-attn：当前帧 token 当 query，检索到的历史帧当 key/value，两侧都拼 Plücker 坐标编码相对位姿 | 给 spatial 记忆一个显式注入口 |
| **Hybrid Training** | 窗口 16 帧对半分：8 帧固定为最近 temporal 上下文；另外 8 帧**按数据来源分配**——VPT（人类游玩，探索向）给更早的 temporal 帧，MineDojo（模拟，频繁重访）给检索的 spatial 帧 | 两种数据分布 = 两种玩法状态的课程，教模型分场合 |
| **Chained Forward Training（CFT）** | 训练时逐窗口滑动，窗口末帧由模型自身少步去噪生成（无梯度），链入后续窗口当条件 | 制造真实的推理期 drift 分布 |
| **Point-to-Frame Retrieval** | 当前帧可见的 3D 点回溯到来源帧，取被引用最多的 top-8 帧当 spatial 记忆 | $O(1)$ 检索，与序列长度无关 |
| **Incremental 3D Reconstruction** | VGGT 流式估计深度 + 跨窗口尺度对齐，选择性关键帧 + 体素降采样 | 记忆规模随空间覆盖而非时长增长 |

其中 memory cross-attention（论文 Eq. 2，$\tilde{Q}$、$\tilde{K}_{\text{spatial}}$ 为拼了 Plücker 坐标的增广 query/key）：

$$
\mathrm{Attention}\big(\tilde{Q},\ \tilde{K}_{\text{spatial}},\ V_{\text{spatial}}\big) = \mathrm{Softmax}\!\left(\frac{\tilde{Q}\,\tilde{K}_{\text{spatial}}^\top}{\sqrt{d}}\right) V_{\text{spatial}}
\tag{4.0}
$$

检索到的历史帧只走 key/value 通路（不进 self-attention 的序列拼接）——spatial 记忆是**只读的旁路**：窗口内的 temporal 帧管「短期连贯」，旁路的 spatial 帧管「长期锚定」，两条通路在每块注意力后融合。

---

## 4. 核心机制完整推导

### 4.1 Hybrid Training：窗口的分配律

固定窗口 $L$ 帧，前 $L/2$ 帧固定为最近 temporal 上下文 $T_{\text{fixed}}$，后 $L/2$ 按场景分配（论文 Eq. 3）：

$$
W = \big[T_{\text{fixed}},\ M_{\text{context}}\big] = \begin{cases} \big[T_{\text{fixed}},\ M_{\text{spatial}}\big] & \text{重访已观测区域} \\[2pt] \big[T_{\text{fixed}},\ T_{\text{extended}}\big] & \text{探索新场景} \end{cases}
\tag{4.1}
$$

实现上不显式判定「重访与否」，而是**让数据来源决定**：MineDojo 合成数据（构造时就是频繁回访 + 相邻视角）配 $M_{\text{spatial}}$，VPT 人类游玩数据（天然探索向）配 $T_{\text{extended}}$。模型从统计上学会：视野与检索记忆相关时用记忆，不相关时靠 temporal。

<details>
<summary><b>为什么「对半分」而不是全窗口动态分配？</b></summary>

全动态分配会让两种记忆争夺全部预算，训练信号互相打架（重访数据希望窗口全给 spatial、探索数据希望全给 temporal——梯度方向相反）。固定一半给 temporal 保证探索能力永远有保底训练量；只在另一半上做课程切换。这与 mixture-of-experts 的「共享专家 + 路由专家」同构：共享的 temporal 上下文承担普适的短期连贯，路由的另一半按场景切换记忆源。

</details>

### 4.2 Chained Forward Training：把 drift 喂进训练

CFT 的训练目标（论文 Eq. 4，Diffusion Forcing 框架下）：

$$
\mathcal{L}_{\text{chain}} = \frac{1}{T}\sum_{j=0}^{T-1} \mathbb{E}_{t,\,\epsilon}\Big[\big\|\epsilon - \epsilon_\theta\big(W_j(x, \hat{x}),\ C_j,\ t\big)\big\|^2\Big], \quad t \sim U(0, T_{\text{noise}})
\tag{4.2}
$$

其中窗口 $W_j$ 覆盖帧 $[j,\ j+W-1]$：位置 $k$ 上,若 $k$ 已被早先窗口预测过（$k \in F_{\text{pred}}$）就用预测帧 $\hat{x}_k$，否则用 GT $x_k$；条件 $C_j = \{A_j, P_j, M_{\text{spatial}}\}$（动作、位姿、检索的空间记忆）。每个窗口末帧由该窗口自身**少步、无梯度**地去噪生成，存入 $F_{\text{pred}}$ 供后续窗口使用——预测像链一样一节节传下去。

<details>
<summary><b>Algorithm 1 逐步拆解（论文伪代码的逐行走查）</b></summary>

```
输入: 视频 x, 条件 C, 前进步数 T, 窗口 W, 模型 ε_θ
1  F_pred ← ∅, L_total ← 0
2  for j = 0 .. T-1:                       # 窗口滑动, 一次前进一格
3-10   构造 W_j: 帧 k ∈ [j, j+W-1] 上, k ∈ F_pred 用预测帧 x̂_k, 否则用 GT x_k
11    L_j ← ||ε - ε_θ(W_j, C_j, t)||², 累加                # 有梯度: 训练损失
12    if j < T-1:
13      x̂_{j+W-1} ← denoise(W_j, C_j)                      # 无梯度: 少步去噪
14      F_pred[j+W-1] ← x̂_{j+W-1}                          # 链入后续窗口
15  return L_chain ← L_total / T
```

三个容易看漏的细节：

1. **窗口每次只前进一帧**（$j \to j+1$），不是整窗跳——所以 $F_{\text{pred}}$ 里的预测帧是被最多 $W-1$ 个后续窗口反复当条件用的，drift 的影响被充分采样进损失；
2. **损失与生成解耦**：第 11 行带梯度（训练信号），第 13 行无梯度（只造条件，不传播误差链）——生成帧的「错」通过当条件间接影响后续窗口的损失，而不是直接进损失项。这与 GAN 的前向类似：合成数据用于制造训练状态，监督仍在真值上；
3. **噪声级 $t \sim U(0, T_{\text{noise}})$ 逐窗独立采样**（Diffusion Forcing 框架）——同一序列里不同窗口以不同噪声水平训练，等于天然的数据增广。

</details>

**两个机制，两个目的**：

1. **链式预测制造更大位姿变化**：预测帧的位姿带误差，后续窗口从「错位的位置」出发——模型被迫在「temporal 条件已经不可靠」的状态下完成任务，此时唯一跨窗口一致的信息源是 spatial 记忆（全局坐标系、不受链式误差污染）→ 学会重访时信 spatial；
2. **缩小训练-推理分布错配**：推理时 AR rollout 的条件帧全是模型自己的输出；CFT 让训练时也出现大量预测帧当条件。

<details>
<summary><b>teacher forcing 到底低估了多少？（机制走查）</b></summary>

把生成器抽象成「位姿估计器」：每步输出帧的位姿估计带独立误差 $e_{\text{step}} \sim \mathcal{N}(0, \sigma^2)$，rollout 误差 $e_t = \sum_{\tau \le t} e_{\text{step}}$ 是随机游走，$\mathbb{E}|e_t| \propto \sigma\sqrt{t}$。于是：

- **teacher forcing 的训练分布**：条件帧全 GT → 条件位姿误差是 $t=0$ 处的质点分布（恒 0）；
- **推理的真实分布**：条件窗口（$L$ 帧）的平均误差 $\propto \sigma\sqrt{t}$，随 $t$ 无界增长；
- **错配**：推理时遇到的任何非零 drift 都在训练分布的支撑集之外——不是「少见」，是**从未见过**。

CFT 链长 $n$ 时，训练里链入预测帧的误差 $\approx$ 其条件窗口误差 + 新增生成误差，量级 $\sigma\sqrt{nL}$——训练分布的支撑集随链长扩展。这就是 §6.3 toy 要定量验证的对象。

顺带一提：CFT 与 self-forcing / 推理时搜索的关系在 [VG 10](../../video_generation/VG_10_video_t1/) 的 AR rollout 语境里已经出现过——同一族「训练-推理分布错配」问题在记忆场景的实例化。后两篇 [WM 04](../WM_04_relic/)（self-forcing 蒸馏）与 [WM 05](../WM_05_worldplay/)（Context Forcing）把这条线推到实时。

</details>

### 4.3 几何记忆：point-to-frame 与尺度对齐

**检索**（论文 Eq. 5）：当前帧 $t$ 可见的 3D 点集 $P^t_{\text{visible}}$（把全局点云投影到当前位姿），每个点带来源帧索引 $\mathrm{source}(p_i)$，取被引用最多的 top-8 帧：

$$
H_t = \arg\max_{k=1,\dots,8}\ \mathrm{Count}\big(\mathrm{source}(p_i) : p_i \in P^t_{\text{visible}}\big)
\tag{4.3}
$$

与 [WM 02](../WM_02_vmem/) 的 surfel 投票同族——区别在投票粒度：VMem 投「哪个视图看见过我这块 surfel」（渲染属性图统计），这里投「哪个帧是可见点的来源」（点云投影统计）。**检索复杂度与序列长度无关**：可见点数有界（分辨率 × 体素密度上限），top-8 的候选统计是常数开销。

**关键帧判定**（论文 Eq. 6）：一帧成为关键帧，当且仅当它揭示了未观测区域，或历史上下文不足：

$$
\mathrm{IsKeyframe}(t) = \mathrm{NovelCoverage}(I_t, G_{\text{global}})\ \ \text{or}\ \ \big(|H_t| < \tau_{\text{hist}}\big), \qquad \tau_{\text{hist}} = L/2
\tag{4.4}
$$

**跨窗口尺度对齐**（附录 A.1）：VGGT 输出**相对**深度，各窗口尺度不一。对齐用窗口间的重叠帧：置信度过滤（双侧 > 0.1 + 各自 top 60% 百分位）后，最小二乘解新窗口深度到全局尺度的缩放因子——有闭式解：

$$
s^* = \frac{\sum_{i \in V} D_{\text{old},i}\, D_{\text{new},i}}{\sum_{i \in V} D_{\text{new},i}^2}
\tag{4.5}
$$

这就是普通最小二乘 $s^* = \arg\min_s \sum_i (D_{\text{old},i} - s\,D_{\text{new},i})^2$ 的正规方程解。**置信度加权 + 闭式缩放让全局几何误差有界**——不随序列长度累积，这是 spatial 记忆相对 temporal 链的可靠性来源（§6.3 toy 的第三组数字）。

**写入**：深度图反投影（外参由四元数位姿合成，论文 Eq. 7）→ 空间感知体素降采样——任意位姿区域的点密度有上限，**记忆消耗随空间覆盖而非时间长度增长**。

---

## 5. 训练与推理细节

| 项 | 值 |
|----|-----|
| 骨干 | DiT + 时空 self-attn + adaLN-zero（动作）+ memory cross-attn（每块） |
| 帧化 | 2D VAE 16× 空间压缩，每帧 24×14 连续 token，384×224 分辨率 |
| 训练 | ~400k 步，24 GPU，batch 4/GPU，Adam lr 4e-5（H20/H100） |
| 数据 | VPT（人类游玩，25 维动作，剔除无动作/GUI 帧）+ MineDojo 合成（11k 视频 × 1500 帧，WorldMem 配置：频繁重访 + 相邻视角），合计 3500+ 小时 |
| 评测数据 | 长程记忆 150×1500 帧（WorldMem 协议）；泛化 150×800 帧 × 9 未见地形；生成 300×800 帧（前 600 帧全动作 + 后段限行动作） |
| 公平性 | 全部基线 16 帧窗口、同一合成数据、500–600k 步 |

---

## 6. 实验与消融

### 6.1 主结果：两难的两端同时守住

| 方法 | 长程记忆 FVD ↓ / LPIPS ↓ | 泛化 FVD ↓ / LPIPS ↓ | 生成 FVD ↓ / LPIPS ↓ |
|------|------------------------|---------------------|---------------------|
| Oasis（temporal-only） | 196.8 / 0.379 | 477.3 / 0.512 | 285.7 / 0.470 |
| NFD（temporal-only） | 220.8 / 0.389 | 442.6 / 0.464 | 349.6 / 0.434 |
| WorldMem（spatial） | 122.2 / 0.277 | 328.3 / 0.434 | 290.8 / 0.453 |
| **Memory Forcing** | **84.9 / 0.216** | **253.7 / 0.290** | **185.9 / 0.303** |

关键读法：**WorldMem 的两难在表里清晰可见**——长程记忆好（122.2）但泛化差（328.3）；Memory Forcing 三列全优，且泛化列对 WorldMem 的优势（328.3 → 253.7）证明「分场合」训练成功教会了模型在新场景里少用 spatial 记忆（Fig. 4 定性图：WorldMem 在冻结海洋地形上生成出训练集的平原样子——spatial 记忆的反噬）。

**效率**（4000 帧序列）：WorldMem 检索 4.27 FPS、存 4000 帧；本篇 31.21 FPS、存 72.65 帧——**7.3× 检索加速、98.2% 存储削减**，且差距随序列变长拉大（3000–3999 帧段 25.7×）：WorldMem 的记忆库随时长线性增长，几何记忆只随空间覆盖增长。

**消融**（按论文 Table 3 / 消融叙述重组）：

| 配置 | FVD ↓ | LPIPS ↓ | 说明 |
|------|-------|---------|------|
| 全参微调 + pose 检索 | 366.1 | 0.412 | 模型在两种记忆间摇摆，顾此失彼 |
| + Hybrid Training | 230.4 | 0.360 | 分场合的数据有了,但 drift 下不会用 spatial |
| + CFT（= MF 完整训练） | 225.9 | 0.372 | 训练协议完整 |
| + 3D 检索换 pose 检索 | **165.9** | **0.288** | 检索质量最后一锤定音 |

两个设计缺一不可：训练协议（366.1 → 225.9）与几何检索（225.9 → 165.9）贡献相当——**给模型好字典（检索）和教它查字典（协议）是两件事**，正好对应本篇与 [WM 02](../WM_02_vmem/) 的分工。

### 6.2 记忆机制谱系（论文附录 Table 4 整理）

把本篇放进 Minecraft 世界模型的记忆谱系里（[WM 01](../WM_01_why_memory/) §3 坐标系的实例化）：

| 方法 | 记忆类型（复杂度） | 存储 | 记忆范围 | 训练数据 |
|------|------------------|------|---------|---------|
| Oasis / MineWorld / NFD | 无显式记忆 | — | 上下文窗口内 | VPT 4000+ h |
| LSVM（状态空间） | 隐状态压缩 $O(1)$ | 循环隐状态 | 受训练长度限制 | TECO 800+ h |
| VRAG（相似度 RAG） | 相似度检索 $O(1)$ | 定长缓冲 | 受缓冲大小限制 | MineRL 30+ h |
| WorldMem（位姿 RAG） | 位姿检索 $O(n)$ | 全部帧 | 长程 | MineDojo 800+ h |
| **Memory Forcing** | 3D 检索 $O(1)$ | 随空间覆盖稀疏存储 | 长程 | VPT + MineDojo 3500+ h |

谱系的演化方向清晰：**从「无记忆」到「有记忆」，再到「记忆的复杂度与序列长度解耦」**——本篇用几何锚定把检索复杂度从 $O(n)$ 压到 $O(1)$，付出的代价是要维护一套流式 3D 重建（VGGT + 尺度对齐）。这个「显式几何换复杂度」的交易在 [WM 04](../WM_04_relic/) 会被反向执行一次：RELIC 把记忆压进 KV cache 的 latent token,不要显式几何——实时优先的另一种取舍。

### 6.3 toy：drift 分布错配的定量

脚本 `wm03_toy.py`，把生成器抽象成位姿链（每步估计误差 $\sigma = 0.15$ 格，窗口 $L=16$，rollout $T=1500$）。三组数字：

**(A) 推理时 temporal 条件误差无界增长**（窗口内平均 $|e_t|$ 的分位数）：

| $t$ | P50 | P95 |
|-----|-----|-----|
| 16 | 0.27 | 0.72 |
| 256 | 1.54 | 4.91 |
| 1499 | 4.00 | **11.23** |

**(B) 训练分布对推理 drift 的覆盖率**（推理条件误差 ≤ 训练分布 95 分位的概率）：

| 训练协议 | q95 | 总覆盖率 | 链覆盖范围内 |
|---------|-----|---------|-------------|
| Teacher Forcing | 0.000 | **0.0%** | — |
| CFT（链 4 窗 = 64 帧） | 1.92 | 43.8% | 前 64 步 98.4% |
| CFT（链 16 窗 = 256 帧） | 4.34 | 74.3% | 前 256 步 97.7% |

teacher forcing 的训练分布是 $\delta_0$——推理 rollout 的**任何**非零 drift 都在支撑集外，覆盖率精确为 0。CFT 链长指数级扩展支撑集：链 16 覆盖推理四分之三的 drift 分布。这组数字是「teacher forcing 低估 drift」的定量版，也给出 CFT 的设计指南——**链长决定 drift 覆盖半径**。

**(C) 为什么 drift 大 → 该信 spatial**：temporal 条件误差（无界）vs spatial 对齐后残差（$\sigma_{\text{align}} = 0.30$ 有界）的 P95 之比：

| $t$ | temporal P95 | spatial P95 | 倍数 |
|-----|-------------|-------------|------|
| 16 | 0.72 | 0.59 | 1.2× |
| 256 | 4.91 | 0.56 | 8.7× |
| 1499 | 11.23 | 0.61 | **18.5×** |

rollout 开始时两者相当；越往后 temporal 越不可靠而 spatial 不变。**CFT 专门在大 drift 区间训练 = 把模型暴露在「temporal 已废、spatial 是唯一信号」的状态下**——学出来的行为自然是 drift 大时切换到 spatial 记忆。论文正文那句 "chained predictions create larger pose variations and encourage reliance on spatial memory" 的机制版。

> 数字与理论的核对：随机游走误差 $\mathbb{E}|e_t| = \sigma\sqrt{2t/\pi} \approx 0.12\sqrt{t}$，窗口 16 帧平均再打约八折——$t=1499$ 处理论值 ≈ 3.7，实测 P50 = 4.00，吻合；链 16 的训练分布 q95 = 4.34 ≈ $\sigma\sqrt{nL}$（= 0.15 × 16 = 2.4 的窗口均值 + 生成误差累积），量级自洽。模拟随机游走的输出先对闭式量级做一遍核对再进正文。

---

## 7. 与其他论文的联系

| 连接 | 说的是什么 |
|------|-----------|
| [WM 02](../WM_02_vmem/) | 几何检索同族（surfel 投票 ↔ point-to-frame）；VMem 只解决了「检索什么」，本篇解决「何时信检索」——消融显示两者贡献相当 |
| [WM 04](../WM_04_relic/) / [WM 05](../WM_05_worldplay/) | CFT 是 self-forcing 蒸馏与 Context Forcing 的直系前驱：同一族「链入自身预测以对齐训练-推理分布」的技术,后两篇推向 16–24 FPS 实时 |
| [VG 10](../../video_generation/VG_10_video_t1/) | AR rollout 误差累积的语境；teacher forcing 低估 drift 是其训练侧版本 |
| [WM 08](../WM_08_wrbench/) | 本篇的「scaling 不如协议对」是 WRBench scaling 反例（Wan 1.3B→14B 重访状态反降）的先声——结构性缺陷不随参数量消失 |
| WorldMem | 两难的 spatial 极端；效率对比的直接对象（$O(n)$ 检索 + 全帧存储 vs $O(1)$ + 稀疏关键帧） |
| [WM 01](../WM_01_why_memory/) §1.3 | 「训练目标决定记忆行为」：Hybrid/CFT 是给「看不见时的行为」补梯度的第一次系统尝试——不过本篇补的是**相机位姿**的 drift,不是**世界状态**的演化（后者要等 WM 10–14） |

---

## 8. 局限与思考

- **静态画布仍在**：Minecraft 地形不变，本篇的「一致性」仍是 WM 01 三层中的第一层（静态外观）；实体状态演化（第三层） untouched；
- **两难是调和不是消除**：Hybrid Training 用数据分布教「分场合」，但场合判定是隐式学的——边界情况（半新半旧的区域、检索结果部分相关）没有显式控制；显式的状态分解要等 [WM 10](../WM_10_liveworld/)；
- **Minecraft-only + 384×224**：论文自认域适配与分辨率是主要限制；
- **$O(1)$ 检索的前提**：可见点数有界依赖体素密度上限——极端多房间大场景里「空间覆盖」本身也会膨胀（虽然远慢于时长）；
- **CFT 链长的预算问题**：toy 显示链 16 才覆盖 74%——更长序列需要更长链，训练成本线性上升；链长与序列长度的最优配比论文未讨论；
- 面试可能被问的点：**为什么不在推理时直接检测 drift、切换记忆源？**——drift 检测需要 GT 位姿（推理时没有）；隐式学到的「分场合」是唯一不依赖外部信号的做法。以及 **CFT 与 self-forcing 的区别**：CFT 链入的是**少步无梯度的预测帧**且保留 GT 混合（$W_j$ 里两种帧并存），self-forcing 通常整段自回归——混合比例本身是课程变量。

---

*下一篇 [WM 04：RELIC](../WM_04_relic/) 把这套记忆 + 蒸馏推向实时：14B 模型、单图起步、16 FPS——KV cache 里的压缩记忆如何同时带住相对动作与绝对位姿。*
