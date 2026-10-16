---
title: "Forcing: 13. Causal Forcing"
date: 2026-10-14
math: true
tags:
  - "Forcing"
  - "Video Generation"
  - "Causal Forcing"
categories:
  - Forcing
series: forcing
weight: 13
description: "Causal Forcing（ICML 2026，arXiv:2602.02214v5）精读：双向教师蒸馏 AR 学生时，ODE 初始化违反帧级单射性——同一噪声帧对应多个干净帧，MSE 回归塌缩到条件期望（Prop. 3.3）导致模糊。三阶段解法：teacher forcing 训 AR 教师（VisionReward 3.343 vs DF 的 1.583）→ 因果 ODE 蒸馏 → Self Forcing 同款 DMD。全面超越 Self Forcing：Dynamic Degree 68 vs 57（+19.3%）、VisionReward 6.326（+8.7%）、指令遵循 56（+16.7%），吞吐 17.0 FPS 不变（Table 1/2）。"
paper_id: "2602.02214v5"
---

# Forcing 13：Causal Forcing——ODE 初始化的因果单射性

> **Forcing 系列 · 第 13 篇 · 第四部分「少步蒸馏与初始化」开篇**
> 上一篇：[Context Forcing（12）](../forcing_12/) ｜ 下一篇：[Diagonal Distillation（14）](../forcing_14/)
> 阅读前提：扩散与 PF-ODE 见 [VG 01](../video_generation/VG_01_diffusion_basics/)，DMD 蒸馏见 [VG 06](../video_generation/VG_06_videoscene/)；[CausVid（06）](../forcing_06/) 与 [Self Forcing（07）](../forcing_07/) 是本文批判的对象，建议先读。正文只留主线，长证明、超参标本与消融全表收进折叠块（点击展开）。

## 论文信息

- **原题**：Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation
- **阅读版本**：arXiv:2602.02214v5（ICML 2026，PMLR 306），对应本地收集的 PDF 版本。
- **原文**：[arXiv 摘要页](https://arxiv.org/abs/2602.02214v5) · [PDF](https://arxiv.org/pdf/2602.02214v5)

---

## 1. 这篇论文解决什么问题

实时交互视频生成的标准配方（[CausVid（06）](../forcing_06/)、[Self Forcing（07）](../forcing_07/)）是**非对称蒸馏**：拿一个预训练双向视频扩散模型当教师，蒸馏一个少步自回归（AR）学生。相比普通的步数蒸馏，这里多了一道**架构鸿沟（architectural gap）**——把看得见未来的双向注意力换成只依赖过去的因果注意力。

经验事实（Fig. 1）：同样从双向底座蒸馏，SOTA 的 AR 蒸馏（Self Forcing）在视觉质量、Dynamic Degree、指令遵循上仍显著落后于「标准 DMD 蒸馏双向学生」。这篇论文回答两个问题：

1. **这道鸿沟该在哪个阶段填**？——论文用受控实验证明 DMD 阶段填不了，必须在更早的 **ODE 初始化**阶段填（Fig. 2：先用标准 DMD 初始化 AR 学生，消除步数差距、只留架构差距，性能仍显著落后）。
2. **为什么现有 ODE 初始化在理论上就是错的**？——因为 MSE 回归式的 ODE 蒸馏要求配对数据**单射**：每个噪声样本唯一对应一个干净样本。双向教师的 PF-ODE 只在**视频级**单射，不在**帧级**单射；而 AR 学生学的是帧级映射。于是同一个噪声帧对应多个干净帧，回归解塌缩成**条件期望**——也就是模糊。

解法（Causal Forcing 三阶段）：**teacher forcing 训练一个原生 AR 扩散教师 → 用它做因果 ODE 蒸馏初始化 → 接 Self Forcing 同款非对称 DMD**。结果：同等训练预算下全面超越 Self Forcing——Dynamic Degree +19.3%、VisionReward +8.7%、指令遵循 +16.7%，吞吐不变（17.0 FPS）。

---

## 2. 最小背景

**扩散与 PF-ODE**。前向过程 $x_t = \alpha_t x_0 + \sigma_t \epsilon$；取 $\alpha_t = 1-t,\ \sigma_t = t,\ T=1$ 的流匹配调度，速度目标 $v_t = \epsilon - x_0$。采样即解概率流 ODE：

$$
\mathrm{d}x_t = v_\theta(x_t, t)\,\mathrm{d}t, \qquad x_T \sim \mathcal{N}(0, I), \quad t: T \to 0
\tag{Eq. 1}
$$

**AR 视频扩散**。$p_\theta(x^{1:N}_0) = \prod_{i=1}^{N} p_\theta(x^i_0 \mid x^{<i}_0)$，每个条件由标准扩散建模；实践中按 chunk 生成（本文 chunk = 3 latent 帧）。两种训练范式：

| 范式 | 条件 | 训练时前缀 | 推理时前缀 |
|------|------|-----------|-----------|
| **Teacher Forcing（TF）** | $p(x^i_0 \mid x^{<i}_0)$ | 干净前缀（干净视频与噪声拼接 + 因果 mask） | 干净（自生成）前缀 —— **一致** |
| **Diffusion Forcing（DF，[04](../forcing_04/)）** | $p_{\mathrm{DF}}(x^i_0 \mid x^{<i}_t)$ | 每帧独立加噪的前缀 | 干净前缀 —— **不一致** |

**ODE 蒸馏**。一致性蒸馏（CD）的简化版：学生 $G_\theta$ 直接回归教师 PF-ODE 轨迹上的干净端点，$\theta^* = \min_\theta \mathbb{E}_{t, x_t}[\|G_\theta(x_t, t) - x_0\|^2]$，其中 $(x_t, x_0)$ 在教师同一条轨迹上。

**DMD**。分数蒸馏的 KL 视角（[VG 06](../video_generation/VG_06_videoscene/)），梯度由真假分数差驱动：

$$
\nabla_\theta \mathbb{E}_t\big[D_{\mathrm{KL}}(p_{\theta,t} \| p_{\mathrm{data},t})\big] = -\mathbb{E}_{\tilde{x}, t, \tilde{x}_t}\big[(s_{\mathrm{real}}(\tilde{x}_t, t) - s_{\mathrm{fake}}(\tilde{x}_t, t))\, \partial \tilde{x} / \partial \theta\big]
\tag{Eq. 2}
$$

---

## 3. 方法总览

Causal Forcing 是一条三阶段流水线，每一段都有明确的理论职责：

1. **阶段 1：teacher forcing AR 扩散训练**。从双向底座出发，用干净前缀做条件训练一个**多步**原生 AR 扩散模型。它将充当 ODE 蒸馏的教师。选择 TF 而非 DF 有理论依据（Prop. 3.4，见第 4.5 节）：DF 的「噪声前缀条件」与推理时的「干净前缀」分布不一致。
2. **阶段 2：因果 ODE 蒸馏**。用阶段 1 的 AR 教师采样 PF-ODE 轨迹：从真实数据取干净历史 $x^{<i}_{\mathrm{gt}}$，教师在其条件下从高斯噪声解 ODE 生成当前帧，存下各时间步的中间态；学生在**同一干净历史**下从中间态回归干净端点。因为教师是 AR 的，其 PF-ODE 天然满足帧级单射性——学生学到的是真正的流映射，而不是条件期望。
3. **阶段 3：非对称 DMD**。与 Self Forcing 完全相同：学生以自生成前缀做自 rollout，双向底座当真分数教师（$s_{\mathrm{real}}$）、在线假分数模型当 $s_{\mathrm{fake}}$。区别只在初始化——阶段的接力从「双向教师的 ODE 轨迹」换成了「AR 教师的因果 ODE 轨迹」。

另有一个理论外延：**因果一致性蒸馏（causal CD）**——同一帧级单射性原则套到 CD 框架（Eq. 9），证明「非对称 CD」（双向教师 + 因果学生）同样违反单射性，实验中大幅劣于因果 CD。

---

## 4. 核心原理完整推导

本节是全文的理论心脏，也是系列 14–17 篇的地基，从零推。

### 4.1 先把病灶定位：DMD 阶段填不了架构鸿沟

Self Forcing 的两阶段里，架构转换发生在 ODE 蒸馏阶段（双向教师 → AR 学生），DMD 只是后续打磨。受控实验（Fig. 2）：把 AR 学生直接用**标准 DMD 蒸馏的双向少步模型**初始化——步数差距被消除、只剩架构差距——性能仍显著落后标准 DMD。结论：**初始化时留下的大架构鸿沟，DMD 阶段补不回来**；要填就得在 ODE 初始化时填对。

### 4.2 ODE 蒸馏的隐含前提：单射性

ODE 蒸馏是 MSE 回归：给定配对数据 $(x_t, x_0)$（同一条教师轨迹上），最小化 $\|G_\theta(x_t, t) - x_0\|^2$。**从零推 MSE 解**：对每个固定的 $(x_t, t)$，对 $\|G_\theta(x_t,t) - x_0\|^2$ 逐点求导置零，平方误差的最优解是条件期望（回归的经典结论，Bishop 2006）——这正是论文给出的塌缩形态：

$$
G^*_\theta(x^i_t, x^{<i}_t, t) = \mathbb{E}[x_0 \mid x^i_t, x^{<i}_t, t]
\tag{Eq. 5}
$$

于是有两种世界：

- **配对是函数**（$x_t$ 唯一决定 $x_0$）：条件期望退化成那个函数本身，$G^*_\theta$ = 教师流映射，学生学得干净；
- **配对不是函数**（同一 $x_t$ 对应多个 $x_0$）：条件期望把这些 $x_0$ **平均**——多帧平均在视觉上就是模糊。

所以 ODE 蒸馏的必要条件是**单射性**（Liu et al. 2022）：任何时间步上，每个噪声样本唯一对应样本空间中的一个干净样本。蒸馏双向学生时这天然成立（视频级：PF-ODE 的流映射 $\phi_{\mathrm{Bi}}: (x^{1:N}_t, t) \mapsto x^{1:N}_0$ 是良定义函数）。但 AR 学生逐帧生成，映射单位是**帧**——单射性必须下沉到帧级：

**定义 3.1（帧级单射性）**。对映射 $\phi_{\mathrm{AR}}: (x^i_t, t) \mapsto x^i_0$（条件历史省略），若对任意两个噪声视频 $\{x^j_t\}, \{y^j_t\}$：

$$
\forall i \in [N]:\quad x^i_t = y^i_t \;\Rightarrow\; \phi_{\mathrm{AR}}(x^i_t, t) = \phi_{\mathrm{AR}}(y^i_t, t)
\tag{Eq. 4}
$$

即 $\phi_{\mathrm{AR}}(x^i_t, t)$ 是把 $x^i_t$ 映到第 $i$ 个干净帧的良定义函数。

### 4.3 为什么双向教师必然违反帧级单射性（Lemma 3.2）

**引理 3.2（非正式）**。设 $x^{1:N}_t$ 满足双向模型的 PF-ODE，$x^{\mathrm{other}}_t := x^{[N] \setminus \{i\}}_t$ 为除第 $i$ 帧外的其余帧。若流映射的第 $i$ 帧分量 $\phi_{\mathrm{Bi}}(x^{1:N}_t, t)^i$ **不对 $x^{\mathrm{other}}_t$ 几乎处处取常值**，则

$$
\forall t \in (0, 1],\ \forall x^{1:N}_t,\ \exists\, y^{1:N}_t:\quad y^i_t = x^i_t \ \text{且}\ \phi_{\mathrm{Bi}}(x^{1:N}_t, t)^i \ne \phi_{\mathrm{Bi}}(y^{1:N}_t, t)^i
\tag{Eq. 6}
$$

且碰撞发生在非零测度上：$\mathbb{P}\big(\mathrm{Var}(\phi_{\mathrm{Bi}}(x^{1:N}_t, t)^i \mid x^i_t, t) > 0\big) > 0$。

**证明思路（附录 B.1）**：固定 $(x^i_t, t)$，把 $\phi_{\mathrm{Bi}}$ 的第 $i$ 帧输出看成其余帧的函数 $f_{x^i_t, t}(z) := \phi_{\mathrm{Bi}}((x^i_t, z), t)^i$。若 $f$ 不几乎处处为常数，则对任何 $z_1$ 都存在 $z_2$ 使 $f(z_2) \ne f(z_1)$——取 $y_t = (x^i_t, z_2)$ 即得碰撞。再由 $x_T \sim \mathcal{N}(0,I)$ 密度处处为正，$z_2$ 附近有不可数多个同类点，碰撞集非零测。而「$f$ 不几乎处处常数」由架构保证：DiT 的注意力让第 $i$ 帧的去噪依赖**所有**帧（已有注意力图实验佐证），改动未来帧必然改动第 $i$ 帧的输出。

**直觉**：双向模型去噪第 $i$ 帧时**看得见未来**。固定 $x^i_t$，换一组未来帧 $x^{>i}_t$，干净端点 $x^i_0$ 就跟着变——可 AR 学生推理时根本没有 $x^{>i}_t$。监督信号里被丢掉的那部分信息，正是碰撞的来源。

### 4.4 推论：Self Forcing 的 ODE 初始化学到的是条件期望（Prop. 3.3）

Self Forcing 的 ODE 目标（配对来自**双向**教师的 PF-ODE，学生以历史 $x^{<i}_t$ 为条件）：

$$
\theta^* = \min_\theta\, \mathbb{E}_{t,\, x^{1:N}_t,\, i}\big[\|G_\theta(x^i_t, x^{<i}_t, t) - x^i_0\|^2\big]
\tag{Eq. 3}
$$

**命题 3.3**。该回归的最优解不服从数据分布：

$$
G^*_\theta(x^i_t, t) = \mathbb{E}[x^i_0 \mid x^i_t, t] \;\nsim\; p_{\mathrm{data}}(x^i_0)
\tag{Eq. 7}
$$

**证明（附录 B.1，用二阶矩恒等式，值得完整看一遍）**。记 $U := x^i_t$、$Y := x^i_0$、$\hat{Y} := \mathbb{E}[Y \mid U, t]$。由 $L^2$ 正交投影恒等式：

$$
\mathbb{E}\|Y\|^2 = \mathbb{E}\|\hat{Y}\|^2 + \mathbb{E}[\mathrm{Var}(Y \mid U, t)]
\tag{Eq. 18}
$$

Lemma 3.2 给了 $\mathbb{P}(\mathrm{Var}(Y \mid U, t) > 0) > 0$，故 $\mathbb{E}[\mathrm{Var}(Y \mid U, t)] > 0$，于是 $\mathbb{E}\|Y\|^2 > \mathbb{E}\|\hat{Y}\|^2$。若 $\hat{Y}$ 与 $Y$ 同分布，则两者的 $\mathbb{E}\|\cdot\|^2$ 必相等——矛盾。所以条件期望的分布**不是**数据分布：它系统性缺掉一块二阶矩，缺的那块正是被平均掉的帧间方差——视觉上即模糊与不一致。$\blacksquare$

**直觉总结**：这不是「训练不充分」的工程问题，而是目标函数的最优解本身就不在数据流形上——再多数据、再多步数，MSE 都会把你推向同一个错误答案。

### 4.5 阶段 1 的选型依据：TF 优于 DF（Prop. 3.4）

要造 AR 教师，先要在 TF 与 DF 之间选。**命题 3.4**：AR 场景下 DF 训出的模型不服从干净前缀条件下的数据分布——存在分布错配：

$$
\mathbb{E}_{y \sim p_{\mathrm{data}}(x^{<i}_0)}\Big[D_{\mathrm{KL}}\big(p_{\mathrm{DF}}(x^i_0 \mid y)\,\|\,p_{\mathrm{data}}(x^i_0 \mid y)\big)\Big] > 0
\tag{Prop. 3.4}
$$

<details>
<summary><b>展开：Prop. 3.4 的证明思路（马尔可夫链 + 条件期望算子的不动点）</b></summary>

记 $Y := x^{<i}_0$（干净前缀）、$X := x^i_0$、$Z := x^{<i}_t$（$Y$ 经前向核独立加噪得到）。DF 模型学的是 $p_{\mathrm{DF}}(X \mid Z)$；推理时却在干净前缀处查询：$p_{\mathrm{DF}}(x \mid y) = p_{\mathrm{DF}}(x \mid Z = y) = p_{\mathrm{data}}(x \mid Z = y)$（由假设 A1：DF 训到了最优条件分布）。于是只需证 $\mathbb{E}_Y[D_{\mathrm{KL}}(p_{\mathrm{data}}(X \mid Z = Y) \,\|\, p_{\mathrm{data}}(X \mid Y))] > 0$。

反证：若上述期望为零，则对几乎处处的 $y$，$p_{\mathrm{data}}(X \mid Z = y) = p_{\mathrm{data}}(X \mid Y = y)$。另一方面，独立加噪给出马尔可夫链 $X \to Y \to Z$（$X$ 只通过 $Y$ 影响 $Z$），塔性质给出 $p_{\mathrm{data}}(X \in A \mid Z = y) = \mathbb{E}[f_A(Y) \mid Z = y]$（$f_A(y) := p_{\mathrm{data}}(X \in A \mid Y = y)$）。两式合并：$f_A(y) = \mathbb{E}[f_A(Y) \mid Z = y]$——$f_A$ 成了「加噪条件期望算子」的不动点。由正则性假设 A3（后验核非退化），该算子的有界不动点只有几乎处处常数，于是 $f_A$ 对一切可测 $A$ 几乎处处常数，即 $X \perp Y$——与假设 A2（同一视频的帧彼此相关）矛盾。$\blacksquare$

**直觉**：把前缀加噪会**不可逆地抹掉**前缀里与当前帧相关的信息；条件在加噪前缀上训练的模型，在干净前缀处查询时答非所问。注意边界（附录 C.1）：这个批评只针对「AR 训练里用噪声前缀当条件」的用法；DF 原本为**双向**视频续写设计（续写时输入本来就是干净前缀 + 噪声拼接，训练推理一致），不受此限。

</details>

### 4.6 阶段 2 的正确目标：因果 ODE 蒸馏（Eq. 8）

用 AR 教师采样配对数据：从真实数据取干净历史 $x^{<i}_{\mathrm{gt}}$，教师在其条件下从 $x^i_T \sim \mathcal{N}(0, I)$ 解 ODE 生成当前帧，轨迹存为 $\{x^i_t\}_{t \in \mathcal{S} \cup \{0\}}$；学生在同一历史下回归：

$$
\theta^* = \min_\theta\, \mathbb{E}_{x^{<i}_{\mathrm{gt}},\, t \in \mathcal{S},\, i,\, x^i_t}\big[\|G_\theta(x^i_t, x^{<i}_{\mathrm{gt}}, t) - x^i_0\|^2\big]
\tag{Eq. 8}
$$

**为什么它满足帧级单射性**：历史 $x^{<i}_{\mathrm{gt}}$ 在配对中是**给定且固定**的；AR 教师的 ODE 是确定性微分方程，固定条件与初始状态后，$(x^i_t, t) \mapsto x^i_0$ 沿轨迹是良定义函数——Eq. 4 自动成立，Prop. 3.3 的塌缩不会发生，学生恢复的是教师的真流映射。这也解释了数据构造的一个细节：AR 教师自己生成当前帧（而非用真值），保证配对数据落在**教师自己的 ODE 轨迹**上。

<details>
<summary><b>展开：两个关键对照实验——错的是数据，不是初始化权重（附录 C.2/C.3）</b></summary>

- **多步 AR 直接当 DMD 初始化不够**（附录 C.2，Fig. 7/8，Table 4）：TF 训出的**多步** AR 模型直接初始化 DMD 已比 Self Forcing 的 ODE 初始化好得多（VisionReward 5.863 vs 3.330），但它在少步（4 步）采样下 chunk 间仍有突变——训练条件是高质量真值前缀，少步自回归生成的前缀质量下降，条件错配在 chunk 间累积。所以中间必须插一段因果 ODE 蒸馏把它变成少步模型（完整版 6.326 / 56）。
- **配对数据才是瓶颈**（附录 C.3，Fig. 9）：把学生初始化换回双向模型、但配对数据仍用 AR 教师生成的 $D_{\mathrm{Causal}}$——质量与 AR 初始化相当，仍远好于 Self Forcing 的 ODE 蒸馏。**性能差距来自配对数据的构造（教师该是 AR 的），不来自学生初始权重**。这是对「因果 ODE」这一主张最干净的隔离验证。

</details>

### 4.7 外延：因果一致性蒸馏（Eq. 9）

ODE 蒸馏可视为 CD 的简化版，单射性原则自然推广。用 AR 教师做 teacher forcing 式 CD：

$$
\theta^* = \min_\theta\, \mathbb{E}_{x_{\mathrm{gt}}, \epsilon, t, i}\big[w(t)\, d\big(G_\theta(x^i_t, x^{<i}_{\mathrm{gt}}, t),\; G_{\theta^-}(\hat{x}^i_{t-\Delta t}, x^{<i}_{\mathrm{gt}}, t - \Delta t)\big)\big]
\tag{Eq. 9}
$$

其中 $\hat{x}^i_{t-\Delta t}$ 由 AR 教师从 $x^i_t$ 解一步 ODE 得到，$\theta^-$ 是停止梯度的滑动平均。CD 需要边界条件 $G_\theta(x, 0) \equiv x$；流匹配的 v 预测参数化**自动满足**它，无需 wrapped 网络：

$$
G_\theta(x^i, x^{<i}_{\mathrm{gt}}, t) = x^i - t\, v_\theta(x^i, x^{<i}_{\mathrm{gt}}, t)
\tag{Eq. 31}
$$

（$t = 0$ 时 $\alpha_0 = 1$、$\sigma_0 = 0$，$x_0 = x - 0 \cdot v = x$。）反之，「非对称 CD」（双向教师 + AR 学生）同样撞上 Lemma 3.2，实验中高度模糊。这个外延为系列的 CD 路线（如 [Causal-rCM（17）](../forcing_17/)）埋了种子。

---

## 5. 训练与推理细节

<details>
<summary><b>展开：三阶段流水线的完整标本（附录 D）</b></summary>

| 阶段 | 数据 | 步数 | 要点 |
|------|------|------|------|
| ① TF AR 训练 | $D_{\mathrm{Bi}}$：Wan 双向模型用 VidProM prompt 合成的约 3K 样本 | 2K steps | 同时为基线 ODE 蒸馏存好噪声中间态 |
| ② 因果 ODE 蒸馏 | $D_{\mathrm{Causal}}$：AR 教师采样的 3K 条因果 ODE 轨迹 | 1K steps | teacher forcing，条件为 $D_{\mathrm{Bi}}$ 中对应真值；学生从 AR 教师初始化 |
| ③ 非对称 DMD | VidProM | 750 steps（至收敛） | $s_{\mathrm{real}}$ = Wan2.1-**14B**，$s_{\mathrm{fake}}$ = Wan2.1-1.3B，严格沿用 Self Forcing 协议 |

- **底座**：Wan2.1-T2V-1.3B，81 帧、832×480；chunk-wise 实现，每 chunk 3 个 latent 帧（frame-wise 设置另报）。
- **优化**：全程 batch 64，Adam，lr $2 \times 10^{-6}$，$\beta_1 = 0$、$\beta_2 = 0.999$，其余设置与 Self Forcing 一致。
- **推理**：4 步采样，时间步 $1,\ 0.9375,\ 0.8333,\ 0.625$（因果 ODE 与 DMD 共享）。
- **因果 CD 变体**：LCM 方案，48 个离散时间步，UniPC 求解器，EMA 率 0.99，$D_{\mathrm{Bi}}$ 上 3K steps，推理同样 4 步。
- **公平性**：基线（含 Self Forcing 的 ODE 初始化）与本文都用满 3K 步初始化预算；$D_{\mathrm{Bi}}$ 与 $D_{\mathrm{Causal}}$ 用同一批 prompt 内部合成，数据质量对齐。
- **评测**：VBench 主基准；VisionReward（子分可正可负，本文乘 100 报告）及其指令遵循子分；因 VBench prompt 普遍缺运动，另建 100-prompt 高运动集专测 Dynamic Degree / VisionReward / 指令遵循；10 人 10 prompt 用户研究排名。

</details>

---

## 6. 实验与消融

**主表（Table 1，同规模对比）**：

| 模型 | 类别 | FPS | Latency | Total | Quality | Semantic | Dyn.↑ | VisionReward↑ | Instruct.↑ | 用户排名↓ |
|------|------|-----|---------|-------|---------|----------|-------|----------------|------------|-----------|
| Wan2.1-1.3B | 双向 | 0.78 | 103 s | 83.37 | 84.30 | 79.65 | 61 | 5.275 | 42 | 2.29 |
| SkyReels-V2-1.3B | AR | 0.49 | 112 s | 81.97 | 83.96 | 74.01 | 37 | 3.584 | 32 | 6.57 |
| CausVid | 蒸馏 AR | 17.0 | 0.69 s | 81.33 | 83.98 | 70.72 | 62 | 5.741 | 12 | 4.27 |
| Self Forcing | 蒸馏 AR | 17.0 | 0.69 s | 83.74 | 84.48 | 80.77 | 57 | 5.820 | 48 | 2.87 |
| **Causal Forcing** | 蒸馏 AR | 17.0 | 0.69 s | **84.04** | **84.59** | **81.84** | **68** | **6.326** | **56** | **1.64** |

三个读数：(1) 对蒸馏 AR 同类，Dynamic Degree 68 vs 57（**+19.3%**）、VisionReward 6.326 vs 5.820（**+8.7%**）、指令遵循 56 vs 48（**+16.7%**），吞吐延迟分毫未动；(2) 对可比规模的双向 SOTA（Wan2.1），总分反超且吞吐高 **2079%**；(3) 对原生 AR 模型的最佳基线，Dynamic Degree +47.8%、VisionReward +56.0%、指令遵循 +75.0%。用户研究排名第一（1.64，越小越好）。

**消融（Table 2）**——每一行都是一次理论主张的实证：

| 设置 | Total | Dyn. | VisionReward | Instruct. |
|------|-------|------|--------------|-----------|
| DF 训练 AR 教师 | 81.76 | 60 | 1.583 | 30 |
| **TF 训练 AR 教师** | 82.12 | 50 | **3.343** | 32 |
| Self Forcing ODE + DMD（chunk） | 82.00 | 24 | 3.330 | 38 |
| **因果 ODE + DMD（chunk）** | **84.04** | **68** | **6.326** | **56** |
| Self Forcing ODE + DMD（frame） | 81.83 | 2 | 1.951 | −4 |
| **因果 ODE + DMD（frame）** | 83.75 | 64 | 6.204 | 42 |
| 非对称 CD | 79.07 | 59 | −7.983 | −42 |
| **因果 CD** | 81.48 | 51 | 1.798 | 18 |

<details>
<summary><b>展开：消融的三层读法与附录补充表</b></summary>

1. **TF vs DF（Prop. 3.4 的实证）**：TF 的 VisionReward 比 DF 高 **111.2%**（3.343 vs 1.583）。DF 的 Dynamic Degree 更高（60 vs 50），但那是**视频塌缩病理性抬高了运动指标**——运动大是因为内容崩了，不是动态好。这是「只看单一指标会得出相反结论」的教科书案例。
2. **因果 ODE vs Self Forcing ODE**：chunk-wise 下 VisionReward +90.0%、Dynamic Degree +183.3%（24→68）、指令遵循 +47.4%；frame-wise 差距更夸张——Dynamic Degree **+3100%**（2→64）、VisionReward +218.0%（1.951→6.204）。Self Forcing 的 ODE 初始化在 frame-wise 下几乎全军覆没（Dyn=2、Instruct=−4），与 Prop. 3.3「条件期望=模糊」的预言吻合；chunk-wise 缓一些只是因为 chunk 内还有双向注意力兜底。
3. **因果 CD vs 非对称 CD**：VisionReward 差 **9.781**（1.798 vs −7.983）、指令遵循差 **60**（18 vs −42）——非对称 CD 高度模糊，单射性原则在 CD 框架同样成立。作者坦承当前因果 CD 是直接套 vanilla LCM 的粗糙实例（仍逊于分数蒸馏），改进留给后续（为 [Causal-rCM（17）](../forcing_17/) 留了入口）。

**附录 Table 3（AR 训练策略的近亲）**：PFVG 61/2.857/32、BAgger 53/2.715/28、Resampling Forcing 51/3.336/22，与 TF（50/3.343/32）持平——这些为长视频设计的策略在 5 秒设置里没有额外收益。

**附录 Table 4（初始化梯度实验）**：Self Forcing ODE + DMD 82.00/24/3.330/38 → 多步 AR + DMD 84.02/66/5.863/48 → 因果 ODE + DMD 84.04/68/6.326/56——**架构对齐（换 AR 教师）贡献了绝大部分增益，少步化（ODE 蒸馏）再补最后一段**。

</details>

---

## 7. 与系列其他篇目的关系

| 对照 | 本文 | 对方 | 分歧点 |
|------|------|------|--------|
| [CausVid（06）](../forcing_06/) | 证明其「双向教师 ODE 轨迹初始化 AR 学生」违反帧级单射性 | 开创了非对称蒸馏管线（ODE 初始化 + 非对称 DMD + KV cache） | 本文不换管线，只换 ODE 初始化的**教师**——错配命题（06 篇末埋的疑问）在这里被定理化 |
| [Self Forcing（07）](../forcing_07/) | 继承其 DMD 阶段与自 rollout 协议（原样照抄以保证公平） | 其 ODE 初始化被 Lemma 3.2 + Prop. 3.3 判了死刑 | 「自生成历史对齐」解决的是历史分布，解决不了**架构错配**——两个错配要分开治 |
| [Diffusion Forcing（04）](../forcing_04/) | Prop. 3.4 判定「AR 训练中用噪声前缀当条件」是分布错配 | DF 的原始设计场景（双向续写，干净前缀+噪声拼接）不受此限 | 对 DF 的批评是**用法级**的，不是方法级——附录 C.1 特意划清边界 |
| [Deep Forcing（11）](../forcing_11/) / LongLive / Rolling Forcing | 5 秒注意力内做到最优，长视频外推与之**正交** | 长视频适配（训练型或零训练型） | 论文原文点名可组合：Causal Forcing 的少步 AR 学生 + Deep Forcing 的零训练缓存管理 |
| [Context Forcing（12，上一篇）](../forcing_12/) | 教师与学生的**因果结构**要一致 | 教师与学生的**上下文长度**要一致 | 「师生信息一致性」的两个正交投影：12 对齐条件的内容，13 对齐条件的方向 |
| [Diagonal Distillation（14，下一篇）](../forcing_14/) | 给出单射性判据与因果 ODE 初始化 | 第四部分后续工作在此地基上继续 | 本节的推导（Eq. 4–8）是 14–17 的理论公共底座 |
| [VG 06](../video_generation/VG_06_videoscene/) | 「蒸馏错配」主题的深化 | DMD 家族总览 | 本篇补上 VG 06 留下的「非对称蒸馏为什么难」的机制答案 |

系列视角：第四部分「少步蒸馏与初始化」由此开篇。前三部分反复出现「师生看的世界不一样」的母题（历史的分布、历史的长度），本篇把它推到最底层——**因果结构本身**——并第一次给出可证明的必要条件（帧级单射性）而不只是经验观察。

---

## 8. 局限与个人评价

**边界自认**（论文讨论节）：Causal Forcing 在 5 秒视频、至多 5 秒注意力上训练，直接外推长视频仍有训练-推理差距，需与长视频适配方法组合；因果 CD 是粗糙的 vanilla LCM 实例，还有大量设计空间。

**顺着系列主线的四条评价**：

1. **这是系列里罕见的「先证伪、再施工」的论文**。Lemma 3.2 → Prop. 3.3 的两步（碰撞非零测 → 条件期望不在流形上）把 Self Forcing 的失败从「经验上欠佳」升级为「理论上必然」；而且 Prop. 3.3 的二阶矩证明干净得可以进教材——$\mathbb{E}\|Y\|^2 = \mathbb{E}\|\hat{Y}\|^2 + \mathbb{E}[\mathrm{Var}]$ 一行就说明了「平均掉的方差 = 模糊」。附录 C.3 的隔离实验（换数据不换权重、性能不变）则是把因果解释做实的点睛之笔。
2. **frame-wise 消融是最锋利的一组数字**。Dynamic Degree 2 → 64（+3100%）说明 Self Forcing 的 ODE 初始化在逐帧设置下**彻底失效**，chunk-wise 的 24→68 只是同一病灶的轻症版——chunk 内的双向注意力恰好在做「用上下文消歧」的补丁，掩盖了单射性缺失。这也提示读 chunk-wise 结果时要意识到 chunk 大小是一个隐含的缓解变量。
3. **TF 对 DF 的翻案要按场景读**。Prop. 3.4 只否定「AR 训练 + 噪声前缀条件」这一种用法；DF 在双向续写场景（训练与推理的输入拼接方式一致）依旧正当。系列第 04 篇的核心贡献没有被推翻，被推翻的是把它搬进 AR 蒸馏管线的移植方式。DF 的 Dynamic Degree 反而更高（60 vs 50）但来自塌缩——单一指标结论会整个翻面，这个反例值得在每个消融表旁边挂一辈子。
4. **「教师复现学生的信息结构」在此收官成定理**。12 篇说教师的**上下文长度**必须与学生一致，本篇说教师的**因果方向**必须与学生一致；两者合起来，「师生信息一致性」从系列反复出现的直觉升格为带必要条件的判据：**凡前向 KL / MSE 型蒸馏（ODE、CD），教师的条件信息不得少于学生的生成条件**。14–17 篇的各路初始化方案（Diagonal Distillation、Causal Forcing++、One-Forcing、Causal-rCM）都可以在这个判据下重新排队。

下一篇 [Diagonal Distillation（14）](../forcing_14/)顺着初始化这条线继续往下走。

{{< forcing_nav >}}
