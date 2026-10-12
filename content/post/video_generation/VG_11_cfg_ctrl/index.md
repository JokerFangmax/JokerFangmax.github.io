---
title: "视频生成 11: CFG-Ctrl——无分类器引导的控制论重构"
date: 2026-10-18
math: true
tags:
  - Notes
  - Video Generation
categories:
  - Video Generation
description: "引导三部曲终章（09 对齐 → 10 搜索 → 11 引导）：CFG-Ctrl（arXiv:2603.03281，CVPR 2026 Highlight，Hanyang Wang 与 Yiyang Liu 共同一作，清华）把流匹配采样 ODE 重构为控制仿射系统 dx/dt = v + u，控制律 u = K·Π(e)——K=w、Π=I 代回恰好是 CFG，即 CFG 数学上等价于增益恒定的比例控制（P 控制），guidance scale 就是比例增益；大 scale 崩 = 高增益下的超调与振荡。在此框架下 APG = 投影反馈、CFG-Zero⋆ = 投影到无条件方向、Rectified-CFG++ = MPC、权重调度 = 增益调度，全部 CFG 变体归位为控制律设计。SMC-CFG 引入滑模控制：滑模面 s = ė + λe，切换律 Δe = −k·sign(s)，Lyapunov 分析给出有限时间收敛 ‖s(t)‖=0, t ≤ ‖s(0)‖/(kb_min−δ)，离散实现导出稳定性走廊 δ/w < k < 2‖s‖/(wΔt)——k 太小压不住模型漂移，太大颤振。三个流匹配 T2I 模型（SD3.5 8B / Flux-dev 12B / Qwen-Image 20B）+ Wan2.2 视频：SMC-CFG 几乎全指标最优且 FLOPs 与 CFG 完全相同（3590870.89 vs 3590870.93）；大 scale 下 CFG 快速退化而 SMC-CFG 持续提升。附 toy 相平面实验四组：CFG 在 w > w_c 后进入极限环且半径精确等于饱和反阻尼的平衡解（w=12 时实测 5.345 vs 闭式解 5.35）；k 扫描呈现完整走廊 U 型曲线；λ 不敏感（论文网格搜索三模型全选 λ=6 的机制版）；漂移超过控制权限 w·k 后温和劣化而非爆炸——sign 控制的软失败"
---

# 视频生成 11：CFG-Ctrl——无分类器引导的控制论重构

> 📺 视频生成系列第 11 篇。前篇：[视频生成 10：Video-T1——视频生成的测试时扩展](../VG_10_video_t1/)。
> 论文：*CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance*（arXiv:2603.03281，CVPR 2026 Highlight——项目页 🔥 确认，arXiv 页未标注；Hanyang Wang 与 Yiyang Liu 共同一作，Jiawei Chi、Fangfu Liu、Ran Xue、Yueqi Duan（通讯）；清华大学）。
> 「如何变好」的第三个杠杆：引导。09 把偏好信号写进训练损失，10 把它留到推理时做选择，本篇两个信号都不用——它改的是采样动力学本身：CFG 这个用了五年的「条件减无条件再加权」操作，被证明是一个比例控制器，而它的全部已知修补方案，都是控制律设计。

---

## 1. 这篇论文解决什么问题？

**为什么大 guidance scale 会崩？** 每个用过 Stable Diffusion 的人都知道：CFG scale 调高，语义对齐变好，但过了某个点图像开始过饱和、颜色失真、结构扭曲、细节丢失。工程上的修补从未停过——时变 scale、投影、预测-校正（本篇 Table 1 把它们全部收编）——但修补的依据是各自的直觉，没有一个统一的语言回答「崩的机制是什么」。

CFG-Ctrl 的答案分两层：

1. **诊断**：CFG 的引导速度 $\hat{\mathbf v} = \mathbf v(\varnothing) + w\,(\mathbf v(\mathbf c) - \mathbf v(\varnothing))$ 是条件与无条件预测的**线性外推**。它隐含假设语义误差 $\mathbf e(t)$ 沿直线收敛到原点（$\dot{\mathbf e}$ 与 $\mathbf e$ 共线）。模型容量大、guidance scale 高时这个假设失效——$(\mathbf e, \dot{\mathbf e})$ 相平面上的轨迹开始振荡甚至发散，宏观表现就是过饱和与伪影。**「崩」不是玄学，是控制系统的失稳**。
2. **药方**：既然 CFG 是控制器，就按控制论换一个更聪明的控制器。论文选了滑模控制（Sliding Mode Control, SMC）——一个以「对建模误差和扰动鲁棒」著称的非线性控制范式——配上 Lyapunov 稳定性证明，保证有限时间收敛到流形。

贡献四条：① CFG-Ctrl 统一框架（采样 ODE → 控制仿射系统，现有 CFG 变体全部归位）；② SMC-CFG 算法（滑模面 + 切换控制律，training-free，只改采样循环里几行）；③ Lyapunov 有限时间收敛证明 + 离散实现的「稳定性走廊」；④ 三个流匹配 T2I 模型 + 一个 T2V 模型的系统实验。

---

## 2. 所需背景最小集

1. **CFG（VG 01 已推导）**：训练时同一样本以概率丢弃条件，推理时 $\hat{\mathbf v} = \mathbf v(\varnothing) + w\,\Delta\mathbf v$，$\Delta\mathbf v = \mathbf v(\mathbf c) - \mathbf v(\varnothing)$。$w=1$ 退化为条件生成，$w$ 越大语义越强、伪影越多。本篇只需要这一个公式。（教学版完整推导——贝叶斯分解 → Gaussian 路径 → 训练算法——见 [MITS.6184 第 5 篇](../../MITS6184/MITS6184_5/)；该篇 §16 把「大 scale 副作用」停在 heuristic，本篇就是那个问题的答案。）
2. **流匹配（flow matching）**：直接回归确定性速度场，采样是解 ODE $d\mathbf x_t/dt = \mathbf v_\theta(\mathbf x_t, t)$。选它的理由（论文原话）：直接估计确定性速度场、训练更稳、采样更快——且 SD3.5 / Flux / Qwen-Image 三个当红 T2I 模型都是流匹配底座，改它们的采样循环有最大的现实意义。
3. **控制论最小集**（本篇特有前置，一张表够用）：

| 控制概念 | 一句话 | 在 CFG 里的对应 |
|---|---|---|
| 被控对象 | 你想操纵的动力学系统 | 采样 ODE $d\mathbf x_t/dt = \mathbf v_\theta + \mathbf u$ |
| 状态反馈 | 控制输入由当前状态（误差）决定 | $\mathbf u$ 是 $\mathbf e(t)$ 的函数 |
| P 控制（比例控制） | 控制量 ∝ 误差，增益恒定 | **vanilla CFG** |
| 增益调度 | 增益随时间/状态变化 | CFG 权重调度（时变 $w(t)$） |
| 投影反馈 | 只在误差的某些方向上施加控制 | APG、CFG-Zero⋆ |
| MPC（模型预测控制） | 用系统模型预测未来、滚动优化 | Rectified-CFG++ |
| 滑模控制 | 切换律把状态强行压到一个流形上 | SMC-CFG |
| Lyapunov 函数 | 系统的「能量」，能量单调下降 ⇒ 收敛 | $V = \frac12\|\mathbf s\|^2$ |

---

## 3. 方法总览

**一句话主线**：输入一个不重训的预训练流匹配模型和一句提示词，先把它采样时「条件速度减无条件速度再放大」的操作重写成一套控制系统语言，发现 CFG 就是一个力度恒定的比例控制器、大 scale 崩是这个控制器失稳；再换上一个按误差状态切换修正方向的滑模控制器（SMC-CFG），只改采样循环里几行，输出大引导尺度下依然干净的图像与视频。

| 步骤 | 做什么（白话） | 为什么需要这一步 |
|---|---|---|
| ① 换语言：把采样写成受控系统 | 把采样过程改写成「模型自身速度 + 一个引导控制项」，控制项 = 引导强度 × 修正方向，作用在语义误差（条件与无条件速度之差）上（§4.1） | 代回去会发现 vanilla CFG 就是「强度恒定、方向不变」的比例控制，guidance scale 就是控制力度——不换语言，「为什么崩、往哪修」就没有讨论框架，已有修补也只能各自凭直觉 |
| ② 诊断：找出大 scale 崩的机制 | 指出 CFG 隐含假设语义误差沿直线收敛到零；模型容量大、引导尺度高时假设失效，误差在相平面上打转甚至发散（§4.2） | 把「崩」从玄学变成控制系统的失稳，顺带让 APG、CFG-Zero⋆ 等已有修补全部归位为控制律设计空间里的点——新控制器该往哪个空间找，一目了然 |
| ③ 换控制器：SMC-CFG | 每步先量出语义误差及其变化趋势，两者合成一个「滑模面」信号；再按这个信号的符号以固定力度反向修正误差，把误差强行压回理想收敛轨道，最后照常合成速度（§4.3） | 线性控制（比例、投影、调度）对高度非线性的对象力不从心；滑模的切换式修正对模型误差与漂移天生鲁棒，且能在有限时间（而非慢慢趋近）把误差压上轨道 |
| ④ 划安全边界：稳定性走廊 | 用 Lyapunov「能量单调降」证明误差必在有限时间内被压到目标轨道；再针对离散实现导出修正力度 $k$ 的可行区间——太小压不住模型漂移，太大会来回过冲（颤振）（§4.3） | 定理给定性保证（安全区间存在），走廊给调参指南——$k$ 是唯一必须认真调的超参，走廊告诉你往哪个范围调 |

```
预训练流匹配模型 (不重训, 只改采样循环)
        │
        ▼
┌─ CFG-Ctrl 重构 ─────────────────────────────────────┐
│  采样 ODE:  dx/dt = v_θ(x,t) + u_t     (控制仿射)   │
│  控制律:    u_t = K_t · Π_t ( e(t) )                │
│             K_t = 引导调度 (强度)  Π_t = 方向算子     │
│  e(t) = v_θ(x,t,c) − v_θ(x,t,∅)   (语义误差)        │
│                                                     │
│  K_t = w, Π_t = I  ──代入──►  恰好是 vanilla CFG     │
│  ⇒ CFG ≡ 增益恒定的 P 控制, w = 比例增益             │
└─────────────────────────────────────────────────────┘
        │  Table 1: 现有变体全部归位
        │  (APG=投影反馈 / CFG-Zero⋆=投影 / Rect-CFG++=MPC / 调度=增益调度)
        ▼
┌─ SMC-CFG (Alg.1) ───────────────────────────────────┐
│  每步:  e(t) ← v(c) − v(∅)                          │
│         s_t ← (e(t) − e(t+1)) + λ·e(t+1)   (滑模面)  │
│         Δe ← −k·sign(s_t)                   (切换律) │
│         v̂_t ← v(∅) + w·(e(t) + Δe)                  │
│  Lyapunov: V̇ ≤ −(k·b_min − δ)‖s‖  ⇒ 有限时间收敛    │
│  走廊: δ/w < k < 2‖s_t‖/(wΔt)   (离散实现约束)      │
└─────────────────────────────────────────────────────┘
```

training-free、模型无关：只动每步的速度合成公式，FLOPs 与标准 CFG 几乎逐位相同（Table 8）。

---

## 4. 核心机制完整推导

### 4.1 重构：采样 ODE 是控制仿射系统（Eq.8–13）

把采样过程写成连续时间受控动力学：状态 $\mathbf x_t \in \mathbb R^d$（latent），演化满足

$$
\frac{d\mathbf x_t}{dt} = \mathbf v_\theta(\mathbf x_t, t) + \mathbf G(\mathbf x_t, t)\,\mathbf u_t, \tag{8}
$$

$\mathbf u_t \in \mathbb R^m$ 是引导控制输入，$\mathbf G$ 是输入映射矩阵。引导直接作用在 latent 坐标上、无跨空间变换，故取全驱动 $\mathbf G = \mathbf I_d$（$m = d$）：

$$
\frac{d\mathbf x_t}{dt} = \mathbf v_\theta(\mathbf x_t, t) + \mathbf u_t. \tag{9}
$$

定义**语义误差**（Eq.6）：

$$
\mathbf e(t) = \mathbf v_\theta(\mathbf x_t, t, \mathbf c) - \mathbf v_\theta(\mathbf x_t, t, \varnothing),
$$

控制律取一般状态反馈形式（Eq.10）：

$$
\mathbf u_t = K_t\,\Pi_t(\mathbf e(t)), \tag{10}
$$

$K_t$ 是**引导调度**（guidance schedule，控制强度），$\Pi_t$ 是**方向算子**（direction operator，整形修正方向——恒等、归一化或投影）。

**关键一步（Eq.11–13）**：取 $K_t = w$（常数标量）、$\Pi_t = \mathbf I$，代回：

$$
\frac{d\mathbf x_t}{dt} = \mathbf v_\theta(\mathbf x_t, t, \varnothing) + w\,\big(\mathbf v_\theta(\mathbf x_t, t, \mathbf c) - \mathbf v_\theta(\mathbf x_t, t, \varnothing)\big),
$$

**恰好是 vanilla CFG**。结论：CFG 数学上等价于作用在语义误差上的**比例状态反馈控制器（P 控制）**，guidance scale $w$ 就是比例增益。这一步没有任何近似——是同一公式的两种记法，但两种记法的「直觉接口」完全不同：外推视角看不见的东西（增益太高会失稳、需要阻尼、需要流形），控制视角一目了然。

### 4.2 动机：线性收敛假设在哪里失效（§3.2）

理想情况下语义误差应指数收敛（Eq.18 的雏形）：

$$
\dot{\mathbf e}(t) = -\lambda_0\,\mathbf e(t), \qquad \lambda_0 = -\frac{\dot{\mathbf e}(T)\cdot\mathbf e(T)}{\|\mathbf e(T)\|^2},
$$

即相平面 $(\mathbf e, \dot{\mathbf e})$ 上轨迹沿过原点的直线收敛。论文指出：**这个共线假设在实践中很少成立**，尤其当模型容量增大、guidance scale 提高时——系统高度非线性，标准 CFG 表现出振荡或发散行为（Fig.1 左），宏观上就是颜色失真、细节丢失、纹理紊乱。P 控制的经典病灶：增益恒定、不含阻尼项、对非线性对象高增益必然超调。修补它的全部已有方案，在 $(K_t, \Pi_t)$ 分解下各就各位（Table 1）：

| 方法 | $K_t$ | $\Pi_t$ | 控制论解读 |
|---|---|---|---|
| CFG | $w$ | $\mathbf I$ | 比例控制 |
| 权重调度 | $w(t)$ | $\mathbf I$ | 时变增益调度 |
| APG | $w[\mathbf I\ \eta\mathbf I]$，$\eta\le 1$ | $[\mathbf I - P_tP_t]$，$P_t = \frac{\mathbf v(\mathbf c)\mathbf v(\mathbf c)^\top}{\|\mathbf v(\mathbf c)\|^2}$ | 投影反馈：把 $\mathbf e$ 分解为平行/垂直于 $\mathbf v(\mathbf c)$ 的分量，压低平行分量（超调主源） |
| CFG-Zero⋆ | $[w\mathbf I\ \ \frac{s_t}{1-s_t}\mathbf I]$，$s_t = \frac{\mathbf v(\mathbf c)^\top\mathbf v(\varnothing)}{\|\mathbf v(\varnothing)\|^2}$ | 投影到 $\mathbf v(\varnothing)$ 的正交补 | 投影反馈（流匹配专用） |
| Rectified-CFG++ | $[\mathbf I\ \ \alpha(t)\mathbf I]$，$\alpha(t) = \lambda_{\max}(1-t)^\gamma$ | $\mathbf I$，误差取 $[\Delta\mathbf v(t),\ \Delta\mathbf v(t-\tfrac{\Delta t}{2})]$ | 模型预测控制：用上一步误差预判偏差、提前修正 |
| **SMC-CFG** | $w$ | $\mathbf I$，修正项 $-\,k\cdot\mathrm{sign}(\mathbf s_t)$ | **滑模控制** |

这张表是本篇最有复用价值的东西：它把散落的 trick 组织成一个设计空间——**强度轴（$K_t$）与方向轴（$\Pi_t$）**，每个新方法都是这个空间里的一个点。

### 4.3 SMC-CFG：滑模面、切换律与有限时间收敛（Eq.18–28）

线性控制（P、增益调度、投影都是线性的）对高度非线性的对象力不从心，论文引入滑模控制。

**第一步：定义流形（Eq.18–19）**。目标误差动力学仍是理想指数衰减 $\dot{\mathbf e} = -\lambda_0\mathbf e$，但不再假装它自动成立，而是定义滑模面：

$$
\mathbf s(t) = \dot{\mathbf e}(t) + \lambda\,\mathbf e(t), \tag{19}
$$

$\lambda$ 是流形形状参数。$\mathbf s = \mathbf 0$ 的流形隐含编码了目标动力学：**一旦状态贴在流形上（$\dot{\mathbf e} = -\lambda\mathbf e$），误差自动按速率 $\lambda$ 指数收敛**。控制问题从「让 $\mathbf e$ 收敛」改写为「把 $\mathbf s$ 压到零」——后者可以对任意非线性对象做到。

**第二步：Lyapunov 候选（Eq.20）**：

$$
V(\mathbf s) = \tfrac12\|\mathbf s\|^2, \qquad \dot V = \mathbf s^\top\dot{\mathbf s} < 0.
$$

**第三步：误差动力学（Eq.21–23）**。对 $\mathbf e$ 求链式导数（Eq.21 含 $\partial\mathbf v_\theta/\partial\mathbf x$ 的雅可比差与 $\partial/\partial t$ 项），引入修正项 $\Delta\mathbf e$，完整控制为 $\mathbf u = w(\mathbf e + \Delta\mathbf e)$——**修正在误差动力学里乘着增益 $w$ 进入**（这正是 toy 实验里控制仿射入口的依据）。代入受控动力学（Eq.8）后：

$$
\dot{\mathbf e}(t) = \mathbf\Phi(t, \mathbf x) + w\Big(\frac{\partial\mathbf v_\theta(\mathbf c)}{\partial\mathbf x} - \frac{\partial\mathbf v_\theta(\varnothing)}{\partial\mathbf x}\Big)\mathbf G\,\Delta\mathbf e(t), \tag{22}
$$

$\mathbf\Phi$ 吸收所有与 $\Delta\mathbf e$ 无关的项。再求导代回滑模面定义：

$$
\dot{\mathbf s}(t) = \mathbf\Phi_s(t, \mathbf x) + \mathbf\Gamma_s(t)\,\Delta\mathbf e(t), \tag{23}
$$

两条标准假设（滑模文献惯例）：有效控制增益矩阵的最小奇异值有下界 $\sigma_{\min}(\mathbf\Gamma_s)\ge b_{\min} > 0$；内禀漂移有界 $\|\mathbf\Phi_s\|\le\delta$。

**第四步：切换律（Eq.25–26）**。取经典切换控制：

$$
\Delta\mathbf e(t) = -k\cdot\mathrm{sign}(\mathbf s(t)), \tag{25}
$$

代入 Lyapunov 导数（$\sigma_{\min}(\mathbf\Gamma_s\mathbf K)\ge k\,b_{\min}$）：

$$
\dot V \le \|\mathbf s\|\,\delta - k\,b_{\min}\|\mathbf s\| = -(k\,b_{\min} - \delta)\,\|\mathbf s\|. \tag{26}
$$

**第五步：有限时间收敛（Eq.27–28）**。只要 $k\,b_{\min} > \delta$，令 $\eta = k\,b_{\min} - \delta > 0$，有 $\frac{d}{dt}\|\mathbf s\|\le -\eta$，积分：

$$
\|\mathbf s(t)\| = 0 \quad \text{for some } t \le \frac{\|\mathbf s(0)\|}{\eta}. \tag{28}
$$

**有限时间**（不是渐近）到达流形——这是切换律 $\mathrm{sign}$ 的功劳：修正量大小与距离无关（恒为 $k$），远处近处同样硬压，代价是贴近流形后来回穿越（颤振，见下）。

**定理 1（附录 6.3，鲁棒版本）**：把 $\mathbf\Gamma_s$ 分解为名义各向同性增益加偏差 $\mathbf\Gamma_s = w\mathbf I + \Delta\mathbf\Gamma$，假设 $\|\Delta\mathbf\Gamma\|_2\le\rho$ 且 $w > \rho D$（$D$ 为 $\mathbf s$ 的维度），则

$$
k > \frac{\delta}{w - \rho D} + \epsilon \quad (\epsilon > 0) \Longrightarrow \dot V \le -\epsilon(w-\rho D)\|\mathbf s\|_2,
$$

同样的有限时间收敛结论。读法：**切换增益必须盖过「漂移 $\delta$ + 维度放大的雅可比失配 $\rho D$」的最坏组合**——这是 sign 控制对建模误差鲁棒性的来源，也是它需要手调 $k$ 的原因（$\delta$、$\rho$ 未知）。

**稳定性走廊（Eq.45，离散实现）**。上述是连续时间设计；实际采样步长 $\Delta t$ 有限，高增益切换会颤振。由离散演化 $\|\mathbf s_{t+1}\| \approx \big|\|\mathbf s_t\| - \Delta t\,(w_{\mathrm{eff}}k - \delta)\big|$ 导出启发式**走廊**：

$$
\underbrace{\frac{\delta_{\mathrm{est}}}{w}}_{\text{收敛：压过漂移}} < k < \underbrace{\frac{2\|\mathbf s_t\|_2}{w\,\Delta t}}_{\text{稳定：防数值颤振}}. \tag{45}
$$

$k$ 太小压不住模型漂移（欠修正），太大被逆步长卡住（过修正→颤振）。「适中的固定 $k$」就是走廊内部——Table 3 的消融精确复现这条 U 型曲线（§5 toy 实验 2 给它的机制版）。

**Algorithm 1（离散实现，完整步骤）**：初始 $\mathbf x_T\sim\mathcal N(0,\mathbf I)$，对 $t = T$ 到 $1$：① 算 $\mathbf v_t(\mathbf c), \mathbf v_t(\varnothing)$，$\mathbf e(t) \leftarrow$ 两者之差；② 首步 $\mathbf e(t+1)$ 缺省取 $\mathbf e(t)$；③ $\mathbf s_t \leftarrow (\mathbf e(t) - \mathbf e(t+1)) + \lambda\,\mathbf e(t+1)$（差分近似 $\dot{\mathbf e}$）；④ $\Delta\mathbf e \leftarrow -k\cdot\mathrm{sign}(\mathbf s_t)$；⑤ $\mathbf e(t) \leftarrow \mathbf e(t) + \Delta\mathbf e$；⑥ $\hat{\mathbf v}_t \leftarrow \mathbf v_t(\varnothing) + w\,\mathbf e(t)$；⑦ ODE 步进。注意 $\mathbf e(t)$ 每步从网络**重新测量**——$\Delta\mathbf e$ 只通过速度合成影响这一步的动力学，不持久改变状态（控制仿射语义）。

---

## 5. toy 实验：相平面上的 P 控制失稳与滑模收敛

把语义误差的相平面 $(\mathbf e, \dot{\mathbf e})$ 直接做成 toy（论文 Fig.1 就是这个平面的示意）：$\mathbf e\in\mathbb R^2$，被控对象（CFG 闭环、控制器未知部分）取

$$
\dot{\mathbf e} = m(w,\|\mathbf e\|)\,\mathbf e + (w-1)\kappa\,J\mathbf e + \mathbf d(t), \qquad m = -\lambda_0 + \alpha(w-1)(w - w_c)\frac{\rho^2}{\rho^2 + \|\mathbf e\|^2},
$$

$J$ 为 90° 旋转矩阵。三个部件各有含义：$-\lambda_0$ 是去噪推进带来的误差自然衰减（无引导也收敛）；$(w-1)\kappa J$ 是高增益引入的旋转耦合（$\mathbf e$ 与 $\dot{\mathbf e}$ 非共线的来源，对应论文「大 scale 下线性假设失效」）；饱和反阻尼项在 $w < w_c$ 时是额外阻尼（适度引导有益）、$w > w_c$ 时变反阻尼且随 $\|\mathbf e\|$ 饱和——**有界反阻尼必然造出极限环**（过引导 = 持续伪影的动力学形态）。$\mathbf d(t)$ 是有界模型漂移（Assumption 1 的 $\delta$）。SMC 完全按 Alg.1：差分估 $\dot{\mathbf e}$、$\mathbf s = \dot{\mathbf e} + \lambda\mathbf e$、$\Delta\mathbf e = -k\,\mathrm{sign}(\mathbf s)$、经增益 $w$ 进入 $\dot{\mathbf e}$（Eq.22 的控制仿射入口）。参数 $\Delta t = 0.05$、$T = 20$、$\lambda_0 = 1$、$\kappa = 2.2$、$\alpha = 0.04$、$w_c = 2.5$、$\rho = 3$；SMC 默认 $\lambda = 2$、$k = 0.3$；$\mathbf e_0 = (2, 0.5)$。**积分用 2×2 线性部分的精确矩阵指数**（$e^{m\Delta t}\cdot R(\theta)$，$\theta = (w-1)\kappa\Delta t$）——显式 Euler 对纯旋转项恒数值不稳定（放大系数 $\sqrt{1+\omega^2\Delta t^2} > 1$），首版 toy 在 $w\ge 6$ 全线爆炸就是踩了这个坑，与动力学无关。

**实验 1：引导尺度 $w$ 扫描（论文 Fig.4/6 的机制版）**：

| $w$ | CFG 终 $\|\mathbf e\|$ | CFG mean | CFG TV | SMC 终 $\|\mathbf e\|$ | SMC mean | SMC TV |
|---|---|---|---|---|---|---|
| 1 | 0.006 | 0.054 | 3.4 | 0.005 | 0.055 | 10.0 |
| 2 | 0.059 | 0.056 | 6.1 | 0.069 | 0.060 | 19.1 |
| 4 | 0.018 | 0.017 | 16.4 | 0.027 | 0.045 | 39.6 |
| 6 | 0.007 | 0.074 | 59.4 | 0.067 | 0.067 | 60.3 |
| 8 | **2.197** | 2.204 | **659** | 0.014 | 0.099 | 84.0 |
| 10 | **3.906** | 3.911 | **1456** | 0.192 | 0.144 | 118 |
| 12 | **5.345** | 5.350 | **2384** | 0.291 | 0.209 | 224 |

（TV = $\sum\|\mathbf e_{t+1} - \mathbf e_t\|$，误差振荡能量代理——过饱和/伪影的代理量。）

**实验 2：切换增益 $k$ 扫描（$w = 8$，$\lambda = 2$，走廊 Eq.45）**：

| $k$ | 0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 1.0 | 2.0 | 5.0 |
|---|---|---|---|---|---|---|---|---|---|
| 终 $\|\mathbf e\|$ | 2.197 | 0.021 | 0.041 | 0.039 | 0.014 | 0.275 | 0.434 | 0.894 | 1.953 |
| mean | 2.204 | 0.396 | 0.037 | 0.074 | 0.099 | 0.194 | 0.348 | 0.795 | 1.618 |
| TV | 659 | 206 | 69 | 67 | 84 | 129 | 248 | 501 | 1232 |

**实验 3：流形形状参数 $\lambda$ 扫描（$w = 8$，$k = 0.3$）**：

| $\lambda$ | 0.3 | 0.5 | 1.0 | 2.0 | 4.0 | 8.0 |
|---|---|---|---|---|---|---|
| 终 $\|\mathbf e\|$ | 0.308 | 0.308 | 0.124 | 0.014 | 0.069 | 0.051 |
| TV | 85.2 | 85.2 | 85.3 | 84.0 | 84.1 | 84.2 |

**实验 4：模型漂移 $\delta$ 扫描（$w = 8$，控制权限 $w\cdot k = 2.4$）**：

| $\delta$ | 0 | 0.6 | 1.0 | 2.0 | 4.0 | 8.0 | 12.0 |
|---|---|---|---|---|---|---|---|
| CFG mean | 2.204 | 2.204 | 2.203 | 2.200 | 2.190 | 2.146 | 2.060 |
| SMC 终 $\|\mathbf e\|$ | 0.132 | 0.193 | 0.163 | 0.129 | 0.228 | 0.420 | 0.872 |
| SMC mean | 0.109 | 0.118 | 0.115 | 0.166 | 0.273 | 0.521 | 0.753 |

相平面轨迹（$w = 8$，前 200 步采样）：CFG 从 $(2.0, 0.5)$ 出发旋出并锁定在半径约 2.2 的极限环上往复；SMC **一步**跳到 $(0.15, -0.05)$，之后在原点附近小幅颤振。

六条读法：

- **CFG 的失稳是极限环，且半径有闭式解（实验 1）**：反阻尼随 $\|\mathbf e\|$ 饱和，平衡在 $m = 0$ 处——$r^2 = \rho^2(\alpha(w-1)(w - w_c) - \lambda_0)/\lambda_0$。代入 $w = 8/10/12$ 得 $r = 2.20/3.91/5.35$，实测 $2.197/3.906/5.345$，三位有效数字吻合。**过引导的伪影不是「噪声变大」，是误差被锁进一个由饱和非线性决定的稳定周期轨道**——它有确定的幅度、会自我维持、加大 $w$ 只会按闭式规律推大半径。对应论文 Fig.4/6：大 scale 下 CFG 各项指标快速退化、且退化是平滑可预期的。
- **SMC 的优势恰好出现在失稳之后（实验 1）**：$w\le 6$ 时两者都收敛（对象本身稳定），SMC 反而多付 2–3 倍 TV（颤振代价）；$w = 8$ 起 CFG 进极限环（mean 2.2），SMC 稳在 0.1–0.3、TV 低一个量级。**滑模不是无条件更好，它买到的是「失稳区的稳定性」**——这与论文 §9.1（Flux 在默认 $w=2$ 处各方法最优、更大 scale 基线快速退化而 SMC-CFG 持续提升）逐点对应。
- **$k$ 的走廊 U 型完整可见（实验 2）**：$k = 0$ 即 CFG（2.197）；$k = 0.05$ 时控制权限 $w\,k = 0.4$ 压不住反阻尼增长率 0.54，收敛慢（mean 0.396）；$k\in[0.1, 0.3]$ 走廊内最优（TV 67–84）；$k\ge 0.5$ 颤振主导——每步修正量 $\Delta t\,w\,k$ 超过 $2\|\mathbf s\|$，误差被 sign 切换来回抽打，$k = 5$ 时 TV 反超 CFG、终值 1.953 接近不控。**下界对应 Eq.45 的 $\delta/w$（权限压漂移），上界对应 $2\|\mathbf s\|/(w\Delta t)$（步长内不穿越）**——论文 Table 3「适度 $k$ 最优、过大 $k$ 引起 abrupt pulls / erratic sampling / vibrations」的精确机制版。
- **$\lambda$ 不敏感是结构性的（实验 3）**：$\lambda\ge 1$ 后全平（终值 0.01–0.12）——一旦贴上流形，$\lambda$ 只决定**流形上**的指数衰减速率，而 $T = 20$ 个时间单位内速率 1–8 都收敛干净；$\lambda = 0.3$ 才因流形上收敛过慢 + 颤振地板相对抬高而劣化到 0.308。这解释了论文网格搜索的一个表面反常：**三个模型全部选中 $\lambda = 6$**（对 $\lambda$ 不敏感、选个中偏大值），真正按模型分化的是 $k$（SD3.5/Qwen 0.1，Flux 0.7）——与消融表 $\lambda$ 行平缓、$k$ 行陡峭完全一致。$k$ 是唯一必须认真调的参数。
- **对匹配漂移不变，越界后软失败（实验 4）**：$\delta$ 从 0 加到 2.0（≈权限 $w\,k = 2.4$）SMC 几乎不动（mean 0.109→0.166）——sign 项以恒定力度对抗有界扰动，这是滑模「对匹配扰动不变性」的招牌性质；超过权限后（$\delta = 8/12$，越界 3–5 倍）**温和劣化**（0.52/0.75）而非爆炸。对照组 CFG 全程趴在极限环上（mean ≈ 2.2）——它连「稳态」都没有，漂移无从谈起。
- **数值方法的教训（附录 A 详述）**：首版 toy 用显式 Euler，$w\ge 6$ 时旋转耦合频率 $\omega = (w-1)\kappa$ 使 $\omega\Delta t\approx 0.77$，Euler 对纯旋转的每步放大 $\sqrt{1 + \omega^2\Delta t^2} > 1$ 恒成立——全线爆到 $10^{20}$。这不是动力学发散，是积分器发散。改成线性部分精确指数积分后同样的动力学在全部 $w$ 下行为良好。**做动力系统 toy，先问积分器配不配得上频率**——否则会把数值伪影当成机制发现。

---

## 6. 训练与推理细节 + 论文实验

### 6.1 实验设置速查

| 项 | 值 |
|---|---|
| 数据 | MS-COCO 子集 5,000 图文对（T2I-CompBench 6,000 组合式 prompt 用于 §8.1） |
| 模型（全部不重训） | SD3.5-large (8B)、Flux-dev (12B)、Qwen-Image (20B)，1024×1024 |
| Flux 特殊处理 | guidance-distilled 模型，无 CFG 基线的嵌入引导设 1（保证公平） |
| 基线 | CFG、CFG-Zero⋆、Rectified-CFG++（后两者按官方代码在全部骨干上复现） |
| 超参确定 | 网格搜索：$\lambda\in\{2..8\}$、$k\in\{0.01..0.8\}$，200 例辅助集（与评测集不相交，防泄漏）→ SD3.5 $(6, 0.1)$、Flux $(6, 0.7)$、Qwen $(6, 0.1)$ |
| 硬件 | 单卡 A100 40GB |

### 6.2 主结果（Table 2，MS-COCO，八指标）

| 模型 / 方法 | FID ↓ | CLIP ↑ | Aesthetic ↑ | ImageReward ↑ | HPSv2.1 ↑ | MPS ↑ |
|---|---|---|---|---|---|---|
| SD3.5 无引导 | 41.725 | 0.3399 | 5.4256 | 0.3591 | 0.2372 | 6.5554 |
| SD3.5 + CFG | 21.421 | 0.3681 | 5.5875 | 0.8889 | 0.2842 | 7.2476 |
| SD3.5 + CFG-Zero⋆ | 20.317 | 0.3691 | **5.6124** | 0.9312 | 0.2862 | 7.0430 |
| SD3.5 + Rect-CFG++ | 20.550 | 0.3655 | 5.5663 | 0.7097 | 0.2748 | 6.7854 |
| SD3.5 + SMC-CFG | **20.044** | **0.3694** | 5.5790 | **0.9486** | **0.2875** | **7.5719** |
| Flux + CFG | 27.323 | 0.3692 | 5.5397 | 0.8749 | 0.2828 | 7.8387 |
| Flux + SMC-CFG | **26.398** | **0.3743** | **5.7342** | **1.0558** | **0.3021** | **8.2307** |
| Qwen + CFG | 35.431 | 0.3815 | 5.5995 | 1.1063 | 0.3038 | 8.1852 |
| Qwen + SMC-CFG | **33.371** | **0.3856** | **5.6289** | **1.2035** | **0.3105** | **8.4320** |

（完整表还有 PickScore、HPSv2 两列，格局相同；CFG-Zero⋆ 与 Rect-CFG++ 在 Flux/Qwen 上也全面优于 CFG——流匹配专用变体确实有效，SMC-CFG 在绝大多数格子上再压一头。）

三点诚实读表：① **SMC-CFG 不是每格都赢**——SD3.5 的 Aesthetic 输给 CFG-Zero⋆（5.5790 vs 5.6124），Flux 的 HPSv2 与 CFG-Zero⋆ 基本打平，「全指标最优」要打折成「绝大多数指标最优」；② **Qwen 无引导的 FID 反而全场最低（24.894）**，论文自己加注「Qwen-Image 保持自然图像统计」——引导本质上是**拿自然图像统计换语义对齐**的交易，所有引导方法都站在这条交换曲线上，SMC-CFG 只是让曲线的斜率更有利；③ ImageReward 提升幅度（SD3.5 +0.06、Flux +0.18）与 VG 09 的对齐增益同一量级，但零训练成本。

### 6.3 消融（Table 3）与网格搜索的错位

$\lambda$ 行（$k = 0.1$）：$\lambda = 3/4/5/6 \to$ FID 26.193 / 26.006 / **25.951** / 26.143——平缓，极值两侧均劣化（流形形状被扭曲）。$k$ 行（$\lambda = 5$）：$k = 0.1/0.4/0.7/1.0 \to$ FID 25.951 / 26.143 / 26.416 / 26.281，CLIP 0.3709 / 0.3719 / 0.3739 / **0.3741**，Aesthetic 5.7128 / 5.7218 / 5.7175 / 5.7054——**小 $k$：CLIP 低但 FID 好（欠修正保真实感）；大 $k$：CLIP 拉满但美感与人类偏好受损（过修正引颤振）**。论文原话：过大的 $k$ 引起 abrupt pulls、erratic sampling、vibrations。

一个值得点破的细节：消融表中 FID 最优在 $(\lambda{=}5, k{=}0.1)$，但 Flux 的部署配置是 $(\lambda{=}6, k{=}0.7)$——网格搜索是多指标综合后的选择（$k = 0.7$ 换来 CLIP +0.003 与 ImageReward +0.016，代价 FID +0.47）。「走廊内的最优位置」取决于你在交换曲线上想要哪个点，定理给不了这个答案。

### 6.4 组合性与视频泛化（Table 5/6/7）与效率（Table 8）

**T2I-CompBench（Table 5，Color / Shape / Texture / Spatial 四维全部上涨，Spatial 涨幅最大）**：SD3.5 Spatial 0.1625→0.2563、Qwen Spatial 0.2968→**0.4085**（相对 +38%）、Flux Shape 0.5751→0.6199。**VQAScore GenAI-Bench（Table 6，SD3.5）**：Base/Advance/Overall 0.83/0.64/0.72（CFG）→ **0.89/0.68/0.77**（SMC-CFG），超过 CFG-Zero⋆（0.88/0.66/0.75）与 Rect-CFG++（0.87/0.64/0.73）。**T2V（Table 7，Wan2.2-TI2V-5B，VBench）**：Total 0.5594→0.5839，Quality 0.6581→0.6747，Semantic 0.4607→0.4931，Color 0.9087→**0.9818**，Human Action 0.5313→0.6000——**引导层面的改动能跨模态迁移到视频**，这是「SMC 修的是采样动力学而非图像特有统计」的最强证据。

**效率（Table 8）**：1024×1024 下 Memory 33.59 = 33.59 GB、FLOPs 3590870.89 vs 3590870.93（**逐位几乎相同**）、Runtime 44.78 vs 45.09 s；512×512 同样格局。每步只多一次 sign 与逐元素加法——工程上真正的零成本即插即用。

---

## 7. 与其他论文的联系

**系列内**：

- **VG 01（CFG 的身世补完）**：01 推导了 CFG「为什么能引导」（条件分数与无条件分数的差指向条件流形），本篇回答了补完的问题——**它是一个 P 控制器**。这个重构不是同义反复：外推视角问「外推多少」（$w$ 调大调小），控制视角问「这个对象的闭环稳定域在哪、需要什么阻尼结构」——同一公式，后者的设计空间大得多（Table 1 整张表都是它的产出）。
- **VG 09 / VG 10（引导三部合流：同一设计空间的三个轴）**：偏好/语义信号现在有了三种花法——**进损失**（09 DreamFL，梯度被 hack 出分布、无界风险）、**进选择**（10 best-of-N，误选有界、对数回报）、**进动力学**（本篇，修正有界、颤振有界）。三篇合起来的教训对称而漂亮：梯度的收益无界风险也无界，选择与切换的收益有界风险也有界——**干预的「杠杆率」与「失控半径」同源**。VG 12 会把这条轴正式画进版图。
- **VG 06（VideoScene，蒸馏 vs 引导：效率的两条正交轴）**：06 压平步数（50 步→1 步）但每步质量不变；本篇步数不变但每步走得更稳。尖锐的交叉问题：**一步模型还有「动力学」可修吗？** 走廊上界 $2\|\mathbf s\|/(w\Delta t)$ 随 $\Delta t$ 增大而收窄——步数越少、单步越大，切换控制的可用增益越小。蒸馏与滑模在极限处冲突，与 VG 10 提出的「蒸馏是否压平搜索空间」是同构的问题。
- **系列外**：

  - **FA 系列（Banach 不动点 / 压缩映射）**：Lyapunov 稳定性与压缩映射是收敛性证明的两个家族——前者配能量函数（$V$ 单调降），后者配度量（迭代收缩）。本篇的 $\dot V \le -\eta\|\mathbf s\|$ 给的是**有限时间**收敛，比压缩映射的渐近收敛更强，代价是需要切换律这种非光滑结构。
  - **控制论 × 深度学习的迁移范式**：强化学习里的控制论根源（PID 调参、MPC 与 model-based RL）早已是常识，本篇把同一条路走在了生成模型采样上。值得盯的方向：论文 §9.2 自己指出的自适应引导控制（用误差微分反馈动态调增益）就是控制论里自适应控制的直接移植。
  - **APG / CFG-Zero⋆ / Rect-CFG++ 的原作者视角**：这些方法的设计直觉各不相同（分解平行分量 / 修速度估计 / 预测-校正），Table 1 证明它们其实是控制律空间里的近邻——**好的统一框架会让下一个「trick」变成「查表」**。

---

## 8. 局限与思考

**论文自认的局限（§9.2）**：引入 $\lambda$、$k$ 两个超参增加部署复杂度、需按模型手调；未来方向是自适应引导控制（误差微分反馈动态调整有效引导强度）。

**我的补充思考**：

1. **定理的假设在边界处最不成立**。Assumption 1 自己承认「扩散模型的梯度在时间边界（$t\to 0$ 或 $t\to T$）可能发散」——而采样恰恰要穿越这些边界。定理保证的是有效采样区间内的鲁棒性，边界附近的行为靠的是实践宽裕度（论文 Remark：上界由 $1/\Delta t$ 主导、可行 $k$ 的范围很宽）而非证明。读这类控制论证明要分清：**它锚定的是定性结构（存在走廊），不是定量保证（走廊精确位置）**。
2. **维度放大项 $\rho D$ 是个深坑**。Theorem 1 的条件 $w > \rho D$ 里 $D$ 是 $\mathbf s$ 的维度——图像 latent 是 $16\times128\times128\approx 2.6\times 10^5$ 量级。若各向异性偏差 $\rho$ 真随维度积累，名义增益 $w$（个位数）根本压不住。实践有效说明实际的 $\Delta\mathbf\Gamma$ 高度结构化（能量集中在少数方向），但「名义-偏差」分解在超高维下的解释力值得怀疑——这可能是控制论迁移到生成模型最需要重新发明的部件。
3. **评估的循环性第三次出现（VG 09/10 同病）**：Table 2 的八项指标里 ImageReward、PickScore、HPSv2/2.1、MPS 五项是人类偏好模型——SMC-CFG 在偏好指标上全面最优，与 VG 09 的「裁判由被告家属担任」是同一结构。独立一些的检验恰好是那两处没赢的格子（SD3.5 Aesthetic 输给 CFG-Zero⋆）和组合性 benchmark（Table 5/6 相对独立，且涨得最多——这是本文最硬的证据）。
4. **「零成本」的账要算到样本空间**。Table 8 的 FLOPs 逐位相同是事实，但 toy 实验 1/2 显示切换律有**恒定颤振地板**（$w = 1$ 时 SMC 的 TV 已是 CFG 的 3 倍——对象稳定时纯属白付）。FLOPs 度量的是算术量，不度量误差轨迹的高频抖动；「几乎免费」在算力维度成立，在采样轨迹维度至少要打个问号。
5. **$w$ 的双重身份没有被拆开**。在框架里 $w$ 既是 P 增益（进 $\mathbf u = w\,\mathbf e$）又是控制通道增益（Eq.22 里乘在 $\Delta\mathbf e$ 前）——SMC-CFG 想稳就得容忍大 $w$ 带来的大权限。但反过来说：如果只需要权限而不需要外推强度，为什么不把两者拆成 $w_{\mathrm{ext}}$（外推）与 $w_{\mathrm{ctrl}}$（控制）？论文 §9.1 显示 SMC-CFG 在大 scale 下持续提升，暗示的正是这种拆分方向，但没有明说。
6. **与 09/10 合读的终章教训**：对齐（梯度）、搜索（选择）、引导（切换）是干预一个预训练生成器的三条正交通道——三条通道的收益上限与风险上限全部同构于「干预信号的调制方式」：连续可微（可被 hack、收益无界）、离散选择（误选有界、对数回报）、非光滑切换（颤振有界、走廊受限）。**没有免费的方向，只有形状不同的代价**。VG 12 把这个观察整理成四类干预的总版图。

---

## 附录 A：toy 实验设定

控制 toy（numpy，确定性，无随机种子依赖——漂移用固定频率的余弦/正弦）：状态 $\mathbf e\in\mathbb R^2$；被控对象 $\dot{\mathbf e} = m(w,\|\mathbf e\|)\mathbf e + (w-1)\kappa J\mathbf e + \mathbf d(t)$，$m = -\lambda_0 + \alpha(w-1)(w-w_c)\rho^2/(\rho^2+\|\mathbf e\|^2)$（饱和反阻尼）、$\mathbf d(t) = \delta(\cos 1.7t, \sin 1.3t)$（有界漂移）；参数 $\Delta t = 0.05$、$T = 20$（400 步）、$\lambda_0 = 1$、$\kappa = 2.2$、$\alpha = 0.04$、$w_c = 2.5$、$\rho = 3$、$\mathbf e_0 = (2, 0.5)$。SMC 控制器按 Alg.1：$\dot{\mathbf e}$ 用一步后向差分估计、$\mathbf s = \dot{\mathbf e} + \lambda\mathbf e$、$\Delta\mathbf e = -k\,\mathrm{sign}(\mathbf s)$、修正经增益 $w$ 进入 $\dot{\mathbf e}$（Eq.22 控制仿射形式；论文 Alg.1 中修正改写 $\mathbf e$ 后合成速度，每步重新测量，语义等价）；默认 $\lambda = 2$、$k = 0.3$。指标：终值 $\|\mathbf e\|$、瞬态后（$t > 5$）mean、TV $= \sum_t\|\mathbf e_{t+1} - \mathbf e_t\|$（振荡能量代理，过饱和/伪影的代理量）、max。积分：每步线性部分（$m\mathbf I + (w-1)\kappa J$ 冻结在当前值）用 2×2 精确指数 $e^{m\Delta t}(\cos\theta\,\mathbf I + \sin\theta\,J)$、$\theta = (w-1)\kappa\Delta t$，控制与漂移项 Euler。相对论文的简化：误差空间取 2 维（真实 $\mathbf e$ 是 $10^5$ 维 latent 速度差，Theorem 1 的 $\rho D$ 项在 toy 中不存在）；被控对象的反阻尼/旋转是手工构造的失败机制（论文未给出真实失稳的定量刻画——Fig.1 是示意图）；漂移为确定性周期函数（论文 Assumption 1 只要求有界）。实现教训两条：① 显式 Euler 对纯旋转项数值恒不稳定（$|1 + \Delta t\lambda_{\text{rot}}| > 1$），首版在高 $w$ 段爆到 $10^{20}$，是积分器发散不是动力学发散——线性部分必须精确积分；② 控制项符号：$\dot{\mathbf e}$ 的差分估计方向写反（$(\mathbf e_{\mathrm{prev}} - \mathbf e)/\Delta t$）时控制器反向泵能量、SMC 全面劣于 CFG——滑模 toy 的第一检查项就是符号。

---

*下一篇：[视频生成 12：技术版图总结——引导 · 蒸馏 · 对齐 · 搜索](../VG_12_summary/)（系列终章：九篇论文收进一张「对预训练生成模型的四类干预」地图——梯度、选择、切换、蒸馏四种调制方式各自的收益形状与代价形状；01–11 的概念在版图上重新归位，附易混淆概念辨析总表）*
