---
title: "Forcing: 17. Causal-rCM"
date: 2026-10-14
math: true
tags:
  - "Forcing"
  - "Video Generation"
  - "Causal-rCM"
categories:
  - Forcing
series: forcing
weight: 17
description: "Causal-rCM（NVIDIA/清华，arXiv:2606.25473）精读：把 rCM 的 forward/reverse 散度互补性搬进自回归视频蒸馏——TF-CM 做离线一致性初始化，SF-DMD 做在线 rollout 细化，三阶段 recipe；用自定义因果 mask 的 FlashAttention-2 JVP kernel 首次实现 teacher-forcing 连续时间一致性模型（sCM/MeanFlow），TF-sCM 收敛比 TF-dCM 快 10 倍以上；2 步因果 Wan2.1-1.3B 在 VBench-T2V 拿到 84.63，帧级 1 步 15.9 FPS（单卡 H100）"
paper_id: "2606.25473v1"
---

# 统一一致性初始化与分布匹配细化

> **Forcing 系列 · 第 17 篇 · 少步蒸馏与初始化（第四部分收尾）**
> 论文：*Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation in Streaming Video Generation and Interactive World Models*（NVIDIA / 清华 / UT Austin）。
>
> 📂 **阅读方式**：正文只留主线；长推导、超参标本与消融全表收折叠块。

## 论文信息

- **原题**：Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation in Streaming Video Generation and Interactive World Models
- **阅读版本**：arXiv:2606.25473v1，对应本地收集的 PDF 版本。
- **原文**：[arXiv 摘要页](https://arxiv.org/abs/2606.25473v1) · [PDF](https://arxiv.org/pdf/2606.25473v1)

## 1. 这篇论文解决什么问题

[Self Forcing](../forcing_07/)（SF）解决了 teacher forcing 的训练-推理错配，但它依赖的 DMD/GAN 类目标建立在自己的生成样本上——这类 reverse 型目标对初始化极其敏感，容易模式坍缩。所以现有系统都在 SF 之前塞一段「初始化」：CausVid/SF 系用 ODE-pair 知识蒸馏（KD），[Diffusion Forcing](../forcing_04/) 系用 DF 式因果适配，还有 TF/DF 混合初始化。问题在于：**初始化目标、因果训练范式、蒸馏损失三者的组合空间没有人系统性地做过对照**——哪种初始化对 SF-DMD 最好？连续时间一致性模型能不能搬进因果设置？

Causal-rCM 的回答分三层：

1. **统一视角**：把 rCM 的 forward/reverse 散度互补性对应到自回归设置——teacher-forcing 提供离线、模式覆盖（mode-covering）的 forward 型信号，self-forcing 提供在线、模式追逐（mode-seeking）的 reverse 型信号（论文 Table 1 把 DDO、DiffusionNFT、DDRL、rCM 都纳入这张互补表）；
2. **系统性消融**：对 SF-DMD 的六种初始化（DF / TF / DF-KD / TF-KD / TF-dCM / TF-sCM）做了帧级与 chunk 级两套对照；
3. **基础设施**：首次实现 teacher-forcing 下的连续时间一致性模型（sCM / MeanFlow），靠的是支持自定义因果 mask 的 FlashAttention-2 JVP kernel——TF-sCM 收敛比 TF-dCM 快 **10 倍以上**（迭代数计）。

主结果：2 步因果 Wan2.1-1.3B 达 VBench-T2V **84.63**（帧级 1 步同样 84.63），超过 4 步的 Causal Forcing（81.56）与 Self Forcing chunk 级（83.76）；全部训练只用 rCM 提供的合成数据。

## 2. 最小背景

- **扩散与流匹配**：加噪、速度参数化、PF-ODE 的记号见 [VG 01：扩散基础](../video_generation/VG_01_diffusion_basics/)。本篇一律用整流流（rectified flow, RF）调度 $x_t = (1-t)x_0 + t\epsilon$，速度目标 $v = \epsilon - x_0$。
- **一致性模型（consistency model, CM）**：学一个映射 $f_\theta(x_t, t) \mapsto x_0$，把教师 ODE 轨迹上任意点直接送回起点；边界条件 $f_\theta(x, 0) \equiv x$ 由参数化 $f_\theta(x,t) = x - t F_\theta(x,t)$ 自动满足。
- **分布匹配蒸馏（distribution matching distillation, DMD）**：少步学生与教师分布在加噪边缘分布上做 reverse-KL 匹配，由学生分数与教师分数之差驱动更新；少步蒸馏的整体动机见 [VG 06](../video_generation/VG_06_videoscene/)。
- **三种因果训练范式**：TF（干净历史条件）、DF（逐 chunk 独立噪声等级）、SF（自回归 rollout 自生成历史）——系列 [04](../forcing_04/)/[07](../forcing_07/) 已分别展开；本篇关心的是「蒸馏目标 × 因果范式」的联合选择。

## 3. 方法总览：三阶段 recipe

Causal-rCM 与 rCM 的关键差别：rCM 把 CM 与 DMD **联合训练**，Causal-rCM 改为**顺序三阶段**——因为因果设置下联合训练会压低 VBench 上限（作者归因于因果教师与双向教师的分布差，见 §8）。

| 阶段 | 做什么 | 产出 |
|------|--------|------|
| **① TF 因果适配** | 用 teacher forcing 把双向 Wan2.1 蒸馏/转换为因果扩散模型 | 因果教师 + 学生初始化 |
| **② TF-CM 一致性蒸馏** | 在干净上下文 + 因果教师轨迹上做一致性蒸馏（dCM 或 sCM/MeanFlow） | 少步因果学生（SF-DMD 的初始化） |
| **③ SF-DMD 在线细化** | 学生自回归 rollout，DMD 损失直接优化推理时的少步分布 | 最终模型 |

模块逐个说：

**打包因果前向（TF 系列共用的底座）**。把「干净上下文 + 噪声目标」拼成一条序列，在自定义 TF 注意力 mask 下一次前向：每个噪声块只能看自己允许的干净历史和自己的噪声 token，损失只算噪声部分。mask 用 FlexAttention / MagiAttention 实现；替代的两遍方案（先缓存干净 KV 再让噪声 token attend 过去）要求干净 KV 留在计算图里，与激活检查点不兼容。

**TF-sCM 的 JVP 通路**。连续时间一致性目标需要对网络输出沿教师 ODE 求切线（tangent），通用 `torch.func.jvp` 过未融合注意力会物化巨大的中间量。Causal-rCM 在 rCM 的 FlashAttention2-JVP kernel 基础上扩展出自定义 mask 版本：mask 表示为「查询组 + 容许 key 区间」的稀疏矩形列表，在 FlashAttention-2 的 online-softmax 循环里同步流式计算 primal 与 tangent（附录 B 算法 1，五个累积器 $O, A, B, \ell, r$）。

**基础设施组合**。FSDP2 分片 + 展平 Ulysses 式上下文并行（CP）+ 选择性激活检查点（SAC）+ KV cache（disabled / append / readonly 三模式，pre/post-RoPE 两种缓存）+ 可选 replayed back-propagation（SF-DMD 省显存：先无梯度 rollout 存终态，再逐 chunk 重算最后一步可微部分）。所有两两组合（SAC × FlexAttention、JVP × FSDP2、JVP × CP、KV cache × CP）都在执行语义层做兼容而非独立开关。

## 4. 核心原理完整推导

### 4.1 一致性蒸馏：从相邻一致到连续时间切线

dCM 强制同一教师 ODE 轨迹上相邻两点的学生输出一致：

$$
\mathcal{L}_{\text{dCM}}(\theta) = \mathbb{E}_{x_0\sim p_{\text{data}},\,\epsilon,\,t} \Big[ w(t)\, d\big(f_\theta(x_t, t),\; f_{\theta^-}(\hat{x}_{t-\Delta t},\, t-\Delta t)\big) \Big]
\tag{Eq.1}
$$

其中 $\hat{x}_{t-\Delta t}$ 由教师 ODE 数值求解，$\theta^-$ 是停止梯度（EMA）版本。直觉：**同一条轨迹上的点必须映射到同一个起点**——满足边界条件后，任何 $t$ 一步就能跳到 $x_0$。

<details>
<summary><b>展开：Δt → 0 极限怎么变成切线目标（Eq.2 的来历）</b></summary>

把 $d$ 取为 MSE。对 $f_{\theta^-}(\hat{x}_{t-\Delta t}, t-\Delta t)$ 在 $(x_t, t)$ 处泰勒展开：$f_{\theta^-}(\hat{x}_{t-\Delta t}, t-\Delta t) = f_{\theta^-}(x_t, t) - \Delta t \cdot \frac{d f_{\theta^-}(x_t,t)}{dt} + O(\Delta t^2)$。代入 MSE 展开后，零阶项 $f_\theta - f_{\theta^-}$ 是参数化自身决定的偏差（RF 参数化下为 $\Delta v$ 的来源），一阶项给出瞬时一致性损失 $\mathbb{E}[w(t)\, f_\theta(x_t,t)^\top \frac{d f_{\theta^-}(x_t,t)}{dt}]$。其中全导数沿教师轨迹取：

$$
\frac{d f_{\theta^-}(x_t,t)}{dt} = \nabla_{x_t} f_{\theta^-}(x_t,t)\,\frac{dx_t}{dt} + \partial_t f_{\theta^-}(x_t,t) = \mathrm{JVP}\big(f_{\theta^-}, (x_t, t), (\tfrac{dx_t}{dt}, 1)\big)
$$

直觉：不需要真的解一步 ODE，只要在当前点沿教师速度方向做一次前向模式自动微分，就得到「轨迹移动时一致性输出怎么变」的一阶近似——这是连续时间目标省掉数值求解器的原因，也是必须写 JVP kernel 的原因。

sCM 再做两步数值稳定化：MSE 重整化 + 切线归一化，得到

$$
\mathcal{L}_{\text{sCM}}(\theta) = \mathbb{E}_{x_0,\epsilon,\,t\sim p_G}\left[ \left\| F_\theta(x_t, t) - F_{\theta^-}(x_t, t) - \frac{g}{\|g\|_2^2 + c} \right\|_2^2 \right], \qquad g = w(t)\,\frac{d f_{\theta^-}(x_t,t)}{dt}
\tag{Eq.2}
$$

切线归一化的直觉：一致性条件只约束切线的**方向**（为零），不约束幅度；把 $g$ 除以自身范数（加常数 $c$ 防除零）让回归目标量级稳定，梯度不随切线幅度爆炸。

</details>

MeanFlow 可以看成 sCM + CTM：CTM 给一致性函数加第二个时间条件 $s < t$（映射 $x_t \to x_s$），$s=t$ 的无穷小跳跃退化为扩散去噪器、可以锚定扩散损失；在 RF 调度下其连续时间目标（论文 Eq.3–4）化简出的 $\Delta_\theta$ 恰好是 MeanFlow 目标——所以 TF-sCM 的实现同时覆盖 MeanFlow（把 $dx_t/dt$ 换成教师速度即用于蒸馏）。

### 4.2 DMD：reverse-KL 的分数差形式

DMD 在加噪边缘分布上最小化 reverse-KL：

$$
\mathcal{L}_{\text{DMD-raw}}(\theta) = \mathbb{E}_t\big[\, D_{\mathrm{KL}}(p_\theta^t \,\|\, p_{\text{teacher}}^t)\,\big], \qquad x_t \sim q_{t|0}(x_t \mid x_0^\theta)
\tag{Eq.5}
$$

<details>
<summary><b>展开：从 Eq.5 到可执行的 Eq.7</b></summary>

对 $\theta$ 取路径导数（$x_0^\theta$ 依赖 $\theta$，$x_t$ 由重参数化跟着动）：

$$
\nabla_\theta \mathcal{L}_{\text{DMD-raw}}(\theta) = \mathbb{E}_{z,\epsilon,t}\left[ w(t) \Big( \nabla_{x_t}\log p_\theta^t(x_t) - \nabla_{x_t}\log p_{\text{teacher}}^t(x_t) \Big)^\top \frac{d x_t}{d\theta} \right]
\tag{Eq.6}
$$

直觉：reverse-KL 的梯度 = **学生分数 − 教师分数**，作用在学生样本的运动方向上——学生在自己生成的地方被教师分数「拉」向教师的高密度区，这正是 mode-seeking：只在有学生的地方修正，不保证覆盖。

两个执行困难与解法：（i）少步学生的分数不可得 → 训一个**假分数网络** $\varphi$（在学生样本上跑去噪回归 $\mathcal{L}_{\text{fake}}$，类似 GAN 判别器）；（ii）分数差在去噪器参数化下等价于「假去噪器 − 教师去噪器」（差一个可被 $w(t)$ 吸收的时间标量），再加自适应归一化（除以残差绝对均值），最终是停止梯度的 MSE：

$$
\mathcal{L}_{\text{DMD}}(\theta) = \mathbb{E}_{x_0^\theta\sim p_\theta,\,\epsilon,\,t\sim p_D}\left[ \left\| x_0^\theta - \mathrm{sg}\!\left[ x_0^\theta - \frac{f_{\text{fake}}(x_t,t) - f_{\text{teacher}}(x_t,t)}{\mathrm{mean}(\mathrm{abs}(x_0^\theta - f_{\text{teacher}}(x_t,t)))} \right] \right\|_2^2 \right]
\tag{Eq.7}
$$

学生与 critic 交替更新，构成类似 GAN 的对抗动态——这也是 DMD 对初始化敏感的根源：早期学生样本质量差时，假分数网络给出的修正方向噪声很大。

</details>

### 4.3 TF 与 TF-dCM：干净上下文上的一致性

TF 的核心是把单状态前向换成打包因果前向（上标 `noisy` 表示只取噪声部分的输出）：

$$
\Big[\, v_\theta\big( [x_0^{\text{clean}}, x_t^{\text{noisy}}],\, [0^{\text{clean}}, t^{\text{noisy}}];\, M_{\text{TF}} \big) \,\Big]^{\text{noisy}}
\tag{Eq.8}
$$

配 RF 回归目标 $v = \epsilon - x_0$ 就是普通 TF 目标（Eq.9）。TF-dCM 在此之上让噪声部分沿**因果教师** ODE 从 $t$ 走到 $t - \Delta t$（干净上下文不动），学生对两点做一致性回归（Eq.10）——与 Eq.1 的区别仅在于：函数评估永远带着 TF mask 与干净历史，教师轨迹也是因果教师的。

### 4.4 TF-sCM：因果 mask 下的切线目标（本篇技术核心）

定义噪声分支上的因果教师速度与 RF 一致性映射：

$$
v_{\text{teacher}}^{\text{TF}} = \Big[\, v_{\text{teacher}}\big( [x_0^{\text{clean}}, x_t^{\text{noisy}}],\, [0^{\text{clean}}, t^{\text{noisy}}];\, M_{\text{TF}} \big) \,\Big]^{\text{noisy}}
\tag{Eq.15}
$$

$$
\big[\, f_\theta^{\text{TF-RF}} \,\big]^{\text{noisy}} = x_t^{\text{noisy}} - t\, \Big[\, v_\theta\big( [x_0^{\text{clean}}, x_t^{\text{noisy}}],\, [0^{\text{clean}}, t^{\text{noisy}}];\, M_{\text{TF}} \big) \,\Big]^{\text{noisy}}
\tag{Eq.16}
$$

对 Eq.16 沿因果教师轨迹求全导数（干净分支切线为零，只有噪声分支跟教师速度）：

$$
h_{\text{TF-sCM}} = v_{\text{teacher}}^{\text{TF}} - \big[\, v_{\theta^-}(\cdots; M_{\text{TF}}) \,\big]^{\text{noisy}} - t\, \Big[\, \mathrm{JVP}\big( v_{\theta^-},\, ([x_0^{\text{clean}}, x_t^{\text{noisy}}], [0^{\text{clean}}, t^{\text{noisy}}]),\, ([0^{\text{clean}}, v_{\text{teacher}}^{\text{TF}}], [0^{\text{clean}}, 1^{\text{noisy}}]);\, M_{\text{TF}} \big) \,\Big]^{\text{noisy}}
\tag{Eq.17}
$$

三部分的直觉：$v_{\text{teacher}}$ 是轨迹的运动方向；减 $v_{\theta^-}$ 是「学生若当去噪器会怎么走」；减 $t\cdot\mathrm{JVP}$ 是一致性映射里 $-t v_\theta$ 项随轨迹的变化率。JVP 必须穿过**同一个** TF-mask 打包前向——这就是自定义 mask JVP kernel 存在的意义。最终目标：

$$
\mathcal{L}_{\text{TF-sCM}}(\theta) = \mathbb{E}_{x_0,\epsilon,t}\left[ \left\| \Delta v_\theta^{\text{TF}} - \frac{w(t)\, h_{\text{TF-sCM}}}{w^2(t)\, \|h_{\text{TF-sCM}}\|_2^2 + c} \right\|_2^2 \right]
\tag{Eq.18}
$$

其中 $\Delta v_\theta^{\text{TF}}$ 是同一输入下学生与 EMA 学生输出之差（Eq.19）。

<details>
<summary><b>展开：为什么必须用 RF-native 形式而不是 rCM 的 TrigFlow 包装（附录 A）</b></summary>

rCM 在双向设置下把 RF 速度模型包进 TrigFlow 再做 sCM 更稳。附录 A 证明这在因果设置下不成立。要点：

1. **坐标变换**：RF 时间 $u$ 与 TrigFlow 时间 $\tau$ 满足 $C=\cos\tau, S=\sin\tau, Z=C+S, b=Z^{-1}, u=S/Z$，状态间只差缩放 $x_\tau = Z x_u$（Eq.21）。TrigFlow 包装后的网络（Eq.23）与直接 RF 参数化定义**同一个**一致性映射（Eq.25 化简回 $x_u - u v_\theta$）。
2. **切线层面**：精确算术下 $h_{\text{trig}} = b^2 h_{\text{RF}}$（Eq.37）——包装不引入新的 JVP 结构，只差一个标量因子。
3. **但切线归一化打破等价**：把 $h_{\text{trig}}$ 代入归一化目标后（Eq.39），RF 坐标下分子分母的 $Z$ 幂次不匹配，梯度关系 $\nabla_\theta \mathcal{L}_{\text{TrigFlow-sCM}} = \frac{Z w_{\text{RF}}}{w_{\text{trig}}} \nabla_\theta \mathcal{L}_{\text{RF-sCM}}$ 只在 $c = 0$ 时成立（Eq.40）；$c>0$ 时分母分别是 $w_{\text{trig}}^2\|h_{\text{RF}}\|^2 + cZ^4$ 与 $w_{\text{RF}}^2\|h_{\text{RF}}\|^2 + c$，不可约化为重加权。
4. **有限精度**：$\mathrm{fl}[\mathrm{JVP}(v_{\theta^-}, \cdot, b^2(V,1))] \neq b^2\,\mathrm{fl}[\mathrm{JVP}(v_{\theta^-}, \cdot, (V,1))]$（Eq.41）——缩放放进 JVP 方向里会逐层传播，放在外面则事后重缩放，混合精度/融合 kernel 下两者不等价；且 TrigFlow 推导中的状态项相消是代数精确而非比特精确。

实证结论：因果 TF 设置下 TrigFlow 包装的 TF-sCM 生成质量退化，RF-native 版本输出更平滑。**「调度解析等价 ⇒ 训练目标等价」在归一化 MSE 目标下不成立**——这个反例本身值得记住。

</details>

### 4.5 SF-DMD：rollout、缓存追加与梯度截断

阶段③里学生做时间自回归 rollout（Eq.11）：第 $i$ 个 chunk 基于已缓存的历史生成 $\tilde{x}_0^i = G_\theta(z^i \mid \mathrm{KV}_{<i})$；生成完做一次 cache-update 前向，把该 chunk 的干净 KV 追加进缓存（Eq.12，$\mathrm{sg}[\cdot]$ 表示缓存天然不传梯度）。chunk 内部是从纯噪声出发的少步自去噪（Eq.13），每步转移是 CM 式「反去噪 + 前向加噪」（Eq.14）：

$$
\tilde{x}_{t_{n-1}}^i = (1 - t_{n-1})\, f_\theta\big(\tilde{x}_{t_n}^i,\, t_n \,\big|\, \mathrm{KV}_{<i}\big) + t_{n-1}\,\epsilon_n, \qquad \epsilon_n \sim \mathcal{N}(0, I)
\tag{Eq.14}
$$

**梯度截断**：中间去噪步与历史 KV 全部 detach，只有每个 chunk 的最后一步 $t_1 \to 0$ 可微，DMD 损失（Eq.7）只穿过这一步反传。训练时步数 $N$ 从 $[1, N_{\max}]$ 随机采样。

**自定义步数调度**（Eq.20）：不同 chunk 可用不同步数（如 2 步模型用 $[4, 2, 2, \dots]$——首 chunk 建立全局场景，多给计算）。SF-DMD 训练按迭代轮换 $[1,1,1] \to [2,2,2] \to [3,2,2] \to [4,2,2] \to \cdots$，因为只有最后一步可微，轮换让**不同的去噪区间**轮流成为被监督的终步，而不是永远只监督最大步数采样器的最后一段。

### 4.6 noisy context：N+1 → N

干净上下文的因果推理每 chunk 需要一次额外的干净编码前向，$N$ 步模型实际每 chunk 花 $N+1$ 次函数评估（NFE）。noisy context 复用最后一次去噪前向的 KV 作为后续 chunk 的上下文，把延迟从 $N+1$ 压到 $N$；残余噪声同时充当低通滤波器，抑制累积的高频伪影。TF 阶段引入 noisy context 的方式是把打包前向里的干净历史换成对应上下文时间步的噪声 token；实践上只在最后 SF 阶段用就够。

## 5. 训练与推理细节

主设置：Wan2.1 T2V，480p（832×480），81 RGB 帧 → 21 潜帧；两种 chunk 模式：帧级 c1-1（1 初始潜帧 + 逐 1 帧 chunk）、chunk 级 c3-3（3 + 3）。训练数据全部是 rCM 提供的合成 T2V 数据（Wan2.1-14B 教师，100 步 Euler，shift 3.0，CFG 5.0 生成）；主学生 Wan2.1-1.3B，蒸馏教师为 14B。TF-CM 用 TF 训练的 14B 因果教师；SF-DMD 用 14B **双向**教师与假分数网络。

推理调度（潜空间时间步）：

| 模型 | 调度 | NFE |
|------|------|-----|
| 4 步 | $[15/16,\ 5/6,\ 5/8]$ | 5 |
| 2 步 | $[15/16, 5/6, 5/8] + [5/6]$ | 3 |
| 2 步 noisy ctx | $[15/16, 5/6, 5/8] + [5/8]$（复用终步前向当上下文缓存） | 2 |
| 1 步 | $[15/16, 5/6, 5/8] + [\,]$ | 2 |

（首 chunk 一律 4 步。）效率测量：batch 1、单卡 H100，FPS 与延迟含扩散采样和 VAE 解码。

<details>
<summary><b>展开：三阶段训练超参标本（论文 Table 3）</b></summary>

| 配置 | ① TF/DF 1.3B | ① TF/DF 14B | ② TF-dCM 1.3B | ② TF-sCM 1.3B | ③ SF-DMD 1.3B |
|------|------|------|------|------|------|
| 全局 batch | 256 | 64 | 32 | 32 | 64 |
| CP 规模 | 1 | 8 | 4 | 4 | 4 |
| 学生优化器 | AdamW lr $10^{-5}$, $\beta{=}(0.9, 0.999)$, wd 0.01 | 同左 | AdamW lr $2{\times}10^{-6}$, $\beta{=}(0, 0.999)$ | 同左 | 同左 |
| 假分数优化器 | – | – | – | – | AdamW lr $4{\times}10^{-7}$, $\beta{=}(0,0.999)$ |
| CFG scale | – | – | 3.0 | 3.0 | 5.0 |
| 时间采样/加权 | TF: $p_G{=}$UniformShift(5)、共享 $t$、高斯钟权；DF: 逐 chunk 随机 $t$、无权 | 同左 | 均匀 RF 网格 shift 3、48 步、skip 1 | $p_G{=}$LogitNormal$(\mu{=}-0.8,\ \sigma{=}1.6)$ | $p_D{=}$UniformShift(5) |
| 特殊超参 | – | – | – | tangent warmup 1000 | 最大 rollout 步 4；学生更新频率 6 |
| 迭代数 | 30k | 30k | 10k | 1k | 随设定变化（1.2k–3.2k） |

注意 TF-sCM 只训练 1k 迭代就够——10 倍收敛优势的直接体现。SF-DMD 的最大 rollout 步数 4、随机 $N \in [1, 4]$。

</details>

## 6. 实验与消融

### 6.1 主结果（VBench-T2V，Wan2.1-1.3B 底座）

| 方法 | 设置 | NFE | Total ↑ | FPS ↑ | 二级延迟 (s) ↓ | SF-DMD 迭代 |
|------|------|-----|--------|-------|--------------|------------|
| Wan2.1-1.3B（双向） | – | 50×2 | 82.78 | 0.72 | – | – |
| Wan2.1-14B（双向） | – | 50×2 | 83.35 | 0.18 | – | – |
| Causal Forcing | 帧级 4 步 | 5 | 81.56 | 8.3 | 0.46 | – |
| **Causal-rCM** | 帧级 4 步 | 5 | 84.29 | 8.3 | 0.46 | 1200 |
| **Causal-rCM** | 帧级 2 步 | 3 | **84.63** | 12.2 | 0.31 | 3000 |
| **Causal-rCM** | 帧级 2 步 noisy | 2 | 83.11 | 15.9 | 0.23 | 1500 |
| **Causal-rCM** | 帧级 1 步 | 2 | **84.63** | 15.9 | 0.23 | 3000 |
| Self-Forcing | chunk 4 步 | 5 | 83.76 | 17.4 | 0.64 | – |
| LongLive | chunk 4 步 | 5 | 83.62 | 17.4 | 0.64 | – |
| Causal Forcing | chunk 4 步 | 5 | 83.96 | 17.4 | 0.64 | – |
| AnyFlow | chunk 4 步 | 5 | 84.31 | 17.4 | 0.64 | – |
| **Causal-rCM** | chunk 4 步 | 5 | 84.37 | 17.4 | 0.64 | 1250 |
| **Causal-rCM** | chunk 2 步 | 3 | 84.30 | 22.2 | 0.49 | 2500 |
| **Causal-rCM** | chunk 2 步 noisy | 2 | 84.24 | 25.6 | 0.41 | 1750 |
| **Causal-rCM** | chunk 1 步 | 2 | 84.01 | 25.6 | 0.41 | 3000 |

两个反直觉读法：

1. **帧级 1/2 步反超 4 步**：单潜帧 chunk 内部没有时间结构可去噪，给未来每个 chunk 都配 4 步反而放大自回归反馈误差（叠加梯度截断）——4 步帧级 SF-DMD 出现一致的相机漂移，只能稳定训练约 1k 迭代；1–2 步压住漂移、可训约 3k 迭代。chunk 级相反：3 潜帧 chunk 有内部时间相关性，4 步更深采样最好（84.37 领先）。
2. **noisy context 的最优步数依设置而变**：帧级 1 步 > 2 步 noisy（单帧 chunk 对上下文残余噪声敏感，细化收益有限）；chunk 级 2 步 noisy > 1 步（三维 token 组冗余度高，高斯扰动难以均匀破坏整个 chunk 的结构，多一步保内部时间连贯）。

### 6.2 TF-sCM vs TF-dCM：10 倍收敛

进入 SF-DMD 之前，TF-sCM 一致性给出更强初始化且迭代数少一个量级：帧级 TF-sCM 在 1–2k 迭代内 VBench 超 81.8，已超过 TF-dCM 训 10k 的结果；chunk 级 TF-sCM 1–2k 迭代破 83，TF-dCM 长训后仍更低（论文 Fig. 6）。

### 6.3 初始化策略消融（4 步 SF-DMD）

| 初始化 | 帧级 Total ↑ | 帧级 SF-DMD 迭代 | chunk 级 Total ↑ | chunk 级迭代 |
|--------|------------|----------------|-----------------|-------------|
| DF | 83.11 | 2000 | **84.80** | 1500 |
| TF | 82.62 | 1000 | **84.95** | 1000 |
| DF-KD | 80.59 | 2000 | 83.61 | 1500 |
| TF-KD | 83.49 | 1250 | 83.79 | 1000 |
| **TF-dCM** | **84.29** | 1200 | 84.33 | 3200 |
| TF-sCM | 83.84 | 1000 | 84.37 | 1250 |

结论三连：**（i）帧级 TF-CM 最优**；**（ii）chunk 级 DF/TF 分数最高（逼近 85）但有代价**——定性检查（论文 Fig. 8）显示这些模型输出过平滑、过饱和的纹理（水面、头发、树叶），细粒度细节明显更少，综合看 TF-CM 仍是最可靠选择；**（iii）初始化质量 ≠ 细化稳定性**：TF-sCM 起点更强，但帧级下 TF-dCM 在长 SF-DMD 细化中更稳、峰值更高（作者在局限里再次承认这一点）。

### 6.4 交互世界模型：Cosmos 3

把 Causal-rCM 应用到 Cosmos 3（双塔 Mixture-of-Transformers：理解塔 UND + 生成塔 GEN）：把 GEN 视觉流改成时间因果自回归堆栈——每个潜帧当作一个视觉超 token（supertoken），超 token 间时间因果、超 token 内空间全双向；前向动力学布局里 $A_i$ 控制 $V_i \to V_{i+1}$ 的转移，$V_0$ 前插一个空动作 token 保持布局统一。同一因果堆栈支持 T2V / I2V / 前向动力学三种模式；自动驾驶场景下左转/右转/直行给出可分辨的不同未来轨迹（论文 Fig. 10）。

## 7. 与系列其他篇目的关系

本篇是第四部分（少步蒸馏与初始化）的收束，核心是把 15/16/17 三条初始化-细化路线放进同一张对照表。15/16 各自论文的数字不在本文核对范围内，表内只放本文可核对的部分：

| 维度 | [15: Causal Forcing++](../forcing_15/) | [16: One-Forcing](../forcing_16/) | 17: Causal-rCM |
|------|------|------|------|
| 一致性初始化 | causal CD：相邻时间步**在线**教师 ODE 单步监督 | 无 CM 初始化（DMD 为主） | TF-CM：离线教师轨迹，dCM 或 sCM/MeanFlow |
| 细化目标 | SF-DMD | DMD + 辅助 GAN（高噪声端轨迹曲率动机） | SF-DMD |
| 技术增量 | 可扩展少步因果蒸馏流程 | 单步清晰度-动态性权衡 | 自定义 mask JVP kernel + 系统性初始化消融 + 开源 infra |
| 本文可核对数字 | –（Causal Forcing 4 步帧级 81.56 / chunk 83.96 可作同底座参照） | – | 2 步 84.63；TF-sCM 快 10×；1 步 15.9 FPS |

- **[13: Causal Forcing](../forcing_13/)** 的单射性分析解释了为什么 TF 阶段用因果教师（TF 让教师看见干净历史，匹配 TF-CM 蒸馏时的上下文分布）；论文也提到 APT2 用 TF-dCM 初始化 SF-GAN 的做法后来获得 frame-level injectivity 的理论支持。
- **[07: Self Forcing](../forcing_07/)**：SF-DMD 阶段基本沿用其 rollout + KV cache + DMD 执行方式（含梯度截断），本篇的新增是把它放进了「forward 初始化 → reverse 细化」的解释框架。
- **[04: Diffusion Forcing](../forcing_04/)**：DF 初始化在 chunk 级分数高但纹理过平滑——为「DF 暴露于合成噪声 ≠ rollout 误差分布」的论点提供了新证据。
- **蒸馏基础**（[VG 06](../video_generation/VG_06_videoscene/)）：dCM→sCM→DMD 的谱系与少步蒸馏动机。
- 论文 Table 6 给出更远的路线图：CM 路线（dCM → sCM → DMD/GAN）与 CTM 路线（MeanFlow-FD → MeanFlow-JVP → SF-DMD），双向/因果各一支——现有方法（APT、rCM、APT2、CF++、AnyFlow、Transition Matching）都是两条终极 pipeline 的子集。

## 8. 局限与个人评价

**论文自认的四条局限**：**（i）** 帧级长 rollout 脆弱——4 步帧级 SF-DMD 长训出现方向一致的相机漂移，只能训约 1k 迭代（作者推测动作条件能靠显式运动先验消除歧义）；**（ii）** 初始化质量与细化稳定性错位（§6.3-(iii)）；**（iii）** 联合训练仍是难题——因果设置下 CM+DMD 联合训练压低 VBench 上限，只能退回分阶段；**（iv）** kernel 层面——自定义 mask JVP kernel 是 Triton 实现，TF-sCM 单迭代速度只与标准 FlashAttention-2 的 TF-dCM 相当，落后 FlashAttention-3/4。

**我的评价**：贡献排序应该是「系统性消融 + 开源 recipe > JVP kernel > 84.63 这个分数」。三点观察：

- 「forward 离线保覆盖、reverse 在线提质量」的互补叙事有 Table 1 的跨领域旁证（DDO/DiffusionNFT/DDRL），但叙事与消融之间有一道缝：消融只覆盖了初始化目标的组合，**没有**做「TF-CM 与 SF-DMD 联合 vs 顺序」的受控对照（联合训练只在局限里一句带过）；
- RF-native vs TrigFlow 的附录 A 是全文最扎实的部分——「解析等价的调度在归一化目标下不等价、在有限精度下更不等价」对所有想换调度包装蒸馏目标的工程决策都适用；
- 全合成数据训练（继承 rCM）拿到 84.63，对数据侧是一个强信号，但语义分（80–81）相对质量分（85+）始终偏低，合成数据的语义瓶颈没有讨论。

*下一篇 [Forcing 18：Separable Causal Diffusion](../forcing_18/) 转入第五部分「效率与上下文梯度」：不再问蒸馏目标怎么配，而问**架构上**——时间推理为什么要在每个去噪步、每一层重复做一遍。*

{{< forcing_nav >}}
