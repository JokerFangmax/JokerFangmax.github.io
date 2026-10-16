---
title: "Forcing: 16. One-Forcing"
date: 2026-10-14
math: true
tags:
  - "Forcing"
  - "Video Generation"
  - "One-Forcing"
categories:
  - Forcing
series: forcing
weight: 16
description: "把因果视频生成压到单步的两大障碍：一致性蒸馏被教师轨迹几何卡住——Wan2.1 视频轨迹 92.5% 的曲率质量集中在 t≥0.9 高噪端（EDM2 图像教师无此现象），把中间态拉直后 4 步 VBench Total 从 61.16 升到 71.11；纯 DMD 单步又发糊。One-Forcing 在 DMD 上加与 fake-score 共享骨干的 GAN 判别损失、并用帧级自回归稳定对抗训练：单步 VBench Total 83.76，超此前单步因果基线 4–7 分，逼近 50 步 Wan2.1 的 84.26"
paper_id: "2605.23458v2"
---

# 单步生成的清晰度与动态性

> **Forcing 系列 · 第 16 篇 · 少步蒸馏与初始化**
> 论文：*One-Forcing: Towards Stable One-Step Autoregressive Video Generation*（arXiv:2605.23458v2，预印本）。
> 前置：[13 Causal Forcing](../forcing_13/)（本篇生成器的 ODE 初始化来源）、[07 Self Forcing](../forcing_07/)（自 rollout 训练框架）、[15 Causal Forcing++](../forcing_15/)（初始化与分布匹配的分工）；扩散与蒸馏基础见 [VG 01](../video_generation/VG_01_diffusion_basics/)、[VG 06 VideoScene](../video_generation/VG_06_videoscene/)。
> 本篇所有数字按论文 v2 正文与附录逐条核对。
>
> 📂 **阅读方式**：正文只保留主线——结论、关键公式与总结表；长推导、超参标本与消融全表收进折叠块（点击展开）。

---

## 论文信息

- **原题**：One-Forcing: Towards Stable One-Step Autoregressive Video Generation
- **阅读版本**：arXiv:2605.23458v2（预印本），对应本地收集的 PDF 版本。
- **原文**：[arXiv 摘要页](https://arxiv.org/abs/2605.23458v2) · [PDF](https://arxiv.org/pdf/2605.23458v2)

---

## 1. 这篇论文解决什么问题

### 1.1 单步 regime 的两种病

论文的中心问题：**因果视频生成器在单步（one-step）设定下能否保住视觉质量与运动动态？** 现有两类少步目标在单步下各得一种病：

| 目标类型 | 代表 | 单步症状 | 论文给出的病因 |
|---------|------|---------|---------------|
| 轨迹式（一致性蒸馏） | CD、rCM 系列 | 动态与结构丢失 | 教师轨迹的**曲率高度集中在高噪端**——单步生成完全依赖从最高噪声输入做端点预测，几何上先天困难 |
| 分布式（DMD） | Self Forcing | 帧发糊、细节缺失 | DMD 更新方向被**低频分量主导**，锐利边界与纹理的恢复被推迟（引 SAP-DMD 的分析） |

诊断数字（§4.1 详述）：Wan2.1 视频轨迹 **92.5%** 的曲率质量集中在 $t\ge0.9$ 的高噪区，而图像教师 EDM2 ImageNet-512 没有这种集中；把教师轨迹中间态拉直（端点不动）后，4 步一致性蒸馏的 VBench Total 从 61.16 升到 71.11——**轨迹几何本身就在卡蒸馏质量**。

### 1.2 答案与达到的数字

One-Forcing 的药方是 **DMD + 真实视频对抗监督 + 帧级自回归** 三件套：单步 VBench Total **83.76**、Quality 85.22，超此前所有单步因果基线（Self Forcing 77.18、Causal-Forcing 78.39、ASD 79.12）4–7 分，甚至超过 4 步的 Self Forcing（83.46）与 CausVid（81.18），距 50 步的 Wan2.1-1.3B（84.26）只差 0.5。

### 1.3 在系列中的位置

第四部分的第三篇。[15 Causal Forcing++](../forcing_15/) 解决「初始化怎么便宜地拿对」，本篇解决「初始化之后的单步分布匹配怎么做好」——论文自己把两者定位为互补（complementary）。GAN 类对抗目标是本系列首次出现的第三种训练信号（前两种：回归/一致性、score 蒸馏）。

---

## 2. 最小背景

- **一致性蒸馏（consistency distillation, CD）**：训练端点预测器 $f_\theta(x_t,t,c)$，使教师轨迹上相邻状态的预测一致，支持单步生成。相邻状态的一致性损失：

$$
\mathcal{L}_{\mathrm{CD}}(\theta) = \mathbb{E}_{x_t,t,c}\Big[\mathrm{MSE}\big(f_\theta(x_t,t,c),\ \mathrm{sg}\big[f_{\bar{\theta}}\big(\Phi_{\Delta t}(x_t,t,c),\ t-\Delta t,\ c\big)\big]\big)\Big],
\tag{Eq.1}
$$

其中 $\Phi_{\Delta t}$ 把教师轨迹从 $t$ 推进到 $t-\Delta t$，$f_{\bar{\theta}}$ 是学生的指数移动平均（EMA），$\mathrm{sg}$ 是 stop-gradient；

- **Self Forcing 的 DMD**：学生在自己的 rollout 上训练，给定生成序列 $x_\theta=G_\theta(z,c)$，DMD 通过反向 KL 梯度监督其分布：

$$
\nabla_\theta\mathrm{KL}(p_\theta\,\|\,p_{\mathrm{data}}) = \mathbb{E}_{x_\theta}\Big[J_\theta(z,c)^{\top}\big(\nabla_x\log p_\theta(x_\theta\mid c) - \nabla_x\log p_{\mathrm{data}}(x_\theta\mid c)\big)\Big],
\tag{Eq.2}
$$

$J_\theta$ 是生成器对 $\theta$ 的雅可比；score 差由可训练的 fake-score 网络与冻结教师估计（在加噪 rollout 上）。细节见 [07 Self Forcing](../forcing_07/) 与 [VG 06](../video_generation/VG_06_videoscene/)；
- **KV 缓存因果生成**：每个块条件于之前生成的内容（[06 CausVid](../forcing_06/) 起的标准架构）；
- **DMD2**：DMD 的对抗增强版（真实数据判别器）——本篇的直接基础。

---

## 3. 方法总览

从 Self Forcing 出发加两件事、改一件事：

| 模块 | 做什么 | 关键设计 |
|------|--------|---------|
| ① 共享骨干的对抗监督 | 在**可训练的 fake-score 网络**上加判别头 $D_\phi$，与去噪任务共享骨干 | 不引入独立判别器 backbone；判别头聚合所选 transformer 层的特征，对整段序列输出一个标量 logit；双向 critic 同时看帧外观与帧间关系 |
| ② 帧级自回归 rollout | 生成器一次预测 **1 个 latent 帧**，一步去噪，更新 KV 缓存后再预测下一帧 | 对比 chunkwise（一次 3 帧）：更细的时间分解让序列级对抗反馈能定位到单个因果预测 |
| ③ 联合训练调度 | 每次迭代更新 critic（fake-score 去噪 + 判别）；**每 5 次迭代**用独立 minibatch 更新一次生成器 | 生成器更新时 critic 参数冻结、但梯度穿过其输入；生成器梯度限制在选定 rollout 窗口内，KV 缓存更新被 detach |

生成器从 [Causal Forcing](../forcing_13/) 的 ODE 蒸馏检查点初始化；推理时沿用 ASD 的首帧增强（First-Frame Enhancement, FFE）：第一个块用 4 步预热建立 KV 缓存，之后单步流式。

---

## 4. 核心原理完整推导

### 4.1 轨迹几何：单步一致性蒸馏的先天障碍

度量教师轨迹「弯在哪」。取轨迹上相邻两点，比较**局部速度**与连接噪声端 $x_1$、数据端 $x_0$ 的**常值速度**之差：

$$
C(t_i) = \frac{1}{d}\left\|\frac{x_{t_i}-x_{t_{i-1}}}{t_i-t_{i-1}} - (x_1-x_0)\right\|_2^2,
\tag{Eq.3}
$$

$d$ 是 latent 坐标数、$t=0$ 为数据端、$t=1$ 为最高噪声端。直觉：如果轨迹是直线，局部速度处处等于端点连线方向，$C\equiv0$；$C$ 的大小刻画「这一段偏离直线演化多远」。

测量结果（论文 Fig. 3）：**Wan2.1-T2V-1.3B 的视频轨迹把 92.5% 的曲率质量集中在 $t\ge0.9$**，而一致性蒸馏常用的图像教师 EDM2 ImageNet-512 没有可比的集中。也就是说：视频生成**最偏离直线的转变发生在生成的起点**——最高噪声处。

对单步蒸馏的含义：4 步采样允许后续预测在更低噪声处**修正**初始估计；单步生成则完全依赖从最高噪声输入一次做对的端点预测——恰好撞在曲率最集中的地方。

**对照实验（轨迹拉直）**：保持教师的噪声端与输出端不动，只把三个中间态换成时间感知的线性插值，其余（初始化、样本顺序、优化设置）全部相同。300 步训练后的 4 步 VBench：

| 教师轨迹 | Total ↑ | Quality ↑ | Semantic ↑ |
|---------|--------|----------|-----------|
| 原始弯曲 | 61.16 | 63.92 | 50.11 |
| 拉直 | **71.11** | **75.37** | **54.05** |

端点对不变、只简化中间轨迹就涨近 10 分——**轨迹几何本身**（而非端点映射的难度）在消耗学生的训练预算。这同时论证了：与其压学生去复现教师轨迹（一致性路线），不如直接匹配输出分布（DMD 路线），让学生自己学噪声到数据的映射。

### 4.2 DMD 单步为什么糊：缺一路高频信号

Self Forcing 提供了分布匹配，但单步输出仍发糊、丢细节，且这些瑕疵会进入后续帧的条件上下文。论文引两条已有分析拼出解释：SAP-DMD 发现**低空间频率主导 DMD 更新方向**（先学粗结构、推迟锐边界与细纹理）；SDXL-Lightning 则报告其对抗损失**在低噪声处强调高频细节**。结论：DMD 的 score 差信号与真实数据的判别信号**频谱上互补**——后者正好补前者学得慢的那部分。

### 4.3 对抗监督：判别器嫁接在 fake-score 骨干上

判别头 $D_\phi$ 挂在可训练 fake-score 网络上（共享骨干），在**匹配的噪声等级**上比较真实视频 latent 与生成 rollout：真实视频加噪为 $x_{\mathrm{real},t}$，生成序列加噪为 $x_{\theta,t}$。对抗损失用 softplus 形式：

$$
\mathcal{L}^{\mathrm{adv}}_G = \mathbb{E}\big[\mathrm{softplus}\big(-D_\phi(x_{\theta,t},\ t,\ c)\big)\big],
\tag{Eq.4}
$$

$$
\mathcal{L}^{\mathrm{adv}}_D = \mathbb{E}\big[\mathrm{softplus}\big(-D_\phi(x_{\mathrm{real},t},\ t,\ c)\big) + \mathrm{softplus}\big(D_\phi(x_{\theta,t},\ t,\ c)\big)\big],
\tag{Eq.5}
$$

其中 $\mathrm{softplus}(u)=\log(1+\exp(u))$；更新判别器时生成样本被 detach。

**直觉**：Eq.4 只有一项——生成器希望判别器把假样本判真（logit 越大损失越小）；Eq.5 两项——判别器要把真样本判真、假样本判假。与 DMD2 的区别在**真实分布的来源**：这里「真」是训练集里的真实视频加噪（固定的高质量目标），而不是自蒸馏目标（见 §6.4 对 ASD 的判别器分析——这一步是成败关键）。

### 4.4 联合目标

生成器把教师引导与对抗监督相加：

$$
\mathcal{L}_G = \mathcal{L}_{\mathrm{DMD}} + \lambda_G\,\mathcal{L}^{\mathrm{adv}}_G,
\tag{Eq.6}
$$

共享的 critic 同时学 fake-score 去噪与判别：

$$
\mathcal{L}_{\mathrm{critic}} = \mathcal{L}_{\mathrm{fake}} + \lambda_D\,\mathcal{L}^{\mathrm{adv}}_D,
\tag{Eq.7}
$$

$\mathcal{L}_{\mathrm{fake}}$ 是 fake-score 在生成 latent 上的流去噪损失，$\lambda_G,\lambda_D$ 控制对抗贡献（实现都取 0.03，见 §5）。DMD 梯度用真实 score 残差的平均绝对值归一化（附录 A）。

### 4.5 为什么帧级自回归能稳定对抗训练

对抗目标对**整段序列**给一个标量 logit——序列级反馈要能定位到「哪个因果预测出了问题」才有效。帧级 rollout 每生成一个 latent 帧就把最新内容写进上下文，把生成分解成更小的条件预测问题；chunkwise 一次联合预测 3 帧才更新上下文。论文的假设（hypothesis，非证明）：**更细的时间分解让序列级对抗反馈更好地分配到单个因果预测上**，使学习在生成上下文持续演化的过程中更有效。实验支持：帧级不需要 R1/R2 正则就稳定、200 次迭代收敛；chunkwise 需要 R1/R2 且要 750 次迭代（§6.4）。

---

## 5. 训练与推理细节

| 项 | 设置 |
|----|------|
| 生成器 | Wan2.1-T2V-1.3B，从 [Causal Forcing](../forcing_13/) 的 ODE 蒸馏检查点初始化；每块 1 个 latent 帧、1 步去噪 |
| critic | 可训练 fake-score：Wan2.1-T2V-1.3B（+判别头）；冻结 real-score 教师：Wan2.1-T2V-14B |
| 训练长度 | 21 个 latent 帧（16×60×104/帧，对应 832×480）；8×H100，每卡 batch 1 |
| 更新调度 | 每迭代 1 次 critic 更新；每 5 迭代 1 次生成器更新（独立 minibatch） |
| 收敛 | 200 次迭代（步 400 复验仍稳定） |
| 推理 | 单张 H100；832×480、16 FPS；FFE——首块 4 步预热，之后单步 |

<details>
<summary><b>展开：噪声日程与完整超参标本（附录 A / Table 6）</b></summary>

**噪声日程**（Wan2.1 流匹配，偏移因子 $k=5$）：

$$
\sigma_t = \frac{k\,(t/1000)}{1+(k-1)(t/1000)},\qquad
x_t = (1-\sigma_t)\,x_0 + \sigma_t\,\epsilon,\qquad \text{目标} = \epsilon - x_0,
$$

生成时模型预测 $v_\theta(x_t,t,c)$ 并换算干净估计 $\hat{x}_0 = x_t - \sigma_t v_\theta(x_t,t,c)$。

**超参表（Table 6）**：

| 超参 | 值 |
|------|-----|
| 优化器 | AdamW，$\beta_1=0$、$\beta_2=0.999$、weight decay 0.01 |
| 学习率 | $1.0\times10^{-5}$（生成器与 fake-score critic 同） |
| EMA | 0.99，50 迭代后启动 |
| GAN 分支 | 层 {21, 29}，2 个 register token，特征维 1536、FFN 2048、12 头 |
| GAN 损失 | 非相对论 logistic 损失，$\lambda_G=\lambda_D=0.03$ |
| 判别器正则 | 无（帧级配置不需要 R1/R2） |
| CFG | real-score 引导 scale 5.0；生成器与 fake-score 不用 CFG |
| 系统 | 混合精度、梯度检查点、全分片 FSDP、激活 CPU offload |

**评测协议注记**：VBench 测试提示用 Qwen2.5-7B-Instruct 重写（沿 Self Forcing 惯例）；但轨迹拉直实验（Table 1）、GAN 目标消融（Table 4a）与 20 秒长视频评测用原始提示；部分消融用步 200 原始权重（无 EMA）、944 提示、单样本协议——与主表的五样本排行榜协议不同，对表时注意。

</details>

---

## 6. 实验与消融

### 6.1 主表：单步 83.76，越过多步基线

论文 Table 2（VBench，832×480，one-step 与 many-step 同表）：

| 模型 | 参数量 | NFE | Total ↑ | Quality ↑ | Semantic ↑ |
|------|-------|-----|---------|----------|-----------|
| MAGI-1 | 4.5B | 64 | 79.18 | 82.04 | 67.74 |
| Wan2.1（教师） | 1.3B | 50 | 84.26 | 85.30 | 80.09 |
| SkyReels-V2 | 1.3B | 30 | 82.67 | 84.70 | 74.53 |
| NOVA | 0.6B | 25 | 80.12 | 80.39 | 79.05 |
| LTX-Video | 1.9B | 20 | 80.00 | 82.30 | 70.79 |
| Pyramid Flow | 2B | 20 | 81.72 | 84.74 | 69.62 |
| CausVid | 1.3B | 4 | 81.18 | 84.41 | 68.30 |
| Self Forcing | 1.3B | 4 | 83.46 | 84.77 | 78.24 |
| Self Forcing | 1.3B | 1 | 77.18 | 79.40 | 68.34 |
| Causal-Forcing | 1.3B | 1 | 78.39 | 80.67 | 69.25 |
| ASD | 1.3B | 1 | 79.12 | 81.35 | 70.19 |
| One-Forcing（chunkwise） | 1.3B | 1 | 81.60 | 83.65 | 73.41 |
| **One-Forcing（framewise）** | 1.3B | 1 | **83.76** | **85.22** | **77.91** |

三个读法：单步因果基线被拉开 4–7 分（83.76 vs 77.18–79.12）；单步的 One-Forcing **超过** 4 步 Self Forcing（83.46）与 CausVid（81.18），也超过 LTX-Video、Pyramid Flow、NOVA 等 20–25 步模型；距 50 步教师 84.26 差 0.5——单步与多步的质量差距被压到边际。帧级比 chunkwise 高 2.16 分（83.76 vs 81.60），而 chunkwise 本身也已超过全部既有单步方法——目标函数跨块大小有效，帧级再添一截。

### 6.2 长视频：优势随 rollout 延长放大

20 秒视频（论文 Table 3，均单步推理）：

| 模型 | Total ↑ | Quality ↑ | Semantic ↑ | Multiple Objects ↑ |
|------|---------|----------|-----------|-------------------|
| Self Forcing | 75.30 | 77.47 | 66.61 | 39.33 |
| **One-Forcing** | **78.52** | **79.38** | **75.11** | **67.91** |

长 rollout 下 Semantic 差距从短片的 77.91−78.24（持平）扩大到 +8.50，多目标组合更是 39.33 → 67.91——自生成上下文越积越多时，真实数据锚定的对抗信号越显价值。

### 6.3 消融：GAN、FFE、初始化各值多少分

| 消融 | 对照 | Total ↑ | 结论 |
|------|------|---------|------|
| 对抗目标（Table 4a） | DMD-only vs DMD+GAN | 75.27 vs **80.36** | GAN 分支贡献约 5 分，且不依赖 FFE |
| 首帧增强（Table 4b） | 无 FFE vs 有 FFE | 80.69 vs **83.30** | FFE 贡献约 2.6 分；无 FFE 也已超既有单步基线 |
| 初始化（Table 5） | 因果 ODE vs 因果 CD | **83.76** vs 82.36 | ODE 初始化 Total 与动态占优（Dynamic 52.76 vs 23.61），CD 初始化语义更好（80.50 vs 77.91） |

初始化一行值得细读：ODE 轨迹数据里更丰富的运动信息给了生成器更强的运动先验；CD 初始化从已训练一致性的模型出发，利于语义对齐但**压抑大运动**。这与 [15 Causal Forcing++](../forcing_15/) 「CD 初始化全面替 ODE」的结论形成对照——初始化的优劣取决于后续目标（本篇是 GAN 增强的单步分布匹配）。

### 6.4 判别器健康度与帧级稳定性的证据

- **判别器 logit 差**（论文 Fig. 4，$|l_r-l_f|$，参考与生成视频的判别 logit 之差）：One-Forcing 全程保持大且活跃变动的 gap（$\mu=1.53$、$\sigma=1.20$）；ASD 从第一步起就贴近零（$\mu=0.001$、max < 0.006）——ASD 的「真」是 (n+1) 步模型输出对 n 步学生的自蒸馏目标，两侧都是不完美的模型输出，判别器拿不到有意义的学习信号、实质坍塌。**把对抗信号锚在真实数据上，是 GAN 类视频蒸馏的成败手筋**；
- **帧级 vs chunkwise 训练**：帧级无需 R1/R2 正则即稳定，chunkwise 需要；各自稳定配置下帧级 **200 次迭代**收敛、chunkwise 要 **750 次**。省掉 R1/R2 同时减少 28.59% 的训练 FLOPs（附录 J）。

<details>
<summary><b>展开：多样性、f-KL 正则、四步版与计时全表</b></summary>

**多样性与覆盖（附录 H，Table 9）**：加 GAN 不掉多样性——LPIPS-VGG 0.6437 → 0.6440（100 提示 × 4 种子）；recall 从 0.457 升到 **0.711**（256 视频）——对抗目标反而改善了分布覆盖。

**前向 KL 正则的反面教材（附录 G）**：给生成器加「回归到存储的 ODE 端点」的平方误差代理（$\lambda_{\mathrm{fkl}}=1$）严重伤害：Quality 掉到 75.03、Total 74.83、**Dynamic 1.30**（几乎无运动）——确定性回归锚与分布目标冲突，压制运动而非提升保真。

**四步版本（附录 I，Table 10）**：同样的 DMD+GAN 目标训 4 步模型——Total 83.85 / Quality 85.35，均高于 4 步 Self Forcing（83.46 / 84.77）；成像质量 71.12 vs 69.19、多目标 77.74 vs 75.76。收益不限于单步。

**推理计时（附录 J，Table 11；H200，21 个 latent 帧，20 次平均，端到端含 VAE 解码）**：

| 日程 | 稳态块延迟 | 扩散 | 端到端 |
|------|-----------|------|--------|
| 全单步 | 117.8 ms | 2.46 s | 4.13 s |
| FFE（首块 4 步） | 120.8 ms | 2.53 s | 4.19 s |
| 全 4 步 | 290.9 ms | 6.07 s | 7.72 s |

FFE 只比全单步多 0.06 s，比全 4 步快 **1.84×**——首帧增强的额外预算被限制在第一个块里。

**训练更久（附录 D，Table 7）**：步 200 → 400，Total 83.76 → 83.93、Quality 85.22 → 85.41——双倍预算下稳定不崩。

**曲率测量的稳健性（附录 B）**：100 条 Wan2.1-T2V-1.3B 的 50 步轨迹（shift 8、CFG 6）逐提示估计 92.49%±0.13%（95% bootstrap CI [92.24, 92.73]%），高噪/中噪比 33.1±0.7；去掉静态外观、只看时间差分的版本仍有 88.6% 集中（比值 19.3）；图像侧对照为 EDM2 ImageNet-512 教师的 8 条 256 步轨迹。

</details>

---

## 7. 与系列其他篇目的关系

| 篇目 | 连接点 |
|------|--------|
| [13 Causal Forcing](../forcing_13/) | 生成器直接从它的 ODE 蒸馏检查点初始化；Table 5 的初始化消融把 ODE 与 CD 初始化放在同一单步目标下对比 |
| [15 Causal Forcing++](../forcing_15/) | 互补定位：15 管「初始化怎么拿」（causal CD，帧级 2 步），本篇管「单步分布匹配怎么做」（DMD+GAN，帧级 1 步）；两篇对 CD vs ODE 初始化的结论不完全一致，合读能看出初始化选择依赖下游目标 |
| [07 Self Forcing](../forcing_07/) | 训练框架的直接底座（自 rollout + DMD）；本篇加上真实数据对抗分支并改帧级 |
| [06 CausVid](../forcing_06/) | CausVid 的 4 步成绩（81.18）被本篇单步超过——双向→因果蒸馏管线的质量上限被进一步推高 |
| [14 Diagonal Distillation](../forcing_14/) | 同样面对「少步伤运动/伤质量」：14 用流分布匹配在损失侧对齐运动分布（无判别器），本篇用 GAN 在分布侧补高频信号；14 的日程是 chunk 级前重后轻，本篇是帧级全 1 步 + 首块 FFE——两种单步化的路径 |
| [17 Causal-rCM](../forcing_17/) | 下一篇：把一致性初始化与分布匹配细化统一进一个框架——本篇与 15 的分工问题在那里被正面回答 |
| [VG 06 VideoScene](../video_generation/VG_06_videoscene/) | 少步蒸馏的图像/场景侧对照；DMD2 式对抗增强的用法可互参 |

**跨篇排榜警告**：本篇推理计时在 H100/H200、评测用 Qwen 重写提示；[14](../forcing_14/) 在 H100 + Tiny VAE；[15](../forcing_15/) 在 A800 且不含 VAE——三篇的 VBench/延迟数字口径不同，不能直接放一张榜上。

---

## 8. 局限与个人评价

1. **需要真实数据做「真」分布**：与 Self Forcing、ASD 等 data-free 方法不同，判别器要吃训练集真实视频——作者辩护说标准 forcing 类蒸馏本来就有训练视频可用，但这确实改变了方法的数据依赖性质；
2. **「单步」带星号**：FFE 让第一个块跑 4 步，严格说是一步流式 + 多步首帧；与 [15](../forcing_15/) 的 ASD 技巧同构，单步叙事的纯粹性要打折扣；
3. **Semantic 仍落后多步教师**：77.91 vs Wan2.1 的 80.09——单步的质量追平了（Quality 85.22 vs 85.30），语义对齐还有 2 分差距；
4. **帧级优势是假设不是定理**：§4.5 的「更细时间分解 → 更好的对抗信用分配」是 hypothesis，实验证据是相关性（帧级更稳、更快、更好），未做机制层面的受控验证；
5. **规模与长度**：只有 1.3B、21 帧 + 20 秒验证；更大骨干、更长视频、动作条件都留在 future work；
6. **评测协议的多口径**：主表五样本协议、消融单样本协议、部分用原始提示部分用重写提示——论文披露得很诚实，但跨表对数字时必须逐表核对脚注。

**个人评价**：这篇的两段论非常干净——先用一个可测量的几何量（曲率集中 92.5%）+ 一个只动中间态的对照实验（61.16 → 71.11）论证「轨迹式单步蒸馏先天困难」，再顺理成章地转向分布式目标并补上它缺的高频信号。判别器 logit 差的 ASD 对照是全文最锋利的一刀：同一个架构、同一种损失，只因「真」的来源不同就分出健康与坍塌。**面试可能被问的三个问题**：为什么视频单步一致性蒸馏比图像难？（曲率集中在高噪端，单步全靠端点预测——有 92.5% 的测量与拉直实验支撑）；GAN 损失和 DMD 各补什么？（DMD 给分布方向但低频主导，GAN 给真实数据的高频细节判别，频谱互补）；为什么判别器的「真」必须用真实视频？（自蒸馏目标两侧分布几乎重合，logit gap 贴零、判别器学不到东西——ASD 的 μ=0.001 是实证）。

---

*下一篇 [Forcing 17：Causal-rCM](../forcing_17/)——一致性初始化与分布匹配细化的统一。全系列目录见[总览](../forcing/)。*

{{< forcing_nav >}}
