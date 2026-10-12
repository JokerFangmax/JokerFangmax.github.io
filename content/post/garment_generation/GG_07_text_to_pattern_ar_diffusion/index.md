---
title: "GG 07: 文本到版型 I——自回归与并行扩散（DressCode 与 GarmentDiffusion）"
date: 2026-09-29
math: true
tags:
  - Notes
  - Garment Generation
  - Sewing Pattern
  - Autoregressive Models
  - Diffusion Models
categories:
  - Garment Generation
description: "版型生成的第一次正面交锋：DressCode 把版型量化成离散 token 用自回归 Transformer 逐个生成（PolyGen 式），GarmentDiffusion 用连续边 token + DiT 并行去噪——本篇用三个数值实验拆开'自回归为什么慢'（串行依赖墙，不是 FLOPs 墙：屋顶线模型 10²×）、'量化破坏什么'（闭合 Σv=0 与吃势一致性被逐分量量化随机游走式摧毁）、'50 步扩散够不够'（解析 score 下跨步 50 与 DDIM 50 都落在噪声地板附近；真正的墙在 <20 步，且 score 误差对确定性采样器的破坏远大于随机采样器——本篇 §5.3 含一次自我勘误）"
---

# GG 07：文本到版型 I——自回归与并行扩散

> 📖 服装生成系列第七篇（前篇：[GG 06：从观测恢复版型](../GG_06_pattern_recovery/)）。GG 05 讲了"程序造数据"，GG 06 讲了"观测反演版型"；从本篇起进入生成链路主干：**给一段文本，直接生成可制造的缝纫版型**。两篇主角论文代表这个任务的两代解法：DressCode 的**自回归离散 token**（PolyGen 路线）与 GarmentDiffusion 的**连续向量 + DiT 并行扩散**。论文细节全部对照 arXiv HTML 全文与官方仓库核实（含多处流行误传的勘误，见 §8.2）。

---

## 1. 这组论文解决什么问题？

**任务**：输入一句自然语言（"a long-sleeved hoodie dress with a-line silhouette"），输出完整的缝纫版型——panels + 边环 + 缝合 + 3D 放置，能直接进布料仿真（GG 01）与打版裁剪。与 GG 06 的恢复任务不同：这里没有观测，全部信息来自**条件 + 训练分布的先验**；与 GG 05 的 DSL 不同：不经过显式程序，由神经网络直接吐几何。

| 论文 | 条件 | 版型表示 | 生成模型 | 发表 |
|---|---|---|---|---|
| DressCode | 文本（+ GPT-4 交互解释） | 量化离散 token 序列 | SewingGPT：decoder-only 自回归 | SIGGRAPH 2024 (TOG 43(4)) |
| GarmentDiffusion | 文本 / 图像 / 草图 | 连续边 token 矩阵 | DiT：DDPM 并行去噪 | IJCAI 2025 Main |

两篇相隔一年，正好构成一场**受控对照**：同一个任务、近亲的边表示（都源自 NeuralTailor 的 4 维底座，GG 02 §3.1），唯独生成机制相反——**逐 token 串行** vs **全序列 50 步并行**。GarmentDiffusion 自报快约 100×（单 A10 不到 1 秒 vs DressCode 约分钟级）。

本篇要拆开三个问题，每个配一个数值实验：

1. **自回归为什么慢？** 直觉答案是"序列长"（18k token）。实验 1 证明这个直觉**错了一半**：真正的墙是**串行依赖结构**，不是浮点量——并行扩散的浮点量其实更多；
2. **量化 token 化付出什么代价？** GG 02 §6 用坐标 RMS（$h/\sqrt{12}$）分析过量化误差，但那漏掉了**跨 token 结构约束**：闭合条件 $\sum_i v_i = 0$ 与缝合长度一致性是全局约束，逐分量量化按随机游走累积破坏它们（实验 2）——这是 AR 离散路线的一个隐藏失分点，也是转向连续向量的动机之一；
3. **50 步扩散够吗？** GarmentDiffusion 训练 1000 步、推理只用 50 步。实验 3 用解析 score（零训练误差）隔离出纯"步数-积分误差"权衡，并顺带回答"为什么不干脆用 DDIM"。

### 方法流程鸟瞰

两篇的流程与每步动机（细节在 §3–4）：

**DressCode（离散 token + 自回归）**：

| 步骤 | 做什么 | 目的 |
|---|---|---|
| 1. 量化 | 版型参数 ×50 取整成离散 token，配 SOS/EOS/PAD（词表 ~2000） | 版型变成"语言"——能用 softmax + 交叉熵建模 |
| 2. 嵌入 | 每个 token = 位置 + 参数类别 + 数值 三元嵌入之和 | 同一个数字出现在不同参数槽位含义不同——类别嵌入消歧 |
| 3. 生成 | 24 层 decoder-only Transformer 逐 token 自回归；CLIP 文本条件经 cross-attention 注入 | 长度即面板数——"有什么"与"有多少"都在序列里（GPT-4 只是把多轮对话整理成标准描述的交互解释器，不参与生成） |
| 4. 反量化 | 整数序列还原成版型 | — |

代价：串行依赖 → 分钟级/件（实验 1：墙是串行结构不是浮点量）；逐分量量化随机游走破坏闭合/缝合约束（实验 2）。

**GarmentDiffusion（连续向量 + 并行扩散）**：

| 步骤 | 做什么 | 目的 |
|---|---|---|
| 1. 表示 | 版型拼成 1443 × 13 连续 token 矩阵（无量化、无词表） | 躲开量化的结构破坏；连续值可端到端回归 |
| 2. 去噪 | 12 层 DiT 全序列并行去噪（标准 DDPM 回归目标） | 并行：50 次前向 vs AR 的上千次串行 |
| 3. 条件 | 文本/图像/草图走解耦双通路 cross-attention（IP-Adapter 式，不用 adaLN） | 多模态各走各的支路——论文消融显示比调制进归一化层更好 |
| 4. 采样 | 从 1000 步训练 schedule 等距取 50 步跨步采样 | 步数-积分误差折中（实验 3：真正的墙在 <20 步） |

---

## 2. 背景最小集

| 需要 | 出处 |
|---|---|
| 4 维边编码 $(e_x, e_y, c_x, c_y)$、序列表示、stitch tags | [GG 02](../GG_02_pattern_representations/) §3.1, §5.1, §4.2 |
| 2021 Korosteleva & Lee 数据集（DressCode 的训练数据）、GarmentCodeData | [GG 05](../GG_05_garmentcode_data/) §4 |
| 版型序列的规范化/槽位/缝合分类器（生成侧全部沿用这些约定） | [GG 06](../GG_06_pattern_recovery/) §3 |
| DDPM 前向/反向、ε-预测、DiT、CFG | [VG 01](../../video_generation/VG_01_diffusion_basics/) |
| BRepGen（Xu et al., SIGGRAPH 2024，CAD 边 token 生成的先行者） | 本篇 §4.2 引用 |

---

## 3. 表示法详解：同一个版型的两种 token 化

### 3.1 DressCode：量化离散 token 序列

SewingGPT 把版型压成一个**整数序列**，每个整数是一个词表项。量化规则（对照论文 Sec. 4.2）：

- **边几何参数**（NeuralTailor 4 维底座）：乘 $C_E = 50$ 后取整——量化步长 2%（版型坐标单位下约 1–2mm 级）；
- **旋转/平移/stitch tag**（值域跨正负）：乘 1000、加 +1000 偏移再取整，保证非负；
- **特殊符号**：SOS = 2001、EOS = 2002、PAD = 2003，词表约 2000+。

**每个 panel 的 token 数**：$K=14$ 条边（槽位上限）× 每边 8 个 token（4 个边参数 + 3 个 stitch tag + 1 个 free/stitched flag）+ 7 个 panel 位姿 token（四元数 4 + 平移 3）$= 8K + 7 = 119$。序列上限 1500 token，约对应 **12 个 panel**。

**嵌入设计**（PolyGen 式三元嵌入）：一个 token 的 embedding 是三项之和——

$$
\text{emb}(p, k, v) = E_{\text{pos}}(p) + E_{\text{cat}}(k) + E_{\text{val}}(v),
$$

其中 $p$ 是**panel 内位置**（哪条边）、$k$ 是**参数类别**（$v_x$ / $c_y$ / 四元数分量…）、$v$ 是量化值。同一个数值 0.5 出现在 $v_x$ 与 $c_y$ 槽位上含义完全不同——类别嵌入负责消歧，这是 PolyGen 在 CAD 生成里验证过的设计。

### 3.2 GarmentDiffusion：连续边 token 矩阵

边 token 直接拼**连续向量**（BRepGen 式，论文明确致谢其启发）：

$$
\text{edge token} = \underbrace{e_3}_{\text{3D 起点}} \oplus \underbrace{c_3}_{\text{Bezier 控制}} \oplus \underbrace{a_3}_{\text{圆弧参数}} \oplus \underbrace{s_3}_{\text{stitch tag}} \oplus \underbrace{f_1}_{\text{stitch flag}},
$$

13 维（数据集相关——GarmentCodeData 上的示例是 9 维几何折算，GG 02 §3.1 表里的口径）。圆弧参数 $a_3$ 是半径 + 长/短轴标志 + 扫角。stitch tag $s_3$ 取"对应边的 3D 中点平均"——确定性几何特征而非可学习标签（SewingLDM 沿用同一招，GG 02 §4.2）。

**载体是定长矩阵**：$M \times N \times d_{\text{token}}$（GarmentCodeData V2 上 $M=37$ panel、$N=39$ 边/panel）——$37 \times 39 = 1{,}443$ 个 token 位置，缺席 panel/边靠 flag 维标记。没有量化、没有词表、没有 SOS/EOS。

### 3.3 对照：离散 token vs 连续向量买到了什么

| | DressCode 离散 token | GarmentDiffusion 连续向量 |
|---|---|---|
| 生成模型接口 | 词表 softmax，天然接 AR | 连续回归，天然接扩散 |
| 精度 | 量化步长封顶（$C_E=50$） | 浮点连续 |
| 结构约束 | 逐 token 独立量化——闭合/缝合一致性**无保证**（实验 2） | 连续值可被端到端 MSE 惩罚，但也**无硬保证** |
| 拓扑弹性 | 序列长度可变（≤1500） | 定长矩阵，$M \times N$ 封顶 |
| 训练目标 | 交叉熵（分类） | MSE（回归，单阶段端到端） |

注意第三行的一个微妙事实：**转向连续向量并没有解决结构约束问题，只是把"量化噪声破坏"换成了"回归误差破坏"**——缝合一致性在两代方法里都是软约束。实验 2 量化的那一层损失，是 AR 路线**独有**的额外失分。

---

## 4. 方法核心与数学推导

### 4.1 SewingGPT：decoder-only 自回归 + CLIP 条件注入

**架构**（论文 Sec. 4.3）：24 层 decoder-only Transformer，$d_{\text{model}} = 512$，PolyGen 式三元嵌入（§3.1）。

**文本条件**是 DressCode 最容易被误传的部分（§8.2 勘误 4）：文本先过 **CLIP 文本塔（1024 维）**，接一个**可训练 MLP** 投到 512 维，在生成过程中通过 **cross-attention** 注入。**不是接 LLM**——GPT-4 在 DressCode 里的角色是**交互式解释器**（把用户的自然语言/多轮对话整理成喂给 SewingGPT 的标准描述），GPT-4V 负责给训练数据**写 caption**（看正/背两面渲染图生成描述）。生成服装的"语言理解"全部由 CLIP + MLP 承担。

**训练目标**是标准 AR 交叉熵。记 token 序列 $x = (x_1, \dots, x_L)$（量化整数），条件 $c$（CLIP+MLP 后的 512 维向量）：

$$
\mathcal{L}_{\text{AR}} = -\sum_{t=1}^{L} \log p_\theta\!\left(x_t \mid x_{<t},\ c\right).
$$

推理从 SOS 起，逐 token 采样到 EOS。**长度即面板数**——序列结构里同时编码了"有什么"与"有多少"。

**训练配置**：数据是 2021 Korosteleva & Lee 数据集（约 19,264 个样本、11 个基础类型——**不是 GarmentCodeData**，见勘误 5）；单张 A6000 训练 30 小时；推理约 1 分钟/件（含交互流程约 3 分钟）。

### 4.2 GarmentDiffusion：DiT 并行去噪 + 解耦双模态条件注入

**去噪骨干**是标准 DiT：12 个 block、$d_{\text{model}}=768$、FFN 1024、8 heads。输入是 $1{,}443 \times d_{\text{token}}$ 的**整版 token 矩阵**（加位置编码），每步同时对全部 token 去噪。

**扩散过程**是教科书 DDPM（ε-预测，与 [VG 01](../../video_generation/VG_01_diffusion_basics/) 同一套记号）：

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_t}\, x_0,\ (1-\bar\alpha_t) I\right),
\qquad
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\big\| \epsilon - \epsilon_\theta(x_t, t, c) \big\|^2\right],
$$

$\beta_t$ 线性 $10^{-4} \to 2 \times 10^{-2}$、1000 步。训练是**单阶段端到端 MSE**：全部维度（几何 + tag + flag）一起回归，没有分阶段、没有分类分支。

**条件注入**是这篇架构上最值得注意的选择：**解耦多模态 cross-attention（IP-Adapter 式）**——文本特征与图像/草图特征各自走一条并行的 cross-attention 支路再融合，而**不是**标准 DiT 的 adaLN-Zero（把条件调制进 LayerNorm 的 scale/shift）。论文消融显示 adaLN-Zero 反而更差。文本塔是**冻结的 CLIP ViT-L/14-336 双塔**（文本塔 + 视觉塔），三份训练数据分别提供文本/图像/草图条件。

**推理 50 步**：从 1000 步训练 schedule 里等距取 50 个时间步做跨步采样（DDPM 式含噪更新，不是 DDIM η=0——这两者的差别正是实验 3 的内容）。消融比较过 50/200/500/1000 步，选定 50。

**训练配置**：三个数据集——SewFactory 13,707 件（图像条件）、DressCodeData 19,683 件（Llama-3.1-8B 生成文本描述 + MistoLine/SDXL 生成草图）、GarmentCodeData V2 115,195 件（文本）；8×A10，AdamW lr $10^{-4}$。

### 4.3 串行墙 vs 并行步：延迟的结构分解

自回归生成 $L$ 个 token 需要 $L$ 次**串行**前向（第 $t+1$ 个 token 依赖第 $t$ 个的输出）；DiT 生成需要 $T$ 次**并行**前向（每次全序列同时更新）。两边延迟：

$$
t_{\text{AR}} \approx L \cdot \underbrace{\frac{P \cdot b}{B_{\text{mem}}}}_{\text{单步: 读全部权重 } P,\ b\text{ bytes/参数}},
\qquad
t_{\text{DiT}} \approx T \cdot \frac{\text{FLOPs}(L, d, n)}{S_{\text{compute}}}.
$$

关键的不对称：AR 的每一步只处理**一个 token**——$d^2$ 量级的权重从显存搬运出来只喂一次乘加，**访存受限**（memory-bound），算力大量闲置；DiT 的每一步处理**全部 $L$ 个 token**——同一份权重复用 $L$ 次，矩阵乘法打满算力，**计算受限**（compute-bound）。这是屋顶线模型（roofline model）的两个极端。实验 1 把这个分解实测出来。

---

## 5. 实验与关键数字

实验脚本：`gg07_experiments.py`（numpy/scipy，无学习成分）。三个实验分别对应 §1 的三个问题。

### 5.1 实验 1：自回归的墙是串行依赖，不是浮点量

**(a) token 数**。同一件 GarmentCodeData 版型：SewingGPT 需逐个生成 **18,135+2** 个 token（GarmentDiffusion 论文口径；SewingLDM v2 论文口径是 "over 30k"——两文不一致，见 §8.2 勘误 6）；GarmentDiffusion 是 1,443 个 token 全并行 × 50 步。序列长度差 12.6–20×。

**(b) 等量 FLOPs 的原语对照**（numpy 实测，$d=1024$、12 层、$L=9000$）：**相互依赖**的单 token 前向链 vs **批量**全序列一次前向——浮点量完全相同：

| 结构 | 墙钟 | 有效算力 |
|---|---|---|
| 串行依赖链（AR 的结构） | 10.98 s | 20.6 GFLOP/s |
| 批量一次（DiT 每步的结构） | 1.08 s | 210.5 GFLOP/s |

**相同浮点量，仅"串行 vs 并行之差 = 10×**。单 token 前向喂不满运算管线（访存受限）；批量把权重复用 $L$ 次。

**(c) A10 屋顶线延迟模型**（代入两论文真实配置：DiT 12 层 $d=768$、$T=50$、$L=1{,}443$；AR $L=18{,}137$；A10 按带宽 600 GB/s、fp16 算力 ~100 TFLOP/s）：

| SewingGPT 参数量 | AR 墙钟（访存受限） | DiT 墙钟（计算受限） | 加速比 |
|---|---|---|---|
| 60M | 3.6 s | 0.063 s | 58× |
| 124M | 7.5 s | 0.063 s | 119× |
| 350M | 21.2 s | 0.063 s | 337× |

三个读数：

1. **DiT 的总浮点量更多**（6.3 vs ~4.5 TFLOPs）——并行扩散赢的不是"算得少"，是"算得满"。AR 是 18k 步串行、每步只搬权重算一个 token；DiT 是 50 步、每步打满 tensor core；
2. **模型复现了论文口径**：DiT 单 A10 一秒内 ✓、对 $10^8$ 参数级 AR 加速 ~10² ✓（论文自称 100× 但未附耗时表——本模型是该数字的机理还原，对参数量敏感：SewingGPT 真实参数量未公布，124M 附近正好落在 119×）；
3. 端到端加速 = **序列内并行**（访存受限 → 计算受限，~10–10²×，随参数量）× **步数缩减**（1000 → 50，20×）——两个正交的因子，下一实验验证第二个因子不赔质量。

### 5.2 实验 2：逐分量量化破坏跨 token 结构约束

GG 02 §6 的量化分析是**单坐标 RMS**（$h/\sqrt{12}$）。但版型的有效性条件是**全局**的：边环闭合（$\sum_i v_i = 0$，按位移向量）与缝合两侧长度一致（袖山 = 袖窿 × 吃势比 1.3043，两侧设计上就不等长）。DressCode 的量化对每个 token 独立取整——这两类约束怎么样了？

用 GG 05 实验 1 同款玩具 bodice 打版器（8 项随机身体测量 → 12 顶点前片），量化边向量 $v_i \mapsto \mathrm{round}(v_i / h) \cdot h$，20 个随机身体：

**(a) 闭合残差** $\|\sum_i \hat v_i\|$（缝纫容差 5mm）：

| $h$ | 中位 | 最大 | 超容差 |
|---|---|---|---|
| 20mm | 2.00 cm | 2.83 cm | 19/20 |
| 10mm | 1.00 cm | 2.24 cm | 18/20 |
| 5mm | 0.50 cm | 1.12 cm | 7/20 |
| 2mm | 0.20 cm | 0.28 cm | 0/20 |
| 1mm | 0.10 cm | 0.22 cm | 0/20 |

**(b) 吃势畸变** $|\delta_C - \delta_A|$（袖窿 3 边、袖山 5 边各自量化后的长度偏差之差——两侧本就不等长，比的是**量化引入的相对漂移**）：

| $h$ | 中位 | 最大 | 超 5mm |
|---|---|---|---|
| 20mm | 0.85 cm | 3.91 cm | 17/20 |
| 10mm | 1.17 cm | 2.46 cm | 13/20 |
| 5mm | 0.42 cm | 0.78 cm | 9/20 |
| 2mm | 0.18 cm | 0.49 cm | 0/20 |
| 1mm | 0.11 cm | 0.21 cm | 0/20 |

理论：量化误差各分量独立近似均匀 $[-h/2, h/2]$，闭合残差按随机游走累积 $\sim \frac{h}{\sqrt{12}}\sqrt{m}$（$m$ = 边数，含 $x/y$ 两分量）；吃势畸变 $\sim \frac{h}{\sqrt{12}}\sqrt{m + n}$。实测与该尺度吻合（$h=20$mm、12 边环：$\frac{2}{\sqrt{12}}\sqrt{12} \approx 2.0$ cm ✓）。

**结论**：$h \geq 1$ cm 时闭合必裂；$h = 5$mm（DressCode 的 $C_E = 50$ 大致落在这一档附近，取决于坐标归一化）已在容差边缘、吃势超差率 45%。**单坐标 RMS 看不出这些**——$h/\sqrt{12} \approx 1.4$mm 的"平均误差"背后是 7/20 的版型直接缝不上。AR 离散 token 的量化层在几何有效性上是有真实代价的；连续向量表示（GarmentDiffusion / SewingLDM）从根上没有这一层（它们的误差来源换成了回归误差，§3.3）。两篇论文的评测指标里**都没有**闭合/缝合长度一致性这一项——这本身就是个信号（§8.1）。

### 5.3 实验 3：50 步够吗——解析 score 下的步数消融

GarmentDiffusion 推理用 50 步（训练 1000 步）。步数砍 20× 赔了多少质量？为了让"质量退化"只含积分误差、不含 score 网络误差，用一个**解析已知 score** 的 toy 分布模拟"边向量分布"：三模态 GMM（权重 [0.5, 0.3, 0.2]、模内 σ = 50mm——版型里"这条边属于哪个结构"的多模态性）；VP 扩散下加噪分布仍是 GMM（均值缩放 $\sqrt{\bar\alpha_t}$、方差 $\bar\alpha_t\sigma^2 + (1-\bar\alpha_t)$）、score 解析精确；DDPM 训练 schedule 同论文（线性 $\beta$：$10^{-4}\to 2\times10^{-2}$、1000 步）。采样器对齐论文实践：**DDPM 式跨步**（等距取子序列、保留随机项，η=1），对照 DDIM（η=0）。质量用无偏 RBF-MMD²（中值启发式带宽）：

| 采样器 | MMD² | 模式权重 | 模内 σ |
|---|---|---|---|
| DDPM 1000 步 | 1.22e-04 | [0.49 0.30 0.20] | 65.8 mm |
| DDPM 跨步 200 | **−1.81e-04** | [0.50 0.30 0.20] | 60.6 mm |
| **DDPM 跨步 50** | **9.34e-04** | [0.49 0.30 0.20] | 64.9 mm |
| DDPM 跨步 20 | 2.59e-02 | [0.53 0.28 0.20] | 122.2 mm |
| DDPM 跨步 10 | 1.48e-01 | [0.53 0.23 0.24] | 178.2 mm |
| **DDIM 50（η=0）** | **4.21e-04** | [0.50 0.30 0.20] | 68.9 mm |
| DDIM 20（η=0） | 1.32e-02 | [0.51 0.29 0.20] | 116.7 mm |
| DDIM 10（η=0） | 1.12e-01 | [0.53 0.26 0.22] | 172.0 mm |

（噪声地板 1.18e-04：真分布两半互测的有限样本偏差；跨步 200 的负值即"低于地板"。）

三个读数：

1. **50 步是甜点，且对采样器不敏感**：跨步 DDPM 50（9.34e-04）与 DDIM 50（4.21e-04）都落在地板的一个数量级内、模式权重与真值几乎重合（真值 [0.495 0.305 0.20]）——快 20× 没有实质质量损失。真正的墙在 **20 步以下**：两种采样器同步退化，且**先爆模内 σ**（122 → 178mm，边向量整体变"糊"）、再漂模式权重——积分误差的表现顺序有诊断价值（GG 08 §7 会把这个顺序用作步数消融的诊断特征）；
2. **精确 score 下确定性 ODE 没有模式漂移**——这句是勘误后的结论。本篇初版曾报"DDIM 50 MMD² 9.08e-03 且模式权重漂移"，那是 toy 实现自己的坑：VP 动力学里误用了 VE 约定的解析 score（噪声均值不缩放）。同一份偏差 score 下，随机采样器几乎无感（跨步 50：1.01e-03 vs 正确 9.34e-04），确定性 DDIM 却被放大 ~45×（9.08e-03 vs 正确 4.21e-04）。**score 误差对确定性采样器的破坏远大于随机采样器**——噪声在不断重新注入能量、部分抵消 drift；ODE 没有校正项，偏差全程累积。这是工程上偏爱随机采样器的一个真实机理，也是 toy 实验最危险的失败模式：**score 写错时，随机采样器替你把 bug 藏起来**；
3. 本实验的边界：解析 score 下这是**纯步数-积分权衡**；真实模型再叠加 score 网络误差——由读数 2，这个误差对 DDIM 的威胁比对 DDPM 大，所以 GarmentDiffusion 选跨步 DDPM 在真实模型上大概率是稳妥的（但"toy 上 DDIM 更差"不成立）。论文的 50 步消融（真实模型）与本实验（解析 score）互补而非互相替代。

### 5.4 两篇论文自报的关键数字

**DressCode**（CLIP score 与用户研究，**没有 Panel L2 / FID**）：

| 方法 | CLIP score ↑ |
|---|---|
| Wonder3D* | 0.302 |
| RichDreamer | 0.324 |
| **DressCode** | **0.327** |

对比对象是 3D 生成方法（先生成 3D 再展版型），差距在噪声量级；主要说服力来自用户研究。失败模式论文自列：多层服装、单肩裙 → 双吊带、连帽与衣身混淆（§8.1）。

**GarmentDiffusion**（三数据集，vs SewingGPT / SewFormer）：

| 数据集 | 指标 | GarmentDiffusion | 对照 |
|---|---|---|---|
| DressCodeData（详细文本） | Panel L2 ↓ | **6.53** | SewingGPT 8.35 |
| | stitch F1 ↑ | **0.989** | SewingGPT 0.973 |
| GarmentCodeData（子集） | Panel L2 ↓ | **1.19e1** | SewingGPT 1.34e1 |
| | stitch F1 ↑ | **0.553** | **SewingGPT 0.589** |
| SewFactory（图像） | Panel L2 ↓ | **3.73** | SewFormer† 3.76 |

注意第二行：**几何全赢、GarmentCodeData 上的 stitch F1 反而输给 SewingGPT**（0.553 vs 0.589）。一个直觉解释：缝合一致性是**跨 token 约束**（实验 2 的主题）——AR 的串行条件链恰好把"前面生成的 token"作为上下文传播给了后续 token，等于免费带了一点跨 token 一致性；并行扩散每步对全部 token 独立去噪，缝合一致性完全依赖模型从训练数据里学。这解释了为什么 SewingLDM（GG 08）要专门把 stitch 信息拼进边向量并做潜空间联合建模。†SewFormer 单级 decoder 消融版。

---

## 6. 论文间关系

```
NeuralTailor 4 维边向量（GG 02/06 的公共底座）
        ↓ 量化成离散 token + PolyGen 三元嵌入
DressCode / SewingGPT (SIG'24)：文本 → 版型 = 自回归序列生成
        ↓ 同任务换生成机制：连续边 token + DiT 并行扩散（BRepGen 式 token）
GarmentDiffusion (IJCAI'25, Shenfu Research)
        ↓ 压缩潜空间 + 两步 IDDPM 训练 + 多模态条件（下一章主角）
SewingLDM (ICCV'25, 上交大) —— Work 3 P0 baseline
```

**两条外部技术输入线**在本篇汇合：

- **语言模型线**：PolyGen（CAD 生成）证明"程序化几何可以离散 token 化 + 自回归"——DressCode 是它在服装版型上的移植；
- **CAD 扩散线**：BRepGen（SIG'24）证明"边 token 可以连续向量 + 扩散并行"——GarmentDiffusion 致谢的直接灵感来源。GarmentImage 的栅格路线（GG 09）是第三条输入（图像生成线）。

**数据线的分岔值得单独记**：DressCode 训练在 2021 Korosteleva & Lee 数据集（约 19,264 件、11 类、上限 12 panel）上——这是它 1500 token 上限的来源；GarmentDiffusion 用 GarmentCodeData V2（115,195 件、1443 token 上限——GD 论文记 37 panel × 39 边；SewingLDM 代码实为 39 panel 槽 × 37 边/panel，乘积同、方向相反，见 GG 08 勘误）等三个数据集。SewingGPT 被**移植**到 GarmentCodeData 时 token 序列暴涨到 18k+（GarmentDiffusion 口径）或 30k+（SewingLDM 口径），SewingLDM 还报告 DressCode 在其上 11.4% 样本因超 GPU 限制无法训练——AR 路线跨数据集迁移的代价直接写进了后续论文的动机里。

**两篇互为对照实验**：任务相同、边表示同源、生成机制相反。实验 1–3 把"快 100×"分解成可复算的机理（串行墙 × 步数缩减），把"该不该离散化"分解成可量化的代价（结构约束随机游走破坏）——GG 08 会看到 SewingLDM 的答案：连续向量 + 压缩潜空间 + FSQ，同时回应本篇的全部三个问题。

---

## 7. 与 Work 3 的连接

Work 3（视频先验蒸馏 → 缝纫版型服装生成，SewingLDM P0 baseline）从本篇直接拿走三件事：

1. **表示决策的依据**：我们的 baseline 是连续向量 + 潜空间（SewingLDM），实验 2 给出这个选择的量化论证——离散量化的损失不在单坐标 RMS 而在跨 token 结构约束（闭合 19/20 超差 @ $h=20$mm），且这层损失**评测指标看不见**。P0 复现做误差分析时，闭合残差与缝合长度一致性应作为自建诊断指标补上（两篇论文都没有报）；
2. **推理预算的机理模型**：实验 1(c) 的屋顶线分解（访存/计算 × 串行/并行 × 步数）可以直接套在 Work 3 的推理管线预算上——SewingLDM 实测 ~6.5s/件，分解成潜空间 DiT 步数、token 数、条件注入开销后，哪些是可压缩的、哪些是硬下限，用同一套模型算。Work 3 要上视频条件（多帧注入），条件分支的开销增长也先在这个框架里估；
3. **stitch F1 的弱点是机会**：GarmentDiffusion 在 GarmentCodeData 上 stitch F1 输给 AR（0.553 vs 0.589）——并行扩散的跨 token 一致性缺口是当前 SOTA 的已知失分点。Work 3 的视频先验里天然含有**缝合结构的 3D 观测**（缝线在视频里可见），蒸馏信号恰好打在这个缺口上：这是"视频先验蒸馏"除了几何精度外的第二个可主张收益点。

另：实验 3 的"模内 σ 先爆、模式权重后漂"退化顺序，可作为 P0 复现时步数消融的诊断特征——如果 SewingLDM 降到某步数以下先出现边向量糊化（Panel L2 尚可但视觉版型"软"），说明撞的是积分误差墙，加步数即恢复；如果先出现拓扑漂移（panel 数错），那是 score 网络误差，加步数无用。

---

## 8. 局限与勘误

### 8.1 局限

**DressCode 自列**：多层服装（衬里/外套）不支持；单肩裙被生成成双吊带（对称先验压过了文本）；连帽设计易与衣身混淆。**方法层面**：评测只有 CLIP score + 用户研究——**没有 Panel L2、FID、缝合一致性等几何指标**（与 GarmentDiffusion 的对比数字全部来自后者的论文）；1500 token / 12 panel 上限继承自 2021 数据集，跨数据集迁移即撞墙（§6）。

**GarmentDiffusion 自列**：文本描述无法表达缝合/连接信息（"袖子接在哪"说不出来——靠训练分布先验补）；无数值参数控制（"胸围 96cm"做不到）；无身体条件（生成的版型不知道给谁穿）。**方法层面**：GarmentCodeData 上 stitch F1 0.553 低于 SewingGPT（§5.4）——跨 token 一致性是并行扩散的已知缺口；**官方仓库只挂了项目网页，无代码无权重**（截至写作时），复现只能靠论文描述；"100× faster, <1s on single A10" 是自报口径、**论文未附耗时表**（实验 1(c) 的屋顶线模型是其机理还原，但对参数量假设敏感）。

**两篇共有**：闭合性与缝合长度一致性都不在评测里（实验 2 量化的正是这块盲区）；文本条件都依赖 CLIP（冻结），语言-几何的对齐深度有限（DressCode 用 GPT-4 在外面绕、GarmentDiffusion 用 Llama-3.1-8B 造数据——都在数据/交互层打补丁，模型本体没变）。

### 8.2 勘误表（写作时逐一踩过、核对后修正）

**DressCode**：

| 误传 | 事实 | 依据 |
|---|---|---|
| 作者是 Korosteleva 组 | **Kai He et al.**（上海科技大学/UPenn/Deemos/NeuDim） | arXiv 作者页 |
| arXiv:2401.14159 | **arXiv:2401.16465**（2401.14159 是 Grounded SAM） | arXiv |
| 代码在 `maria-korosteleva/DressCode` | 该仓库 404；官方实现是 **`IHe-KaiI/DressCode`** | GitHub |
| "接入了 GPT-4/LLM 做文本编码" | 文本编码是 **CLIP(1024) + 可训练 MLP(→512) + cross-attention**；GPT-4 只是**交互解释器**、GPT-4V 只负责**给训练数据写 caption** | 论文 Sec. 4 |
| 训练数据是 GarmentCodeData | **2021 Korosteleva & Lee 数据集**（~19,264 样本、11 类）——GarmentCodeData 是 GarmentDiffusion/SewingLDM 的数据 | 论文 Sec. 5 |
| "SewingGPT 在 GarmentCodeData 上 18k token" | 18,135+2 是 **GarmentDiffusion 论文**的口径；**SewingLDM v2 论文说 "over 30k"**——两文不一致，引用时须注明出处 | arXiv:2504.21476 / arXiv:2412.14453 |
| 指标有 Panel L2/FID | 只有 **CLIP score + 用户研究**；Panel L2 口径的对比全部出自 GarmentDiffusion 论文 | 论文 Sec. 6 |

**GarmentDiffusion**：

| 误传 | 事实 | 依据 |
|---|---|---|
| 作者含 Yunjie Sun / Shengqi Liu（等） | **Xinyu Li, Qi Yao, Yuanda Wang（深赋 Shenfu Research）**；Shengqi Liu 是 **SewingLDM** 的一作 | arXiv 作者页 |
| 官方仓库有代码 | `Shenfu-Research/GarmentDiffusion` **只有项目网页，无代码无权重** | GitHub |
| "快 100×" 有耗时表支撑 | 论文正文自报（18,135+2 vs 1,443 token、<1s/A10），**无耗时表**；本篇实验 1(c) 是机理还原 | arXiv:2504.21476 |
| 条件注入是 adaLN-Zero（标准 DiT 式） | **解耦多模态 cross-attention（IP-Adapter 式）**；adaLN-Zero 在消融里更差 | 论文 Sec. 4 + 消融 |
| 推理是 DDIM | **DDPM 式跨步采样（保留随机项）**、50 步由消融选定；DDIM η=0 的对照见本篇实验 3 | 论文 Sec. 4.3 |

**本篇自勘误**（§5.3 读数 2 的初版错误，发布当日修正）：初版实验 3 在 VP 动力学里误用 VE 约定的解析 score（噪声均值不缩放），得出"DDIM 50 有模式漂移、比跨步 DDPM 差 7×"的错误结论。正确 VP score 下 DDIM 50 落在噪声地板（4.21e-04）、无漂移；真实结论反而是"score 误差对确定性采样器的破坏远大于随机采样器"（同一偏差：DDIM 恶化 45×、跨步 DDPM 几乎无感）。两个 toy 实现坑（σ 比值方向、VE/VP 混淆）都以注释形式留在实验脚本里（`scripts/garment_generation_experiments/gg07_experiments.py`）。

---

## 9. 习题

**题 1**：实验 1(c) 的屋顶线模型里，AR 延迟 $t_{\text{AR}} = L \cdot Pb/B$ 与 DiT 延迟 $t_{\text{DiT}} = T(L \cdot 12nd^2 + L^2 \cdot 2d)/S$。取 $L=18{,}137$、$T=50$、$d=768$、$n=12$、$B = 600$ GB/s、$S = 100$ TFLOP/s、fp16（$b=2$），求两者相等的临界参数量 $P^\ast$，并解释为什么 $P^\ast$ 与带宽成正比、与算力成反比。
*提示*：注意 DiT 的 FLOPs 里有 $L^2$ 项（self-attention）——$L=18{,}137$ 时它不可忽略；代入后 $P^\ast$ 在 $10^8$ 量级，正好解释"对 60M/124M/350M 加速比 58×/119×/337×"的跨度来自哪里。

**题 2**：证明实验 2(a) 的闭合残差尺度：若各边量化误差独立、每分量均匀分布于 $[-h/2, h/2]$，则 $\|\sum_i \delta v_i\|$ 的期望约为 $\frac{h}{\sqrt{12}}\sqrt{m}$（$m$ 条边）。再说明为什么实测的"超容差率"在 $h$ 减半时不是减半而是按 $\sqrt{}$ 下降。
*提示*：$\sum$ 的两分量各为 $m$ 个独立均匀变量之和（Irwin–Hall 分布），方差 $mh^2/12$；向量模长再走一遍瑞利分布的期望。

**题 3**：实验 2(b) 的吃势畸变 $\sim \frac{h}{\sqrt{12}}\sqrt{m+n}$（袖窿 $m$ 边、袖山 $n$ 边）。若袖山被加密成 9 条边（CAD 习惯），畸变尺度怎么变？这对"曲线边该用多少控制点"（GG 02 实验 1 的每参数表达力分析）加上了一条什么**工艺**约束？
*提示*：$\sqrt{3+9}/\sqrt{3+5} \approx 1.22$——加密本身抬量化噪声；约束：边数的选择在"表达力"与"量化/回归误差累积"之间取舍，不是越多越好。

**题 4**：推导 DDIM 的广义方差（Song 2020 Eq. 12）：

$$
\sigma_t(\tau') = \eta\,\sqrt{\tfrac{1-\bar\alpha_{\tau'}}{1-\bar\alpha_t}}\;\sqrt{1-\tfrac{\bar\alpha_t}{\bar\alpha_{\tau'}}},
$$

并验证 $\eta = 1$、$\tau' = t-1$（逐步）时 $\sigma^2 = \tilde\beta_t = \beta_t\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}$ 恰为 DDPM 后验方差。
*提示*：$1 - \bar\alpha_t/\bar\alpha_{t-1} = 1 - \alpha_t = \beta_t$。注意比值方向：$\tau'$ 比 $t$ 更接近数据，$\bar\alpha_{\tau'} > \bar\alpha_t$——**写反了会让根号内变负、被 `max(0,·)` 静默钳成 0，采样器无声退化成 η=0**（本篇实验 3 的真实踩坑：两组"不同"采样器出了逐位相同的结果，任何"两个采样器结果完全一样"都是 bug 信号）。

**题 5**（思考）：GarmentDiffusion 在几何指标全赢的情况下，GarmentCodeData 上的 stitch F1 输给自回归的 SewingGPT（0.553 vs 0.589）。给出两个候选解释并设计消融区分它们：(a) AR 的串行条件链天然传播跨 token 信息；(b) GarmentCodeData 的缝合模式更复杂，连续回归的 MSE 对离散缝合决策（stitch tag 的匹配）不是好的损失形状。
*提示*：(a) 的检验：给 GarmentDiffusion 加"teacher forcing 式两遍采样"（第一遍出全版型、第二遍以第一遍为条件重生成 stitch 维）——若 F1 回升则支持 (a)；(b) 的检验：把 stitch 维单独改成分類头（几何仍回归），若 F1 回升则支持 (b)。GG 08 会看到 SewingLDM 的实际选择：stitch tag 用**确定性几何特征**（对应边 3D 中点）而非自由学出的匹配——相当于在表示层绕开这个问题。

---

*下一篇：[GG 08：文本到版型 II——SewingLDM 深度精读](../GG_08_sewingldm_deep_dive/)（Work 3 P0 baseline）——29 维边向量、256×6 有限标量量化潜空间（codebook 0%→100% 的 Goldilocks 机理）、多模态条件注入，逐模块推导 + 论文自身 5 处错误的勘误 + 我们复现的实测经验。*
