---
title: "GG 08: 文本到版型 II——SewingLDM 深度精读（Work 3 P0 baseline）"
date: 2026-09-29
math: true
tags:
  - Notes
  - Garment Generation
  - Sewing Pattern
  - Latent Diffusion
  - Reproduction
categories:
  - Garment Generation
description: "我们的 P0 baseline 逐模块拆解：29 维边向量（代码权威分解 19 连续+10 离散，论文 Eq.1 自己只有 28 维）、MLP-Mixer 自编码器 + 有限标量量化 256×6 潜空间（codebook 利用率 0%→100% 的 Goldilocks 消融）、PixArt-α 式 DiT 24×1152 + IDDPM、T5-XXL + 15 维体型 PCA + 草图的统计归一化残差注入——配三个数值实验（FSQ 大潜空间两侧都亏：重建饱和 + 码字复用崩塌；CFG guidance 压结构多样性的解析机理：模式质量按似然比几何衰减，P0 实测 n_panels {2:18, 4:2} 的解释）、论文自身 5 处错误的勘误表、以及我们复现的实际经验"
---

# GG 08：SewingLDM 深度精读

> 📖 服装生成系列第八篇（前篇：[GG 07：文本到版型 I](../GG_07_text_to_pattern_ar_diffusion/)），也是全系列写得最深的一篇——**这是我们 Work 3（视频先验蒸馏 → 缝纫版型服装生成）的 P0 复现 baseline**，本篇除了论文精读还带上我们复现的实测经验。论文细节对照 arXiv v1/v2 两个版本全文 + 官方代码逐文件核对（仓库与权重均已放出）；论文本身有 5 处错误，全部列在 §8.2。
> 一个坦白：本篇三处纠错（GG 02 的 29 维分解、GG 07 的 DDIM 勘误、我们自己笔记里的"冻结 CLIP"）都是这次逐文件读代码才发现的——**baseline 论文不读到代码级，等于没读**。

---

## 1. 这篇论文解决什么问题？

GG 07 的两篇主角各留了一个结构性缺口：

- **DressCode 的自回归路线撞墙了**：它的量化 token 序列在 2021 数据集上 ~1500 封顶，但复杂版型（GarmentCodeData 一代）需要 **over 30k token**，超出现有 GPU 显存**无法训练**——SewingLDM v2 原文（Sec 4.3）：
  > "Dresscode, which requires over 30k tokens for complex patterns and exceeds current GPU limitations, unabling to train on the GarmentCode dataset, was evaluated only for failure rate (11.4%) using its open-source code and checkpoint."

  即 DressCode 只能拿开源 checkpoint 评一个 simulation failure rate（11.4%），其余指标全部缺席；
- **GarmentDiffusion 快但能力封顶**：无身体条件（版型不知道给谁穿）、无草图条件、仓库无代码无权重，且在 GarmentCodeData 上 stitch F1 输给 AR（0.553 vs 0.589，GG 07 §5.4）。

SewingLDM（**Multimodal Latent Diffusion Model for Complex Sewing Pattern Generation**，Shengqi Liu et al.，上海交大 MoE AI Lab + Xueshen AI + 网易伏羲，ICCV 2025 pp.17640–17650）的答案是一套组合拳：

1. **表示升级**：29 维边向量（GG 02 §3.3）——比 GarmentDiffusion 的 13 维多出三次曲线、圆弧、panel 位姿与附着约束，直接对齐 GarmentCode DSL 的表达力；
2. **潜空间扩散**：Stable Diffusion 范式（[VG 01](../../video_generation/VG_01_diffusion_basics/) §latent diffusion）移植到版型域——先把 1443 token 压到 256×6 的**有限标量量化潜空间**，再在潜空间跑 DiT；
3. **多模态条件**：文本（T5-XXL）+ **人体体型**（15 维 PCA）+ **服装草图**（前/背二值边缘图）三模态注入——版型生成第一次"知道给谁穿"。

主结果（GarmentCodeData v2）：clothes-to-body distance **2.20cm**（DressCode 3.69 / SewFormer 5.45 / RichDreamer 6.19 / Wonder3D 6.54），用户研究 4.60/5（次名 DressCode 3.56），生成失败率 0。

本篇按数据流走一遍全链路，每个模块配"论文怎么说 / 代码怎么做 / 数学是什么"，最后用三个数值实验回答三个"论文没说透为什么"的问题：

1. **潜空间为什么是 256×6**（消融给了 what：256×32/256×12 codebook 利用率 0% 无法生成；本篇实验 1 给 why）；
2. **同 prompt 结构多样性为什么低**（我们 P0 复现实测 20 seeds 只有 2 种 panel 数；实验 2 给机理）；
3. **潜压缩在计算上买到了什么**（GG 07 屋顶线模型的延续，实验 3）。

### 方法流程鸟瞰

一句话：**先把版型压进小潜空间，再在潜空间学扩散，条件分两步注入**。§4 开头有 ASCII 数据流图，这里先给每步的目的：

| 步骤 | 做什么 | 目的 |
|---|---|---|
| Step 0 | MLP-Mixer 自编码器把 1443 token 版型压到 256×6 有限标量量化潜码（FSQ，§4.1） | 甩掉 padding 的结构性浪费；潜码离散且复用率高——扩散先验才有重复结构可学（§5.5） |
| Step 1 | 只带文本条件（T5-XXL）在潜空间训 DiT（IDDPM 1000 步） | 先学会"版型分布本身"——主体能力在这里 |
| Step 2 | 注入体型（15 维 PCA）+ 草图条件，只微调注意力 proj 层 | 小数据条件分支不许带崩 Step 1 的先验；条件 25% 概率置零保住纯文本能力 |
| 推理 | 100 步采样 → 解码 → 反量化成版型 JSON | 端到端 6.5s/件（P0 实测），版型直接进缝合仿真 |

---

## 2. 背景最小集

| 需要 | 出处 |
|---|---|
| 29 维边向量、39×37 padding、四种版型编码谱系 | [GG 02](../GG_02_pattern_representations/) §3.3, §5 |
| GarmentCodeData（数据基座、v2 split） | [GG 05](../GG_05_garmentcode_data/) §4 |
| AR vs 并行扩散的计算结构、屋顶线模型 | [GG 07](../GG_07_text_to_pattern_ar_diffusion/) §5.1 |
| DDPM/IDDPM、latent diffusion 范式、CFG | [VG 01](../../video_generation/VG_01_diffusion_basics/) |
| FSQ vs VQ-VAE、直接量化坐标不可行 | GG 02 §6 |

---

## 3. 表示法详解：29 维边向量（代码权威版）

### 3.1 逐维分解

论文 Eq. 1 写 $E^f_{i,j} = V \oplus C \oplus C^b \oplus C^r \oplus S \oplus R_i \oplus T_i \oplus E^t \oplus E^m \oplus A \oplus M'$ 并称 29 维——**但按论文自己的分量描述加起来只有 28**（论文把 $R_i$ 写成 $\mathbb{R}^3$）。代码才是权威（`configs/ae/ae.py`）：

```python
continous_channels = (element_size + 3) + 7   # (9+3)+7 = 19
discrete_channels  = (edge_type_size + 2 + 1) # (7+2)+1 = 10
```

**19 维连续**：

| 分量 | 维 | 含义 |
|---|---|---|
| $V$ | 2 | 边位移向量（NeuralTailor 底座） |
| $C$ | 2 | 二次 Bezier 控制 $(c_x, c_y)$（相对坐标） |
| $C^b$ | 4 | 三次曲线两控制点 |
| $r$ | 1 | 圆弧半径（论文写成 $\mathbb{R}^3$"半径+转角"，代码只有 1 维连续半径，方向是离散 flag——见 §8.2） |
| $S$ | 3 | stitch tag = 该缝合两条边**3D 中点的平均值**（"版型穿在 T-pose 身体上时缝合处的 3D 位置"）——确定性几何特征，不可学习，继承 NeuralTailor |
| $R_i$ | 4 | panel 旋转，**四元数**（`scipy.as_quat`） |
| $T_i$ | 3 | panel 平移（panel 包围盒**顶边中点**的 3D 位置——与 NeuralTailor 同款稳定参考点选择） |

**10 维离散**：

| 分量 | 维 | 含义 |
|---|---|---|
| edge_type | 7 | [0:2] 圆弧方向两 flag（大弧/优角、顺/逆时针）+ [2:4] 线型两位（直 $[0,0]$/二次 $[0,1]$/圆弧 $[1,0]$/三次 $[1,1]$）+ [4:7] 附着约束 |
| stitch mask | 1 | 该边是否参与缝合 |
| reverse flag | 1 | 缝合方向反转 |
| existence flag | 1 | 边存在（padding 区为 0） |

**附着约束 6 类**（3 位编码）：lower_interface（下缘，防裙/裤滑落）、left/right_collar、strapless_top（抹胸）、left/right_armhole——直接对应 GarmentCode 的 Interface 概念（GG 05 §3.2），作用是**让生成的版型在仿真里穿得住**（无肩带上衣、宽松裤不会滑落）。这是"版型语义进边向量"的最深一步：GG 02 说的"4 维→29 维 = 几何→语义"，在 SewingLDM 这里走到了"工艺约束"层。

### 3.2 载体与 padding

版型 = 边向量矩阵 $(39 \times 37, 29)$：**39 个 panel 槽位**（`max_pattern_len`，`panel_order.json` 恰好 39 个键）× **每 panel 37 条边**（`max_panel_len`）= **1,443 token**。注意 config 键名是反直觉的（`max_pattern_len` 是 panel 数不是序列长）——GG 02 初版把方向记反了（37 panel × 39 边），GarmentDiffusion 论文也写的是 37 panel × 39 边/panel；乘积相同（1443），方向以 SewingLDM 代码为准（§8.2 勘误 8）。

这份 padding 的浪费是结构性的：$R_i, T_i$ 是 panel 级量被广播到每条边（重复 37 次）、不存在的 panel 整块填零——**1,443 个 token 位置里，有效信息密度很低**。这正是潜空间压缩的出发点（§4.1）。

### 3.3 与前代的对照

| | GarmentDiffusion (13 维) | SewingLDM (29 维) |
|---|---|---|
| 曲线类型 | 直/二次/三次/圆弧 | 同（但圆弧半径独立连续维 + 方向离散 flag） |
| panel 位姿 | 无（靠 token 序列位置隐含？） | 四元数 + 平移，显式拼进边向量 |
| stitch | tag 3 + flag 1 | tag 3 + mask 1 + reverse 1 |
| 附着约束 | 无 | 6 类 3 flag |
| 数据上限 | 1443 token | 1443 token（同） |

多出来的 16 维全在买**语义与工艺**（位姿显式化、缝合方向、附着约束）——代价是回归目标更高维、更稀疏。SewingLDM 补充材料 Sec 10 的消融：只用"origin vector"（老 4 维系）只能生成简单服装，extended 29 维才能生成 curve neckline、tail dress——**表示升级是这次能力跃迁的必要条件**，不只是工程细节。

---

## 4. 方法核心与数学推导

全链路（数据流顺序）：

```
版型 JSON ──序列化──► X ∈ (1443, 29)
   │
   ▼  MLP-Mixer AE（15 层, hidden 256）+ FSQ
z_q ∈ (256, 6), 每维 5 级 {−1,−0.5,0,0.5,1}      ← Step 0: 自编码器
   │
   ▼  DiT（24 层, d=1152, 12 heads, learned σ）+ IDDPM 1000 步训练
   │   条件: T5-XXL 文本 ‖ 15 维体型 PCA ‖ 草图 embedder 160 维
   │   注入: block 0 后统计归一化残差相加                ← Step 1/2: 扩散
   ▼  100 步采样 + 解码 + 反量化
版型 JSON（panels + edges + stitches + placement）
```

### 4.1 Step 0：MLP-Mixer 自编码器 + 有限标量量化

**架构**（论文只说 "auto-encoder"，以下全部来自代码 `fsqae.py`）：**MLP-Mixer 风格，不是 Transformer**——encoder 是 Linear(29→256) + 15 个 MixerLayer（token-mixing 与 channel-mixing 各一个 1024 隐层 MLP，hidden 256，dropout 0.2）+ LayerNorm + Linear(1443→256)（token 数压缩）+ Linear(256→6)（每 token 6 维）+ tanh。decoder 对称（Linear(256→1443) 还原 token 数 → 15 MixerLayer → Linear(256→29)）。

**量化**是 FSQ（Mentzer et al. 2023，论文引用但从未叫这个名字）的最简形式：

$$
\hat z = \frac{\mathrm{round}(n \tanh(z))}{n}, \qquad n = 2,
$$

每维取 $\{-1, -0.5, 0, 0.5, 1\}$ 共 $2n+1 = 5$ 级；6 维 → 每 token 码本 $5^6 = 15{,}625$ 码字，整个潜空间 $5^{6\times256}$——但没有码本查找、没有 EMA、没有 commitment loss，**量化器零参数**。梯度用直通估计（straight-through）：

$$
\nabla_z \mathcal{L}\big|_{\text{通过量化}} = \nabla_{\hat z} \mathcal{L},
$$

代码里就是 `z + (zq - z).detach()`——论文没提这一句，但没它训练根本不会收敛（§8.2 勘误 4）。

**训练损失**（论文 Eq. 3）：

$$
\mathcal{L}_{AE} = 5\,\mathcal{L}_{rec} + \mathcal{L}_{panel} + \mathcal{L}_{stitch} + \mathcal{L}_{BCE},
$$

代码口径：$\mathcal{L}_{rec}$ 是**全 29 维逐元素 MSE（含离散维）**；$\mathcal{L}_{panel}/\mathcal{L}_{stitch}$ 取自 NeuralTailor 的 cloth loss 族（shape/loop/rotation/translation/stitch）；$\mathcal{L}_{BCE}$ 约束三个二值 flag（edge_type/自由边/存在）。

**重建质量**（v2 Table 2，GarmentCodeData test）：

| | Panel L2 ↓ | Panel Acc ↑ | Edge Acc ↑ | Rot L2 ↓ | Transl L2 ↓ | Stitch Acc ↑ | Failure ↓ |
|---|---|---|---|---|---|---|---|
| AE（Ours） | **0.64 cm** | 99.8% | 88.5% | 0.0004 | 0.6 cm | 90.8% | **0** |

自编码器的重建是整个系统的**上限**：Panel L2 0.64cm 意味着无论扩散学得多好，输出误差的地板在毫米级——对缝纫这是够用的（对照 GG 07 实验 2 的 5mm 容差）。

### 4.2 Step 1/2：潜空间 IDDPM-DiT

**骨干**：PixArt-α 代码库改造的 DiT——**depth 24、hidden 1152、12 heads**（注意不是原版 PixArt-α XL/2 的 depth 28），输入 256 token × 6 维，输出 ε 与 learned σ（12 通道）。

**扩散目标**：IDDPM（Nichol & Dhariwal 2021）的 ε-预测 + 学习方差（LEARNED_RANGE），**linear β schedule**（$10^{-4} \to 0.02$，非 cosine——又一个论文没写、代码里才有的选择）、训练 1000 步、MSE：

$$
\mathcal{L}_{simple} = \mathbb{E}_{t, z_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2\right], \qquad z_t = \sqrt{\bar\alpha_t} z_0 + \sqrt{1-\bar\alpha_t}\,\epsilon.
$$

**推理**：默认 **IDDPM 100 步**（可选 DPM-Solver 20 步 / SA-Solver 25 步）；CFG scale 脚本默认 4.5、`generate.sh` 用 1.0。潜变量除以 scaling_factor 1.8407 归一化。

**两步训练策略**（消融见 §5.3）：

- **Step 1（2 天）**：只用文本条件训练潜扩散——让模型先学会"版型分布本身"；
- **Step 2（+10 小时）**：注入体型 + 草图条件，**只微调注意力模块的输出投影层**（config 正则 `.*attn.*proj.*`，self-attn 与 cross-attn 的 proj 都放开）——冻结其余全部参数，防止小数据条件分支带崩大模型；
- Step 2 里 sketch 或 text 条件以 **0.25 概率置零**——保持单条件（纯文本）生成能力不退化。这个细节是复现时最容易漏的：**不是数据增广，是条件 dropout**（CFG 的 null-token 家族）。

### 4.3 条件注入：统计归一化残差（不是 cross-attention 全家）

三路条件：

1. **文本**：**T5-XXL**（PixArt-Σ 的 diffusers 权重，4096 维，fp16，**冻结**，最多 120 token；训练时 10% 概率置 null 做 CFG）——**全文无 CLIP**（我们此前笔记写错，§8.2）；
2. **体型**：15 维 PCA 体重参数（GarmentCodeData 的 5000 体型库做 PCA），`Linear(15→32)`；
3. **草图**：前/背两张 512×512 二值边缘图（PiDiNet 提取），5 级 Conv+GroupNorm+ReLU 下采样到 128 通道。

三路 concat（32+128 = 160 维）→ Linear → 1152，过一层轻量 transformer（`MultiModalNet`）。**注入位置与方式**是这篇最特别的架构决定：不做 cross-attention、不做 adaLN、不做 ControlNet 双分支（论文明确讨论并否决了 ControlNet）——而是在 **DiT block 0 之后**，把条件特征做自适应池化 + 归一化到当前潜特征分布，乘 control_scale 后**残差相加**（论文 Eq. 4）：

$$
F \leftarrow F + \gamma \cdot \frac{F_{bs} - \mu_{bs}}{\sigma_{bs} + \epsilon}\,\sigma_z + \mu_z \quad (\text{代码版}),
$$

即"把条件特征的分布**搬家**到潜特征的分布上再相加"。论文 Eq. 4 的分式把 $\sigma_{bs}$ 与 $\sigma_z$ 写反了（§8.2 勘误 3）——代码为准。消融显示注入位置极其敏感：block 5/10/15/20 注入会逐渐丢失袖子/腰头等部件直到不可穿戴，**block 0 最优**。

这个设计的直觉：潜 token（FSQ 后是离散格点）的分布与连续条件特征差异巨大，直接 cross-attention 会让早期训练产生分布错配；归一化残差是"最温和的融合"。它省掉的 cross-attention 也是 GG 07 实验 1(c) 里 DiT FLOPs 的大头之一——注入越轻，延迟越稳。

---

## 5. 实验与关键数字

### 5.1 论文主表

**Table 1**（生成质量 + 效率 + 用户研究，30 人 × 10 prompt、1–5 分）：

| 方法 | Runtime ↓ | Clothes-to-body ↓ | Users ↑ |
|---|---|---|---|
| RichDreamer | ~4 min | 6.19 cm | 1.89 |
| Wonder3D | ~4 h | 6.54 cm | 1.88 |
| SewFormer | ~3 min | 5.45 cm | 2.10 |
| DressCode | ~3 min | 3.69 cm | 3.56 |
| **SewingLDM** | ~3 min | **2.20 cm** | **4.60** |

（runtime 含仿真渲染，各方法口径相同。）注意 **Panel L2 等版型几何指标不在主表**——主表是"渲染出的 3D 服装质量"（clothes-to-body = 服装每点到人体最小距离的均值）+ 用户研究；版型几何指标在 Table 2 是**重建口径**（AE/SewingLDM 3.13 Panel L2），与 SewFormer*（GarmentCodeData 微调版 12.3）对比。GG 05 §5.4 说过这个评测生态的割裂——SewingLDM 也没有跳出它。

### 5.2 消融一：潜空间压缩（本篇实验 1 的主角）

Table 3（v1/v2 同）：

| | 不压缩 | 256×32 | 256×12 | 256×8 | **256×6** | 256×4 |
|---|---|---|---|---|---|---|
| 重建 | ✓ | ✓ | ✓ | ✓ | ✓ | **×** |
| 生成 | × | **×** | **×** | ✓ | ✓ | − |
| Clothes-to-body | − | − | − | 2.87 | **2.20** | − |
| Codebook 利用率 | − | **0%** | **0%** | 91% | **100%** | − |

三个坑：**大潜空间（256×32/12）生成失败——codebook 利用率 0%**（自编码器重建没问题！）；**过度压缩（256×4）连重建都崩**；256×6 是 Goldilocks。论文给了 what，why 要自己拆——实验 1（§5.5）给出机理：这不是调参运气，是计数约束。

### 5.3 消融二：定性消融（论文无定量表）

- **注意力微调范围**（Step 2）：只放开 self-attn → 领口/袖子错；只放开 cross-attn → 圆弧领口差、草图对齐差；全放开最优；
- **注入位置**：block 5/10/15/20 → 部件逐步丢失至不可穿戴；block 0 最优；
- **训练策略**：一步到位（sketch 条件从零训练）→ 失败（丢 midi 铅笔裙）；两步成功；
- **表示**：origin vector（老 4 维系）→ 只能生成简单服装；extended 29 维 → curve neckline/tail dress。

**论文没有 guidance scale、采样步数的定量消融**——这两项空缺正是我们实验 2、GG 07 实验 3 补的位置。

### 5.4 我们的复现经验（P0，8×A6000 服务器）

Work 3 的 P0 阶段在 `ssh lab` 上复现（Apache-2.0，代码+权重+数据三全——这是选它做 baseline 而非 GarmentDiffusion 的直接原因）：

- **推理 6.5s/件**（100 步 IDDPM，A6000，含 T5 编码）；生成 spec → NvidiaWarp 缝合仿真闭环通过（468 帧、0 body collision、0 self-intersection）；
- **序列化 bug**：生成的 spec `panel_order` 含 None padding（39 槽定长的副作用），仿真前需 `[x for x in order if x]` 过滤——论文/README 均未提，是我们踩的第一个坑；
- **同 prompt 结构多样性低**："long sleeve shirt loose fit" 20 seeds → n_panels 分布 **{2: 18, 4: 2}**（90% 样本只有 2 个 panel）。valid_json / 仿真 / 碰撞全过——**质量没问题，多样性有问题**。这个现象的机理是实验 2 的内容；
- **环境坑**：依赖锁定（torch 2.0.0+cu118 / transformers 4.36.1 / diffusers 0.25.0），mmcv 需要 setuptools<81，warp 得用已编译的 fork；官方 `train.sh` 调用 `train_fisrt_step.py`（拼写笔误），开箱即跑会报错（§8.2 勘误 6）。

### 5.5 实验 1：FSQ 潜空间的 Goldilocks——大潜空间两侧都亏

**设置**：玩具版型世界（30 类服装 × 每类 24 维风格子空间 → 16 token × 29 维观测，训练样本 19,683 与 DressCodeData 同量级），线性自编码器 + 每 token FSQ（5 级、直通梯度、L-BFGS 训练），扫每 token 维数 $k$：

| $k$ | 测试 RMS ↓ | 每 token 平均码字数 | 码字平均复用 | 测试整版码字在训练集复现率 | NN Hamming/$k$ |
|---|---|---|---|---|---|
| 4 | 0.0551 | 24 | 803× | 98.9% | 0.000 |
| **6** | **0.0481** | 41 | 477× | **95.7%** | 0.000 |
| 8 | 0.0467 | 57 | 343× | 86.7% | 0.003 |
| 12 | 0.0473 | 107 | 183× | 58.0% | 0.003 |
| 32 | 0.0519 | 1,114 | 18× | **0.0%** | 0.016 |

两个读数，对应消融表的两行失败：

1. **重建不是单调改善**：$k=4\to8$ 下降后饱和甚至回升（$k=32$ 已排除欠拟合，maxiter 600 收敛验证）——FSQ 每维只有 2.3 bits，加一维 = 加一个 $\pm0.25$ 均匀量化噪声通道；重建侧大 $k$ **没有**收益。256×4 的重建失败（消融表第一行坑）就是这个效应加上容量不足的极限；
2. **码字结构单调崩塌**：重建压力总想把样本摊开到不同码字（区分度），容量小时量化强迫共享（$k=6$：每码字复用 477 次、95.7% 测试整版码字在训练中出现过——**离散先验有大量重复证据可学**）；容量大时样本被摊成孤点（$k=32$：复用 18 次、整版码字零复现）——扩散先验面对 19,683 个互不重复的码组合，只能 memorize 不能泛化。**256×32/12 的 codebook 0% 利用率**（消融表第二行坑）是同一现象在真实模型上的样子：不是码字没人用，是**每个码字几乎没有共现结构，离散扩散学不出下一码字的分布**。

**Goldilocks 的机理**：重建饱和点（$k\approx6\text{–}8$）∩ 复用结构尚存 → $k=6$。计数上：每 token $5^6 = 15{,}625$ 码字 ≈ 训练样本数 19,683（DressCodeData 口径；GarmentCodeData 上乘以类别结构后同量级）——**码本大小匹配数据的有效自由度**，这与 GG 02 §6 "瓶颈不是越小越好，是要匹配数据的有效自由度"的论断在玩具上定量成立。

### 5.6 实验 2：CFG guidance 压结构多样性——P0 实测 {2:18, 4:2} 的机理

**设置**：4 个"结构模式"（2D toy，模式 = panel 布局类），条件模型 $p_c$ 与无条件模型 $p_u$ 对各模式的权重不同（条件模型只把与 prompt 不符的模式 3 压到 7%——**它懂一点 prompt 但不完美**）。CFG 采样等效于从

$$
p_{cfg} \propto \frac{p_c^{\,1+w}}{p_u^{\,w}}
$$

采样（组合 score $(1{+}w)\nabla\log p_c - w\nabla\log p_u$ 的精确恒等式）。解析预测 + 全 1000 步 DDPM（解析 score）采样双重验证：

| $w$ | 解析 $p_{cfg}$ 模式权重 | DDPM 采样权重 | 非法模式质量 | 合法模式熵 |
|---|---|---|---|---|
| 0 | [.55 .22 .070 .16] | [.61 .21 .071 .12] | 0.070 | 0.955 |
| 1 | [.74 .12 .014 .13] | [.79 .10 .010 .10] | 0.014 | 0.735 |
| **2** | [.86 .055 .0026 .084] | [.89 .038 .0005 .073] | 0.0026 | 0.500 |
| 3 | [.92 .024 .0004 .053] | [.94 .015 .0000 .043] | 0.0004 | 0.318 |
| 5 | [.98 .004 .0000 .019] | [.98 .001 .0000 .016] | 0.0000 | 0.120 |

机理：**$w$ 是似然比的指数放大器**——每个模式的质量按 $\pi_c \cdot (\pi_c/\pi_u)^w$ 几何衰减。压制非法模式（ratio 0.28）与保留少数合法模式（ratio 0.73）的分离度 = 两者 log-ratio 之比：$w$ 调到非法模式 <1% 时（$w\ge2$），少数合法模式已被压到 5% 以下、合法模式熵从 0.96 崩到 0.50。

**对接 P0 实测**：{2:18, 4:2} 即少数结构存活率 10%，正对应表中 $w\approx1\text{–}2$ 的模式 2 质量量级（SewingLDM 脚本默认 cfg 4.5——但那是文本对齐用途，结构与文本的相关弱，等效分离度低）。**结论：guidance 调不出"既压制非法又保留少数"**——旋钮只有一档，压非法的同时必然压少数合法。要多样性只能提高分离度本身：让 $p_c$ 更懂 prompt（哪些结构都合法），这正是 Work 3 视频先验蒸馏的切入点——视频观测天然携带结构信息（袖子数量、裙撑形状直接可见），把 $\pi_c/\pi_u$ 在"合法结构"内部拉平、在"非法结构"上拉开。

（边界说明：本实验是机理玩具——真实模型的条件分布、CFG 的 null-drop 训练细节都会偏移数字，但"单旋钮几何衰减"的结构不变。）

### 5.7 实验 3：潜压缩的计算预算（GG 07 屋顶线延续）

代入两模型真实配置（DiT FLOPs $\approx L\cdot12nd^2 + L^2\cdot2d$ 每 step，A10 100 TFLOP/s）：

| 模型 | $L$ | $d$ | $n$ | $T$ | 总 FLOPs | DiT 参数 | A10 理想延迟 |
|---|---|---|---|---|---|---|---|
| GarmentDiffusion（观测空间） | 1443 | 768 | 12 | 50 | 6.3T | 85M | 0.063s |
| **SewingLDM（潜空间）** | 256 | 1152 | 24 | 100 | 9.8T | 382M | 0.098s |
| 假想：SewingLDM 不压缩 | 1443 | 1152 | 24 | 100 | 55.6T | 382M | 0.556s |

潜压缩（1443→256 token）让 SewingLDM 用 **[更大 DiT（24 层 × 1152）× 2 倍步数]** 的总 FLOPs 仍与 GarmentDiffusion 同量级——$L^2$ 注意力项从 $1443^2$ 降 32× 是主因；同等 DiT 直接吃 1443 token 是 8× 计算量。P0 实测 6.5s/件 对上：扩散本体 ~1s 量级，其余是 T5 编码 + 序列化/反量化开销。**GG 07 结论的补全**：GarmentDiffusion 已把延迟打到 <1s，版型域的延迟竞争已经结束；SewingLDM 的潜压缩不是为快，是把省下的算力花在更大模型 + 复杂版型能力 + 多模态条件上。

---

## 6. 论文间关系

```
GG 07: DressCode（AR, 30k token 撞墙）        GarmentDiffusion（并行, 无代码）
              │                                       │
              │ 30k/11.4% 的失败记录（v2 Sec 4.3）      │ 同任务、观测空间扩散、无身体条件
              ▼                                       ▼
SewingLDM (ICCV'25)：29 维表示 + FSQ 潜空间 + 多模态条件 ★ Work 3 P0
              │
              │ 表示/数据同源（GarmentCodeData v2, 1443 token 上限）
              ▼
concurrent works（补充材料 Sec 8, 定性对比）: ChatGarment / AIpparel / Design2GarmentCode
```

三个值得单独记的点：

1. **SewingLDM 全文没有引用 GarmentDiffusion**——两篇是同期工作（arXiv 2412 与 2504），互不引用；concurrent works 列的是 ChatGarment（GPT-4V 版型编辑）、AIpparel、Design2GarmentCode。所以"GarmentDiffusion vs SewingLDM"的任何直接数字对比都不存在，GG 07 §5.4 的对比是 GarmentDiffusion 论文自己 vs SewingGPT——别把两篇的表拼在一起当同口径；
2. **DressCode 的 11.4% 是 SewingLDM v2 测的**（用 DressCode 开源代码 + checkpoint 在 GarmentCodeData 上评 simulation failure rate）——不是 DressCode 自报，也不是 GarmentDiffusion 测的。三个口径链（18,135 GD 测 / 30k+11.4% SewingLDM 测）在 GG 07 勘误表里的"两文不一致"至此完全理清：**两个数字都是真的，测的东西不同**（GD 测 token 数、SewingLDM 测不能训练+失败率）；
3. **谱系定位**：SewingLDM 是"向量序列 → 潜空间"断层的完成者（GG 02 §7 谱系第三断层）——NeuralTailor（观测序列）→ DressCode（离散 token AR）→ GarmentDiffusion（连续 token 扩散）→ SewingLDM（压缩潜空间扩散）。每一步都在回应上一步的瓶颈：AR 慢（GD 解）→ 观测空间扩散仍受 1443 token 拓扑上限约束、无条件身体（SewingLDM 解）→ SewingLDM 的开放问题：FSQ 潜空间的表达上限（拉链/口袋失败）、guidance 多样性（§5.6）——GG 09 的三条新路线（栅格/隐式场/LLM 程序合成）正是对着这些剩余瓶颈。

---

## 7. 与 Work 3 的连接（最重的一节）

Work 3 的目标：**视频基础模型先验 → 蒸馏 → 缝纫版型服装生成**。P0 选 SewingLDM 做 baseline 后，本篇精读直接变成 Work 3 的设计输入：

1. **蒸馏的接口已经存在且很干净**：条件注入是"160 维特征 → 一层轻量 transformer → block 0 残差相加"（§4.3）——**这就是一个天然的蒸馏注入口**。Work 3 的视频先验（Wan2.1 中层特征的 3D 对应/遮挡补全信号，P2 Probe 已确认 block8/14/17/21/29 携带）可以走同一条路：视频特征 → 轻量投影 → 残差注入。不需要动 DiT 主体（Step 2 的"只微调 attn proj"策略原样可用）；
2. **结构多样性的机理 → 蒸馏的收益主张**：实验 2 证明 guidance 单旋钮压非法必压少数——Work 3 要主张的"视频先验带来结构多样性"有了机理论证：视频观测把 $\pi_c/\pi_u$ 的分离度本身变大（合法结构内部拉平），这是 guidance 调不出来、只能靠更好的 $p_c$ 得到的东西。P0 的 {2:18, 4:2} 是对照基线，P3 蒸馏后的 n_panels 分布是第一个要测的指标；
3. **FSQ 潜空间对蒸馏是约束也是机会**：约束——潜码离散（5 级），视频先验的连续几何信号要过量化瓶颈，梯度只能直通；机会——实验 1 的复用结构视角给了蒸馏的优化目标：视频条件应该提高码字序列的**条件可预测性**（同一视频 → 同一码字序列的映射更确定），而不是提高码字多样性。评测上可以直接数：加视频条件前后，同 prompt 的整版码字复现率/码字转移熵怎么变；
4. **复现已踩平的坑**（P0 记录，全部已解）：`panel_order` None padding 过滤、依赖锁定、`train_fisrt_step.py` 拼写、mmcv/setuptools 版本、warp fork 编译——这些是复现成本，也是"为什么 P0 选它"的隐性答案（GarmentDiffusion 无代码，这些坑会以不可见的方式存在）；
5. **评测的坑要提前知道**：主表口径是 clothes-to-body（渲染 3D 质量），不是版型几何；GG 07 实验 2 的闭合/缝合一致性指标两代论文都没报——Work 3 的评测管线应自建这一层（P0 已加 valid_json/sim/碰撞全过的基础版）。

---

## 8. 局限与勘误

### 8.1 局限

**论文自列**（Sec 5 + Fig 8）：拉链、口袋等现代设计失败；婚纱等复杂草图对齐困难；复杂服装的复杂草图偶尔失配。未来工作：更全面的日常服装表示、更多条件模态。

**方法层面**（本篇实验的推论）：

- **FSQ 表达上限**：256×6 = 每 token 13 bits 的离散瓶颈 + 0.64cm 重建地板——拉链/口袋的失败首先撞的是**表示**（29 维里没有"口袋"的语义位）而非扩散，这解释了为什么表示消融（Sec 10）比采样消融更决定能力边界。GG 09 的 LLM 程序合成路线（GarmentoPIA）正是绕开这个瓶颈的方案；
- **guidance 多样性**（实验 2）：结构多样性被 CFG 的几何衰减机制系统性压低——这不是 SewingLDM 独有，是所有"CFG + 结构多模态"组合的通病（GG 09 Latent Flow Matching 的无 CFG 采样是一个可能的解法方向）；
- **无定量步数/guidance 消融**：100 步 IDDPM 与 cfg 4.5 都是工程默认值，论文没给敏感性——复现者要自己做（P0 已开始）。

### 8.2 勘误表（论文自身 5 处错误 + 我们笔记 3 处修正）

**论文自身的错误**（逐文件读代码才能发现，以代码为准）：

| # | 论文写法 | 事实（代码） | 依据 |
|---|---|---|---|
| 1 | Eq. 1 各分量加总 = 29 维 | 按论文自己的分量描述加总只有 **28**（$R_i$ 写成 $\mathbb{R}^3$）；代码是**四元数 4 维**，29 的缺口在此 | `pattern_converter.py`, `scipy_rot.as_quat()` |
| 2 | $C^r \in \mathbb{R}^3$（半径+转角，连续） | 代码是 **1 维连续半径** + 方向是 edge_type 里的 **2 维离散 flag** | `_edge_as_vector` |
| 3 | Eq. 4 注入公式 $\sigma_{bs}$ 与 $\sigma_z$ 写反 | 代码为 $(F_{bs}-\mu_{bs})\cdot\sigma_z/(\sigma_{bs}+\epsilon)+\mu_z$ | `SewingLDM.forward` |
| 4 | 量化只引 [41]、未提直通梯度 | 代码 `z + (zq - z).detach()`——没它不收敛 | `fsqae.py` |
| 5 | Sec 6 线型编码表 cubic = $[0,0]$（与 straight 撞码）、circle = $[1,1]$ | 代码：circle = $[1,0]$、cubic = $[1,1]$ | `_edge_dict` |
| 6 | （仓库）`train.sh` 调用 `train_fisrt_step.py` | 实际文件名 `train_first_step.py`，开箱即跑报错 | 仓库单 commit |
| 7 | 补充材料 panel 顺序表 "29" 键重复 | `panel_order.json` 实为 39 项 | `configs/data_configs/` |

**我们笔记的修正**（本篇写作时发现，前帖已同步改）：

| 误记 | 事实 |
|---|---|
| "SewingLDM 冻结 CLIP" | 文本塔是 **T5-XXL**（PixArt-Σ diffusers 权重），全文无 CLIP |
| GG 02 初版"config 9+7+4+3=29" | 真实算式 $(9{+}3{+}7)+(7{+}2{+}1) = 19+10 = 29$（漏了 stitch_tags +3 与三个 mask +3） |
| GG 02/GG 07 "37 panel × 39 边" | **39 panel 槽 × 37 边/panel**（`max_pattern_len`=39 是 panel 数）；GarmentDiffusion 论文的方向相反，乘积同 |

**另注**：论文正文说 GarmentCodeData "120,000 sewing patterns"，代码 `dataset_v2_info.json` 实际 125,865 件（split 102,660/6,270/6,265）——口径差未解释（推测 v1/v2 计数），引用时以 v2 代码口径为准。

---

## 9. 习题

**题 1**：实验 1 的玩具里，$k=6$ 时每 token 平均只用 41 个码字（$5^6=15{,}625$ 个里）——为什么码本利用率这么低反而是好事？把"利用率"与"复用率"区分为两个指标，分别说明哪个对应重建、哪个对应可学习性。
*提示*：利用率 = 用过的码字/总码字（SewingLDM 消融表报的是它）；复用 = 每码字平均出现次数。256×32 的 0% 利用率不是"码字没用"而是"分布没有重复结构"。低利用率 + 高复用 = 模型找到了数据的低维流形——这恰是要蒸馏的东西。

**题 2**：推导 $p_{cfg} \propto p_c^{1+w}/p_u^w$ 时用到了 $\nabla\log p^a = a\nabla\log p$。(a) 证明组合 score $(1{+}w)\nabla\log p_c - w\nabla\log p_u$ 确实是某个分布的 score（归一化常数不影响）；(b) 若 $p_c$ 与 $p_u$ 完全相同（条件没学到任何东西），$p_{cfg}$ 是什么？此时采样分布与 $w$ 无关吗？
*提示*：(b) $p_{cfg} \propto p_c$——CFG 对"条件没学到"是盲的：用户看到输出随 $w$ 变化（对比度变化）会误以为条件在起作用。这是用 CFG 输出判断条件模态有效性的经典陷阱，Work 3 评测要用"同 prompt 多样性"这类条件外指标。

**题 3**：SewingLDM 的 Step 2 只微调注意力 proj 层（`.*attn.*proj.*`）。设 DiT 参数 382M，估算 proj 层占比（$d=1152$，每 block 4 个 proj：qkv 合并算 1 个 $d \to 3d$、out proj $d\to d$、cross-attn 两个）；再从优化角度论证为什么"冻结构干 + 微调投影"在小数据条件注入里比全参数微调稳。
*提示*：每 block $\approx 3d^2 + d^2 + d\cdot 4096\cdot2$（T5 cross）；占比 ~10% 量级。全参数微调会把 Step 1 学到的版型分布先验冲掉（灾难性遗忘），proj 层是"注意力重定向"——只改看哪、不改怎么算。

**题 4**：实验 3 里 SewingLDM 潜空间（256 token）与 GarmentDiffusion 观测空间（1443 token）的 $L^2$ 注意力项差 32×。若把 SewingLDM 的 100 步换成 DPM-Solver 20 步（仓库支持），总延迟估计降多少？结合 GG 07 实验 3 的"score 误差对确定性采样器更致命"结论，讨论换 solver 的风险与验证步骤（P0 有什么现成指标可以做 sanity check？）。
*提示*：T5 编码等固定开销不变，扩散本体 ~5×；验证：n_panels 分布、闭合残差（GG 07 实验 2 的指标）、clothes-to-body 三个口径对比 20 vs 100 步——若 n_panels 分布显著收窄，先怀疑 score-solver 失配而非"更快了"。

**题 5**（思考）：Work 3 要把视频先验注入 SewingLDM。候选方案 A：走 §4.3 的条件注入口（视频特征 concat 进 160 维）；方案 B：直接在潜码序列上加视频对齐损失（要求同视频帧间码字转移可预测）。用实验 1/2 的结论分别论证 A、B 各自动的是哪个量（分离度？复用结构？码字转移熵？），并预测哪个先在 P3 见效。
*提示*：A 动 $\pi_c/\pi_u$ 分离度（实验 2 的量）——直接攻结构多样性，但受 0.25 条件置零与轻量注入层的容量限制；B 动码字条件熵（实验 1 的复用结构的对偶）——更根本但要过 FSQ 直通梯度，训练不稳风险高。P3 最小蒸馏应该先 A 后 B：A 不动训练目标、只加一路条件，风险与 Step 2 同级。

---

*下一篇：[GG 09：版型生成的表示革新](../GG_09_representation_shifts/)——GarmentImage（栅格化，unseen 组合 IoU 0.81 vs 序列 0.31/0.43 的局部规则机理）、Latent Flow Matching（隐式场 + 可微网格化 + 无 CFG 的标准 FM）、GarmentoPIA（LLM agent 把打版文献翻译成尺规 DSL 程序）：对着 SewingLDM 剩余三个结构性瓶颈的三条路线。*
