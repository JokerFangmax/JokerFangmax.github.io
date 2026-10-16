---
title: "Forcing: 11. Deep Forcing"
date: 2026-10-14
math: true
tags:
  - "Forcing"
  - "Video Generation"
  - "Deep Forcing"
categories:
  - Forcing
series: forcing
weight: 11
description: "Deep Forcing（KAIST AI，arXiv:2512.05081v1）精读：零微调长视频生成。Deep Sink 把滑窗约一半（S=10/共 21 帧）留给持久 sink token 并重对齐时间 RoPE 相位；Participative Compression 按近期注意力参与度对中间 token 做 Top-C 筛选（预算 N=16、保护 R=4，仅在窗口满且 t=1000 时执行一次）。5 秒训练外推 12× 到 60 秒以上：30 秒 VBench-Long Dynamic Degree 57.56（Self Forcing 仅 36.62，Table 1/4），24 人 2AFC 用户研究对 CausVid 整体质量偏好 100%（Table 2）。"
paper_id: "2512.05081v1"
---

# Forcing 11：Deep Forcing——零训练的 KV 缓存手术

> **Forcing 系列 · 第 11 篇 · 第三部分「长视频与上下文」收尾（历史记忆维度）**
> 上一篇：[Self-Forcing++（10）](../forcing_10/) ｜ 下一篇：[Context Forcing（12）](../forcing_12/)
> 阅读前提：扩散模型基础见 [VG 01](../video_generation/VG_01_diffusion_basics/)。正文只留主线，长推导、超参标本与消融全表收进折叠块（点击展开）。

## 论文信息

- **原题**：Deep Forcing: Training-Free Long Video Generation with Deep Sink and Participative Compression
- **阅读版本**：arXiv:2512.05081v1，对应本地收集的 PDF 版本。
- **原文**：[arXiv 摘要页](https://arxiv.org/abs/2512.05081v1) · [PDF](https://arxiv.org/pdf/2512.05081v1)

---

## 1. 这篇论文解决什么问题

Self Forcing 一类的自回归视频扩散模型靠滚动 KV cache 逐帧外推，长程上有两大病症：**保真退化**（颜色过饱和、纹理模糊，即误差累积）与**运动停滞**。直觉的解法是从 LLM 搬 attention sink——保留最初几个 token 稳定注意力。但论文开篇就给出反例：把 StreamingLLM 式的浅 sink 直接用于视频扩散，会导致保真退化与运动停滞（LongLive 自己的生成结果里也有）。

Deep Forcing 的立场是：**预训练 Self Forcing 本来就自带 attention sink 行为，问题不在「要不要 sink」，而在推理时怎么管理缓存**。于是整篇论文是一道零微调的推理时手术，两个机制：

| 组件 | 攻击的病症 | 一句话机制 |
|------|-----------|-----------|
| **Deep Sink** 深沉降 | 全局上下文漂移 | sink 不再是前 3 帧，而是滑窗的 40–60%（本文 S=10/共 21 帧），并重对齐时间 RoPE 相位 |
| **Participative Compression** 参与式压缩 | 超长外推后的退化（degeneration） | 中间区 token 按「近期注意力的参与度」Top-C 筛选，冗余的直接踢出 |

结果口径：5 秒训练的模型外推 **12×**（60 秒以上），在 VBench-Long 上与显式为长视频训练的 Rolling Forcing / LongLive 打平甚至更好，Dynamic Degree 大幅领先，全程保持实时吞吐（15.75 FPS）。

---

## 2. 最小背景

**自回归视频扩散的链式分解**。$N$ 帧视频的联合分布按因果链拆开：

$$
p(x_{1:N}) = \prod_{i=1}^{N} p(x_i \mid x_{<i})
\tag{Eq. 1}
$$

每个条件分布 $p(x_i \mid x_{<i})$ 由扩散模型参数化。

**Self Forcing 的滚动 KV cache**。每帧经 VAE 编码为 latent token 后进入定长 cache；cache 满后按 **FIFO** 驱逐最旧帧——不看重要性。去噪用蒸馏过的 4 步流程，噪声调度 $\{t_0=0,\, t_1=250,\, t_2=500,\, t_3=750,\, t_4=1000\}$（共 5 个噪声级）。第 $i$ 帧在第 $j$ 步的迭代：

$$
x^i_{t_{j-1}} = \Psi\big(G_\theta(x^i_{t_j},\, t_j,\, \mathrm{KV}),\, t_{j-1}\big)
\tag{Eq. 2}
$$

其中 $x^i_{t_4} \sim \mathcal{N}(0, I)$，KV 是已生成干净帧的缓存，$\Psi$ 是前向加噪算子。

**三条既有路线与死穴**：

| 路线 | 代表 | 死穴 |
|------|------|------|
| 训练时加噪历史 | Diffusion Forcing、History-guided | 人工噪声 ≠ 生成噪声，长程漂移仍在 |
| 自生成历史训练 | Self Forcing | 依赖历史 KV cache，误差照样累积 |
| LLM attention sink | StreamingLLM → LongLive / Rolling Forcing（各训 3 帧 sink） | 需要蒸馏/微调；浅 sink 对视频不够 |

---

## 3. 方法总览

整个系统挂在预训练 chunk-wise Self Forcing 上，**不含任何可学参数**。推理时每个 chunk 的处理分四步：

1. **Deep Sink（模块一）**：把滑窗（本文最大 21 帧）的前 $S=10$ 帧标为持久 sink，永不驱逐——约一半窗口。依据是第 4.1 节的注意力分析：中间帧也拿到可观权重，锚要「深」不要「浅」。
2. **Temporal RoPE Adjustment（模块二）**：sink 的时间戳停留在 $t\approx 1$，当前帧已到 $t\approx 200$，3D RoPE 的时间维直接暴露这个鸿沟（闪烁/退化/回卷）。做法是把 sink 的 K 的时间相位**旋转**一个偏移，使其在相对位置上贴近当前时间线，空间维不动。
3. **Participative Compression（模块三）**：窗口满 21 帧后，把中间区（sink 与 recent 之间的候选）按「近期 query 的注意力参与度」打分，只保留 Top-C 个 token，cache 压回 $N=16$ 帧；最近 $R=4$ 帧免压缩。
4. **Temporal RoPE Unification（模块四）**：压缩后的三段（sink ∥ top ∥ recent）时间上不连续，对 Top-C 段再做一次相位旋转，让整个 cache 在时间索引上连续。

与基线的对照一句话：Self Forcing 是「满了就丢最旧的」（FIFO，不看内容）；Deep Forcing 是「前段钉死、中段按参与度精选、尾段全保，且每段都对齐时间相位」（selective eviction）。

---

## 4. 核心原理完整推导

### 4.1 为什么 sink 要「深」：注意力分布的证据

第一步是诊断。论文观察预训练 Self Forcing 生成最后一个 chunk（帧 19–21）时，query 平均注意力在早期 KV 上的分布（Fig. 4，两个代表性头 L1H1 / L5H10）：**大部分注意力头不只给最初 token 大权重，序列中段的 token 也拿到可比的注意力**。附录 H（Fig. 16）进一步显示这个「深注意力」模式跨层跨头普遍存在——每帧 1,560 个 latent token，注意力铺满整个上下文窗口。

由此假设：**直到序列中段的更多 token 都是高质量生成的关键锚**。消融（Fig. 5，50 秒视频、MovieGen 前 21 个 prompt）验证：sink 帧数增大时，Overall Consistency（CLIP）单调升、Aesthetic Drift（LAION 美学预测器，取视频前后各 5 秒美学分差的绝对值）单调降。

<details>
<summary><b>展开：sink 深度的定性标本（附录 B，60 秒生成）</b></summary>

- Sink 0：怪物纹理 frame 230 明显退化、颜色漂移，frame 690 完全崩溃；Sink 4 / 9 逐步减轻但细节仍可见；Sink 14（超过 10 帧）保真退化大幅消除；Sink 18 过大反而**重复生成**（早期帧被过度保留，模型开始复读）。

结论是有效 sink 区间 **10–15 帧（滑窗的 40–60%）**。「深而不过深」。

</details>

### 4.2 Deep Sink 的时间 RoPE 重对齐

深 sink 引出新问题：sink 的时间戳停留在视频开头，当前帧的时间戳已经很大。3D RoPE（时间/高/宽分维编码）下，sink 与当前 token 的**时间相对距离**随生成无界增长，最终超出训练插值范围。症状有三：闪烁（flickering）、保真退化、**回卷**（roll-back——已沉降的历史帧被「重新生成」一遍）。

**推导**。把当前滑窗的 K、V 按位置切成 sink 与 tail 两段（$\|$ 表示拼接）：

$$
K = K_{\mathrm{sink}} \,\|\, K_{\mathrm{tail}}
\tag{Eq. 3}
$$

$$
V = V_{\mathrm{sink}} \,\|\, V_{\mathrm{tail}}
\tag{Eq. 4}
$$

记 $s_{\mathrm{tail}}$ 为 tail 的首帧索引、$s_{\mathrm{sink}}$ 为 sink 的末帧索引，定义两段的时间间隙：

$$
\Delta_{\mathrm{sink}} = s_{\mathrm{tail}} - s_{\mathrm{sink}}
\tag{Eq. 5}
$$

对 $K_{\mathrm{sink}}$ 的**时间分量**施加复数相位旋转（$\omega_t$ 是 RoPE 的时间频率向量，$\odot$ 为逐元素乘）：

$$
K^{(\mathrm{time})}_{\mathrm{sink}} \leftarrow K^{(\mathrm{time})}_{\mathrm{sink}} \odot \exp\big(i\,\omega_t\, \Delta_{\mathrm{sink}}\big)
\tag{Eq. 6}
$$

**直觉**：RoPE 的相对位置信息编码在 Q、K 的**相位差**里。对已旋转的 K 整体再乘 $\exp(i\omega_t\Delta)$，等价于把它的时间坐标平移 $\Delta$——sink 在模型眼里不再是「200 帧之前」，而贴到了 tail 的时间线上。这是一次只动时间维、不动空间维的等距变换（保范数、只转角度）。

<details>
<summary><b>展开：为什么必须是「旋转」而不是「重新编码」——三种 sink 的对照（附录 A）</b></summary>

KV cache 里存的是**已经应用过 RoPE 的 K**，推理时没有原始 token 可供重新编码，所以唯一无损的可逆操作就是复数域相位旋转。三种 sink 策略的 training-free 对照（Fig. 9，60 秒生成）：

| 策略 | 做法 | 60 秒末的表现 |
|------|------|--------------|
| LongLive 式 | 前 3 帧 sink，**不调 RoPE** | frame 800 保真退化 → 801 闪烁 → 802 **回卷**（生成回退到早期沉降帧） |
| Rolling Forcing 式 | 3 帧 sink + dynamic RoPE（滚动时对整个 cache 重放位置编码） | frame 800–801 仍严重保真退化 |
| Deep Sink（本文） | 10 帧 sink + 时间相位重对齐 | 显著最好 |

两个教训：sink 要深（3 帧不够）、RoPE 要调（不调必回卷）。而且这些失败模式在**专门训练过 sink 的 LongLive 自己身上也会出现**——训练并不能豁免位置编码的外推失真。

</details>

### 4.3 Participative Compression：按参与度筛 token

Deep Sink 止住了漂移，但 12× 外推后另一个病冒头：**退化（degeneration）**。机理（LLM 长文本生成中已有记录）：超出训练长度后，无差别保留 token 让注意力被摊薄在相关与无关上下文上，噪声复利累积。而视频扩散的注意力本来就高度集中在少数语义关键 token 上（引 Sparse VideoGen2 的分析）——**剪掉低参与度 token 既省算力又不伤质量**。这就是把 H2O / SnapKV 一族的 LLM KV 压缩思想（累积注意力分数 / 观察窗口选 token）搬到视频 DiT。

**设定**。三个超参：预算 $N$（压缩后保留的 token 数）、保护长度 $R$（最近 $R$ 帧免压缩）、sink 帧数 $S$；cache 涨到窗口上限 $M$（本文 21 帧）时触发压缩，目标 $N \le M$。三段划分：**Sink**（前 $S$ 帧，永不动）、**Recent**（最近 $R$ 帧，免压缩保局部连贯）、**Candidate**（两者之间的全部中间 token，被压缩）。

**重要度打分**。对每个候选 key $k_j$，用最近 $R$ 帧的全部 query 对它打出的重要性分数求和（query 维聚合）：

$$
\phi_j = \sum_{r=1}^{R} q_r^{\top} k_j
\tag{Eq. 7}
$$

直觉：$\phi_j$ 是「这个 key 被所有近期 query 共同打了多高的分」。一个被当前生成**反复查询**的 token 是活跃参与者；谁都不查的是冗余历史。注意 Eq. 7 用的是未过 softmax 的原始内积之和——它度量的是 logits 水平的参与度，而不是归一化后的注意力份额（后者会被 sink 的吸票压扁）。

**选择与组装**。取分数最高的 $C = N - R - S$ 个 token：

$$
K_{\mathrm{top}} = \mathrm{Top\text{-}C}(\phi)
\tag{Eq. 8}
$$

按时间顺序拼接成压缩后的 cache（V 侧同样处理；每层 Transformer 独立执行，各层有各自的语义重要性）：

$$
K_{\mathrm{compressed}} = K_{\mathrm{sink}} \,\|\, K_{\mathrm{top}} \,\|\, K_{\mathrm{rct}}
\tag{Eq. 9}
$$

**效率**：压缩只在两个条件同时满足时触发——窗口满 **且** 处于第一个扩散时间步 $t = T$（即 $t=1000$）。选定后，本 chunk 后续 3 个去噪步直接复用，不再重算。gather + sort 的开销由压缩后 16 帧（而非 21 帧）注意力的节省摊销（见第 5 节）。

<details>
<summary><b>展开：为什么 t=1000 选一次就够，以及「去噪 query 不是噪声」的两个标本（附录 C/D）</b></summary>

**标本 1：注意力模式跨扩散时间步一致（Fig. 11）**。生成帧 19–21 时，在 $t = 1000 / 750 / 500 / 250$ 及最终干净 KV 上分别看 query 平均注意力对每个 key 帧的分布——四个时间步的图样几乎相同。所以 $t=1000$ 选出的 Top-C 在整个去噪过程中保持上下文相关。

**标本 2：去噪 query 携带真信号（Fig. 12/13）**。把 Top-C 换成「给每个候选 token 采一个 $\mathcal{N}(0,1)$ 随机分数再选」的对照：随机选择造成严重场景重复与上下文丢失（脸突然转向、头悬空、场景漂移），去噪 query 选择则保持主体一致。Fig. 13 的 token 选择频率热图（1 分钟生成）给出机制：横轴 token 0–32,760，其中 0–15,600 是 Deep Sink、15,600–28,080 是压缩候选、28,080 以后是 recent——去噪 query 的选择**高度集中在 sink 边界（15,600）紧后方**的位置槽位。由于压缩会驱逐 token、后续 token 滑入这些槽位，这说明被反复选中的是「桥接既有上下文与当前生成」的语义锚，而非固定帧或随机噪声。作者的归因：(1) Self Forcing 的 4 步蒸馏让注意力在 $t=1000$ 就快速收敛出有意义的模式；(2) 逐层缓存允许各层按各自的上下文相关性独立选择。

</details>

### 4.4 Temporal RoPE Unification

压缩完还有一个隐患：sink、Top-C、recent 三段的**真实时间戳彼此跳变**——Top-C 里可能同时躺着第 15 帧和第 60 帧的 token。不给它们一个连续的时间索引，等于让模型看一个时间上「打洞」的序列，第 4.2 节的三种症状（闪烁/退化/回卷）会换一个位置回来。做法与 Deep Sink 完全同构：记 $s_{\mathrm{top}}$ 为 Top-C 块应对齐的目标时间位置、$s^{\mathrm{base}}_{\mathrm{top}}$ 为每个缓存 key 的当前时间位置：

$$
\Delta_{\mathrm{top}} = s_{\mathrm{top}} - s^{\mathrm{base}}_{\mathrm{top}}
\tag{Eq. 10}
$$

$$
K^{(\mathrm{time})}_{\mathrm{top}} \leftarrow K^{(\mathrm{time})}_{\mathrm{top}} \odot \exp\big(i\,\omega_t\, \Delta_{\mathrm{top}}\big)
\tag{Eq. 11}
$$

空间维照旧不动。至此整个 cache 在时间维上是连续序列：长程锚（sink）、精选锚（top）、新鲜上下文（recent）各就各位。

---

## 5. 训练与推理细节

**没有训练**。方法直接实现在预训练 chunk-wise Self Forcing 之上，零微调、零新增参数——这也是它与 Rolling Forcing / LongLive（都需要蒸馏或微调）的根本分野。

<details>
<summary><b>展开：超参与吞吐标本</b></summary>

| 量 | 值 | 说明 |
|----|-----|------|
| sink 帧数 $S$ | 10 | 窗口 21 帧的约一半（有效区间 10–15 帧 = 40–60%） |
| 压缩预算 $N$ | 16 帧 | 压缩后的 cache 大小 |
| 保护帧数 $R$ | 4 | 最近 4 帧免压缩 |
| 窗口上限 $M$ | 21 帧 | 触发压缩的阈值（约 32,760 token，每帧 1,560 token） |
| Top-C 容量 | $C = N - R - S$ | token 级选择，逐层独立；触发时机为窗口满且 $t=1000$，每个 chunk 只选一次 |

**吞吐（附录 E，单张 H100，60 秒视频，首次滚动后计）**：

| 方法 | FPS | 延迟 Min/Max |
|------|-----|--------------|
| Self Forcing | 15.78 | 0.770 / 0.776 s |
| Deep Forcing | 15.75 | 0.747 / 0.797 s |

压缩步稍慢、生成步稍快（cache 在 21↔16 帧间振荡），平均下来与基线几乎一致——压缩开销被注意力开销的下降精确摊销。

</details>

---

## 6. 实验与消融

**主表（Table 1，VBench-Long，30 秒）**——注意 Deep Forcing 是唯一不打「attention sink 训练」标签却排在第一行的方法：

| 模型 | 训练 | FPS | Dynamic Degree | Overall Cons. | Imaging Qual. | Aesthetic Qual. |
|------|------|-----|----------------|---------------|----------------|-----------------|
| Rolling Forcing | ✔（含 sink） | 15.79 | 30.71 | 20.99 | 70.58 | 60.24 |
| LongLive | ✔（含 sink） | 18.16 | 45.55 | 20.16 | 69.07 | 61.51 |
| CausVid | ✔ | 15.78 | 47.21 | 19.15 | 66.36 | 59.77 |
| Self Forcing | ✔ | 15.78 | 36.62 | 20.50 | 68.58 | 59.44 |
| **Deep Forcing** | **✘（零训练）** | 15.75 | **57.56** | **20.54** | **69.31** | **60.68** |

60 秒设置下结论不变：Deep Forcing Dynamic Degree **57.19**，仍领先全部基线（LongLive 43.49、CausVid 46.44）；Imaging Quality 69.27 超过所有同规模基线（LongLive 69.11）。Dynamic Degree 的大幅领先有一半是「零训练红利」：为锚定 sink 而显式训练的方法可能顺带压抑了运动，本篇没有这个约束。

**组件消融（Table 4，30 秒）**：

| 变体 | Dynamic Degree | Overall Cons. | Image Quality |
|------|----------------|---------------|---------------|
| Self Forcing（基线） | 36.62 | 20.50 | 68.58 |
| + Deep Sink（S=10） | 48.58 | 20.54 | 68.54 |
| + DS + PC（完整） | **57.56** | 20.54 | **69.31** |

Dynamic Degree 从 36.62 → 48.58 → 57.56 两级跳；图像质量在 30 秒上先微降（68.58→68.54）再由 PC 拉起——论文诚实注明 30 秒时 DS 对图像质量略有负作用，其对保真的正作用在 50 秒设置（Fig. 5）里才显形。

<details>
<summary><b>展开：用户研究、VLM 评测与 PC 内部消融全表</b></summary>

**用户研究（Table 2，24 人 2AFC，「偏好 Deep Forcing 的票数占比」）**：

| 对手 | 颜色一致性 | 动态运动 | 主体一致性 | 整体质量 |
|------|-----------|---------|-----------|---------|
| CausVid | 98.9% | 95.8% | 96.8% | **100%** |
| Self Forcing | 85.9% | 86.9% | 84.8% | 87.9% |
| LongLive | 71.2% | 83.5% | 72.2% | 72.2% |
| Rolling Forcing | 76.7% | 76.7% | 80.0% | 78.9% |

值得注意的是主体一致性：VBench-Long 上 Deep Forcing 的 Subject Consistency（97.34）低于训练型方法（LongLive 97.97、Rolling Forcing 98.12），但用户研究里对 LongLive 的主体一致性偏好仍有 72.2%——指标与人对「身份漂移」的感知并不对齐。

**VLM 视觉稳定性（Table 3，Gemini 2.5-Pro，30 秒，越高越好）**：CausVid 42.84、Self Forcing 43.94、Rolling Forcing 72.6、LongLive 78.58、**Deep Forcing 75.44**——零训练方法达到与训练型方法同档，明显甩开无 sink 的基线。

**PC 查询来源消融（附录 Table 5，30 秒）**：

| Top-C 查询来源 | Motion Smooth. | Overall Cons. | Image Quality |
|----------------|----------------|---------------|---------------|
| 只用当前去噪 query | 97.86 | 20.44 | 68.24 |
| 只用过去干净 query | 97.91 | 20.47 | 68.54 |
| 两者都用（完整） | **98.27** | **20.54** | **69.31** |

解释：干净历史 query 提供稳定的重要度估计，去噪 query 保证选中 token 与「正在生成的内容」相关，二者互补——只用去噪 query 时，$t=1000$ 的噪声性让选择不稳定，图像质量掉到 68.24。

</details>

---

## 7. 与系列其他篇目的关系

| 对照 | 本文 | 对方 | 分歧点 |
|------|------|------|--------|
| [Self-Forcing++（10）](../forcing_10/) | 推理侧手术：改缓存管理，模型冻结 | 训练侧手术：短片段教师监督长 rollout | 同样对抗长程退化，一个不动权重、一个只动权重 |
| Rolling Forcing / LongLive | sink 深达 10 帧 + 时间相位旋转，零训练 | 3 帧 sink，靠蒸馏/微调获得 | 附录 A 证明浅 sink + 不调 RoPE 在 60 秒必然失败——「训过的 sink 也会回卷」 |
| [Context Forcing（12，下一篇）](../forcing_12/) | 历史记忆维度：历史**怎么存取** | 教师与目标维度：蒸馏时教师**看不看得到**长历史 | 本篇管缓存，下篇管监督——长视频问题的两条正交轴 |
| [WM 04（RELIC）](../world_memory/WM_04_relic/) | 缓存管理：控制历史对**注意力**的可见性 | replayed back-propagation：控制历史对**梯度**的传播 | 同族手术——都在「过去如何进入当前计算图」上做文章，一个改前向、一个改反向 |
| [WM 05（WorldPlay）](../world_memory/WM_05_worldplay/) | Temporal RoPE Adjustment：旋转 K 的相位，把 sink 拉近当前时间线 | Temporal Reframing：丢弃绝对时间索引，给检索帧重赋固定小距离 | 同一堵墙（RoPE 时间外推失真）的两种翻法：旋转已缓存的 K vs 重赋位置编码；一个保留真实时间差、一个直接声明时间差无关 |
| [Causal Forcing（13）](../forcing_13/) | 被引为长视频正交适配手段 | 专注 5 秒内的蒸馏初始化 | 13 明确说长视频外推与本篇方法正交可组合 |

在系列的坐标系里，本篇把「长视频 = 训练问题」的默认假设掀开了一角：**预训练模型自带的长程结构（深 attention sink）只是被 FIFO 缓存浪费了**。这也是第三部分「历史记忆」维度的收官论点。

---

## 8. 局限与个人评价

**论文自认**：推理时方法受冻结主干的容量与偏置约束；**没有显式长期记忆**，反复遮挡的极长序列仍会缓慢漂移；未来方向是分层记忆模块。

**顺着系列主线的四条评价**：

1. **「深 sink」是对 StreamingLLM 移植假设的修正**。LLM 里 sink 是少数初始 token，视频方法照抄「前 3 帧」；本文的注意力分析说明视频 DiT 的 sink 是**深而分散**的——中间帧也是锚。attention sink 的本质是「低频全局信息的汇聚点」，其深度应按信息密度定，而不是按位置定——与 [WM 05](../world_memory/WM_05_worldplay/) 里「首 chunk 永久保留 + 空间记忆检索」的构图遥相呼应。
2. **参与度筛选是 LLM 技术的诚实移植，但有视频特有的效率设计**。「$t=1000$ 选一次、跨去噪步复用」利用了扩散多步共享 KV 的结构，LLM 里没有对应物。附录 D 对「去噪 query 是否携带信号」的随机对照（Fig. 12/13）是全文最扎实的机制证据。
3. **Dynamic Degree 的暴涨要打折扣读**。57.56 vs 36.62 里既有缓存管理的贡献，也有「没有为锚定而训练」的红利——训练型 sink 方法可能把运动一起锚死了。论文自己给出这个解释，但无法单独消融「零训练」因素（所有训练型基线都训过）。
4. **与下一篇的分工是本篇最好的定位**。历史 KV 的**存取**（本篇）解决之后，长视频蒸馏剩下的瓶颈在**监督**：教师看不见长历史，学生的长上下文就没有对的梯度来源——这正是 [Context Forcing（12）](../forcing_12/) 的靶子，「师生信息一致性」主题的第二次登场。

{{< forcing_nav >}}
