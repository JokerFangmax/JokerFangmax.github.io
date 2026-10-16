---
title: "Forcing: 20. Self Gradient Forcing"
date: 2026-10-14
math: true
tags:
  - "Forcing"
  - "Video Generation"
  - "Self Gradient Forcing"
categories:
  - Forcing
series: forcing
weight: 20
description: "Self Gradient Forcing（arXiv:2607.20368）精读：Self Forcing 把历史 KV 当冻结状态，未来损失只能监督「读」不能监督「写」——历史上下文梯度缺口。SGF 两遍训练修复它：Pass 1 无梯度自 rollout 并记录采样退出步，Pass 2 并行重建上下文 KV 与未来-历史因果注意力，历史 latent 保持 stop-gradient。5s 训练窗口外推 240s：TF 初始化下 subject consistency 0.965→0.972、chunk-wise 240s flickering 0.934→0.970；重建保真 cosine 0.999886、相对 L2 1.41%（约为 bf16 精度 1.8×）；代价仅峰值显存 79.01→87.01GB、每 5 步 10.39→11.71s，而直接可微 cache 直接 OOM。GSB 人评全部为正（最高 +48.7%）"
paper_id: "2607.20368v2"
---

# Self Gradient Forcing：未来损失能否监督历史记忆写入

> **Forcing 系列 · 第 20 篇 · 效率与上下文梯度**
> 原文：Zhuang et al., *Self Gradient Forcing: Native Long Video Extrapolation*, Joy Future Academy, JD。
> 阅读前提：[Self Forcing（第 07 篇）](../forcing_07/)的 rollout 训练与 DMD、[扩散模型基础](../video_generation/VG_01_diffusion_basics/)。
>
> 📂 **阅读方式**：正文只留主线；显存代价推导、sink 消融与全表收进折叠块（点击展开）。

## 论文信息

- **原题**：Self Gradient Forcing: Native Long Video Extrapolation
- **阅读版本**：arXiv:2607.20368v2，对应本地收集的 PDF 版本。
- **原文**：[arXiv 摘要页](https://arxiv.org/abs/2607.20368v2) · [PDF](https://arxiv.org/pdf/2607.20368v2)

## 1. 这篇论文解决什么问题

[Self Forcing](../forcing_07/) 让学生在自己 rollout 出的历史上训练，解决了「历史来源」的训推错配。但仔细看它的梯度流，会发现一个没被修复的缺口：rollout 时，每个生成块 $\tilde{x}_i$ 会被模型在**干净上下文时间步** $t_{\text{ctx}} = 0$ 重新编码成 K/V 写进 cache，后续块读的是这份**记忆表示**而非 $\tilde{x}_i$ 本身。而 Self Forcing 出于计算可行性，把历史 KV 当作**冻结的 rollout 状态**——

| 梯度通路 | Self Forcing 里是否存在 | 后果 |
|---------|------------------------|------|
| 未来损失 → 噪声去噪 token（**读**缓存） | 存在 | 模型学会怎么读历史 |
| 未来损失 → $t_{\text{ctx}}=0$ 的写入计算（**写**缓存） | **被截断** | 「早先的 latent 该被编码成什么样更好用的记忆」无人监督 |

论文把这条缺失的信用分配路径命名为**历史上下文梯度缺口（historical context-gradient gap）**。缺口不是纯理论问题，它会被两个机制放大：同一个 DiT 的参数在所有时间步共享，噪声时间步上的 DMD 更新会顺带改写 $t_{\text{ctx}}=0$ 的 cache 写入行为，而后续块的损失又无法回传去纠正——**写入通路在训练中漂移，却收不到任何来自下游的反馈**。定性表现（论文 Fig. 1/3）：Self Forcing 的 rollout 局部始终合理，但主体身份、视角、场景布局随外推逐渐崩坏。

直接解法是把历史 cache 保持可微，让未来损失反传进写入计算——但那要求把每次 cache 写入的 autograd 图保留到所有消费者处理完为止，图随 rollout 长度增长，实际直接 OOM（§6.4 有实测）。

一句话总结：**SGF 把「串行保留反传图」的问题变成「有界并行重算」的问题——Pass 1 无梯度 rollout 记录退出状态，Pass 2 丢掉串行 cache、在固定窗口内并行重建同一计算并打开上下文梯度，历史 latent 本身保持 stop-gradient。**

## 2. 最小背景

三块前置。

**Self Forcing 的两遍结构（本篇直接继承的骨架）。** SF 的可训练化技巧是：逐块串行 rollout，但只有随机采样的一个「退出步」保留梯度。用两遍视角看：Pass 1 做无梯度 rollout 并记录退出状态；Pass 2 并行重建该退出步计算。关键在于 SF 的 Pass 2 里，上下文 K/V 路径是 **detached** 的——SGF 改的恰恰只有这一处梯度边界。

**DMD 与因果学生。** 少步因果学生 + 双向教师 + DMD 分布匹配损失，这套蒸馏管线沿用 [CausVid（第 06 篇）](../forcing_06/)/[Self Forcing](../forcing_07/)；初始化可在 TF / 一致性蒸馏（CD）/ ODE 三种里选，本文对三种初始化都做了对照。

**sink + FIFO 流式上下文。** 长时推理用「固定 sink 前缀 + FIFO 近期窗」的 cache 策略。本文用 Wan 视频 VAE，其时间分组是不对称的（开头 1 帧一组、之后 4 帧一组），由此产生的「流边界前缀」问题决定了 sink 的取值（§5 展开）。

## 3. 方法总览

SGF 的训练循环由两遍构成，模块职责如下：

| 模块 | Pass 1（串行，无梯度） | Pass 2（并行，开梯度） |
|------|----------------------|----------------------|
| 自回归 rollout | 逐步去噪、逐块写入串行 KV cache——**与推理完全一致** | 不做 rollout |
| 记录 | 采样退出步 $t^\star$：每块的噪声输入 $z^{t^\star}_i$ 与干净预测 $\tilde{x}_i$ | 读取记录值当固定数据 |
| 上下文编码 | $t_{\text{ctx}}=0$ 写入串行 cache（无梯度） | **重算**：$\mathrm{sg}(\tilde{x}_i)$ 经 $t_{\text{ctx}}=0$ 前向得到可微 KV |
| 目标计算 | 无 | 记录的 $Z^\star$ 作预测输入，在重建 mask $M_{\text{rec}}$ 下并行算 $\hat{X}_{\text{tar}}$ |
| 损失 | 无 | $\mathcal{L}_{\text{DMD}}(\hat{X}_{\text{tar}})$，梯度同时穿过去噪侧与上下文 KV 侧 |

两条边界设计是方法的全部要点：

1. **历史 latent 保持 stop-gradient**：$\tilde{x}_i$ 是记录的固定数据，SGF 不优化采样轨迹本身——它训练的是「给定这些 latent，如何把它们编码成对未来更有用的 K/V」，而不是「如何生成更好的 latent」（后者属于完整 BPTT 的职责，SGF 明确不主张）。
2. **Pass 2 是固定窗口的并行计算**：在 $M_{\text{rec}}$（复现串行 cache 诱导的 sink+窗口注意力关系）下一次性前向，autograd 图不随 rollout 长度增长——这就是「有界」的含义。

直觉版总结：Pass 1 负责「生成什么样的世界」，Pass 2 负责「教模型怎么记住这个世界」；两遍之间唯一的桥梁是记录下来的 $(Z^\star, \tilde{X})$。

## 4. 核心原理完整推导

### 4.1 写入计算与缺口的 formal 化

设 $C_\theta$ 为因果生成器在 $t_{\text{ctx}}=0$ 诱导的 cache 写入计算。串行 rollout 中它是递归的——写第 $i$ 块的 K/V 要读之前的 cache（论文 Eq. 1）：

$$
\mathrm{KV}^0_i(\theta) = C_\theta\big(\tilde{x}_i,\ t_{\text{ctx}};\ \mathrm{KV}^0_{<i}\big), \qquad t_{\text{ctx}} = 0
\tag{Eq. 1}
$$

如果未来损失能对这个历史条目求导，它就能把信用分配给「把 $\tilde{x}_i$ 编码进记忆」的那次干净上下文计算——既训练后面的噪声 token 怎么读，也训练前面的 latent 该怎么被写。但冻结 cache 的 Self Forcing 把 $\mathrm{KV}^0_i$ 当 detached 状态。

**为什么缺口会随训练恶化**（论文 Eq. 2）：共享参数被噪声时间步的 DMD 损失更新，

$$
\theta_{r+1} = \theta_r - \eta\, \nabla_\theta \mathcal{L}_{\text{SF}}(\theta_r), \qquad \mathrm{KV}^0_i(\theta_{r+1}) \not\equiv \mathrm{KV}^0_i(\theta_r)
\tag{Eq. 2}
$$

训练会改变干净上下文的写入行为，但冻结 cache 的 SF 对「自生成 latent 被编码成什么样」没有任何损失侧的修正信号。直觉：一个共享笔记本的两个人，一个只被考核「读得准不准」，另一个的写法却一直在变——读的人迟早发现笔记跟不上。

### 4.2 直接可微 cache 为什么不可行

naive 想法：把串行历史 cache 保持可微不就行了？问题是 cache 的形成是**递归**的——第 $i$ 块的 K/V 由读着更早 cache 的前向算出，若那些更早条目也可微，autograd 图就在 rollout 上递归生长。瓶颈不是 KV 张量本身，而是挂在递归 cache 轨迹上的保存激活。附录 C 给出 scaling 论证（详见折叠块）：直接可微 cache 的显存 $\gtrsim$ 原始 cache 张量 + 全部写入前向的激活 + 注意力中间量，随块数线性增长；实测直接 OOM（§6.4）。而 SGF 的重建只有固定窗口 $N$ 的大小（论文 Eq. 9–10）：

$$
M_{\text{SGF}} \approx M_{\text{pass1 cache data}} + M_{\text{records}}(\tilde{X}, Z_{t^\star}) + M_{\text{parallel window}}(N)
\tag{Eq. 9}
$$

$$
M_{\text{SGF}} < M_{\text{direct}} < M_{\text{full-BPTT}}
\tag{Eq. 10}
$$

### 4.3 Pass 2 的重建与梯度边界

冻结 cache SF 的 Pass 2 视角（论文 Eq. 3）：给定记录的噪声 latent $Z^\star$ 与上下文 latent $\tilde{X}_{\text{ctx}}$，在因果重建 mask $M_{\text{rec}}$ 下并行重建目标预测：

$$
\hat{X}_{\text{tar}} = G_\theta\big(Z^\star,\ t^\star;\ \tilde{X}_{\text{ctx}},\ t_{\text{ctx}},\ M_{\text{rec}}\big)
\tag{Eq. 3}
$$

$M_{\text{rec}}$ 复现串行 cache 诱导的 sink+窗口注意力关系。SF 在这一步把上下文侧 K/V 视为 detached 记忆；**SGF 保持 Pass 1 与重建几何完全不变，唯一改动是移除重建上下文 K/V 路径上的 stop-gradient**。上下文 latent 本身仍是 $\mathrm{sg}(\tilde{X}_{\text{ctx}})$ 输入，但模型在 $t_{\text{ctx}}=0$ 重新编码它们，得到的 K/V 条目对未来 token 可见时保持可微。于是 DMD 梯度包含（论文 Eq. 4）：

$$
\nabla_\theta \mathcal{L}_{\text{DMD}}(\hat{X}_{\text{tar}}) \supset \frac{\partial \mathcal{L}_{\text{DMD}}}{\partial \mathrm{KV}^{\text{rec}}_{\text{ctx}}} \cdot \frac{\partial \mathrm{KV}^{\text{rec}}_{\text{ctx}}}{\partial \theta}
\tag{Eq. 4}
$$

直觉读法：损失对「重建出的上下文 KV」求偏导（未来帧觉得这份记忆哪里不好用），再链到参数（写入计算该怎么改）——**「写」第一次收到了来自「读」的反馈**。

梯度边界总结：$\tilde{X}_{\text{ctx}}$、噪声样本、scheduler 状态、Pass 1 的串行 cache 轨迹全部 stop-gradient；梯度只流过 Pass 2 的干净上下文前向、K/V 投影、未来-上下文注意力、目标侧去噪。

<details>
<summary><b>展开：附录 C 的显存 scaling 论证（Eq. 5–8）</b></summary>

冻结 cache 的原始 KV 张量占用（论文 Eq. 5；$B$ 批、$T$ 保留块、每块 $L$ token、宽度 $D$、$H$ 层、因子 2 是 K 与 V）：

$$
M_{\text{cache}} \approx H \cdot 2 \cdot B \cdot T\, L\, D \cdot \text{bytes}
\tag{Eq. 5}
$$

直接可微 cache（latent 已 detach，但 cache 形成可微）还要加上每次 $t_{\text{ctx}}=0$ 写入前向的保存激活与其对更早 cache 状态的依赖（论文 Eq. 7）：

$$
M_{\text{direct}} \gtrsim M_{\text{cache}} + \sum_{i=1}^{T} M_{\text{KV formation}}(i) + M_{\text{saved attention}}
\tag{Eq. 7}
$$

完整 BPTT（latent 也不 detach）再加上全部去噪步的激活（论文 Eq. 8）：

$$
M_{\text{full-BPTT}} \gtrsim M_{\text{direct}} + \sum_{i=1}^{T} \sum_{s=1}^{K_i} M_{\text{denoise}}(i, s)
\tag{Eq. 8}
$$

注意论文的诚实标注：Eq. 7/8 是 scaling 论证而非 allocator 精确公式；它的作用是说明**即使是最小版的直接可微 cache（latent detach）也打开了串行历史形成图**。三项的排序 Eq. 10 就是 SGF 选择「固定窗口并行重建」的依据。

</details>

### 4.4 重建保真：Pass 2 凭什么等于 Pass 1

梯度恢复的前提是 Pass 2 忠实重放了 Pass 1 的退出步计算。理论上：确定性层、匹配的位置索引、相同的因果重建几何下，$\hat{X}_{\text{tar}}$ 应与记录的 $\tilde{X}_{\text{ctx}}$ 完全一致。实测（附录 D，Table 7）：24 个 prompt × 4 个退出步 $\{1000, 750, 500, 250\}$ 共 96 组对照，整体相对 $\ell_2$ 误差 **1.41%**、余弦相似度 **0.999886**；把误差归一到 bf16 的相对精度 $\epsilon_{\text{bf16}} = 2^{-7}$ 上，整体只有 **1.80×**，且从最噪退出步的 2.73× 单调降到最干净退出步的 1.04×——**重建误差就是浮点舍入的量级，不是计算错配**。两个执行路径（串行 cache vs 并行重建）只是 bf16 运算的累加顺序不同。

这一节是方法成立的实验地基，但注意它的边界（论文自己写明）：验证的是**前向**恢复局部注意力计算到混合精度数值误差；**不**意味着 SGF 恢复了完整串行 rollout 的精确梯度。

### 4.5 流式上下文策略：VAE 边界决定 sink 数

帧式（frame-wise）流式生成用 sink + FIFO：sink 4 + FIFO 16 + 当前块 1，总窗 21 个 latent 帧。sink=4 不是拍的：Wan VAE 时间编码是不对称分组（开头 1 帧边界组 + 后续 4 帧组），新编码流的前几个 latent 不等价于稳态流内 latent。边界诊断（附录 E）：把目标帧组当「新鲜流」编码，与全前缀编码的第 21 个 latent 比相对 $\ell_2$ 误差 **0.607**；加 1 帧边界锚点降到 **0.271**；再逐个加之前的 4 帧组降到 0.124 / 0.084 / 0.067 / 0.060（W=1..4），之后基本不变——**前几个 VAE latent 构成一个短「边界过渡区」，sink 必须把整个过渡区罩住**。sink 消融（Table 8）确认 sink=1 的 aesthetics（0.627）与 flickering（0.990）弱于 sink=4（0.653 / 0.991），sink=8 只有边际收益还挤占 FIFO 预算。

<details>
<summary><b>展开：sink 消融全表（Table 8，帧式 SGF + TF 初始化，60s）</b></summary>

| 指标 | sink 1 | sink 2 | sink 4 | sink 8 |
|------|--------|--------|--------|--------|
| Aesthetics | 0.627 | 0.645 | 0.653 | 0.655 |
| Background | 0.970 | 0.974 | 0.974 | 0.976 |
| Dynamics | 0.743 | 0.733 | 0.730 | 0.728 |
| Imaging | 0.715 | 0.713 | 0.714 | 0.714 |
| Motion | 0.982 | 0.982 | 0.982 | 0.983 |
| Subject | 0.983 | 0.982 | 0.983 | 0.983 |
| Flickering | 0.990 | 0.990 | 0.991 | 0.992 |

总上下文预算固定，加大 sink 就是在压缩 FIFO。趋势不是单调改善，而是「小 sink 罩得住 VAE 边界过渡区即可」——sink 4 是覆盖观测到的边界前缀的最小取值。

</details>

## 5. 训练与推理细节

| 项 | 值 |
|---|---|
| 训练窗口 | **全部模型统一 5s**——60s/240s 均为超出训练视界的原生外推 |
| 初始化 | TF / 因果 CD / 因果 ODE 三种对照（因果 ODE 基线用 Causal Forcing 发布 checkpoint） |
| 帧式配置 | 训练用全 5s 窗（无滑窗驱逐），Pass 2 用 teacher-forcing 式因果 mask；推理 sink 4 / FIFO 16 / 当前 1（总窗 21） |
| 块式配置 | 训练 Pass 1 就带滑窗：sink 3 / FIFO 6 / 当前块 3（chunk size 3，总窗 12），Pass 2 mask 匹配同一几何 |
| Pass 2 实现 | FlexAttention + 编译的静态块稀疏因果 mask |
| 更新节奏 | 假分数网络:生成器 5:1 交替；每 5 步里 Pass 2 的额外开销只落在生成器更新上 |
| 评测 | 5s 标准 VBench（sanity check）；60s 用 VBench-Long prompt；240s 用 128 条 MovieGen prompt |

帧式与块式的对照设计本身是实验的一部分：帧式在**最细时间粒度**上写记忆（错误累积最快），块式在粗粒度上写（每块要当好几个块之后的上下文）——两种粒度都成立，说明上下文梯度信号不依赖特定的记忆更新粒度。

## 6. 实验与消融

### 6.1 长时生成：一致性与稳定性全面占优

Table 1（帧式）TF 初始化一列（SF → SGF）：

| 视界 | 指标 | SF | SGF |
|------|------|-----|-----|
| 60s | Aesthetics | 0.650 | **0.653** |
| 60s | Subject | 0.976 | **0.983** |
| 60s | Flickering | 0.988 | **0.991** |
| 240s | Subject | 0.965 | **0.972** |
| 240s | Imaging | 0.701 | **0.718** |
| 240s | Flickering | 0.967 | **0.968** |

更陡的差距出现在因果 ODE 初始化下（240s flickering 0.937 → **0.972**；subject 0.936 → 0.969）——初始化越弱、写入漂移越重，SGF 修复的收益越大。块式（Table 2）TF 初始化 240s：subject 0.946 → **0.975**、background 0.944 → **0.970**、flickering 0.934 → **0.970**、aesthetics 0.557 → **0.629**——块式下差距整体更大，粗粒度记忆写入更需要未来监督。

**Dynamic degree 是唯一系统性例外**（如块式 TF 240s：0.943 → 0.566）。论文的解释：Self Forcing 长 rollout 里的场景跳变、破碎相机几何、物体形变制造了「大而不连贯的表观运动」，反而抬高 dynamic degree；SGF 画面更稳，动态度分数反而低。这个辩护与附录 H 的定性失败模式（场景跳变/裁切漂移/身份漂移）对得上，但读者应记住：**该指标在长时设定下不能按字面读**。

### 6.2 人类偏好：GSB 全部为正

盲测 GSB（$(G-B)/(G+S+B)$，正 = 偏好 SGF），**1,900+ 配对判断、10 组配对比较**（Table 3）：

| 设置 | 60s | 240s |
|------|-----|------|
| 帧式 · 因果 ODE | +29.6% | +38.6% |
| 帧式 · 因果 CD | +36.8% | +35.9% |
| 帧式 · TF | +45.1% | **+48.7%** |
| 块式 · 因果 CD | +32.9% | +34.3% |
| 块式 · TF | +37.5% | +44.9% |

十组全正、TF 初始化下最强——自动指标之外，人眼也确认了修复的方向。

### 6.3 5s 视界：不伤短时质量

设计原则是「SGF 改的是长时外推，不该退化 5s 质量」。附录 B 的完整 16 维表显示各配对内互有胜负、整体相当（如帧式 TF：subject 0.961→0.968，dynamics 0.633→0.653，action 0.962→0.958）——sanity check 通过。

### 6.4 训练可行性：修复梯度的真实价格

Table 4 是全文最硬的一张表：

| 训练变体 | 峰值显存 | 稳态显存 | 每 5 步耗时 | 结果 |
|---------|---------|---------|------------|------|
| Self Forcing（冻结 cache） | 79.01 GB | 79.01 GB | 10.39 s | 可训练 |
| **SGF（Pass 1 + Pass 2）** | 87.01 GB | **63.73 GB** | 11.71 s | 可训练 |
| 直接可微 cache | OOM | OOM | — | **OOM** |

三个读数：(i) 直接可微 cache 实测 OOM，验证 §4.2 的 scaling 论证；(ii) SGF 峰值只涨 8 GB（+10%），稳态反而**降** 15.28 GB——Pass 1 的串行 cache 是临时数据、Pass 2 丢掉它重算，稳态 footprint 更小；(iii) 耗时 +12.7%（10.39→11.71s/5 步），因为 SF 本来就要做 $n-1$ 次无梯度前向 + 1 次带梯度前向/反传，SGF 只是 Pass 1 多做一次前向、Pass 2 做一次带梯度前向/反传——**增量主要是一次前向，不是反传**。

## 7. 与系列其他篇目的关系

| 对照 | SGF 的立场 | 对方 | 分歧点 |
|------|-----------|------|--------|
| [第 07 篇 Self Forcing](../forcing_07/) | 恢复被截断的上下文写入梯度 | 梯度截断换可行性 | 全系列「梯度职责」线的第一组对照：SF 截断、SGF 重建——两者共享 Pass 1 与重建几何，**唯一区别是 Pass 2 的梯度边界**（这使全部实验都能做成配对对照） |
| [第 19 篇 Sparse Forcing](../forcing_19/)（上一篇） | 记忆的梯度 | 记忆的内容与结构 | 19 的 Top-C 写入是无梯度评分规则，20 问「写入该受谁监督」——两篇正交，SGF 的上下文编码可以换成任何 cache 设计 |
| [第 21 篇 SGF+](../forcing_21/)（下一篇） | 修复缺口 | 处理修复后的梯度冲突 | 直接续篇：SGF 把两种职责的梯度放进同一组参数，SGF+ 发现它们方向相反 |
| [第 13 篇 Causal Forcing](../forcing_13/) | 提供因果 ODE 基线 checkpoint | 解决初始化错配 | 「初始化质量」与「记忆写入监督」是两个可叠加的修复，实验里三种初始化的配对都成立 |
| [WM 03 Memory Forcing](../world_memory/WM_03_memory_forcing/) | 未来损失监督历史表示的写入 | Chained rollout 把模型自己的预测链进后续训练窗口 | 同族思想：都是「训练时让模型面对自己犯的错」——WM 03 的 CFT 链的是**条件帧分布**（让训练覆盖推理期 drift），SGF 补的是**梯度通路**（让写入行为收到下游反馈）；一个改数据分布，一个改梯度流 |
| [WM 04 RELIC](../world_memory/WM_04_relic/) | 有界重建 | replayed back-propagation 分块重放反传 | 工程结构惊人相似：都是「先无梯度全 roll、记录、再有界地重放反传」；差别在 RELIC 重放的是 DMD 分数差、SGF 重放的是上下文编码 + 因果注意力 |

另一个值得记录的学术史细节：附录 F 承认 Solaris（Minecraft 多人世界模型）的 Checkpointed Self Forcing 与 SGF 结构相近（无梯度 rollout 缓存干净估计与噪声状态、再并行重放采样去噪步）——SGF 的独有贡献是把该结构形式化为「上下文梯度重建」并给出缺口论证与系统实验。诚实标注先行工作这点值得肯定。

## 8. 局限与个人评价

**论文自认的三条**（附录 G）：SGF 是缺失上下文梯度的**有界替代**而非完整 rollout BPTT——它不更新采样的 latent 本身，也不优化产生它们的去噪决策序列；方法假设 Pass 2 忠实重放串行上下文关系，若 mask/sink/FIFO/RoPE/chunk 对齐与推理有任何偏差，SGF 会「为错误的注意力关系训练错误的写入器」；SGF 不替代其他长视频技术（流式长调、检索、稀疏注意力、长上下文教师都作用于互补部位）。

**顺着主线再补三条**：

1. **监督只到「一个窗口内的未来」**。Pass 2 的重建窗口固定（帧式即 5s 训练窗），「未来损失」其实只是「近未来损失」——一个 latent 要在几十秒后才变得关键（长程重访型内容）时，它的写入行为收不到任何信号。这解释了为什么 SGF 修复的是身份/布局/闪烁这类**局部持续量**，而不是真正的长程记忆检索。与 [WM 03](../world_memory/WM_03_memory_forcing/) 的链式 rollout 对读会发现：CFT 用数据分布覆盖长时 drift，SGF 用梯度覆盖短时写入——**两者合起来仍然没有「长时写入监督」**。
2. **梯度的「质」没有被单独验证**。Table 7 验证的是前向重建保真（cosine 0.999886），但 Pass 2 恢复的梯度与「真正穿过串行 rollout 的梯度」的夹角从未被测量——理论上 Pass 2 丢弃了「写入对更早 cache 的递归依赖」这条路径（串行 cache 形成是递归的，并行重建把它摊平了）。重建的前向一致不蕴含梯度一致，这是「有界替代」最需要警惕的暗角。
3. **「写入漂移导致长时退化」的因果链缺一环**。Eq. 2 证明参数更新会改变写入行为，实验证明修复梯度能改善长时一致性，但「漂移的写入 → 长时崩坏」这个中间环节没有直接观测（例如没有跟踪训练过程中 KV 写入分布的漂移量）。辩护证据是间接的——最弱的初始化（ODE）受益最大这一点强烈暗示写入质量是瓶颈，但机制层面的直接测量仍然缺席。

总体评价：这是「梯度职责」维度的开山篇——问题定义（读/写梯度不对称）干净利落，解法（挪一处 stop-gradient）极简，实验设计（一切配对化、只变梯度边界）近乎教科书级。它的价值不在单个数字，而在把一个所有人都在默默忍受的截断变成了可见、可命名、可修复的对象。下一篇 [SGF+](../forcing_21/) 将顺着这个修复往下问：两种职责的梯度进了同一组参数之后，会发生什么。

{{< forcing_nav >}}
