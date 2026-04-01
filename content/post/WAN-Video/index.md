---
title: "WAN-Video Technical Report"
date: 2026-04-01
math: true
tags:
  - Notes
  - Video Generation
categories:
  - Video Generation
description: "WAN视频生成模型"
---

# Wan Technical Report 学习笔记

> 论文：**Wan: Open and Advanced Large-Scale Video Generative Models**  
> 适用场景：博客笔记 / 技术报告导读 / 复习提纲  
> 原始报告核心信息见：摘要、目录、模型设计、评测与扩展应用部分。

---

## 1. 这篇报告在讲什么

Wan 是阿里巴巴提出的一套**开源大规模视频生成模型体系**。整篇 technical report 不只是给出一个模型结果，而是尽量完整地公开了一条视频基础模型路线：

- 如何构建大规模图像/视频训练数据；
- 如何设计适合视频的 **spatio-temporal VAE**；
- 如何训练基于 **Diffusion Transformer + Flow Matching** 的视频生成主干；
- 如何做大规模训练和推理加速；
- 如何扩展到 I2V、视频编辑、个性化视频、实时生成、视频配音等任务；
- 如何建立自动化 benchmark 来评估视频生成质量。

从定位上看，Wan 想解决开源视频模型长期存在的三个问题：

1. **性能不如闭源模型**；
2. **能力覆盖面不足**，很多模型只支持基础 T2V；
3. **推理成本高**，普通创作者难以真正使用。

---

## 2. Wan 的几个核心卖点

报告中把 Wan 的特点概括得很明确：

### 2.1 性能强
Wan 的 14B 模型在大规模图像和视频数据上训练，在内部与外部 benchmark 上都表现突出，并在报告中宣称优于多个开源与部分闭源对手。

### 2.2 覆盖任务广
Wan 不仅做文本生成视频，还扩展到了：

- 图像生成视频（I2V）
- 指令驱动视频编辑
- 个性化视频生成
- 实时视频生成
- 音频生成等任务。

### 2.3 小模型效率高
除了 14B 模型外，Wan 还提供了 1.3B 模型。报告强调 1.3B 版本仅需 **8.19GB 显存** 就可运行，面向消费级显卡更友好。

### 2.4 强调开源
作者希望通过公开模型、代码、训练细节、评估体系，推动开源视频生成社区发展。

---

## 3. 文章整体结构

从目录可以看出，这篇报告的逻辑非常完整，大致分为六块：

1. **引言与相关工作**
2. **数据处理流水线**
3. **模型设计与加速**
4. **评测与消融**
5. **扩展应用**
6. **局限与结论**。fileciteturn0file0L25-L67

可以把它理解成：  
**“从数据 → 表示学习 → 生成主干 → 训练系统 → 推理系统 → 下游应用”的完整工程化视频基础模型说明书。**

---

## 4. 背景：Wan 站在什么技术脉络上

报告先回顾了当前视频生成的发展脉络。

### 4.1 闭源方向
以 Sora、Kling、Runway、Veo 2 等为代表，特点是资源投入大、质量高、商业落地快。

### 4.2 开源方向
开源社区主要沿着扩散模型路线推进，通常包括三个基础组件：

- **Autoencoder / VAE**：把像素压到 latent space；
- **Text Encoder**：把文本 prompt 编码成条件；
- **Diffusion / Flow-based Generator**：在 latent 空间中建模视频分布。fileciteturn0file0L137-L167

在主干结构上，视频模型大致经历了：

- 从 **2D U-Net** 扩展到 **3D U-Net**；
- 或采用 **空间注意力 + 时间注意力** 的混合设计；
- 再进一步转向更可扩展的 **Diffusion Transformer (DiT)**。

Wan 明确站在这个主流路线之上：  
**VAE + Text Encoder + Diffusion Transformer + Flow Matching**。

---

## 5. 数据：Wan 为什么很重视数据工程

报告很强调一点：**高质量数据是大模型效果的核心前提**。  
他们的数据构建原则是：

- 高质量
- 高多样性
- 大规模。

而且不是简单“多抓数据”，而是构建了比较系统的自动化数据流水线。

### 5.1 预训练数据的清洗

作者对候选视频/图像做了多维过滤，主要包括：

- 文本检测：去掉文本覆盖太多的素材；
- 美学评分；
- NSFW 过滤；
- 水印 / logo 检测；
- 黑边检测；
- 过曝检测；
- 合成图检测；
- 模糊检测；
- 时长与分辨率约束。

这一步的目的非常明确：  
**先快速去掉低质量、无效、污染性强的数据。**

其中一个很值得注意的观点是：  
报告认为即便少量生成图污染（<10%）也可能显著伤害模型性能，因此他们专门训练了分类器去过滤生成图。

### 5.2 视觉质量筛选

他们把候选数据先分成 100 个 cluster，再在各个 cluster 内评分与筛选。这样做是为了避免长尾数据在全局筛选中被淹没。

这背后的思路很值得记：

> **高质量筛选不能只盯着“平均质量”，还要保住数据分布的多样性。**

### 5.3 运动质量筛选

对于视频生成来说，视频里“有没有合理运动”非常关键。  
因此 Wan 不只看画质，还专门评估 motion quality，把视频分成：

- 最优运动
- 中等运动
- 静态视频
- 相机驱动运动
- 低质量运动
- 抖动相机视频。

其中：

- 静态视频不会完全删除，但会降低采样比例；
- 低质量运动和严重抖动素材会被排除。

这说明他们很清楚视频生成模型最容易学坏的地方：  
**如果训练集中大量都是“动得不自然”或者“几乎不动”的视频，模型就很难学会有表现力的时空动力学。**

### 5.4 视觉文本数据

Wan 还特别强调 **visual text generation**，即让模型在视频中生成清晰、正确的中英文文字。

他们同时使用两类数据：

1. **合成数据**：在纯白底上渲染大量中文字符；
2. **真实数据**：从真实图像/视频中用 OCR 提取文字，再借助 Qwen2-VL 生成包含文字内容的自然描述。

这套思路很重要，因为视频里“生成可读文字”通常很难。  
Wan 的做法可以概括为：

> **用 synthetic text data 保 glyph 准确性，用 real text data 保场景自然性和分布一致性。**

---

## 6. 后训练数据与 Dense Caption

### 6.1 Post-training 数据

在 post-training 阶段，他们用更精选的数据提升：

- 视觉保真度
- 运动动态质量。

图像方面，会从高分数据中继续筛 top 20%，并人工补齐缺失概念。  
视频方面，会选取简单运动与复杂运动并重、类别均衡的数据。

### 6.2 Dense Video Caption 的作用

报告明确引用了 DALL·E 3 的经验：  
**更细致、更密集的 caption 能显著提升生成模型的 prompt-following 能力。**

所以他们训练了一个内部 caption model，为每个图像/视频生成 dense caption。

他们的 in-house caption 数据不仅有一般图文对，还有：

- 名人 / 地标 / 角色识别
- 物体计数
- OCR
- 相机角度与相机运动
- 细粒度类别
- 空间关系
- re-caption
- 编辑指令描述
- 多图组描述
- 人工高质量 caption。

这说明 Wan 不把 caption 当成“顺手配个文本”，而是把它当成**训练时的重要监督层**。  
其目标是把视频中的多个维度都说清楚，例如：

- 谁在做什么；
- 相机怎么动；
- 场景是什么；
- 风格是什么；
- 有没有文字；
- 颜色、数量、关系是什么。

---

## 7. Caption 模型本身怎么做

Wan 的 caption model 采用 **LLaVA-style 架构**：

- ViT 负责提取图像/视频帧视觉特征；
- 两层 MLP 做投影；
- Qwen LLM 负责语言生成。

### 7.1 图像处理
图像采用动态高分辨率输入，最多切成 7 个 patch，并把每个 patch 的视觉特征池化成 12×12 网格，以降低计算量。

### 7.2 视频处理
视频每秒采样 3 帧，最多 129 帧；  
使用 **slow-fast 编码策略**：

- 每 4 帧保留一帧的原始高分辨率；
- 其余帧做全局池化。

这个思路很实用，本质是在长视频理解里平衡：

- 时序覆盖范围
- token 数量
- 计算成本

### 7.3 三阶段训练
训练分三步：

1. 先冻结 ViT 和 LLM，只训练 MLP 对齐视觉与语言空间；
2. 再全部参数一起训练；
3. 最后在少量高质量数据上 end-to-end 微调。

### 7.4 自动评估
他们还做了 caption 自动评测管线，评估十个维度：

- action
- camera angle
- camera motion
- category
- color
- counting
- OCR
- scene
- style
- event。

结果上，他们的 caption 模型在 **event、camera angle、camera motion、style、color** 等维度优于 Gemini 1.5 Pro，而 Gemini 在 action、OCR、scene 等方面更强。

---

## 8. Wan-VAE：为什么它是整篇报告的关键基础设施

对于视频生成模型来说，VAE 不只是“压缩器”，而是决定：

- latent 是否足够紧凑；
- 时序信息是否能保住；
- 训练和推理是否能扩展到长视频；
- 重建质量是否足够高。

Wan 提出的 VAE 是一个 **3D causal VAE**，报告中称为 **Wan-VAE**。

### 8.1 输入输出与压缩比

给定输入视频 `V ∈ R^{(1+T)×H×W×3}`，Wan-VAE 会把它压到：

- 时间维：`1 + T/4`
- 空间维：`H/8, W/8`
- 通道维：扩成 `C=16`。

也就是说，其压缩比可以理解为：

- 时间压缩 4 倍
- 空间压缩 8×8 倍。

### 8.2 为什么强调 causal

视频生成尤其是长视频、流式视频时，必须保证**未来帧不能泄漏给过去帧**。  
因此他们强调 temporal causality，并为此：

- 把 GroupNorm 换成 RMSNorm；
- 配合 feature cache 机制提升推理效率。

### 8.3 设计细节上的优化

报告提到几项具体设计：

- 用 RMSNorm 保护因果性；
- 空间上采样层把输入通道减半，降低推理显存；
- 把模型规模控制在 **127M 参数**。

这意味着作者不是盲目追求“更大 VAE”，而是在：

- 重建质量
- 压缩率
- 推理速度
- 显存占用

之间做了精细平衡。

### 8.4 Wan-VAE 的训练方式

Wan-VAE 训练分三阶段：

1. 先训练同结构的 2D image VAE；
2. 再 inflate 成 3D causal VAE，用低分辨率、少帧视频加速收敛；
3. 最后在高质量、多分辨率、不同帧数视频上微调，并引入 GAN loss。

loss 包括：

- L1 reconstruction
- KL loss
- LPIPS perceptual loss
- 最后阶段再加 GAN loss。

### 8.5 Feature Cache：支持长视频的关键技巧

报告中一个很有价值的工程设计是 **feature cache mechanism**。  
做法是：

- 把长视频按 latent chunk 切分；
- 编码 / 解码时每次只处理少量帧；
- 保存前一 chunk 的必要历史特征；
- 在后续 chunk 的 causal convolution 中复用这些缓存。fileciteturn0file0L347-L359

这样做的效果是：

- 避免长视频直接进 VAE 导致显存爆炸；
- 保持 chunk 边界处的时序连续性；
- 支持“任意长视频”稳定编码解码。fileciteturn0file0L347-L361

### 8.6 Wan-VAE 的效果

报告用 PSNR 和速度来评估多个视频 VAE，结论是：

- Wan-VAE 在重建质量和效率上都很有竞争力；
- 在相同硬件环境下，其重建速度是 Hunyuan Video VAE 的 **2.5 倍**。fileciteturn0file0L362-L376

并且从图示和文字描述看，Wan-VAE 在以下场景表现较好：

- 纹理细节
- 人脸
- 文本
- 高运动场景。fileciteturn0file0L376-L385

---

## 9. Wan 的主干生成模型：DiT + Flow Matching

Wan 的基础 T2V 模型由三部分组成：

1. **Wan-VAE**
2. **Diffusion Transformer**
3. **Text Encoder（umT5）**。fileciteturn0file0L386-L392

### 9.1 DiT 主干是怎么处理视频的

视频先经过 VAE 编到 latent，再通过 3D patchify 变成 token 序列，送入 Transformer。  
每个 block 中：

- 用 self-attention 建模时空关系；
- 用 cross-attention 注入文本条件；
- 用时间步嵌入调制 block。fileciteturn0file0L393-L404

报告还提到一个细节：  
它们把时间嵌入经 MLP 后预测 6 个 modulation 参数，而且这个 MLP 在所有 block 中共享，仅让每个 block 学不同 bias。这样做可以减少约 **25% 参数量**，同时提升效果。fileciteturn0file0L402-L407

这是一个很值得记的设计点：

> **不是所有层都要完全独立；适当共享条件调制网络，能提升参数效率。**

### 9.2 为什么选 umT5 作为文本编码器

他们最终选了 **umT5**，理由是：

1. 多语言能力强，适合中英双语与视觉文本；
2. 在相同条件下，比部分单向注意力 LLM 在组合性上更强；
3. 收敛更快。fileciteturn0file0L408-L412

这也说明 Wan 对“视频中中英文文字生成”这件事是从 text encoder 层面就开始考虑的。

---

## 10. Flow Matching 训练目标

Wan 采用 **flow matching / rectified flow** 风格目标来训练视频生成主干。

给定真实 latent `x1`、高斯噪声 `x0` 和时间步 `t`，中间状态定义为：

$
x_t = t x_1 + (1-t) x_0
$

对应的目标速度为：

$
v_t = \frac{d x_t}{dt} = x_1 - x_0
$

模型学习预测速度场，loss 是预测速度与真实速度之间的 MSE。

这一点很关键，因为它说明 Wan 不是传统 DDPM 的 epsilon-prediction 表述，而是站在 **flow / velocity prediction** 框架下。

---

## 11. 训练策略：为什么先做图像，再做图像+视频联合训练

报告的训练路线很有代表性。

### 11.1 先做低分辨率 T2I 预训练

作者发现，如果一开始就直接用高分辨率、长视频联合训练，会有两个问题：

- 序列太长，训练吞吐太低；
- 显存太大，batch 太小，训练不稳定。 ***为什么会不稳定***

所以他们先做 **256px 低分辨率文本到图像预训练**，先把：

- 文本-视觉对齐
- 基本几何结构理解

学起来。

### 11.2 再做 image-video joint training

之后采用分阶段联合训练：

1. 256px 图像 + 5 秒 192px 视频；
2. 升到 480px；
3. 最后升到 720px。 ***为什么分段学习会比较稳定***

这本质上是一个 **resolution-progressive curriculum**。  
好处在于：

- 先学基础语义与运动；
- 再逐步学高分辨率细节；
- 保证训练稳定与吞吐。

### 11.3 优化配置

预训练使用：

- bf16 mixed precision ***为什么要强调这个，还有其他设置方式吗***
- AdamW
- 初始学习率 1e-4
- 根据 FID 与 CLIP Score plateau 动态降低学习率。

---

## 12. 大规模训练系统：Wan 的工程价值在哪里

这一节是很多博客容易略过，但实际上很有价值的部分。

### 12.1 训练瓶颈主要在 attention

报告分析 DiT 训练开销时指出：

- DiT 占总训练计算量 **85% 以上**；
- 当序列很长时，attention 的计算成本按 `O(s^2)` 增长；
- 在百万 token 级别时，attention 可能占到端到端训练时间的 **95%**。fileciteturn0file0L444-L455

所以长视频生成的根本瓶颈并不神秘：

> **就是超长时空 token 上的注意力计算。**

### 12.2 显存压力也非常夸张

报告估计：14B DiT 在 1M token、micro-batch=1 的情况下，仅 activation 显存需求就可能超过 **8TB**。fileciteturn0file0L456-L459

这也解释了为什么视频大模型必须高度依赖复杂并行策略与内存优化。

### 12.3 并行策略：FSDP + 2D Context Parallel

Wan 的并行设计中：

- VAE 可以简单 DP；
- Text encoder 需要 DP + FSDP；
- DiT 则重点使用 **FSDP + Context Parallelism (CP)**。fileciteturn0file0L460-L469

更具体地说，他们设计了 **2D Context Parallelism**：

- 外层是 Ring Attention
- 内层是 Ulysses。fileciteturn0file0L470-L476

其目标是：

- 减少跨机通信开销；
- 提升长序列分片下的效率；
- 保持更好的通信-计算重叠。fileciteturn0file0L471-L476

### 12.4 激活卸载与梯度检查点

由于长序列下计算时间已经很长，CPU/GPU 间 offload 的 PCIe 时间有机会被计算隐藏，因此作者优先使用 **activation offloading** 降低显存，而不是只用 gradient checkpointing。  
同时再结合 checkpointing 处理高显存占比层。fileciteturn0file0L480-L484

这个判断很工程化：  
在某些 regime 下，**offload 并不一定拖慢整体训练**，反而能更优雅地解决显存问题。

---

## 13. 推理优化：怎么把大模型真正跑快

Wan 在推理端主要做了三类事：

1. 并行推理；
2. diffusion cache；
3. 量化。fileciteturn0file0L485-L499

### 13.1 并行推理
推理时继续使用 FSDP + 2D Context Parallel，使 Wan 14B 能在多 GPU 上接近线性加速。fileciteturn0file0L489-L494

### 13.2 Diffusion Cache

报告观察到两个现象：

- 同一个 DiT block 在不同 sampling step 的 attention 输出很相似；
- 在采样后期，CFG 中 conditional / unconditional 输出也很相似。fileciteturn0file0L495-L503

因此他们缓存：

- attention 结果
- unconditional 分支结果

并在部分 step 复用。  
结果是 14B T2V 模型的推理效率提升 **1.62×**。fileciteturn0file0L501-L508

### 13.3 量化

他们尝试了：

- **FP8 GEMM**：DiT 模块提速约 **1.13×**；
- **8-bit FlashAttention**：整体再提速 **1.27× 以上**。fileciteturn0file0L509-L526

这里可以看出 Wan 在推理优化上不是单点技巧，而是把：

- 结构缓存
- 低精度矩阵乘
- 低精度注意力

组合使用。

---

## 14. Prompt Alignment：为什么用户短 prompt 往往不够好

作者指出，用户输入通常很短，而训练时的 dense caption 往往很长、很细致。  
这会带来 **train-test distribution mismatch**。

因此他们用 LLM 对用户 prompt 做重写，原则包括：

1. 不改变原意，只补充细节；
2. 补充自然运动属性；
3. 尽量把 prompt 改写成接近训练 caption 的风格。

用一句话总结：

> **Wan 把 prompt rewriting 当作推理时的重要“分布对齐”模块。**

---

## 15. Wan-Bench：他们如何评估视频生成

报告认为传统 FVD / FID 之类指标与人类感知不够一致，因此提出了 **Wan-Bench**。

它分成三大类：

- dynamic quality
- image quality
- instruction following。

细分指标一共 14 个，包括：

- large motion generation
- human artifacts
- physical plausibility
- smoothness
- pixel-level stability
- ID consistency
- comprehensive image quality
- scene generation quality
- stylization
- single/multiple object
- spatial positions
- camera control
- action instruction following。

评价器也不是单一模型，而是混合使用：

- 传统检测器
- optical flow
- DINO
- CLIP
- Qwen2-VL
- 人类反馈加权。

这里最值得注意的是最后一步：  
他们收集了 5000+ 人类成对偏好比较，用 Pearson correlation 来给不同维度赋权，得到最终综合分。

这说明 Wan-Bench 不是简单平均，而是想更接近“用户真正关心什么”。

---

## 16. Wan 的主要结果

### 16.1 Wan-Bench 结果
在报告给出的表格中，Wan 14B 的加权分数最高，为 **0.724**，优于对比模型；Wan 1.3B 也具有竞争力。fileciteturn0file0L583-L597

### 16.2 Human Evaluation
在人类评测中，Wan 14B 在 T2V 任务上在视觉质量、运动质量、匹配度、整体质量等维度上都表现强势。fileciteturn0file0L598-L609

### 16.3 VBench
报告还说 Wan 14B 在 VBench 上取得了 **86.22%** 的总分，Wan 1.3B 也达到了 **83.96%**。fileciteturn0file0L604-L617

这说明作者想表达的不是“只有一个大模型很强”，而是：

- 大模型追求 SOTA；
- 小模型追求可用性与性价比。

---

## 17. 消融实验：从 Wan 的选择中能学到什么

### 17.1 AdaLN 共享比不共享更好
他们比较了不同 adaLN 配置，发现与其把参数砸在每层独立 adaLN 上，不如把 adaLN 共享并增加网络深度，效果更好。fileciteturn0file0L618-L635

这个结论很有启发性：

> **条件调制层不一定越“独立”越好，参数预算可能更应该投给主干深度。**

### 17.2 umT5 比一些强 LLM 编码器更合适
他们比较了 umT5、Qwen2.5-7B-Instruct、GLM-4-9B 等，发现 umT5 的训练损失表现更好。  
报告给出的解释之一是：T5 的双向注意力比 decoder-only LLM 更适合 diffusion 模型。fileciteturn0file0L636-L646

### 17.3 传统 VAE 优于 VAE-D
他们还比较了普通 VAE 与一个 diffusion-loss 版本的 VAE-D，结果 VAE 的 FID 更低。fileciteturn0file0L647-L651

---

## 18. 扩展应用一：Image-to-Video（I2V）

Wan-I2V 的核心思路是：  
把条件图像作为首帧，通过 mask 和条件 latent 让模型知道：

- 哪些帧是已知的；
- 哪些帧需要生成。fileciteturn0file0L652-L666

此外还加入了：

- CLIP 图像编码器提取全局图像特征；
- decoupled cross-attention 注入全局 context。fileciteturn0file0L659-L666

更重要的是，这套 mask 框架不只支持 I2V，还能扩展到：

- 视频续写
- 首尾帧过渡
- 随机帧插值。fileciteturn0file0L666-L669

数据构建上，他们会用 SigLIP 过滤：

- 首帧与视频内容差异过大的样本；
- 首段与末段不够连续的视频。fileciteturn0file0L670-L681

这说明 I2V 的关键不是“只给一张图就完了”，而是要让训练数据本身满足：

- 首帧能代表视频主要内容；
- 视频时间延续性强。

---

## 19. 扩展应用二：统一视频编辑（VACE）

Wan 报告中把视频编辑放在一个更通用的统一框架里，即 **VACE**。  
它统一支持：

- reference-to-video
- video-to-video editing
- masked editing
- 多条件控制等任务。fileciteturn0file0L682-L706

### 19.1 Video Condition Unit (VCU)
VCU 把输入统一成三部分：

- 文本 `T`
- 帧序列 `F`
- mask 序列 `M`。fileciteturn0file0L706-L715

### 19.2 Concept Decoupling
一个很有意思的设计是把输入帧按 mask 拆成：

- `Fc = F × M`：需要修改的区域
- `Fk = F × (1-M)`：需要保留的区域。fileciteturn0file0L716-L725

这样可以让模型明确知道：

- 哪些内容该保持；
- 哪些内容该编辑。

这个思路很适合写成一句博客里的总结：

> **统一编辑框架的关键不是“塞更多条件”，而是先把条件语义拆清楚。**

### 19.3 两种训练模式
他们支持：

1. 完全微调 Wan；
2. 加 Context Adapter，不改原始主干参数。fileciteturn0file0L700-L706

这意味着它既能追求最好性能，也能追求更灵活的插件式适配。

---

## 20. 扩展应用三：Text-to-Image

由于 Wan 在训练时做了大规模 image-video joint training，因此它不仅能生视频，也能生图。  
报告强调，图像数据量接近视频数据的十倍，因此图像与视频任务之间存在明显的 cross-modal transfer。fileciteturn0file0L742-L747

换句话说，Wan 不是单纯“顺便支持生图”，而是在统一框架下把图像生成也做成了主能力之一。

---

## 21. 扩展应用四：个性化视频生成

这一部分的目标是：  
**给一张或几张用户参考脸，生成保持身份一致的视频。**fileciteturn0file0L748-L754

Wan 的一个关键选择是：

- 不依赖 ArcFace、CLIP 这类外部 identity extractor 作为唯一身份表达；
- 而是直接把参考人脸图像放到 Wan-VAE 的 latent 空间中作为条件。fileciteturn0file0L755-L764

作者的理由是：

- ArcFace 更偏人脸识别，不一定保留所有细节；
- 通用视觉编码器又往往太粗粒度。fileciteturn0file0L755-L761

在结构上，他们通过在时间维前面 prepend 若干参考脸帧，再用自注意力统一建模。fileciteturn0file0L764-L770

这套方案的优点是：

- 条件与生成目标处在同一 latent 空间；
- 不必强行把 identity 压成单一 embedding；
- 更容易保留细粒度身份特征。

---

## 22. 扩展应用五：相机运动控制

Wan 也支持 camera motion controllability。  
其做法是把每帧相机的：

- extrinsic `[R, t]`
- intrinsic `K`

转为更细粒度的位置表征，再通过 camera pose encoder 编码，最后用 adaptive normalization 注入 DiT。fileciteturn0file0L776-L786

可以理解为：  
**不是只在 prompt 里说“镜头推进”，而是显式提供相机轨迹条件。**

这对更可控的视频生成非常重要。

---

## 23. 扩展应用六：实时视频生成

这一部分非常值得关注，因为它展示了视频生成从“高质量离线生成”走向“实时交互系统”的方向。

### 23.1 为什么要做实时
报告认为当前视频生成普遍太慢，很难用于：

- 实时交互
- 游戏
- 直播
- 虚拟世界模拟。fileciteturn0file0L787-L798

### 23.2 Streamer：把固定长度视频生成改成流式生成
他们提出的核心机制是 **sliding temporal window**：

- 在一个滑动窗口里做 token 去噪；
- 每次生成最左侧低噪 token 后，将其出队；
- 再在右侧加入新的高斯噪声 token；
- 这样窗口长度固定，但视频可以无限延长。fileciteturn0file0L799-L818

这相当于把普通“生成一个固定 5 秒视频”的模型，改成了：

> **持续滚动地产生后续帧的流式世界模型。**

### 23.3 一致性蒸馏进一步加速
为了达到实时，Streamer 进一步结合了 **LCM / VideoLCM** 思想，把原始 diffusion 过程蒸馏为更少步数的 consistency model。  
报告称这样可以带来 **10–20×** 加速，使系统达到 **8–16 FPS**。fileciteturn0file0L819-L829

### 23.4 消费级设备上的量化部署
他们还讨论了：

- int8 quantization
- TensorRT quantization。fileciteturn0file0L830-L836

其中 TensorRT 量化可以让系统在单张 4090 上达到约 **8 FPS**，但会带来一定误差和质量损失。fileciteturn0file0L833-L838

---

## 24. 扩展应用七：视频到音频（V2A）

Wan 的最后一个扩展方向是给视频自动生成同步音频。  
目标是合成：

- 环境音
- 背景音乐

但不包含语音 / 人声。fileciteturn0file0L839-L846

### 24.1 为什么不用 mel-spectrogram + 图像 VAE
他们指出，若把音频转成 spectrogram 再 patchify，可能破坏与视频的严格时间对齐。  
因此他们训练了一个 **1D-VAE** 直接压缩原始 waveform。fileciteturn0file0L847-L856

### 24.2 多模态对齐方式
- 用 CLIP 提取逐帧视觉特征；
- 复制 / 对齐到音频采样率；
- 再和音频 latent 进行融合；
- 文本则仍用 umT5 编码。fileciteturn0file0L856-L862

### 24.3 数据构建
他们从视频数据中筛选出带音轨但不含 speech/vocal 的样本，并用 Qwen2-audio 生成：

- 环境音描述
- 音乐描述。fileciteturn0file0L863-L867

### 24.4 局限
目前方法不擅长生成人声，例如笑声、哭声、说话声，因为训练数据中刻意去掉了这部分。fileciteturn0file0L868-L874

---

## 25. Wan 的局限

报告最后也比较坦诚地指出了几个局限：

### 25.1 大运动场景细节仍难保持
虽然 Wan 在大幅运动上有提升，但剧烈运动中的细粒度细节保真仍然困难。fileciteturn0file0L875-L881

### 25.2 大模型推理成本仍高
14B 模型在没有额外优化时，单张高端 GPU 推理大约仍需 **30 分钟**。fileciteturn0file0L881-L883

### 25.3 专业领域能力不足
作为通用基础模型，它在教育、医疗等专业场景下仍可能缺乏足够领域知识。fileciteturn0file0L883-L885

---

## 26. 我对这篇报告的理解与总结

如果把这篇 technical report 压缩成一句话，我会这样概括：

> **Wan 不只是“一个视频生成模型”，而是一套完整的视频基础模型工程体系。**

它的价值主要体现在以下几个层面。

### 26.1 它不是只堆模型，而是系统工程
很多文章只会强调“某个网络结构更强”。  
但 Wan 明显是从系统角度来做：

- 数据怎么清洗
- caption 怎么构造
- VAE 怎么设计
- DiT 怎么训练
- 长序列怎么并行
- 推理怎么缓存与量化
- 下游应用怎么统一。fileciteturn0file0L168-L171 fileciteturn0file0L444-L484

### 26.2 它很强调“可扩展性”
Wan-VAE、DiT、2D Context Parallel、Streamer，这些设计都在服务一个核心目标：

> **让视频生成模型不仅能生成好看的短视频，还能走向更长、更实时、更可控的场景。**

### 26.3 它体现了“统一视频基础模型”的趋势
从 T2V 到 I2V，再到 editing、personalization、camera control、audio generation，Wan 的扩展说明未来视频模型很可能会向一个统一底座发展，而不是每个任务单独训练一套模型。fileciteturn0file0L652-L874

---

## 27. 适合复习时重点记住的结论

### 27.1 一句话速记版
- **数据质量决定上限**：Wan 非常重视清洗、聚类筛选、motion quality 与 dense caption。  
- **视频 VAE 是根基**：Wan-VAE 的 3D causal + feature cache 是长视频可扩展的关键。  
- **主干路线是 DiT + Flow Matching**。  
- **训练先图像后视频、先低分辨率后高分辨率**。  
- **训练瓶颈主要是长序列 attention**。  
- **推理优化靠缓存、量化、并行组合拳**。  
- **统一底座可以扩到 I2V、编辑、个性化、实时、音频等多任务**。  

### 27.2 最值得抄到自己的研究思路里的几点
1. **caption 不是附属品，而是关键 supervision。**  
2. **高质量 motion data 对视频生成极其关键。**  
3. **VAE 的设计直接影响长视频建模上限。**  
4. **长序列生成不能只看模型结构，还要看并行与内存系统设计。**  
5. **统一任务框架比为每个任务单独造模型更有扩展性。**

---

## 28. 参考原文阅读建议

如果你之后打算继续深入读原报告，建议优先按下面顺序看：

1. **第 4.1 节 Wan-VAE**：最关键的基础设施。  
2. **第 4.2 节主干模型训练**：理解 DiT + Flow Matching。  
3. **第 3 节数据流水线**：理解为什么它能做 strong prompt-following 和 visual text。  
4. **第 4.3/4.4 节训练与推理优化**：适合做系统方向复习。  
5. **第 5 节扩展应用**：理解 unified video foundation model 的意义。fileciteturn0file0L25-L67

---

## 29. 博客标签建议

```text
Video Generation, Diffusion Transformer, Flow Matching, Video VAE, Wan, Image-to-Video, Video Editing, Personalized Video, Real-time Video Generation
```

---

## 30. 原文信息

- 标题：**Wan: Open and Advanced Large-Scale Video Generative Models**
- 团队：Wan Team, Alibaba Group
- arXiv 版本：**arXiv:2503.20314v2**
- 报告中提到代码与模型已开源。fileciteturn0file0L1-L24

