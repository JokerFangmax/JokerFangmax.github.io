---
title: "MITS.6184: 6.从 U-Net 到 DiT，再到 VAE：Stable Diffusion 3 背后的完整工程栈"
date: 2026-03-19
tags:
  - Diffusion Model
  - U-Net
  - DiT
  - VAE
  - Stable Diffusion 3
categories:
  - Generative Models
description: "从网络结构与 latent space 出发，理解现代文生图/文生视频模型为什么会采用 U-Net、DiT、VAE 与多模态条件编码。"
---

# 从 U-Net 到 DiT，再到 VAE：Stable Diffusion 3 背后的完整工程栈

前面几篇我们主要都在讲“原理”：

- 生成模型本质上是在学一个分布
- flow / diffusion 用连续动态过程来做采样
- flow matching / score matching 给出了训练目标
- classifier-free guidance 让模型更贴 prompt

但如果你真的去看现代文生图、文生视频系统，会发现它们远不只是一个“训练目标”那么简单。  
一个可用的大模型系统，通常至少包含下面几层东西：

1. **条件表示**：时间 $t$、文本 prompt $y$ 怎么编码  
2. **生成主干网络**：U-Net 还是 Transformer / DiT  
3. **数据空间选择**：是在原始像素空间做，还是在 latent space 做  
4. **采样策略**：Euler、SDE/ODE 模拟、CFG scale 等  
5. **完整系统拼装**：文本编码器 + latent autoencoder + backbone + guidance

这一篇就从工程视角把这些组件串起来。  
目标不是把每个实现细节都展开，而是让你明白：

> **现代 Stable Diffusion / Movie Gen 这一类系统，到底是怎样由前面学过的那些原理拼出来的。**

---

# 1. 先记住一句总纲：现代生成系统 = 训练目标 + 条件编码 + backbone + latent space

从讲义的组织方式看，这一章其实是在回答一个非常现实的问题：

> 前面那些概率路径、向量场、score function，最终到底由谁来实现？

答案就是：

- 用神经网络去表示 $u_t^\theta(x|y)$ 或相应的 score/noise predictor
- 用时间嵌入、prompt 嵌入来注入条件
- 用 U-Net 或 DiT 作为主干结构
- 为了节省计算与内存，在 autoencoder 的 latent space 里训练生成模型
- 再用具体大模型系统把这些模块组合起来 

所以从工程角度看，现代生成模型不是某一个单独算法，而更像是一整套模块化系统。

---

# 2. 条件变量首先得能“喂进网络”：时间嵌入与 prompt 嵌入

不管你用的是 Flow Matching 还是 DDPM，模型输入都不只是当前样本 $x$。  
至少还会有两个条件：

- 当前时刻 $t$
- 外部条件 $y$（例如文本 prompt、类别标签）

因此，第一步总是要把这些变量编码成网络能处理的向量表示。

---

## 2.1 时间嵌入：告诉模型“现在是第几步”

讲义在 DiT 部分把时间嵌入记成：

$
\tilde t = \mathrm{TimeEmb}(t)\in\mathbb R^d
$

也就是说，一个标量时间 $t$ 会被映射成一个 $d$ 维向量，再送进主干网络。

为什么这一步是必要的？

因为模型在不同时间步要做的事完全不一样：

- 在接近纯噪声时，要先恢复全局结构
- 在接近终点时，要做更细致的局部修正

所以模型必须知道“现在走到哪一步了”，才能输出合适的更新方向。

从直觉上讲，时间嵌入就是在告诉模型：

> **当前输入 $x_t$ 是“很脏的噪声”，还是“已经快干净了的样本”。**

---

## 2.2 Prompt 嵌入：把文本变成可计算的条件向量

类似地，条件变量 $y$ 也要编码。  
讲义在 DiT 中记作：

$
\tilde y=\mathrm{PromptEmb}(y)\in\mathbb R^{S\times d}
$

这里 $S$ 是 token 序列长度，$d$ 是隐藏维度。:contentReference[oaicite:3]{index=3}

这说明现代条件生成往往不是把整段文本压缩成一个单独向量，而是：

> **把 prompt 编成一个 token 序列，让模型能按位置去关注文本中的不同部分。**

这对复杂 prompt 尤其重要。  
比如一句话里既有主体、又有动作、又有背景、又有风格描述，如果只压成一个向量，很多细节会丢掉；而 token 级表示则允许模型在生成不同区域时，关注不同词语。

这也为后面 DiT 里的 **cross-attention** 做了准备。

---

# 3. 两大主干网络：U-Net 与 DiT

讲义把现代生成主干大致分成两类：

- **U-Net**
- **Diffusion Transformer（DiT）** 

你可以把它们理解成两种不同的“函数逼近器”，都在实现类似的目标：

$
u_t^\theta(x|y)
$

或者等价地实现 score / noise predictor。

它们最根本的区别，不在训练目标，而在于：

> **怎么处理输入张量、怎么建模空间关系、怎么融合条件信息。**

---

# 4. U-Net：为什么早期扩散模型几乎都用它？

讲义指出，U-Net 原本是为图像分割设计的一类卷积网络，它的一个关键特点是：

> **输入和输出都具有图像形状。** :contentReference[oaicite:5]{index=5}

这使它特别适合做 diffusion / flow 模型里的向量场预测：

- 输入是一个 noisy image $x_t$
- 输出也是与图像同形状的张量 $u_t^\theta(x|y)$

也就是说，从张量形状上看，这件事非常自然。

---

## 4.1 U-Net 的基本结构：编码器 + 中间块 + 解码器

讲义给出了一个典型例子。  
假设输入是一张

$
x_t^{\text{input}}\in\mathbb R^{3\times 256\times 256}
$

的图像，那么它在 U-Net 里大致经历：

$
x_t^{\text{input}}\in\mathbb R^{3\times256\times256}
\;\to\;
x_t^{\text{latent}}=E(x_t^{\text{input}})\in\mathbb R^{512\times32\times32}
$

再经过中间处理块：

$
x_t^{\text{latent}}=M(x_t^{\text{latent}})\in\mathbb R^{512\times32\times32}
$

最后通过解码器恢复回输出图像：

$
x_t^{\text{output}}=D(x_t^{\text{latent}})\in\mathbb R^{3\times256\times256}.
$

讲义还特别说明：在下采样过程中，通道数会增加、空间分辨率会减小；而解码阶段则反过来恢复空间分辨率。编码器和解码器之间通常还会有 skip / residual connections。:contentReference[oaicite:6]{index=6}

所以你可以把 U-Net 理解成：

- 先把图像压到低分辨率高通道表示，提取高层语义
- 再逐步把这些信息恢复回原图大小
- 同时利用跳连保留局部细节

---

## 4.2 为什么 U-Net 很适合早期 diffusion？

因为图像生成里有两类信息都很重要：

1. **全局语义**：这是一只狗、一辆车、还是一个房间  
2. **局部细节**：边缘、纹理、局部结构、颜色分布

U-Net 正好很擅长同时处理这两类信息：

- 编码器负责聚合全局信息
- 解码器负责恢复空间细节
- skip connections 则避免低层细节在压缩中丢失

所以早期大多数 diffusion 模型都大量采用 U-Net。讲义也明确说，U-Nets 在早期 diffusion literature 中被广泛使用。:contentReference[oaicite:7]{index=7}

---

# 5. DiT：为什么后来 Transformer 也开始主导生成模型？

随着模型规模越来越大、条件越来越复杂，Transformer 开始进入 diffusion 领域。  
这类模型在讲义里被称为 **diffusion transformer (DiT)**。

它的基本思路来源于 ViT：

> 不再把图像当成一个卷积特征图来处理，  
> 而是先切成 patches，再把 patch 当成 token 序列送进 Transformer。

---

## 5.1 Patchify：先把图像切成 patch token

讲义把 patchify 写成：

$
\mathrm{Patchify}(x)\in\mathbb R^{N\times C'}
$

其中：

- $P$ 是 patch size
- $N=(H/P)\cdot(W/P)$ 是 patch 数量
- $C'=CP^2$ 是每个 patch 展平后的维度

然后再乘一个可学习矩阵，得到 patch embedding：

$
\mathrm{PatchEmb}(x)=\mathrm{Patchify}(x)W\in\mathbb R^{N\times d}
$

此时，每个 patch 都变成了一个 $d$ 维 token。:contentReference[oaicite:9]{index=9}

所以 DiT 的输入实际上是三类对象：

$
\tilde t=\mathrm{TimeEmb}(t)\in\mathbb R^d,\qquad
\tilde y=\mathrm{PromptEmb}(y)\in\mathbb R^{S\times d},\qquad
\tilde x_0=\mathrm{PatchEmb}(x)\in\mathbb R^{N\times d}.
$

这一步非常关键，因为它把：

- 图像 patch
- 时间信息
- 文本条件

都放进了同一个“隐藏维度 $d$”的表示空间。:contentReference[oaicite:10]{index=10}

---

## 5.2 DiT block 里到底在做什么？

讲义对 DiTBlock 的总结非常清楚：它主要包含三种机制：

1. **self-attention on patches**  
2. **cross-attention to the prompt**  
3. **time conditioning via adaptive normalization (AdaLN)** 

所以可以这样理解：

### 第一层：patch 之间先彼此交流
自注意力让不同空间位置的 patch 彼此通信，建模全局依赖。

### 第二层：patch 再去看 prompt
交叉注意力让图像 token 可以从文本 token 中读取相关条件信息。

### 第三层：时间嵌入调节整个 block 的行为
AdaLN 用时间嵌入生成 scale/shift 参数，调节当前层在当前时间步的行为。:contentReference[oaicite:12]{index=12}

这三步合起来，DiT 就实现了：

> **在全局注意力框架下，把图像结构、文本条件、时间阶段统一融合。**

---

## 5.3 为什么 DiT 越来越重要？

因为与 U-Net 相比，DiT 有几个很明显的优势：

### 第一，更适合大规模扩展
Transformer 往往更容易随着参数量增大继续提升性能。

### 第二，更适合复杂条件融合
特别是文本 prompt 是序列时，cross-attention 是非常自然的接口。

### 第三，更擅长全局关系建模
对于长程依赖、复杂布局、多对象组合，attention 往往比纯卷积更灵活。

讲义虽然没有做一长串对比表，但从它把 Stable Diffusion 3 的主干放到 MM-DiT 来讲，也很能看出这一趋势。

---

# 6. 但有个更现实的问题：为什么不直接在像素空间训练？

到目前为止，我们一直默认输入是图像 $x$。  
但现代大模型其实很少直接在原始像素空间做 diffusion / flow。  
原因很简单：

> **像素空间太大、太贵、太冗余。**

尤其是高分辨率图像和视频：

- 维度极高
- 内存消耗大
- 计算成本大
- 许多高频细节其实对语义生成不重要

所以现代方法几乎都采用 **latent diffusion / latent flow** 范式：  
先用 autoencoder 把图像压到一个更低维、更语义化的 latent space，再在 latent 上训练生成模型。

---

# 7. VAE / Autoencoder：为什么它是现代生成系统的“前置模块”？

讲义在 latent space 部分强调：

> 要训练一个 latent generative model，只需要遵循原来的训练流程，但直接在 latent space 里工作。  
> 训练时从 $q_\phi(z|x)$ 采样 latent，推理时由 latent diffusion / flow model 产生 $z$，再用 decoder 解码回 $x$。:contentReference[oaicite:15]{index=15}

这句话其实已经概括了整个 latent diffusion pipeline：

1. 用 encoder 把原图 $x$ 编到 latent $z$
2. 在 latent 上训练 flow/diffusion 模型
3. 推理时先生成 latent，再由 decoder 还原成图像

---

## 7.1 VAE 的训练目标：重建 + KL 约束

讲义给出的 $\beta$-VAE 训练过程包括两项核心损失：

### Reconstruction loss
$
L_{\text{recon}}
\propto
\|x-\hat x\|^2
$

用来保证 decoder 能把 latent 恢复成原图。:contentReference[oaicite:16]{index=16}

### KL loss
$$
L_{\text{KL}}=
\frac12\sum_j
\big(
\mu_{i,j}^2+\sigma_{i,j}^2-\log \sigma_{i,j}^2-1
\big)
$$

用来把编码后的 latent 分布往简单先验（通常标准高斯）拉近。

总损失是：

$
L = L_{\text{recon}}+\beta L_{\text{KL}}.
$

这意味着 VAE 同时做两件事：

- 保证 latent 里保留足够信息，能重建输入
- 保证 latent 分布别太乱，尽量靠近简单先验，便于后续建模

---

## 7.2 为什么要先训练 autoencoder，再训练 diffusion/flow？

讲义里有一句非常重要的话：

> 现在几乎所有最先进的图像和视频生成方法，都采用 latent diffusion paradigm；但要做到这一点，你必须先把 autoencoder 训练好。最终性能也取决于 autoencoder 是否能把图像压缩到好的 latent，并恢复出足够美观的结果。:contentReference[oaicite:18]{index=18}

这说明 latent diffusion 并不是“只靠 diffusion 模型自己就够了”。  
系统质量有两个关键来源：

1. **autoencoder 压得好不好、还原得好不好**
2. **latent generative model 生成得好不好**

所以现代生成系统其实是一个“两阶段工程”：

- 第一阶段：学一个好的潜空间
- 第二阶段：在这个潜空间上做强大的生成建模

---

## 7.3 为什么 latent space 往往更适合生成？

讲义给了一个非常好的直觉：

> 一个训练良好的 autoencoder 可以把高频或语义上不重要的细节过滤掉，让生成模型把容量集中在更重要、更感知相关的特征上。:contentReference[oaicite:19]{index=19}

你可以把它理解成：

- 像素空间里既有“语义”也有大量“局部微小噪声式细节”
- 但生成模型最难学的，其实是主体、布局、形状、风格这些高层特征
- autoencoder 先把这些“更本质的因素”抽出来
- 于是后面的 diffusion / flow 可以更专注于真正重要的生成难点

这也是为什么 latent diffusion 对大模型训练来说几乎是必选项。

---

# 8. 为什么不能只靠 VAE 自己生成，而还要再接一个 diffusion/flow model？

这个问题讲义也专门讨论了。  
理论上，VAE 自己当然也能做生成：

- 从先验 $p_{\text{prior}}(z)$ 采样
- 再用 decoder 生成 $x$

但讲义指出，实践中编码器聚合后的 latent 分布 $q_\phi(z)$ 并不一定真的等于先验 $p_{\text{prior}}(z)$，这里存在所谓 **amortization gap**。同时，decoder 训练时看到的是编码器产生的 latent 分布，而不是真正的 prior 采样。:contentReference[oaicite:20]{index=20}

更重要的是，讲义明确说：

> 实践证明，flow 和 diffusion 模型通常比 VAE decoder 这种卷积栈更有能力承担复杂的生成建模任务，因此把一部分生成复杂性“外包”给 latent generative model 是更合理的。:contentReference[oaicite:21]{index=21}

所以现代系统的分工通常是：

- **autoencoder** 负责压缩与还原
- **diffusion / flow** 负责真正强大的生成建模

这就是“VAE + latent diffusion”组合会长期存在的原因。

---

# 9. 把这些模块真正拼起来：Stable Diffusion 3

现在终于可以看具体案例了。

讲义在 case study 里对 Stable Diffusion 3 的总结非常清楚：

1. 它采用了本文讲过的 **conditional flow matching** 目标  
2. 作者比较了多种 flow/diffusion 方案后，认为 flow matching 最好  
3. 它使用 **classifier-free guidance** 训练  
4. 它在 **预训练 autoencoder 的 latent space** 中训练  
5. 它为了增强文本条件，使用了多种文本嵌入  
6. 它把原始 DiT 扩展成 **multi-modal DiT (MM-DiT)** 来同时处理图像 patch 和文本 token 

这几句话其实已经把前几篇的知识全部串起来了。

---

## 9.1 Stable Diffusion 3 用了哪些文本条件？

讲义提到，Stable Diffusion 3 使用三类文本嵌入，其中包括：

- **CLIP embeddings**
- **T5-XXL encoder 的 sequential outputs**

并指出：

- CLIP embedding 提供较粗粒度、整体性的文本语义
- T5 embeddings 提供更细粒度的上下文，使模型有机会关注 prompt 中的特定元素 

这点很重要，因为它反映了现代文本条件设计的一个趋势：

> **不再满足于“整句一个向量”，而是希望同时保留全局语义和局部 token 级细节。**

---

## 9.2 为什么要有 MM-DiT？

原始 DiT 更像是“图像 patch + 类别标签”的结构。  
但当条件变成一整串文本 token 时，模型需要能直接 attend 到文本序列。  
因此讲义说，Stable Diffusion 3 把 diffusion transformer 扩展到不仅能看图像 patch，还能看文本 embeddings，这个修改后的模型被称为 **MM-DiT**。

所以 MM-DiT 的本质，就是：

> **让图像 token 与文本 token 在同一个 Transformer 框架里交互。**

从工程角度说，这是一种非常自然的多模态融合方式。

---

## 9.3 Stable Diffusion 3 的采样长什么样？

讲义里给出的描述是：

- 使用 **50 steps**
- 使用 **Euler simulation**
- CFG 权重大约在 **2.0–5.0** 之间 

这说明实际系统里：

- 不需要无限小步
- 会选择一个相对可接受的离散步数
- guidance 仍然是非常核心的推理技巧

也就是说，从工程落地角度看，Stable Diffusion 3 其实就是：

> **latent autoencoder + MM-DiT backbone + conditional flow matching + CFG + Euler sampler**

---

# 10. 再往视频扩展：Meta Movie Gen

讲义接着讨论了 Meta 的 Movie Gen Video，并指出它很多设计本质上是把图像方法扩展到多了一个时间维度的视频空间。视频数据位于：

$
\mathbb R^{T\times C\times H\times W}
$

相比图像，多了时间轴 $T$。

---

## 10.1 训练目标仍然是 conditional flow matching

讲义明确说，Movie Gen Video 也采用 conditional flow matching，且使用同样的直线调度：

$
\alpha_t=t,\qquad \sigma_t=1-t.
$

也就是说，从训练目标角度，它和 Stable Diffusion 3 是同一条主线。:contentReference[oaicite:27]{index=27}

---

## 10.2 视频更离不开 latent space

讲义强调：对视频来说，autoencoder 降低内存的重要性甚至比图像更高，因为视频维度大得多，所以现在大多数视频生成器在时长上仍然受限。Movie Gen 采用一个 **temporal autoencoder (TAE)**，把原始视频压到 latent video 表示。:contentReference[oaicite:28]{index=28}

这说明 latent paradigm 在视频里更是刚需。

---

## 10.3 视频版的 DiT：同时 patchify 时间和空间

Movie Gen 的 backbone 仍然是 DiT-like 结构，但现在 patchify 不只是沿空间维度，而是沿**时间和空间维度一起**切块。然后：

- latent video patches 之间做 self-attention
- 再和语言模型 embeddings 做 cross-attention

这和 Stable Diffusion 3 的 MM-DiT 在思想上是一致的，只是从图像 token 扩展到了时空 token。:contentReference[oaicite:29]{index=29}

---

## 10.4 Movie Gen 的文本条件更丰富

讲义里提到 Movie Gen 使用三类文本嵌入：

- **UL2 embeddings**
- **ByT5 embeddings**
- **MetaCLIP embeddings**

它们分别对应不同层面的文本条件，例如更细粒度的文本推理、字符级细节，以及共享图文空间的语义表示。:contentReference[oaicite:30]{index=30}

这说明随着任务越来越复杂，条件编码也在不断变得更丰富、更分工明确。

---

# 11. 现在可以把整套工程栈压缩成一个统一图景了

如果你把前面几篇加上这一篇一起看，其实现代系统已经非常清楚了：

## 第一步：训练 autoencoder / VAE
把原始图像或视频压到 latent space，并保证可还原。

## 第二步：设计生成训练目标
通常是 flow matching、score matching 或 DDPM 风格目标。

## 第三步：选择 backbone
图像任务可以是 U-Net 或 DiT；大模型趋势更偏向 DiT / MM-DiT。

## 第四步：注入条件
把时间、文本、类别等编码成向量或 token，通过 AdaLN、cross-attention 等机制注入。

## 第五步：推理时用采样器 + CFG
例如 Euler steps + classifier-free guidance。

所以现代文生图/文生视频的本质并不是某个单独神秘组件，而是：

> **latent space + 条件编码 + 大主干网络 + 生成训练目标 + 推理启发式** 的组合系统。

---

# 12. 如果只想抓住 U-Net、DiT、VAE 三者的分工，应该怎么记？

这是最值得记的一张“脑内结构图”：

## U-Net
更偏卷积式图像处理，擅长多尺度局部结构恢复。  
早期 diffusion 模型大量使用。:contentReference[oaicite:36]{index=36}

## DiT / MM-DiT
更偏 Transformer，擅长全局依赖建模和复杂条件融合。  
现代大规模文生图、文生视频越来越常见。

## VAE / Autoencoder
不是直接负责“生成内容细节”的主干，而是先把数据压进一个更适合建模的 latent space。  
现代系统几乎都离不开它。

一句话概括就是：

> **VAE 决定“在哪个空间里生成”，U-Net/DiT 决定“用什么网络生成”。**

---

# 13. 本章小结

这一篇最核心的内容可以总结成下面几条：

1. 现代生成系统不仅有训练目标，还需要条件编码、backbone 与 latent space 设计。

2. U-Net 适合输入输出同为图像张量的任务，通过 encoder–midcoder–decoder 结构和 skip connections 同时保留语义与细节。:contentReference[oaicite:40]{index=40}

3. DiT 把图像切成 patch token，再通过 self-attention、cross-attention 与时间条件融合；MM-DiT 进一步把文本 token 也纳入统一 Transformer 框架。

4. 现代图像/视频生成几乎都采用 latent diffusion / latent flow：先训练 autoencoder，再在 latent space 中训练生成模型。

5. VAE 训练由 reconstruction loss 和 KL loss 组成，用来学一个既可重建、又较接近先验的潜空间。

6. Stable Diffusion 3 可以看成：pretrained autoencoder + conditional flow matching + CFG + MM-DiT + 多种文本嵌入。

7. Movie Gen 则把这一思路扩展到视频：temporal autoencoder + 时空 patchified DiT + richer text conditioning。:contentReference[oaicite:45]{index=45}

如果只记一句话，那就是：

> **现代生成大模型不是“某一个算法”，而是 latent autoencoder、条件编码、主干网络和生成训练目标拼出来的完整系统。**

---

# 14. 下一篇要讲什么？

到这里，连续空间里的生成模型主线已经差不多齐了。  
接下来最后还剩一个很自然的问题：

> 如果数据不是连续向量，而是离散 token，比如文本，那 diffusion 思想还能用吗？

下一篇我们就接着写：

# 《离散扩散模型入门：文本生成为什么也能用 diffusion 思想？》

---