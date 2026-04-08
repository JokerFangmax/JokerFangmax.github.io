---
title: "MITS.6184: 5.Classifier-Free Guidance 原理：生成模型为什么会更“听 prompt 的话”？"
date: 2026-03-19
tags:
  - Diffusion Model
  - Guidance
  - Classifier-Free Guidance
  - Conditional Generation
categories:
  - Generative Models
description: "从 vanilla guidance、classifier guidance 到 classifier-free guidance，理解生成模型为什么能更贴合 prompt。"
---

# Classifier-Free Guidance 原理：生成模型为什么会更“听 prompt 的话”？

前面几篇我们已经把无条件生成的核心逻辑讲清楚了：

- 用概率路径把噪声分布逐步连接到数据分布
- 用 flow matching 或 score matching 学到相应的局部更新规则
- 最终从噪声一步步生成样本

但现实里，大家更关心的通常不是“随便生成一张图”，而是：

- 给一句文本，生成对应图像
- 给一个类别标签，生成该类图像
- 给一段 prompt，让结果尽量贴合 prompt

这就进入了 **条件生成（conditional generation）** 的问题。

而现代文生图、文生视频模型里最重要的条件控制技巧，几乎就是 **classifier-free guidance（CFG）**。

很多人第一次见到 CFG 时，只记住了一个公式：

$
\tilde u_t(x|y)=(1-w)u_t(x|\emptyset)+wu_t(x|y)
$

然后知道“把 $w$ 调大，图会更贴 prompt”。  
但这个公式为什么长这样、它在放大什么、它和 classifier guidance 有什么关系，往往并没有真正理解。

这一篇就把这件事完整讲清楚。

---

# 1. 条件生成到底是什么？

在无条件生成里，我们的目标是从整体数据分布采样：

$
X_1 \sim p_{\text{data}}
$

而在条件生成里，目标变成：

$
X_1 \sim p_{\text{data}}(\cdot|y)
$

其中 $y$ 是条件变量，比如：

- 文本 prompt
- 类别标签
- 语义描述
- 图像编辑指令

也就是说，我们不是想“随便生成一个合理样本”，而是想生成一个：

> **既合理，又满足条件 $y$ 的样本。**

---

# 2. 最直接的做法：vanilla guidance / 直接条件建模

最朴素的想法其实很简单：

> 既然想生成满足条件 $y$ 的样本，那就把 $y$ 直接送进模型。

于是原来的无条件向量场 $u_t^\theta(x)$ 变成条件向量场：

$
u_t^\theta(x|y)
$

训练时，数据也不再只是样本 $z$，而是成对的数据 $(z,y)\sim p_{\text{data}}(z,y)$，即z和条件y有关

因此 guided conditional flow matching 的目标可以写成：

$
L_{\text{guided}}^{\text{CFM}}(\theta)=
\mathbb{E}_{(z,y)\sim p_{\text{data}}(z,y),\; t\sim \mathrm{Unif}[0,1],\; x\sim p_t(\cdot|z)}
\left[
\|u_t^\theta(x|y)-u_t^{\text{target}}(x|z)\|^2
\right].
$

这里要注意一个很关键的点：

- 条件 $y$ 决定的是我们想生成哪一类内容
- 但条件概率路径 $p_t(\cdot|z)$ 本身仍然是“围绕具体数据点 $z$”定义的
- 所以训练目标仍然是在回归 $u_t^{\text{target}}(x|z)$，只不过网络现在多看到了条件 $y$ (后面的目标速度中没有y是因为z和y是联合生成的，即z就已经隐含了是y的条件)

从实现角度说，这件事也很好理解：

- dataloader 不再只返回图像 $z$
- 而是返回 $(z,y)$
- 模型输入也不再只有 noisy sample 和时间 $t$
- 还包括条件 $y$

这就是最基础的 **vanilla guidance**。

---

# 3. 问题来了：为什么只做条件训练还不够？

按道理说，如果我们已经在学 $p_{\text{data}}(\cdot|y)$，那应该足够了。  
讲义里也明确说了：

> 从理论上，vanilla guidance 应该能忠实地生成 $p_{\text{data}}(\cdot|y)$ 的样本。

但现实里很快发现一个问题：

> **模型虽然“知道条件是什么”，但生成出来的样本往往还不够贴 prompt。**

例如：

- 文本说的是 “corgi dog”
- 结果图里确实有狗
- 但未必足够像柯基
- 或者文本里有多个细节，模型只满足了一部分

讲义里指出，vanilla guidance 在实践中经常“不够 fit 到 desired label $y$”；原因可能包括：

- 模型欠拟合，并没有学到真正的条件边缘向量场 ***为什么会欠拟合***
- 数据本身不完美，比如网页抓取的 text-image pairs 有错误
- 条件信号相对于图像先验来说太弱，推理时容易被“平均掉”  ***为什么会太弱，我要怎么解决***

于是一个自然的问题就出现了：

> 能不能在推理时，**人为增强 prompt 的作用**？

这就是 guidance 的本质动机。

---

# 4. Guidance 的核心思想：放大“与条件有关”的那一部分

先把最重要的直觉说出来：

生成模型里其实同时存在两种力量：

1. **无条件先验**：什么样的图像整体上看起来合理
2. **条件约束**：什么样的图像更符合当前 prompt $y$

如果只做普通条件训练，这两种力量会一起起作用。  
但很多时候，第二种力量不够强，于是结果会“看起来合理，但不够贴 prompt”。

所以 guidance 的核心想法就是：

> **把条件相关的部分额外放大。**

这个思路既可以通过外部 classifier 来做，也可以不借助 classifier，直接在模型内部完成。  
这就分别对应：

- **classifier guidance**
- **classifier-free guidance**

---

# 5. Classifier Guidance：先用贝叶斯分解看条件信息从哪来

讲义先从 Gaussian probability path 下的 classifier guidance 开始讲。  
为了说明直觉，它先把 guided vector field 写成依赖 guided score 的形式：

$
u_t^{\text{target}}(x|y)=a_t \nabla \log p_t(x|y) + b_t x
$

其中 $a_t,b_t$ 只和时间有关。

接着利用条件概率的分解：

$
p_t(x|y)=\frac{p_t(y|x)p_t(x)}{p_t(y)}
$

对数求梯度后得到：

$
\nabla \log p_t(x|y)=\nabla \log p_t(x)+\nabla \log p_t(y|x)
$

因为 $p_t(y)$ 与 $x$ 无关，所以梯度消失。  
这个式子特别重要，它把条件 score 分成了两部分：

1. $\nabla \log p_t(x)$：无条件部分  
2. $\nabla \log p_t(y|x)$：与 prompt / label 相关的额外部分

也就是说：

> **条件生成 = 无条件生成 + 一个“告诉你如何更符合条件 $y$”的修正项。**

这就是 guidance 能成立的数学基础。

---

# 6. Classifier Guidance 的做法：把分类器项放大

既然条件信息体现在

$
\nabla \log p_t(y|x)
$

这一项上，那最直接的办法当然就是：

> 把这一项乘一个更大的系数 $w>1$

于是得到一个强化后的条件 score：

$
\nabla \log p_t(x)+w \nabla \log p_t(y|x)
$

再对应回向量场，就得到一个被“prompt-reinforced”的 guided vector field。  
讲义里把这件事概括成：classifier guidance 通过分解 guided vector field，然后放大 classifier 的梯度项 $\nabla \log p_t(y|x)$，达到更强条件对齐的效果。

这里的直觉非常清楚：

- 当 $w=1$ 时，就是普通条件生成
- 当 $w>1$ 时，模型会更强烈地朝着“更像 label / prompt $y$”的方向走

所以 classifier guidance 本质上是在做：

> **人为放大“这张图被判定为符合条件 $y$”的驱动力。**

---

# 7. 但 classifier guidance 有个明显缺点：还得额外训练 classifier

虽然这个思路很漂亮，但工程上会有麻烦：

- 你不仅要有生成模型
- 还得单独训练一个 classifier 去估计 $p_t(y|x)$ 或其梯度
- 而且这个 classifier 还得适配不同时间 $t$ 的 noisy sample

这会带来额外训练成本，也增加系统复杂度。

所以很自然就会问：

> 能不能不额外训练 classifier，也达到类似效果？

这就引出了 **classifier-free guidance**。

---

# 8. CFG 的核心观察：有条件模型和无条件模型的差，本身就代表“条件信息”

讲义里给出的关键想法非常漂亮：

如果我们已经有了

- 无条件向量场 $u_t^{\text{target}}(x|\emptyset)$
- 有条件向量场 $u_t^{\text{target}}(x|y)$

那么两者的差值：

$
u_t^{\text{target}}(x|y)-u_t^{\text{target}}(x|\emptyset)
$

其实就可以被理解为：

> **“条件 $y$ 额外带来的那部分方向修正”**

这和 classifier guidance 里分离出来的“prompt-dependent part”是同一类东西。  
所以如果我们想放大条件作用，就不一定非要训练外部 classifier，只需要：

> 放大“条件模型”和“无条件模型”之间的差异即可。 

---

# 9. 于是经典 CFG 公式就出来了

如果要把上面的“差值部分”放大 $w$ 倍，那么最自然的写法就是：

$
\tilde u_t(x|y)=
u_t(x|\emptyset)
+
w\big(u_t(x|y)-u_t(x|\emptyset)\big)
$

展开以后就是：

$
\tilde u_t(x|y)=
(1-w)u_t(x|\emptyset)+wu_t(x|y)
$

这就是我们最熟悉的 **classifier-free guidance** 公式。讲义明确给出了这一形式，并指出它对一般概率路径也成立，不仅仅是高斯路径。

这个式子非常值得从结构上理解：

- $u_t(x|\emptyset)$：无条件的“基础生成方向”
- $u_t(x|y)-u_t(x|\emptyset)$：纯粹由条件带来的修正
- $w$：修正项的放大倍数

所以它本质上是：

> **先按无条件生成走，再把“满足 prompt 的偏移方向”额外放大。**

---

# 10. 为什么这个公式这么合理？

这个公式有几个非常好的性质。

## 10.1 当 $w=1$ 时，退化成普通条件生成

代入 $w=1$：

$
\tilde u_t(x|y)=u_t(x|y)
$

也就是说，不做任何额外强化。

## 10.2 当 $w>1$ 时，条件作用被放大

因为

$
\tilde u_t(x|y)=
u_t(x|\emptyset)+w\big(u_t(x|y)-u_t(x|\emptyset)\big)
$

所以 $w$ 越大，模型越强烈地朝条件信号推动的方向前进。

## 10.3 它不需要单独训练 classifier

这正是 CFG 最实用的地方：  
你不需要额外的 $p_t(y|x)$ 模型，只要同一个网络同时学会：

- 有条件生成
- 无条件生成

就够了。

---

# 11. 训练时怎么让一个模型同时学“有条件”和“无条件”？

这就是 CFG 训练最经典的 trick。

讲义指出，问题在于：如果你从真实数据 $(z,y)\sim p_{\text{data}}(z,y)$ 采样，那么永远不会得到 $y=\emptyset$。  
也就是说，真实数据集中不存在“空条件”这个标签。

所以必须**人为制造**这种情况。

具体做法是：

> 设一个超参数 $\eta$，表示丢弃条件的概率。  
> 在训练时，对于每个样本 $(z,y)$，以概率 $\eta$ 把原来的 $y$ 替换成空条件 $\emptyset$。 

于是，模型有时候看到的是：

- 正常条件 $y$，学习有条件生成

有时候看到的是：

- 空条件 $\emptyset$，学习无条件生成

这样同一个网络就同时学到了两种模式。

---

# 12. CFG 的训练目标长什么样？

讲义给出的 CFG conditional flow matching 目标是：

$
L_{\text{CFG-CFM}}(\theta)=
\mathbb{E}_{(z,y)\sim p_{\text{data}}(z,y),\; t\sim\mathrm{Unif}[0,1],\; x\sim p_t(\cdot|z)}
\left[
\|u_t^\theta(x|y)-u_t^{\text{target}}(x|z)\|^2
\right]
$

但这里有一个附加操作：

> 以概率 $\eta$，把 $y$ 替换成 $\emptyset$。 

注意这个损失的形式本身并没有变复杂。  
变的只是：模型输入的条件有时被 drop 成空条件。

这点非常巧妙，因为它意味着：

- 训练代码几乎不需要两套系统
- 只是偶尔把条件 embedding 换成空 embedding
- 同一个网络参数自然就兼顾了 conditional / unconditional 两种行为

---

# 13. Gaussian 路径下，训练算法会长什么样？

讲义还给出了一个非常具体的训练流程。  
对于高斯路径

$
p_t(x|z)=\mathcal N(\alpha_t z,\beta_t^2 I)
$

训练步骤大致就是：

1. 从数据集中采样一对 $(z,y)$
2. 采样时间 $t\sim \mathrm{Unif}[0,1]$
3. 采样噪声 $\epsilon\sim\mathcal N(0,I)$
4. 构造 noisy sample
   $
   x=\alpha_t z+\beta_t\epsilon
   $
5. 以概率 $p$ 丢弃 label，让 $y\leftarrow\emptyset$
6. 回归目标向量场

讲义里对应的单步损失写成：

$
L(\theta)=\|u_t^\theta(x|y)-(\dot\alpha_t z+\dot\beta_t\epsilon)\|^2
$

也就是说，CFG 在训练阶段本质上仍然是在做 flow matching，只是附加了 **label dropout**。

---

# 14. 推理时怎么用 CFG？

训练完以后，推理时对于固定条件 $y$，我们先分别算出：

- $u_t^\theta(x|\emptyset)$
- $u_t^\theta(x|y)$

再组合成：

$
\tilde u_t^\theta(x|y)=(1-w)u_t^\theta(x|\emptyset)+wu_t^\theta(x|y)
$

然后再用这个强化后的向量场去模拟 ODE 或 SDE。  
讲义里对 flow model 的说法是：

- 初始化 $X_0\sim p_{\text{init}}$
- 然后用 $\tilde u_t^\theta(X_t|y)$ 从 $t=0$ 积分到 $t=1$
- 最终得到样本 $X_1$ 

而对于 diffusion model，讲义也明确说了：  
直接把原来的 $u_t^\theta(x|y)$ 替换成 $\tilde u_t^\theta(x|y)$，再照常按 SDE 采样即可。

所以 CFG 并不是一种新的模型结构，而更像是一种：

> **训练时加入空条件，推理时重组 conditional/unconditional 输出的采样技巧。**

---

# 15. 为什么调大 guidance scale 会更贴 prompt？

现在这个问题就很容易回答了。

因为 $w$ 放大的不是整个向量场，而是：

$
u_t(x|y)-u_t(x|\emptyset)
$

也就是“只由条件带来的额外偏移”。

所以当 $w$ 增大时，模型会更强烈地向“更符合 prompt”的方向走。  
讲义中也提到，随着 guidance scale 增大，生成结果通常会表现出更好的 conditioning alignment；在实践中，很多 AI 生成图像和视频都强依赖较大的 CFG scale。

你可以把它想成：

- 无条件模型知道“什么图总体看起来像真的”
- 条件模型知道“怎样更接近 prompt”
- CFG 则是在说：  
  **“保持真实感的同时，把靠近 prompt 的那部分再推得更猛一点。”**

---

# 16. 但为什么 guidance 太大有时也会出问题？

虽然讲义这一节主要强调的是 CFG 的成功经验，但从公式本身你也能看出它的副作用：

当 $w$ 变得很大时，模型其实不再是在忠实采样

$
p_{\text{data}}(\cdot|y)
$

而是在朝一个“更强条件对齐”的启发式方向走。  
讲义里也明确说明：

> 如果使用 $w>1$，最终 $X_1$ 的分布不一定仍然严格对齐于 $p_{\text{data}}(\cdot|y)$。  
> 但经验上会表现出更好的条件一致性，因此它本质上是一个 heuristic。

这句话很重要，因为它解释了一个常见现象：

- guidance scale 太小：图像不够听 prompt
- guidance scale 太大：虽然更听 prompt，但可能牺牲自然性、多样性，甚至出现过度锐化、细节失真等现象

所以 CFG 并不是“数学上严格更正确”，而是：

> **用一点分布偏移，换来更强的条件可控性。**

---

# 17. CFG 到底为什么这么成功？

因为它同时满足了三件很重要的事：

## 17.1 训练简单

只需要在原来的条件训练里加入 label dropout。

## 17.2 推理简单

只需要多跑一次空条件前向，然后线性组合。

## 17.3 效果极强

它直接提升“prompt adherence”，而这恰恰是文生图/文生视频最重要的体验指标之一。

所以 CFG 虽然从理论上只是 heuristic，但从工程上几乎成了现代生成模型的标准配置。

---

# 18. 本章小结

这一篇最核心的逻辑可以概括成下面几步：

1. 条件生成的目标是从  
   $
   p_{\text{data}}(\cdot|y)
   $
   中采样。

2. 最基础的做法是训练条件模型 $u_t^\theta(x|y)$，这就是 vanilla guidance。

3. 但实践中普通条件训练往往不够贴 prompt，因此需要人为增强条件作用。

4. Classifier guidance 的思路是：把条件相关的 classifier 梯度 $\nabla \log p_t(y|x)$ 放大。

5. Classifier-free guidance 则不再依赖额外 classifier，而是直接放大  
   $
   u_t(x|y)-u_t(x|\emptyset)
   $
   这个“条件额外贡献”。

6. 因而得到经典公式：
   $
   \tilde u_t(x|y)=(1-w)u_t(x|\emptyset)+wu_t(x|y)
   $
   当 $w>1$ 时，条件信号被强化。

7. 训练时通过以概率 $\eta$ 丢弃标签 $y\to\emptyset$，让同一个模型同时学会有条件与无条件生成。

如果只记一句话，那就是：

> **CFG 的本质，是把“有条件模型”和“无条件模型”的差值当成条件信号，并在推理时把这部分信号放大。** 
> 
> **核心章节第五章**

---

# 19. 下一篇要讲什么？

到这里，条件控制已经讲清楚了。  
接下来就该进入现代大模型工程视角了：

> 实际上的文生图 / 文生视频系统，不只是一个训练目标，  
> 它还涉及 U-Net、DiT、VAE、文本编码器、latent space 等一整套组件。

下一篇我们就接着写：

# 《从 U-Net 到 DiT，再到 VAE：Stable Diffusion 3 背后的完整工程栈》

---