---
title: "MITS.6184: 7.离散扩散模型入门：文本生成为什么也能用 diffusion 思想？"
date: 2026-03-19
tags:
  - Discrete Diffusion
  - CTMC
  - Diffusion Language Model
  - Flow Matching
categories:
  - Generative Models
description: "从连续时间马尔可夫链、rate matrix 到 masked diffusion language model，理解离散扩散如何用于文本生成。"
---

# 离散扩散模型入门：文本生成为什么也能用 diffusion 思想？

前面几篇我们讲的 flow model、diffusion model，默认数据都在连续空间里：

$
x \in \mathbb R^d
$

这时我们可以写：

- ODE：
  $
  \frac{d}{dt}X_t=u_t(X_t)
  $
- SDE：
  $
  dX_t=u_t(X_t)dt+\sigma_t dW_t
  $

因为连续空间里，“往某个方向移动一点”这件事是有意义的。  
图像像素、latent 向量、视频表示都可以做这种微小连续变化。

但如果数据变成文本 token 呢？

例如一句话：

$
(\text{The},\ \text{cat},\ \text{sat},\ \text{on},\ \text{the},\ \text{mat})
$

这里每一维都不再是实数，而是离散词表中的一个符号。  
这时你就会发现一个根本问题：

> **离散 token 没有“朝某个方向微小移动一点”这种说法。**

你不能说：

- “cat 往 dog 的方向移动 0.1”
- “the 往 a 的方向平滑移动一点”

这就是为什么连续扩散不能直接原封不动地搬到文本上的原因。

但这并不意味着 diffusion 思想失效了。  
现代离散扩散的关键洞见是：

> **虽然离散 token 不能连续移动，但它们可以随机跳转。**

而描述这种“随时间随机跳转”的最自然工具，就是 **连续时间马尔可夫链（CTMC, continuous-time Markov chain）**。

---

# 1. 连续空间和离散空间最大的区别：有没有“方向”

先回忆一下连续 flow / diffusion 的核心结构：

- 概率路径 $p_t$
- 向量场 $u_t(x)$
- 或者 score $\nabla \log p_t(x)$

这些对象都隐含着一个事实：

> **在连续空间里，状态的变化可以用“方向”和“速度”来描述。**

例如在图像 latent space 里，一个点可以沿着某个向量缓慢移动；  
在 SDE 里，样本虽然有噪声扰动，但每一步仍然是连续的微小变化。

而讲义在离散空间部分特别指出，离散概率路径和高斯概率路径有相似之处，也有本质区别：前者不是“transporting probability mass with direction”，而是更像“fade out 一个分布、fade in 另一个分布”；换句话说，在离散空间里没有连续方向，概率质量更像是被“teleport”到另一个状态上。:contentReference[oaicite:2]{index=2}

这句话非常关键。

你可以这样理解：

- 连续扩散：像一团墨水在水里慢慢流动
- 离散扩散：更像某个 token 以一定概率突然跳成另一个 token

所以在离散空间里，最自然的生成过程不是“连续流动”，而是“随机跳转”。

---

# 2. 离散 diffusion 的基本对象：状态空间与 token 序列

设词表为

$
V=\{v_1,\dots,v_{|V|}\}
$

一句长度为 $d$ 的文本序列可以写成：

$
z=(z_1,\dots,z_d)\in V^d
$

也就是说，整个数据空间不再是 $\mathbb R^d$，而是一个离散状态空间 $S=V^d$。

于是，离散生成模型的目标可以写成：

- 从某个初始“噪声序列”分布 $p_{\text{init}}$ 出发
- 通过一个时间连续、状态离散的随机过程
- 最终得到
  $
  X_1 \sim p_{\text{data}}
  $

这和连续生成模型在宏观目标上完全一样。  
唯一变化是：中间过程从“连续演化”变成了“离散跳转”。

---

# 3. CTMC：离散空间里的连续时间随机过程

在离散空间中，最自然的过程就是 **连续时间马尔可夫链（CTMC）**。

它的直觉是：

- 时间仍然是连续的
- 但状态不是连续变化，而是在某些随机时刻突然跳到另一个离散状态

所以 CTMC 可以看成是：

> **连续时间版的随机跳转系统。**

它和前面连续 diffusion 的关系非常紧密：

- 连续空间里，局部更新规则由向量场 $u_t(x)$ 决定
- 离散空间里，局部更新规则由 **rate matrix** 决定

讲义里明确把 rate matrix 说成是向量场在离散空间里的对应物。

---

# 4. 向量场在离散空间里的对应物：Rate Matrix

在连续空间里，向量场 $u_t(x)$ 回答的问题是：

> 如果当前在状态 $x$，下一瞬间应该往哪个方向移动？

而在离散空间里，我们不能谈“方向”，于是换成另一个问题：

> 如果当前在状态 $x$，下一瞬间跳到另一个状态 $y$ 的速率是多少？

这个“跳转速率”就由 **rate matrix** $Q_t(y|x)$ 给出。

所以可以把它理解成：

- $Q_t(y|x)$ 越大，表示从 $x$ 跳到 $y$ 越容易发生
- 它刻画的是一个局部随机跳转机制
- 整个 CTMC 就由这套跳转机制定义 

一句话概括：

> **连续空间里是“朝哪走”，离散空间里是“往哪跳、跳多快”。**

---

# 5. 离散 probability path：仍然先规定“每个时刻应该长什么样”

和 Flow Matching 一样，离散扩散也不是直接硬学一个 CTMC。  
第一步仍然是定义一条概率路径 $p_t$，满足：

- $p_0=p_{\text{init}}$
- $p_1=p_{\text{data}}$

讲义在 Figure 19 的解释里强调，这条离散路径更像是“把初始分布逐渐淡出，把终点分布逐渐淡入”，而不是像连续高斯路径那样有真实的平移/运输。:contentReference[oaicite:6]{index=6}

这点很重要，因为它说明：

> 离散 diffusion 仍然保留了“先定义路径，再学局部规则”的总体思想，  
> 只是路径的几何直觉不再是连续运动，而是分布权重的渐变。

---

# 6. 条件 probability path：仍然先围绕单个数据点来定义

和连续 Flow Matching 完全平行，离散情形里也先对单个终点数据 $z$ 定义条件概率路径：

$
p_t(\cdot|z)
$

它的含义仍然是：

- 初始时刻像噪声分布
- 最终时刻塌缩到具体数据点 $z$

讲义里随后引入 **conditional rate matrix** $Q_t^z(y|x)$，并定义：

> 如果
> $
> X_0\sim p_{\text{init}},\quad X_t \text{ 是由 }Q_t^z\text{ 定义的 CTMC}
> $
> 那么在任意时刻都满足
> $
> X_t\sim p_t(\cdot|z)
> $
> 就称 $Q_t^z$ 是一个 conditional rate matrix。:contentReference[oaicite:7]{index=7}

这和连续情形中的 conditional vector field 是一模一样的角色分工：

- continuous: conditional vector field
- discrete: conditional rate matrix

---

# 7. 离散版 marginalization trick：把“通往每个样本的跳转机制”加权成整体生成机制

连续 Flow Matching 最漂亮的地方，是 marginalization trick：

$
u_t(x)=\int u_t(x|z)p_t(z|x)\,dz
$

离散空间里，讲义给出了它的完全平行版本，也就是 **discrete marginalization trick**：

$$
Q_t(y|x)=
\sum_{z\in S}
Q_t^z(y|x)\,
\frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)}=
\sum_{z\in S}
Q_t^z(y|x)\,p_{1|t}(z|x).
$$

并且这个 $Q_t$ 是一个有效的 marginal rate matrix，它对应的 CTMC 会满足：

$
X_0\sim p_{\text{init}},\quad X_t \text{ 是 }Q_t\text{ 的 CTMC}
\Longrightarrow
X_t\sim p_t,
$

特别地，

$
X_1\sim p_{\text{data}}.
$

也就是说，这个 marginal rate matrix 真正实现了“把噪声变成数据”的离散生成过程。:contentReference[oaicite:8]{index=8}

所以离散扩散和连续 Flow Matching 的深层逻辑是完全一致的：

- 先围绕每个终点样本定义条件局部规则
- 再用后验概率加权，得到整体生成规则

---

# 8. 但真正让离散扩散变得实用的，是 factorized mixture path

如果只到这一步，框架很漂亮，但训练还不一定方便。  
讲义接下来选了一类非常实用的离散路径，叫做 **factorized mixture path**。

它的核心做法是：

- 对序列中每个位置 $j$
- 独立地决定：保留真实 token，还是用噪声 token 替换

更具体地说，在训练时：

1. 对每个位置 $j$，采样一个 mask 变量
   $
   m_j\sim \text{Bernoulli}(\kappa_t)
   $
2. 再采样一个噪声 token $\xi_j\sim p_{\text{init}}^{(j)}$
3. 然后令
   $
   x_j = m_j z_j + (1-m_j)\xi_j
   $

这意味着：

- 当 $m_j=1$ 时，该位置保留原 token $z_j$
- 当 $m_j=0$ 时，该位置被噪声 token 替换 :contentReference[oaicite:10]{index=10}

讲义解释说，这种路径会随着时间逐步“destroy”信息：  
在 $t=0$ 时，信息几乎全被毁掉；在 $t=1$ 时，不再破坏任何信息。:contentReference[oaicite:11]{index=11}

所以它和高斯路径的类比是：

- 连续高斯路径：逐渐加大/减小噪声幅度
- 离散 mixture path：逐渐提高/降低每个位置被保留的概率

---

# 9. 最关键的一步：marginal rate matrix 可以重参数化成“终点 token 的后验概率”

这一点几乎就是离散扩散最漂亮的结论。

讲义在 factorized setting 下推导出：

$$
Q_t(v_i,j|x)=
\frac{\dot\kappa_t}{1-\kappa_t}
\Big(
p_{1|t}(z_j=v_i|x)-\delta_{x_j}(v_i)
\Big)
$$

然后强调：

> 这说明 marginal rate matrix 本质上可以重参数化为  
> $
> p_{1|t}(z_j=v_i|x)
> $
> 也就是：给定当前 noisy sequence $x$，第 $j$ 个位置最终 token 是 $v_i$ 的后验概率。:contentReference[oaicite:12]{index=12}

这件事非常关键，因为它把一个看起来很“随机过程 / CTMC / 速率矩阵”的问题，转成了一个极其熟悉的深度学习任务：

> **对每个位置做分类。**

也就是说，模型不一定要显式输出整个 rate matrix，  
只需要输出：

$
p_{1|t}(z_j=v_i|x)
$

即“当前位置最终应该是什么 token”的概率分布。

---

# 10. 于是离散 diffusion 的训练，最后竟然只是交叉熵分类

既然我们只需要预测每个位置最终 token 的后验分布，那就可以直接定义一个网络：

$
p_{1|t}^\theta : x \mapsto
\big(
p_{1|t}^\theta(z_j=v_i|x)
\big)_{j=1,\dots,d,\; v_i\in V}
$

讲义指出，这个网络输出的形状就是 $d\times |V|$，本质上就是：

- 每个位置一个 softmax
- 每个位置预测终点 token 的类别分布
- 网络本身可以是标准的序列到序列模型，比如 Transformer :contentReference[oaicite:13]{index=13}

于是训练损失直接变成：

$$
L_{\text{DFM}}(\theta)=
\mathbb E_{z\sim p_{\text{data}},\; t\sim\text{Unif}[0,1],\; x\sim p_t(\cdot|z)}
\left[
\sum_{j=1}^d -\log p_{1|t}^\theta(z_j|x)
\right]
$$

这就是讲义里的 **Discrete Flow Matching loss**。

讲义还专门强调，这一点很 remarkable：

- 连续 Flow Matching 最后变成简单回归
- 离散 Flow Matching / discrete diffusion 最后则变成简单分类 

所以从工程实现角度看，离散 diffusion 其实比你想象中更“正常”：

> **训练一个能根据 noisy sequence 预测原 token 的分类器就行。**

---

# 11. 这时候就很像 BERT / Masked LM 了，对吗？

是的，直觉上会非常像。  
因为训练过程就是：

- 把一部分 token 破坏掉
- 给模型看 noisy sequence
- 让它恢复原始 token

但和普通 Masked Language Modeling 相比，离散 diffusion 有两个更系统化的地方：

## 第一，它有明确的时间变量 $t$
不同 $t$ 对应不同的破坏强度 / 噪声水平。

## 第二，它有完整的生成路径与 CTMC 采样解释
它不是单纯的“mask 再恢复”任务，而是被嵌在一整套从 $p_{\text{init}}$ 到 $p_{\text{data}}$ 的生成框架里。

所以可以说：

> **Masked LM 可以看作离散 diffusion 思想的一种局部近亲；  
> 而离散 diffusion 则把这件事系统地放进了概率路径和马尔可夫过程框架中。**

---

# 12. 具体例子：Masked Diffusion Language Model (MDLM)

讲义最后给出的最直观例子就是 **Masked Diffusion Language Model**。

它的做法是：

- 在原词表
  $
  V=\{v_1,\dots,v_{|V|}\}
  $
  之外，再加入一个特殊 token：
  $
  [\text{mask}]
  $
- 然后把初始点设成全 mask 序列：
  $
  p_{\text{init}}=\delta_{[\text{mask}]^d}
  $

也就是说，采样起点就是：

$
[\text{mask}], [\text{mask}], \dots, [\text{mask}]
$

讲义明确写道，这就是 MDLM 的设定。

---

## 12.1 MDLM 的生成过程长什么样？

讲义在 Figure 20 给了一个非常直观的例子：

- $t=0$：全是 `[MASK]`
- $t=0.25$：少数位置开始出现词
- $t=0.75$：大部分位置已经被恢复
- $t=1$：完整句子出现 

例如：

- $t=0$：`[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]`
- $t=0.25$：`[MASK] [MASK] [MASK] on [MASK] [MASK] [MASK]`
- $t=0.75$：`[MASK] cat [MASK] on the mat [MASK]`
- $t=1$：`The cat sat on the mat .`

这和连续扩散的“从纯噪声逐渐清晰”非常像，只不过现在的“噪声”不再是高斯噪声，而是：

> **未知 token / mask token。**

所以 MDLM 的直觉可以总结成一句话：

> **不是从高斯噪声去噪，而是从一串被遮蔽或破坏的 token 序列中逐步把句子补全。**

---

# 13. 离散 diffusion 的完整训练流程其实非常简单

讲义在 Algorithm 8 里总结了 factorized CTMC model 的训练流程，大意就是：

1. 从数据集中采一个真实序列 $z\sim p_{\text{data}}$
2. 采样时间 $t\sim \text{Unif}[0,1]$
3. 按 factorized mixture path 采 noisy state $x$
4. 用网络输出每个位置的 logits / softmax 概率
5. 对真实 token $z_j$ 做 token-wise cross entropy
6. 用梯度下降更新参数 

这说明一个很重要的事实：

> **离散 diffusion 的理论框架虽然看起来有 CTMC、rate matrix、Kolmogorov forward equation 等高级对象，  
> 但最终实现落地时，训练 loop 非常像标准的序列分类训练。**

这也是它能真正扩展到大规模语言建模的重要原因。

---

# 14. 离散 diffusion 和连续 diffusion 的平行关系，应该怎样记？

这是这一篇最值得建立的“对照表”。

## 连续空间
- 状态：$x\in\mathbb R^d$
- 局部规则：vector field $u_t(x)$
- 过程：ODE / SDE
- 训练：回归 score / noise / vector field
- 典型噪声：高斯噪声

## 离散空间
- 状态：$x\in V^d$
- 局部规则：rate matrix $Q_t(y|x)$
- 过程：CTMC
- 训练：预测终点 token 后验分布，做交叉熵分类
- 典型噪声：mask token 或随机替换 token 

所以离散 diffusion 不是“和连续 diffusion 完全不同的一套东西”，而是：

> **把连续 diffusion 的思想迁移到了另一个状态空间，并把“微小移动”替换成“随机跳转”。**

---

# 15. 为什么 flow / diffusion 思想能这么自然地推广到离散空间？

讲义最后有一句很值得记住的话：

> flow / diffusion 的原则并不只属于 flows 或 CTMC，它们本质上是更一般的 **Markov process generative modeling** 原则。  
> 这进一步通向一个更统一的 **Generator Matching** 框架，把连续与离散的生成模型统一起来。:contentReference[oaicite:21]{index=21}

这句话的意义非常大。

它告诉我们：

- 真正核心的不是 ODE、也不是高斯噪声本身
- 而是“先设计路径，再学局部马尔可夫生成规则”这一更抽象的思想

于是：

- 连续空间：局部规则是 vector field
- 离散空间：局部规则是 rate matrix
- 其他更一般的状态空间，也可能有自己的 generator

所以你可以把离散 diffusion 看成：

> **生成建模框架的一次成功“跨状态空间迁移”。**

---

# 16. 本章小结

这一篇最核心的内容可以总结成下面几条：

1. 连续扩散不能直接搬到文本上，因为离散 token 没有“连续方向移动”的概念。:contentReference[oaicite:22]{index=22}

2. 在离散空间中，更自然的生成过程是连续时间马尔可夫链（CTMC）：时间连续，但状态以随机跳转的方式变化。

3. 连续空间里的 vector field，在离散空间中对应为 rate matrix $Q_t(y|x)$。

4. 和连续 Flow Matching 一样，离散情形也先定义条件 probability path，再定义 conditional rate matrix，并通过 discrete marginalization trick 得到 marginal rate matrix。:contentReference[oaicite:25]{index=25}

5. 在 factorized mixture path 下，marginal rate matrix 可以重参数化为终点 token 的后验概率  
   $
   p_{1|t}(z_j=v_i|x)
   $
   因而训练可化成每个位置的分类问题。:contentReference[oaicite:26]{index=26}

6. 于是离散 Flow Matching / 离散 diffusion 的训练损失就是 token-wise cross entropy。

7. MDLM 是最典型的例子：把 `[MASK]` 加入词表，令初始状态为全 mask 序列，再逐步恢复出完整文本。

如果只记一句话，那就是：

> **离散扩散不是在 token 上加高斯噪声，而是把连续扩散的“局部生成规则”思想迁移到 CTMC 上，把“连续移动”换成“离散跳转”。**

---

# 17. 整个系列到这里就收束了

回头看这 7 篇，其实正好是一条很完整的主线：

1. 生成模型 = 从数据分布采样  
2. 用 ODE / SDE 把简单分布变成复杂分布  
3. Flow Matching：先设计路径，再回归方向场  
4. Score Matching：为什么扩散模型会预测噪声  
5. Guidance / CFG：为什么模型会更“听 prompt 的话”  
6. U-Net / DiT / VAE：现代系统如何工程落地  
7. 离散扩散 / CTMC：为什么文本也能用 diffusion 思想

如果你把这些博客连起来，其实已经能构成一套比较完整的 diffusion / flow 生成模型入门到系统理解的博客系列了。

---