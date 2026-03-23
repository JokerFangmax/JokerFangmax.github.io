---
title: "MITS.6184: 4.扩散模型为什么要预测噪声？从 Score Matching 讲清楚"
date: 2026-03-19
tags:
  - Diffusion Model
  - Score Matching
  - DDPM
  - Denoising Score Matching
categories:
  - Generative Models
description: "从 score function、denoising score matching 到 DDPM 的噪声预测参数化，理解扩散模型为什么训练成“预测噪声”。"
---

# 扩散模型为什么要预测噪声？从 Score Matching 讲清楚

前一篇我们讲了 Flow Matching。  
它的核心思想是：

- 先设计一条从噪声到数据的概率路径
- 再学习一个向量场，让 ODE 沿着这条路径演化

但如果你接触过扩散模型，就会发现它们在训练时经常不是预测“方向场”，而是预测：

- score
- 或者更常见地，直接预测 noise

这时一个很自然的问题就是：

> 扩散模型为什么总在预测噪声？  
> 这和概率路径、向量场、生成过程到底是什么关系？

这一篇就把这件事系统讲清楚。

---

# 1. 从“向量场”到“score function”

在前面的 flow 视角里，我们主要关心的是向量场 $u_t(x)$。  
它告诉我们：

> 在时间 $t$ 的时候，如果样本位于位置 $x$，那它下一步应该往哪里移动(速度)。

而在 score matching 视角里，我们关心的是另一个对象：

$
\nabla \log p_t(x)
$

这就是 **score function**。

它看起来只是一个数学定义，但直觉上很好理解。  
因为 $\log p_t(x)$ 表示当前位置的对数概率密度，所以它的梯度

$
\nabla \log p_t(x)
$

就表示：

> **如果你想让当前位置的概率密度增加得最快，应该往哪个方向走。**

所以 score function 其实是在告诉你：

- 哪个方向更靠近高概率区域
- 哪个方向更像真实数据分布的“内部”

从这个角度看，score 和向量场很像，都是“局部方向信息”。  
只不过：

- 向量场是人为指定的动力系统方向(知道初始分布和数据分布，我们自己选择一条路径，如高斯路径)
- score 是由概率分布本身诱导出来的“上升方向”

---

# 2. 我们真正想学的，是随时间变化的 marginal score

和 Flow Matching 一样，我们这里仍然有一条概率路径 $p_t$。

所以在时刻 $t$，最理想的目标其实是学习这条边缘路径的 score：

$
\nabla \log p_t(x)
$

讲义里把它叫做 **marginal score function**。

如果我们能学到它，那么它就能告诉我们：

> 在每个时刻 $t$，对于任意中间状态 $x$，怎样朝着更高概率的区域移动。

于是就可以用它来构造采样过程。

---

# 3. 但问题来了：marginal score 一般并不好直接算

虽然目标看起来很清楚，但现实里有一个老问题：

> 我们通常并不知道 $p_t(x)$ 的显式公式，因此也不知道 $\nabla \log p_t(x)$。

这和 Flow Matching 中学 marginal vector field 时遇到的问题几乎一样：

- 真正想学的东西是边缘量
- 但边缘量往往涉及对所有数据点的积分
- 实际训练时不好直接拿来监督

所以如果只停留在“直接回归 marginal score”，那训练并不可行。

---

# 4. 和 Flow Matching 一样，先看 conditional score

解决办法也和前一篇非常类似。  
我们先看更 tractable 的对象：**conditional score**

$
\nabla \log p_t(x|z)
$

它的含义是：

> 如果当前中间状态 $x$ 是从某个真实数据点 $z$ 沿条件概率路径加噪得到的，那么当前位置的条件概率密度梯度是什么。

于是，讲义里定义了两个训练目标：

## 4.1 理想但不可直接计算的 score matching loss

$$
L_{\text{SM}}(\theta)=
\mathbb{E}_{t, z, x\sim p_t(\cdot|z)}
\big[
\|s_t^\theta(x)-\nabla \log p_t(x)\|^2
\big]
$$

这里 $s_t^\theta(x)$ 是神经网络预测的 score。

---

## 4.2 可计算的 conditional / denoising score matching loss

$$
L_{\text{CSM}}(\theta)=
\mathbb{E}_{t, z, x\sim p_t(\cdot|z)}
\big[
\|s_t^\theta(x)-\nabla \log p_t(x|z)\|^2
\big]
$$

注意两者唯一的区别在于：

- 一个目标是 marginal score $\nabla \log p_t(x)$
- 一个目标是 conditional score $\nabla \log p_t(x|z)$

而后者是可以显式算出来的。

---

# 5. 最关键的结论：这两个 loss 只差一个常数

这一步和 Flow Matching 的味道几乎一模一样。

讲义给出的核心定理是：

$
L_{\text{SM}}(\theta)=L_{\text{CSM}}(\theta)+C
$

其中 $C$ 与模型参数 $\theta$ 无关。  
因此它们的梯度相同，最优解也一致。

这意味着：

> **虽然我们真正想学的是 marginal score，但训练时完全可以只对 conditional score 做回归。**

这就是 **denoising score matching** 的核心理由。

也就是说，扩散模型训练并不需要显式知道整个边缘分布的 score，  
只需要知道：

> 某个 noisy sample $x$ 是由哪个干净样本 $z$ 加噪得到的，  
> 然后回归它对应的 conditional score 即可。

这一步是整个方法可训练的根本。

---

# 6. 高斯概率路径下，conditional score 长什么样？

现在进入最常见也最重要的情况：

$
p_t(x|z)=\mathcal N(\alpha_t z,\beta_t^2 I)
$

也就是说，在时刻 $t$：

- 信号部分是 $\alpha_t z$
- 噪声部分是 $\beta_t \epsilon$

因此我们可以写：

$
x = \alpha_t z + \beta_t \epsilon,\qquad \epsilon\sim\mathcal N(0,I)
$

对于这个高斯分布，它的 conditional score 可以直接求出来：

$
\nabla \log p_t(x|z)
=
-\frac{x-\alpha_t z}{\beta_t^2}
$

这个式子特别重要，因为它一下就把 score 和“加噪残差”联系起来了。

你可以这样理解：

- $x-\alpha_t z$ 表示当前位置相对于干净信号部分的偏移
- 这个偏移本质上就是噪声部分
- score 则是把这个“偏移量”按高斯分布的结构变成一个梯度方向

所以在高斯路径下，score 本质上就是：

> **告诉你当前 noisy sample 偏离干净信号多少，以及应该往回修正的方向。**

---

# 7. 把上式代进去，就得到 denoising score matching

把

$
\nabla \log p_t(x|z)
=
-\frac{x-\alpha_t z}{\beta_t^2}
$

代入 conditional score matching loss，有：

$
L_{\text{CSM}}(\theta)
=
\mathbb{E}_{t,z,x\sim p_t(\cdot|z)}
\left[
\left\|
s_t^\theta(x)+\frac{x-\alpha_t z}{\beta_t^2}
\right\|^2
\right]
$

再利用

$
x=\alpha_t z+\beta_t\epsilon
$

可得：

$
x-\alpha_t z=\beta_t\epsilon
$

于是 loss 可以改写成

$
L_{\text{CSM}}(\theta)
=
\mathbb{E}_{t,z,\epsilon}
\left[
\left\|
s_t^\theta(\alpha_t z+\beta_t\epsilon)+\frac{\epsilon}{\beta_t}
\right\|^2
\right]
$

这就是讲义里说的 **denoising score matching** 形式。

为什么叫 denoising？

因为从这个式子里已经很明显了：  
网络本质上是在利用 noisy sample 去恢复其噪声成分。

---

# 8. 这时就已经能看出：score network 其实在“预测噪声”

观察上面的目标：

$
s_t^\theta(\alpha_t z+\beta_t\epsilon)
\approx
-\frac{\epsilon}{\beta_t}
$

可以发现：

- 网络表面上是在预测 score
- 但在高斯路径下，score 与噪声 $\epsilon$ 只差一个简单的线性变换

也就是说：

> **学 score，几乎等价于学噪声。**

这就是为什么很多扩散模型论文里，会把“预测 score”和“预测噪声”视作两种等价参数化方式。

从直觉上讲也很合理：

- 如果你知道一个 noisy sample 里混入了什么噪声
- 那你就能把它减掉
- 也就能逐步往干净数据方向恢复

所以“预测噪声”不是一个偶然技巧，而是高斯概率路径下 score matching 的自然结果。

---

# 9. 为什么后来大家更喜欢直接预测噪声？

虽然上面的 denoising score matching 已经很好了，但它还存在一个数值问题：

$
\left\|
s_t^\theta(\alpha_t z+\beta_t\epsilon)+\frac{\epsilon}{\beta_t}
\right\|^2
$

这里有一个 $\frac{1}{\beta_t}$ 因子。  
当 $\beta_t$ 很小时，也就是噪声很弱、接近干净数据时，这个系数会变得很大，训练容易数值不稳定。

讲义里就明确指出了这一点：  
当 $\beta_t \approx 0$ 时，上述损失会变得不稳定，所以早期 DDPM 工作提出把这个常数项去掉，并重参数化成一个 **noise predictor**。  

也就是说，不再让网络输出 score $s_t^\theta(x)$，而是直接定义：

$
-\beta_t s_t^\theta(x)=\epsilon_t^\theta(x)
$

这里 $\epsilon_t^\theta(x)$ 就是噪声预测网络。

这样一来，训练目标就变成了最熟悉的 DDPM 形式：

$
L_{\text{DDPM}}(\theta)
=
\mathbb{E}_{t,z,\epsilon}
\left[
\|\epsilon_t^\theta(\alpha_t z+\beta_t\epsilon)-\epsilon\|^2
\right]
$

这个式子一下就清爽很多了：

- 输入：noisy sample $x_t=\alpha_t z+\beta_t\epsilon$
- 输出：预测噪声 $\epsilon_t^\theta(x_t)$
- 目标：真实噪声 $\epsilon$

这就是大家最常见到的“扩散模型训练 = 预测噪声”。

---

# 10. 所以 DDPM 到底在学什么？

很多人第一次看 DDPM 会有一种错觉：

> 它是不是只是一个很奇怪的去噪自编码器？

其实不完全是。  
更准确地说，它是在做下面这件事：

1. 先定义一条高斯概率路径
2. 在这条路径上学习 marginal score
3. 由于 marginal score 不可直接监督，于是改学 conditional score
4. 由于高斯 conditional score 与噪声线性等价，于是进一步改成预测噪声

所以 DDPM 的“预测噪声”只是最终外观。  
它背后的理论本体其实是：

> **score matching on a Gaussian probability path**

也就是说，DDPM 不是凭空发明了一个“猜噪声”的任务，  
而是从 score matching 的推导中自然得到的。

---

# 11. score、vector field、noise predictor 三者的关系

到这里，其实可以把前面几篇的内容连起来了。

在概率路径框架下，我们会遇到三种常见对象：

## 11.1 向量场 $u_t(x)$

它定义 ODE / SDE 的动力学方向。

## 11.2 score $ \nabla \log p_t(x) $

它是分布在当前位置的概率密度上升方向。

## 11.3 noise predictor $ \epsilon_t^\theta(x) $

它是高斯概率路径下对 score 的一个重参数化。

所以三者并不是三个互不相干的方法，而是同一套生成建模语言里的不同表示。

特别是在高斯路径下，讲义还指出：

> score network 和 vector field 之间可以互相转换。  
> 因而没有必要把它们当成完全独立的两套系统。

也就是说：

- 你可以从向量场出发理解生成
- 也可以从 score 出发理解生成
- 在最常见的高斯设定里，这两者经常只是表达方式不同

这也是为什么 Flow Matching、Score Matching、DDPM 这些看似不同的方法，底层其实是能接起来的。

---

# 12. 学到 score 之后，怎么拿来采样？

光训练还不够，我们还得采样。

讲义在这一节最后总结说：  
如果我们知道 marginal vector field $u_t^{\text{target}}(x)$ 和 marginal score $\nabla \log p_t(x)$，那么对任意 diffusion coefficient $\sigma_t\ge 0$，下面这个 SDE 的轨迹都会遵循概率路径：

$
dX_t =
\left[
u_t^{\text{target}}(X_t)
+
\frac{\sigma_t^2}{2}\nabla \log p_t(X_t)
\right]dt
+
\sigma_t dW_t
$

这句话很重要，因为它告诉我们：

> **score 不是只在训练时有用，它还直接决定了采样时 SDE 的漂移项。**

也就是说，score learning 最终服务的仍然是“从噪声生成样本”。

对于高斯概率路径，讲义还给出了 score network 和 vector field 之间的显式转换，因此采样时可以用 score 形式，也可以转成向量场形式再去模拟相应的 SDE。

---

# 13. 为什么“预测噪声”这件事这么自然？

如果只从实现角度看，预测噪声好像只是一个经验技巧。  
但从现在的推导回头看，它其实一点也不神秘。

因为整个逻辑链条非常顺：

1. 生成模型要学习一条概率路径
2. 学这条路径的一种方式是学习 marginal score
3. marginal score 不好直接监督，于是改成 denoising score matching
4. 在高斯路径下，conditional score 与噪声成线性关系
5. 为了数值稳定和实现简洁，进一步重参数化为噪声预测

所以：

> **扩散模型预测噪声，不是“为了方便随便选的目标”，而是高斯 denoising score matching 的最自然实现方式。**

---

# 14. 本章小结

这一篇最核心的内容可以总结成下面几条：

1. Score function  
   $
   \nabla \log p_t(x)
   $
   表示当前位置沿哪个方向移动会让概率密度增长最快。

2. 理想目标是学习 marginal score，但它通常不可直接计算。

3. 和 Flow Matching 一样，可以改为学习 tractable 的 conditional score  
   $
   \nabla \log p_t(x|z)
   $
   而且对应的 loss 与真正的 score matching loss 只差一个常数。

4. 在高斯概率路径下，
   $
   p_t(x|z)=\mathcal N(\alpha_t z,\beta_t^2I)
   $
   条件 score 有显式公式
   $
   \nabla \log p_t(x|z)=-\frac{x-\alpha_t z}{\beta_t^2}
   $

5. 代入后可得 denoising score matching，其本质是在恢复加到数据上的噪声。

6. 由于数值稳定性问题，进一步把 score network 重参数化为 noise predictor，得到经典 DDPM loss：
   $
   L_{\text{DDPM}}(\theta)
   =
   \mathbb{E}
   \big[
   \|\epsilon_t^\theta(x_t)-\epsilon\|^2
   \big]
   $

如果只记一句话，那就是：

> **DDPM 之所以预测噪声，是因为在高斯概率路径下，score matching 正好等价于预测噪声。**

---

# 15. 下一篇要讲什么？

到这里，扩散模型为什么要预测噪声已经清楚了。  
接下来就很自然会问：

> 如果我要做文生图，模型是怎么“听懂 prompt”的？  
> 为什么 classifier-free guidance 一加，生成结果就更贴合文本了？

下一篇我们就接着写：

# 《Classifier-Free Guidance 原理：生成模型为什么会更“听 prompt 的话”？》

---