---
title: "SOL 01: VG 01 习题 1——反向高斯的均值与方差"
date: 2026-10-03
math: true
tags:
  - Solutions
  - Diffusion Model
  - 概率论
categories:
  - Solutions
description: "DDPM 反向条件分布 q(x_{t-1}|x_t,x_0) 均值与方差公式的完整推导（配方 + 联合高斯双路线 + Monte Carlo 验证）"
---

# SOL 01：VG 01 习题 1——反向高斯的均值与方差

> 出处：[VG 01：扩散模型基础](../../video_generation/VG_01_diffusion_basics/) §8 习题 1（正文 §2.2 直接使用其结论）  
> 依赖：VG 01 §2.1（前向过程与闭式解 (2.1)）、一维高斯的配方 (complete the square)  
> 结论预览：$\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$，$\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$

---

## 1. 题目与材料

**要证的东西**：反向条件分布

$$
q(x_{t-1}\mid x_t, x_0) = \mathcal{N}\left(x_{t-1};\ \tilde\mu_t(x_t, x_0),\ \tilde\beta_t \mathbf{I}\right)
$$

中的两个参数是

$$
\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,\qquad
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t.
$$

**出发点**（全部来自前向过程，无新假设）。两个已知的高斯：

| 分布 | 来源 | 显式形式 |
|---|---|---|
| $q(x_t\mid x_{t-1})$ | 前向定义（一步加噪） | $\mathcal N(x_t;\ \sqrt{\alpha_t}\,x_{t-1},\ \beta_t\mathbf I)$ |
| $q(x_{t-1}\mid x_0)$ | 闭式解 (2.1) 取 $t-1$ | $\mathcal N(x_{t-1};\ \sqrt{\bar\alpha_{t-1}}\,x_0,\ (1-\bar\alpha_{t-1})\mathbf I)$ |

**符号速查**：

| 记号 | 含义 | 本篇数字标本中的值 |
|---|---|---|
| $\beta_t$ | 第 $t$ 步噪声强度 | $0.02$ |
| $\alpha_t$ | $1-\beta_t$ | $0.98$ |
| $\bar\alpha_t$ | $\prod_{s=1}^t\alpha_s$ | $\bar\alpha_2=0.9604$，$\bar\alpha_3=0.941192$ |
| $(\star)$ | 方差守恒账本：$\alpha_t(1-\bar\alpha_{t-1})+\beta_t = 1-\bar\alpha_t$ | $0.98\times0.0396+0.02=0.058808$ ✓ |

账本 $(\star)$ 在 VG 01 §2.1 已证（它就是"$x_{t-1}$ 自己那份方差 $\times\alpha_t$，加上新噪声 $\beta_t$，等于 $x_t$ 的总方差 $1-\bar\alpha_t$"）。本篇所有化简的最后一行都是它。

推导只做标量（一维）；多维时各维独立、系数全部相同，逐维照抄即可。

---

## 2. 预备知识：「读系数」引理

这是整个推导的核心工具，先把它彻底讲透。

**引理**：若一个关于 $x$ 的密度形如

$$
p(x)\ \propto\ \exp\left(-\frac{1}{2}\left[a\,x^2 - 2b\,x + c\right]\right)\qquad (a>0),
$$

其中 $b,c$ 是与 $x$ 无关的常数，则 $p(x)$ 就是高斯 $\mathcal N\left(\dfrac{b}{a},\ \dfrac{1}{a}\right)$——**均值是一次项系数除以二次项系数，方差是二次项系数的倒数**。

**证明（配方）**：把方括号强行配成完全平方：

$$
a\,x^2-2b\,x+c = a\left(x-\frac{b}{a}\right)^2 + \underbrace{\left(c-\frac{b^2}{a}\right)}_{\text{与 }x\text{ 无关，吸进归一化常数}}.
$$

（展开右边验证：$a x^2 - 2b x + \frac{b^2}{a} + c - \frac{b^2}{a}$，两头吻合。）于是

$$
p(x)\ \propto\ \exp\left(-\frac{a}{2}\left(x-\frac{b}{a}\right)^2\right),
$$

对照高斯密度 $\mathcal N(x;\mu,\sigma^2)\propto\exp\left(-\frac{1}{2\sigma^2}(x-\mu)^2\right)$：$\frac{a}{2}=\frac{1}{2\sigma^2}$ 给 $\sigma^2=\frac{1}{a}$，中心 $\frac{b}{a}$ 给 $\mu=\frac{b}{a}$。$\blacksquare$

**为什么要 "$\propto$"（正比）就够了**：密度差一个常数因子不影响它是哪个分布——那个因子（以及 $c$、以及之后贝叶斯里的分母）最后都会被"密度积分为 1"自动补齐。所以我们全程只需要盯着 $x$ 的二次项和一次项。

**迷你标本**：$p(x)\propto\exp(-\frac{1}{2}[4x^2-8x+5])$。读系数：$a=4\Rightarrow$ 方差 $\frac14$；$b=4\Rightarrow$ 均值 $\frac{4}{4}=1$。所以 $p=\mathcal N(1,\ 0.25)$。验证：$4(x-1)^2=4x^2-8x+4$，剩下的 $5-4=1$ 是常数，确实被吸收。

有了这个引理，整个习题就变成机械劳动：**把 $q(x_{t-1}\mid x_t,x_0)$ 的指数整理成 $a x_{t-1}^2-2b x_{t-1}+c$ 的形状，读出 $a$ 和 $b$，除一下就完事。**

---

## 3. 路线一：贝叶斯 + 配方（题目提示的路线）

### 3.1 第一步：贝叶斯拆解

按定义展开联合分布再拆链（分子分母同乘 $q(x_{t-1}\mid x_0)$ 是戏法，"多除再乘回"）：

$$
q(x_{t-1}\mid x_t, x_0)
= \frac{q(x_t,\,x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
= \frac{q(x_t\mid x_{t-1},\,x_0)\;q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}.
$$

其中第一个因子可以再化简：

$$
q(x_t\mid x_{t-1}, x_0) = q(x_t\mid x_{t-1}).
$$

**理由（马尔可夫性）**：前向过程中 $x_t$ 只由 $x_{t-1}$ 加一步噪声生成——$x_0$ 对 $x_t$ 的全部影响都**经过** $x_{t-1}$ 传递；一旦 $x_{t-1}$ 已知，多告诉它 $x_0$ 不再提供任何新信息。

代入得（分母 $q(x_t\mid x_0)$ 与 $x_{t-1}$ 无关，是常数，按 §2 的讨论直接丢掉）：

$$
q(x_{t-1}\mid x_t,x_0)\ \propto\ q(x_t\mid x_{t-1})\cdot q(x_{t-1}\mid x_0).
$$

### 3.2 第二步：两个密度相乘，指数相加

把两个高斯的密度完整写出来（$x$ 是 $x_{t-1}$ 的简写）：

$$
q(x_t\mid x_{t-1}) = \frac{1}{\sqrt{2\pi\beta_t}}\exp\left(-\frac{(x_t-\sqrt{\alpha_t}\,x)^2}{2\beta_t}\right),
$$

$$
q(x_{t-1}\mid x_0) = \frac{1}{\sqrt{2\pi(1-\bar\alpha_{t-1})}}\exp\left(-\frac{(x-\sqrt{\bar\alpha_{t-1}}\,x_0)^2}{2(1-\bar\alpha_{t-1})}\right).
$$

相乘时：两个归一化常数（都不含 $x$）乘起来丢进 $\propto$；两个 $\exp$ 相乘则**指数相加**（$e^A\cdot e^B=e^{A+B}$）：

$$
q(x_{t-1}\mid x_t,x_0)\ \propto\ \exp\left[-\frac{1}{2}\left(\frac{(x_t-\sqrt{\alpha_t}\,x)^2}{\beta_t}+\frac{(x-\sqrt{\bar\alpha_{t-1}}\,x_0)^2}{1-\bar\alpha_{t-1}}\right)\right].
$$

### 3.3 第三步：展开两个平方，按 $x_{t-1}$ 的幂次归项

这是纯代数，一步不跳。先各自展开（用 $(u-v)^2=u^2-2uv+v^2$）：

**第一个平方**（除以 $\beta_t$）：

$$
\frac{(x_t-\sqrt{\alpha_t}x)^2}{\beta_t}
= \frac{\alpha_t x^2-2\sqrt{\alpha_t}\,x_t\,x+x_t^2}{\beta_t}
= \frac{\alpha_t}{\beta_t}x^2-\frac{2\sqrt{\alpha_t}\,x_t}{\beta_t}x+\frac{x_t^2}{\beta_t}.
$$

**第二个平方**（除以 $1-\bar\alpha_{t-1}$，简记 $s^2:=1-\bar\alpha_{t-1}$）：

$$
\frac{(x-\sqrt{\bar\alpha_{t-1}}x_0)^2}{s^2}
= \frac{x^2-2\sqrt{\bar\alpha_{t-1}}x_0\,x+\bar\alpha_{t-1}x_0^2}{s^2}
= \frac{1}{s^2}x^2-\frac{2\sqrt{\bar\alpha_{t-1}}x_0}{s^2}x+\frac{\bar\alpha_{t-1}x_0^2}{s^2}.
$$

**相加、按幂次归项**（同类项系数相加；注意两个一次项都自带负号，合并后仍是 $-2(\cdots)x$ 的形状）：

| 幂次 | 来自第一项 | 来自第二项 | 合并后系数 |
|---|---|---|---|
| $x^2$ | $\dfrac{\alpha_t}{\beta_t}$ | $\dfrac{1}{s^2}$ | $a=\dfrac{\alpha_t}{\beta_t}+\dfrac{1}{1-\bar\alpha_{t-1}}$ |
| $x$ | $-\dfrac{2\sqrt{\alpha_t}x_t}{\beta_t}$ | $-\dfrac{2\sqrt{\bar\alpha_{t-1}}x_0}{s^2}$ | $-2b$，$b=\dfrac{\sqrt{\alpha_t}x_t}{\beta_t}+\dfrac{\sqrt{\bar\alpha_{t-1}}x_0}{1-\bar\alpha_{t-1}}$ |
| 常数 | $\dfrac{x_t^2}{\beta_t}$ | $\dfrac{\bar\alpha_{t-1}x_0^2}{s^2}$ | $c$（后面不再用它） |

于是

$$
q(x_{t-1}\mid x_t,x_0)\ \propto\ \exp\left(-\frac{1}{2}\left[a\,x^2-2b\,x+c\right]\right).
$$

指数是 $x_{t-1}$ 的二次函数 ⟹ 反向分布**仍是高斯**——这正是"高斯的反向还是高斯"（Anderson 1982）在离散时间下的证明本体。剩下的工作：按引理算 $\frac{1}{a}$（方差）和 $\frac{b}{a}$（均值）。

### 3.4 第四步：算方差 $\tilde\beta_t=1/a$（用一次账本）

先通分求 $a$（公分母 $\beta_t(1-\bar\alpha_{t-1})$；第一项分子分母同乘 $1-\bar\alpha_{t-1}$，第二项同乘 $\beta_t$）：

$$
a=\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}
=\frac{\alpha_t(1-\bar\alpha_{t-1})+\beta_t}{\beta_t(1-\bar\alpha_{t-1})}.
$$

分子恰好是账本 $(\star)$ 的左边，等于 $1-\bar\alpha_t$。所以

$$
a=\frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})},
\qquad
\tilde\beta_t=\frac{1}{a}=\boxed{\ \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t\ }
$$

**读一眼结构**：$a$ 是两个信息源的精度**相加**（$\frac{\alpha_t}{\beta_t}$ 来自"已知 $x_t$"这条线索，$\frac{1}{1-\bar\alpha_{t-1}}$ 来自"已知 $x_0$"这条线索）。精度相加 ⟹ 方差 $\tilde\beta_t$ 比任何单一来源对应的方差都小——同时知道两条线索，当然比只知道一条更确定。具体地，$\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}<1$（分母更大，因为 $\bar\alpha_t<\bar\alpha_{t-1}$），所以 $\tilde\beta_t<\beta_t$：反向分布甚至比前向一步加噪还尖。

### 3.5 第五步：算均值 $\tilde\mu_t=b/a=b\cdot\tilde\beta_t$（再用两次账本）

引理给 $\tilde\mu_t=\frac{b}{a}$，而 $\frac{1}{a}=\tilde\beta_t$ 刚算完，所以 $\tilde\mu_t=b\cdot\tilde\beta_t$。把 $b$ 乘进去，**两项分别约分**：

$$
\tilde\mu_t=\left(\frac{\sqrt{\alpha_t}\,x_t}{\beta_t}+\frac{\sqrt{\bar\alpha_{t-1}}\,x_0}{1-\bar\alpha_{t-1}}\right)\cdot\frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}.
$$

**第一项**：$\beta_t$ 约掉——

$$
\frac{\sqrt{\alpha_t}\,x_t}{\beta_t}\cdot\frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}
=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t.
$$

**第二项**：$(1-\bar\alpha_{t-1})$ 约掉——

$$
\frac{\sqrt{\bar\alpha_{t-1}}\,x_0}{1-\bar\alpha_{t-1}}\cdot\frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}
=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0.
$$

合起来：

$$
\tilde\mu_t=\boxed{\ \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t\ }
$$

（注意这一步根本没用到账本——账本只在求 $a$ 时用了一次，均值只是"乘回来再约分"。）

### 3.6 全程数字标本

设定：$\beta_t=0.02$、$t=3$、$x_0=2$、观测 $x_t=1.5$。逐步代入 §3.3–§3.5 的每个公式：

| 步骤 | 公式 | 数值 |
|---|---|---|
| 基本量 | $\alpha_t=0.98$，$\bar\alpha_2=0.9604$，$\bar\alpha_3=0.941192$ | $1-\bar\alpha_2=0.0396$，$1-\bar\alpha_3=0.058808$ |
| 账本 $(\star)$ 核对 | $\alpha_t(1-\bar\alpha_2)+\beta_t$ | $0.98\times0.0396+0.02=0.058808=1-\bar\alpha_3$ ✓ |
| 二次系数 $a$ 的两块 | $\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_2}$ | $49+25.2525=74.2525$ |
| **方差** $\tilde\beta_t=1/a$ | $\frac{0.0396\times0.02}{0.058808}$ | $0.0134676$ |
| 一次系数 $b$ 的两块 | $\frac{\sqrt{0.98}\times1.5}{0.02}+\frac{\sqrt{0.9604}\times2}{0.0396}$ | $74.2462+49.4949=123.7412$ |
| **均值** $\tilde\mu_t=b/a$ | $123.7412/74.2525$ | $1.6665$ |
| 交叉验证 | $w_0 x_0+w_t x_t$，$w_0=0.3333,\ w_t=0.6666$ | $0.3333\times2+0.6666\times1.5=1.6665$ ✓ |

两个独立算路（$b/a$ 与权重展开）给出同一均值；§5 的 Monte Carlo 也给出 $1.6669$。三种算法互相咬合，公式不会错。

顺带一提配方的"余项"：$c-\frac{b^2}{a}=209.5101-206.2135=3.2966$——一个正数，连同所有归一化常数一起被吸收，这正是 §2 引理里"常数不用管"的具体体现。

### 3.7 均值的解读：精度加权平均

把 $b\cdot\tilde\beta_t$ 改写成另一种形状。定义两个"线索各自对 $x_{t-1}$ 的估计"：

| 线索 | 观测到的量 | 它认为 $x_{t-1}$ 的中心 | 精度（信息量） |
|---|---|---|---|
| 前向一步 $q(x_t\mid x_{t-1})$ | $x_t$ | $\dfrac{x_t}{\sqrt{\alpha_t}}$（把衰减 $\sqrt{\alpha_t}$ 补回去） | $\lambda_1=\dfrac{\alpha_t}{\beta_t}$ |
| 边缘闭式 $q(x_{t-1}\mid x_0)$ | $x_0$ | $\sqrt{\bar\alpha_{t-1}}\,x_0$ | $\lambda_2=\dfrac{1}{1-\bar\alpha_{t-1}}$ |

（验证第一行：$x_t\mid x_{t-1}=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\varepsilon$，反过来 $x_{t-1}=\frac{x_t}{\sqrt{\alpha_t}}-\sqrt{\frac{\beta_t}{\alpha_t}}\varepsilon$，期望就是 $\frac{x_t}{\sqrt{\alpha_t}}$。）则

$$
\tilde\mu_t=(\lambda_1+\lambda_2)^{-1}\left(\lambda_1\frac{x_t}{\sqrt{\alpha_t}}+\lambda_2\sqrt{\bar\alpha_{t-1}}x_0\right),\qquad
\tilde\beta_t=(\lambda_1+\lambda_2)^{-1}.
$$

**均值是两个估计按精度加权平均（谁噪声小谁权重大），方差是精度之和的倒数**——这就是题目提示语"信息融合形式"的含义，与卡尔曼滤波 (Kalman filter) 的更新步是同一副骨架（VG 01 §7 的 Kalman 视角会再用它）。数字标本中 $x_t$ 线索权重 $0.6666$ 大于 $x_0$ 线索的 $0.3333$：$x_t$ 离 $x_{t-1}$ 只差一步噪声，当然更可信；$x_0$ 隔了两步噪声，只能做小修正。

**一个容易踩的坑**：两权重之和 $0.3333+0.6666=0.99990$，**并不恰好等于 1**（差约 $1\times 10^{-4}$，与 $\beta_t^2$ 同阶）——均值只是**近似**凸组合 (convex combination)。不要拿"系数和必为 1"当检验条件，会得到假阴性；§5 的 Monte Carlo 表里 $x_t=x_0=2$ 时 $\tilde\mu_t=1.9998\ne 2$，那个 $2\times10^{-4}$ 的偏差正是权重和 $\ne 1$ 的指纹。真正精确的恒等式是 §4.5 的 $\varepsilon$ 形式。

---

## 4. 路线二：联合高斯条件化（标准公式版）

同一条公式换一条路推——既作交叉验证，也顺手得到一个更重要的形式。

### 4.1 第一步：把 $(x_{t-1},x_t)$ 写成"信号 + 独立噪声"

给定 $x_0$（条件分布里 $x_0$ 是常数），由闭式解 (2.1) 和前向定义：

$$
x_{t-1}=\sqrt{\bar\alpha_{t-1}}\,x_0+\underbrace{\sqrt{1-\bar\alpha_{t-1}}}_{=:r}\,\eta,
\qquad
x_t=\sqrt{\alpha_t}\,x_{t-1}+\sqrt{\beta_t}\,\varepsilon,
$$

其中 $\eta\perp\varepsilon$ 都是标准高斯（VG 01 §2.1 的归纳证明保证了 $\eta$ 只含前 $t-1$ 步噪声、与第 $t$ 步的 $\varepsilon$ 独立）。两个"信号 + 噪声"的线性组合 ⟹ $(x_{t-1},x_t)$ 联合高斯，其分布由四个矩完全决定。逐个算：

**均值**（噪声期望为 0）：

$$
\mathbb E[x_{t-1}\mid x_0]=\sqrt{\bar\alpha_{t-1}}\,x_0=:m_1,\qquad
\mathbb E[x_t\mid x_0]=\sqrt{\alpha_t}\,m_1=\sqrt{\bar\alpha_t}\,x_0=:m_2.
$$

**方差**（$\mathrm{Var}(c\,Z)=c^2\mathrm{Var}(Z)$）：

$$
\mathrm{Var}(x_{t-1}\mid x_0)=r^2=1-\bar\alpha_{t-1},\qquad
\mathrm{Var}(x_t\mid x_0)=\alpha_t r^2+\beta_t\overset{(\star)}{=}1-\bar\alpha_t.
$$

（第二个用了独立随机变量方差可加：$\mathrm{Var}(\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\varepsilon)=\alpha_t\mathrm{Var}(x_{t-1})+\beta_t$。）

**协方差**（双线性展开，一次一项）：

$$
\mathrm{Cov}(x_{t-1},x_t\mid x_0)=\mathrm{Cov}(x_{t-1},\ \sqrt{\alpha_t}\,x_{t-1}+\sqrt{\beta_t}\,\varepsilon)
$$
$$
=\sqrt{\alpha_t}\,\underbrace{\mathrm{Cov}(x_{t-1},x_{t-1})}_{=\,r^2}+\sqrt{\beta_t}\,\underbrace{\mathrm{Cov}(x_{t-1},\varepsilon)}_{=\,0\ (\eta\perp\varepsilon)}
=\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1}).
$$

汇总：

| 矩 | 值 | 数字标本中的值 |
|---|---|---|
| $m_1$ | $\sqrt{\bar\alpha_{t-1}}\,x_0$ | $0.98\times2=1.96$ |
| $m_2$ | $\sqrt{\bar\alpha_t}\,x_0$ | $0.97015\times2=1.9403$ |
| $\mathrm{Var}(x_{t-1})$ | $1-\bar\alpha_{t-1}$ | $0.0396$ |
| $\mathrm{Var}(x_t)$ | $1-\bar\alpha_t$ | $0.058808$ |
| $\mathrm{Cov}$ | $\sqrt{\alpha_t}(1-\bar\alpha_{t-1})$ | $0.98995\times0.0396=0.039202$ |

### 4.2 第二步：代条件高斯公式

对联合高斯 $(X,Y)$，有标准公式（把联合密度视为 $X$ 的密度、用 §2 引理读系数即可证出——不是新工具）：

$$
\mathbb E[X\mid Y=y]=m_X+\frac{\mathrm{Cov}(X,Y)}{\mathrm{Var}(Y)}(y-m_Y),
\qquad
\mathrm{Var}(X\mid Y)=\mathrm{Var}(X)-\frac{\mathrm{Cov}(X,Y)^2}{\mathrm{Var}(Y)}.
$$

代入 $X=x_{t-1}$、$Y=x_t$（条件里的 $x_0$ 已冻结成常数）：

**方差**（逐步）：

$$
\mathrm{Var}(x_{t-1}\mid x_t,x_0)=(1-\bar\alpha_{t-1})-\frac{\alpha_t(1-\bar\alpha_{t-1})^2}{1-\bar\alpha_t}
=(1-\bar\alpha_{t-1})\left[1-\frac{\alpha_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\right].
$$

方括号通分：$1-\frac{\alpha_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}=\frac{(1-\bar\alpha_t)-\alpha_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$，分子正是账本 $(\star)$ 的移项，等于 $\beta_t$。所以

$$
\mathrm{Var}(x_{t-1}\mid x_t,x_0)=(1-\bar\alpha_{t-1})\cdot\frac{\beta_t}{1-\bar\alpha_t}=\tilde\beta_t.\ \checkmark
$$

（数字标本：方括号 $=0.3401=\frac{0.02}{0.058808}$，乘 $0.0396$ 得 $0.013468$，与路线一吻合。）

**均值**（逐步）：

$$
\mathbb E[x_{t-1}\mid x_t,x_0]=\sqrt{\bar\alpha_{t-1}}\,x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\left(x_t-\sqrt{\bar\alpha_t}\,x_0\right).
$$

**合并 $x_0$ 的系数**（一行一行来；用 $\sqrt{\bar\alpha_t}=\sqrt{\alpha_t}\sqrt{\bar\alpha_{t-1}}$）：

$$
x_0\text{ 的系数}=\sqrt{\bar\alpha_{t-1}}-\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\cdot\sqrt{\alpha_t}\sqrt{\bar\alpha_{t-1}}
=\sqrt{\bar\alpha_{t-1}}\left[1-\frac{\alpha_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\right].
$$

方括号与方差那一步**完全相同**（都等于 $\frac{\beta_t}{1-\bar\alpha_t}$），于是

$$
x_0\text{ 的系数}=\sqrt{\bar\alpha_{t-1}}\cdot\frac{\beta_t}{1-\bar\alpha_t},\qquad
\mathbb E[x_{t-1}\mid x_t,x_0]=\tilde\mu_t.\ \checkmark
$$

两条路线殊途同归。（方差和均值的化简里反复出现同一个方括号不是巧合——它就是"条件化从 $x_{t-1}$ 的不确定性里扣掉了与 $x_t$ 相关的那部分"的代数痕迹。）

### 4.3 顺手得到的精确恒等式：$\varepsilon$ 形式

从闭式解 $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon$ 反解

$$
x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon}{\sqrt{\bar\alpha_t}},
$$

代入 $\tilde\mu_t$ 的表达式（注意 $\frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}}=\frac{1}{\sqrt{\alpha_t}}$）：

$$
\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\cdot\frac{x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon}{\sqrt{\bar\alpha_t}}
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t.
$$

分别收拢 $x_t$ 与 $\varepsilon$ 的系数：

**$x_t$ 的系数**（通分到 $\sqrt{\alpha_t}(1-\bar\alpha_t)$）：

$$
\frac{\beta_t}{\sqrt{\alpha_t}(1-\bar\alpha_t)}+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}
=\frac{\beta_t+\alpha_t(1-\bar\alpha_{t-1})}{\sqrt{\alpha_t}(1-\bar\alpha_t)}
\overset{(\star)}{=}\frac{1}{\sqrt{\alpha_t}}.
$$

**$\varepsilon$ 的系数**（$\sqrt{1-\bar\alpha_t}$ 与分母约掉一个）：

$$
-\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\cdot\frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}
=-\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}.
$$

合并：

$$
\tilde\mu_t=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon\right).
$$

数字标本：$\varepsilon=\frac{1.5-0.97015\times2}{\sqrt{0.058808}}=\frac{-0.4403}{0.24251}=-1.8157$，代入右边 $=\frac{1.5-0.02\times(-1.8157)/0.24251}{0.98995}=1.6665$，与左边吻合（六位小数）。

**这个形式比原式重要得多**：$\tilde\mu_t$ 里不再有不可知的 $x_0$，只剩"当前观测 $x_t$ + 一步噪声 $\varepsilon$"。DDPM 训练时网络 $\varepsilon_\theta(x_t,t)$ 预测的正是这个 $\varepsilon$，正文 §2.3 的损失函数 $\|\varepsilon-\varepsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon,\,t)\|^2$ 就是从这里长出来的。

---

## 5. Monte Carlo 验证

公式再漂亮也要过数值这一关。标量设定同 §3.6：$\beta_t\equiv 0.02$、$t=3$、$x_0=2$。流程：

1. 采 $x_{t-1}\sim q(x_{t-1}\mid x_0)=\mathcal N(1.96,\ 0.0396)$；
2. 采 $x_t\sim q(x_t\mid x_{t-1})=\mathcal N(0.98\,x_{t-1},\ 0.02)$；
3. 只保留 $x_t$ 落在窄带（宽 $0.02$）内的样本，统计这些样本的 $x_{t-1}$ 经验均值/方差——这就是对 $x_t$ 做数值条件化（$4\times10^6$ 样本）：

| 条件 $x_t\approx$ | MC 均值 | 公式 $\tilde\mu_t$ | MC 方差 | 公式 $\tilde\beta_t$ |
|---|---|---|---|---|
| $1.5$ | $1.6669$ | $1.6665$ | $0.01351$ | $0.0134676$ |
| $2.0$ | $2.0001$ | $1.9998$ | $0.01349$ | $0.0134676$ |

均值 4 位小数吻合，方差在窄带统计涨落内吻合。三处细节再读一遍：

- $x_t=1.5$ 行就是 §3.6 数字标本的"实验版"：公式 $1.6665$、实验 $1.6669$；
- **$x_t=2=x_0$ 时 $\tilde\mu_t=1.9998\ne 2$**：若权重和恰为 1，这里应严格等于 $2$；差出的 $2\times10^{-4}$ 正是 §3.7"权重和 $=0.99990$"的指纹；
- **$\tilde\beta_t=0.0134676<\beta_t=0.02$**：条件化确实让分布更尖了，印证"精度相加"。

---

## 6. 这两条公式在正文里承担什么

| 位置 | 用到的性质 | 作用 |
|---|---|---|
| §2.2 反向过程 | $\tilde\mu_t$ 含 $x_0$ | 反向过程"需要知道答案才能算"——循环依赖，由此引出 DDPM 的解法：用网络从 $x_t$ 猜 $\hat x_0$（或直接猜 $\varepsilon$） |
| §2.3 训练目标 | KL 两两高斯、$\tilde\beta_t$ 不含 $\theta$ | 变分下界拆出的每项 KL 是 $\mathcal N(\tilde\mu_t,\tilde\beta_t)$ 对网络高斯的 KL；方差项与 $\theta$ 无关被丢掉，训练实际只压 $\mu_\theta\to\tilde\mu_t$，等价于 §4.3 的 $\varepsilon$ 回归 |
| §2.2 凝结直觉 | $\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ | $t=1$ 时分子 $1-\bar\alpha_0=0$，方差为 $0$——反向最后一步完全确定；比值随 $t$ 增大，反向越走越确定 |

**一句话总结**：反向高斯 = 两个高斯信息源的贝叶斯融合——精度相加给方差（$\tilde\beta_t$，比前向一步还小），精度加权平均给均值（$\tilde\mu_t$，$x_0$ 与 $x_t$ 的近似凸组合，精确形式是 $\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon)$）；全部代数难点只有一处——方差守恒账本 $\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\bar\alpha_t$。

---

*栏目导航：[习题解答](../../solutions/)* | *本题出处：[VG 01 §8](../../video_generation/VG_01_diffusion_basics/)*
