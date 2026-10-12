---
title: "VG 01: 扩散模型基础——从 DDPM 到 Flow Matching 与 CFG"
date: 2026-09-28
math: true
tags:
  - Notes
  - Video Generation
  - Diffusion Model
categories:
  - Video Generation
description: "DDPM 前向/反向过程完整推导、训练目标的变分下界来源、DDIM 与概率流 ODE、Flow Matching 统一视角、CFG 引导机制与数值实验"
---

# VG 01：扩散模型基础——从 DDPM 到 Flow Matching 与 CFG

> 📖 本系列精读 [Hanyang Wang](https://hanyang-21.github.io) 的 9 篇论文（视频生成 × 3D 方向）。本篇是系列地基：后面每一篇（ReconX 的视频扩散先验、VideoScene 的蒸馏、CFG-Ctrl 的引导重构）都建立在这篇的推导之上。
> 阅读前提：概率论基础（高斯分布、条件期望）、微积分。不需要先验扩散模型知识。
>
> 📂 **阅读方式**：正文只保留主线——结论、关键公式与总结表；推导、数字标本和详细解释收进折叠块（点击展开），带虚线下划线的论断可悬停查看依据。学过的读者沿主线快速回忆，初学者按需展开折叠块深入学习。

---

## 1. 这一篇解决什么问题？

生成模型的核心问题：**怎么从"只会采样噪声"的计算机里变出"符合数据分布"的样本？**

### 1.1 出发点：计算机为什么「只会采样噪声」？

这句话值得拆开说清楚——它是整个领域的公理级出发点，也解释了为什么后面所有模型（GAN、VAE、扩散、流）全部从 $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I)$ 出发。答案一句话：**计算机能直接采样的分布，只有「有闭式结构的简单分布」**。本节把这个句子里的每个零件都拆开检查：什么是「闭式」（一张词典表）、伪随机数发生器到底在做什么、两个经典变换的完整细节、闭式密度/CDF 为什么是一切的门票、以及为什么自然图像这两条都不占。

<details>
<summary><b>先把「闭式」这个词说清楚：五个近义词的词典表</b></summary>

「闭式」（closed-form）在不同语境里指的东西有细微差别，而本篇会用到其中五个。一次说清：

| 术语 | 定义 | 正例 | 反例 |
|---|---|---|---|
| 闭式表达 (closed-form expression) | 用**有限个**初等运算（$+,-,\times,\div$、幂、指数、对数、三角）与已知函数复合出的公式；不含极限、积分、无穷级数、递归 | $f(x)=\frac{1}{\sqrt{2\pi}}e^{-x^2/2}$ | $f(x)=\sum_{n=1}^{\infty}\frac{x^n}{n!}$（无穷级数）、$f(x)=\int_0^x e^{-t^2}\,dt$（积分定义） |
| 闭式解 (closed-form solution) | 方程/问题的解能写成闭式表达——给定参数**一步算出**答案，不用迭代逼近 | 一元二次方程：$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$ | 五次及以上多项式求根（Abel–Ruffini 定理：不存在通用根式公式）；ODE $\frac{dx}{dt}=x^2$ 一般只能数值积分 |
| 闭式密度 | 概率密度 $p(x)$ 有闭式表达 | 高斯密度、指数密度 $p(x)=\lambda e^{-\lambda x}$ | $p_{\text{data}}$（根本没有公式） |
| 闭式 CDF | 累积分布函数 $F(x)=P(X\le x)$ 有闭式表达 | 指数分布 $F(x)=1-e^{-\lambda x}$ | **高斯的 CDF** $\Phi(x)=\int_{-\infty}^{x}\frac{1}{\sqrt{2\pi}}e^{-t^2/2}dt$——**没有初等闭式**（写不出有限个初等函数的组合，只能借助 $\mathrm{erf}$ 函数或数值近似） |
| 闭式结构（本节用法） | 上面几条的统称：一个分布被**有限个参数 + 闭式公式**完整描述，采样链条上的每一步都可算 | $\mathcal N(\mu,\sigma^2)$：两个参数 + 密度公式 + 已知采样算法 | $p_{\text{data}}$：无参数、无公式、无算法 |

一个最容易踩的坑值得单独标出来：**「高斯有闭式密度」$\ne$「高斯有闭式 CDF」**。正态密度本身是闭式表达，但把它积起来的 CDF 积不出初等函数——这不是数学家偷懒，是被证明积不出来的（属于 Liouville 定理一族的结果）。这个差别马上会兑现：逆变换采样需要的是**可逆的 CDF**，所以它对指数分布好用、对高斯却不好用，高斯必须绕道 Box–Muller。

本篇后文还会遇到三次「闭式」，全是同一件事的化身：§2.1 前向过程 $x_t$ 的**闭式解**（加噪 1000 步不用逐步模拟、一步到位）；§2.2 反向高斯的**闭式配方**；§3–§5 toy 实验的**闭式 score**（分布是我们自己造的、score 有公式，所以能拿来校验采样器）。共同点始终是四个字：**公式可算**。

</details>

<details>
<summary><b>伪随机数：确定性机器里的「随机」</b></summary>

计算机是确定性机器——同样的输入永远得到同样的输出，机器里物理上并没有骰子（真随机要外接硬件熵源；统计与机器学习恰恰**不要**真随机，要的是可复现）。所以「随机数发生器」其实是**伪随机**（pseudorandom）：一个确定性递推

$$
s_{n+1} = f(s_n), \qquad u_n = g(s_n)\in(0,1),
$$

初值 $s_0$ 叫**种子**（seed）。整个序列由种子唯一决定——同一种子永远产生同一序列，这是可复现实验的基石。

**线性同余生成器（LCG, linear congruential generator）**，最古老的一类：

$$
s_{n+1} = (a\cdot s_n + c)\ \bmod\ m .
$$

手算标本（$a=5,\ c=1,\ m=16,\ s_0=3$）：

```text
3 → 0 → 1 → 6 → 15 → 12 → 13 → 2 → 11 → 8 → 9 → 14 → 7 → 4 → 5 → 10 → 3（回到起点）
```

周期最多 $m$（这个标本 16 步绕圈，恰好取满）。真实参数如 Park–Miller「最小标准」$m=2^{31}-1,\ a=16807,\ c=0$，周期约 $2.1\times10^9$。LCG 的经典毛病：**低位比特周期极短**——$m$ 取 $2^k$ 时最低位以周期 2 交替 0101…，历史上老虎机与早期加密系统被这类规律坑过真实损失。

**梅森旋转（Mersenne Twister, MT19937）**：Python/NumPy 标准库 `random` 的默认后端。基于 624 个字的移位寄存器 + 位运算混淆（tempering），周期 $2^{19937}-1$（一个约 6000 位十进制数字的素数——「梅森」之名的来源），通过绝大多数统计随机性测试。它快、统计性质好，但**完全可预测**：观察到 624 个连续输出即可复原内部状态、预测之后的每一个数——所以绝不能用于密码学（密码场景换 CSPRNG，如系统熵池）。

要点收拢成一句：无论 LCG 还是梅森旋转，**机器真正「会采」的只有均匀分布**——把状态整数归一化成独立同分布的 $U\sim\mathrm{Unif}(0,1)$（如取 53 个比特 $u=(b_1\cdots b_{53})_2/2^{53}$）。其它一切分布都要从 $U$ **变换**出来。

</details>

<details>
<summary><b>两个经典变换的细节</b></summary>

**逆变换采样（inverse transform sampling）**：若 CDF $F$ 可求逆，则

$$
X = F^{-1}(U),\quad U\sim\mathrm{Unif}(0,1)\quad\Longrightarrow\quad X\sim F .
$$

证明三行（概率积分变换，probability integral transform）：对 $F$ 的连续点 $x$，

$$
P(X\le x) = P\big(F^{-1}(U)\le x\big) = P\big(U\le F(x)\big) = F(x).\ \blacksquare
$$

最后一步只用了「均匀分布的 CDF 是恒等函数」。反过来 $F(X)$ 也均匀——同一枚硬币的两面，统计模拟里做概率纸检验（Q–Q 图）用的就是它。

数字标本（指数分布，$\lambda=1$）：$F(x)=1-e^{-x}$，解出 $F^{-1}(u)=-\ln(1-u)$。采到 $u=0.5$ 时 $x=\ln 2\approx0.693$（中位数）；$u=0.9$ 时 $x=-\ln 0.1\approx2.303$（90 分位）。「查分位数表」的机制就长这样。

**它对高斯为什么失效**：正是上面那个坑——$\Phi$ 没有初等闭式，$\Phi^{-1}$ 更没有；硬走逆变换就得每个样本数值求根，慢而尴尬。

**Box–Muller 变换**：绕开 $\Phi^{-1}$，两个独立均匀样本直接生产两个独立标准高斯：

$$
Z_1=\sqrt{-2\ln U_1}\,\cos(2\pi U_2),\qquad Z_2=\sqrt{-2\ln U_1}\,\sin(2\pi U_2).
$$

推导骨架（极坐标的胜利）：$(Z_1,Z_2)$ 的联合密度 $\frac{1}{2\pi}e^{-(z_1^2+z_2^2)/2}$ 换到极坐标 $(r,\theta)$ 后**变量分离**——$\theta\sim\mathrm{Unif}(0,2\pi)$ 与 $r$ 独立，而 $r^2=Z_1^2+Z_2^2$ 服从 $\mathrm{Exp}(\tfrac12)$（指数分布恰好闭式 CDF 可逆！）。于是 $r^2=-2\ln U_1$、$\theta=2\pi U_2$，展开回直角坐标即得。

数字标本：$U_1=0.5,\ U_2=0.25$ 时 $r=\sqrt{-2\ln 0.5}=\sqrt{2\ln 2}\approx1.177$，$\theta=\pi/2$，得 $(Z_1,Z_2)=(1.177\cos\frac{\pi}{2},\ 1.177\sin\frac{\pi}{2})=(0,\ 1.177)$。

工程注记：生产级标准库实际用 **Marsaglia 极坐标法**（拒绝采样取单位圆内点，免掉三角函数）或 **Ziggurat**（分层查表，最快）——效率在变，结构从不变：**均匀样本 + 闭式公式**。

| 采样技术 | 需要什么 | 能采什么 |
|---|---|---|
| 逆变换 | $F$ 闭式且可逆 | 指数、柯西、Weibull…… |
| Box–Muller / 极坐标法 | 只要均匀样本 + 三角/对数 | 标准高斯 |
| 接受-拒绝 | 目标密度逐点可算 + 一个可采样的包络 | Gamma、Beta 等 |
| 卷积 | 独立和的分布（粗糙版：12 个均匀之和 $\approx$ 高斯——CLT 的手工应用，方差 $\frac{1}{12}\times12=1$） | 可分解的分布 |
| Ziggurat | 密度形状的分层表 | 高斯/指数（生产级速度） |

这张表读法：每一行都在「简单分布 + 闭式公式」的地基上盖楼。它就是「简单分布」的**可操作定义**。

</details>

<details>
<summary><b>为什么自然图像是像素空间的低维流形</b></summary>

这句断言是「生成模型难」的几何根源，拆成四层论证。**流形**（manifold）：局部看起来像 $\mathbb R^k$ 的空间——数据点虽身处 $d$ 维像素空间，却集中在一个维数 $k\ll d$ 的弯曲子集附近。

1. **参数计数（生成自由度）**：一张 $1024\times1024\times3$ 图像是 $d\approx3\times10^6$ 维空间的点，但决定它的是稀少得多的因果参数：3D 场景几何与材质、相机位姿（6 个自由度）、光源方向与强度、物体身份与表情——合计大约几十到几千个自由度。**世界的成像过程本来就是「低维参数 → 高维像素」的映射**；$p_{\text{data}}$ 的支撑集是这个映射的像，维数由参数维数封顶，而不是由像素数决定。
2. **局部光滑（邻域结构）**：流形上「语义相邻」的图像（同一场景旋转 1°、光照微调）对应像素空间里的连续小位移——数据集在局部像一片低维曲面。经验佐证：自然图像做 PCA，前几十到几百个主成分就解释了绝大部分方差；JPEG 能压 10–50 倍而人眼几乎无感——**冗余度高 = 内在维数低**的日常证据。
3. **测度论证（「体积几乎为零」的严格含义）**：在 $d=3\times10^6$ 维空间中**均匀**撒一个点，落在「看起来像自然图像」的集合里的概率几乎为零——典型样本是逐像素独立噪声。合法图像集合是勒贝格零测集。（**勒贝格测度**是长度/面积/体积向任意集合的推广；**零测**指该集合可被可数个区间/方体覆盖且总体积任意小——单点零测，可数个零测集之并仍零测；零测 $\ne$ 稀少或可数，Cantor 集不可数却零测。这个测度从哪来？泛函分析系列从另一个方向撞见了它：$(C[0,1],d_p)$ 的完备化正是 $L^p$——要填平均收敛留下的洞，必须把 Lebesgue 可积函数请进来，见[泛函分析 02：完备性与紧性](../../functional_analysis/FA_02_completeness_compactness/) §3.3–3.4；「不可数却零测」的 Cantor 集标本见[泛函分析 10](../../functional_analysis/FA_10_uniform_boundedness_weak_convergence/) 的范畴表。）「低维流形」正是这个直观的数学化：$k$ 维曲面在 $d\gg k$ 维空间中体积为零，但全部概率质量压在它上面。
4. **反证（扩散自己就是证据）**：DDPM 能从 $x_t$ 一步步去噪恢复图像，说明去噪器能分辨「流形上的点」与「流形外的噪声」；若数据真铺满 $10^6$ 维，处处都是「合法图像」，去噪无从下手。

低维流形 + 无公式，给 $p_{\text{data}}$ 判了**三重死刑**，正好逐条杀死上面的采样技术表：

| 采样技术的前提 | $p_{\text{data}}$ 的现状 |
|---|---|
| 闭式密度（逐点可算） | 无公式——分布只由数据集**隐式定义**，任一点的密度值都算不出 |
| 闭式可逆 CDF（低维查表） | 高维连表都写不出：逐坐标分位数需要全部条件分布 $F(x_2\mid x_1),\,F(x_3\mid x_1,x_2),\dots$（Rosenblatt 变换），一个都拿不到 |
| 低维（几何直观/公式可用） | $10^6$ 维，且质量集中在零测流形上、密度极度尖峰化——把它当普通 $d$ 维密度处理的任何方法都会撞墙 |

（这个几何事实同时解释了「两阶段」设计的合理性：3D VAE / latent diffusion 先把数据压进低维隐空间——**先找流形的坐标**，再在低维空间里建模，§6 的视频管线正是这么做的。）

</details>

<details>
<summary><b>逐条验尸：采样技术表是怎么死的</b></summary>

上表的判决分两个层次，一层比一层深。**表层死因（无公式）**：$p_{\text{data}}$ 只由数据集隐式定义，密度在任何一点的值都算不出——拒绝采样与 Ziggurat（要逐点密度）、逆变换（要 CDF）连门票都出示不了。**深层死因（流形 ⟹ 公式原则上不存在）**：支撑在零测集上的分布对 $d$ 维 Lebesgue 测度是**奇异**的——它在环境空间里根本没有密度函数。不是"公式难写"，是"作为普通函数不存在"：密度在流形上是无限尖峰（delta 型）、流形外恒为零，CDF 退化成魔鬼阶梯般的病态对象。逐条对号：

1. **逆变换**：一维就挂（无公式）；高维更挂——多元逆变换实际是 Rosenblatt 链 $F(x_1),\ F(x_2\mid x_1),\ F(x_3\mid x_1,x_2),\dots$，每个条件分布都要已知且可逆，一个都没有；
2. **Box–Muller**：它不是通用算法，是**高斯专属**的坐标变换——靠的是径向对称 + 极坐标下变量分离这份意外之喜。$p_{\text{data}}$ 没有任何已知对称性可利用（高维标准高斯能采，恰因各坐标独立可分解；而图像的像素强相关）。把均匀样本变成图像的那个变换，**本身就是我们要学的 $T$**——Box–Muller 之于高斯 = 生成模型之于 $p_{\text{data}}$；
3. **接受-拒绝**：流形给出最干脆的死刑——任何连续的 proposal 分布 $g$ 给零测集分配的概率是 $0$，**接受率精确为零**。低维直觉版：单位正方形里均匀撒点、要求恰好落在对角线上——概率就是 0。即便数据只"贴近"流形而非严格在其上，高维接受率也随维数指数衰减；
4. **卷积**：只能组装已会采样的零件，$p_{\text{data}}$ 不是任何已知分布的组合；
5. **Ziggurat**：建表要逐点密度，回到死因一。

一句话：那张表的每一行都默认「分布在环境坐标系里有公式」，而 $p_{\text{data}}$ 三重违约——**无公式（算不出）、高维（数值替代失效）、流形（公式原则上不存在）**。

</details>

#### 汇合：生成 = 学一个搬运变换

所以「生成」问题的本质是：**学一个变换 $T$，把能直接采样的简单分布搬运成采不出来的数据分布**：

$$
\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I)\ \xrightarrow{\quad T\quad}\ \mathbf x\sim p_{\text{data}}.
$$

噪声不是生成模型的审美选择，是**计算机能力边界决定的唯一入口**。而 $T$ 太复杂（要吞下流形几何与数据的全部相关结构），写出闭式绝无可能——所以只能**学**。三条经典路线的分歧只在「怎么学 $T$」。

### 1.2 GAN 的病：博弈结构决定的三个症状

GAN（Generative Adversarial Network，生成对抗网络）的方案：生成器 $G$ 造样本，判别器 $D$ 猜真伪，解极小极大博弈

$$
\min_G\max_D\ \ \mathbb E_{\mathbf x\sim p_{\text{data}}}[\log D(\mathbf x)] + \mathbb E_{\mathbf z}[\log(1-D(G(\mathbf z)))].
$$

**逐项读懂这个目标**：两项期望各自是一场「考试」——$D$ 同时参加两场，两场都想拿高分：

| 期望 | 考题 | $D$ 的最优策略 |
|---|---|---|
| $\mathbb E_{\mathbf x\sim p_{\text{data}}}[\log D(\mathbf x)]$ | 给**真图**打分：$D(\mathbf x)$ 是「我认为它是真的」的概率 | 对每个真图回答 $D(\mathbf x)\to1$（$\log$ 在 1 处取最大值 0） |
| $\mathbb E_{\mathbf z}[\log(1-D(G(\mathbf z)))]$ | 给**假图**（$G(\mathbf z)$）打分 | 对每个假图回答 $D(G(\mathbf z))\to0$，即 $1-D\to1$、$\log$ 项 $\to0$ |

两项加起来是 $D$ 的「判别准确率」的对数似然（这正是二元交叉熵的负号版本）：$\max_D$ 让 $D$ 成为尽可能准的侦探。而 $\min_G$ 站在对面：$G$ 唯一能动的是假图本身——把 $G(\mathbf z)$ 造得让 $D$ 忍不住打高分，第二项的 $\log(1-D(G(\mathbf z)))$ 被推向 $-\infty$ 一侧（$D$ 越确信「这是真的」，这项越小）。**一个目标、两个视角**：固定 $G$ 看，是 $D$ 的最大化问题；固定 $D$ 看，是 $G$ 的最小化问题；二者纠缠成博弈。

**为什么用 $\log$ 而不是裸概率？** 三个理由。①**梯度形状**：$\log(1-D)$ 在 $D\to1$（$G$ 骗术太差被识破）处梯度大、在 $D\to0$ 处饱和——原版 GAN 论文还证明可换成 $-\log D(G(\mathbf z))$（non-saturating 版：让 $G$ 直接最大化「被判为真」的概率），解决 $G$ 早期梯度消失的问题；②**统计解释**：交叉熵就是「用 $D$ 的判断给样本编码所需的信息量」，最大化它 = 让 $D$ 的判断尽量有信息量；③**理论落点**：最优判别器有闭式解 $D^*(\mathbf x)=\frac{p_{\text{data}}(\mathbf x)}{p_{\text{data}}(\mathbf x)+p_G(\mathbf x)}$（贝叶斯后验），代回目标后 $\min_G$ 的问题恰好变成最小化 $\mathrm{KL}(p_{\text{data}}\,\|\,p_G)$ 型的分布距离（JS 散度）——**博弈的均衡点 = 两个分布重合**，这是 GAN 理论成立的全部根据，也是 §1.2 三条病状的出发点。

「训练不稳定」不是工程瑕疵，是博弈结构的必然：

1. **要的是均衡不是极值**：普通训练找一个损失的极小值（有全局下降方向），GAN 找的是双人博弈的纳什均衡——一方的最优解依赖对方的策略。$G$ 的梯度信号质量完全取决于对手 $D$ 的水平，而 $D$ 在整个训练过程中漂移：$D$ 训练得太强，$\log(1-D(G(\mathbf z)))$ 饱和、$G$ 的梯度消失；$D$ 太弱，梯度干脆指向错误方向。**没有「把 loss 降下去」这种安全的推进感**，超参、学习率平衡（TTUR）、谱归一化全是打在这个结构上的补丁；
2. **覆盖率无惩罚 → 模式坍缩**（mode collapse）：$D$ 逐样本打真伪分，从不比较分布整体。$G$ 一旦发现把全部概率质量压到数据的单一模式（比如只画某一种脸）就能持续骗过 $D$，损失函数里没有任何项阻止它——训练出的模型多样性塌掉、缺失大半个数据分布；
3. **无显式似然**：$p(\mathbf x)$ 算不出来。既无法度量模型分配给真数据的概率，也无法在分布上做后续操作（对比本系列 VG 09 那种对生成分布本身做优化的工作——没有密度连讨论的起点都没有）。

### 1.3 VAE 的病：条件均值决定的「糊」

VAE（Variational Autoencoder，变分自编码器）的方案：编码器 $q(\mathbf z\mid\mathbf x)$ 把图像压进隐空间，解码器 $p(\mathbf x\mid\mathbf z)$ 还原，最大化证据下界（ELBO）：

$$
\mathcal L = \underbrace{\mathbb E_{q(\mathbf z\mid\mathbf x)}[\log p(\mathbf x\mid\mathbf z)]}_{\text{重构项}} - \underbrace{\mathrm{KL}\big(q(\mathbf z\mid\mathbf x)\,\|\,p(\mathbf z)\big)}_{\text{隐空间正则项}}.
$$

**逐项读懂这个目标**：ELBO 不是拍脑袋的组合，它是 $\log p_\theta(\mathbf x)$（对数证据）的下界，由 Jensen 不等式一步拆出来——训练时只需采样、不用碰棘手的 $p(\mathbf z\mid\mathbf x)$。两项的分工是一根拔河绳：

| 项 | 一句话 | 它单独会怎样 |
|---|---|---|
| 重构项 $\mathbb E_{q(\mathbf z\mid\mathbf x)}[\log p(\mathbf x\mid\mathbf z)]$ | 「编码再解码，要能还原」——**自编码器的记忆** | 单独最大化 = 过拟合的 autoencoder：每个 $\mathbf x$ 在隐空间里占一个孤立的小岛，隐空间没有结构、无法从中采样生成 |
| KL 正则项 $\mathrm{KL}(q(\mathbf z\mid\mathbf x)\,\|\,p(\mathbf z))$ | 「各图的编码分布要贴着统一的先验 $\mathcal N(\mathbf 0,\mathbf I)$」——**组织隐空间的秩序** | 单独最小化 = 所有编码坍缩到同一个分布：$q(\mathbf z\mid\mathbf x)\equiv\mathcal N(\mathbf 0,\mathbf I)$，隐空间不再区分任何图像——重构必然崩 |

<details>
<summary><b>KL 项逐字读：积分定义、熵分解读法、高斯闭式解</b></summary>

**KL 项逐字读**（别被符号吓住，它就量「两个分布差多远」）：

$$
\mathrm{KL}\big(q\,\|\,p\big)=\int q(\mathbf z)\,\log\frac{q(\mathbf z)}{p(\mathbf z)}\,d\mathbf z
=\underbrace{-H(q)}_{\text{编码的散乱度}}+\underbrace{\mathbb E_q[\log p(\mathbf z)]}_{\text{向先验聚拢}}\ \ge 0,
$$

恒等于零当且仅当 $q=p$。对高斯先验它有闭式解（均值向量与协方差矩阵的显式公式），所以这一项**不需要采样、精确可算**——VAE 训练信号里唯一「干净」的部分。

</details>

上面是速读。下面把两条公式的推导补全：ELBO 从哪来（三行链 + gap 恒等式），KL 的非负性、不对称与高斯闭式解（含具体数字演算）；外加两个前置小补丁——$p$/$q$ 记号谁算得出谁算不出、凹函数到底是什么。
**先摆正五个记号的角色**（最容易混的一步——裸写的 $p(\mathbf x)$ **不是**真实图分布：那是 §1.1 的 $p_{\text{data}}$，它在推导里从不出现。这里的 $p(\mathbf x)$ 是**模型自己的**边缘似然，是 $\theta$ 的函数——省略下标是 VAE 原论文传下来的记号惯例）：

| 记号 | 角色 | 可不可得 |
|---|---|---|
| $p_{\text{data}}(\mathbf x)$ | 真实图分布（世界定的） | **不进推导**——只「出题」：训练图 $\mathbf x$ 从它抽出来，抽到即冻结成常数 |
| $p(\mathbf x\mid\mathbf z)$ | 解码器：给 $\mathbf z$ 输出图像分布 | **可得**——围绕网络输出定义的分布 $\mathcal N\big(f(\mathbf z),\sigma^2\mathbf I\big)$（网络只提供均值参数；「网络 vs 解码器分布」的两层区分见四环链第 0 步），正向问一次就有一个数 |
| $p(\mathbf z)$ | 先验 | **可得**——手工指定为 $\mathcal N(\mathbf 0,\mathbf I)$，密度闭式可算 |
| $p(\mathbf x)=\int p(\mathbf x\mid\mathbf z)\,p(\mathbf z)\,d\mathbf z$ | **模型的边缘似然（「证据」）**——「模型认为这张图有多大概率」，由解码器 × 先验积分而成，$\theta$ 动它就动 | **不可得**——那个对全体 $\mathbf z$ 的积分：百万维空间、每个点要解码一次，原则上算不动。**它是 ELBO 推导想优化的主角**，下界 $\mathcal L$ 就是它的可计算替身 |
| $p(\mathbf z\mid\mathbf x)$ | 真后验：看到图 $\mathbf x$ 后 $\mathbf z$ 的真实分布 | **不可得**——贝叶斯公式 $p(\mathbf z\mid\mathbf x)=\frac{p(\mathbf x\mid\mathbf z)\,p(\mathbf z)}{p(\mathbf x)}$ 的分母正是上面那行算不动的积分 |
| $q(\mathbf z\mid\mathbf x)$ | 编码器（变分分布） | **随便造**——见下 |

**为什么 $q$ 可以随便造**：回看下面的推导——往积分里塞 $q$ 那一步对 $q$ **没有提任何要求**，任何归一化分布都合法。$q$ 不需要「正确」才有资格使用；gap 恒等式只是告诉你，$q$ 离真后验越远、下界越松。于是策略变成：**在一族算得动的分布（对角高斯）里，挑让下界最紧的那个**——「变分近似」四个字的全部含义：用可算的替身顶替算不出的真后验，代价被 gap 精确记账。


<details>
<summary><b>ELBO 推导全过程：三行链与 gap 恒等式</b></summary>

推导的起点 $\log p(\mathbf x)$ 不是真实图分布——真实分布 $p_{\text{data}}$ 全程不进推导（无公式）。$p(\mathbf x)$ 是**模型自己的**边缘似然（原论文省略 $\theta$ 下标；本篇从现在起统一带上，与 §2.3 扩散版的 $p_\theta(x_0)$ 同款记号）。它是「模型认为这张图有多大概率」：这张图可以由任何一个 $\mathbf z$ 解码产生，把所有 $\mathbf z$ 的贡献全部积掉——被积函数 $p(\mathbf x\mid\mathbf z)\,p(\mathbf z)$ 全是自己定义的零件，没有任何 $p_{\text{data}}$ 的成分。**正因为它依赖 $\theta$ 才值得优化**：若它是真实分布，便是一个与 $\theta$ 无关的常数，对它做任何优化都是空转。

真实分布的角色只有一个——**出题人**：训练图 $\mathbf x$ 先从 $p_{\text{data}}$ 抽出来，抽到后在本节推导里就是一个冻结的常数；整个训练目标是把这些题的平均分抬高，$\max_\theta\ \mathbb E_{\mathbf x\sim p_{\text{data}}}[\log p_\theta(\mathbf x)]$。为什么「给真实图打高分」等于「拟合真实分布」？又是认-KL 那一手（式 (1.1)）：

$$
\mathbb E_{p_{\text{data}}}[\log p_\theta(\mathbf x)]
= \underbrace{-H(p_{\text{data}})}_{\text{与 }\theta\text{ 无关}} - \mathrm{KL}\big(p_{\text{data}}\ \big\|\ p_\theta\big)
\qquad\Longrightarrow\qquad \max \text{ 平均似然} \iff \min\ \mathrm{KL}(p_{\text{data}}\|p_\theta).
$$

**推导的起点**：模型给这张图的对数概率——对数证据 $\log p_\theta(\mathbf x)$：

$$
\log p_\theta(\mathbf x) = \log \int p(\mathbf x\mid\mathbf z)\,p(\mathbf z)\,d\mathbf z.
$$

这个积分对神经网络不可算：要知道每个 $\mathbf z$ 处的 $p(\mathbf x\mid\mathbf z)$ 就得解码一次，像素级隐空间里无从枚举。**变分法的全部技巧就一句话：往积分里塞一个自己造的分布 $q(\mathbf z\mid\mathbf x)$，把「算不了的概率」变成「采样能估计的期望」**。塞法是分子分母同乘同除 $q$：

$$
\log p_\theta(\mathbf x) = \log \int q(\mathbf z\mid\mathbf x)\,\frac{p(\mathbf x\mid\mathbf z)\,p(\mathbf z)}{q(\mathbf z\mid\mathbf x)}\,d\mathbf z
= \log\ \mathbb E_{q(\mathbf z\mid\mathbf x)}\!\left[\frac{p(\mathbf x\mid\mathbf z)\,p(\mathbf z)}{q(\mathbf z\mid\mathbf x)}\right].
$$

**凹函数 (concave function)**：图象上任意两点连成的弦都落在曲线下方的函数，等价于

$$
f\big(\lambda a+(1-\lambda)b\big)\ \ge\ \lambda f(a)+(1-\lambda)f(b),\qquad \lambda\in[0,1],
$$

形状直觉：**收益递减**——投入越多、边际回报越少（$\sqrt{x}$、$\log x$、上半圆都是这个形状；镜像概念是凸函数，弦在曲线上方，加速增长）。数字标本（算术-几何平均不等式取对数）：$\log\frac{2+8}{2}=\log 5\approx1.609\ \ge\ \frac{\log 2+\log 8}{2}=\log\sqrt{16}\approx1.386$。下面用的 Jensen 不等式就是这条性质的期望版：$\log\mathbb E[X]\ \ge\ \mathbb E[\log X]$——「先平均再取对数」不小于「先取对数再平均」，因为 $\log$ 把大值压得更狠，先抹平（平均）就少挨压。

现在期望号里是能采样的东西（从 $q$ 采一个 $\mathbf z$，代入右侧公式算一个数），但它还被 $\log$ 罩着。**Jensen 不等式**（$\log$ 是凹函数：$\log\mathbb E[\cdot]\ \ge\ \mathbb E[\log\cdot]$）把 $\log$ 挪进期望号——代价是等号变 $\ge$：

$$
\log p_\theta(\mathbf x)\ \ge\ \mathbb E_{q}\!\left[\log p(\mathbf x\mid\mathbf z)\right] + \mathbb E_{q}\!\left[\log\frac{p(\mathbf z)}{q(\mathbf z\mid\mathbf x)}\right]
= \underbrace{\mathbb E_q[\log p(\mathbf x\mid\mathbf z)]}_{\text{重构项}} - \underbrace{\mathrm{KL}\big(q(\mathbf z\mid\mathbf x)\,\|\,p(\mathbf z)\big)}_{\text{KL 正则项}} = \mathcal L.
$$

右边就是 $\mathcal L$——**证据下界（Evidence Lower BOund）这个名字就是这么来的**。把这「认-KL」一步提炼成一般形式（记作 式 (1.1)，§2.3 推导扩散的 ELBO 时会原样再用）——只有两下：先拆 $\log$，再认 KL 定义：

$$
\mathbb E_q\!\left[\log\frac{p}{q}\right]
\ \overset{\text{拆 }\log}{=}\ \mathbb E_q[\log p]-\mathbb E_q[\log q]
\ \overset{\text{认 KL}}{=}\ -\,\mathrm{KL}(q\ \|\ p),
\qquad \mathrm{KL}(q\|p)\ \triangleq\ \mathbb E_q\!\left[\log\frac{q}{p}\right]
\tag{1.1}
$$

（第一下：$\log\frac{p}{q}=\log p-\log q$，期望号按线性拆成两段；第二下：$\mathbb E_q[\log q]-\mathbb E_q[\log p]=\mathbb E_q\!\left[\log\frac{q}{p}\right]$ 正是 KL 的定义，整体添负号。往后推导中凡出现「期望里的对数比值」，都可用 式 (1.1) 一眼认出它是（负的）KL——下界里所有 KL 项的出处都是这一步。）

**Jensen 这一步丢了什么**：等号成立当且仅当比值 $\frac{p(\mathbf x\mid\mathbf z)\,p(\mathbf z)}{q(\mathbf z\mid\mathbf x)}$ 对 $\mathbf z$ 是常数。下面把这句话的三段论证逐步展开——为什么恰好是「常数」、为什么常数化等价于「$q$ = 真后验」、以及这给训练带来了什么保证。

**第一段：等号条件为什么是「比值为常数」。** 凹函数不等式 $f(\mathbb E[X])\ge\mathbb E[f(X)]$ 的等号条件是 $X$ 取常数值（对严格凹的 $\log$：只要 $X$ 有波动，弦就严格落在曲线下——「抹平后少挨压」严格成立）。这里的随机变量正是 $X=\frac{p(\mathbf x\mid\mathbf z)\,p(\mathbf z)}{q(\mathbf z\mid\mathbf x)}$（在 $q(\mathbf z\mid\mathbf x)$ 下采样），它在 $\mathbf z$ 处取什么值由分子分母共同决定。所以 Jensen 是不是取到等号，完全取决于这个**比值函数是否平**：比值处处相等 ⟹ 期望号内没有波动 ⟹ $\log$ 的压制作用左右抵消 ⟹ 等号。

**第二段：比值平坦当且仅当 $q$ = 真后验。** 关键一步是**用贝叶斯公式改写分子**：联合分布两种分解

$$
p(\mathbf x\mid\mathbf z)\,p(\mathbf z)\;=\;p(\mathbf x,\mathbf z)\;=\;p(\mathbf z\mid\mathbf x)\,p_\theta(\mathbf x),
$$

（左边「先采 $\mathbf z$ 再生成 $\mathbf x$」，右边「先看到 $\mathbf x$ 再反推 $\mathbf z$」——同一块概率质量，两种切法。）代入比值：

$$
\frac{p(\mathbf x\mid\mathbf z)\,p(\mathbf z)}{q(\mathbf z\mid\mathbf x)}
=\frac{p(\mathbf z\mid\mathbf x)}{q(\mathbf z\mid\mathbf x)}\cdot p_\theta(\mathbf x).
$$

$p_\theta(\mathbf x)$ 与 $\mathbf z$ 无关（这张图已经给定），于是「比值对 $\mathbf z$ 是常数」$\iff$「$\frac{p(\mathbf z\mid\mathbf x)}{q(\mathbf z\mid\mathbf x)}$ 对 $\mathbf z$ 是常数」$\iff$ $q(\mathbf z\mid\mathbf x)\propto p(\mathbf z\mid\mathbf x)$。最后钉死比例常数：$q$ 和 $p(\cdot\mid\mathbf x)$ 都是归一化密度（各自积分为 1），若处处 $q=c\cdot p$，两边积分得 $1=c\cdot1$，故 $c=1$。**合起来：Jensen 取等号 $\iff$ 编码器输出的分布精确等于真后验**——「编码器完美逼近」不是近似成立的宽松说法，是等号条件的精确翻译。

**第三段：这个条件给训练的两重含义。**

1. **gap 的来源被精确定位**。Jensen 丢掉的量 = $\log p_\theta(\mathbf x)-\mathcal L$，由上面的分析，它衡量「$q$ 偏离真后验多远」——不是数值误差，不是采样噪声，而是**结构性代价**：我们只肯在「均值 + 对角协方差」的高斯族里找 $q$，而真后验 $p(\mathbf z\mid\mathbf x)$ 对真实数据几乎必然是非高斯、多峰、坐标相关的（一个物体现在隐空间可能对应两个孤岛——朝左或朝右都解释得通这张图）。函数族不含真身，最优解也只能是族内投影，gap 是投影距离。
2. **优化下界时会自动缩 gap**。这一点容易误读成「我在最大化 $\mathcal L$，跟我编码器有什么关系」——有关系的机制是：由下面的 gap 恒等式，$\log p_\theta(\mathbf x)=\mathcal L+\mathrm{KL}(q\|p(\mathbf z\mid\mathbf x))$。对**固定的解码器**，$\log p_\theta(\mathbf x)$ 是定值，抬高 $\mathcal L$ 必然压低那个 KL——**梯度自动流向「把 $q$ 推向真后验」**，不需要任何显式指令。反过来这也解释了一个经典现象：如果解码器太强（比如自回归像素级解码），它可以「绕开」编码器——任意 $\mathbf z$ 都能解码好，$\mathcal L$ 靠重构项就能抬高，KL 项不再被迫缩小，$q$ 学不到后验结构，先验匹配失效（后验坍缩，posterior collapse）。等号条件在这里是诊断书：gap 消失的**唯一**通道是 $q$ 逼近后验，模型可以选择走或不走这条通道。
3. **「完美逼近」在什么时候真的发生**。两种情形值得一提：一是数据本身简单（每张图的真后验恰好接近单峰高斯——接近即可，族内投影几乎无损失），典型如 MNIST 这类低复杂度数据集，VAE 隐空间规整、插值光滑；二是 $q$ 的函数族扩张到足够灵活（normalizing flow 编码器、扩散先验），把「族内最优 ≈ 真后验」变成可达目标——后续 VG 系列讨论的 latent 扩散模型正是沿这条线修补 VAE 的。

一个三行的微型数值标本（离散版，看得见 gap 怎么产生）。设隐变量只有两个取值 $z\in\{a,b\}$，真后验 $p(z\mid\mathbf x)=(0.9,\ 0.1)$，编码器 $q=(0.5,\ 0.5)$（为了例子完整，设 $p(\mathbf x)=0.1$，且 $p(\mathbf x\mid a)=0.18,\ p(\mathbf x\mid b)=0.02$，先验 $p(a)=p(b)=0.5$——满足 $p(\mathbf x)=\sum_z p(\mathbf x\mid z)p(z)$ 自洽）。比值 $X$ 在两点的取值：

$$
X(a)=\frac{0.18\times0.5}{0.5}=0.18,\qquad X(b)=\frac{0.02\times0.5}{0.5}=0.02.
$$

按定义直接算三个量（$\mathcal L$ = 重构项 $-$ 对先验的 KL；对均匀先验的第二项恰为零）：

| 量 | 计算 | 数值 |
|---|---|---|
| $\log p(\mathbf x)$ | $\log 0.1$ | $\approx-2.303$（证据，精确值） |
| $\mathcal L$ | $\mathbb E_q[\log p(\mathbf x\mid\mathbf z)]-\mathrm{KL}(q\|p(\mathbf z))=0.5\log 0.18+0.5\log 0.02-0$ | $\approx-2.813$（下界，采样可估计） |
| gap | $\log p(\mathbf x)-\mathcal L$ | $\approx 0.511$ |

（这里 $\mathrm{KL}(q\|p(\mathbf z))=0.5\log\frac{0.5}{0.5}+0.5\log\frac{0.5}{0.5}=0$——编码器与先验同为均匀分布，两个分布一模一样，KL 恒为零。这是标本的**刻意设计**：正则项归零后 $\mathcal L$ 只剩重构项，gap 的来源被干净地隔离——全部来自「重构项不够高」，没有正则项的干扰。注意它与下一段 $q$ 对**真后验**的 KL $=0.511$ 是两个不同的量：一个进 $\mathcal L$、一个记 gap，别混。）

$q$ 对**真后验**的 KL：$\mathrm{KL}(q\|p(\mathbf z\mid\mathbf x))=0.5\log\frac{0.5}{0.9}+0.5\log\frac{0.5}{0.1}\approx 0.511$——与 gap 逐位吻合（gap 恒等式的数值验证）。比值 $X$ 在 $0.18$ 与 $0.02$ 之间**不平**（差 9 倍），Jensen 严格取不等号，gap $>0$；若把编码器换成 $q=p(\mathbf z\mid\mathbf x)=(0.9,\ 0.1)$，则 $X(a)=\frac{0.18\times0.5}{0.9}=0.1$、$X(b)=\frac{0.02\times0.5}{0.1}=0.1$ 处处相等，gap 精确归零。「完美逼近 ⟹ 下界贴住证据」的完整算术：

| 场景 | 重构项 | $-\mathrm{KL}(q\|p(\mathbf z))$ | $\mathcal L$ | $\log p(\mathbf x)$ | gap |
|---|---|---|---|---|---|
| $q=(0.5,\,0.5)$（均匀） | $0.5\log0.18+0.5\log0.02\approx-2.813$ | $-0$（均匀对均匀） | $-2.813$ | $-2.303$ | $0.511$ |
| $q=(0.9,\,0.1)$（真后验） | $0.9\log0.18+0.1\log0.02\approx-1.935$ | $-\big(0.9\log\tfrac{0.9}{0.5}+0.1\log\tfrac{0.1}{0.5}\big)\approx-0.368$ | $\mathbf{-2.303}$ | $\mathbf{-2.303}$ | $\mathbf{0}$ ✓ |

第二行是这段标本真正值得盯的一眼：**KL 项不再为零**（$-0.368$）——编码器贴向真后验时，重构项升高 $0.879$，但先验惩罚同时付出 $0.368$，一加一减的净差额恰好把 $\mathcal L$ 抬到与 $\log p(\mathbf x)$ **逐位相等**（$-2.302585$）。ELBO 两股力量的非同向性在这里现了形：贴后验要付先验的账，gap 归零靠的是净收支恰好填平——「可以逐位验算」的完整版。

**gap 恰好是另一个 KL：$\mathrm{KL}(q\|p(\mathbf z\mid\mathbf x))$（恒等式）**。上面标本里 gap 与 KL 逐位吻合不是巧合，是下面这条**恒等式**的数值体现。推导只有四步、全是等号——钥匙是：联合密度 $p(\mathbf x,\mathbf z)$ 有两种分解方式，ELBO 用的是生成方向，这里换推断方向重新展开同一个量。

**第 0 步：把 $\mathcal L$ 从「两项式」收回「比值式」。** 回看 Jensen 之前塞入 $q$ 的那个期望（推导里它被 $\log$ 罩着、Jensen 放行后拆成了两项）——不放 $\log$、只取期望，它本身就是 $\mathcal L$ 的另一种等价写法：

$$
\mathcal L=\mathbb E_q\!\left[\log\frac{p(\mathbf x\mid\mathbf z)\,p(\mathbf z)}{q(\mathbf z\mid\mathbf x)}\right]
=\mathbb E_q\!\left[\log\frac{p(\mathbf x,\mathbf z)}{q(\mathbf z\mid\mathbf x)}\right],
$$

最后一步把分子认成**联合分布**（生成方向分解）。这一步给出了下面「取 $\log p(\mathbf x,\mathbf z)$ 的期望」里期望号的出处。

**第 1 步：换一种分解。** 同一块联合概率质量换个切法（推断方向：先看到 $\mathbf x$ 再反推 $\mathbf z$）：

$$
p(\mathbf x,\mathbf z)=p_\theta(\mathbf x)\,p(\mathbf z\mid\mathbf x)
\qquad\Longrightarrow\qquad
\mathcal L=\mathbb E_q\big[\log p_\theta(\mathbf x)\big]+\mathbb E_q\!\left[\log\frac{p(\mathbf z\mid\mathbf x)}{q(\mathbf z\mid\mathbf x)}\right].
$$

**第 2 步：常数提出。** $\log p_\theta(\mathbf x)$ 不含 $\mathbf z$（这张图已冻结），期望等于自身：

$$
\mathcal L=\log p_\theta(\mathbf x)+\mathbb E_q\!\left[\log\frac{p(\mathbf z\mid\mathbf x)}{q(\mathbf z\mid\mathbf x)}\right].
$$

**第 3 步：认-KL（式 (1.1)）再移项。** 最后一个期望正是「期望里的对数比值」，分子分母方向恰好是 $p$ 在上——认出它是 $-\mathrm{KL}\big(q\,\|\,p(\mathbf z\mid\mathbf x)\big)$：

$$
\log p_\theta(\mathbf x) \ =\ \mathcal L \ +\ \mathrm{KL}\big(q(\mathbf z\mid\mathbf x)\,\|\,p(\mathbf z\mid\mathbf x)\big).
\qquad\blacksquare
$$

**为什么是等式而不是不等式**：Jensen 只出现在「$\mathcal L\le\log p_\theta$」那条路线里（$\log$ 挪进期望号那一次）；这条路线是对 $\mathcal L$ **本身**做代数改写，从头到尾没过 Jensen——「不丢信息」的字面意思是四步全是等号，一步近似都没碰。两种分解各司其职：

| 分解方向 | 公式 | $\mathcal L$ 展开成 | 算得动吗 | 用途 |
|---|---|---|---|---|
| 生成方向 | $p(\mathbf x,\mathbf z)=p(\mathbf x\mid\mathbf z)\,p(\mathbf z)$ | 重构项 $-\,\mathrm{KL}(q\|p(\mathbf z))$ | ✓ 两项都可算 | **训练**（梯度下降用它） |
| 推断方向 | $p(\mathbf x,\mathbf z)=p_\theta(\mathbf x)\,p(\mathbf z\mid\mathbf x)$ | 证据 $-\,\mathrm{KL}(q\|\text{真后验})$ | ✗ 后一项不可算 | **记账**（给 gap 命名） |

同一个 $\mathcal L$，训练时用左列、诊断时用右列——「换分解」不是近似，是把同一个量的两副面孔分别派去干不同的活。

这个恒等式逐条读：

| 读法 | 含义 |
|---|---|
| 右边 KL 项 $\ge 0$ | $\mathcal L\le\log p_\theta(\mathbf x)$ 恒成立——「下界」名副其实 |
| gap 是 $q$ 与**真后验**的距离 | 下界松，恰恰因为编码器没把 $p(\mathbf z\mid\mathbf x)$ 模仿到位 |
| 最大化 $\mathcal L$ 一石二鸟 | 既抬高 $\log p_\theta(\mathbf x)$ 本身，又把 $q$ 往真后验推（缩 gap） |
| gap 不可直接算 | $p(\mathbf z\mid\mathbf x)$ 的分母正是开头那个算不了的积分——所以才退而优化下界 |

「变分 (variational)」的名字也在这里：不是对某个固定函数的参数求导，而是**在一个函数族（高斯族）里挑最好的函数**——函数空间里的优化。

**训练怎么算这个期望**：$\mathbb E_q[\log p(\mathbf x\mid\mathbf z)]$ 解析积不出来（解码器是黑箱神经网络），只能采样估计；但「采样」挡梯度。**重参数化 (reparameterization trick)** 把随机性挪出计算图：$\mathbf z=\boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\varepsilon$（$\boldsymbol\varepsilon\sim\mathcal N(\mathbf 0,\mathbf I)$），$\boldsymbol\varepsilon$ 的随机性如同数据噪声般与参数无关，$\mathbf z$ 对 $(\boldsymbol\mu,\boldsymbol\sigma)$ 可导——蒙特卡洛估计与反向传播同时成立。KL 项有闭式解不用采样（前文已说），所以 VAE 的每步梯度 = 重构项的采样梯度 + KL 项的解析梯度。两个方向的「为什么」——梯度到底被什么挡住、换写法后到底通在哪——值得各展开一遍：

<details>
<summary><b>重参数化两端：梯度为什么被挡住、为什么这样能通</b></summary>

**第一端：梯度被什么挡住。** 要算的是 $\nabla_\theta\,\mathbb E_{z\sim q_\theta}[f(z)]$（$f=\log p(\mathbf x\mid\mathbf z)$，$q_\theta=\mathcal N(\boldsymbol\mu(\mathbf x),\mathrm{diag}(\boldsymbol\sigma^2(\mathbf x)))$）。麻烦在**期望的「测度」本身依赖 $\theta$**：蒙特卡洛估计是「掷一个 $z$、代进去算 $f(z)$」，但「掷」这个操作对 $(\boldsymbol\mu,\boldsymbol\sigma)$ 不可导——你只知道 $z$ 的分布随参数变，「这一次掷出的 $z$」与 $(\boldsymbol\mu,\boldsymbol\sigma)$ 之间没有任何函数关系可求导。看计算图：

```text
θ → (μ, σ) → [采样 z ~ N(μ, σ²)] → f(z) → loss
                ↑
        断点：z 不是 (μ,σ) 的函数，反向传播到这里无路可走
```

**第二端：为什么 $z=\boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\varepsilon$ 能通。** 分两问。

*合法吗？*——高斯在仿射变换下封闭（§1.1 两个经典变换的近亲）。一维验证：$\mu+\sigma\varepsilon$（$\varepsilon\sim\mathcal N(0,1)$）的均值是 $\mu+\sigma\cdot0=\mu$，方差是 $\sigma^2\cdot\mathrm{Var}(\varepsilon)=\sigma^2$——分布与 $\mathcal N(\mu,\sigma^2)$ 一模一样。所以这不是近似，是**同一分布的另一种参数化方式**（「重」参数化的字面意思）：不是「分布带参数 $(\mu,\sigma)$」，而是「把 $\mathcal N(0,I)$ 通过映射 $(\mu,\sigma,\varepsilon)\mapsto\mu+\sigma\varepsilon$ 推过去」。数字标本：$\mu=2,\sigma=0.5$，掷 $\varepsilon=1.2$ → $z=2.6$；这么产生的 $z$ 服从 $\mathcal N(2,0.25)$，与直接采 $z\sim\mathcal N(2,0.25)$ 分布完全相同。

*通在哪？*——关键恒等式：$\mathbb E_{z\sim q_\theta}[f(z)]=\mathbb E_{\varepsilon\sim\mathcal N(0,\mathbf I)}[f(\mu+\sigma\varepsilon)]$。**$\theta$ 的依赖从测度挪进了被积函数**：右边期望对 $\varepsilon$ 取，$\varepsilon$ 的分布与 $\theta$ 无关（掷骰子不再携带参数）；而 $z=\mu+\sigma\varepsilon$ 是 $(\mu,\sigma,\varepsilon)$ 的确定性光滑函数，链式法则全程有路：

$$
\frac{\partial z}{\partial\mu}=1,\qquad\frac{\partial z}{\partial\sigma}=\varepsilon\ \ (\text{把掷出的 }\varepsilon\text{ 当数值用}).
$$

计算图断点消失：

```text
θ → (μ, σ) ↘
              z = μ + σ·ε → f(z) → loss      全程可导
ε（像数据一样的外部输入）↗
```

**梯度质量标本**（$f(z)=z^2$，精确答案 $\mathbb E[z^2]=\mu^2+\sigma^2=4.25$、$\partial/\partial\mu=2\mu=4$）：掷 $\varepsilon=1.2$ → $z=2.6$，单样本路径梯度 $=f'(z)\cdot\frac{\partial z}{\partial\mu}=2\times2.6\times1=5.2$；多掷几个 $\varepsilon$ 取平均收敛到 $4$。对照另一条不用重参数化的经典路线（REINFORCE：$\nabla_\theta\mathbb E[f]=\mathbb E[f\cdot\nabla_\theta\log q_\theta(z)]$，从对数导数绕过采样）——同样无偏，但方差大得多。重参数化是**路径梯度**（pathwise gradient），方差小；代价是要求 $z$ 可由噪声经连续变换得到——高斯满足，**离散隐变量就不行**（得换 Gumbel-softmax 那一族技巧）。

**一个伏笔回收**：§2.1 的式 (2.3)——$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon$——正是同一招。DDPM 训练时构造 $x_t$ 能反向传播，靠的就是把随机性收编进与参数无关的 $\varepsilon$；$\sqrt{\bar\alpha_t},\sqrt{1-\bar\alpha_t}$ 扮演 $(\mu,\sigma)$ 的角色，只是它们不靠梯度学习、由调度表给定。扩散模型的训练循环与 VAE 共用这一件底层零件。

</details>

</details>

<details>
<summary><b>KL 散度专题：非负性、不对称、高斯闭式解</b></summary>

**定义与读法**：

$$
\mathrm{KL}(q\,\|\,p)=\mathbb E_q\!\left[\log\frac{q(\mathbf z)}{p(\mathbf z)}\right]=\int q(\mathbf z)\,\log\frac{q(\mathbf z)}{p(\mathbf z)}\,d\mathbf z.
$$

读法：**以 $q$ 为权重，统计 $\log\frac{q}{p}$ 的平均「惊讶度」**——权重是 $q$ 不是 $p$，$q$ 没有质量的地方发生什么都不计费。这个「谁加权」的选择是 KL 一切不对称行为的根源。

**为什么恒 $\ge 0$（Gibbs 不等式，一行证明）**：因为 $-\log$ 是凸函数，Jensen 给出

$$
\mathrm{KL}(q\,\|\,p)=\mathbb E_q\!\left[-\log\frac{p(\mathbf z)}{q(\mathbf z)}\right]\ \ge\ -\log\,\mathbb E_q\!\left[\frac{p(\mathbf z)}{q(\mathbf z)}\right]=-\log\int p(\mathbf z)\,d\mathbf z=-\log 1=0.
$$

等号当且仅当 $\frac{p}{q}$ 在 $q$ 的支撑上为常数（归一化 ⟹ $q=p$）。注意这次用的是**凸**方向（$\mathbb E[-\log X]\ge-\log\mathbb E[X]$），与 ELBO 里凹方向是同一枚硬币的两面。

**性质速查表**（它不是距离）：

| 性质 | 成立？ | 说明 / 反例 |
|---|---|---|
| $\mathrm{KL}(q\|p)\ge 0$，$=0\Leftrightarrow q=p$ | ✓ | 上一行刚证 |
| 对称？ | ✗ | $q=(0.5,0.5)$，$p=(0.9,0.1)$：$\mathrm{KL}(q\|p)=0.5\ln\frac{0.5}{0.9}+0.5\ln\frac{0.5}{0.1}\approx 0.51$；$\mathrm{KL}(p\|q)=0.9\ln\frac{0.9}{0.5}+0.1\ln\frac{0.1}{0.5}\approx 0.36$ |
| 三角不等式？ | ✗ | 由不对称直接破产——它不是度量 (metric) |
| $q$ 有质量而 $p$ 为零 | $+\infty$ | $\log\frac{q}{p}\to\infty$——支撑不包含即无穷罚 |

**不对称的几何意义：谁被优化，谁迁就谁**。$\mathrm{KL}(A\|B)$ 的无穷罚只出现在「$A$ 有质量而 $B$ 没有」处，所以：

| 优化哪个参数 | 约束方向 | 行为画像 |
|---|---|---|
| 第一个（$A$ 可变） | $A$ 有质量而 $B$ 稀疏处被重罚 | $A$ 收进 $B$ 的高密度区——**挤压/模式收敛** |
| 第二个（$B$ 可变） | 原分布有质量而 $B$ 没盖到处被重罚 | $B$ 必须处处盖住——**覆盖 (mass-covering)** |

VAE 的 ELBO 里 $q$ 站在被优化的第一个位置（先验 $p(\mathbf z)=\mathcal N(\mathbf 0,\mathbf I)$ 固定不动）：最小化 $\mathrm{KL}(q\|p)$ 就是把每张图的编码往先验的高密度区里挤——这正是「KL 正则挤压隐空间」的机制本体。读论文遇到 KL 时先看清谁是第一个参数，方向一换行为全换。

**高斯对高斯的闭式解**：设 $q=\mathcal N(\boldsymbol\mu,\mathrm{diag}(\boldsymbol\sigma^2))$、$p=\mathcal N(\mathbf 0,\mathbf I)$，各维独立可拆成 $d$ 个一维 KL 之和。一维情形把 $\log\frac{q}{p}$ 显式写出再取期望：

$$
\log\frac{q(z)}{p(z)}=-\log\sigma+\frac{z^2}{2}-\frac{(z-\mu)^2}{2\sigma^2}
\ \xrightarrow{\ \mathbb E_q\ }\ -\log\sigma+\frac{\mu^2+\sigma^2}{2}-\frac{1}{2}
=\frac{1}{2}\Big(\mu^2+\sigma^2-1-\log\sigma^2\Big),
$$

中间只用了两条二阶矩事实：$\mathbb E_q[z^2]=\mu^2+\sigma^2$（二阶矩）、$\mathbb E_q[(z-\mu)^2]=\sigma^2$（方差的定义）。多维版即

$$
\mathrm{KL}\big(q\|p\big)=\frac{1}{2}\left(\sum_{k=1}^{d}\sigma_k^2+\|\boldsymbol\mu\|^2-d-\sum_{k=1}^{d}\log\sigma_k^2\right).
$$

**用具体数字走一遍**（$d=1$，编码器对某张图输出 $\mu=2,\ \sigma=0.5$）：

$$
\mathrm{KL}=\frac{1}{2}\big(4+0.25-1-\log 0.25\big)=\frac{1}{2}(4+0.25-1+1.386)\approx 2.32.
$$

拆开看三项各自在罚什么：

| 项 | 数值 | 它在罚什么 |
|---|---|---|
| $\mu^2$ | $4$ | 均值离先验中心太远——把编码拉回原点的主力 |
| $\sigma^2-1$ | $-0.75$ | 方差小于先验要罚（部分被下一项抵消） |
| $-\log\sigma^2$ | $+1.386$ | 方差收缩的对数罚——**防坍缩主力**：$\sigma\to 0$ 时它比 $\sigma^2$ 快得多地冲向 $+\infty$ |

后两项合起来 $\sigma^2-1-\log\sigma^2\ \ge\ 0$（令 $t=\sigma^2$，$t-1-\ln t$ 在 $t=1$ 处取零、两侧均正）——「编码既不许缩成点、也不许过分膨胀」的解析表达。若训练时去掉 KL 项，重构压力会把 $\sigma$ 一路压到机器精度（隐变量塌缩的镜像现象）；反之去掉重构项，$\mu$ 归零、$\sigma$ 归 1，编码器对任何图输出同一个分布。

把这套执法放回大图：KL 项在每个编码位置执行「不许缩点、不许乱跑」，它的执法强度就是拔河绳另一端的拉力——下一句「合谋出糊」正是两股力均衡后的落点。

**两项怎么合谋出「糊」**：KL 项把每张图的编码 $q(\mathbf z\mid\mathbf x)$ 摊成一小团有重叠的高斯斑（贴向同一个先验 ⟹ 斑与斑必然部分重叠）。解码器面对重叠区域里的一个 $\mathbf z$，它「见过」多张不同图的编码都覆盖这里——MSE 最优解是输出它们的平均。**重构项要清晰（要求编码分开），KL 项要规整（要求编码重叠），拔河的结果落在中间：既不够分开、又必须重叠——重叠区域的解码 = 平均 = 糊**。训练好之后从 $\mathcal N(\mathbf 0,\mathbf I)$ 采样新 $\mathbf z$，大概率落在「谁都不专属」的中间地带，生成的图像就偏糊——糊被写进了目标函数的均衡里，与训练技巧无关（对应正文病根 1+2 的联合推论）。

</details>

「样本偏糊」同样是数学结构决定的：

1. **逐像素独立的重构似然**：实践中 $p(\mathbf x\mid\mathbf z)$ 取固定方差的各向同性高斯，重构项等价于逐像素 MSE。给定 $\mathbf z$，解码器的**最优输出是条件均值** $\mathbb E[\mathbf x\mid\mathbf z]$；当数据多模态时（同一片 $\mathbf z$ 邻域对应多张都合理的图），条件均值 = 多张图的**像素平均**——平均出来的图像就是模糊的图像。糊不是训练失败，是 MSE 目标的最优解本身；
2. **KL 正则挤压隐空间**：正则项把每个 $q(\mathbf z\mid\mathbf x)$ 都拉向同一先验 $\mathcal N(\mathbf 0,\mathbf I)$，不同图像的隐编码被迫重叠，解码器在重叠区域只能输出「对谁都安全」的平均值——越靠近隐空间中心越糊；
3. **变分近似的 gap**：真实后验 $p(\mathbf z\mid\mathbf x)$ 复杂，$q$ 被限制为简单高斯，ELBO 与 $\log p_\theta(\mathbf x)$ 之间的差距部分地靠「平滑掉后验结构」来缩小——又一次平均化。

上面三条各说了一半，合起来才是完整的因果链。下面把这条链从头到尾走一遍，每步给依据——**为什么「平均」必然「模糊」，而不是经验观察**。

<details>
<summary><b>四环链逐步推导：为什么「平均」必然「模糊」（第 0/0.5/1/2/3 步，含地板论证与 MSE 两证明）</b></summary>

**第 0 步（解码器的最优输出是均值——上面第 1 条的推导）**。先把两个容易滑过去的层次摆正。

*层次一：网络与解码器是两层东西。* $f$ 是**确定性函数**（吃 $\mathbf z$ 吐一个向量），而解码器 $p(\mathbf x\mid\mathbf z)$ 是**围绕网络输出定义的分布** $p(\mathbf x\mid\mathbf z)=\mathcal N(\mathbf x;\,f(\mathbf z),\ \sigma^2\mathbf I)$——网络吐的是分布的**均值参数**，不是「一张图」；$\sigma$ 是手工固定的超参（「传感器噪声宽度」，基本 VAE 不学它）。这正是回归模型的标准结构 $y=f(x)+\varepsilon$：网络给回归函数，模型是围绕它的噪声。

| 层 | 是什么 | 角色 |
|---|---|---|
| 网络 $f$ | 确定性函数，$\theta$ 的载体 | 分布的均值参数化 |
| 解码器 $p(\mathbf x\mid\mathbf z)$ | $\mathcal N\big(f(\mathbf z),\sigma^2\mathbf I\big)$ | 「$\mathbf z$ 对应一族合理的图，中心在 $f(\mathbf z)$」 |

**为什么必须定义成分布、不能就是映射 $\mathbf x=f(\mathbf z)$？** 两个硬理由：①**似然框架需要密度**——训练要算 $\log p(\mathbf x\mid\mathbf z)$，即给**任意**观测图打分；确定性解码器是 delta 分布，对任何 $\mathbf x\neq f(\mathbf z)$ 概率为零（$\log=-\infty$），且 delta 对 Lebesgue 测度根本不是密度，最大似然直接崩塌。§1.1 判了真实分布「无公式」的刑，这里是自己动手给模型装一个**有公式的密度**；②**生成本义就是条件分布**——同一 $\mathbf z$ 应对应一族合理输出（瞳孔朝左/朝右都合理），条件分布才是对这件事的诚实建模。

*层次二：这是单股压力分析，不是整体目标。* 下文「训练要求最大化 $\log p(\mathbf x\mid\mathbf z)$」分析的是**给定 $\mathbf z$，重构项把 $f(\mathbf z)$ 往哪个方向推**——单看一股压力的最优响应。完整目标是整个 ELBO（两股压力的拔河），**重构项单独升不保证 ELBO 升**（反例与「拉高 ELBO 怎么拉高 $\log p_\theta$」的完整论证见下文第 0.5 步）。现在，固定 $\mathbf z$、取 $p(\mathbf x\mid\mathbf z)=\mathcal N(\mathbf x;\,f(\mathbf z),\ \sigma^2\mathbf I)$，则

$$
\log p(\mathbf x\mid\mathbf z)=-\frac{\|\mathbf x-f(\mathbf z)\|^2}{2\sigma^2}+\text{const},
$$

高斯似然**恰好退化为 MSE**（$\sigma^2$ 只是这股压力相对 KL 项的权重）。最大化它 ⟺ 最小化 $\mathbb E\|\mathbf x-f(\mathbf z)\|^2$ ⟺ $f(\mathbf z)=\mathbb E[\mathbf x\mid\mathbf z]$（「MSE 最优解是条件均值」的两个证明收在折叠块）。**逐像素高斯 ⟹ 解码器被逼成均值机器**，这是第一环。

<details>
<summary><b>为什么最小化 MSE 的解恰是条件均值：两个证明 + 一个标本</b></summary>

精确命题：固定 $\mathbf z$，$f(\mathbf z)$ 是常数向量 $c$，$\mathbf x$ 取条件分布，则

$$
\arg\min_{c}\ \mathbb E\big[\|\mathbf x-c\|^2\ \big|\ \mathbf z\big]\ =\ \mathbb E[\mathbf x\mid\mathbf z].
$$

**证明一（求导）**：目标关于 $c$ 是严格凸的二次函数，$\nabla_c\,\mathbb E\|\mathbf x-c\|^2=-2\big(\mathbb E[\mathbf x]-c\big)\overset{!}{=}0\Rightarrow c=\mathbb E[\mathbf x]$；二阶导 $=2\mathbf I\succ0$，凸 ⟹ 唯一全局最小。∎

**证明二（偏差–方差分解，更透明）**：

$$
\mathbb E\|\mathbf x-c\|^2\ =\ \underbrace{\mathbb E\|\mathbf x-\mathbb E[\mathbf x]\|^2}_{\text{方差：与 }c\text{ 无关}}\ +\ \underbrace{\|\mathbb E[\mathbf x]-c\|^2}_{\text{偏差平方：仅 }c=\mathbb E[\mathbf x]\text{ 时为零}}
$$

（展开平方和，交叉项 $2\,\mathbb E[(\mathbf x-\mathbb E\mathbf x)]^\top(\mathbb E\mathbf x-c)=0$——零均值噪声与常数不相关。）第一项是数据的内在随机性，**任何预测器都消不掉**；唯一能压的是第二项，压到零的条件就是 $c=$ 均值。∎

**数字标本**（微型双模态 $\mathbf x\in\{0,10\}$ 各半）：

| $c$ | $\mathbb E\|\mathbf x-c\|^2$ | 分解：方差 $25$ + 偏差² |
|---|---|---|
| $0$（押注左模态） | $50$ | $25+25$ |
| $5$（均值） | $\mathbf{25}$ | $25+0$ |
| $10$（押注右模态） | $50$ | $25+25$ |

押任何一个模态都要付 $25$ 的偏差罚，**骑在中间的均值反而最优**——最小化 MSE 的解**无视多峰结构**，这就是「均值机器」的算术本质。把 $\{0,10\}$ 换成「左瞳图/右瞳图」，$c=5$ 就变成那张半左半右的糊图（第 1–2 步的入口）。另注意 $\mathbb E[\mathbf x\mid\mathbf z]$ 里的「条件」是**真实数据 × 编码器**联合诱导的——KL 项把哪些图推到 $\mathbf z$ 附近，决定了这个平均在平均**什么**（第 3 环的伏笔）。

</details>

**第 0.5 步（地板论证：拉高 ELBO 与 $\log p_\theta$ 的真实关系）**。「优化重构项」如何传导到「抬高似然」？成立的是**地板论证**，不是「重构升 ⟹ ELBO 升 ⟹ $\log p$ 升」的连锁（第一环就不成立——反例就用 ELBO 折叠块里的玩具标本，解码器固定、只动编码器）：

| $q$ | 重构项 | $-\mathrm{KL}(q\|p(\mathbf z))$ | ELBO | gap |
|---|---|---|---|---|
| $(0.5,\,0.5)$ | $-2.813$ | $0$ | $-2.813$ | $0.511$ |
| $(0.9,\,0.1)$=真后验 | $-1.935$ | $-0.368$ | $\mathbf{-2.303}$（峰） | $0$ |
| $(0.99,\,0.01)$ | $-1.737$ | $-0.637$ | $-2.374$ | $0.071$ |
| $(1,\,0)$ | $-1.715$ | $-0.693$ | $-2.408$ | $0.105$ |

第三行到第四行：**重构项还在升**（$-1.737\to-1.715$），KL 项涨得更多，**ELBO 反而降**——重构项的局部最优 $(1,0)$ 的 ELBO 比真后验处的峰低 $0.105$。单项最优 ≠ 整体最优，这就是拔河绳的算术形态。真正成立的链条是三条已证的命题：

1. **点点下界**：$\log p_\theta(\mathbf x)\ge\mathcal L(\theta,q)$ 对每个 $(\theta,q)$ 成立（gap 恒等式 + KL 非负）——ELBO 是似然的**地板**；
2. **地板抬高的含义**：ELBO 改进到 $V$ ⟹ **保证** $\log p_\theta(\mathbf x)\ge V$——不是「$\log p$ 单调跟着走」（两者梯度本来就不同），而是「下界值即保证值」：你不可能把下界推到量本身之上，每抬高一分下界就是抬高一分「有保证的最小似然」；
3. **上确界贴住**：固定 $\theta$，$\sup_q\mathcal L=\log p_\theta(\mathbf x)$（$q=$ 真后验时取到）——bound 家族从下方把似然**包住且可触及**，所以在族内推 $q$ 就是把 bound 往似然贴。

（上表最后一列顺带三次验证记账：解码器不动时 $\log p_\theta(\mathbf x)\equiv-2.303$，且每行 ELBO + gap 加回这个常数；ELBO 的峰恰在 gap 归零处，过峰后重构再升、gap 重新张开、ELBO 掉头向下。另注意别把两个 KL 混了：此表的 gap 是 $q$ 对**真后验**的 KL，「自动缩 gap」的论述以解码器固定为前提；拔河绳说的是 $q$ 对**先验**的正则 KL。）

**第 1 步（多模态 ⟹ 均值落在模态之间的无人区）**。生成的困境在于：给定隐变量 $\mathbf z$，真实数据条件分布 $p(\mathbf x\mid\mathbf z)$ 常常**多峰**——同一低维描述对应多个等价答案（瞳孔朝左/朝右都合理；头发丝细节任意）。设 $\mathbf z$ 的邻域训练数据里有两张图 $\mathbf x^{(1)},\mathbf x^{(2)}$（各占一半概率），则最优输出

$$
f^\*(\mathbf z)=\tfrac12\mathbf x^{(1)}+\tfrac12\mathbf x^{(2)}.
$$

**第 2 步（平均为什么就是模糊——一张图看懂）**。一维信号标本：$\mathbf x^{(1)}$ 是方波、$\mathbf x^{(2)}$ 是它的反相方波（左半为 +1 右半为 −1 vs 左 −1 右 +1——两张「边缘在同一位置、符号相反」的图）：

```text
x1:  +1 +1 +1 +1 | -1 -1 -1 -1      ← 边缘清晰的图（黑白各半）
x2:  -1 -1 -1 -1 | +1 +1 +1 +1      ← 同一结构、颜色对调的图

(x1+x2)/2 =  0  0  0  0 |  0  0  0  0      ← 平均：全灰，边缘消失
```

两张都完全合法的图，平均出来是**全灰**——边缘（图像的信息所在）被精确对消。这不是巧合，是算术：**两幅图在哪些像素上不同，平均值就在哪些像素上向中间灰靠拢**。数学语言：平均只保住低频结构（两图共有的成分），把高频差异（两图各自的细节）对消掉——「模糊」= **丢失高频 = 平均化**，傅里叶意义上几乎是同义词（低通滤波器的白话文就是「取平均」）。多模态越多、模态间差异越大，均值离每个模态越远、越糊。

**第 3 步（KL 保证你一定会踩进多模态区）**。第 2 步的前提「解码器遇到多模态 $\mathbf z$」不是偶然事件，是 KL 项的**制度性安排**：正则把所有编码摊成互相重叠的高斯斑（重叠论证见上方 KL 专题），且**生成时是从先验 $\mathcal N(\mathbf 0,\mathbf I)$ 采样**——训练时从没被任何编码「专属占领」的中间地带，恰是采样最常落入的区域。三环合拢：

| 环 | 结论 | 依据 |
|---|---|---|
| 0 | 解码器 = 均值机器 | 逐像素高斯重构（本节推导） |
| 1 | 均值落在模态之间 | 数据多模态（生成任务的本质不确定性） |
| 2 | 模态之间的均值 = 模糊 | 高频对消（方波标本） |
| 3 | 采样必然落入多模态区 | KL 挤压 + 从先验采样（KL 专题） |

所以「VAE 生成的图偏糊」不是工程缺陷的罗列，是一条**四环逻辑链**的终点——每一环都写进了 ELBO 目标函数本身。对照实验也支持这个诊断：把解码器方差 $\sigma^2$ 当作**可学习**参数（或换离散似然/感知损失），第 0 环被松绑，糊明显缓解——因为最优输出不再被迫是纯均值；把 KL 权重调小，第 3 环松动，生成更锐但隐空间出现「空洞」（采到的 $\mathbf z$ 解码出乱码）——糊与规整是同一枚硬币的两面，只能交换、不能白拿。扩散模型的做法则是釜底抽薪：不要求任何一步给出最终答案，多模态的抉择被拆散到一千步里由随机性逐步敲定（每步只在「这一步往哪边去噪」上做局部单峰回归），根本不存在「一步跳到全局均值」的压力——§2 展开。

</details>

两个病根有一个共同的深层结构：**都在「一步」之内要求模型完成从噪声到图像的全部跳跃**——GAN 把全部压力压在一个博弈均衡点上（所以脆），VAE 把全部压力压在单次解码的条件均值上（所以糊）。

**扩散模型的答案：把"生成"拆成一千个"去噪"小步骤**——每步只是高斯回归（没有对手、就是回归，训练稳定），多模态的不确定性交给逐步去噪的随机性慢慢消解（不需要一步跳到均值，所以不糊）。§2–5 就是这套话的严格版本。

| 路线 | 学 $T$ 的方式 | 一句话病根 |
|---|---|---|
| GAN | 判别器打真伪分，二人博弈 | 博弈无下降方向 + 覆盖率无惩罚（模式坍缩） |
| VAE | 编码-解码 + ELBO | 多模态数据的条件均值 = 像素平均 = 糊 |
| 扩散 | 把 $T$ 拆成一千个去噪小步 | 每步简单（稳定回归），复杂性由步数累积 |

本篇回答四个递进的问题：

1. **DDPM 在训什么**？——前向加噪、反向去噪的完整数学，以及训练目标为什么是"预测噪声"（从变分下界推出来，不是拍脑袋）；
2. **采样一定要随机吗**？——DDIM 与概率流 ODE：扩散采样可以变成完全确定性的 ODE 求解（这是 VideoScene 一步生成的前提）；
3. **Flow Matching 和 DDPM 是什么关系**？——SD3/Flux/视频生成模型普遍用的 Flow Matching，其实是同一套数学的换坐标（CFG-Ctrl 篇直接在 velocity 场上做文章，必须先懂这个视角）；
4. **CFG 为什么能"引导"**？——条件/无条件预测的线性组合，数学上到底做了什么？为什么 guidance scale 大了会崩？（CFG-Ctrl 篇的全部动机）

**数值实验预告**：本篇所有关键结论都在一个二维 toy 上程序验证过——包括一个有趣的现象：**CFG 的 $w>1$ 会把样本系统性推出数据分布之外**，这正是后续 CFG-Ctrl 论文要修复的问题。

---

## 2. DDPM：一千个小去噪步骤

**几何总览：前向 = 蒸发，反向 = 凝结。** 把 $\mathcal M$ 记为数据流形。$t=0$ 时全部概率质量压在 $\mathcal M$ 上（零测、极度尖峰）；前向过程每步加噪，把质量往环境空间抹开一点，$t=T$ 时是各向同性高斯、充满整个 $\mathbb R^d$——流形的痕迹消失（一根铁丝加热蒸发、均匀充满房间）。反向过程则把质量逐步收回流形，每步去噪都是**一次指向流形的收缩**。三个等价镜头：

| 镜头 | 去噪在做什么 | 对应 |
|---|---|---|
| 条件均值 | $\mathbb E[x_0\mid x_t]$ = "站在流形外的这一点看，数据最可能在哪"——指向流形 | §2.2 式 (2.2)：朝流形走一步 + 再加一点小噪声 |
| score | $\nabla_{x_t}\log p_t(x_t)$ 指向概率质量增大的方向 = 指向流形；训练出的 $\varepsilon_\theta$ 是一张定义在全空间、**处处指回流形**的向量场 | §2.4 |
| ODE | PF-ODE 的流线族把满空间的高斯质量整体搬运回流形，像漏斗收缩，每条流线终点落在 $\mathcal M$ 上 | §3.3 |

**为什么是一千小步、不是一步**（§1.3 VAE 之病的镜像）：噪声大时 $\mathbb E[x_0\mid x_t]$ 是对多个模式的**平均**——一步跳过去就是模糊图像；小步走则每步只问"局部往哪偏"，不对多模态表态，模式选择交给逐步注入的随机性慢慢消解。且 $\mathcal M$ 是**弯曲**的：指向流形的方向只在局部近似正确，步子大了"方向"本身失效（数值积分的线性化只在小区间成立）——这也是 §4.2 直线插值的 Flow Matching 更好采样的几何原因。这两句话各藏着一段值得展开的论证——但需要式 (2.2) 与 score 视角（§2.4）垫底，展开收在 §2.5「回头看」。

### 2.1 前向过程：把数据"平稳地"变成噪声

给定数据 $x_0 \sim q_{\text{data}}$，定义**前向过程**（forward process）：一阶马尔可夫链，每步加一点高斯噪声

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t \mathbf{I}), \qquad t = 1, \dots, T,
$$

其中 $\beta_t \in (0,1)$ 是**噪声表**（noise schedule），$T \approx 1000$。

<details>
<summary><b>记号读本：$\mathcal N(x_t;\ m,\ V)$ 怎么读、这一行逐件在做什么</b></summary>

**先把 $\mathcal N(\cdot\,;\cdot,\cdot)$ 这个记号读开**。它记「多元正态分布」，分号前是随机变量、分号后是参数——$\mathcal{N}(x_t;\ m,\ V)$ 读作「$x_t$ 服从均值为 $m$、协方差为 $V$ 的正态分布」，密度为

$$
\mathcal N(x;\,m,\,V)=\frac{1}{(2\pi)^{d/2}(\det V)^{1/2}}\exp\!\Big(-\tfrac12(x-m)^\top V^{-1}(x-m)\Big).
$$

代入本题：$q(x_t\mid x_{t-1}) = \mathcal{N}(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t\mathbf I)$ 的意思是——**在上一步 $x_{t-1}$ 给定的条件下，$x_t$ 这样产生**：

$$
x_t = \sqrt{1-\beta_t}\;x_{t-1} + \sqrt{\beta_t}\;\varepsilon,\qquad \varepsilon\sim\mathcal N(\mathbf 0,\mathbf I),
$$

其中 $\varepsilon$ 是与 $x_{t-1}$ 独立的标准高斯噪声（$\mathbf I$ 是单位协方差：各维独立、方差全为 1）。逐件拆开这一行在做什么：

| 部件 | 值 | 在做什么 |
|---|---|---|
| 均值 $\sqrt{1-\beta_t}\,x_{t-1}$ | 原图缩水一点 | **信号**：向原点收缩，保留 $x_{t-1}$ 的「形状」 |
| 协方差 $\beta_t\mathbf I$ | 各向同性、逐维独立 | **噪声**：在每个方向等量撒胡椒面 |
| $\varepsilon\sim\mathcal N(\mathbf 0,\mathbf I)$ | 标准高斯 | 每步都新掷一次的骰子（Box–Muller/极坐标法可采，§1.1） |

直觉版本：「**原图乘以略小于 1 的系数，再叠加一份小幅高斯噪声**」。均值收缩 + 噪声膨胀这两件事合起来，让信号的「份额」逐步下降。注意记号约定：$\mathcal N(x_t; m, V)$ 里 $m,V$ 是参数（固定），$x_t$ 是随机变量；密度是 $x_t$ 的函数，$x_{t-1}$ 只作为条件出现。

</details>

<details>
<summary><b>为什么要乘 $\sqrt{1-\beta_t}$、而不是直接 $x_t = x_{t-1} + \text{noise}$？</b></summary>因为若不收缩、只叠噪声，方差会**逐 步累积膨胀**：$\mathrm{Var}(x_t) = \mathrm{Var}(x_0) + t\beta$，一千步后彻底爆炸（方差 $\sim 10^3$ 量级），终点根本不是标准高斯。乘上收缩系数后，「收缩掉的方差」恰好抵消「新增的方差」（下面验证：$\alpha_t + \beta_t = 1$ 恰是方差守恒的账本）——终点才能精确落在 $\mathcal N(0,\mathbf I)$ 上，反向过程才有明确的起点可采。
</details>

<details>
<summary><b>为什么是 $\sqrt{1-\beta_t}$ 这个系数？</b></summary>为了让任意时刻的 $x_t$ 有**闭式解**。定义 $\alpha_t = 1-\beta_t$，$\bar\alpha_t = \prod_{s=1}^t \alpha_s$，则两次加噪可以合并：

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \mathbf{I}). \tag{2.1}
$$

闭式解说的是：**不必一步步模拟 $x_0\to x_1\to\cdots\to x_t$，从 $x_0$ 直接一步写出 $x_t$ 的分布**——训练时任意时间步的样本一次构造（式 (2.3) 的训练循环之所以便宜，全靠它）。推导分三步。
</details>

<details>
<summary><b>三步推导 + 数学归纳三件套：式 (2.1) 的完整证明（连 $\varepsilon\perp x_0$ 一起证）</b></summary>

**第一步：两个独立高斯之和还是高斯（合并的基本功）。** 若 $A\sim\mathcal N(\mu_1,\sigma_1^2 I)$、$B\sim\mathcal N(\mu_2,\sigma_2^2 I)$ 且独立，则

$$
A + B \sim \mathcal N\big(\mu_1+\mu_2,\ (\sigma_1^2+\sigma_2^2)\,\mathbf I\big)
$$

——均值相加、**方差相加**（不是标准差相加！独立时无交叉项）。标准差层面要说的是：$\sqrt{\sigma_1^2+\sigma_2^2}$ 才是合成噪声的总幅值。这就是下面反复「合并两个 $\sqrt{\cdot}$ 项」的依据。

**第二步：往前推两步看结构。** 由定义（$t$ 与 $t-1$ 步）：

$$
x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{\beta_t}\,\varepsilon_t,\qquad
x_{t-1} = \sqrt{\alpha_{t-1}}\,x_{t-2} + \sqrt{\beta_{t-1}}\,\varepsilon_{t-1},
$$

代换（$\varepsilon_t\perp\varepsilon_{t-1}\perp x_{t-2}$，各步骰子独立）：

$$
x_t = \sqrt{\alpha_t\alpha_{t-1}}\;x_{t-2} + \underbrace{\sqrt{\alpha_t\beta_{t-1}}\;\varepsilon_{t-1} + \sqrt{\beta_t}\;\varepsilon_t}_{\text{两个独立零均值高斯}}.
$$

信号系数**相乘**（$\sqrt{\alpha_t}\cdot\sqrt{\alpha_{t-1}}$，缩水再缩水），噪声项变成两个独立高斯之和——按第一步合并，总方差 $\alpha_t\beta_{t-1}+\beta_t$，总幅值 $\sqrt{\alpha_t\beta_{t-1}+\beta_t}$。**每往前推一步，信号系数多乘一个 $\sqrt{\alpha}$，噪声项多并进一个高斯**——结构完全重复。

**第三步：归纳到 $t$ 步（形式化 + 顺手验证方差守恒）。** 信号系数累积为 $\sqrt{\prod_{s=1}^t\alpha_s}=\sqrt{\bar\alpha_t}$。噪声总方差按同样的模式累积：设 $t-1$ 步时噪声方差为 $1-\bar\alpha_{t-1}$（归纳假设），则 $t$ 步时

$$
\underbrace{\alpha_t\,(1-\bar\alpha_{t-1})}_{\text{旧噪声被缩水}\ \sqrt{\alpha_t}\ \text{再乘}}\ +\ \underbrace{\beta_t}_{\text{新添的噪声}}\ =\ \alpha_t - \alpha_t\bar\alpha_{t-1} + \beta_t\ =\ \underbrace{(\alpha_t+\beta_t)}_{=\,1}\ -\ \alpha_t\bar\alpha_{t-1}\ =\ 1-\bar\alpha_t,
$$

中间那行就是「系数平方和为 1」（$\alpha_t+\beta_t=1$）在起作用——**收缩消掉的方差（$\beta_t\bar\alpha_{t-1}$ 那部分）恰好补上新增噪声，总方差守恒地滑向 1**。归纳完成，合并所有噪声为一个等价高斯 $\sqrt{1-\bar\alpha_t}\,\varepsilon$（$\varepsilon$ 仍是标准高斯，把多个独立骰子合成一个等价骰子），即得式 (2.1)。

**补全归纳假设本身的证明**（上面那句「设 $t-1$ 步时噪声方差为 $1-\bar\alpha_{t-1}$」不能凭空假设——完整的数学归纳法要三件套：明确命题、验证起点、完成传递）。要证的命题是对**分布**的断言，比「方差是 $1-\bar\alpha_{t-1}$」更强：

$$
P(t):\qquad x_t \;=\; \underbrace{\sqrt{\bar\alpha_t}\;x_0}_{\text{信号}} \;+\; \underbrace{\sqrt{1-\bar\alpha_t}\;\eta_t}_{\text{噪声}},\qquad \eta_t\sim\mathcal N(0,\mathbf I)\ \text{且}\ \eta_t\perp x_0,
$$

即「$x_t$ 分解为缩水信号 + 一个与 $x_0$ **独立**的标准高斯」（独立性也是命题的一部分——它保证方差能直接相加，且最后合并出的 $\varepsilon$ 是干净的骰子）。

**起点 $P(1)$**：由定义 $x_1=\sqrt{\alpha_1}\,x_0+\sqrt{\beta_1}\,\varepsilon_1$，而 $\bar\alpha_1=\alpha_1$、$1-\bar\alpha_1=\beta_1$，取 $\eta_1=\varepsilon_1$ 即是 $P(1)$——$P(1)$ 就是定义本身。✓

**传递 $P(t-1)\Rightarrow P(t)$**（这一步同时推出方差公式，不是两件事）：设 $P(t-1)$ 成立，

$$
x_{t-1}=\sqrt{\bar\alpha_{t-1}}\,x_0+\sqrt{1-\bar\alpha_{t-1}}\;\eta_{t-1},\qquad \eta_{t-1}\perp x_0 .
$$

代入第 $t$ 步的定义 $x_t=\sqrt{\alpha_t}\,x_{t-1}+\sqrt{\beta_t}\,\varepsilon_t$（其中 $\varepsilon_t$ 与 $x_{t-1}$ 独立，从而与 $x_0,\eta_{t-1}$ 都独立）：

$$
x_t=\sqrt{\alpha_t}\,\Big(\sqrt{\bar\alpha_{t-1}}\,x_0+\sqrt{1-\bar\alpha_{t-1}}\,\eta_{t-1}\Big)+\sqrt{\beta_t}\,\varepsilon_t
=\underbrace{\sqrt{\alpha_t\bar\alpha_{t-1}}}_{=\sqrt{\bar\alpha_t}}\;x_0+\underbrace{\sqrt{\alpha_t(1-\bar\alpha_{t-1})}\;\eta_{t-1}+\sqrt{\beta_t}\;\varepsilon_t}_{\text{待合并}},
$$

- **信号项**：$\alpha_t\bar\alpha_{t-1}=\bar\alpha_t$（$\bar\alpha$ 的定义就是累乘），系数整理完毕；
- **噪声项**：$\eta_{t-1}$ 与 $\varepsilon_t$ 独立、都是零均值高斯（$\eta_{t-1}$ 由归纳假设保证，$\varepsilon_t$ 由前向定义保证）——按第一步的合并公式，两者之和仍是零均值高斯，方差为 $\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\bar\alpha_t$（上式的守恒账本）。于是可定义

$$
\eta_t \;=\; \frac{\sqrt{\alpha_t(1-\bar\alpha_{t-1})}\;\eta_{t-1}+\sqrt{\beta_t}\;\varepsilon_t}{\sqrt{1-\bar\alpha_t}}
\qquad\Longrightarrow\qquad x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\;\eta_t,\ \ \eta_t\sim\mathcal N(0,\mathbf I).
$$

- **独立性 $\eta_t\perp x_0$**：$\eta_t$ 是 $\eta_{t-1}$ 与 $\varepsilon_t$ 的（确定性系数）线性组合；$\eta_{t-1}\perp x_0$（归纳假设）且 $\varepsilon_t\perp x_0$（每步骰子独立于一切过去），故组合出的 $\eta_t$ 也与 $x_0$ 独立（联合高斯情形下，不相关 ⟂ 独立；或直接看：$\eta_t$ 的随机来源与 $x_0$ 无交集）。✓

三件套齐了：$P(1)$ 成立，$P(t-1)\Rightarrow P(t)$ 成立，故 $P(t)$ 对一切 $t$ 成立——**式 (2.1) 连同「$\varepsilon$ 与 $x_0$ 独立」一起被证明**，而不只是「方差碰巧等于 $1-\bar\alpha_t$」。这一点有实际后果：$\eta_t$ 干净地独立于 $x_0$，训练时 $(x_0,\varepsilon)$ 才能当作两个互不干扰的自由变量采样（式 (2.3)）；若独立性没被证明、只是方差对上了，$x_t$ 的分布正确但「猜 $\varepsilon$」这个监督目标会与 $x_0$ 纠缠。

（递归式 $\eta_t=\sqrt{\tfrac{\alpha_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}}\,\eta_{t-1}+\sqrt{\tfrac{\beta_t}{1-\bar\alpha_t}}\,\varepsilon_t$ 顺带解释了 $\eta_t$ 是怎么「攒」出来的：**旧的等价骰子再缩一点 + 新骰子掺一点，两个系数平方和为 1**（用 $1-\bar\alpha_t=\alpha_t(1-\bar\alpha_{t-1})+\beta_t$ 一行可验），所以它始终是标准高斯——方差守恒在骰子层面的表述。）

</details>

用具体数字走一遍（$\beta_t\equiv 0.02$ 恒定，前 3 步）：

| $t$ | $\bar\alpha_t=\prod\alpha_s$ | $\sqrt{\bar\alpha_t}$（信号份额） | $\sqrt{1-\bar\alpha_t}$（噪声份额） | 直观 |
|---|---|---|---|---|
| 1 | $0.98$ | $0.990$ | $0.141$ | 几乎原图 |
| 2 | $0.9604$ | $0.980$ | $0.199$ | 略糊 |
| 3 | $0.9412$ | $0.970$ | $0.243$ | 噪声渐显 |
| 1000 | $\approx 1.7\times10^{-9}$ | $\approx 4\times10^{-5}$ | $\approx 1.000$ | 纯噪声 ✓ |

（每行两个份额的平方和 $=\bar\alpha_t+(1-\bar\alpha_t)=1$——方差守恒的直接体现；$\beta_t$ 实际取随 $t$ 递增的表，这里恒定仅为了手算。）

$$
\text{训练时的用法（重参数化视角）：} x_t = \sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\quad \varepsilon\sim\mathcal N(0,\mathbf I)\tag{2.3}
$$

——注意它和式 (2.1) 是同一件事：给定 $x_0$ 和采出的 $\varepsilon$，$x_t$ **是确定的**（随机性全部收编进 $\varepsilon$）；这正是 §1.3 重参数化技巧的复用——采样不挡梯度，训练时想要哪个 $t$ 的样本就直接构造哪个。

噪声表设计得让 $\bar\alpha_T \approx 0$：终点 $x_T \approx \mathcal{N}(0, \mathbf{I})$——**纯噪声，数据信息全部抹掉**。

### 2.2 反向过程：学一个逆马尔可夫链

如果能把 $q(x_{t-1}\mid x_t)$ 逆向过来，就能从噪声"走回"数据。关键事实（Anderson 1982）：**高斯过程的反向仍是高斯过程**，且

$$
q(x_{t-1}\mid x_t, x_0) = \mathcal{N}\left(x_{t-1};\ \tilde\mu_t(x_t, x_0),\ \tilde\beta_t \mathbf{I}\right),
$$

其中（由贝叶斯公式对高斯做配方可得，是本篇第一个值得亲手推一遍的练习）：

$$
\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\, x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\, x_t, \qquad \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t.
$$

**$\tilde\mu_t$ 怎么理解、怎么记**：原始形式（$x_0$ 与 $x_t$ 的加权组合，[习题 1 解答](../../solutions/SOL_01_vg01_reverse_gaussian/) §3.7 有「精度加权平均」的读法）结构对称但不便记忆；换个等价写法就好记了（推导见解答 §4.3）：

$$
\tilde\mu_t=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\eta\right),
\qquad
\eta=\frac{x_t-\sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}}\ \ (\text{即式 (2.3) 里的 }\varepsilon).
$$

一句话读法：**把前向一步倒着走，但把「这一步的噪声 $\varepsilon_t$」换成它的最优猜测**。分三层（记号沿用 §2.1：$\varepsilon_t$ 是第 $t$ 步的骰子，$\eta$ 是打包后的等价骰子；逐步拆解收进折叠块）：

<details>
<summary><b>三层拆解 + 两个锚点：$t=1$ 精确倒走、$t>1$ 按份额 $c_t$ 缩</b></summary>

| 层 | 内容 |
|---|---|
| ① 想倒走 | 前向定义 $x_t=\sqrt{\alpha_t}\,x_{t-1}+\sqrt{\beta_t}\,\varepsilon_t$ 反解：$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\big(x_t-\sqrt{\beta_t}\,\varepsilon_t\big)$——卡在 $\varepsilon_t$（这一步实际加的噪声）不知道 |
| ② 猜噪声 | 已知 $(x_t,x_0)$ 能反解出等价骰子 $\eta$（§2.1 已构造）；$\varepsilon_t$ 在 $\eta$ 里的**份额**是 $c_t=\sqrt{\beta_t/(1-\bar\alpha_t)}$（就是 §2.1 递归式里 $\varepsilon_t$ 的系数），它的最优猜测（条件期望）是 $c_t\,\eta$ |
| ③ 代回去 | $\tilde\mu_t=\frac{1}{\sqrt{\alpha_t}}\big(x_t-\sqrt{\beta_t}\,c_t\,\eta\big)=\frac{1}{\sqrt{\alpha_t}}\big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\eta\big)$ |

两个锚点：**$t=1$** 时份额 $c_1=\sqrt{\beta_1/(1-\bar\alpha_1)}=1$（骰子就是 $\varepsilon_1$ 本身），猜测退化为真值——$\tilde\mu_1$ 是精确倒走、$\tilde\beta_1=0$，反向最后一步完全确定；**$t>1$** 时 $c_t<1$（数字标本 $\beta_t=0.02$、$t=3$：$c_3=0.583$）——骰子是**全部** $t$ 步噪声的混合，直接当这一步的噪声用会矫枉过正，按份额缩一下正好。

</details>

<details>
<summary><b>为什么「猜噪声」连方差 $\tilde\beta_t$ 也一并给出了</b></summary>

倒走公式 $x_{t-1}=\frac{1}{\sqrt{\alpha_t}}(x_t-\sqrt{\beta_t}\,\varepsilon_t)$ 里，给定 $(x_t,x_0)$ 后唯一的随机量是 $\varepsilon_t$（$x_t$ 已冻结成数），所以均值 = 代入条件期望、方差 = 常数平方倍放大剩余方差：

$$
\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\big(x_t-\sqrt{\beta_t}\,\mathbb E[\varepsilon_t\mid x_t,x_0]\big),\qquad
\tilde\beta_t = \frac{\beta_t}{\alpha_t}\,\mathrm{Var}(\varepsilon_t\mid x_t,x_0).
$$

而 $(\varepsilon_t,\eta)$ 是联合高斯、$\mathrm{Cov}(\varepsilon_t,\eta)=c_t$、两者方差都是 1，标准公式给 $\mathbb E[\varepsilon_t\mid\eta]=c_t\,\eta$、$\mathrm{Var}(\varepsilon_t\mid\eta)=1-c_t^2$。后者代入：

$$
\tilde\beta_t=\frac{\beta_t}{\alpha_t}\left(1-\frac{\beta_t}{1-\bar\alpha_t}\right)
=\frac{\beta_t}{\alpha_t}\cdot\frac{\alpha_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}
=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t,
$$

（中间步用了 $1-\bar\alpha_t-\beta_t=\alpha_t-\bar\alpha_t=\alpha_t(1-\bar\alpha_{t-1})$，账本 $(\star)$ 的亲戚。）数字标本：$\mathrm{Var}(\varepsilon_t\mid\eta)=1-0.583^2=0.660$，$\tilde\beta_t=\frac{0.02}{0.98}\times0.660=0.013468$，与习题 1 解答的蒙特卡洛结果吻合。同一套「份额」语言把均值、方差一网打尽——§2.1 的等价骰子不只是证明工具，它就是反向过程的心脏。

</details>

**注意**：反向转移需要知道 $x_0$——但 $x_0$ 正是我们要生成的东西，循环依赖。**DDPM 的解法：用网络从 $x_t$ 猜 $x_0$**。由式 (2.1)，给定 $x_t$ 猜 $x_0$ 等价于猜产生 $x_t$ 的那个噪声 $\varepsilon$，于是参数化

$$
\text{网络 } \varepsilon_\theta(x_t, t) \approx \varepsilon, \qquad \hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}.
$$

式 (2.2) 从哪来？——**三条线索拼一个高斯**。$p_\theta(x_{t-1}\mid x_t)$ 是我们自己定义的（网络是学生，可以随便定），DDPM 的选择是：**抄老师 $q(x_{t-1}\mid x_t,x_0)$ 的结构，把老师作弊用的 $x_0$ 换成自己的猜测 $\hat x_0$**。具体分三步，每步只做一件事：

<details>
<summary><b>三条线索逐步拼装：$\tilde\mu$' 形式 → 换 $\hat x_0$ → 紧凑式（账本三次记账）</b></summary>

**第一步：把老师的公式改写成 $\hat x_0$ 的函数**。$\tilde\mu_t$ 原始形式是 $x_0$ 与 $x_t$ 的加权组合（§2.2 开头），先把 $x_0$ 反解出来。由式 (2.1)：

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon
\quad\Longrightarrow\quad
x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon}{\sqrt{\bar\alpha_t}},
\qquad
x_t-\sqrt{\bar\alpha_t}\,x_0=\sqrt{1-\bar\alpha_t}\,\varepsilon.
$$

（第二个恒等式是关键中间量：**$x_t$ 与「缩水后的 $x_0$」之差恰是噪声项本身**。）代入 $\tilde\mu_t$ 加权式，两个系数通分合并（又用到账本 $\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\bar\alpha_t$，逐项见习题 1 解答 §4.3），得到等价形式：

$$
\tilde\mu_t=\sqrt{\bar\alpha_{t-1}}\,x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\left(x_t-\sqrt{\bar\alpha_t}\,x_0\right).
\tag{$\tilde\mu$'}
$$

这个形式把 $\tilde\mu_t$ 写成「$\hat x_0$ 的直接贡献 + 从 $x_t$ 出发的修正」——**两个输入（$x_0$ 的先验位置、当前观测 $x_t$）各占一项**，是把 $x_0$ 换成 $\hat x_0$ 最方便的形状。

**第二步：定 $\mu_\theta$——把 $\tilde\mu$' 里的 $x_0$ 换成 $\hat x_0$**。$\hat x_0$ 怎么来？网络不直接猜图像，而是猜噪声 $\varepsilon_\theta(x_t,t)\approx\varepsilon$（为什么猜噪声而不是 $x_0$：§2.3 训练目标因此变成干净的最小二乘）。把猜测代入反解式：

$$
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}.
$$

于是学生答卷的均值定为

$$
\mu_\theta=\sqrt{\bar\alpha_{t-1}}\,\hat x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\left(x_t-\sqrt{\bar\alpha_t}\,\hat x_0\right).
$$

**第三步：整理成式 (2.2) 的紧凑形式**。把 $\hat x_0$ 代入上式，$\hat x_0$ 的两个系数合并——用 $\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$ 与账本：

$$
\sqrt{\bar\alpha_{t-1}}-\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\cdot\sqrt{\bar\alpha_t}
=\sqrt{\bar\alpha_{t-1}}\left[1-\frac{\alpha_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\right]
=\sqrt{\bar\alpha_{t-1}}\cdot\frac{\beta_t}{1-\bar\alpha_t},
$$

（方括号与 $\tilde\beta_t$ 推导里出现的完全相同——同一笔账本第三次记账。）剩下一步通分：$x_t$ 的系数 $\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\overset{\text{通分}}{=}\frac{1}{\sqrt{\alpha_t}}$（分子提出 $\sqrt{\bar\alpha_{t-1}}$ 后恰是账本的又一变体），$\varepsilon_\theta$ 的系数为 $-\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\cdot\frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}=-\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}$。合并：

$$
\mu_\theta=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t,t)\right).
$$

</details>

再加上方差项 $\sigma_t z$（$p_\theta$ 的方差按 DDPM 取 $\sigma_t^2=\beta_t$；DDIM 会说明这个选择不唯一、$\sigma_t=0$ 时整条链变成确定性 ODE——§3.2 的入口），即得采样更新式：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t,t)\right) + \sigma_t z, \qquad z\sim\mathcal{N}(0,\mathbf{I}). \tag{2.2}
$$

<details>
<summary><b>数字标本：三步各是什么数（$\beta_t=0.02$、$t=3$、$x_t=1.5$、真值 $x_0=2$、网络猜 $\varepsilon_\theta=-1.8$）</b></summary>

| 步骤 | 公式 | 数值 |
|---|---|---|
| 反解 $\hat x_0$ | $\dfrac{1.5-\sqrt{0.058808}\times(-1.8)}{\sqrt{0.941192}}$ | $\dfrac{1.5+0.4365}{0.9702}=\mathbf{1.9961}$（真值 2，猜得很准） |
| $\tilde\mu$' 两系数 | $\sqrt{\bar\alpha_2}=0.980$；$\dfrac{\sqrt{0.98}\times0.0396}{0.058808}=0.6667$ | 两项：$0.98\hat x_0+0.6667(x_t-0.9702\hat x_0)$ |
| 第二步代入 | $0.98\times1.9961+0.6667\times(1.5-0.9702\times1.9961)$ | $1.9562+0.6667\times(-0.4365)=\mathbf{1.6652}$ |
| 第三步紧凑式 | $\dfrac{1}{\sqrt{0.98}}\left(1.5-\dfrac{0.02}{\sqrt{0.058808}}\times(-1.8)\right)$ | $\dfrac{1.5+0.14899}{0.98995}=\mathbf{1.6652}$ ✓ 两条路线吻合 |
| 采样（抽到 $z=-0.5$） | $1.6652+\sqrt{0.02}\times(-0.5)$ | $1.6652-0.0707=\mathbf{1.5945}$（$x_2$ 的一个样本） |

中间量的自检：$x_t-\sqrt{\bar\alpha_t}\hat x_0=1.5-0.9702\times1.9961=-0.4365$，而 $\sqrt{1-\bar\alpha_t}\,\varepsilon_\theta=0.2425\times(-1.8)=-0.4365$——**两个数相等正是第一步那个恒等式的数值版**（$\hat x_0$ 的定义保证了它，无论 $\varepsilon_\theta$ 猜成多少都成立）。

</details>


**$q$ 与 $p_\theta$、$\tilde\mu_t$ 与 $\mu_\theta$——一张表分清**。这两个记号是同一道反向一步的「标准答案」与「学生答卷」：

| | $q(x_{t-1}\mid x_t,x_0)$（真反向） | $p_\theta(x_{t-1}\mid x_t)$（学的反向） |
|---|---|---|
| 身份 | 前向过程的数学推论——**不是模型、没有参数** | 神经网络造的分布，参数 $\theta$ |
| 均值 | $\tilde\mu_t$：**偷看 $x_0$ 时**反向一步的真实中心 | $\mu_\theta$：网络（看不到 $x_0$）猜出来的中心 |
| 均值公式 | $\frac{1}{\sqrt{\alpha_t}}\big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon\big)$，$\varepsilon$ 是**真**骰子 | **同一模板**，把 $\varepsilon$ 换成猜测 $\varepsilon_\theta(x_t,t)$ |
| 方差 | $\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$（更小、更尖） | $\beta_t$（占位，见下） |
| 何时在场 | 只在训练时当靶子（采样时没有 $x_0$ 可偷看） | 采样时真正执行的就是它 |

三个记忆钩子：

1. **模板**：式 (2.2) 不是新公式——把 $\tilde\mu_t$ 的 $\varepsilon$-形式里那个真噪声换成网络猜测，得到的正是 $\mu_\theta$。**学生抄了老师的模板，只把「真噪声」换成「猜的噪声」**，采样更新式 (2.2) 就这么来的；
2. **训练**：让学生答卷的中心贴住标准答案的中心——§2.3 的 KL 只罚均值差，把 $\mu_\theta$ 与 $\tilde\mu_t$ 各自代入后连 $x_0$ 都消掉，只剩 $\|\varepsilon-\varepsilon_\theta\|^2$；
3. **方差为什么用 $\beta_t$ 占位而不是更准的 $\tilde\beta_t$**：KL 不罚方差（$\tilde\beta_t$ 不含 $\theta$），糙一点不碍训练；其实 $\tilde\beta_t$ 也不含 $x_0$（只依赖调度表），后续工作（如 IDDPM）用它或可学方差，采样质量更好——这是习题 2 的主题。

数字标本（$\beta_t=0.02$、$t=3$、$x_t=1.5$、$x_0=2$）把整条链走一遍：反解真骰子 $\eta=-1.816$ → 老师均值 $\tilde\mu_t=1.6665$；若网络猜 $\varepsilon_\theta=-1.8$ → 学生均值 $\mu_\theta=1.6652$——训练就是在把这个差往零里压。

### 2.3 训练目标：从变分下界到"预测噪声"

**$p_\theta(x_0)$ 的积分公式**：链式法则 × 边缘化——"一步步采出来"的采样算法与"一次性写出 $x_0$ 的分布"是同一件事的两种表述，与 §1.3 VAE 的 $p(x)=\int p(x\mid z)p(z)\,dz$ 完全同构（隐变量从一层换成一整条链；分步拆解收进折叠块）。

<details>
<summary><b>分步拆解：$p_\theta(x_0)$ 积分公式的来历（以 $T=2$ 看结构）</b></summary>

**$p_\theta(x_0)$ 那条积分公式怎么来的**——它是「采样算法倒着写成积分」。回忆反向采样的实际操作：从纯噪声出发，$x_T\sim p(x_T)=\mathcal N(0,\mathbf I)$，然后对 $t=T,T-1,\dots,1$ 逐步用网络 $p_\theta(x_{t-1}\mid x_t)$ 抽样。这条随机轨迹**产生**某个 $x_0$ 的概率是多少？分步拆（以 $T=2$ 看结构，记号 $x_{0:2}=(x_0,x_1,x_2)$）：

1. 链式法则展开联合分布：$p_\theta(x_{0:2}) = p_\theta(x_0\mid x_1)\,p_\theta(x_1\mid x_2)\,p(x_2)$（第三项是已知的噪声先验，没有 $\theta$）；
2. 我们只关心最终的 $x_0$，把沿途的中间变量 $x_1,x_2$ **全部积分掉**（边缘化）：

$$
p_\theta(x_0) = \int p_\theta(x_0\mid x_1)\,p_\theta(x_1\mid x_2)\,p(x_2)\ dx_1 dx_2.
$$

一般 $T$ 就是那条公式：$p_\theta(x_0) = \int p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1}\mid x_t)\,dx_{1:T}$。一句话：**链式法则 × 边缘化——"一步步采出来"的算法和"一次性写出 $x_0$ 的分布"是同一件事的两种表述**。它与 §1.3 VAE 的 $p(x)=\int p(x\mid z)p(z)\,dz$ 完全同构，只是隐变量从一层 $z$ 变成了整条链 $x_{1:T}$。

</details>

对齐 $p_\theta$ 与数据 $q$ 用最大似然，引入变分下界（$x_T$ 取已知先验，ELBO 的标准推导——就是把 §1.3 三行链里的 $z$ 换成 $x_{1:T}$，Jensen 后每一项按链式法则拆开、逐层配成 KL，细节收进折叠块）：

$$
-\log p_\theta(x_0) \le \sum_{t=2}^{T} \mathbb{E}_q\!\left[D_{\mathrm{KL}}\big(q(x_{t-1}\mid x_t, x_0)\ \|\ p_\theta(x_{t-1}\mid x_t)\big)\right] + C.
$$

<details>
<summary><b>ELBO 到这行求和的三步推导（把 §1.3 的三行链搬到链式隐变量上）</b></summary>

**第一步：塞入前向联合分布 $q$（对 $q$ 无任何要求，同 §1.3）**。记 $q(x_{1:T}\mid x_0)$ 为前向过程把这张图加噪成整条链的联合分布（各步独立定义、马尔可夫），则 $q(x_{1:T}\mid x_0)>0$，可往 $\log p_\theta(x_0)$ 里塞：

$$
\log p_\theta(x_0) = \log \int p_\theta(x_{0:T})\,dx_{1:T}
= \log \int q(x_{1:T}\mid x_0)\,\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\,dx_{1:T}.
$$

**第二步：Jensen（log 是凹的，期望进 log 只会变大）**：

$$
\log p_\theta(x_0) \ge \int q(x_{1:T}\mid x_0)\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\,dx_{1:T}
= \log p_\theta(x_0) - \mathrm{KL}\big(q(x_{1:T}\mid x_0)\ \|\ p_\theta(x_{1:T}\mid x_0)\big).
$$

后一个等号没有新数学——就是 §1.3 的认-KL 一步（式 (1.1)，悬停可见推导）。把 $p_\theta(x_{0:T})=p_\theta(x_0)\cdot p_\theta(x_{1:T}\mid x_0)$ 拆开：$\log p_\theta(x_0)$ 与积分变量 $x_{1:T}$ 无关，提到积分号外；剩下的 $\mathbb E_q\!\left[\log\frac{p_\theta(x_{1:T}\mid x_0)}{q(x_{1:T}\mid x_0)}\right]$ 用 式 (1.1) 认成 $-\mathrm{KL}\big(q(x_{1:T}\mid x_0)\ \|\ p_\theta(x_{1:T}\mid x_0)\big)$。KL 非负顺带自检了不等号方向；移项即是 gap 恒等式（与 §1.3 同构）：$\log p_\theta(x_0)=\mathrm{ELBO}+\mathrm{KL}(q\ \|\ p_\theta(x_{1:T}\mid x_0))$——**Jensen 丢掉的量恰好是「整条链上 $q$ 与 $p_\theta$ 的 KL」**，下界的松紧仍由 $q$ 偏离模型后验的程度记账。

**第三步：把中间那个期望 $\mathbb E_q\!\left[\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]$ 按链式法则拆项、逐项配 KL**。分子按生成方向拆：$p_\theta(x_{0:T})=p_\theta(x_0\mid x_1)\prod_{t=2}^T p_\theta(x_{t-1}\mid x_t)\cdot p(x_T)$；分母按前向链拆：$q(x_{1:T}\mid x_0)=q(x_1\mid x_0)\prod_{t=2}^T q(x_{t-1}\mid x_t,x_0)$（其中 $q(x_{t-1}\mid x_t,x_0)$ 恰是习题 1 的那个反向高斯）。同一个 $t$ 的分子分母配成一对，每一对 $\mathbb E_q\!\left[\log\frac{q(x_{t-1}\mid x_t,x_0)}{p_\theta(x_{t-1}\mid x_t)}\right]$ 再用一次认-KL（式 (1.1)）认成条件 KL——配对时反复用前向的马尔可夫性。逐项整理后：

$$
\log p_\theta(x_0) \ge \sum_{t=2}^{T}\mathbb E_q\!\left[\log\frac{q(x_{t-1}\mid x_t, x_0)}{p_\theta(x_{t-1}\mid x_t)}\right] + \mathbb E_q\big[\log p_\theta(x_0\mid x_1)\big] + \text{(与 }\theta\text{ 无关的项)}.
$$

每一项正是 $-D_{\mathrm{KL}}(q(x_{t-1}\mid x_t,x_0)\ \|\ p_\theta(x_{t-1}\mid x_t))$，取负移项即得正文那行不等式（$C$ 吸收了 $t=1$ 项、重构项与熵项——$t=1$ 时 $x_0$ 已知 $x_1$ 是确定性重构，DDPM 论文用 $\mathcal L_1$ 单独处理，通常也以重加权形式并入）。**要点**：求和号里每一项是"老师 $q$（偷看 $x_0$）与学生 $p_\theta$ 在同一步 $(x_t\to x_{t-1})$ 上的 KL"——训练目标天然分解成逐步的局部比较，这正是扩散模型可训练性的来源。

</details>

**两个高斯分布的 KL 散度只惩罚均值差**。这是本节第一处要展开的推导。对同方差假设 $\sigma_1^2=\sigma_2^2=\sigma^2$，KL 闭式解（§1.3 KL 专题）退化为 $\frac{\|\mu_1-\mu_2\|^2}{2\sigma^2}$；一般情形（$p_\theta$ 方差 $\sigma_t^2$、$q$ 方差 $\tilde\beta_t$）完整展开：

$$
D_{\mathrm{KL}}\big(\mathcal N(\tilde\mu,\tilde\beta_t)\ \|\ \mathcal N(\mu_\theta,\sigma_t^2)\big)
= \underbrace{\log\frac{\sigma_t}{\sqrt{\tilde\beta_t}} + \frac{\tilde\beta_t}{2\sigma_t^2} - \frac{1}{2}}_{\text{与 }\theta\text{ 无关（}\sigma_t\text{ 按调度表取，}\tilde\beta_t\text{ 只依赖调度表）}} +
\underbrace{\frac{1}{2\sigma_t^2}\|\mu_\theta-\tilde\mu\|^2}_{\text{唯一的训练信号}}.
$$

（来自高斯 KL 闭式解 $\log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+\|\mu_1-\mu_2\|^2}{2\sigma_2^2}-\frac12$，代入 $\mu_1=\tilde\mu_t,\sigma_1^2=\tilde\beta_t,\mu_2=\mu_\theta,\sigma_2^2=\sigma_t^2$。）于是

$$
L_t = \mathbb{E}_q\left[\frac{1}{2\sigma_t^2}\|\mu_\theta(x_t,t) - \tilde\mu_t(x_t,x_0)\|^2\right] + \text{const}.
$$

<details>
<summary><b>同模板代入：$x_0$、$x_t$ 相消，只剩噪声差（带权 ε-MSE 的来历）</b></summary>

接下来是 DDPM 论文最漂亮的化简：把 $\mu_\theta$ 与 $\tilde\mu_t$ 各自的显式公式代入，**所有 $x_0$、$x_t$ 的项相消，只剩噪声差**。逐步核（两条均值都已是 $\varepsilon$-形式，见式 (2.2) 推导第三步；记网络猜 $\varepsilon_\theta=\varepsilon_\theta(x_t,t)$，真噪声 $\varepsilon$ 是产生这个 $x_t$ 的那个）：

$$
\mu_\theta-\tilde\mu_t
= \frac{1}{\sqrt{\alpha_t}}\left[\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta\Big)-\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon\Big)\right]
$$

$x_t$ 直接相减消掉（同一行里出现两次），剩下

$$
\mu_\theta-\tilde\mu_t = -\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\,(\varepsilon_\theta-\varepsilon),
\qquad
\|\mu_\theta-\tilde\mu_t\|^2 = \frac{\beta_t^2}{\alpha_t(1-\bar\alpha_t)}\,\|\varepsilon_\theta-\varepsilon\|^2.
$$

**消去发生的原因**：学生与老师用的是**同一张模板**（式 (2.2) 推导的核心观察），两份答卷的差别只剩"猜的噪声减真的噪声"——均值比较变成了噪声回归，$x_0$、$x_t$ 连同调度系数全部被模板自己吸收。代入 $L_t$（并把期望换成对自由变量 $(x_0,\varepsilon)$ 的期望——§2.1 的归纳证明保证了 $\varepsilon\perp x_0$，两者可以独立采）：

$$
L_t = \mathbb{E}_{x_0,\varepsilon}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\,\|\varepsilon - \varepsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon,\ t)\|^2\right].
$$

（第二个参数位写成了 $\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$——就是 $x_t$ 本身，训练时直接构造。）

</details>

<details>
<summary><b>数字标本：$\beta_t=0.02$、$t=3$、$x_0=2$、$x_t=1.5$（真 $\varepsilon=-1.816$）、网络猜 $\varepsilon_\theta=-1.8$</b></summary>

| 步骤 | 公式 | 数值 |
|---|---|---|
| 学生均值 | $\frac{1}{\sqrt{0.98}}\big(1.5-\frac{0.02}{\sqrt{0.058808}}\times(-1.8)\big)$ | $1.66519$ |
| 老师均值 | 同模板，$\varepsilon=-1.816$ 代入 | $1.66649$ |
| 均值差 | $1.66519-1.66649$ | $-0.00130$ |
| 恒等式核对 | $-\frac{0.02}{\sqrt{0.98}\sqrt{0.058808}}\times(-1.8+1.816)$ | $-0.00130$ ✓ |
| KL 的 MSE 项 | $\frac{(0.00130)^2}{2\times 0.02}$ | $4.2\times10^{-5}$ |
| 恒等式核对 | $\frac{0.02^2}{0.98\times0.058808}\times(0.016)^2\div(2\times0.02)$ | $4.2\times10^{-5}$ ✓ |
| KL 全值（含常数项） | $\log\frac{\sqrt{0.02}}{\sqrt{0.013468}}+\frac{0.013468}{0.04}-\frac12+\text{MSE 项}$ | $0.04486$ |

常数项 $0.04480$ 与 MSE 项 $4.2\times10^{-5}$ 的悬殊比例正说明：**梯度几乎全部来自均值差**，丢掉常数项没有损失。此外可算出当 $\sigma_t^2=\beta_t$ 时加权系数 $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}=\frac{\beta_t}{2\alpha_t(1-\bar\alpha_t)}=0.1735$——DDPM 实测丢掉它（置 1）反而更好，原因见下。

</details>

DDPM 发现丢掉前面的加权系数直接训（均匀加权）效果更好：

$$
\boxed{\ L(\theta) = \mathbb{E}_{x_0,\,t,\,\varepsilon}\left[\big\|\varepsilon - \varepsilon_\theta\big(\underbrace{\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon}_{x_t},\ t\big)\big\|^2\right]\ }
$$


**训练算法**因此简单到荒谬：

```text
重复：
  采样数据 x0 ~ 数据集, 噪声 ε ~ N(0,I), 时间步 t ~ Uniform{1..T}
  构造 x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε
  梯度下降 ‖ε - ε_θ(x_t, t)‖²
```

没有对抗、没有平衡技巧——**就是回归**。这个稳定性是扩散模型取代 GAN 的根本原因。

### 2.4 分数视角：去噪 = 学 score

**一句话版本**：DDPM 训练完，网络 $\varepsilon_\theta$ 不只会去噪——乘一个调度表里查得到的系数，它还免费变成了一个「罗盘场」（score）：站在任何 $x_t$ 处，箭头都指向数据密集的方向。本节用两个具体数字标本说清这件事，完整推导收在折叠块里，读正文不需要它。

#### 箭头场是什么：先看一个一维小标本

想象数据全部挤在 $x=2$ 附近（$x\sim\mathcal N(2, 0.6^2)$）。**score** 的定义是 $\nabla_x\log p(x)$（对数密度的梯度），但它的意思用表格看最直接——**站在每个点问一句：往哪边走、数据会更密？**

| 站在 | 箭头 $-\frac{x-2}{0.36}$ | 指向 |
|---|---|---|
| $1.0$ | $+2.78$ | 右（朝 2） |
| $1.7$ | $+0.83$ | 右（朝 2，更缓） |
| $2.0$ | $0$ | 不动（已在最密处） |
| $2.6$ | $-1.67$ | 左（朝 2） |

三个规律：**所有箭头都指向数据密集处**；离中心越远箭头越长（越该快走）；高斯的箭头公式恰好是「负的离中心距离 ÷ 方差」——指向中心、按陡峭度定长度。所谓生成，就是从纯噪声出发**顺着箭头一小步一小步挪**进数据密集区——§2.2 那一千步反向过程，几何上就是这件事。

score 的三个读法，一层比一层「为什么」（第三层是生成模型在乎它的真正原因）：

| 层 | 读法 | 内容 |
|---|---|---|
| 1 定义 | 梯度 | $s(x)=\nabla_x\log p(x)$——一个**向量场**，空间每点配一个箭头 |
| 2 物理 | 弹簧 / 坡度 | 高斯 score $=-\frac{x-\mu}{\sigma^2}$ 是胡克弹簧：离中心越远拉回力越大、按 $1/\sigma^2$ 定劲度；一般分布 = 无数片「局部高斯弹簧」的缝合。等价读法：$\log p$ 是海拔，score 是最陡上升方向 |
| 3 为什么生成模型在乎它 | **只要 score 就能采样** | Langevin 动力学：$x\leftarrow x+\frac\epsilon2 s(x)+\sqrt\epsilon\,z$ 迭代够久，$x$ 的分布收敛到 $p$——**不需要知道 $p$ 本身**。§1.1 判了 $p_{\text{data}}$「无公式」的刑，而 score 只依赖密度的**导数侧**，可从样本学——$\varepsilon_\theta$（式 2.4）就是在每个噪声水平学一个 score 场 |

#### 去噪器为什么天生知道箭头方向

站在 $x_t$ 处问去噪器「干净图最可能是什么」，它的答案是 $\hat x_0=\mathbb E[x_0\mid x_t]$（后验均值）。那么**从 $x_t$ 朝 $\hat x_0$ 的方向走，就是朝数据密集处走**——直觉就这一句。**Tweedie 公式**只是把这句直觉写成等式，并给出精确的换算比例：

$$
\mathbb E[x_0\mid x_t] = \frac{x_t+(1-\bar\alpha_t)\,\nabla_{x_t}\log q(x_t)}{\sqrt{\bar\alpha_t}}
\qquad\Longleftrightarrow\qquad
\nabla_{x_t}\log q(x_t) = -\frac{x_t-\sqrt{\bar\alpha_t}\,\mathbb E[x_0\mid x_t]}{1-\bar\alpha_t}.
$$

（右式分子 =「当前图与缩放后干净图猜想的差」，指向 $\hat x_0$ 那一侧；分母是调度表常数。对任意数据分布都成立——推导见折叠块，不依赖 $x_0$ 的分布形状。）

#### 换到 ε 语言，接上 DDPM 的训练

DDPM 实际训练的不是 $\hat x_0$ 而是 $\varepsilon_\theta$。两者由式 (2.1) 的恒等式 $\big(x_t-\sqrt{\bar\alpha_t}\,x_0\big)\big/\sqrt{1-\bar\alpha_t}=\varepsilon$ 直接连通——把上式分子改写一下：

$$
\nabla_{x_t}\log q(x_t) = -\frac{\mathbb E[\varepsilon\mid x_t]}{\sqrt{1-\bar\alpha_t}}
\qquad\Longrightarrow\qquad
s_\theta(x_t,t) := -\frac{\varepsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}} \;\approx\; \nabla_{x_t}\log q(x_t).
\tag{2.4}
$$

（$\Longrightarrow$ 那步用到 §2.2 已证的事实：MSE 回归的最优解是条件均值，所以训到收敛时 $\varepsilon_\theta=\mathbb E[\varepsilon\mid x_t]$。）

三个量是**同一个去噪器的三种坐标**，互相换算只需调度表：

| 手里有 | 想要 | 换算 |
|---|---|---|
| $\varepsilon_\theta$ | score | $-\varepsilon_\theta\big/\sqrt{1-\bar\alpha_t}$ |
| $\varepsilon_\theta$ | $\hat x_0$ | $\big(x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon_\theta\big)\big/\sqrt{\bar\alpha_t}$ |
| $\hat x_0$ | score | $-\big(x_t-\sqrt{\bar\alpha_t}\,\hat x_0\big)\big/(1-\bar\alpha_t)$ |
| score | $\hat x_0$ | $\big(x_t+(1-\bar\alpha_t)\,s\big)\big/\sqrt{\bar\alpha_t}$ |

（§4.3 那张「三种参数化互相转换」的表正是这一张的推广；CFG-Ctrl（VG 11）能自由选择在哪个量上施加控制，靠的也是这套换算。）

**一个容易踩的理解坑**：式 (2.4) 里的 $\varepsilon$ 是 $\mathbb E[\varepsilon\mid x_t]$（**平均**要去的噪声），不是训练时掷出的那个具体骰子。原因：同一个 $x_t$ 有多种来历（$x_0$ 大配 $\varepsilon$ 小、$x_0$ 小配 $\varepsilon$ 大……都产生同一个 $x_t$），而箭头场是 $x_t$ 的确定函数——一个点只有一个方向，只能取所有来历的平均意见。数字标本里把三种来历逐个列了出来。

这是与 score-based 生成模型（Song & Ermon 的 score matching 系列）统一的桥梁——§3.3 的概率流 ODE 直接把 $\nabla_x\log p_t(x)$ 写进方程（网络输出 $\varepsilon_\theta$，按式 (2.4) 换算即可），后面 CFG、蒸馏、Flow Matching 全部在 score/velocity 场上操作，这个视角比 $\varepsilon$ 参数化更本质。

<details>
<summary><b>完整推导三步：从定义到 式 (2.4)（想把细节钉死时再读）</b></summary>

**第一步：边缘 score = 条件 score 的后验平均。** 想算的是边缘分布 $q(x_t)=\int q(x_t\mid x_0)\,q(x_0)\,dx_0$ 的 score，手里只有条件版本（式 (2.1)：一个高斯）。**不能**把条件 score 对 $x_0$ 直接积分——那会丢掉「每个 $x_0$ 占多少权重」。对密度积分本身求梯度，一步步换装。先写下起点（链式法则 $\nabla\log q=\frac{\nabla q}{q}$，再把 $q(x_t)$ 展开成定义它的积分）：

$$
\nabla_{x_t}\log q(x_t)
= \frac{\nabla_{x_t}\int q(x_t\mid x_0)\,q(x_0)\,dx_0}{q(x_t)}.
$$

**第二个等号：三件小事的连击**（没有一步是近似）：

1. **$\nabla$ 挪进积分号**（Leibniz 积分法则；高斯密度对 $x_t$ 光滑、衰减够快，条件满足）。挪的时候只碰含 $x_t$ 的因子——$q(x_0)$ 与 $x_t$ 无关，原样留下。分子变成 $\int q(x_0)\,\nabla_{x_t}q(x_t\mid x_0)\,dx_0$；
2. **换装** $\nabla q=q\,\nabla\log q$：$\nabla_{x_t}q(x_t\mid x_0)=q(x_t\mid x_0)\,\nabla_{x_t}\log q(x_t\mid x_0)$；
3. **$\frac{1}{q(x_t)}$ 挪进积分号**：它是「$x_t$ 处的边缘密度」，对积分变量 $x_0$ 是常数，自由进出。

$$
\nabla_{x_t}\log q(x_t)
= \int \frac{q(x_0)\,q(x_t\mid x_0)}{q(x_t)}\,\nabla_{x_t}\log q(x_t\mid x_0)\,dx_0.
$$

（换装那步的来历与用意。**①$\nabla q=q\,\nabla\log q$ 是哪来的**：对 $\log q(x)$ 求导，链式法则给 $\frac{d}{dx}\log q=\frac{1}{q}\cdot\frac{dq}{dx}$，两边乘回 $q$ 即得。**②为什么非要这步换装**：积分里的被积项原本是「密度的梯度」$\nabla q(x_t\mid x_0)\cdot q(x_0)$，直接长不出后验结构；换成 $q\,\nabla\log q$ 后多出的 $q$ 因子恰好与分母 $q(x_t)$ 凑成下一步要认的后验权重——没有这步，积分里认不出「后验平均」。（一般化的读法：凡「梯度和密度的乘积积分」$\int f\,\nabla q\,dx$ 都可用这招换成密度加权的对数梯度积分——概率里叫 Fisher 恒等式，score matching 的推导是同一招。））

**第三个等号：认出贝叶斯。** 积分里的权重 $\frac{q(x_0)\,q(x_t\mid x_0)}{q(x_t)}$ 不是碰巧长这样——它正是后验密度的定义。条件密度的乘法公式（同一块联合概率质量，两种切法）：

$$
q(x_0\mid x_t)\,q(x_t)=q(x_t\mid x_0)\,q(x_0)
\qquad\Longrightarrow\qquad
q(x_0\mid x_t)=\frac{q(x_t\mid x_0)\,q(x_0)}{q(x_t)}.
$$

代入积分，「后验加权的条件 score」正是条件期望的定义：

$$
\nabla_{x_t}\log q(x_t)
= \mathbb E_{q(x_0\mid x_t)}\big[\nabla_{x_t}\log q(x_t\mid x_0)\big].
$$

**读法：$x_t$ 处的箭头 = 所有能产生这个 $x_t$ 的干净图的意见取平均**——「平均意见」的严格含义就在这一步。

**第二步：条件 score 显式算出。** $q(x_t\mid x_0)=\mathcal N\big(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)\,\mathbf I\big)$，对数密度只剩一个二次型，求梯度（$\nabla_x\|x-a\|^2=2(x-a)$）：

$$
\log q(x_t\mid x_0) = -\frac{\|x_t-\sqrt{\bar\alpha_t}\,x_0\|^2}{2(1-\bar\alpha_t)} + \text{const}
\quad\Longrightarrow\quad
\nabla_{x_t}\log q(x_t\mid x_0) = -\frac{x_t-\sqrt{\bar\alpha_t}\,x_0}{1-\bar\alpha_t}.
$$

**第三步：代回，解出 Tweedie。** 第二步代入第一步，期望线性拆开——被积式是 $x_0$ 的仿射函数，$x_t$ 在「给定 $x_t$」的后验下是常数（$\mathbb E[x_t\mid x_t]=x_t$），直接提出：

$$
\nabla_{x_t}\log q(x_t)
=\mathbb E_{q(x_0\mid x_t)}\!\left[-\frac{x_t-\sqrt{\bar\alpha_t}\,x_0}{1-\bar\alpha_t}\right]
=-\frac{x_t-\sqrt{\bar\alpha_t}\,\mathbb E[x_0\mid x_t]}{1-\bar\alpha_t}.
$$

这一行已是 Tweedie **右式**。两边乘 $-(1-\bar\alpha_t)$、移项、除 $\sqrt{\bar\alpha_t}$，解出 $\mathbb E[x_0\mid x_t]$ 即正文 Tweedie **左式**：

$$
\mathbb E[x_0\mid x_t] = \frac{x_t+(1-\bar\alpha_t)\,\nabla_{x_t}\log q(x_t)}{\sqrt{\bar\alpha_t}}.
$$

**再换到 ε 语言**：把每样本恒等式 $x_t-\sqrt{\bar\alpha_t}\,x_0=\sqrt{1-\bar\alpha_t}\,\varepsilon$（式 (2.1) 移项，对产生这个 $x_t$ 的每一对 $(x_0,\varepsilon)$ 精确成立）两边取 $\mathbb E[\cdot\mid x_t]$——期望保持等式——得 $x_t-\sqrt{\bar\alpha_t}\,\mathbb E[x_0\mid x_t]=\sqrt{1-\bar\alpha_t}\,\mathbb E[\varepsilon\mid x_t]$，代回上式分子：

$$
\nabla_{x_t}\log q(x_t)=-\frac{\mathbb E[\varepsilon\mid x_t]}{\sqrt{1-\bar\alpha_t}},
$$

即式 (2.4) 的左端（网络收敛时 $\varepsilon_\theta\to\mathbb E[\varepsilon\mid x_t]$，换算即得 $s_\theta$）。全程对 $q(x_0)$ 的形状**零假设**——Tweedie 是普遍公式，不挑数据分布。**非高斯先验下的数值核对**（混合先验 $0.5\mathcal N(-2,0.6^2)+0.5\mathcal N(2,0.6^2)$，$\bar\alpha_t=0.941192$，取三个 $x_t$ 点）：直接算边缘 score（对数密度差分）与走这条链（后验加权）逐位相等（如 $x_t=0.3$ 处两边都是 $3.629366$）；Tweedie 解出的 $\mathbb E[x_0\mid x_t]=0.529233$ 与逐分量共轭计算一致——「对任意分布成立」不是口头的。

</details>

<details>
<summary><b>数字标本：toy 先验 $x_0\sim\mathcal N(2,\,0.6^2)$，$\beta_t=0.02$、$t=3$（$\bar\alpha_3=0.941192$），取 $x_3=2.0$</b></summary>

高斯先验下边缘仍是高斯（§2.1 的「独立高斯相加」：$x_3=\sqrt{\bar\alpha_3}\,x_0+\sqrt{1-\bar\alpha_3}\,\varepsilon$，两个独立项）。先备好边缘的参数：

| 量 | 公式 | 数值 |
|---|---|---|
| 边缘均值 | $\sqrt{0.941192}\times 2$ | $1.94030$ |
| 边缘方差 | $0.941192\times 0.36 + 0.058808$ | $0.397637$ |

**核对 1：两条路算同一个箭头，逐位对上**（一维小标本那条公式 $-\frac{x-\text{中心}}{\text{方差}}$ 直接用）：

| 路线 | 公式 | 数值 |
|---|---|---|
| 直算边缘 score | $-\dfrac{2.0-1.94030}{0.397637}$ | $-0.1501$ |
| 噪声路线：先算 $\mathbb E[\varepsilon\mid x_3]$ | $\dfrac{\sqrt{0.058808}}{0.397637}\times(2.0-1.94030)$ | $0.03641$ |
| 噪声路线：代入 式 (2.4) | $-\dfrac{0.03641}{\sqrt{0.058808}}$ | $-0.1501$ ✓ |

（$\mathbb E[\varepsilon\mid x_3]$ 用联合高斯的线性条件期望：$\mathbb E[\varepsilon\mid x_t]=\frac{\mathrm{Cov}(\varepsilon,x_t)}{\mathrm{Var}(x_t)}\big(x_t-\mathbb E[x_t]\big)$，其中 $\mathrm{Cov}(\varepsilon,x_t)=\sqrt{1-\bar\alpha_t}$——由 $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon$ 与 $\varepsilon\perp x_0$ 一行可验。）

**核对 2：「同一个 $x_3$ 有多种来历」**——$x_3=2.0$ 可以由：

| 来历 | $x_0$ | 配套的 $\varepsilon=\frac{2.0-0.97015\,x_0}{0.24252}$ |
|---|---|---|
| 1 | $2.0$（先验均值附近） | $0.2462$ |
| 2 | $1.9$ | $0.6462$ |
| 3 | $2.1$ | $-0.1539$ |

三个 $\varepsilon$ 各不相同，但箭头只看 $x_3$ 这个点本身：它等于所有来历按后验加权的平均意见 $\mathbb E[\varepsilon\mid x_3]=0.03641$——不是上表中的任何一个数，也不该是。若误把「采样时那个具体 $\varepsilon$」当成箭头的来源，「$\varepsilon_\theta$ 对应 score」这层关系就断了；正确读法是：**$\varepsilon_\theta$ 收敛到后验均值，后验均值按式 (2.4) 换算出箭头**。

</details>

### 2.5 回头看：一千个小步在导航什么

开头「几何总览」留了两句话没有展开——为什么「一步去噪」必然糊而「小步走」不糊，以及 score 如何随噪声水平换挡。当时只给了结论；现在式 (2.2)（每步去噪的实际形状）与 score 视角（§2.4）都齐了，可以把它们讲透。

**第一句：为什么「一步去噪」必然糊、「小步走」就不糊。** 关键在两个量随噪声水平 $\sigma_t$ 的**此消彼长**：

<details>
<summary><b>展开：峰数此消彼长、一步可达的局部均值、随机性是投票的骰子（两高斯混合微缩模型）</b></summary>

- **条件分布的峰数随 $\sigma_t$ 增多**。$\mathbb E[x_0\mid x_t]$ 里的条件是「$x_t$ 固定」。$\sigma_t$ 小时，$x_t$ 贴着某个数据点，能产生它的 $x_0$ 只有小邻域里那一个模式——条件分布**单峰**，均值就是那个峰，不糊。$\sigma_t$ 大时，$x_t$ 离所有数据都远，**很多个相距甚远的模式都能以相近的概率产生它**——条件分布**多峰**，而均值无视峰的结构、落在峰之间的谷底（§1.3 方波标本的放大版）。极端情形 $\sigma_t\to$ 很大时，$\mathbb E[x_0\mid x_t]\to$ 全数据集的平均——一张所有图像的平均图，灰蒙蒙一片。

- **但每一步的去噪目标不是 $\mathbb E[x_0\mid x_t]$ 本身，而是「一步可达的局部均值」**。看式 (2.2) 的结构：$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}(x_t-\cdots)+\sigma_t z$——它输出的是 $x_{t-1}$，一个离 $x_t$ **只有 $\sigma_t$ 量级**的点，不是直接输出 $x_0$。用两个高斯的混合写一个微缩模型（数据 = 两个模式 $\mu_1,\mu_2$ 各占一半）：

$$
\mathbb E[x_0\mid x_t]=\frac{w_1(x_t)\,\mu_1+w_2(x_t)\,\mu_2}{w_1(x_t)+w_2(x_t)},\qquad
w_i(x_t)=\exp\!\Big(-\frac{\|x_t-\mu_i\|^2}{2\sigma_t^2}\Big).
$$

权重 $w_i$ 是**软的最近邻投票**：$\sigma_t$ 小时，离哪个模式近，那个 $w_i$ 指数压倒另一个，均值 $\approx$ 单个 $\mu_i$——决出了模式；$\sigma_t$ 大时两个 $w_i$ 几乎相等，均值 $\approx\frac{\mu_1+\mu_2}{2}$——两模式的中点，谷底。**模式的决出是随 $\sigma_t$ 减小而连续发生的「相变」**，而每步的 $\sigma_t$ 降幅很小——所以从「还没决出」（$w_1\approx w_2$）到「决出」（$w_1\gg w_2$）之间隔着许多步。**每一步只需要在「软投票比例」上挪一小格**（$x_t$ 向被投票的模式靠近一点 ⟹ 下一轮该模式的权重进一步上升 ⟹ 正反馈），而不需要在一步内从谷底跳到某个峰。

- **随机性 $z$ 在这里不是噪声、是投票的骰子**。决定「最终落进哪个模式」的信息不在均值里（均值恰好在谷底、不偏向任何模式），而在逐步注入的 $\sigma_t z$ 里：随机晃动让 $x_t$ 偏向某一侧，权重天平随之倾斜，正反馈放大，轨迹滑进那个模式。这就是「模式选择交给随机性慢慢消解」的机制本义——**均值给方向，随机性做决定**。一步生成没有骰子可掷（或等价地，掷一次骰子就得直接宣布结果），只能输出均值 = 谷底 = 糊。

（把这个链条与 §1.3 对照：VAE 的解码器在 $\sigma\to$「一步之噪声」的尺度上做全局回归，被困在谷底；扩散模型把同一个「回归到均值」的算子用了一千次，每次只作用在一个几乎单峰的局部条件分布上——**均值机器还是均值机器，但喂给它的分布换了**。）

</details>

**第二句：score 为什么早期指向全局质心、后期退化为局部投影。** 先给两个极端情形的推导，再给一张「随 $\sigma$ 连续过渡」的表。

<details>
<summary><b>两个极端的推导：高噪声端 → 全局质心引力；低噪声端 → 垂直投影</b></summary>

**高噪声端（$\sigma_t$ 大）：score ≈ 指向全局质心。** 数据分布近似离散步（$N$ 个训练点，各带质量 $\frac1N$；流形上的点彼此相距 $\ll\sigma_t$，在这个尺度上「抹平了看」）。此时 $p_t(x)\approx\frac1N\sum_i \mathcal N(x;\mu_i,\sigma_t^2\mathbf I)$，score 是各模式「引力」的加权和：

$$
s(x_t)=\nabla\log p_t(x_t)=\frac{\sum_i w_i(x_t)\cdot\frac{\mu_i-x_t}{\sigma_t^2}}{\sum_i w_i(x_t)},
\qquad w_i(x_t)=\mathcal N(x_t;\mu_i,\sigma_t^2\mathbf I).
$$

读法：**每个数据点都按高斯权重往自己方向拉，score 是这些拉力的加权平均**。$\sigma_t$ 很大时各 $w_i$ 几乎相等（所有点都「看不清但都看得见」），加权平均退化为**无差别平均**：

$$
s(x_t)\approx\frac{\frac1N\sum_i\mu_i-x_t}{\sigma_t^2}=\frac{\bar\mu-x_t}{\sigma_t^2},
\qquad \bar\mu=\frac1N\sum_i\mu_i\ \text{（全局质心）},
$$

即「**全部数据的平均位置拉着你**」——决定的是轨迹**宏观落点**（图像的整体构图、哪一大类模式）。注意分母 $\sigma_t^2$ 很大：这个力**很弱**，要积累很多步才见效（导航是低分辨率的、慢的）。

**低噪声端（$\sigma_t$ 小）：score ≈ 局部投影。** $w_i$ 变得极度尖锐——只有 $x_t$ 最近的数据点（或流形上最近的局部补丁 $\mathrm{proj}_{\mathcal M}(x_t)$）有权重，其余指数归零：

$$
s(x_t)\approx\frac{\mathrm{proj}_{\mathcal M}(x_t)-x_t}{\sigma_t^2}=-\frac{x_t-\mathrm{proj}_{\mathcal M}(x_t)}{\sigma_t^2},
$$

即「**最近的流形点垂直拉着你**」——只负责把 $x_t$ 贴回流形并顺着流形微调，做**局部细节精修**（纹理、边缘锐度）。此时分母 $\sigma_t^2$ 很小：力**很强**，一步之内就把法向偏差拉平（精修是高分辨率的、快的）。这正是开头三镜头表里「处处指回流形的向量场」在两个噪声极端下的具体形状——高噪声端是全局质心引力，低噪声端是这记垂直投影。

</details>

**中间的连续过渡**（一张表看懂「多尺度导航」）：

| 噪声水平 | 有效权重 $w_i$ | score 实质 | 起作用的分辨率 | 这一步在决定什么 |
|---|---|---|---|---|
| $\sigma_t$ 大（早期） | 所有模式近乎等权 | $\frac{\bar\mu-x_t}{\sigma_t^2}$：全局质心引力 | 粗——模式间距量级 | 落在哪个**大区域**（构图、类别、全局布局） |
| $\sigma_t$ 中（中期） | 邻近一群模式加权 | 局部质心引力：几个候选模式按距离加权 | 中——邻域内模式间距 | 收窄到哪个**子模式**（姿态、朝向的大方向） |
| $\sigma_t$ 小（后期） | 只有最近流形补丁有权重 | $-\frac{x_t-\mathrm{proj}_{\mathcal M}(x_t)}{\sigma_t^2}$：垂直投影 | 细——流形局部曲率半径量级 | **细节**（纹理、边缘、像素级锐度） |

设计含义也随之落地：**噪声表就是在安排「决策日程表」**——$\bar\alpha_t$ 下降得慢的区间，轨迹在低分辨率力场里漂得久（有大把时间定全局结构）；下降到尾声时高分辨率力场接管，快速钉死细节。「先构图后细节」不是训练涌现的玄学，是 $\sigma_t$ 从大到小单调下降这件事本身的数学结构（coarse-to-fine 的力场序列）。反过来它也解释了一个失败模式：步数砍太少时，中期「收窄模式」的力被跳过，轨迹从全局质心一步跨到局部投影——模式没决干净就精修，于是细节锐利但**语义混乱**（把两个模式的元素各修一半）。

（一个闭环：§1.1 流形论证的第 4 条说"扩散能去噪本身就是流形假设的证据"——现在能看清为什么：去噪方向有定义，前提是"数据集中、噪声各向同性"，二者叠加才有唯一的"回流形方向"；若数据真铺满 $10^6$ 维空间，每个方向的噪声都不比数据更"假"，去噪无从下手。）

---

## 3. 确定性采样：DDIM 与概率流 ODE

### 3.1 问题：为什么式 (2.2) 必须走 1000 步？

式 (2.2) 每步注入新噪声 $\sigma_t z$，步子迈大了误差爆炸——1000 步成了硬约束。**视频扩散一步的计算量巨大**（后面所有论文都在跟这个成本搏斗），所以"能不能少步数、甚至确定性"是生死问题。

### 3.2 DDIM：换个参数化，采样变 ODE

DDIM 的观察：训练目标只涉及 $q(x_t\mid x_0)$（边缘），**根本没用过 $q(x_{t-1}\mid x_t)$ 这个链式结构**——所以反向过程可以自由重新设计，只要各步边缘对得上。这句话是 DDIM 全部自由的来源，值得拆成两层看清楚。

<details>
<summary><b>两层论证：训练循环里采不到「链」（逐行盘点）+ 链式结构是推导的脚手架</b></summary>

**表层：训练循环里根本采不到「链」。** 逐行盘点 §2.3 那个训练算法触碰到的分布：

| 训练循环的行 | 做了什么 | 用到的分布 |
|---|---|---|
| `x0 ~ 数据集` | 抽一张干净图 | $q(x_0)$（数据集） |
| `ε ~ N(0,I)` | 掷一个骰子 | 标准高斯 |
| `t ~ Uniform{1..T}` | 抽时间步 | 均匀分布 |
| `x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε` | **一步构造** $x_t$ | $q(x_t\mid x_0)$ 的闭式解（式 (2.1)） |
| 梯度 ‖ε − ε_θ(x_t, t)‖² | 回归 | 无新分布 |

全程没有出现过 $x_{t-1}$、任何「上一步」——训练从不一步步模拟链，而是靠式 (2.1) 直接跳到任意 $t$。而式 (2.1) 刻画的恰恰是**单个时刻**的分布：$q(x_t\mid x_0)$ 把链上其余时刻 $x_1,\dots,x_{t-1}$ 全部边缘化掉了——只管「$t$ 时刻长什么样」，不管「链是怎么走到这的」。损失函数的每一个量都只挂在单点 $x_t$ 上。

**深层：链式结构是脚手架，化简完就拆掉了。** 推导确实用了链——但只用在途中：

| 推导环节 | 链式结构出现的地方 | 最后去哪了 |
|---|---|---|
| ELBO 三步分解 | 前向联合 $q(x_{1:T}\mid x_0)$ 按链拆开、逐项配 KL | 拆完每项只含**单点对** $(x_t,x_{t-1})$ |
| 老师均值 $\tilde\mu_t$ | 出身是真反向条件 $q(x_{t-1}\mid x_t,x_0)$（习题 1，链的产物） | 显式公式 $\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$——**只剩调度表常数和 $(x_0,x_t)$** |
| 相消 | 学生/老师同模板，$x_0$、$x_t$ 全部被吸收 | 剩 $\|\varepsilon_\theta-\varepsilon\|^2$ |

链的全部贡献最后浓缩进了几个调度系数。链是推导的脚手架——它论证了「<span class="eqref" data-eqref-tip="\begin{gathered} -\log p_\theta(x_0)\ \le\ \sum_{t=2}^{T}\mathbb{E}_q\big[D_{\mathrm{KL}}\big(q(x_{t-1}\mid x_t,x_0)\ \big\|\ p_\theta(x_{t-1}\mid x_t)\big)\big]+C \\[3pt] \Big\Downarrow\ \text{逐步配成 KL：每项是老师 }q(x_{t-1}\mid x_t,x_0)\text{ 与学生 }p_\theta(x_{t-1}\mid x_t)\text{ 的 KL（§2.3 折叠块）} \\[3pt] D_{\mathrm{KL}}\big(\mathcal{N}(\tilde\mu_t,\tilde\beta_t)\ \big\|\ \mathcal{N}(\mu_\theta,\sigma_t^2)\big)=\text{const}+\frac{1}{2\sigma_t^2}\|\mu_\theta-\tilde\mu_t\|^2 \\[3pt] \Big\Downarrow\ \text{高斯 KL 闭式解只罚均值差；}\ \mu_\theta\text{ 与 }\tilde\mu_t\text{ 同模板，}x_0,x_t\text{ 相消（式 2.2 的 }\varepsilon\text{-形式）} \\[3pt] L_t=\mathbb{E}_{x_0,\varepsilon}\Big[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\ \big\|\varepsilon-\varepsilon_\theta\big(\underbrace{\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon}_{x_t},\ t\big)\big\|^2\Big] \\[3pt] \text{（每步 KL 与带权 }\varepsilon\text{-MSE 只差与 }\theta\text{ 无关的常数；均匀加权是 DDPM 的经验重加权）} \end{gathered}">这个 MSE 确实是变分下界的等价形式</span>」（训练它不亏）；脚手架拆掉后，**最终目标本身是链无关的**：损失只通过 $q(x_t\mid x_0)$ 和调度表取值。

</details>

**反过来读，就是 DDIM 的推论。** 把前面两层论证的箭头反过来指：

| 已证（正向读） | 反过来读 |
|---|---|
| 训练目标的每个量只挂在**边缘** $q(x_t\mid x_0)$（式 (2.1)）上 | 换任何前向过程，只要每个 $t$ 的边缘仍是式 (2.1)，损失函数一字不差——同一个 $\varepsilon_\theta$ 对所有这些前向通用 |
| 链式结构只在推导途中出现（脚手架，化简完就拆） | 生成时**不必沿着链走**——「怎么从 $x_t$ 搬到 $x_{t-1}$」是一个自由的设计变量 |

生成侧唯一不能违背的硬约束：**每一步喂给网络的输入要处在它训练时见过的噪声水平**（边缘对上），否则 $\varepsilon_\theta$ 出分布。在这条约束之内怎么搬运都合法——式 (2.2) 的 DDPM 采样只是众多搬运路线中「抄老师后验」的那一条。DDIM 换了一条：

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t-1} - \sigma_t^2}\,\varepsilon_\theta + \sigma_t z.
$$

读法：**一步用两份预测，重新分配噪声预算**。$\hat x_0$ 定「$t-1$ 时刻的信号在哪」（信号系数 $\sqrt{\bar\alpha_{t-1}}$ 由目标边缘定死）；$\varepsilon_\theta$ 顶替**旧噪声**——$x_t$ 之所以处在 $t$ 水平，正是因为携带着这份噪声，降到 $t-1$ 水平就是把旧噪声按剩余份额保留；想要多少随机性，由新骰子 $z$ 的份额 $\sigma_t^2$ 决定（§2.1 递归式「旧骰子缩一点 + 新骰子掺一点」的同款结构）。旋钮拧到两端：

- $\sigma_t=\sqrt{\tilde\beta_t}$：**随机端点**——代数上精确退化成 DDPM 的随机采样（方差取精确的 $\tilde\beta_t$ 而非 §2.2 的占位 $\beta_t$；逐位核对见折叠块）；
- $\sigma_t=0$：**确定端点**——不掷骰子，每步的 $\hat x_0$、$\varepsilon_\theta$ 都由 $x_t$ 唯一决定，整条轨迹由初始 $x_T$ 唯一决定。采样变成一条 **ODE 轨迹**，时间表可以子采样（跳步），20~50 步即可。

这个设计不是「随便换个公式」——合法性有闭环：它对应一族**非马尔可夫前向**（每步直接「看」$x_0$ 的星形桥接，边缘仍是式 (2.1)），DDIM 论文对这族前向重推 ELBO，训练项还是同一个 ε-MSE，与「训练只见过边缘」互为表里（折叠块末段展开）。一句话：**链只负责证明这个损失值得训，不负责规定生成时怎么走。**

<details>
<summary><b>构造与校准：公式是怎么一步步设计出来的（硬约束 → 装配 → DDPM 端点核对）</b></summary>

**第 0 步：明确要设计什么。** 一个生成更新式 $x_{t-1}=g(x_t,t,z)$（$z$ 是这一步新掷的骰子），必须满足两条硬约束：

| 约束 | 内容 | 为什么必须 |
|---|---|---|
| **R1 边缘保真** | 给定 $x_0$，产出必须 $x_{t-1}\sim\mathcal N(\sqrt{\bar\alpha_{t-1}}\,x_0,\ (1-\bar\alpha_{t-1})\,\mathbf I)$ | $\varepsilon_\theta$ 是在 DDPM 边缘上训的——采样中途喂错噪声水平的输入，立刻出分布 |
| **R2 可实现** | 一步只能用 $(x_t,t)$ 加新随机性 | 生成时没有 $x_0$；但式 (2.1) 的恒等式保证 $(x_0,\varepsilon)$ 与 $x_t$ 一一对应，预测出 $\hat x_0$ 与 $\varepsilon_\theta$ 之一即可换算另一个（§2.4 换算表）——所以一步是 $(\hat x_0,\ \varepsilon_\theta,\ z)$ 的函数 |

**第 1 步：R1 把骨架定死，只剩一个自由度。** R1 说 $x_{t-1}$ 必须是「信号 + 噪声」两段。信号一段没得选：$\sqrt{\bar\alpha_{t-1}}\,x_0$（未知量 $x_0$ 按惯例换成预测 $\hat x_0$）。噪声一段只定了**总量**（方差 $=1-\bar\alpha_{t-1}$），**来源**随便——而手里恰好有两份现成的零均值独立噪声：

- **旧噪声 $\varepsilon$**：把 $x_t$ 带到 $t$ 水平的那个骰子，$\varepsilon=\frac{x_t-\sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}}$（式 (2.1) 反解；生成时用 $\varepsilon_\theta$ 顶替）；
- **新骰子 $z\perp\varepsilon$**：这一步现掷。

两份独立噪声混出单位方差，**系数的平方就是各自贡献的方差份额**，所以配方只有一种——两系数平方和为 1：

$$
\varepsilon'=\sqrt{1-\eta^2}\;\varepsilon+\eta\,z .
$$

**第 2 步：装配。** 按 R1 把信号与混合噪声 $\varepsilon'$ 拼起来，再把旋钮从 $\eta$ 换成更好读的记号 $\sigma_t:=\eta\,\sqrt{1-\bar\alpha_{t-1}}$（即新骰子的方差份额；顺带给出取值范围 $\sigma_t^2\le 1-\bar\alpha_{t-1}$，否则旧噪声份额为负）：

$$
x_{t-1}=\sqrt{\bar\alpha_{t-1}}\,x_0+\sqrt{1-\bar\alpha_{t-1}}\;\varepsilon'
=\sqrt{\bar\alpha_{t-1}}\,x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\;\varepsilon+\sigma_t\,z,
$$

（第二个等号只是代入 $\eta$ 的定义：$\sqrt{1-\bar\alpha_{t-1}}\cdot\sqrt{1-\eta^2}=\sqrt{(1-\bar\alpha_{t-1})-\sigma_t^2}$，$\sqrt{1-\bar\alpha_{t-1}}\cdot\eta=\sigma_t$。）最后把两个未知量换成预测（$x_0\to\hat x_0$、$\varepsilon\to\varepsilon_\theta$），就是正文那条公式。方差核算一行可验：

$$
\underbrace{(1-\bar\alpha_{t-1}-\sigma_t^2)}_{\text{旧噪声份额}}+\underbrace{\sigma_t^2}_{\text{新骰子份额}}=1-\bar\alpha_{t-1}\quad\checkmark
$$

三个系数各司其职：$\sqrt{\bar\alpha_{t-1}}$ 定信号水平（R1 定死）、$\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}$ 是旧噪声被保留的份额、$\sigma_t$ 是新注入的份额——**DDIM 与 DDPM 的全部区别，就压缩在「噪声预算怎么分」这一个自由度上**。

**第 3 步：校准——$\sigma_t=\sqrt{\tilde\beta_t}$ 端点核对。** 到这里公式是「设计」出来的，凭什么信它不是乱定义？最强的证据：存在一个 $\sigma$ 取值，让这条公式**精确**退化成 DDPM 的真反向后验 $\mathcal N(\tilde\mu_t,\tilde\beta_t)$（习题 1 的标准答案）。猜 $\sigma_t^2=\tilde\beta_t=\frac{(1-\bar\alpha_{t-1})\beta_t}{1-\bar\alpha_t}$，把**真值**（$x_0,\varepsilon$，不是预测）代回，逐块核对：

*方差（免费的一步）*：真值代入后，公式里唯一的随机源是 $z$，方差 $=\sigma_t^2=\tilde\beta_t$——自动对上 ✓。

*均值，第一块：$x_t$ 的系数*。DDIM 均值里 $\varepsilon$ 的系数先用一次账本（$1-\bar\alpha_t-\beta_t=\alpha_t(1-\bar\alpha_{t-1})$）整理：

$$
1-\bar\alpha_{t-1}-\tilde\beta_t
=\frac{(1-\bar\alpha_{t-1})\,(1-\bar\alpha_t-\beta_t)}{1-\bar\alpha_t}
=\frac{\alpha_t\,(1-\bar\alpha_{t-1})^2}{1-\bar\alpha_t}
\qquad\Longrightarrow\qquad
\sqrt{1-\bar\alpha_{t-1}-\tilde\beta_t}=\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{\sqrt{1-\bar\alpha_t}}.
$$

把 $\varepsilon=\frac{x_t-\sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}}$ 代回，$x_t$ 拿到的系数是 $\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$——**与 $\tilde\mu_t$ 的 $x_t$ 系数逐字相同** ✓。

*均值，第二块：$x_0$ 的系数*。$x_0$ 拿到的系数是 $\sqrt{\bar\alpha_{t-1}}-\frac{\sqrt{\bar\alpha_t}\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$（第二项来自 $\varepsilon$ 代入后的 $x_0$ 部分）。先化简根号乘积 $\sqrt{\bar\alpha_t}\,\sqrt{\alpha_t}=\alpha_t\,\sqrt{\bar\alpha_{t-1}}$（由 $\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$），提出公因子后只剩账本的另一个变体 $(1-\bar\alpha_t)-\alpha_t(1-\bar\alpha_{t-1})=\beta_t$：

$$
\sqrt{\bar\alpha_{t-1}}\left[1-\frac{\alpha_t\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\right]
=\sqrt{\bar\alpha_{t-1}}\cdot\frac{\beta_t}{1-\bar\alpha_t}.
$$

两块合起来：

$$
\text{DDIM 均值}
=\underbrace{\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}}_{\tilde\mu_t\text{ 的 }x_0\text{ 系数}}\,x_0+\underbrace{\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}}_{\tilde\mu_t\text{ 的 }x_t\text{ 系数}}\,x_t=\tilde\mu_t\quad\checkmark
$$

（能对上的底层原因就是账本 $\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\bar\alpha_t$：$\tilde\mu_t$ 的推导与这里的份额核算记的是同一本账，两边必然说同一种语言。）

方差 $\sigma_t^2=\tilde\beta_t$ 自动对上。**数字标本**（$\beta_t=0.02$、$t=3$、$x_t=1.5$、$x_0=2$，即 §2.2/§2.3 的同一组数）：

| 量 | 公式 | 数值 |
|---|---|---|
| 旧噪声（真值） | $\frac{1.5-\sqrt{0.941192}\times2}{\sqrt{0.058808}}$ | $-1.8156$ |
| DDIM 形式均值（$\sigma^2=\tilde\beta_t$） | $\sqrt{0.9604}\times2+\sqrt{0.026132}\times(-1.8156)$ | $1.66649$ |
| 习题 1 老师 $\tilde\mu_t$ | $\frac{\sqrt{0.9604}\times0.02}{0.058808}\times2+\frac{\sqrt{0.98}\times0.0396}{0.058808}\times1.5$ | $1.66649$ ✓ |
| 方差 | $\tilde\beta_t=\frac{0.0396\times0.02}{0.058808}$ | $0.013468$ |

两条路逐位相等（$1.6664910$）——**构造包含了已知正确的答案作为端点**，这是它不是乱定义的最强证据。$\sigma_t=0$ 则是另一个端点：噪声预算全给旧骰子（$\varepsilon_\theta$ 回收），$z$ 完全不掷——每步的 $\varepsilon_\theta$ 由 $x_t$ 决定，整条轨迹由 $x_T$ 唯一决定，这就是「确定性 ODE」的含义。

**「非马尔可夫」在前向侧**。把生成模型反过来读，对应的前向条件 $q_\sigma(x_{t-1}\mid x_t,x_0)$ 依赖 $x_0$——每步直接「看」干净图（星形桥接而非链），联合 $q(x_{1:T}\mid x_0)$ 无法按链式分解，但每步边缘仍是式 (2.1)。前向的桥接结构给了反向同样的信息通道：生成时看 $\hat x_0$ 合法。训练侧的合法性由 DDIM 附录 B 补上：对这族前向重推 ELBO，训练项仍是同一个 ε-MSE——与正文「训练只见过边缘」的论证互为表里。

</details>

**数值验证**（toy，数据 $x_0\sim\mathcal{N}(2, 0.6^2)$ 单高斯，此时闭式 score 已知、DDPM 反向是精确的）：

```text
DDPM  1000 步（随机）:  mean=1.983, std=0.610   （真值 2.0 / 0.6）✓
DDIM  100 步（确定性）:  mean=1.892, std=0.594   （小误差来自 ᾱ_T≈0.0067≠0）
```

### 3.3 概率流 ODE：扩散的确定性骨架

#### 先把两件工具说清楚：ODE 和 SDE 是什么

§3.2 说「$\sigma_t=0$ 时整条链变成确定性 ODE」——ODE/SDE 到底指什么？一张表（详细定义收进折叠块）：

| | 常微分方程 (ODE) | 随机微分方程 (SDE) |
|---|---|---|
| 形状 | $\frac{dx}{dt}=f(x,t)$ | $dx=f(x,t)\,dt+g(t)\,dW$ |
| 直觉 | 一个**向量场**：每个点每时刻告诉你往哪走、走多快 | 向量场 + **沿途撒随机噪声** |
| 解是什么 | 一条**确定轨迹**：给定 $x(0)$ 整条路唯一确定 | 一族随机轨迹：只有「$t$ 时刻的分布 $p_t$」有意义 |
| 数值解法 | Euler：$x\leftarrow x+f\Delta t$ | Euler–Maruyama：$x\leftarrow x+f\Delta t+g\sqrt{\Delta t}\,\varepsilon$ |

$dW$ 是**布朗运动的增量**：$W(t+\Delta)-W(t)\sim\mathcal N(0,\Delta)$、各段独立——非正式读法 $dW\approx\sqrt{\Delta t}\,\varepsilon$，即 SDE 的模拟公式比 ODE 只多一项「$\sqrt{\Delta t}$ 乘标准高斯」。

**为什么扩散模型会牵扯它们**：看 Euler–Maruyama 那行的形状——$x\leftarrow x+(\text{漂移})+(\text{系数})\times\varepsilon$，**这正是 DDPM 前向步的形状**（$\sqrt{1-\beta_t}$ 缩水 + $\sqrt{\beta_t}\,\varepsilon$）。DDPM 的 1000 步链不是「类似于」某个 SDE——它本身就是一条 SDE 的 Euler–Maruyama 离散化；步数 $T\to\infty$、每步噪声 $\sim1/T$ 的连续极限就是那条 SDE（推导收进折叠块）。于是生成侧也有两副面孔：随机的（反向 SDE，式 (2.2) 采样即其离散化）与确定的（本节的 PF-ODE）。

<details>
<summary><b>ODE/SDE 详细定义 + 从前向链到 VP-SDE 的极限推导</b></summary>

**ODE（普通读者的够用版）**：$\frac{dx}{dt}=f(x,t)$ 描述「点在向量场里流动」。给定初值 $x(0)$，解唯一（$f$ 光滑时）——这就是「确定性」的全部含义。数值求解最朴素的是 Euler 法：把时间切成小段，每段走一步切线。

**SDE 的严格意思**：形如 $dx=f(x,t)\,dt+g(t)\,dW$ 的方程。$W(t)$ 是布朗运动（Wiener 过程）：$W(0)=0$、增量 $W(t+\Delta)-W(t)\sim\mathcal N(0,\Delta)$ 且各段独立、轨道处处连续但处处不可导（所以不能写成 $\frac{dW}{dt}$ 的普通微积分——$dt$ 记号是「对时间累积、对 $dW$ 累积」的微分简写）。**解**不再是函数而是一族随机轨迹；描述它的正确对象是**边缘密度 $p_t(x)$**——「$t$ 时刻撒一大把轨迹，落点分布是什么」。

**从 DDPM 链到 VP-SDE（三行推导）**。把每步占的连续时间记 $\Delta t=1/T$，定义**噪声率密度** $\beta(t):=T\beta_t$（单位时间的噪声预算；$\beta_t=0.02$、$T=1000$ 即 $\beta(t)=20$）。前向一步的增量：

$$
x_t-x_{t-1}=\underbrace{(\sqrt{1-\beta_t}-1)}_{\text{Taylor：}\approx-\beta_t/2}\,x_{t-1}+\sqrt{\beta_t}\;\varepsilon
=-\tfrac12\beta(t)\,x_{t-1}\,\Delta t+\sqrt{\beta(t)\Delta t}\;\varepsilon,
$$

对照 Euler–Maruyama 模板 $x\leftarrow x+f\Delta t+g\sqrt{\Delta t}\,\varepsilon$：漂移 $f=-\frac12\beta(t)x$、扩散 $g=\sqrt{\beta(t)}$。$T\to\infty$（$\Delta t\to0$）取极限：

$$
dx=-\tfrac12\beta(t)\,x\,dt+\sqrt{\beta(t)}\,dW\qquad(\text{VP-SDE}).
$$

**名字 Variance Preserving 的意思**：漂移 $-\frac12\beta x$ 收缩掉的方差恰好抵消噪声注入的方差——离散账本 $\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\bar\alpha_t$ 的连续版。平稳分布是 $\mathcal N(0,\mathbf I)$，终点有明确起点可采。（对照：Song 系列的 VE-SDE 用 $dx=g(t)\,dW$ 只加噪声不收缩，方差一直膨胀——VP/VE 之名由此而来。）

**解的逐字对应**（线性 SDE 可用积分因子显式解；$\bar\alpha(t)=\exp(-\int_0^t\beta)$，因为 $\prod(1-\beta_s)\approx e^{-\sum\beta_s}$）：

| 离散（§2.1 的链） | 连续（VP-SDE） |
|---|---|
| 均值缩水 $\sqrt{1-\beta_t}\,x_{t-1}$ | 漂移 $-\frac12\beta(t)x\,dt$ |
| 每步噪声方差 $\beta_t$ | 扩散 $g^2dt=\beta(t)dt$ |
| $\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$ | $\bar\alpha(t)=\exp(-\int_0^t\beta(s)ds)$ |
| 式 (2.1)：$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$ | 解：$x_t=\sqrt{\bar\alpha(t)}\,x_0+\sqrt{1-\bar\alpha(t)}\,\varepsilon$（**逐字相同**） |

数值核对（$\beta_t=0.02$、$T=1000$）：$t=0.003$ 处离散 $\bar\alpha=0.941192$ vs 连续 $e^{-20t}=0.941765$——O$(\beta^2)$ 每步误差内吻合；正向模拟 20 万样本，终点 mean/std $=-0.002/1.003$ vs 闭式 $0.000/1.000$ ✓。

</details>

#### VP-SDE、反向 SDE 与 PF-ODE

更一般的结论（Song et al. 2021）：**任何扩散 SDE 都有一个保持边缘分布相同的 ODE**。VP-SDE（DDPM 的连续极限）：

$$
dx = -\tfrac{1}{2}\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dW
\quad\Longleftrightarrow\quad
\text{PF-ODE: } \frac{dx}{dt} = -\tfrac{1}{2}\beta(t)\,x - \tfrac{1}{2}\beta(t)\,\nabla_x\log p_t(x).
$$

中间还藏着第三个对象——**反向 SDE**（Anderson 1982，式 (2.2) 的 ancestral 采样就是它的离散化）：$dx=\big[f-g^2\nabla_x\log p_t\big]dt+g\,d\bar W$（$t$ 从 $T$ 倒流向 $0$，$\bar W$ 是反向布朗运动）。三者关系一张表：

| | 反向 SDE | PF-ODE |
|---|---|---|
| 方程 | $dx=\big[f-g^2\nabla\log p_t\big]dt+g\,d\bar W$ | $\frac{dx}{dt}=f-\tfrac12 g^2\nabla\log p_t$ |
| 轨迹 | 随机（每步掷骰子） | **确定**（初值定则全轨迹定） |
| 边缘 $p_t$ | 与前向 SDE 的时间反演相同 | **与反向 SDE 相同** |
| 离散化对应 | 式 (2.2) 的 DDPM 采样（$\sigma_t$ 即旋钮） | §3.2 的 DDIM 取 $\sigma_t=0$——「链变成确定性 ODE」是字面成立的 |

直觉：PF-ODE 把 SDE 的漂移减去"噪声的一半效应"（噪声把分布拉宽，score 把分布拉回数据），两者抵消后每个样本点沿确定轨迹流动，但**轨迹族的密度恰好等于 SDE 的边缘密度**。系数为什么是 $\tfrac12 g^2$、以及「减掉一半」如何精确配平——推导只是把两条「密度演化方程」对齐（Fokker–Planck 配平，四行全等号，收进折叠块）。

<details>
<summary><b>PF-ODE 推导：Fokker–Planck 配平 + 时间方向的坑（含数值验证）</b></summary>

**命题**：SDE $dx=f\,dt+g\,dW$ 与 ODE $\frac{dx}{dt}=f-\tfrac12 g^2\nabla_x\log p_t(x)$ 有完全相同的边缘分布 $p_t$。

推导只需两条「密度随时间怎么演化」的方程：

1. SDE 的边缘密度满足 **Fokker–Planck 方程**：$\partial_t p=-\nabla\cdot(fp)+\tfrac12\nabla^2(g^2p)$——漂移搬运质量、扩散项把质量抹开；
2. ODE $\frac{dx}{dt}=F$ 的边缘密度满足**连续性方程**：$\partial_t p=-\nabla\cdot(pF)$——纯搬运、不抹。

要让 ODE 搬出同样的密度，取 $F=f-\tfrac12 g^2\nabla\log p$ 验算：

$$
-\nabla\cdot(pF)=-\nabla\cdot(fp)+\tfrac12\nabla\cdot\big(g^2\,p\,\nabla\log p\big)
\overset{p\nabla\log p=\nabla p}{=}-\nabla\cdot(fp)+\tfrac12\nabla\cdot(g^2\nabla p)
\overset{g\text{ 只依赖 }t}{=}-\nabla\cdot(fp)+\tfrac12\nabla^2(g^2p).\ \checkmark
$$

与 Fokker–Planck 右端逐项相同——同一个密度演化方程，两条轨道族（一条带噪声一条不带）共享同一套边缘。$\blacksquare$

代入 VP（$f=-\frac12\beta x$、$g^2=\beta$）即正文那条 PF-ODE。**「减掉一半」的读法**：扩散项把分布抹宽的速率是 $\tfrac12 g^2$（Fokker–Planck 扩散项的系数），score 是「往密处拉的力」；$-\tfrac12 g^2\nabla\log p$ 恰好抵掉抹宽效应，剩下的净流量与 SDE 相同。一个立即可验的极端：平稳分布处（$p_t\approx\mathcal N(0,1)$，$\nabla\log p=-x$），VP 的 PF-ODE 漂移 $=-\frac12\beta x+\frac12\beta x=0$——分布本来就不动，ODE 冻结 ✓。

**反向 SDE 的时间方向是个坑**（实现时亲手踩过）：反向 SDE 的 $t$ 从 $T$ 倒流向 $0$；若改用「生成时递增」的步进变量 $\tau=T-t$，漂移要变号成 $-f+g^2\nabla\log p$、噪声项照旧。我用错了方向的第一版模拟 std 爆炸到 $8\times10^4$；改对后（数据 $x_0\sim\mathcal N(2,0.6^2)$、$\beta(t)=20$、20 万样本、Euler 步长 $1/1000$）：

```text
反向 SDE: mean=+1.997 std=0.607   （真值 2.0 / 0.6）✓
PF-ODE : mean=+1.994 std=0.605   （真值 2.0 / 0.6）✓
```

两条路在误差内一致——「同边缘」的数值版。DDIM σ=0 的离散化落在 PF-ODE 一侧（无噪声项），DDPM 采样落在反向 SDE 一侧（噪声系数即 §3.2 的 $\sigma_t$ 旋钮）。

</details>

**数值验证**（toy，双峰混合分布 $0.5\mathcal{N}((-2,0),0.6^2\mathbf{I}) + 0.5\mathcal{N}((2,0),0.6^2\mathbf{I})$，闭式 score，PF-ODE 用 Euler 积分）：

```text
PF-ODE 采样 4000 个样本：
  左右模式占比 49.0% / 51.0%（真值 50/50）✓
  右模式 +2.02 ± 0.58（真值 +2.0 ± 0.6）✓
  左模式 -1.98 ± 0.60（真值 -2.0 ± 0.6）✓
```

（这里有个实现细节值得记录：score 必须用**被 ᾱ_t 缩放过的边缘分布** $p(x_t)=\sum_k w_k\,\mathcal{N}(\sqrt{\bar\alpha_t}\mu_k,\ (1-\bar\alpha_t)+\bar\alpha_t\sigma_k^2)$ 来算。我第一版直接用了 $t=0$ 的均值/方差，PF-ODE 在高斯数据上通过、在混合数据上模式外推到 ±4.9——单高斯的偶然正确掩盖了 bug。**边缘分布随 $t$ 缩放**这件事本身就是一个好的理解测试：$t$ 时刻的"目标分布"不是数据分布，而是被压缩后的版本。）

**两个关键推论**（后面论文反复使用）：

1. **确定性采样 ⟹ 初始噪声到输出的双射**：同一噪声永远生成同一视频/图像。这是 Video-T1（VG 10）"在噪声空间搜索"的前提；
2. **生成 = ODE 求解 ⟹ 可以蒸馏**：ODE 求解器的步数压缩（consistency distillation、progressive distillation、VideoScene 的 Leap Flow）成为独立的加速研究方向（VG 06）。

---

## 4. Flow Matching：换一套坐标的同一套数学

### 4.1 从"加噪"到"插值"

SD3、Flux、以及现代视频生成模型（CogVideoX、WAN 等——见本栏目此前的阅读笔记）普遍改用 **Flow Matching / Rectified Flow** 训练。出发点：把前向过程从"逐步加噪"换成**直线插值**：

$$
x_t = (1-t)\,x_0 + t\,\varepsilon, \qquad x_0\sim q_{\text{data}},\ \varepsilon\sim\mathcal{N}(0,\mathbf{I}),\ t\in[0,1]. \tag{4.1}
$$

$t=0$ 是数据，$t=1$ 是噪声——**每条"噪声→数据"的路径是一条直线**（对给定的 $(x_0,\varepsilon)$ 配对）。生成就是从 $t=1$ 沿速度场积分回 $t=0$。

**为什么可以这样"直线采样"？** 因为前向过程从来就是**设计选择，不是物理定律**——§3.2 已证：链是脚手架，训练只碰边缘。DDPM 那条曲线（$\sqrt{\bar\alpha_t}$ 指数衰减 + 噪声份额增长）只是马尔可夫链脚手架留下的残余形状。对一条概率路径 $\{p_t\}$ 的硬性要求只有两条：$p_0=$ 数据分布、$p_1=$ 能直接采样的简单分布——中间长什么样完全自由。直线插值两个端点**精确**成立（$t=0$ 时 $x_t=x_0$ 原封是数据，$t=1$ 时 $x_t=\varepsilon$ 原封是噪声），且条件分布仍是高斯：$p_{t\mid x_0}=\mathcal N\big((1-t)\,x_0,\ t^2\mathbf I\big)$，与 DDPM 的 $q(x_t\mid x_0)$ 同款结构，只是「信号/噪声份额」的日程表从曲线换成了直线。

一张图景对照（DDPM → FM 不是换理论，是换日程表）：

| | DDPM | Flow Matching |
|---|---|---|
| 路径（$x_0$ 与 $\varepsilon$ 的配比日程） | $\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon$：信号指数缩水、噪声渐涨（**曲线**） | $(1-t)\,x_0+t\,\varepsilon$：匀此消彼长（**直线**） |
| $t$ 两端 | $t=0$ 数据，$t=T$ 近似 $\mathcal N(0,\mathbf I)$ | $t=0$ **精确**数据，$t=1$ **精确**噪声 |
| 条件分布 $p_{t\mid x_0}$ | $\mathcal N\big(\sqrt{\bar\alpha_t}x_0,\ (1-\bar\alpha_t)\mathbf I\big)$ | $\mathcal N\big((1-t)x_0,\ t^2\mathbf I\big)$ |

### 4.2 训练目标：边缘速度场

**速度场是什么、干嘛用的**——三步直觉（推导与数字标本收进折叠块）：

1. **每条轨迹是匀速直线**。固定一对 $(x_0,\varepsilon)$，式 (4.1) 对 $t$ 求导：$\frac{dx_t}{dt}=\varepsilon-x_0$——**常数**，与 $t$ 无关。单条轨迹上「速度」毫无含糊；
2. **但一个点有多条轨迹经过**。生成时手里只有 $\varepsilon$，不知道它与哪个 $x_0$ 配对（有多个$x_0$所以取平均）；同一个 $x_t$ 可由许多 $(x_0,\varepsilon)$ 对产生（§2.4「同一个 $x_3$ 有多种来历」的同款困境）。ODE 的向量场每点只能指**一个**方向——FM 取所有经过 $(x,t)$ 的轨迹的**条件平均**：

$$
u_t(x) := \mathbb{E}\left[\frac{dx_t}{dt}\ \Big|\ x_t = x\right] = \mathbb{E}\left[\varepsilon - x_0\ \big|\ (1-t)x_0 + t\varepsilon = x\right].
$$

3. **训练自动给出这个平均**。回归目标 $\varepsilon-x_0$ 对 $x_t$ 做 MSE 回归，最优解 = 条件均值（§1.3 已证两次）——$\mathbb E[\varepsilon-x_0\mid x_t]$ 正是 $u_t(x_t)$。**与 $\varepsilon_\theta\to\mathbb E[\varepsilon\mid x_t]$（§2.4）是同一个机制，连收敛理由都一字不差。**

**Flow Matching 定理**（Lipman et al. 2023）：如果网络 $v_\theta(x_t,t)$ 拟合了 $u_t$，则 ODE $\frac{dx}{dt} = v_\theta(x, t)$ 从 $t=1$ 积分到 $0$ 产生的分布恰好是 $q_{\text{data}}$。定理的机制与 §3.3 的 PF-ODE 是**同一台机器**：$p_t$ 是云的形状随 $t$ 变化的「电影」（$t=0$ 云贴在数据流形上，$t=1$ 云摊成高斯），速度场就是**把这部片子倒放的导航场**——在任意 $(x,t)$ 告诉粒子下一步往哪挪，使整片云从噪声凝结回流形。理想配对下每个粒子本该走匀速直线；粒子不知道自己从哪来，场给它「所有可能来历的后验平均方向」。所以 score（§2.4）与 $u_t$ 是同一类对象——后验平均的罗盘场，只是服务的路径不同：

| | score（§2.4） | FM 速度场（本节） |
|---|---|---|
| 服务的路径 | DDPM 曲线（加噪日程） | 直线插值 |
| 场 | $\nabla_{x_t}\log p_t$：后验平均的**拉力** | $u_t=\mathbb E[\varepsilon-x_0\mid x_t]$：后验平均的**速度** |
| 网络收敛到 | $\mathbb E[\varepsilon\mid x_t]$ | $\mathbb E[\varepsilon-x_0\mid x_t]$ |
| 收敛理由 | MSE 最优 = 条件均值 | MSE 最优 = 条件均值（同一个） |
| 生成 | 沿 PF-ODE 积分 $T\to0$ | 沿 ODE 积分 $1\to0$ |

训练目标：

$$
\boxed{\ L(\theta) = \mathbb{E}_{x_0,\,\varepsilon,\,t}\left[\big\|v_\theta\big((1-t)x_0 + t\varepsilon,\ t\big) - (\varepsilon - x_0)\big\|^2\right]\ }
$$

**为什么回归目标从 $\varepsilon$ 换成了 $\varepsilon-x_0$？** 总原则一句话：**学的 = 采样用的**。DDPM 学 $\varepsilon$ 不是因为「噪声」本身重要，而是因为采样引擎是 score、$\varepsilon$ 经式 (2.4) 换算就是 score；FM 的采样机器是 ODE $\frac{dx}{dt}=v_\theta$，右端要的量是**速度场**——学的量随引擎走。而 $\varepsilon-x_0$ 不是新发明的量：把式 (4.1) 对 $t$ 求导，$\frac{dx_t}{dt}=-x_0+\varepsilon$，它**就是路径本身的速度**。那能不能继续回归 $\varepsilon$ 或 $x_0$、采样时再换算成速度？可以（§4.3 的三种坐标），但换算 $u_t=\frac{x_t-\mathbb E[x_0\mid x_t]}{t}$ 要**除以 $t$**——在生成收尾的 $t\to0$ 一侧数值病态、误差被放大 $\tfrac1t$ 倍；直接回归 $\varepsilon-x_0$ 既避开换算，目标方差 $\mathrm{Var}(\varepsilon-x_0)$ 又与 $t$ 无关、训练信号全程稳定（数值核对收进折叠块）。

<details>
<summary><b>数字标本：换算恒等式、$/t$ 误差放大、目标方差恒定（$x_0\sim\mathcal N(2,0.6^2)$）</b></summary>

**换算恒等式**（$\mathbb E[\varepsilon\mid x_t]-\mathbb E[x_0\mid x_t]=\frac{x_t-\mathbb E[x_0\mid x_t]}{t}$，由塔性质 $(1-t)\,\mathbb E[x_0\mid x_t]+t\,\mathbb E[\varepsilon\mid x_t]=x_t$ 一行可得）：$t=0.5$ 处 50 万样本 Monte-Carlo 验证，两种算法的最大偏差 $9\times10^{-16}$ ✓。

**$/t$ 的误差放大**：设 $x_0$-预测器带固定误差 $\delta=0.01$，换算成速度后误差 $=\delta/t$：

| $t$ | 0.5 | 0.1 | 0.05 | 0.01 |
|---|---|---|---|---|
| 换算后速度误差 | 0.02 | 0.10 | 0.20 | **1.00** |
| 直接回归 $v$ 的误差 | 0.01 | 0.01 | 0.01 | 0.01 |

$t=0.01$ 处放大 100 倍——生成恰要在 $t\to0$ 一侧收尾，这是换算路线的硬伤。

**目标方差全程恒定**：$\mathrm{Var}(\varepsilon-x_0)=\mathrm{Var}(\varepsilon)+\mathrm{Var}(x_0)=1+0.36=1.36$，与 $t$ 无关（回归目标是两条端点分布的「位移」，天然不随路径位置缩放）。对照 $\mathbb E[\varepsilon\mid x_t]$ 的方差：$t=0$ 时 $0$、$t=0.25$ 时 $0.085$、$t=1$ 时 $0.36$——$\varepsilon$-回归的信号强度随 $t$ 漂移，早期极弱；$v$-回归全程同一尺度。

</details>

<details>
<summary><b>定理为什么成立：连续性方程配平 + 高斯 toy 数字标本</b></summary>

**引理（与 §3.3 PF-ODE 的配平同构）**：若向量场 $u_t$ 满足**连续性方程** $\partial_t p_t=-\nabla\cdot(p_t\,u_t)$，则 ODE $\frac{dx}{dt}=u_t$ 的流把 $p_1$ 搬运回 $p_0$。连续性方程就是质量守恒：某区域密度变化 = 净流出，其中「单位面积的流量」= 密度 × 速度。

**条件平均速度恰好满足它**（机制版论证）：把每条轨迹 $x_t=(1-t)x_0+t\varepsilon$ 看作一个携带概率质量的粒子，速度恒为 $\varepsilon-x_0$。大量粒子过同一点 $(x,t)$、速度各异，**净流量 = 流量对粒子求平均**：

$$
p_t(x)\,u_t(x)=\underbrace{p_t(x)}_{\text{粒子密度}}\times\underbrace{\mathbb E[\varepsilon-x_0\mid x_t=x]}_{\text{平均速度}}
=\mathbb E\big[\delta(x-x_t)\,(\varepsilon-x_0)\big],
$$

（最后一步是「密度 × 条件平均 = 对粒子求平均」的测度论写法。）对每个光滑测试函数 $\phi$，$\frac{d}{dt}\mathbb E[\phi(x_t)]=\mathbb E[\nabla\phi(x_t)\cdot(\varepsilon-x_0)]=\int\nabla\phi\cdot u_t\,p_t\,dx$，分部积分即得连续性方程——「平均速度」正是让质量守恒成立的那一个选择。$\blacksquare$

**数字标本**（$x_0\sim\mathcal N(2,0.6^2)$，此时 $u_t$ 有闭式：联合高斯条件期望给

$$
u_t(x)=\frac{t-(1-t)\,\sigma_0^2}{(1-t)^2\sigma_0^2+t^2}\,\big(x-(1-t)\mu\big)-\mu,
\qquad \mu=2,\ \sigma_0^2=0.36;
$$

注意常数项 $-\mu$——它来自 $-\mathbb E[x_0\mid x_t]$，漏掉它 ODE 会积到错误的分布，我第一版就漏了）：

| 核对 | 结果 | 真值 |
|---|---|---|
| $t=0.5$ 处 40 万样本 MC 回归 $(\varepsilon-x_0)\sim x_t$ | 斜率/截距 $0.938/-2.939$ | 闭式 $0.941/-2.941$ ✓ |
| ODE 从 $t=1$（$x\sim\mathcal N(0,1)$）积回 $0$ | $2.00\pm0.60$ | $2.0/0.6$ ✓ |
| ODE 从 $t=0$ 积到 $1$（正向搬运） | $0.00\pm1.00$ | $0.0/1.0$ ✓ |
| 双峰混合 $0.5\mathcal N(\pm2,0.6^2)$ 生成 | 左右 50.0%/50.0%，$\pm2.00\pm0.60$ | 50/50，$\pm2.0\pm0.6$ ✓ |

（闭式推导：$x_t$ 与 $(x_0,\varepsilon)$ 联合高斯，$\mathrm{Cov}(\varepsilon,x_t)=t$、$\mathrm{Cov}(x_0,x_t)=(1-t)\sigma_0^2$、$\mathrm{Var}(x_t)=(1-t)^2\sigma_0^2+t^2$，标准条件期望公式分别算 $\mathbb E[\varepsilon\mid x_t]$、$\mathbb E[x_0\mid x_t]$ 相减即得。双峰时对两个模式做后验加权平均——「平均意见」机制与 §2.4 的 $\mathbb E[\varepsilon\mid x_3]$ 标本完全平行。）

</details>

训练循环的全景——**形式与 DDPM 完全同构**（只有回归目标一行变了）：

| | DDPM | Flow Matching |
|---|------|---------------|
| 回归目标 | $\varepsilon$ | $\varepsilon - x_0$（速度） |
| 采样 | 反向 SDE 或 ODE，$T\to 0$ | ODE，$t\to 0$ |

**为什么大家迁移到 FM？** 路径是直线 ⟹ ODE 轨迹更直 ⟹ **少步采样的离散误差更小**。对视频生成（每步计算量巨大）这是决定性优势。（Rectified Flow 还可以"重整流"——反复拉直轨迹——进一步逼近一步生成；VideoScene 的 Leap Flow Distillation 可以看作这个思路在 3D 场景任务上的变体，VG 06 详述。）

### 4.3 三种参数化的互相转换（一张表）

先把两条公式放进同一个坐标系。任何插值路径统一写成 $x_t=a_t\,x_0+\sigma_t\,\varepsilon$（DDPM：$a_t=\sqrt{\bar\alpha_t},\ \sigma_t=\sqrt{1-\bar\alpha_t}$；FM：$a_t=1-t,\ \sigma_t=t$）——「手里的观测」永远是 $x_t$，「想要的干净量」永远是 $\mathbb E[x_0\mid x_t]$，**同一个去噪器**。两条公式于是统一成一行（Tweedie 统一式，DDPM 情形即 §2.4 右式）：

$$
\boxed{\ \text{score}=-\frac{x_t-a_t\,\mathbb E[x_0\mid x_t]}{\sigma_t^2}\qquad\Longleftrightarrow\qquad \varepsilon\text{-pred}=\frac{x_t-a_t\,\mathbb E[x_0\mid x_t]}{\sigma_t}=-\sigma_t\cdot\text{score}\ }
$$

（与式 (2.4) 是同一条公式的两件外套：代入 DDPM 系数后统一式即 §2.4 的 Tweedie 右式；把分子用每样本恒等式 $x_t-\sqrt{\bar\alpha_t}x_0=\sqrt{1-\bar\alpha_t}\,\varepsilon$ 的条件期望版改写（$x_t-\sqrt{\bar\alpha_t}\,\mathbb E[x_0\mid x_t]=\sqrt{1-\bar\alpha_t}\,\mathbb E[\varepsilon\mid x_t]$），再约去一个 $\sqrt{1-\bar\alpha_t}$，就得到 $-\mathbb E[\varepsilon\mid x_t]/\sqrt{1-\bar\alpha_t}$——纯代数、无近似。两件外套的核心量不同：这边用去噪器本尊 $\mathbb E[x_0\mid x_t]$ 讲「统一坐标」，§2.4 用网络实际回归的 $\mathbb E[\varepsilon\mid x_t]$ 讲「训练怎么接上 score」。）

<details>
<summary><b>逐步推导：从统一式到 ε 语言的那行代数（含卡点自查）</b></summary>

§2.4 折叠块给过一步版，这里拆到每条性质。起点是已证的 Tweedie 右式（代入 DDPM 系数 $a_t=\sqrt{\bar\alpha_t}$、$\sigma_t^2=1-\bar\alpha_t$）：

$$
\nabla_{x_t}\log q(x_t)=-\frac{x_t-\sqrt{\bar\alpha_t}\,\mathbb E[x_0\mid x_t]}{1-\bar\alpha_t}.
$$

**第 1 步：把每样本恒等式两边取 $\mathbb E[\cdot\mid x_t]$。** 前向定义式移项：

$$
x_t-\sqrt{\bar\alpha_t}\,x_0=\sqrt{1-\bar\alpha_t}\,\varepsilon\qquad\text{a.s.}
$$

关键读法：这不是「均值层面的等式」，而是**对每一次投骰子的结果 $(x_0,\varepsilon)$ 逐点成立**——$x_t$ 本来就是由这一对定义出来的，两边是**同一个随机变量**，取任何条件期望都保持相等。左边逐项拆：

| 性质 | 用法 |
|---|---|
| 线性（无需独立性） | $\mathbb E[x_t-\sqrt{\bar\alpha_t}\,x_0\mid x_t]=\mathbb E[x_t\mid x_t]-\sqrt{\bar\alpha_t}\,\mathbb E[x_0\mid x_t]$ |
| 取出已知量 | $\mathbb E[x_t\mid x_t]=x_t$——给定 $x_t$ 后它已冻结成数（§2.4 第三步同一招） |
| 常数提出 | $\mathbb E[\sqrt{1-\bar\alpha_t}\,\varepsilon\mid x_t]=\sqrt{1-\bar\alpha_t}\,\mathbb E[\varepsilon\mid x_t]$ |

得条件期望版恒等式 $x_t-\sqrt{\bar\alpha_t}\,\mathbb E[x_0\mid x_t]=\sqrt{1-\bar\alpha_t}\,\mathbb E[\varepsilon\mid x_t]$。两边现在都是 $x_t$ 的确定函数——随机性被两边**一致地**平均掉（给定 $x_t$ 后 $x_0$ 与 $\varepsilon$ 完全负相关，这两个条件期望不是各自独立算的平均，但恒等式是线性的，一致平均恰好保住等式）。

**第 2 步：代回分子、认出平方再约分。**

$$
-\frac{x_t-\sqrt{\bar\alpha_t}\,\mathbb E[x_0\mid x_t]}{1-\bar\alpha_t}
=-\frac{\sqrt{1-\bar\alpha_t}\,\mathbb E[\varepsilon\mid x_t]}{\underbrace{1-\bar\alpha_t}_{=(\sqrt{1-\bar\alpha_t})^2}}
=-\frac{\mathbb E[\varepsilon\mid x_t]}{\sqrt{1-\bar\alpha_t}}.
$$

能恰好约掉一个因子不是巧合：score 的分母是 $\sigma_t^2$（**方差**），恒等式的系数是 $\sigma_t$（**幅值**），前者本来就是后者的平方。

**卡点自查**（推不出来时，多半踩了其中一条）：

| 卡点 | 症状 | 解法 |
|---|---|---|
| 想从零**计算** $\mathbb E[\varepsilon\mid x_t]$ | 套联合高斯公式算 Cov/Var，越算越繁，还误以为只对高斯先验成立 | 不用算——恒等式逐点成立，取条件期望即可，任意先验都对 |
| 「$\varepsilon$ 不是 $x_t$ 的函数，怎能在 $\mathbb E[\cdot\mid x_t]$ 里操作」 | 卡在第 1 步不敢动手 | 从不需要 $\varepsilon$ 由 $x_t$ 单独决定；条件期望替你把所有来历平均掉，线性保证等式不破 |
| 把 $\mathbb E[x_t\mid x_t]=x_t$ 当循环论证 | 在「$x_t$ 两边都出现」上犹豫 | 这是「取出已知量」 |
| 没认出分母是系数的平方 | 约分像变魔术 | 先写 $1-\bar\alpha_t=(\sqrt{1-\bar\alpha_t})^2$ 再代 |
| 错拿习题 1 的路线（在 $(x_t,x_0)$ 上条件） | 那条路里 $\varepsilon_t$ 被完全决定，推出来的是另一个命题 | 必须在 $x_t$ 上条件——score 只是 $x_t$ 的函数 |

</details>

$$
\text{DDPM 的 } \varepsilon\text{-预测} = \frac{x_t - \sqrt{\bar\alpha_t}\,\mathbb{E}[x_0|x_t]}{\sqrt{1-\bar\alpha_t}}, \qquad
\text{FM 的 } v = \frac{x_t - \mathbb{E}[x_0|x_t]}{t}\ \text{（线性路径下）}.
$$

**统一读法：同一行代数，不同的分母**。核心只有一个：$\mathbb E[x_0\mid x_t]$（去噪器）。每套日程表都给出「缩放后干净猜想 vs 观测」的**残差** $x_t-a_t\,\mathbb E[x_0\mid x_t]$，三种参数化只是残差的归一化方式不同：

| 参数化 | 读数 | 分母含义 |
|---|---|---|
| score（§2.4） | 残差 $\div\sigma_t^2$ | 噪声**方差**——弹簧劲度式的读法（拉力按 $1/\sigma^2$ 定标） |
| $\varepsilon$-pred（§2.3） | 残差 $\div\sigma_t$ | 噪声**幅值**——还原「产生 $x_t$ 的那个骰子」 |
| $v$-pred（FM） | 残差的**时间归一化**（线性路径下 $\div t$） | 速度——「单位时间该走多少」（§4.2：$v=\frac{dx_t}{dt}$） |

所以**$\varepsilon$-预测、$x_0$-预测、$v$-预测（速度）是同一个去噪器 $\mathbb{E}[x_0\mid x_t]$ 的三种坐标**。这个事实让 CFG-Ctrl（VG 11）可以自由选择"在哪个量上施加控制"。

<details>
<summary><b>数字标本：同噪声水平下两套日程的逐项对照（高斯数据 $\mathcal N(2,0.6^2)$，$x_t=1$）</b></summary>

取 $\bar\alpha_t=0.5$（噪声份额 $\sqrt{1-\bar\alpha_t}=0.707$），对照 FM 取 $t=0.707$（同噪声水平），观测同一点 $x_t=1$：

| 量 | DDPM 日程 | FM 日程 |
|---|---|---|
| 边缘均值/方差 | $\sqrt{0.5}\cdot2\,/\;0.5\cdot0.36+0.5$ | $0.293\cdot2\,/\;0.293^2\cdot0.36+0.5$ |
| $\mathbb E[x_0\mid x_t]$ | $1.8449$ | $2.0823$ |
| score $=-\frac{x_t-a_t\mathbb E[x_0\mid x_t]}{\sigma_t^2}$ | $+0.6091$ | $-0.7802$ |
| $\varepsilon$-pred / $v$ | $-0.4307$ | $-1.5306$ |
| 统一式核验 $-\sigma_t^2\cdot\text{score}=x_t-a_t\mathbb E[x_0\mid x_t]$ | $-0.3046$ ✓ | $+0.3901$ ✓ |

两件事值得盯：① 最末行统一式在两套日程下都逐位成立——「同一行代数」的数值版；② **score 依赖日程表**：同一空间点、同一噪声水平，两套日程的 score 连符号都不同（$+0.61$ vs $-0.78$）——因为 score 是「**这条特定路径**的 $p_t$」的梯度（$\mathbb E[x_0\mid x_t]$ 也随日程变），路径不同则 $p_t$ 不同。换参数化不是「换个记号」，是换了一整条概率路径；同一路径内部的三种读数才只是坐标变换。

</details>

---

## 5. CFG：无分类器引导的数学

### 5.1 问题：条件生成怎么"听话"？

给扩散模型加上条件 $c$（文本、类别）最直接的做法是训 $\varepsilon_\theta(x_t, t, c)$。但实测发现条件模型的输出常常"不够极端"——模型学到了条件与数据的**平均关联**而非强绑定。（这多半不是训练失败：模型忠实学到了 $p(x\mid c)$，而这个分布本身就是概率性的宽分布——成因拆解、直观例子与数字标本见下方折叠块。）**引导**（guidance）的思想：在采样时人为放大条件的影响。

<details>
<summary><b>展开：「平均关联而非强绑定」——三个成因、一张对照表、一个数字标本</b></summary>

**先说结论**：这不是训练失败。条件模型收敛到 $\mathbb E[\varepsilon\mid x_t,c]$——与 §2.4 同一机制（MSE 回归的最优解是条件均值），而这个条件均值天然是「所有与 $c$ 相容的数据来历的平均意见」（§2.4「理解坑」的条件版）。**平均意见的宿命就是不极端**：模型忠实建模了 $p(x\mid c)$，而 $p(x\mid c)$ 本身就是概率性、多对多的关联分布。

三个成因，每一个都在把条件分布拉宽：

| 成因 | 机制 | 直观例子 |
|---|---|---|
| ① 条件粒度粗 | caption 只锁住少数属性，其余属性在数据里自由变化 | 「一只猫」对应所有品种/姿态/光照的猫——条件分布几乎和无条件一样宽 |
| ② 关联多对多且带噪 | 弱关联样本也带着这个条件进了训练集 | 「落日」的 caption 下混着阴天、伪彩、随手拍的糊图 |
| ③ 逐步后验平均 | 每一步的箭头都把强关联与弱关联来历按后验权重混合，方向永远偏向「典型」 | 「戴帽子的猫」：帽子样式五花八门，平均出来是一只帽子特征模糊的猫 |

落到提示词上，「不够极端」的具体样子：

| 提示词 $c$ | 数据里「符合 $c$」的样本谱 | 不加引导时的输出（典型/平均） | 用户想要的（分布右尾） |
|---|---|---|---|
| 「落日」 | 颜色从灰蒙到浓烈的连续谱 | 中等饱和度的落日 | 最浓烈的那张 |
| 「奔跑的狗」 | 慢跑到疾驰 | 中速 | 全速飞奔 |
| 「鲜红的汽车」 | 各种深浅的红 | 偏暗的「平均红」 | 饱和的鲜红 |

**数字标本（标签噪声版）**：一维双峰数据 $0.5\mathcal N(-2,0.6^2)+0.5\mathcal N(2,0.6^2)$，条件 $c=$「右」来自带噪标签——$x$ 在右模式时标对的概率 $0.8$，在左模式时误标成「右」的概率 $0.1$。贝叶斯：

$$
p(x\mid c{=}\text{右})\propto p(x)\,p(c{=}\text{右}\mid x)
\;\Longrightarrow\;
\text{右模式权重}=\frac{0.5\times0.8}{0.5\times0.8+0.5\times0.1}=\frac89\approx88.9\%.
$$

条件模型（不加引导、老老实实采样 $p(x\mid c)$）**正确地**学到这个分布：约 $11\%$ 的样本整个落在左模式，条件分布均值只到 $\tfrac{8}{9}\times2+\tfrac{1}{9}\times(-2)=1.56$（强绑定应是 $2.0$）。「不够极端」= 模型没错，数据里的关联本来就只有这么强。

**数学骨架（一行）**：$p(x\mid c)\propto p(x)\,p(c\mid x)$，两边取对数求梯度正是下一节 §5.2 要拆开的形式——条件 score = 无条件 score + $\nabla_x\log p(c\mid x)$。关联越弱（$p(c\mid x)$ 越平坦），修正项越小，条件模型越像无条件模型——「平均关联」的严格含义。

**引导为什么是解药**：它不再采样 $p(x\mid c)$，而是沿 $\nabla_x\log p(c\mid x)$ 方向外推——专门放大「$x$ 越像 $c$ 所描述」的那部分证据，样本自然往「最像 $c$」的那一小撮聚。实测的两幅面孔——不加引导时样本偏灰、欠饱和（本节说的「不够极端」），引导强度拉太大时过饱和（§5.3 的表会量化这一点：样本 std 从 $0.59$ 压到 $0.25$）——是同一枚硬币的两面：引导把分布沿「更像 $c$」的方向压缩，不压不够极端、压过头走向另一个极端（引导强度这个旋钮 $w$ 下一节 §5.2 正式定义）。

</details>

### 5.2 Classifier Guidance → Classifier-Free

**Classifier Guidance**（Dhariwal & Nichol 2021）：用贝叶斯公式把条件 score 拆开

$$
\nabla_x\log p(x\mid c) = \nabla_x\log p(x) + \nabla_x\log p(c\mid x),
$$

第二项用一个分类器网络近似，采样时加上去（可选权重）。缺点：要**额外训一个噪声鲁棒的分类器**，且对复杂条件（文本）不可行。

<details>
<summary><b>展开：「分类器网络」是什么 + 怎么形象区分 $p(x\mid c)$ 与 $p(c\mid x)$</b></summary>

**分类器网络**：就是普通的图像分类器——输入一张图，输出它属于每个类别的概率 $p_\phi(c\mid x)$（$\phi$ 是网络参数），结构上与 ImageNet 分类用的 CNN（ResNet 一类）没有本质区别。Classifier Guidance 用它时**不取类别输出，取概率对输入图像素的梯度** $\nabla_x\log p_\phi(c\mid x)$——这个梯度回答「把像素往哪个方向改一点，会更像类别 $c$」，一个现成的**编辑方向**，采样每步加到 score 上。与普通分类器的唯一差别在训练数据：

| | 普通分类任务 | Classifier Guidance |
|---|---|---|
| 输入 | 干净图 | 加噪中间态 $x_t$ |
| 用什么输出 | 类别概率（argmax） | 概率对**输入像素**的梯度 |
| 训练数据 | 干净图 + 标签 | **各噪声水平的加噪图** $(x_t,t,c)$ |

「噪声鲁棒」的含义：引导发生在带噪的 $x_t$ 上，只在干净图上训的分类器遇到重噪声图会输出接近均匀的概率——梯度 $\approx0$，引导失效，所以必须见过所有噪声水平。「对文本不可行」的原因：分类器输出层是 $N$ 个类别的 softmax，要求条件是**有限标签集**；文本开放词表、一句一个新条件，没法做成输出层。CFG 的办法是同一扩散模型做「有条件 / 空条件」两次前向，取差 $\nabla\log p(x\mid c)-\nabla\log p(x)$ 把这一项**逼出来**——不再需要任何分类器。

**$p(x\mid c)$ 与 $p(c\mid x)$：方向相反的一对**（贝叶斯 $p(x\mid c)\propto p(x)\cdot p(c\mid x)$ 连接两者）：

| | $p(x\mid c)$ | $p(c\mid x)$ |
|---|---|---|
| 谁在变 | 图 $x$ 在变、条件 $c$ 固定 | 标签 $c$ 在变、图 $x$ 固定 |
| 对谁归一 | 对**图**积分 $=1$：图空间上的分布 | 对**标签**求和 $=1$：作为图的函数**不归一** |
| 形象 | **画廊**：符合描述的所有图的散布（§5.1 说的「宽」就是它宽） | **裁判/地形图**：图像空间上一座「像不像 $c$」的山，每个位置一个分数 |
| 能否采样 | 能——生成模型采的就是它 | 不能——不是图的分布，只能顺着坡「爬」 |
| 扩散里的对应 | 条件模型 $\varepsilon_\theta(x_t,t,c)$ | 分类器（或 CFG 的两次前向之差） |
| 梯度的角色 | —— | $\nabla_x\log p(c\mid x)$ =「往哪改更像 $c$」，引导加的正是它 |

经典锚点（医学检验）：$p(\text{阳性}\mid\text{患病})$ 与 $p(\text{患病}\mid\text{阳性})$ 不是一回事——前者固定病看结果（灵敏度），后者拿结果倒推病因。这里同构：**固定条件生成图 vs 固定图判断条件**。最要紧的是第二行：$p(c\mid x)$ 作为 $x$ 的函数不归一（对图积分无意义），只是地形图——**正因为不归一，它对 $x$ 的梯度才携带纯粹的「更像 $c$」方向信息**，这正是引导项想要的。贝叶斯两边对 $x$ 取梯度时分母 $p(c)$ 是常数、梯度为零消失，剩下的正是正文那行分解：$\nabla_x\log p(x\mid c)=\nabla_x\log p(x)+\nabla_x\log p(c\mid x)$。

</details>

**CFG**（Ho & Salimans 2022）的天才简化：把"无条件"也当成一种条件（实现：训练时以 10–20% 概率随机**丢弃**条件，替换为空条件 $\varnothing$），然后

$$
\nabla_x\log p(x\mid c) = \nabla_x\log p(x) + \underbrace{\nabla_x\log\frac{p(x\mid c)}{p(x)}}_{\text{用两个网络输出之差近似}},
$$

**引导后的 score**（guidance scale $w$）：

$$
\boxed{\ s_w(x_t, t) = s_u(x_t,t) + w\,\big(s_c(x_t,t) - s_u(x_t,t)\big)\ }
$$

其中 $s_c = \nabla\log p(x_t\mid c)$（条件模型），$s_u = \nabla\log p(x_t)$（无条件模型）。实践中写作 $\varepsilon_{\text{cfg}} = \varepsilon_u + w(\varepsilon_c - \varepsilon_u)$。$w=1$ 退化为纯条件采样；$w=0$ 无条件；**$w>1$ 是"外推"**。（本节的教学版推导——Gaussian 路径下从 classifier guidance 的贝叶斯分解一步步推到 CFG 公式与训练算法——见 [MITS.6184 系列第 5 篇](../../MITS6184/MITS6184_5/)。）

### 5.3 CFG 到底在采样什么分布？（数值实验）

**逐点恒等式**：在固定的 $(x_t, t)$ 处，引导 score 是

$$
s_w = (1-w)\,s_u + w\,s_c = \nabla_x\log\left[p_u(x_t)^{1-w}\, p_c(x_t)^{w}\right],
$$

即引导 score 恰好是"重加权分布" $p_u^{1-w}p_c^w$ 的 score——**在每个时刻逐点成立**。但要小心：不同 $t$ 的重加权分布族 $\{p_{u,t}^{1-w}p_{c,t}^w\}$ 一般**不对应任何单一数据分布的扩散过程**，所以"$w>1$ 采样自 $p_u^{1-w}p_c^w$"这个常见说法是不精确的。

**数值实验**（二维 toy，双峰混合 $0.5\mathcal{N}((-2,0),0.6^2)+0.5\mathcal{N}((2,0),0.6^2)$；条件 $c$ = "右模式"，即 $p_c$ 为右模式单高斯；闭式 score，PF-ODE 采样 4000 样本）：

| guidance scale $w$ | 右模式占比 | 模式均值 ± 标准差 | 解读 |
|---|---|---|---|
| 0（无条件） | 49.0% | $+2.02\pm0.58$ / $-1.98\pm0.60$ | 精确复现数据分布 ✓ |
| 1（纯条件） | 99.9% | $+1.91\pm0.59$ | 精确复现条件分布 ✓ |
| 3 | 100.0% | $\mathbf{+2.45\pm0.35}$ | **推出数据支撑之外 + 方差收缩** |
| 7 | 100.0% | $\mathbf{+2.93\pm0.25}$ | 更强的外推 + 更狠的收缩 |

（外推量在步长细化 16 倍下稳定不变，是引导动力学的真实现象而非数值误差。）

**读懂数字**：$w=3$ 时样本均值 $2.45 > 2.0$——**样本被系统性推到条件分布之外**。这正是真实场景中 CFG 的两幅面孔：

- 好的一面：外推 = 更强的条件绑定、更"锐"的输出（图像饱和度高、视频动作更明显）——所以实践中大家爱用 $w=3\sim7$；
- 坏的一面：**过饱和（oversaturation）、多样性坍缩（std $0.6\to0.25$）、大 $w$ 下数值不稳定**。

**这就是 CFG-Ctrl（VG 11）的全部动机**：CFG 本质是一个固定增益的反馈控制器，$w$ 大了必然超调。他们用控制论（滑模控制 + Lyapunov 稳定性）重新设计这个控制器——那是系列的理论终章。

### 5.4 CFG 的实现细节（训练侧）

```text
训练：以概率 p_drop（如 15%）把条件 c 替换为空条件 ∅，其余同普通条件训练
采样：每步算两次前向（ε_c 与 ε_u），按 ε_u + w(ε_c − ε_u) 合并
```

代价：采样时**前向计算翻倍**（这也是后续很多工作的优化对象）。SD3/Flux 等 FM 模型中把 $\varepsilon$ 换成 velocity $v$，公式形式不变。

---

## 6. 视频扩散：把这套数学搬到时空立方体上

（本节给个最小图景，细节见本栏目 [CogVideoX](../CogVideoX/)、[WAN-Video](../WAN-Video/) 笔记。）

视频 = $F$ 帧 × $H\times W$ 像素的大张量。三个工程要点：

1. **3D 压缩**：像素空间太大，先过 3D VAE（时空联合下采样，如 $4\times8\times8$），在隐空间做扩散；
2. **架构**：DiT（Diffusion Transformer）——把隐 token 化，时间步 $t$、文本条件（CLIP/T5 嵌入）通过 cross-attention 或 adaLN 注入；
3. **时间一致性**：全注意力（CogVideoX 的做法——所有帧的 token 一起 attend，天然建模帧间依赖）或时空交替注意力。**时间注意力是视频扩散与图像扩散的唯一本质区别**，也是"视频先验为什么能帮 3D 重建"（ReconX 的核心洞察）的来源。

**ReconX / VideoScene / LangScene-X / Physics3D 的共同底牌**：预训练视频扩散模型内部存了"世界应该怎么运动/长什么样"的先验，这个先验可以当作 3D/4D 任务的免费监督信号——这个范式是 VG 05–08 四篇的主线。

---

## 7. 与系列后续的连接

| 后续论文 | 用到本篇的哪个结论 |
|---------|------------------|
| VG 05 ReconX | 视频扩散作为生成先验；3D 结构条件注入（§6 的时间注意力） |
| VG 06 VideoScene | 确定性 ODE 采样（§3）⟹ 蒸馏可行的前提；FM 直线路径（§4.2） |
| VG 07 LangScene-X | 多模态条件扩散（§5.4 的条件注入推广） |
| VG 08 Physics3D | SDS 型蒸馏（本篇 §2.4 score 视角是 SDS 的前置，完整推导在 VG 02） |
| VG 09 DreamReward-X | CFG 采样流程是被评估/被优化的对象 |
| VG 10 Video-T1 | **确定性采样的噪声→输出双射**（§3.3 推论 1）：在噪声空间做测试时搜索 |
| VG 11 CFG-Ctrl | **引导 score 的控制论重构**（§5.3 的外推现象是它的靶子） |

前置补齐：3D 表示（NeRF/3DGS）与 SDS 的完整推导在 **VG 02**。课程版前置：Flow 与 Diffusion 的统一视角、Flow Matching 回归目标、Score Matching 的教学推导见 [MITS.6184 系列](../../MITS6184/MITS6184_1/)（Generative Models 栏目，第 2–4 篇）——本篇是它的论文精读版对应物。

---

## 8. 习题

**题 1**：推导 §2.2 中 $q(x_{t-1}\mid x_t, x_0)$ 的均值与方差公式。
*提示*：贝叶斯 $q(x_{t-1}\mid x_t, x_0) \propto q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)$，两个高斯相乘配方。$\tilde\beta_t$ 的系数是标准的"信息融合"形式。
*解答*：见 [SOL 01：反向高斯的均值与方差](../../solutions/SOL_01_vg01_reverse_gaussian/)（配方 + 联合高斯条件化双路线 + Monte Carlo 验证，含 $\varepsilon$ 形式的精确恒等式）。

**题 2**：证明 DDPM 训练目标中的加权系数（§2.3 里被丢掉的那个）在 $\bar\alpha_T\to 0$、$\beta_t$ 小时近似与 $t$ 无关。
*提示*：$\sigma_t^2 = \tilde\beta_t \approx \beta_t$（当 $\bar\alpha_{t-1}\to$ 小），代入化简。

**题 3**：对线性插值路径 (4.1)，验证 $u_t(x) = \mathbb{E}[\varepsilon - x_0\mid x_t = x]$ 且数据为单高斯 $x_0\sim\mathcal{N}(\mu,\Sigma)$ 时有闭式解。
*提示*：$(x_0, \varepsilon)$ 给定 $x_t$ 的条件分布是联合高斯，$\mathbb{E}[\varepsilon\mid x_t]$ 用标准条件高斯公式。可得上式 $\text{Cov}$ 形如 $\frac{t}{t^2+(1-t)^2}$ 类的系数。

**题 4**（思考）：为什么 CFG 训练时随机丢条件（而不是单独训一个无条件模型）？两个网络共享参数带来了什么？
*提示*：共享参数 ⟹ $s_c$ 与 $s_u$ 高度相关 ⟹ 差值 $s_c - s_u$ 噪声小、且恰好近似 $\nabla\log p(c\mid x_t)$（分类器那一项）。单独训两个大模型的话差值会有系统性偏差。

**题 5**（实验）：把 §5.3 的 toy 代码思路复现：双峰混合 + 闭式条件/无条件 score + PF-ODE，扫 $w\in\{0,1,3,7\}$，验证"外推 + 方差收缩"现象。如果把两个模式的距离从 4 拉到 8，外推量会怎么变？
*提示*：距离越大，$s_c - s_u$ 在边界处的幅值越大，同样 $w$ 下外推更猛——这解释了为什么条件越"稀有"（数据中越少见），CFG 的副作用越强。

---

*下一篇：[VG 02：视频扩散与 3D 先验——架构、渲染与蒸馏](../VG_02_video_3d_priors/)*
