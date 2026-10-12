---
title: "泛函分析 08: 标准正交基与 Riesz 表示"
date: 2026-10-04
math: true
tags:
  - Notes
  - FunctionalAnalysis
categories:
  - FunctionalAnalysis
description: "无穷维空间有'基'吗？Hamel 基不可构造、Schauder 基不总存在，但 Hilbert 空间的标准正交基总存在且近乎完美：Fourier 系数 ⟨x,eₙ⟩ 显式可算、Bessel 不等式 ⟵ Parseval 等式只差'完全性'一步、可分 Hilbert 空间在等距同构意义下只有 Kⁿ 和 ℓ² 两类。Gram–Schmidt 正交化与 Zorn 引理两条存在性路线；Riesz 表示定理：Hilbert 空间的对偶就是自己（FA 06 (ℓ²)'=ℓ² 的机制揭晓）；伴随算子 T* = 共轭转置的抽象，‖T*T‖=‖T‖²；RKHS 一瞥"
---

# 泛函分析 08：标准正交基与 Riesz 表示

> 📖 本系列基于步尚全《泛函分析基础》整理，保留教材定理编号。本篇覆盖 §3.3（标准正交集与标准正交基）+ §3.4（Hilbert 空间上有界线性泛函的表示）。
> 前篇：[泛函分析 07：内积空间——当空间有了几何](../FA_07_inner_product_spaces/)。
> 编号已对照教材 PDF 核实（§3.3–3.4 全部正文 + 习题 3 题 14–29）。

---

## 1. 这一篇解决什么问题？

FA 04 留下的"基"的三角债：

| 基的类型 | 定义 | 状态 |
|---|---|---|
| Hamel 基 | 有限组合张成全空间 | 无穷维时**不可构造**（Zorn 引理产物，FA 04 §2.1） |
| Schauder 基 | 唯一收敛展开 $\sum a_ne_n$ | **不总存在**（Enflo 1973 反例，FA 04 §3.5） |
| ??? | ??? | Hilbert 空间里有更好的吗？ |

答案：**有，而且近乎完美**。Hilbert 空间的**标准正交基**（本篇主角）同时享有：

1. **总存在**（定理 3.3.6——连 Enflo 的阴影都照不进来，正交性太强反而好办事）；
2. **系数显式**：展开系数就是内积 $a_n=\langle x,e_n\rangle$——不用解方程、不用归纳，**量一下就得到**（对比 Schauder 基的系数要靠极限程序递归求解）；
3. **能量守恒**：$\|x\|^2=\sum|\langle x,e_n\rangle|^2$（Parseval 等式）——"时域能量 = 频域能量"，信号处理的 Fourier 定律即此；
4. **分类定理**（本篇最震撼的结果）：可分 Hilbert 空间在等距同构意义下**只有两类**——$\mathbb{K}^n$ 与 $\ell^2$。所有可分无穷维 Hilbert 空间都是 $\ell^2$ 换皮。

例 3.3.3 是这一切的原型：$C[0,2\pi]$ 上的三角函数系 $e_n(t)=\frac{1}{\sqrt{2\pi}}e^{int}$（$n\in\mathbb{Z}$）——**Fourier 级数**就是标准正交基展开，Fourier 系数就是 $\langle x,e_n\rangle$，Parseval 就是能量守恒。本篇把大学数学里"背公式"的 Fourier 分析变成"看图说话"的正交几何。

第二主题收 FA 06 的尾：对偶表里 $(\ell^2)'=\ell^2$ 的**机制**是什么？**Riesz 表示定理（定理 3.4.1）**：Hilbert 空间上每个有界线性泛函都长成 $f(x)=\langle x,y_0\rangle$ 的样子——**对偶就是自己**。这是"自对偶"的完整故事，也是核方法（RKHS）的数学根基。

**路线**：正交集与勾股 → Bessel 不等式 → 收敛定理（$l^2$ 系数 ⟺ 级数收敛）→ 标准正交基四等价 → Gram–Schmidt 与存在性 → Riesz 表示 → 伴随算子与酉算子。

---

## 2. 核心定义与直觉

### 2.1 正交集：互相测量为零的坐标架

**定义 3.3.1**　$M$ 中元素两两正交称**正交集**；再要求 $\|x\|=1$ 称**标准正交集**（orthonormal）；可数时称标准正交序列、有限时称标准正交组。

**例 3.3.1–3.3.3（本篇三件套）**：

| 空间 | 标准正交系 | 备注 |
|---|---|---|
| $\mathbb{K}^n$ | $e_1,\dots,e_n$（坐标单位向量） | 有限维原型 |
| $\ell^2$ | $e_1,e_2,\dots$ | 无穷维原型（FA 04 的 Schauder 基就是它） |
| $C[0,2\pi]$ | $e_n(t)=\frac{1}{\sqrt{2\pi}}e^{int}$（$n\in\mathbb{Z}$，复）；实版 $\{\frac{1}{\sqrt{2\pi}},\frac{\cos nt}{\sqrt\pi}\}$ | **三角函数系 = Fourier 分析**（正交性 = 积分 $\int e^{i(m-n)t}dt=0$，$m\ne n$） |

### 2.2 Fourier 系数与 Bessel 不等式：投影的量产

给定标准正交序列 $\{e_n\}$，每个 $x$ 产出**Fourier 系数** $\langle x,e_n\rangle$——FA 07 "投影系数万金油"的量产版。核心恒等式：**部分和是正交投影**。

$$
y_n=\sum_{i=1}^n\langle x,e_i\rangle e_i \;\in\; \operatorname{span}\{e_1,\dots,e_n\}, \qquad x-y_n\ \perp\ \operatorname{span}\{e_1,\dots,e_n\}.
$$

（验证：$\langle y_n,x-y_n\rangle=\sum\langle x,e_i\rangle\langle e_i,x\rangle-\sum|\langle x,e_i\rangle|^2=0$。）于是勾股定理 $\|x\|^2=\|y_n\|^2+\|x-y_n\|^2\ge\|y_n\|^2$ 给出**Bessel 不等式（式 3.11）**：

$$
\boxed{\ \sum_{i=1}^\infty|\langle x,e_i\rangle|^2\ \le\ \|x\|^2\ }
$$

读法：**投影能量永远不超过总能量**；Fourier 系数平方可和，且平方和有预算上限。Bessel 是"预算约束"，什么时候预算花满（取等号）？答案是完全性——本篇主戏。

### 2.3 标准正交基：无穷维的完美坐标系

**定义 3.3.2**　标准正交集 $M$ 满足 $\overline{\operatorname{span}(M)}=H$（即 $M$ 为完全集，FA 07 定理 3.2.6），称 $M$ 为 $H$ 的**标准正交基**。

**注 3.3.1（分类定理的零件）**　$\ell^2(M)=\{f:M\to\mathbb{K}:\ \sum_{s\in M}|f(s)|^2<\infty\}$。每个 Hilbert 空间 $H$ 配上标准正交基 $M$ 后，映射 $x\mapsto\{s\mapsto\langle x,s\rangle\}$ 是 $H\to\ell^2(M)$ 的**等距同构**——$H$ 的全部信息（范数、内积）被坐标 **无损** 打包。$M$ 有限 $n$ 个元素 ⟹ $\ell^2(M)=\mathbb{K}^n$；$M$ 可数 ⟹ $\ell^2(M)=\ell^2$。这给出书页 92 的分类结论：

> **可分 Hilbert 空间在等距同构意义下只有两类：$\mathbb{K}^n$（有限维）与 $\ell^2$（无穷维）。**

$L^2[0,2\pi]$、$\ell^2$、任何可分无穷维 Hilbert 空间——**全是同一个空间的不同皮肤**。这就是为什么 $\ell^2$ 在 FA 06 对偶表里自对偶、在 ML 里（NTK、GP、RKHS）反复出现的结构原因：**没有别的可分无穷维 Hilbert 空间**。

### 2.4 Riesz 表示：泛函 = 内积；伴随 = 共轭转置

FA 06 说对偶空间是"观测者的全体"。Hilbert 空间的观测者出奇地简单：

$$
f\in H' \;\Longleftrightarrow\; \exists!\,y_0\in H:\ f(x)=\langle x,y_0\rangle,\qquad \|f\|=\|y_0\|.
$$

**每个连续观测 = 与某个固定向量的内积**。直觉：$y_0$ 是泛函的"代表向量"——观测 $x$ 就是让 $x$ 与 $y_0$ 比夹角。二阶版本（定理 3.4.2）把双线性泛函 $h(x,y)$ 表示成 $\langle Tx,y\rangle$，其中 $T$ 是**有界算子**——由此定义**伴随算子（adjoint）**：

$$
\langle Tx,y\rangle_2=\langle x,T^*y\rangle_1, \qquad T^*\in B(H_2,H_1),\ \|T^*\|=\|T\|.
$$

有限维里 $T^*$ 就是**共轭转置矩阵** $A^{\dagger}=(\bar a_{ji})$（例 3.4.1），对角算子的伴随是逐项共轭（例 3.4.2）。FA 07 投影的"对称性"自此有名字：$P_M=P_M^*$（自伴）；酉算子（定理 3.4.4）满足 $TT^*=T^*T=I$——**保内积的同构 = 旋转/反射的抽象**。

---

## 3. 主要定理与证明

### 3.1 定理 3.3.1：勾股 + 无关

**定理 3.3.1**　标准正交集 $M$：(1) 任取 $M$ 中 $n$ 个不同元，$\big\|\sum_{i=1}^n a_ie_i\big\|^2=\sum_{i=1}^n|a_i|^2$（**勾股定理**）；(2) $M$ 线性无关。

**证明**　(1) 展开内积，交叉项 $\langle e_i,e_j\rangle=0$ 全灭：$\|\sum a_ie_i\|^2=\sum_{i,j}a_i\bar a_j\langle e_i,e_j\rangle=\sum_i|a_i|^2$。(2) 零组合的范数平方 $=\sum|a_i|^2=0$ ⟹ 全部系数为零。∎

（FA 07 辨析表"正交 ⟹ 无关"的正式出处。勾股定理是本篇一切收敛性论证的地基——它把**向量级数的收敛**翻译成**数项级数的收敛**。）

### 3.2 定理 3.3.2：$\ell^2$ 系数 ⟺ 级数收敛（Riesz–Fischer 型定理）

**定理 3.3.2**　$\{e_n\}$ 为 Hilbert 空间 $H$ 的标准正交序列，$\{a_i\}\subset\mathbb{K}$：

1. $\sum_{i\ge1}a_ie_i$ 在 $H$ 中收敛 $\iff$ $\sum_{i=1}^\infty|a_i|^2<\infty$（系数 $\ell^2$）；
2. 若收敛且 $x=\sum a_ie_i$，则 $a_i=\langle x,e_i\rangle$；
3. 任取 $x\in H$，**Fourier 级数** $\sum_{i\ge1}\langle x,e_i\rangle e_i$ 收敛。

**证明（完整——勾股定理三连击）**

(1) 部分和 $S_n=\sum_{i\le n}a_ie_i$，勾股给 $\|S_m-S_n\|^2=\sum_{i=n+1}^m|a_i|^2=s_m-s_n$（数项部分和）。**向量 Cauchy ⟺ 数列 Cauchy**——完备性接收 $\ell^2$ 数列的收敛。

(2) 内积连续（FA 07 定理 3.1.2）：$\langle x,e_i\rangle=\lim_n\langle\sum_{j\le n}a_je_j,e_i\rangle=a_i$。

(3) Bessel 不等式给 $\sum|\langle x,e_i\rangle|^2\le\|x\|^2<\infty$，由 (1) 收敛。∎

**两件免费赠品**：**重排不变**——正交级数绝对按范数收敛（$\sum\|a_ie_i\|^2=\sum|a_i|^2<\infty$，FA 05 引理 2.3.3 的绝对收敛版），重排不改变和（教材紧接着的注）。**展开系数唯一**——(2) 说明收敛展开的系数**必须**是 Fourier 系数，别无选择。

### 3.3 定理 3.3.3：每个向量只"看见"可数个基

**定理 3.3.3**　$M$ 标准正交集，则 $M_x=\{e\in M:\langle x,e\rangle\ne0\}$ **至多可数**。

**证明（完整——Bessel 分层计数）**　令 $M_{x,m}=\{e\in M:|\langle x,e\rangle|\ge\frac1m\}$。若 $e_1,\dots,e_k\in M_{x,m}$ 两两不同，Bessel 给

$$
\frac{k}{m^2}\le\sum_{j=1}^k|\langle x,e_j\rangle|^2\le\|x\|^2 \;\Longrightarrow\; k\le m^2\|x\|^2<\infty.
$$

每层有限，$M_x=\bigcup_{m=1}^\infty M_{x,m}$ 可数个有限集之并，至多可数。∎

（这条让"不可数基"也能写展开式：$x=\sum_{e\in M}\langle x,e\rangle e$ 中非零项至多可数个——记号 (3.12)/(3.13) 的合法性来源。**每个向量只与可数个坐标有来往**，无穷维的"不可数"只藏在基的势里。）

### 3.4 定理 3.3.4：标准正交基四等价（本篇主定理）

**定理 3.3.4**　$H$ Hilbert，$M$ 标准正交集。以下等价：

$$
(1)\ M\ \text{为标准正交基} \iff (2)\ x=\sum_{e\in M}\langle x,e\rangle e\ \ \forall x \iff (3)\ \langle x,y\rangle=\sum_{e\in M}\langle x,e\rangle\overline{\langle y,e\rangle} \iff (4)\ \|x\|^2=\sum_{e\in M}|\langle x,e\rangle|^2\ (\text{Parseval}).
$$

**证明（完整——环环相扣的四步）**

**(1)⟹(2)**：定理 3.3.2(3) 给 $y=\sum_i\langle x,e_i\rangle e_i$ 收敛。残差 $z=x-y$ 与**每个** $e\in M$ 正交（内积连续逐项验证：$\langle z,e_i\rangle=\langle x,e_i\rangle-\langle y,e_i\rangle=0$；$e\in M\setminus M_x$ 时 $\langle x,e\rangle=0$ 且 $\langle e_i,e\rangle=0$）。故 $z\in(\operatorname{span}M)^\perp=(\overline{\operatorname{span}M})^\perp=H^\perp=\{0\}$（FA 07 引理 3.2.1 + 完全集）——$x=y$。

**(2)⟹(3)**：展开式的内积 = 极限交换 + 勾股交叉项：

$$
\langle x,y\rangle=\lim_n\Big\langle\sum_{i\le n}\langle x,e_i\rangle e_i,\ \sum_{i\le n}\langle y,e_i\rangle e_i\Big\rangle=\lim_n\sum_{i\le n}\langle x,e_i\rangle\overline{\langle y,e_i\rangle}.
$$

**(3)⟹(4)**：取 $x=y$。

**(4)⟹(1)**：Parseval ⟹ $y\perp M$ 时 $\|y\|^2=\sum_{e\in M}|\langle y,e\rangle|^2=0$ ⟹ $(\operatorname{span}M)^\perp=\{0\}$ ⟹ 完全集（FA 07 定理 3.2.6 零检测器）。∎

**四张面孔一张表**：

| # | 内容 | 别名/读法 |
|---|---|---|
| (2) | $x$ 展开 | **Fourier 展开公式** |
| (3) | 内积 = 系数配对 | **Parseval 内积版**（坐标计算内积） |
| (4) | 范数平方 = 系数平方和 | **Parseval 等式 = 能量守恒** |
| ⟸ | Bessel 取等 | **完全性 = 预算花满** |

**品味**：Bessel（$\le$）对任何标准正交序列成立；**完全性恰好是取等的那一步**。"投影能量何时打满总能量"——答案是"投满了整个空间"。信号处理里"三角函数系完备 ⟹ Parseval 能量守恒成立"，数学内核就是这一行。

### 3.5 定理 3.3.5 / 3.3.6：Gram–Schmidt 与基的存在性

**定理 3.3.5（Gram–Schmidt 正交化）**　$\{x_1,x_2,\dots\}$ 线性无关，则存在标准正交序列 $\{e_n\}$ 使 $\operatorname{span}\{e_1,\dots,e_n\}=\operatorname{span}\{x_1,\dots,x_n\}$（**逐级保张成**）。

**证明（完整——"减掉投影再归一"的流水线）**　$e_1=x_1/\|x_1\|$。归纳：设 $e_1,\dots,e_n$ 已造好，令

$$
v_{n+1}=x_{n+1}-\sum_{i=1}^n\langle x_{n+1},e_i\rangle e_i, \qquad e_{n+1}=\frac{v_{n+1}}{\|v_{n+1}\|}.
$$

则 $\langle v_{n+1},e_i\rangle=\langle x_{n+1},e_i\rangle-\langle x_{n+1},e_i\rangle=0$（**减掉全部已有投影**），$v_{n+1}\ne0$（否则 $x_{n+1}$ 落入 $\operatorname{span}\{e_1..e_n\}=\operatorname{span}\{x_1..x_n\}$，与无关矛盾）。张成关系：$v_{n+1}$ 是 $x_{n+1}$ 与旧基的线性组合、$x_{n+1}$ 是 $v_{n+1}$ 与旧基的线性组合——双向包含。∎

**定理 3.3.6（存在性）**　$H\ne\{0\}$ Hilbert，$M_0$ 为任一标准正交集，则存在标准正交基 $M\supset M_0$。特别地 **$H$ 必有标准正交基**。

**证明（两条路线）**

**路线一（Zorn 引理，一般情形）**：$E=\{M\supseteq M_0:\ M\ \text{标准正交}\}$ 按包含关系半序。任何全序链之并仍是标准正交集（有限个元素总落在链中某个成员里——链的全序性保证），故为链的上界。Zorn 引理给极大元 $M_1$。若 $M_1$ 不完全，$(\overline{\operatorname{span}M_1})^\perp\ne\{0\}$（FA 07 定理 3.2.3），挑一个单位向量 $x_0$ 加入——$M_1\cup\{x_0\}$ 更大，矛盾。

**路线二（可分情形，免 Zorn）**：$H$ 可分 ⟹ 有可数稠密集 $\{y_n\}$，提取线性无关子列 $\{x_n\}$ 保持张成；Gram–Schmidt 产出 $\{e_n\}$，其张成的闭包仍稠密——标准正交基。∎

**品味**：Enflo 反例（可分 Banach 空间可以没有 Schauder 基）与这里形成尖锐对照——**加一层正交结构，存在性从"可以失败"变成"永远成功"**，且可分时证明完全是构造性的（Gram–Schmidt 一遍走完）。教材特意给出免 Zorn 的路线：**可分 Hilbert 空间的基可以真的算出来**（习题 3 题 21 对 $1,t,t^2$ 走一遍得到 Legendre 多项式）。

### 3.6 定理 3.4.1：Riesz 表示定理（对偶 = 自己）

**定理 3.4.1（F. Riesz）**　$H$ Hilbert，$f\in H'$。则**存在唯一** $y_0\in H$ 使 $f(x)=\langle x,y_0\rangle$（$\forall x$），且 $\|f\|=\|y_0\|$。

**证明（完整——两个方向先立好，再一个漂亮的"碰消"构造）**

**"内积泛函"方向**（正文先行给出）：固定 $y_0$，$f_{y_0}(x)=\langle x,y_0\rangle$。Schwarz 给 $\|f_{y_0}\|\le\|y_0\|$；试验向量 $x=y_0$ 给 $|f_{y_0}(y_0)|=\|y_0\|^2$，即 $\|f_{y_0}\|\ge\|y_0\|$——**取等**。（FA 06 算子范数两步法的泛函版。）

**"泛函 ⟹ 内积"方向**：$f=0$ 取 $y_0=0$；设 $f\ne0$。$N(f)$ 是**闭**真子空间（FA 06 推论 2.4.1 + $f\ne0$），故 $N(f)^\perp\ne\{0\}$（FA 07 正交分解）。固定单位化前的 $z_0\in N(f)^\perp$，$z_0\ne0$。**关键构造**：任取 $x$，令

$$
v=f(x)z_0-f(z_0)\,x \;\in\; H \qquad\Rightarrow\qquad f(v)=f(x)f(z_0)-f(z_0)f(x)=0,
$$

故 $v\in N(f)$，而 $v$ 又是 $z_0$（$\in N(f)^\perp$）与 $x$ 的组合。用 $\langle v,z_0\rangle=0$（$v\in N(f)\perp z_0$）：

$$
f(x)\langle z_0,z_0\rangle-f(z_0)\langle x,z_0\rangle=0 \;\Longrightarrow\; f(x)=\Big\langle x,\ \frac{\overline{f(z_0)}\,z_0}{\|z_0\|^2}\Big\rangle \;\Longrightarrow\; y_0=\frac{\overline{f(z_0)}\,z_0}{\|z_0\|^2}.
$$

**唯一性**：$\langle x,y_0-y_1\rangle\equiv0$ 取 $x=y_0-y_1$ 得 $\|y_0-y_1\|^2=0$。∎

**品味**：证明的漂亮之处在 $v=f(x)z_0-f(z_0)x$ 这个**碰消组合**——把任意 $x$ 拉进 $N(f)$（$f$ 值归零），再用 $z_0$ 的垂直性交换出 $x$ 的信息。几何上 $y_0$ 就是**等值超平面 $\{f=c\}$ 的法向量**（FA 04 §2.1 超平面伏笔的兑现）。

**完备性不可去**（教材明示的反例）：$X$ 不完备内积空间，$H$ 为其完备化，取 $y_0\in H\setminus X$，则 $f(x)=\langle x,y_0\rangle$（$x\in X$）是 $X$ 上合法有界泛函，但**在 $X$ 内无表示向量**（若有 $y_1\in X$，则 $y_0-y_1\perp X$ 而 $X$ 稠密 ⟹ $y_0=y_1$，矛盾）。**Riesz 表示是完备性的赠品**。

### 3.7 定理 3.4.2 + 引理 3.4.1：双线性表示与 $\|h\|=\|T\|$

**引理 3.4.1**　$\|x_0\|=\max_{x\ne0}\frac{|\langle x_0,x\rangle|}{\|x\|}$——**max 真的取到**（在 $x=x_0$ 处）。

（对比 FA 06 习题 2 题 22：$C[-1,1]$ 上 $\|f\|=2$ 无人取到——**内积空间的"够得着"**：表示向量自己就是试验向量。一般赋范空间的 sup 是"看得见摸不着"，Hilbert 空间送你一个够得着的。）

**定理 3.4.2（Riesz，双线性版）**　$H_1,H_2$ Hilbert，$h$ 有界共轭双线性泛函（$|h(x,y)|\le C\|x\|\|y\|$，定义 3.4.1）。则**存在唯一** $T\in B(H_1,H_2)$ 使 $h(x,y)=\langle Tx,y\rangle_2$，且 $\|T\|=\|h\|$。

**证明思路（三次 Riesz 表示接力）**：固定 $x$，$y\mapsto\overline{h(x,y)}$ 是 $H_2$ 上有界线性泛函（注意共轭线性翻成线性），定理 3.4.1 给唯一向量 $Tx$ 使 $h(x,y)=\langle Tx,y\rangle$；线性性由"对一切 $y$ 内积相等 ⟹ 向量相等"传递；有界性与 $\|T\|=\|h\|$ 由引理 3.4.1（$\|Tx\|=\sup_y\frac{|\langle Tx,y\rangle|}{\|y\|}=\sup_y\frac{|h(x,y)|}{\|y\|}\le\|h\|\|x\|$，反向取 $y=Tx$）。∎

### 3.8 伴随算子：定义、例子、六条性质、酉算子

**定义**　$T\in B(H_1,H_2)$，把 $h(y,x)=\langle y,Tx\rangle_2$（$H_2\times H_1$ 上的有界共轭双线性泛函）喂给定理 3.4.2，得到唯一的 $T^*\in B(H_2,H_1)$：

$$
\boxed{\ \langle Tx,y\rangle_2=\langle x,T^*y\rangle_1\ },\qquad \|T^*\|=\|T\|.
$$

**两个落地例子**：

| 例子 | $T$ | $T^*$ |
|---|---|---|
| 例 3.4.1（矩阵） | $A=(a_{ij})_{n\times m}$：$(Tx)_i=\sum_ja_{ij}x_j$ | 共轭转置 $(\bar a_{ji})_{m\times n}$——逐分量验证 $\langle Tx,y\rangle=\sum_jx_j\overline{\sum_i\bar a_{ij}y_i}$ |
| 例 3.4.2（对角） | $(Tx)_i=a_ix_i$（$a\in\ell^\infty$） | $(T^*y)_i=\bar a_iy_i$（逐项共轭） |

**定理 3.4.3（伴随六件套，证明全部是"对一切 $y$ 内积相等"的一句话论证）**：

| # | 性质 | 备注 |
|---|---|---|
| (1) | $(S+T)^*=S^*+T^*$ | 线性 |
| (2) | $(\alpha T)^*=\bar\alpha T^*$ | **共轭**出号 |
| (3) | $(T^*)^*=T$ | 对合 |
| (4) | $\|T^*T\|=\|TT^*\|=\|T\|^2$ | 证明见下 |
| (5) | $T^*T=0\iff T=0$ | (4) 的推论 |
| (6) | $(PS)^*=S^*P^*$ | **反序**（矩阵转置的老朋友） |

**(4) 的证明（一行妙招）**：$\|Tx\|^2=\langle Tx,Tx\rangle=\langle x,T^*Tx\rangle\le\|x\|\,\|T^*Tx\|\le\|T^*T\|\|x\|^2$ ⟹ $\|T\|^2\le\|T^*T\|$；反向 $\|T^*T\|\le\|T^*\|\|T\|=\|T\|^2$。∎

**配套**：$I^*=I$；$T$ 可逆且 $T^{-1}$ 有界时 $(T^*)^{-1}=(T^{-1})^*$（两边取伴随 + 唯一性）。

**定理 3.4.4（酉算子）**　$T\in B(H_1,H_2)$ 一一映射。$T$ 为（内积意义）等距同构 $\iff$ $TT^*=I_{H_2}$ 且 $T^*T=I_{H_1}$。满足者称**酉算子（unitary）**——**保内积 = 保长度 + 保角度**的算子，有限维里就是正交/酉矩阵。FA 14 谱定理的舞台构件（自伴 $T=T^*$、酉 $U$、正规 $T^*T=TT^*$ 三分类的起点，习题 3 题 28–29 预热）。

---

## 4. 例子与反例速查

| 命题 | 真假 | 例子/反例 |
|------|------|----------|
| Hilbert 空间必有标准正交基 | ✅ | 定理 3.3.6（可分时免 Zorn，Gram–Schmidt 构造） |
| 可分 Banach 空间必有 Schauder 基 | ❌ | Enflo 1973（FA 04 §3.5）——正交性的存在论红利 |
| 标准正交序列必为基 | ❌ | $\ell^2$ 中 $\{e_{2n}\}$（偶数位）：$e_1\in\{e_{2n}\}^\perp$，不完全——Bessel 严格不等 |
| $C[0,2\pi]$ 的三角函数系完全 | ✅ | 其完备化 $L^2[0,2\pi]$ 中 Parseval 成立（Fourier 级数理论核心） |
| 标准正交基下 Fourier 系数唯一 | ✅ | 定理 3.3.2(2)：收敛展开的系数别无选择 |
| 正交级数重排不变 | ✅ | 依范数绝对收敛（$\sum\|a_ne_n\|^2=\sum\|a_n\|^2$） |
| Parseval 对任意标准正交序列成立 | ❌ | 完全性恰好是取等条件（定理 3.3.4 (4)⟹(1)） |
| Riesz 表示在完备内积空间成立 | ✅ | 定理 3.4.1 |
| Riesz 表示在不完备内积空间成立 | ❌ | $y_0\in H\setminus X$：$f=\langle\cdot,y_0\rangle$ 无 $X$ 内表示（教材反例） |
| $\sup\frac{\|f\|}{\text{试验}}$ 恒可取到 | ❌→✅ | 一般赋范：题 22 不可达；内积空间：引理 3.4.1 **max 取到**（$x=y_0$ 自己） |
| $\|T^*\|=\|T\|$、$\|T^*T\|=\|T\|^2$ | ✅ | 定理 3.4.3(4)；矩阵奇异值 $\sigma_1=\sqrt{\lambda_{\max}(T^*T)}$ 的算子版 |
| 可分无穷维 Hilbert 空间互不同构 | ❌ | **全部等距同构于 $\ell^2$**（注 3.3.1 + 定理 3.3.6 路线二） |

---

## 5. 易混淆辨析

| 概念对 | 辨析 |
|--------|------|
| **Hamel vs Schauder vs 标准正交基** | Hamel：有限组合、代数、不可构造；Schauder：唯一收敛展开、分析、**可以不存在**（Enflo）；标准正交基：正交 + 完全、系数**显式**（$\langle x,e\rangle$）、**总存在**、附送 Parseval。层级递强、条件递苛刻（要内积），回报递丰厚。 |
| **Bessel vs Parseval** | Bessel $\sum|\langle x,e\rangle|^2\le\|x\|^2$ 对任何标准正交系成立（预算约束）；Parseval 取等 ⟺ 完全（预算花满）。**从 $\le$ 到 $=$ 的一步之差就是"基"与"不是基"的差别**——习题 3 题 20 的扰动稳定性（$\sum\|e_n-f_n\|^2<1$ 保完全性）说明这步之差的拓扑厚度。 |
| **Fourier 系数存在 vs 级数收敛 vs 级数等于 $x$** | 系数永远存在且有 Bessel 预算（定理 3.3.2(3) 收敛免费）；但**部分和的极限未必是 $x$**——完全性负责"投满"（定理 3.3.4(2)）。三档：算得出系数 < 级数收敛 < 级数还原原函数。经典 Fourier 分析的难处（处处收敛、Carleson 定理等）全在第三档的**逐点**版本；本篇的收敛是**依范数**（$L^2$），容易得多。 |
| **Riesz 表示（Hilbert）vs Hahn–Banach（一般赋范）** | Hilbert：对偶**就是自己**，表示向量显式、范数相等；一般赋范空间：对偶只能**延拓**保证非平凡（FA 09），表示要逐空间研究（FA 06 对偶表）。**完备 + 内积买到了"完美对偶"**，一般空间只能买"够用的对偶"。 |
| **$T^*$ vs 矩阵转置** | 有限维内积下伴随 = 共轭转置 $A^\dagger$；无穷维 $T^*$ 由内积**定义**（$\langle Tx,y\rangle=\langle x,T^*y\rangle$），存在性靠 Riesz 表示。$(\alpha T)^*=\bar\alpha T^*$（共轭出号）与 $(PS)^*=S^*P^*$（反序）是初学两大坑。 |
| **酉 vs 等距 vs 可逆** | 可逆：代数（双射 + 有界逆）；等距：$\|Tx\|=\|x\|$（保长度）；酉：$T^*T=TT^*=I$（保内积 = 保长度**和**角度）。线性等距未必满射（右移算子，题 27）；酉 = 等距 + 双射。 |

---

## 6. ML 连接

### 6.1 Fourier / 小波：正交基展开的工业化

信号与深度学习里"频域分析"的全部合法性来自本篇：$\{e^{int}/\sqrt{2\pi}\}$ 是 $L^2[0,2\pi]$ 的标准正交基（例 3.3.3），于是

- **变换 = 取系数**：$c_n=\langle x,e_n\rangle$（变换是"量夹角"，不是解方程）；
- **Parseval = 能量守恒**：$\|x\|_{L^2}^2=\sum|c_n|^2$——滤波（丢高频系数）丢掉多少能量，误差平方就是多少，**预算制**；
- **spectral bias**：神经网络学低频先、高频后——在正交基视角下即"先学能量大的系数（Parseval 排序），难学能量小的"；小波基的局部性（时频双赢）是**换一组正交基**的工程学——基不变的是定理（展开/Parseval 全套随基走），变的是系数的**可解释性**。

### 6.2 RKHS 一瞥：核方法 = 隐式的 $\ell^2$

正定核 $k(x,y)$ 定义函数空间 $\mathcal H_k$（RKHS），其中 $k(x,\cdot)$ 自己住在 $\mathcal H_k$ 里且满足**再生性** $f(x)=\langle f,k(x,\cdot)\rangle$——**"求值泛函"有表示向量**，这正是 Riesz 表示定理的姿态（RKHS 的定义本质就是强制求值泛函连续 + Riesz 向量显式）。由此：

- **representer theorem**：正则化经验风险的最优解落在 $\operatorname{span}\{k(x_i,\cdot)\}$——有限个"基函数"的组合，无穷维优化塌缩成有限维；
- **kernel PCA / GP**：在 RKHS 里做投影/贝叶斯推断，几何全是 FA 07 的投影 + 本篇的基；
- **$\ell^2$ 分类定理的 ML 回响**：任何可分无穷维 Hilbert 空间（包括一切 RKHS）都是 $\ell^2$——**核方法的"高维特征空间"在 Hilbert 结构层面没有多样性**，多样性全在"嵌入映射 $\phi$ 怎么摆"。

### 6.3 伴随算子的 ML 面

- **反向传播 = 链式法则的伴随**：复合算子的伴随反序 $(PS)^*=S^*P^*$——**backprop 沿计算图反向传播正是伴随算子的运作方式**（$J^\top$ 乘在梯度上）；$\|T^*T\|=\|T\|^2$ 是谱范数（最大奇异值平方）的身份证——FA 06 §6.1 谱归一化的正式数学对象（$\|W\|_2^2=\lambda_{\max}(W^*W)$）；
- **自伴 = 对称矩阵**：$T=T^*$ 的算子有实谱、谱定理伺候（FA 14：协方差算子/PCA/Kernel PCA）；注意力里的 $QK^\top$ 虽非自伴，其谱结构（低秩 + softmax 非线性）是表达能力分析的常规入口；
- **酉初始化**：权重矩阵初始化为酉/正交（$\|Wx\|=\|x\|$）防止训练初期激活/梯度的范数漂移——定理 3.4.4 的工程化，"旋转不拉伸"。

---

## 7. 逻辑推导概念图

自上而下读这张图：顶部是出发点（正交规范性、完备性），箭头是推理方向，每一步只用上游已到手的结论。读者可据此核对"哪个定理用了哪条假设"——例如收敛定理专门花完备性、Parseval 取等专门花完全性，各有各的供货商。

```text
标准正交性（定义 3.3.1：两两正交 + 模为 1）        Hilbert 完备性
│                                                    │
├─► 定理 3.3.1 勾股：||Σ a_i e_i||^2 = Σ |a_i|^2 ◄───┤
│     │（向量级数收敛 ⟺ 数项级数收敛）               │
│     ▼                                               │
│   定理 3.3.2：Σ a_i e_i 收敛 ⟺ {a_i} ∈ ℓ^2（完备性接收极限）
│     └─ Fourier 级数 Σ <x,e_i> e_i 自动收敛（Bessel 喂 ℓ^2 预算）
│
├─► Bessel 不等式（式 3.11：部分和是正交投影，投影能量 ≤ 总能量）
│     ▼
│   定理 3.3.4 四等价：基 ⟺ 人人可展开 ⟺ Parseval 内积版 ⟺ Parseval 等式
│     │（完全性 = Bessel 取等；零检测器 M⊥ = {0} 借自 FA 07 定理 3.2.6）
│     ├─► 定理 3.3.5 Gram–Schmidt（逐级保张成）
│     │      └─► 定理 3.3.6 基总存在（Zorn；可分时免 Zorn 纯构造）
│     │             └─► 注 3.3.1 分类：可分 Hilbert 只有 K^n 与 ℓ^2
│     └─ 反例：ℓ^2 中 {e_2n} 不完全——Bessel 严格不等，Parseval 失效
│
└─► 定理 3.4.1 Riesz 表示（N(f) 闭 + FA 07 正交分解，碰消构造）
      ├─► f(x) = <x,y0>，||f|| = ||y0||：对偶就是自己
      ├─► 定理 3.4.2 + 引理 3.4.1（max 取到）──► 伴随 T*（||T*T|| = ||T||^2）
      │                                            └─► 定理 3.4.4 酉算子
      └─ 反例：X 不完备时 f = <·,y0>（y0 ∈ H\X）在 X 内无表示向量
```

| 出发点 | 买到什么 | 撤掉它失去什么 |
|---|---|---|
| 标准正交性（定义 3.3.1） | 勾股定理（定理 3.3.1）；系数显式 $a_n=\langle x,e_n\rangle$，量一下就得到 | Schauder 基的系数要靠极限程序递归求解，且可分 Banach 空间可以根本没有基（Enflo 反例，FA 04） |
| 完全性（$\overline{\operatorname{span}(M)}=H$） | 定理 3.3.4 全套：展开公式 + Parseval 内积版 + 能量守恒 | $\ell^2$ 中 $\{e_{2n}\}$：$e_1$ 与全体垂直——Bessel 依旧成立但严格不等，系数照算、能量守恒失效 |
| Hilbert 完备性 | 定理 3.3.2 收敛定理（完备性接收 $\ell^2$ 数列的极限）；Riesz 表示（定理 3.4.1） | 不完备内积空间 $X$：$f=\langle\cdot,y_0\rangle$（$y_0\in H\setminus X$）是合法有界泛函却在 $X$ 内无表示向量（教材反例） |
| Riesz 表示（定理 3.4.1） | 对偶就是自己；引理 3.4.1 的 max 取到；伴随算子存在且 $\|T^*\|=\|T\|$（定理 3.4.3） | 一般赋范空间只能退回 Hahn–Banach 延拓（FA 09）：对偶仅保证"够用"，表示向量与范数等式全部失去 |

---

## 8. 习题精选（教材习题 3 选摘）

**题 21**：$C[-1,1]$（内积 $\int_{-1}^1xy$）中把 $x_0=1,x_1=t,x_2=t^2$ Gram–Schmidt 正交化。
*提示*：$e_0=\frac{1}{\sqrt2}$（$\|1\|^2=\int_{-1}^1=2$）。$v_1=t-\langle t,\frac1{\sqrt2}\rangle\frac1{\sqrt2}=t-0=t$（奇偶正交免费送一步），$e_1=\sqrt{\frac32}\,t$（$\|t\|^2=\frac23$）。$v_2=t^2-\langle t^2,e_0\rangle e_0-0=t^2-\frac13$（$\langle t^2,1\rangle\cdot\frac12=\frac13$），$e_2=\sqrt{\frac58}\,(3t^2-1)$。**这就是 Legendre 多项式**——正交多项式全家（Chebyshev/Hermite/Jacobi）都是"权函数 + Gram–Schmidt"的产物，FA 12 逼近论的主角提前登场。

**题 18**：求 $\min_{a,b,c}\int_{-1}^1|t^3-a-bt-ct^2|^2dt$。
*提示*：$\min\|t^3-P\|$，$P=\operatorname{span}\{1,t,t^2\}$——**正交投影**（FA 07 定理 3.2.2），残差 $t^3-P^*t^3\perp P$。用奇偶分解免算一半：$t^3$ 是奇函数，$\{1,t^2\}$ 全是偶函数（内积自动为零），所以最优解必为 $P^*=bt$，由残差正交 $\int_{-1}^1(t^3-bt)\,t\,dt=0$，即 $\frac25-\frac{2b}3=0$，$b=\frac35$。最小值 $=\|t^3-\frac35t\|^2=\int(t^3-\frac35t)^2=\frac27-2\cdot\frac35\cdot\frac25+\frac9{25}\cdot\frac23=\frac{8}{175}$。**最小二乘 = 投影，正规方程 = 残差正交**（FA 07 §6.1 的手算版）；奇偶正交免费砍掉两个待定系数，是题 21 Gram–Schmidt 里"$v_1=t-0$"的同一份红利。

**题 27**：$\ell^2$ 上右移算子 $T(x_1,x_2,\dots)=(0,x_1,x_2,\dots)$。求 $T^*$、$D(T^*)$、$R(T^*)$、$\|T^*\|$。
*提示*：$\langle Tx,y\rangle=\sum_{i\ge2}x_{i-1}\bar y_i=\sum_{j\ge1}x_j\bar y_{j+1}=\langle x,(y_2,y_3,\dots)\rangle$——**$T^*$ 是左移** $T^*(y)=(y_2,y_3,\dots)$，定义在全 $\ell^2$。$\|T^*\|=1$（$\|T^*e_1\|=0$ 但 $\|T^*e_2\|=1$；上界 $\|T^*y\|\le\|y\|$）。**等距但不酉**：$T$ 保范数却不满射（第一坐标永远空着）——"酉 = 等距 + 双射"辨析行（§5）的活标本；$T^*T=I\ne TT^*$——正规性的反例一角。

**题 20**：$\{e_n\},\{f_n\}$ 均标准正交，$\sum_{n\ge1}\|e_n-f_n\|^2<1$，$\{f_n\}$ 为基。求证 $\{e_n\}$ 也是基。
*提示*：反设有 $x\ne0$ 与全体 $e_n$ 正交。由 $x\perp e_i$ 得 $\langle x,f_i\rangle=\langle x,f_i-e_i\rangle$，Schwarz 逐项控制：
$$
\sum_i|\langle x,f_i\rangle|^2\le\sum_i\|x\|^2\,\|f_i-e_i\|^2=\|x\|^2\sum_i\|e_i-f_i\|^2<\|x\|^2.
$$
但 $\{f_n\}$ 是基，Parseval（定理 3.3.4(4)）给 $\sum_i|\langle x,f_i\rangle|^2=\|x\|^2$——矛盾。**基是稳定的**：平方可和的小扰动不丢完全性——数值实现 Gram–Schmidt 的舍入误差不至于毁掉基的理论保障。

**题 23**：$X$ 内积空间，$z\mapsto f_z$（$f_z(x)=\langle x,z\rangle$）若为 $X\to X'$ 的**满射**，则 $X$ 完备。
*提示*：Riesz 表示定理的逆命题。$X'$ 总 Banach（FA 06 定理 2.4.5）；映射 $z\mapsto f_z$ 是**等距**（$\|f_z\|=\|z\|$，定理 3.4.1 的"内积泛函"方向）。若它还是满射，则 $X$ 等距同构于 Banach 空间 $X'$——**完备性被对偶倒灌回来**。"对偶够好 ⟹ 空间本身够好"的奇观，一般赋范空间绝无此事（对偶可以非平凡而空间千疮百孔）。

---

*下一篇：[泛函分析 09：Hahn-Banach 定理——泛函的延拓](../FA_09_hahn_banach_extension/)*
