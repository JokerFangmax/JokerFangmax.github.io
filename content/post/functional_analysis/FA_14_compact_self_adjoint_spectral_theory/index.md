---
title: "泛函分析 14: 紧算子与自伴算子的谱论"
date: 2026-10-10
math: true
tags:
  - Notes
  - FunctionalAnalysis
categories:
  - FunctionalAnalysis
description: "系列收官篇：紧算子——'无穷维里最像矩阵的算子'。Riesz–Schauder 理论：非零谱全是特征值、特征空间有限维、谱至多可数且只以 0 为聚点（证明引擎是 FA 05 Riesz 引理的全额兑现——每次都靠'紧性拒收两两距离 ≥θ 的序列'）；Fredholm 二择一：积分方程 (λ−A)x=y 可解 ⟺ y 与共轭齐次方程的解正交——有限维秩定理的无穷维翻版；σ(A)=σ(A*) 与 dim N(λ−A)=dim N(λ−A*)（有限秩扰动 F 的精妙构造）。自伴算子：⟨Ax,x⟩∈R、谱落在实轴、剩余谱空、数值半径=范数=谱半径（FA 13 的 r(A) vs ‖A‖ 辨析在此结账）、m_A 与 M_A 都是谱点；谱定理（定理 5.3.7）：A=Σλ_n⟨x,e_n⟩e_n——自伴紧算子必可对角化，PCA / Karhunen–Loève 展开 / Mercer 定理 / kernel PCA 的数学本体，低秩近似误差 ‖A−Σλ_iP_i‖=|λ_{n+1}| 是截断 SVD 的无穷维版"
---

# 泛函分析 14：紧算子与自伴算子的谱论

> 📖 本系列基于步尚全《泛函分析基础》整理，保留教材定理编号。本篇覆盖 §5.2（紧算子的谱论）+ §5.3（自伴算子的谱论）——**系列收官篇**。
> 前篇：[泛函分析 13：谱论入门——算子的"特征值"](../FA_13_spectral_theory_introduction/)。
> 编号已对照教材 PDF 核实（§5.2、§5.3 全部正文 + 习题 5 题 14–24）。

---

## 1. 这一篇解决什么问题？

FA 13 的三个样本算子留下了一个没有言明的分裂：对角算子（$\alpha_n\searrow0$）的谱是干净的 $\{\alpha_n\}\cup\{0\}$——全是特征值加一个聚点；而乘法算子的谱是整条线段 $[0,1]$——一个特征值都没有。**同是有界算子，谱的"形状"何以一个离散一个连续？**

本篇给出完整答案，而且是一个漂亮的两因子分解：

> **紧性管谱的形状（离散化），自伴性管谱的位置（实轴化）。**

| 算子 | 紧？ | 自伴？ | 谱的形状 | 能对角化？ |
|---|---|---|---|---|
| 对角算子 $\alpha_n\searrow0$ | ✅ | ✅（实 $\alpha_n$） | 离散点列 + 聚点 0 | ✅（标准正交特征向量） |
| 乘法算子 $(Ax)(t)=tx$ | ❌ | ✅ | 连续线段 $[0,1]$ | ❌（无特征向量） |
| 对角算子 $\alpha_n=\mathrm{i}/n$ | ✅ | ❌ | 离散但落在虚轴 | ✅（但不在实轴） |
| 一般有界算子 | ❌ | ❌ | 任意紧集 | 不一定 |

两个条件叠加——**自伴紧算子**——得到全系列的最强结构定理（定理 5.3.7 谱定理）：

$$
A\ \text{自伴紧}\quad\Longrightarrow\quad Ax=\sum_{n=1}^{\infty}\lambda_n\langle x,e_n\rangle e_n,\qquad Ae_n=\lambda_ne_n,\ \lambda_n\in\mathbb{R},\ \lambda_n\to0 .
$$

**无穷维算子被完全还原成"带正交基的对角矩阵"**——线性代数的谱理论在无穷维的最高复刻。这不是装饰性定理：协方差算子、积分算子（对称核）、Hessian、Gram 算子全是自伴紧算子——**PCA、Karhunen–Loève 展开、Mercer 定理、kernel PCA 是它的直接化身**（§6 展开）。

历史线（教材开篇原话）：紧算子理论就是从积分方程 $(T-\lambda)x=y$（$T$ 连续核）里长出来的——**分析中遇到的很多算子都是紧算子**。本篇两大定理的分工：

- **Riesz–Schauder 理论**（§5.2）：紧算子的谱长什么样（离散化）+ **Fredholm 二择一**（积分方程可解性的完整判据）；
- **自伴谱论**（§5.3）：谱落在哪里（实轴）+ 端点 $m_A,M_A$ 是谱点 + 谱定理（对角化）。

而证明的引擎，是全系列埋线最长的一条：**FA 05 的 Riesz 引理**（$\varepsilon<1$ 折扣）——本篇两次核心使用（$\frac{|\lambda|}{2}$ 与 $\frac{t}{2}$ 矛盾），每一次都是同一句话：**紧算子的像不容忍两两距离 $\ge\theta$ 的序列**。

---

## 2. 核心定义与直觉

### 2.1 紧算子：把单位球压成"几乎有限维"

**定义 5.2.1**　$A:X\to Y$ 线性。**紧算子**：$X$ 的任意有界列 $\{x_n\}$，$\{Ax_n\}$ 在 $Y$ 中总有收敛子列。全体记 $K(X,Y)$（$X=Y$ 时简记 $K(X)$）。

等价刻画：$A$ 紧 $\iff$ $\overline B_X(0,1)$ 的像 $A\big(\overline B(0,1)\big)$ 相对紧。**相对紧 ⟹ 有界**，故 $K(X,Y)\subset B(X,Y)$——紧是比有界强一档的性质。

**直觉**：FA 05 证明了无穷维单位闭球**不紧**（两两距离 $\frac12$ 的球面序列）。紧算子把这个"不紧的大球"压成**相对紧的小像**——像看起来"几乎住在有限维里"（完全有界：任意 $\varepsilon$ 下有限网覆盖）。**紧算子 = 无穷维里最像矩阵的算子**：矩阵（有限维）把球映成椭球（紧集），一般无穷维算子把球摊成"无穷维胖集"，紧算子守住了椭球性。

**注 5.2.1 四件套**（紧算子的代数作业簿）：

| 命题 | 内容 | 依据 |
|---|---|---|
| 有限秩 ⟹ 紧 | $Ax=\sum_{k=1}^n f_k(x)y_k$（值域有限维）则 $A$ 紧 | 有界列的像落在有限维空间，定理 2.3.3 给收敛子列 |
| 有限维全员紧 | $\dim X<\infty$ 时 $B(X,Y)=K(X,Y)$ | 单位球紧（FA 05）+ 紧集的连续像紧 |
| 紧 ∘ 有界 = 紧 | $A$ 紧、$B$ 有界 ⟹ $AB$、$BA$（有意义时）紧 | 紧性在复合中"传染" |
| $K(X,Y)$ 是子空间 | 线性组合保紧 | 对角线提取法（子列的子列） |

### 2.2 自伴算子与数值值域

**定义**　$H$ 复 Hilbert 空间，$A\in B(H)$ **自伴**：$A=A^*$，即 $\langle Ax,y\rangle=\langle x,Ay\rangle$。最简单的自伴算子是**正交投影** $P_M$（$M$ 闭子空间）——勾股分解一行验证。

**定义 5.3.1**　**数值值域** $\omega(A)=\{\langle Ax,x\rangle:\|x\|=1\}$，**数值半径** $R(A)=\sup_{\mu\in\omega(A)}|\mu|$。

自伴算子的全部几何浓缩在一个实数区间里：$\omega(A)\subset\mathbb{R}$（定理 5.3.1），谱被夹在 $[m_A,M_A]=[\inf\omega(A),\sup\omega(A)]$ 中且两端都是谱点（定理 5.3.6）。**自伴算子 = "带实谱的实数型算子"**——量子力学里可观测量的数学身份（测量值=谱点）。

### 2.3 预备：两个对偶零化子

**定义 5.2.2**　$M\subset X$：${}^\perp M=\{f\in X':f|_M=0\}$；$N\subset X'$：$N^\perp=\{x\in X:f(x)=0,\ \forall f\in N\}$。（都是闭线性子空间；FA 09 题 4 的双向推广——正交补的对偶版，没有内积时用泛函零化代偿。）

---

## 3. 主要定理与证明

### 3.1 定理 5.2.1 + 例 5.2.1 + 定理 5.2.2：紧算子的三层身份

> **定理 5.2.1**　(1) $K(X,Y)$ 是 $B(X,Y)$ 的线性子空间；(2) $Y$ Banach 时 $K(X,Y)$ 是 $B(X,Y)$ 的**闭**子空间。

**(2) 的证明（完整——$\varepsilon/3$-网，紧性逼近的标准样板）**　$A_n\in K(X,Y)$、$A\in B(X,Y)$、$\|A_n-A\|\to0$，证 $A$ 紧：只需证 $A\big(\overline B(0,1)\big)$ 完全有界（$Y$ Banach + FA 02：完备空间中完全有界 $\iff$ 相对紧）。任取 $\varepsilon>0$，取 $N$ 使 $\|A_N-A\|<\frac\varepsilon3$。$A_N$ 紧给 $A_N\big(\overline B\big)$ 完全有界：设 $y_1,\dots,y_k$ 是其 $\frac\varepsilon3$-网，$y_i=A_Nx_i$。下证 $Ax_1,\dots,Ax_k$ 是 $A\big(\overline B\big)$ 的 $\varepsilon$-网：任取 $x\in\overline B$，取 $i$ 使 $\|A_Nx-A_Nx_i\|<\frac\varepsilon3$，则

$$
\|Ax-Ax_i\|\le\|Ax-A_Nx\|+\|A_Nx-A_Nx_i\|+\|A_Nx_i-Ax_i\|<\frac\varepsilon3+\frac\varepsilon3+\frac\varepsilon3=\varepsilon.\;\blacksquare
$$

**例 5.2.1（FA 13 例 5.1.2 的身份揭晓）**　$A(x_1,x_2,\dots)=(\frac{x_1}1,\frac{x_2}2,\frac{x_3}3,\dots)\in B(\ell^2)$。截断 $A_m=(x_1,\frac{x_2}2,\dots,\frac{x_m}m,0,\dots)$ 是**有限秩**算子（紧），且 $\|A-A_m\|=\frac1{m+1}\to0$——定理 5.2.1(2) 给 $A\in K(\ell^2)$。**FA 13 里那个谱干净的对角算子本来就是紧算子**——它的好谱型不是巧合，是 Riesz–Schauder 定理（§3.3）的特例。同时这也给出紧算子的第三种身份：

$$
\text{有限秩}\ \subsetneq\ \text{紧}\ \subsetneq\ \text{有界},\qquad\text{紧}=\text{有限秩的范数极限（}Y\ \text{Banach 时）}.
$$

（两个包含都真：对角算子紧而非有限秩；无穷维恒等算子有界而不紧。）

> **定理 5.2.2**　$A\in K(X,Y)$ ⟹ $A^*\in K(Y',X')$。

**证明思路（双层网——对偶的紧性由原像的紧性支付）**　$A\big(\overline B_X(0,1)\big)$ 完全有界：取 $\frac\varepsilon4$-网 $Ax_1,\dots,Ax_n$。映射 $T:Y'\to\mathbb{K}^n$，$Tg=(g(Ax_1),\dots,g(Ax_n))$ **有限秩故紧**，$T\big(\overline B_{Y'}(0,1)\big)$ 完全有界：取 $\frac\varepsilon4$-网 $Tg_1,\dots,Tg_m$。则 $A^*g_1,\dots,A^*g_m$ 是 $A^*\big(\overline B_{Y'}\big)$ 的 $\varepsilon$-网：任取 $g$、任取 $x\in\overline B_X$，取 $j$（对偶球上）与 $i$（原像球上），
$$
|g(Ax)-g_j(Ax)|\le|g(Ax)-g(Ax_i)|+|g(Ax_i)-g_j(Ax_i)|+|g_j(Ax_i)-g_j(Ax)|\le\frac\varepsilon4+\frac\varepsilon4+\frac\varepsilon4<\varepsilon,
$$
对 $x$ 取 sup 得 $\|A^*g-A^*g_j\|<\varepsilon$。∎（**用值域的网造对偶球面的网**——每层 $\varepsilon/4$ 的预算分配是紧性论证的精细账本。）

### 3.2 引理 5.2.1 + 定理 5.2.3：特征空间有限维、值域闭

**引理 5.2.1（有限维子空间有闭补）**　$N$ 有限维 ⟹ 存在闭子空间 $M$ 使 $X=M\oplus N$。
（证明：$N$ 的 Hamel 基 $e_1,\dots,e_n$，坐标泛函 $a_i\in N'$ 有限维自动有界（FA 06：有限维空间上线性泛函免费有界），**HB 延拓**到 $\varphi_i\in X'$，$M=\bigcap_iN(\varphi_i)$ 闭；分解 $z=\sum\varphi_i(x)e_i$、$y=x-z\in M$、交为零。——**HB 延拓的又一种用法**（FA 09 保范延拓、分离定理之外的第三种姿态：造闭补）。）

> **定理 5.2.3**　$A\in K(X)$，$\lambda\ne0$。则 $\dim N(\lambda-A)<\infty$ 且 $R(\lambda-A)$ 闭。

**证明（完整——两件事共用一个机制）**　数乘归一 $\lambda=1$（$\lambda-A=\lambda(1-\frac A\lambda)$）。

**特征空间有限维**：$x_n\in N(1-A)$、$\|x_n\|\le1$，则 $Ax_n=x_n$——$A$ 紧给 $\{Ax_n\}$ 有收敛子列，即 $\{x_n\}$ 自己有收敛子列且极限 $x$ 满足 $Ax=x$（$x_{n_k}\to x$ ⟹ $Ax_{n_k}\to Ax$，但 $Ax_{n_k}=x_{n_k}\to x$，极限唯一）。**单位闭球序列紧 ⟹ 有限维**（FA 05 定理 2.3.3）。

**值域闭**：$N=N(1-A)$ 有限维，引理 5.2.1 给闭补 $M$，$X=M\oplus N$。$B=(1-A)\big|_M$ 单射（$Bx=0$ ⟹ $x\in M\cap N=\{0\}$）且 $R(B)=R(1-A)$（$N$ 上 $1-A=0$）。**核心估计**：存在 $a>0$ 使
$$
a\|x\|\le\|Bx\|,\qquad x\in M. \tag{$*$}
$$
反证：$x_n\in M$、$\|x_n\|=1$、$Bx_n\to0$，即 $x_n-Ax_n\to0$。$A$ 紧给 $Ax_{n_k}\to y$，故 $x_{n_k}=(x_{n_k}-Ax_{n_k})+Ax_{n_k}\to y$；$M$ 闭给 $y\in M$；$By=\lim Bx_{n_k}=0$，单射给 $y=0$——但 $\|x_{n_k}\|=1\to\|y\|$，矛盾。有 $(*)$ 后：$Bx_n\to y$ 给 $\{x_n\}$ Cauchy（$a\|x_n-x_m\|\le\|Bx_n-Bx_m\|$），$M$ 闭（Banach 的闭子空间完备）给 $x_n\to x\in M$，$y=Bx\in R(B)$。∎

**读法**：$(*)$ 是"**受控可逆**"——$B$ 在 $M$ 上有下界，这是 FA 11 题 34（双向控制）的紧算子版。整个证明的模式"**紧性提取 + 单射对账**"会在 §3.3 反复出现：紧算子的世界里，**有界序列的收敛子列是唯一弹药**。

### 3.3 定理 5.2.4（Riesz–Schauder）：紧算子谱的三连画像

> **定理 5.2.4**　$X$ 复 Banach，$A\in K(X)$。
> (1) $\sigma(A)\setminus\{0\}\subset\sigma_p(A)$——**非零谱点全是特征值**；
> (2) $\sigma(A)$ 至多可数，且 $0$ 是唯一可能的聚点；
> (3) $\dim X=\infty$ ⟹ $0\in\sigma(A)$。

**证明（完整——Riesz 引理的全额兑现）**

**(1)** 设 $\lambda\ne0$、$\lambda\notin\sigma_p(A)$（即 $\lambda-A$ 单射），要证 $\lambda\in\rho(A)$。二项式分解：$(\lambda-A)^n=\lambda^nI-AB$，其中 $B\in B(X)$、$AB\in K(X)$（每项含 $A$，紧 ∘ 有界）。于是 $\mu-AB$ 型（$\mu=\lambda^n\ne0$）套定理 5.2.3：$R\big((\lambda-A)^n\big)$ **闭**，且链
$$
R\big((\lambda-A)^{n+1}\big)\subset R\big((\lambda-A)^n\big)
$$
在 $\lambda-A$ 不满射时**严格下降**（若某层取等：$y\in R((\lambda-A)^{n-1})$ 则 $(\lambda-A)y\in R((\lambda-A)^n)=R((\lambda-A)^{n+1})$，即 $(\lambda-A)y=(\lambda-A)^{n+1}x$，**单射**消去一个因子给 $y\in R((\lambda-A)^n)$——等式逐层回退，最终逼出 $R(\lambda-A)=X$，与不满射矛盾）。**Riesz 引理**（FA 05 引理 2.3.2，$\varepsilon=\frac12$）逐层取 $y_n\in R\big((\lambda-A)^n\big)$、$\|y_n\|=1$、$\rho\big(y_n,R((\lambda-A)^{n+1})\big)\ge\frac12$。对 $m>n$：三项 $(\lambda-A)y_n-(\lambda-A)y_m+\lambda y_m$ 全落在 $R\big((\lambda-A)^{n+1}\big)$ 里，故

$$
\|Ay_n-Ay_m\|=\Big\|\lambda y_n-\big[(\lambda-A)y_n-(\lambda-A)y_m+\lambda y_m\big]\Big\|\ge\frac{|\lambda|}{2}.
$$

——$\{Ay_n\}$ 无收敛子列，**与 $A$ 紧矛盾**。故 $\lambda-A$ 满射，$\lambda\in\rho(A)$。

**(2)** 固定 $t>0$，证 $\{\lambda\in\sigma(A):|\lambda|\ge t\}$ 有限。反设有互异 $\lambda_n$、$|\lambda_n|\ge t$，由 (1) 取特征向量 $x_n$。**引理 5.2.2**（互异特征值的特征向量线性无关——归纳 + "作用 $A$ 再相减"）给 $M_n=\operatorname{span}\{x_1,\dots,x_n\}$ 严格升链（有限维子空间自动闭，FA 05）。Riesz 引理取 $y_n\in M_n$、$\|y_n\|=1$、$\rho(y_n,M_{n-1})\ge\frac12$。关键观察：$A(M_n)\subset M_n$（特征空间不变）且 $(\lambda_n-A)y_n\in M_{n-1}$（逐项 $(\lambda_n-\lambda_i)x_i$ 只用到前 $n-1$ 个基）。对 $m>n$：括号 $(\lambda_m-A)y_m+Ay_n$ 落在 $M_n$（前者 $\in M_{m-1}\subseteq M_n$，后者 $\in M_n$），而 $M_n\subseteq M_{m-1}$ 给 $\rho(y_m,M_n)\ge\rho(y_m,M_{m-1})\ge\frac12$，故

$$
\|Ay_m-Ay_n\|=\Big\|\lambda_my_m-\big[(\lambda_m-A)y_m+Ay_n\big]\Big\|\ge\frac{|\lambda_m|}{2}\ge\frac{t}{2}.
$$

——又与紧性矛盾。故 $\{\sigma\ge t\}$ 有限，$0$ 是唯一可能聚点；$\sigma(A)\setminus\{0\}=\bigcup_n\{\sigma\ge\frac1n\}$ 是可数个有限集之并，至多可数。

**(3)** $\dim X=\infty$ 时反设 $0\in\rho(A)$：$A$ 双射且 $A^{-1}\in B(X)$（FA 11 逆算子定理），于是 $I=AA^{-1}\in K(X)$（紧 ∘ 有界）——单位闭球紧——FA 05 定理 2.3.3 判 $\dim X<\infty$，矛盾。∎

**品味**：三段的引擎是同一句话的三个变奏——**"紧算子的像不容忍两两距离 $\ge\theta$ 的序列"**。FA 05 造出两两距 $\frac12$ 的球面序列证"无穷维单位球不紧"（$\varepsilon<1$ 的折扣），本篇把这个序列**塞进紧算子的像里**引爆矛盾：折扣全额兑现，还带利息（两次都拿到 $\frac{|\lambda|}2$、$\frac t2$ 的定量下界）。**Riesz 引理从"维数的探针"升级为"谱形状的雕刻刀"**。

**$0$ 的三重身份**（无穷维紧算子必有 $0\in\sigma$，但 $0$ 的谱型自由）：

| 算子 | $0$ 的身份 | 现场 |
|---|---|---|
| 对角 $\alpha_n\searrow0$ 全非零 | **连续谱** | FA 13 例 5.1.2（值域稠密不满） |
| Volterra $\int_0^t$ | **剩余谱** | FA 13 例 5.1.4（值域 $y(0)=0$ 不稠密） |
| 对角某 $\alpha_{n_0}=0$ | **点谱** | $e_{n_0}$ 是特征向量 |

——FA 13 的四分法在 $0$ 这一点上"三谱俱全"。

### 3.4 定理 5.2.5 + 5.2.6：Fredholm 二择一

> **定理 5.2.5**　$A\in K(X)$，$\lambda\ne0$。
> (1) $R(\lambda-A)=N(\lambda-A^*)^\perp$；
> (2) $R(\lambda-A^*)={}^\perp N(\lambda-A)$。

**证明要点**　(1) "$\subset$"：$y=(\lambda-A)x$、$f\in N(\lambda-A^*)$ 时 $f(y)=[(\lambda-A^*)f](x)=0$。"$\supset$"：$R(\lambda-A)$ 闭（定理 5.2.3！），若 $y\notin R$，**定理 4.1.7 分离**（FA 09）：存在 $f$ 零化 $R$ 而 $f(y)\ne0$；零化值域恰是 $f\in N(\lambda-A^*)$，与 $y\in N^\perp$ 矛盾。(2) "$\subset$" 同上；"$\supset$"：$f\in{}^\perp N(\lambda-A)$，在 $R(\lambda-A)$ 上定义 $g\big((\lambda-A)x\big)=f(x)$（**良定**：$f$ 零化核，同一个 $y$ 的两个原像之差在核上）、有界（**商映射** $\tilde\lambda:X/N\to R$ 双射，**开映射定理**给 $\tilde\lambda^{-1}$ 有界，$\|x\|$ 换成商范数 $\|\tilde x\|$ 控制 $|g(y)|$），**HB 延拓** $g$ 到 $g_0\in X'$，验证 $(\lambda-A^*)g_0=f$。∎

（(2) 的证明是第 4 章工具的密集调用：**闭值域（5.2.3）+ 商映射与商范数 + 开映射（FA 11）+ HB 延拓（FA 09）**——四件套在同一个构造里合体。）

> **定理 5.2.6**　$A\in K(X)$，$\lambda\ne0$。
> (1) $\sigma(A)=\sigma(A^*)$；
> (2) $\lambda\in\sigma_p(A)$、$\mu\in\sigma_p(A^*)$、$\lambda\ne\mu$、$x\in N(\lambda-A)$、$f\in N(\mu-A^*)$ ⟹ $f(x)=0$；
> (3) $\dim N(\lambda-A)=\dim N(\lambda-A^*)$。

(2) 一行：$\lambda f(x)=f(Ax)=(A^*f)(x)=\mu f(x)$，$\lambda\ne\mu$ 逼 $f(x)=0$。(1)：$\lambda\in\rho(A)\iff\lambda-A$ 满射/单射 $\iff$（经 5.2.5 的值域—零空间字典 + HB）$\lambda-A^*$ 单射/满射 $\iff\lambda\in\rho(A^*)$——**值域与零空间的对偶字典逐条翻译**。(3) 的证明是本篇最精巧的构造：反设 $n<n^*$，用**引理 5.2.3（双对偶基）**造一个**有限秩扰动** $F(x)=\sum_{i\le n}g_i(x)y_i$，使 $\lambda-(A+F)$ 单射（核被 $g_i$ 的对偶性逐层清空）⟹ 紧算子谱论（5.2.4(1)）给满射；再用 $n^*$ 侧的第 $n+1$ 个对偶向量 $y_{n+1}$ 找原像，对账得 $1=f_{n+1}(y_{n+1})=0$ 矛盾。反向把 $X$ 经典范映射嵌入 $X''$（$(\lambda-A)^{**}$ 是 $\lambda-A$ 的延拓）双向夹逼。（完整证明见教材；**"加一个有限秩算子消掉核"**——Fredholm 交替理论的技术心脏。）

**合流：Fredholm 二择一**（5.2.5(1) + 5.2.6 的合订本，教材理论的出口）：

> **二择一**　$A$ 紧、$\lambda\ne0$，积分方程 $(\lambda-A)x=y$：
> **第一择**：齐次方程 $(\lambda-A)x=0$ 只有零解 ⟹ $\lambda\in\rho(A)$，**任取 $y$ 方程唯一可解**；
> **第二择**：齐次方程有 $d\ge1$ 维解空间 ⟹ 方程可解 $\iff$ $y$ 与共轭齐次 $(\lambda-A^*)f=0$ 的**全部解正交**（$f(y)=0$，恰好 $d$ 个独立条件，因 $\dim$ 相等）。

这是有限维"秩—零化度定理 + $R(A)={}^\perp N(A^T)$"的无穷维翻版——**线性代数对偶理论在紧算子上的完整复活**。历史上 Fredholm 就是在积分方程上发现这件事的（"齐次方程的非平凡解恰好解释非齐次方程的全部不可解性"），它是微分方程边值问题、数值分析、正则化理论的通用判据。

### 3.5 定理 5.3.1 / 5.3.2 / 5.3.3：自伴算子的三块基石

> **定理 5.3.1**　$A\in B(H)$ 自伴 $\iff$ $\langle Ax,x\rangle\in\mathbb{R}$（一切 $x$）。

（⟹ 一行；⟸ 用**极化恒等式的四方和**：$4\langle Ax,y\rangle=\langle A(x{+}y),x{+}y\rangle-\langle A(x{-}y),x{-}y\rangle+i\langle A(x{+}iy),x{+}iy\rangle-i\langle A(x{-}iy),x{-}iy\rangle$——实值性把前两项变成实的、后两项经换元 $x\pm iy=\pm i(y\mp ix)$ 搬运后恰好给出 $4\langle x,Ay\rangle$。**"实二次型决定对称双线性型"**——与实分析里"二次型恢复内积"（FA 07 极化恒等式）同源。）

> **定理 5.3.2**　$\sigma(A^*)=\{\bar\lambda:\lambda\in\sigma(A)\}$（Hilbert 伴随）。

（$\lambda\in\rho(A)$ ⟹ 预解式恒等式两边取伴随：$R(\lambda,A)^*(\bar\lambda-A^*)=(\bar\lambda-A^*)R(\lambda,A)^*=I$——**可逆性经伴随原样传递，只是 $\lambda$ 共轭**。对照定理 5.2.6(1) 的 Banach 版 $\sigma(A')=\sigma(A)$ **无共轭**——差别全在 Hilbert 伴随是共轭线性的（FA 09 两个 $T^*$ 辨析的谱论续篇）。）

> **定理 5.3.3**　$A$ 自伴 ⟹ $\|A\|=\sup_{\|x\|=1}|\langle Ax,x\rangle|=R(A)$（**数值半径 = 范数**）。

（$\le$ 显然；$\ge$：**相位旋转**先行——换 $e^{i\theta}x$ 可把 $|\langle Ax,y\rangle|$ 吸收成实部 $\operatorname{Re}\langle A(e^{i\theta}x),y\rangle=\frac14\big[Q(x{+}y)-Q(x{-}y)\big]$（$Q(z)=\langle Az,z\rangle$，实值），于是
$$
|\langle Ax,y\rangle|\le\frac r4\big(\|x+y\|^2+\|x-y\|^2\big)\ \overset{\text{平行四边形}}{=}\ \frac r2\big(\|x\|^2+\|y\|^2\big);
$$
取 $y=\frac{\|x\|}{\|Ax\|}Ax$（$\|y\|=\|x\|$）收账 $\|x\|\|Ax\|\le r\|x\|^2$。——FA 12 定理 4.5.3 证明里"符号追踪"的孪生技巧：先把辐角转正，再用实部只花一半预算。）

### 3.6 定理 5.3.4：谱落在实轴、剩余谱空

> **定理 5.3.4**　$A\in B(H)$ 自伴。(1) $\sigma(A)\subset\mathbb{R}$；(2) 互异 $\lambda,\mu\in\sigma_p(A)$ 的特征空间互相正交；(3) $\sigma_r(A)=\varnothing$。

**证明（完整——虚部控制一行定乾坤）**　(1) 核心恒等式：

$$
\langle(\lambda-A)x,x\rangle-\langle x,(\lambda-A)x\rangle=2\,i\,\mathrm{Im}(\lambda)\|x\|^2
\tag{5.17}
$$

（自伴性消掉 $\langle Ax,x\rangle$ 项）。取模 + Schwarz：

$$
2|\mathrm{Im}(\lambda)|\|x\|^2\le2|\langle(\lambda-A)x,x\rangle|\le2\|(\lambda-A)x\|\|x\|
\quad\Longrightarrow\quad
|\mathrm{Im}(\lambda)|\|x\|\le\|(\lambda-A)x\|. \tag{5.19}
$$

$\mathrm{Im}(\lambda)\ne0$ 时：(5.19) 给 $\lambda-A$ 单射；**值域稠密**（若 $R\ne\overline R$，FA 07 的正交分解给 $x\ne0$ 垂直于 $R$，特别 $\langle(\lambda-A)x,x\rangle=0$，代回上式左边逼 $2|\mathrm{Im}(\lambda)|\|x\|^2\le0$ 即 $x=0$，矛盾）；**值域闭**（(5.19) 让像收敛回推原像 Cauchy——FA 11 题 33 的同款回推）。稠密 + 闭 = 满射，$\lambda\in\rho(A)$。故 $\sigma\subset\mathbb{R}$。

(2) $\lambda\langle x,y\rangle=\langle Ax,y\rangle=\langle x,Ay\rangle=\mu\langle x,y\rangle$，$\lambda\ne\mu$ 逼正交。（**不同特征值的特征空间正交——谱定理的地基**：特征向量自动排成正交族。）

(3) 反设 $\lambda\in\sigma_r(A)$：由 (1) $\lambda$ 实，$R(\lambda-A)$ 不稠密 ⟹ $R^\perp\ne\{0\}$（FA 07）。$x\in R^\perp$：对一切 $y$，$0=\langle(\lambda-A)y,x\rangle=\langle y,(\lambda-A)x\rangle$（$\lambda$ 实 + 自伴），$y$ 任意逼 $(\lambda-A)x=0$——$x$ 是特征向量，$\lambda\in\sigma_p$，与 $\sigma_r\cap\sigma_p=\varnothing$ 矛盾。∎

**读法**：自伴算子的 FA 13 四分法**塌缩成三分**（$\sigma_r$ 空、$\sigma\subset\mathbb{R}$）：只剩"特征值 + 连续谱（实轴上）"。乘法算子（自伴）就是纯连续谱样本。**(1) 的机制值得记**：虚部被不等式 (5.19) 直接判死刑——"虚数方向的谱会破坏能量守恒 $\langle Ax,x\rangle\in\mathbb{R}$"。

### 3.7 定理 5.3.5 + 5.3.6：数值域夹逼与端点定理

> **定理 5.3.5**　$A$ 自伴。(1) $\sigma(A)\subset\overline{\omega(A)}$ 且 $\omega(A)\subset\mathbb{R}$；(2) $R(A)=\|A\|$。
> （教材记 $\sigma(A)\subset\omega(A)$；证明中 $d=\rho(\lambda,\omega(A))>0$ 实际用到 $\lambda\notin\overline{\omega(A)}$，故此处取闭包——对端点 $m_A,M_A$ 的使用无影响。）

（(1)：$\lambda\notin\overline{\omega(A)}$ 时 $d=\mathrm{dist}(\lambda,\omega(A))>0$，对单位向量 $d\le\big|\lambda-\langle A\frac x{\|x\|},\frac x{\|x\|}\rangle\big|$ 放大成 $d\|x\|\le\|(\lambda-A)x\|$——与 5.3.4(1) 同款"距离控制"，再走一遍稠密 + 闭 ⟹ 满射。）

> **定理 5.3.6**　$m_A=\inf\omega(A)$、$M_A=\sup\omega(A)$。则 $\sigma(A)\subset[m_A,M_A]$ 且 $m_A,M_A\in\sigma(A)$——**端点都是谱点**。

**证明（$M_A$ 侧完整——正算子的判别式技巧）**　$B=M_A-A$ 是**正算子**：$\langle Bx,x\rangle=M_A\|x\|^2-\langle Ax,x\rangle\ge0$。对一切 $t\in\mathbb{R}$：$0\le\langle B(tBx+x),tBx+x\rangle=t^2\langle B^2x,Bx\rangle+2t\langle Bx,Bx\rangle+\langle Bx,x\rangle$——**关于 $t$ 的二次函数恒非负 ⟹ 判别式 $\le0$**：

$$
\langle Bx,Bx\rangle^2\le\langle B^2x,Bx\rangle\langle Bx,x\rangle\le\|B\|^3\|x\|^2\langle Bx,x\rangle\qquad(5.21)
$$

由 $M_A$ 的定义 $\inf_{\|x\|=1}\langle Bx,x\rangle=0$，(5.21) 给 $\inf_{\|x\|=1}\|Bx\|=0$（5.22）。反设 $M_A\in\rho(A)$：$B$ 可逆且 $B^{-1}$ 有界，$\|x\|=\|B^{-1}Bx\|\le\|B^{-1}\|\|Bx\|$ 给 $\inf_{\|x\|=1}\|Bx\|\ge\frac1{\|B^{-1}\|}>0$——与 (5.22) 矛盾。$m_A$ 侧对 $A-m_A$ 同理。∎

**合账**（FA 13 的 $r$ vs $\|A\|$ 辨析结账）：自伴时 $\|A\|=R(A)=\max(|m_A|,|M_A|)$（5.3.3 + 5.3.5），且 $m_A,M_A\in\sigma$ 给 $r(A)=\max(|m_A|,|M_A|)$——**三量合一**：

$$
A\ \text{自伴}\quad\Longrightarrow\quad r(A)=\|A\|.
$$

Volterra 的 $\|A\|=1$、$r(A)=0$ 是"非自伴"的代价——**$\|A\|-r(A)$ 是非自伴性的谱论度量**，FA 13 的悬案在此结清。

### 3.8 定理 5.3.7：谱定理——系列的终点站

> **定理 5.3.7（自伴紧算子的谱定理）**　$H$ 复 Hilbert，$A\in B(H)$ 自伴紧。则：
> (1) 存在有限或无限的**非零实**特征值列 $\lambda_1,\lambda_2,\dots$（$|\lambda_1|\ge|\lambda_2|\ge\cdots$，无穷时 $\lambda_n\to0$）与标准正交特征向量 $e_n$，使
> $$Ax=\sum_{n}\lambda_n\langle x,e_n\rangle e_n,\qquad x\in H;$$
> (2) 记 $P_n$ 为到 $\mathbb{C}e_n$ 的正交投影，$A=\sum_n\lambda_nP_n$（依算子范数收敛）；
> (3) 若 $0\notin\sigma_p(A)$，则 $\{e_n\}$ 是 $H$ 的完全标准正交基。

**证明（构造完整——贪心剥谱）**　不妨 $A\ne0$。$\|A\|=\sup_{\|x\|=1}|\langle Ax,x\rangle|>0$ 给 $M_A\ne0$ 或 $m_A\ne0$；取 $\lambda_1=\pm\|A\|$（绝对值较大的那个端点）。**定理 5.3.6** 给 $\lambda_1\in\sigma(A)$，$\lambda_1\ne0$ + **定理 5.2.4(1)** 给 $\lambda_1\in\sigma_p(A)$——取单位特征向量 $e_1$。**不变子空间链**：$Q_1=\mathbb{C}e_1$、$H_1=Q_1^\perp$（都闭）；$A(Q_1)\subset Q_1$ 显然，而 $x\in H_1$、$y\in Q_1$ 时 $\langle Ax,y\rangle=\langle x,Ay\rangle=0$（$Ay\in Q_1$）给 $A(H_1)\subset H_1$——**正交补也是不变的**（自伴性的红利：特征方向剥掉后算子还留在余下空间里）。$A_1=A|_{H_1}$ 仍是自伴紧算子；若 $A_1\ne0$ 就重复构造（$\lambda_2=\pm\|A_1\|$，$|\lambda_1|\ge|\lambda_2|$）……

- **有限步归零**（某 $A_n=0$）：$x=y+z$（$y\in Q_n$、$z\in H_n$，$Az=0$），$Ax=Ay=\sum_{i\le n}\lambda_i\langle x,e_i\rangle e_i$——有限和。
- **永不归零**：得无穷列 $|\lambda_1|\ge|\lambda_2|\ge\cdots$。$\lambda_n\to0$：任取 $\varepsilon>0$，$|\lambda|\ge\varepsilon$ 的特征值**个数有限**（5.2.4(2) 互异值有限）且每个**重数有限**（5.2.3 特征空间有限维），故 $\{n:|\lambda_n|\ge\varepsilon\}$ 有限。**余维归零**：$H_\infty=Q_\infty^\perp$（$Q_\infty=\overline{\operatorname{span}}\{e_n\}$）上 $x\in H_\infty\subset H_n$ 给 $|\langle Ax,x\rangle|\le\|A|_{H_n}\|\|x\|^2=|\lambda_{n+1}|\|x\|^2\to0$，即 $\langle Ax,x\rangle=0$ 对一切 $x\in H_\infty$；**极化恒等式**（定理 5.3.1 的四方和）把"对角元全零"升级为 $A|_{H_\infty}=0$。分解 $x=y+z$（$y=\sum\langle x,e_n\rangle e_n\in Q_\infty$）收尾：$Ax=Ay=\sum_n\lambda_n\langle x,e_n\rangle e_n$。

**(2) 的余项估计（Bessel 的精确回收）**：

$$
\Big\|A-\sum_{i\le n}\lambda_iP_i\Big\|^2=\sup_{\|x\|=1}\Big\|\sum_{i>n}\lambda_i\langle x,e_i\rangle e_i\Big\|^2=\sup_{\|x\|=1}\sum_{i>n}|\lambda_i|^2|\langle x,e_i\rangle|^2\ \overset{\text{Bessel}}{\le}\ |\lambda_{n+1}|^2\to0.
$$

**(3)**：$0\notin\sigma_p$ ⟹ $A$ 单射；$x\perp\{e_n\}$ ⟹ $Ax=0$ ⟹ $x=0$——正交系完全（FA 08 的完全性判据：与一切基向量正交的向量只有零）。∎

**品味**：构造是**贪心算法**——每轮取"当前空间里绝对值最大的谱端点"当下一块对角元。三台引擎按帧出场：**端点定理（5.3.6）供特征值、紧性（5.2.4）供"非零即特征值"、自伴性供正交不变链**；收尾两处"对角元全零 ⟹ 算子全零"都是**极化恒等式**把二次型信息还原成双线性信息。**(2) 的余项公式是全定理最实用的输出**：截断误差**精确等于**下一个特征值——低秩近似的误差账单一目了然。

---

## 4. 例子与反例速查

| 命题 | 真假 | 例子/反例 |
|------|------|----------|
| 紧 ⟹ 有界 | ✅ | 相对紧 ⟹ 有界；$K(X,Y)\subset B(X,Y)$ |
| 有界 ⟹ 紧 | ❌（$\dim X=\infty$） | 恒等算子；右移 $\|Se_n\|=1$ 无收敛子列 |
| 有限秩 ⟹ 紧 | ✅ | 注 5.2.1(2)：像住在有限维，列紧接管 |
| 紧 ⟹ 有限秩 | ❌ | 对角算子 $\frac{x_n}n$（有限秩逼近的极限）；Volterra |
| 紧算子的范数极限仍紧 | ✅ | 定理 5.2.1(2)（$Y$ Banach）——"紧 = 可有限秩逼近" |
| 紧算子的非零谱点全是特征值 | ✅ | 定理 5.2.4(1)；一般算子 ❌（乘法算子无特征值但 $\sigma=[0,1]$） |
| 乘法算子 $(Ax)(t)=tx$ 不紧 | ✅ | 题 16：$\sigma=[0,1]$ 不可数，紧算子谱至多可数（5.2.4(2)）一步矛盾 |
| 无穷维上 $I$ 紧 | ❌ | $I$ 紧 ⟺ $\dim<\infty$（5.2.4(3) 逆否 + 正向都成立） |
| 无穷维紧算子必有 $0\in\sigma$ | ✅ | 5.2.4(3)：$0\in\rho$ ⟹ $I=AA^{-1}$ 紧 ⟹ 有限维 |
| $0$ 的谱型可以是点谱/连续谱/剩余谱 | ✅ 三种都有 | 对角含零项（点）/ $\frac{x_n}n$ 全非零（连续，FA 13 例 5.1.2）/ Volterra（剩余，例 5.1.4） |
| 紧 + 幂等 ⟹ 有限秩 | ✅ | 题 17（$R(A)$ 的单位球紧 + $R(A)$ 闭 ⟹ 有限维） |
| $\sigma(A^*)=\sigma(A)$（Banach 共轭，紧算子） | ✅ | 定理 5.2.6(1)；（对一般 $A\in B(X)$ 也成立，经预解式取共轭） |
| $\sigma(A^*)=\overline{\sigma(A)}$（Hilbert 伴随） | ✅ | 定理 5.3.2——共轭线性之差，与 Banach 版差一个复共轭 |
| 自伴 ⟹ $\sigma\subset\mathbb{R}$、$\sigma_r=\varnothing$ | ✅ | 定理 5.3.4；非自伴可全在虚轴（对角 $\alpha_n=\frac in$ 紧而非自伴） |
| 自伴 ⟹ $r(A)=\|A\|$ | ✅ | 5.3.3 + 5.3.6 端点；非自伴可 $r=0<\|A\|=1$（Volterra） |
| 自伴但谱连续（无特征值） | ✅ | 乘法算子：自伴管位置不管形状——**对角化还需要紧性** |
| 正交投影 $P_M$ 的 $m_A,M_A$ | $0$ 与 $1$ | 题 21：$\omega(P)=[0,1]$（$M\ne0,H$ 时） |
| 自伴紧 ⟹ $\sigma_p\ne\varnothing$ | ✅ | 题 22（= 定理 5.3.7 证明的第一步；对照一般自伴可 $\sigma_p=\varnothing$） |
| 对角算子 $\lambda_i\in\mathbb{R}$ 有界：$\sigma=[a,b]$ 的条件 | ✅ ⟺ 特征值稠密 | 题 23：$\sigma=\overline{\{\lambda_i\}}$，故 $[a,b]$ ⟺ $\{\lambda_i\}$ 在 $[a,b]$ 中稠密 |

---

## 5. 易混淆辨析

| 概念对 | 辨析 |
|--------|------|
| **有限秩 vs 紧 vs 有界（三层塔）** | 有限秩（像有限维）⊂ 紧（像相对紧）⊂ 有界（像有界），无穷维时两层包含都真。中间层的美德：**紧 = 有限秩的范数极限**（K 闭）——紧算子是"可以用矩阵逼近的算子"，这正是它谱型像矩阵的原因（5.2.4）。Volterra 证明紧性通常也走这条路（有限秩逼近）或 Arzelà–Ascoli（教材外）。 |
| **紧管形状 vs 自伴管位置** | 紧 ⟹ 谱离散化（非零全是特征值、至多可数、聚点只许 0）；自伴 ⟹ 谱实轴化（$\sigma\subset[m_A,M_A]$、剩余谱空）。两性质独立：乘法算子自伴不紧（谱连续）、对角 $\frac in$ 紧不自伴（谱离散但虚轴）。**只有叠加（自伴紧）才得到"实轴上离散 + 正交特征向量系 + 完全对角化"**——谱定理要两张门票。 |
| **Fredholm 二择一 vs 有限维秩定理** | 有限维：$R(A)={}^\perp N(A^T)$、$\dim N(A)=\dim N(A^T)$、齐次零解 ⟹ 处处唯一可解。紧算子（$\lambda\ne0$）：定理 5.2.5 + 5.2.6 逐条复活，唯一新增条款是"$0$ 处的一切都可以塌"（$0\in\sigma$ 恒发生、三种谱型都可能）。**"非零谱 = 矩阵世界，$0$ = 无穷维的门户"**。 |
| **$\sigma(A')=\sigma(A)$ vs $\sigma(A^*)=\overline{\sigma(A)}$** | Banach 共轭（$f\mapsto f\circ A$，线性）保谱不加共轭；Hilbert 伴随（$\langle Ax,y\rangle=\langle x,A^*y\rangle$，共轭线性一侧）保谱但复共轭。根源：Riesz 映射 $y\mapsto\langle\cdot,y\rangle$ 本身共轭线性（FA 08/FA 09 两个 $T^*$ 的辨析）。自伴算子上两者和解：$\sigma$ 实，共轭不动它。 |
| **$0$ 的三重身份** | 无穷维紧算子恒有 $0\in\sigma$，但 $0$ 的谱型三选一自由：点谱（对角含零）、连续谱（$\frac{x_n}n$）、剩余谱（Volterra）。诊断 $0$ 的身份 = 诊断"核与值域的相对位置"：核非零→点谱；值域稠密不满→连续；值域不稠→剩余（自伴时第三种被禁止——5.3.4(3)）。 |
| **数值半径 $R(A)$ vs $\|A\|$ vs $r(A)$** | 一般算子：$r(A)\le R(A)\le\|A\|$（且 $R(A)\ge\frac12\|A\|$，教材外），三个量可互异；自陪时**三量合一**（5.3.3 + 5.3.5 + 5.3.6）。$\|A\|$ 是一次作用的放大、$R(A)$ 是"平均放大"（量子力学期望值的波动上界）、$r(A)$ 是长期放大——自伴算子三种放大率没有分别，因为它"没有隐藏的旋转抵消"。 |
| **可对角化：有限维 vs 谱定理** | 有限维：可对角化是**例外**（Jordan 块普遍存在）；无穷维自伴紧：可对角化是**定理**（永远成立）。差别在于门票：有限维矩阵要 $n$ 个独立特征向量（一般的没有），自伴紧算子的正交不变链（5.3.7）自动供给它们。**自伴性 + 紧性 = 无穷维里"保证可对角"的两个条件**——PCA 世界的数学安全感来源。 |

---

## 6. ML 连接

### 6.1 PCA / Karhunen–Loève / 核方法：谱定理的三个化身

**协方差算子是自伴紧算子**——这句话是 PCA 的全部数学根基。数据 $x\in H$（有限维 $\mathbb{R}^d$ 或无穷维函数空间）的二阶统计造出 $C=\mathbb{E}[x\otimes x]$（$Cv=\mathbb{E}[\langle x,v\rangle x]$）：自伴（对称核/对称性）显然，紧性在无穷维由 $\sum\lambda_n<\infty$ 型衰减保证（Hilbert–Schmidt / 核型算子，教材外标注）。**谱定理（5.3.7）逐字翻译**：$\lambda_n$ = 主成分方差、$e_n$ = 主方向、$x=\sum\sqrt{\lambda_n}\xi_ne_n$（$\xi_n$ 不相关、$\mathrm{Var}=\lambda_n$）——这正是 **Karhunen–Loève 展开**（随机过程的谱定理版本）；取 $H=\mathbb{R}^d$、$C$ 换成样本协方差就是 **PCA**；取 $H=$ RKHS、$C$ 换成积分算子 $(T_kf)(x)=\int k(x,y)f(y)\,d\mu(y)$（对称核 ⟹ 自伴；HS ⟹ 紧）就是 **kernel PCA / Mercer 定理**——$\sum_n\lambda_ne_n(x)e_n(y)=k(x,y)$ 是谱定理 (2) 的核函数版。**"找主成分 = 解特征方程 = 谱定理的贪心构造"**——三百年前的变分法（Rayleigh 商 $\max\frac{\langle Cx,x\rangle}{\|x\|^2}$，定理 5.3.6 端点的变分面孔）与今天的 PCA 是同一件事。

### 6.2 谱衰减 = 有效维度：低秩近似的误差账单

定理 5.3.7(2) 的余项公式 $\|A-\sum_{i\le n}\lambda_iP_i\|=|\lambda_{n+1}|$ 是**截断 SVD / 低秩近似**的无穷维原型（正半定时即 Eckart–Young 定理）：**用前 $n$ 个特征方向重构，误差精确等于下一个特征值**。ML 读法：协方差算子（或 NTK/Gram 算子）的特征值衰减速度就是数据的"**本质维度**"——$\lambda_n\sim n^{-\alpha}$ 快衰减 ⟹ 信息集中在少数方向 ⟹ 泛化好、样本效率高；慢衰减 ⟹ 本质高维。核方法的选择核、early stopping、随机特征（random features）截断，全都在消费这条余项公式；"无穷维模型 + 快谱衰减 = 可学习"是 kernel 方法泛化理论的标准叙事。**紧性不是技术假设，是"可学习的二阶刻画"**。

### 6.3 Fredholm 二择一：逆问题的可解性判据与正则化的边界

带紧算子的方程 $(\lambda-T)x=y$（积分方程、CT 重建、光滑化算子反演）的可解性由二择一完全判决：要么处处唯一可解，要么不可解集恰是有限个正交条件。但紧性在 $\lambda=0$ 处**必然塌方**（$0\in\sigma$）：紧算子的逆（若存在）必无界（FA 06 题 20 积分/求导的现场），且谱聚点 $0$ 使 $\|R(\lambda,T)\|$ 在 $0$ 附近爆炸（FA 13 定理 5.1.2 的稳定半径 $1/\|R\|$ 收缩到零）——**第一类积分方程天然病态**。ML 的应对全部写在本篇的谱语言里：Tikhonov 正则化 $=$ 把谱 $\lambda_n$ 换成 $\frac{\lambda_n}{\lambda_n+\alpha}$（大特征值透过、小特征值压制——**谱域滤波**）；早停 $=$ Neumann/迭代级数的截断；先验光滑 $=$ 对特征向量基加权重。**正则化就是谱定理坐标系下的"衰减器设计"**——系列从 FA 01 的度量公理出发，到今天以"如何驯服一个紧算子的谱"收官，恰好走完从空间到算子的全程。

---

## 7. 逻辑推导概念图

全篇是一个两因子分解：紧性管谱的形状，自伴性管谱的位置，两者叠加才兑换出谱定理。下图顶部是出发点（紧算子定义、自伴算子定义），中间层是两组各自的中间性质，底层是主定理；缩进与箭头是推理方向，每个分叉标明"哪几条前提合起来推出哪个结论"——例如谱定理的贪心构造需要三台引擎同时在场：端点定理供特征值、紧性供"非零即特征值"、自伴性供正交不变链。读者可沿树核对每个定理的门票清单。

```text
出发点（定义与假设）
├── 紧性：A∈K(X)（定义 5.2.1：有界列的像总有收敛子列）
│   ├── 定理 5.2.3：λ≠0 时 dim N(λ-A)<∞ 且 R(λ-A) 闭
│   │   （紧性提取 + 单射对账；引理 5.2.1 闭补 + 受控可逆估计 (*)）
│   ├── Riesz 引理（FA 05，ε=1/2）逐层引爆
│   │   └── 定理 5.2.4（Riesz–Schauder）：非零谱全是特征值；
│   │       谱至多可数、0 是唯一可能聚点；dim X=∞ ⟹ 0∈σ(A)
│   └── 定理 5.2.5 + 5.2.6（值域=零化子、σ(A)=σ(A*)、dim 相等）
│       └── Fredholm 二择一：可解 ⟺ y 与共轭齐次方程的解正交
│
├── 自伴性：A=A*（⟨Ax,y⟩=⟨x,Ay⟩）
│   ├── 定理 5.3.1：⟨Ax,x⟩∈R ⟺ 自伴（极化恒等式四方和）
│   ├── 定理 5.3.4：σ(A)⊂R、σ_r(A)=∅、互异特征值的特征空间正交
│   └── 定理 5.3.3 + 5.3.5：‖A‖=R(A)（数值半径=范数）、σ⊂ω(A) 闭包
│       └── 定理 5.3.6：σ⊂[m_A,M_A] 且端点 m_A、M_A 都是谱点
│           └── 合账：自伴 ⟹ r(A)=‖A‖（三量合一）
│
▼ 两因子叠加（A 自伴紧）
└── 谱定理（定理 5.3.7）：贪心剥谱
    ├── λ_1=±‖A‖（端点定理），λ_1≠0 + 5.2.4(1) ⟹ λ_1∈σ_p
    ├── 剥掉 Ce_1，正交补不变（自伴红利），循环取 λ_2, λ_3, ...
    └── A=Σλ_n⟨x,e_n⟩e_n，余项 ‖A-Σ_{i≤n}λ_iP_i‖=|λ_{n+1}|
```

| 出发点 | 买到什么 | 撤掉它失去什么 |
|---|---|---|
| 紧性（$A\in K(X)$） | 定理 5.2.4：非零谱全是特征值、谱至多可数、聚点只许 $0$；Fredholm 二择一 | 乘法算子 $(Ax)(t)=tx$ 自伴但谱是整条线段 $[0,1]$，一个特征值都没有——紧性才是"谱离散化"的真正供款人（题 16） |
| 自伴性（$A=A^*$） | $\sigma(A)\subset[m_A,M_A]$、$\sigma_r$ 空、特征空间正交、$r(A)=\|A\|$ 三量合一（5.3.4 + 5.3.6） | 对角算子 $\alpha_n=\frac in$ 紧但谱全在虚轴；Volterra $\|A\|=1$ 而 $r(A)=0$——$\|A\|-r(A)$ 这个非自伴性的谱论度量失去意义（§3.7 合账） |
| $\dim X=\infty$ 且 $A$ 紧 | $0\in\sigma(A)$ 恒发生（5.2.4(3)） | 若 $0\in\rho(A)$ 则 $I=AA^{-1}$ 紧 ⟹ 有限维矛盾；$0$ 的三重身份（对角含零=点谱、$\frac{x_n}n$=连续谱、Volterra=剩余谱）无从展览（§3.3 表） |
| Riesz 引理（FA 05 的埋线） | 5.2.4 三连画像的证明引擎：$\|Ay_n-Ay_m\|\ge\frac{|\lambda|}2$ 与紧性矛盾 | "紧算子的像不容忍两两距离 $\ge\theta$ 的序列"的引爆机制失效——FA 05 的折扣（$\varepsilon<1$）无处兑现，"谱形状的雕刻刀"失灵 |
| 谱定理的两张门票（紧 + 自伴） | $A=\sum_n\lambda_n\langle x,e_n\rangle e_n$，截断误差精确等于 $|\lambda_{n+1}|$（低秩近似的误差账单） | 乘法算子自伴不紧：谱连续、无可对角化——PCA / Karhunen–Loève / Mercer 定理的数学本体消失（§6.1） |

---

## 8. 习题精选（教材习题 5 选摘，题 14–24）

**题 16**：$C[0,1]$ 上 $(Ax)(t)=tx(t)$。求证 $A$ 不紧。
*提示*：**谱论一步矛盾**：FA 13 例 5.1.1 已算出 $\sigma(A)=[0,1]$ **不可数**；若 $A$ 紧，定理 5.2.4(2) 给 $\sigma(A)$ 至多可数——矛盾。（用理论反推不紧性，比构造振荡序列便宜得多——这正是"谱是算子的指纹"的用法。）直接构造备选：$x_n(t)=\sin(2\pi nt)$ 有界，而 $\{t\sin(2\pi nt)\}$ 在 $t\approx1$ 附近两两距离不小于一个正常数，无收敛子列。

**题 17**：$A$ 紧且 $A^2=A$。求证 $\dim R(A)<\infty$。
*提示*：幂等给 $R(A)=N(I-A)$ **闭**（连续算子的核闭，FA 06）；$A$ 在 $R(A)$ 上是恒等（$Ax\in R(A)$，$A(Ax)=Ax$）。于是 $R(A)$ 的单位闭球 $=A\big(R(A)\cap\overline B\big)\subset A\big(\overline B\big)$ 相对紧且闭于 $R(A)$ ⟹ **紧**——FA 05 定理 2.3.3 判 $\dim R(A)<\infty$。**紧幂等 = 有限维投影**：无穷维空间上的"投影到无穷维子空间"永远不紧——这解释了为什么 GNN/滤波器的"无限容量投影"必须配有限秩截断。

**题 20**：$T\in B(H)$。求证 $T$ 是正交投影 $\iff$ $T$ 自伴且 $T^2=T$。
*提示*：（⟹）FA 07 投影三件套。（⟸）$R(T)=N(I-T)$ 闭；$x-Tx\in N(T)$（幂等）且 $\langle Tx,x-Tx\rangle=\langle Tx,x\rangle-\langle Tx,Tx\rangle=\langle T^2x,x\rangle-\|Tx\|^2=0$（自伴把 $\langle Tx,Tx\rangle=\langle T^*Tx,x\rangle=\langle T^2x,x\rangle=\langle Tx,x\rangle$）——**分解 $x=Tx+(x-Tx)$ 是正交分解**，$T$ 恰是把 $x$ 投到 $R(T)$ 的那个投影。∎（对照题 8/FA 13 题 8：一般幂等算子 $\sigma=\{0,1\}$ 但两块可以斜交——**加上自伴性才把斜投影掰正**；对称化 $T=\frac{T+T^*}2$ 是 ML 里把"斜投影"换成"正交投影"的标准修正。）

**题 22**：$H\ne\{0\}$，$A$ 自伴紧。求证 $\sigma_p(A)\ne\varnothing$。
*提示*：$A=0$ 时任何 $x\ne0$ 是特征向量（$\lambda=0$）。$A\ne0$ 时 $\|A\|=\sup_{\|x\|=1}|\langle Ax,x\rangle|>0$（定理 5.3.3）给 $M_A\ne0$ 或 $m_A\ne0$；**定理 5.3.6** 给该端点 $\in\sigma(A)$；非零谱点 + **定理 5.2.4(1)** 给它 $\in\sigma_p(A)$。∎（三行串起本篇两大定理——也是谱定理证明的第一步；对照：一般自伴算子可以 $\sigma_p=\varnothing$（乘法算子），**紧性供给特征值**。）

**题 23**：$\ell^2$ 上对角算子 $A(x_1,x_2,\dots)=(\lambda_1x_1,\lambda_2x_2,\dots)$，$\lambda_i\in\mathbb{R}$ 有界，$a=\inf\lambda_i$、$b=\sup\lambda_i$。求证 $\lambda_i\in\sigma_p(A)$；何时 $\sigma(A)=[a,b]$？
*提示*：$e_i$ 是特征向量。**谱 = 特征值集的闭包**：$\lambda\notin\overline{\{\lambda_i\}}$ 时 $\delta=\inf_i|\lambda-\lambda_i|>0$，逐坐标除法可逆（$\|R\|\le\frac1\delta$，FA 13 例 5.1.2 同款）；反之 $\overline{\{\lambda_i\}}\subset\sigma$（谱闭）。故
$$
\sigma(A)=\overline{\{\lambda_i:n\ge1\}},\qquad \sigma(A)=[a,b]\iff\{\lambda_i\}\ \text{在}\ [a,b]\ \text{中稠密}.
$$
样本：$\lambda_i$ 取遍 $[0,1]$ 中有理数 ⟹ $\sigma=[0,1]$ 而 $\sigma_p$ 可数——**连续谱段 $[0,1]\setminus\mathbb{Q}$ 全部由聚点织成**（每个无理点是可数特征值列的极限，自己不是特征值）。这是"自伴不紧 ⟹ 谱可以连续"的最干净标本（此 $A$ 紧 ⟺ $\lambda_i\to0$ 型衰减，稠密于区间时永不紧——两种谱形状在此分道）。

---

***系列完结**。十四篇，从 FA 01 的四条度量公理出发，经完备性与紧性（02）、不动点（03）、赋范结构（04）、有限维分水岭（05）、算子与对偶（06）、内积几何（07）、正交基与 Riesz（08）、四大定理（09–11）、逼近论（12）、谱论（13），到今天的谱定理——**一条主线贯穿始终：结构换定理**——每加一层公理（完备、线性、范数、内积、自伴、紧），就兑换一批更强的存在/唯一/结构结论。泛函分析的地图至此完整；回到[系列第一篇](../FA_01_metric_spaces/)再走一遍，路上的每个概念如今都有了归宿。*
