---
title: "线性代数 11: 对称矩阵与正定矩阵"
date: 2026-11-11
math: true
tags:
  - Notes
  - LinearAlgebra
categories:
  - LinearAlgebra
description: "Strang 第 5 版 §6.4-6.5 精读——谱定理 S=QΛQᵀ、实特征值与正交特征向量的两个证明、特征值与主元符号配对、Schur 分解、正定矩阵五重判据（主元、左上角行列式、特征值、能量 xᵀSx、AᵀA）、Cholesky 与三种平方和、半正定、椭圆与主轴定理、Hessian 极小值判定、Schur 补与 -1,2,-1 矩阵的谱"
---

# 线性代数 11：对称矩阵与正定矩阵

> 📖 本篇覆盖 Strang《Introduction to Linear Algebra》第 5 版 §6.4–6.5，是 Chapter 6 的收官篇。前置：[LA 10](../LA_10_eigenvalues_diagonalization/)（特征值、对角化、AM vs GM、矩阵指数）。

---

## 1. 这一篇解决什么问题？

LA_10 的结尾是一张警告牌：可对角化需要 $n$ 个无关特征向量，而重根可能短缺——$[[5,1],[0,5]]$ 的 5,5 病灶仍历历在目。本篇宣布一个好消息：**只要矩阵对称，警告牌立刻撤掉**。Strang 的原话毫不夸张：对称矩阵是理论与应用中**全世界最重要的矩阵**（the most important matrices the world will ever see）。两个问题：

1. **$S=S^T$ 到底买到了什么？**（§6.4）——三件套保票：**特征值全是实数**、**特征向量可以选成标准正交**（orthonormal）、**永远可对角化**（重根也不短缺）。对角化从 $A=X\Lambda X^{-1}$（逆矩阵伺候）升级为
$$
\boxed{\ S=Q\Lambda Q^T\ }\qquad(Q^TQ=I,\ \Lambda\ \text{实对角}),
$$
这就是**谱定理**（Spectral Theorem）——数学里的名字；几何与物理里叫**主轴定理**（Principal Axis Theorem），§8 的椭圆会解释为什么。本篇给出 $\lambda$ 全实与特征向量正交的**两个完整证明**，外加一个隐藏彩蛋：**特征值与主元符号配对**——LA_04 的消元世界与 LA_10 的特征值世界在此会师；

2. **正定（positive definite）怎么快速判定、有什么用？**（§6.5）——全部特征值 $>0$ 的对称矩阵叫正定矩阵，它站在无数应用的正中央（协方差、Hessian、图拉普拉斯、核矩阵）。但**算特征值是苦役**——本篇的目标是绕开它：五个两两等价的判据（主元、左上角行列式、特征值、能量 $x^TSx$、$A^TA$），**验一个即可定乾坤**。

暗线：LA_07 证明的 $A^TA$ 定理（列无关 $\iff$ $A^TA$ 可逆）在本篇升级——列无关时 $A^TA$ 不只可逆，而是**正定**；LA_04 的 $LDL^T$、LA_08 的 Cholesky 纽带（$A^TA=R^TR$）在本篇全部转正。

---

## 2. 谱定理：S=QΛQᵀ（§6.4）

### 2.1 从一个老朋友开始

$$
S=\begin{bmatrix}1&2\\2&4\end{bmatrix}=S^T
\qquad(\text{LA 05/07 的常客}),\qquad
\det(S-\lambda I)=\lambda^2-5\lambda.
$$

$\lambda=0$ 与 $\lambda=5$（都实）。$\lambda=0$ 不用算——$S$ 奇异（第二行 = 2×第一行）；$\lambda=5$ 也不用算——trace $=1+4=5$ 配平 $\lambda$ 之和。两个特征向量：

$$
\lambda=0:\ (2,-1)\ \ (\text{零空间}),\qquad
\lambda=5:\ (1,2)\ \ (\text{列空间}).
$$

**它们垂直**（$2-2=0$ ✓）——为什么？零空间的特征向量自然在 $N(S)$ 里；$\lambda=5\neq0$ 的特征向量在 $C(S)$ 里（$Sx=5x$ 是列的组合）。FTLA 说 $N(S)\perp C(S^T)$（零空间 ⊥ **行**空间）——但 $S$ 对称，**行空间 = 列空间**！于是零空间 ⊥ 列空间，两个特征向量被迫垂直。除以长度 $\sqrt5$ 排进 $Q$：

$$
Q=\frac{1}{\sqrt5}\begin{bmatrix}2&1\\-1&2\end{bmatrix},
\qquad
Q^{-1}SQ=Q^TSQ=\begin{bmatrix}0&0\\0&5\end{bmatrix}=\Lambda.
$$

### 2.2 谱定理

> **谱定理**：每个实对称矩阵都有分解 $S=Q\Lambda Q^T$——$\Lambda$ 里是 $n$ 个**实**特征值，$Q$ 的列是 $n$ 个**标准正交**特征向量。

三个立刻的注解：

- **"choose" 的含义**：特征向量本身不是单位向量——长度由我们**选**（除以 $\|x\|$ 即可）；选成单位长，正交（orthogonal）升级为正交规范（orthonormal），$X$ 就换成专属字母 $Q$；
- **$Q\Lambda Q^T$ 自动对称**：转置得 $(Q^T)^T\Lambda^TQ^T=Q\Lambda Q^T$——原样回来。难的方向是反过来（每个对称阵都有这副模样），必须证明，见 §3；
- **展开成秩一之和**（§7 会用到）：

$$
S=Q\Lambda Q^T=\lambda_1q_1q_1^T+\lambda_2q_2q_2^T+\cdots+\lambda_nq_nq_n^T.
$$

每个对称矩阵是 $n$ 个**正交投影** $\boldsymbol{q_iq_i^T}$（LA_07 的秩一投影！）的加权和——权重恰是特征值。

---

## 3. 两个证明：λ 全实、特征向量正交

### 3.1 实特征值：共轭两连击

设 $Sx=\lambda x$，$S$ 实对称。在证明之前，$\lambda$ 完全可能是复数 $a+ib$，$x$ 的分量也可能是复数。**取共轭**（$\bar\lambda=a-ib$，$\bar x$ 逐分量翻转虚部）：因为 $S$ 是**实**矩阵，共轭穿过 $S$：

$$
S\bar x=\bar\lambda\bar x
\qquad\Longrightarrow\qquad
\text{转置：}\ \bar x^TS=\bar\lambda\,\bar x^T\ (\text{用了}\ S^T=S).
$$

现在两连击——第一式左乘 $\bar x^T$、第二式右乘 $x$，**左边是同一个数** $\bar x^TSx$：

$$
\bar x^TSx=\lambda\,\bar x^Tx,
\qquad
\bar x^TSx=\bar\lambda\,\bar x^Tx.
$$

相减：$(\lambda-\bar\lambda)\,\bar x^Tx=0$。关键在 $\bar x^Tx=|x_1|^2+|x_2|^2+\cdots$——**长度的平方，恒正**（$x\neq\mathbf 0$）。于是 $\lambda=\bar\lambda$，即 $b=0$，$\lambda$ 是实数。∎

特征向量随后解实方程 $(S-\lambda I)x=\mathbf 0$（系数全实）即可取实解——对称矩阵的整个特征世界都是实的。**对照**（LA_10 §4）：非对称实矩阵的复 $\lambda$ 成**共轭对**出现（旋转矩阵 $\lambda=\cos\theta\pm i\sin\theta$、特征向量 $(1,\mp i)$）；对称性恰好把虚部掐死。

### 3.2 不同特征值的特征向量正交：两行证明

设 $Sx=\lambda_1x$，$Sy=\lambda_2y$，$\lambda_1\neq\lambda_2$。第一式左乘 $y^T$，第二式左乘 $x^T$ 再转置：

$$
\lambda_1x^Ty=(Sx)^Ty=x^TS^Ty=x^TSy=x^T\lambda_2y.
$$

（中间一步就是 $S^T=S$ 的唯一出场。）于是 $(\lambda_1-\lambda_2)\,x^Ty=0$，而 $\lambda_1\neq\lambda_2$，故 $x^Ty=0$。∎——与 LA_10 的"$AB$ vs $BA$"证明同款节奏：**同一个量算两遍，逼出系数为零**。

**Example 2**（教材注：not widely known）：$2\times2$ 对称阵 $S=\begin{bmatrix}a&b\\b&c\end{bmatrix}$ 的特征向量有闭式

$$
x_1=\begin{bmatrix}\lambda_1-c\\ b\end{bmatrix},\qquad
x_2=\begin{bmatrix}b\\ \lambda_2-a\end{bmatrix},
$$

（验证 $x_1$：第二行 $b(\lambda_1-c)+(c-\lambda_1)b=0$ ✓。）内积一行清零：

$$
x_1^Tx_2=b(\lambda_1-c)+(\lambda_2-a)b=b\underbrace{(\lambda_1+\lambda_2-a-c)}_{=\,\text{trace}-\text{trace}\,=\,0}=0.
$$

trace 又立功了。边界情形 $S=I$：$b=0$、$\lambda_1=\lambda_2=1$ 重根、公式给出零向量——但 $I$ 的特征向量当然可以选正交（$e_1,e_2$）。**重根时正交性不是自动的，是可选择的**——这正是 §4 要补的最后一环。

---

## 4. 特征值 vs 主元：符号配对与 Schur

### 4.1 到目前为止唯一的联系

LA_04（主元）与 LA_10（特征值）两个世界此前只有一根细线：**主元之积 = 行列式 = 特征值之积**。数值本身毫无关系——$[[2,0],[0,1]]$ 与 $[[1,1],[1,2]]$... 但对称矩阵藏着一条强得多的定律：

> **符号配对**（对称矩阵）：$S=S^T$ 的**正特征值个数 = 正主元个数**。特例：全部 $\lambda_i>0\iff$ 全部主元 $>0$（§6.5 正定性的地基）。

**Example 4**：
$$
S=\begin{bmatrix}1&3\\3&1\end{bmatrix}:
\quad\text{主元 }1,\ -8;\qquad
\text{特征值 }4,\ -2
\quad(\text{trace }2,\ \det=-8\ \text{两边都对}).
$$

一正一负，符号精确配对。**非对称时立刻失效**：$B=\begin{bmatrix}1&-2\\3&-4\end{bmatrix}$ 主元 $1,\ 2$ 全正，特征值却是 $-1,\ -2$（trace $=-3$、$\det=2$ 验算 ✓）——对称性是符号配对的必要条件。

### 4.2 连续性证明：LDLᵀ 里把 L 推向 I

对称消元（LA_04）给 $S=LDL^T$——对 Example 4：

$$
\begin{bmatrix}1&3\\3&1\end{bmatrix}
=\begin{bmatrix}1&0\\3&1\end{bmatrix}
\begin{bmatrix}1&0\\0&-8\end{bmatrix}
\begin{bmatrix}1&3\\0&1\end{bmatrix}
\qquad(S=LDL^T).
$$

**证明思想**：把 $L$ 的非对角元连续地推向 $0$，矩阵从 $LDL^T$ 连续变形到 $IDI^T=D$。过程中**主元全程不变**（$D$ 不动），所以矩阵**永不满秩**——特征值想变号就必须穿过 $0$，而穿过 $0$ 的那一刻矩阵奇异。矛盾。于是特征值从 $\lambda_i$（起点）一路滑到 $d_i$（终点）都**无法变号**：起点 $\lambda$ 的符号分布 = 终点主元的符号分布。∎

一行总结：**$\lambda$ 们不能越过 $0$，因为消元的世界里没有 $0$ 主元**。这条证明连接了应用线性代数的两半——主元与特征值。

### 4.3 Schur：每个方阵都能三角化，对称时自动对角化

§3.2 只证明了**不同** $\lambda$ 的特征向量正交；重根怎么办（LA_10 的 $[[5,1],[0,5]]$ 噩梦）？补刀来自一个对所有方阵成立的新分解：

> **Schur 分解**：每个方阵 $A=QTQ^{-1}$，其中 $T$ 上三角。若 $A$ 的特征值全实，$Q$ 与 $T$ 可取实正交（$Q^{-1}=Q^T$）。

对对称 $S$ 使用：$T=Q^TSQ$，转置 $T^T=Q^TS^TQ=Q^TSQ=T$——**上三角且对称 $\Rightarrow$ 必是对角阵**（$T$ 的 $(1,3)$ 元在三角里是零、在对称里等于 $(3,1)$ 元也是零，逐个剿灭）。于是 $T=\Lambda$，$S=Q\Lambda Q^T$——**重根也挡不住**：对称矩阵永远有一套完整的正交特征向量。∎

对照 LA_10：可对角化对一般矩阵是**运气**（$GM=AM$），对对称矩阵是**保票**。最后一块拼图来自 §7.2（LA_12 再详述）：$\lambda_1$ 可以**逐个**刻画——

$$
\lambda_1=\max_x\frac{x^TSx}{x^Tx},\qquad
\lambda_2=\text{同式}\max\ \text{但限制}\ x\perp q_1,\quad\dots
$$

**Rayleigh 商**（Rayleigh quotient）的最大值恰是最大特征值——谱定理的第三种证明路线，也是 PCA 与幂法的接口（§12）。

---

## 5. 对称世界的两个明星例（Worked 6.4 A/B）

### 5.1 Worked 6.4 A：由谱反推矩阵——一切皆可预知

**问题**：哪个矩阵 $A$ 有 $\lambda=1,-1$ 与正交特征向量 $x_1=(\cos\theta,\sin\theta)$、$x_2=(-\sin\theta,\cos\theta)$？以下性质哪些能**提前**预言：$A=A^T$？$A^2=I$？$\det A=-1$？主元一正一负？$A^{-1}=A$？

**答案：全部可以**。$x_1,x_2$ 正交规范，$A=Q\Lambda Q^T$ 自动对称；$\lambda^2=1$ 给 $A^2=I$ 与 $A^{-1}=A$；$\det=\lambda_1\lambda_2=-1$；对称 + 特征值一正一负 ⟹ 主元一正一负（§4）。实算（$c=\cos\theta,\ s=\sin\theta$）：

$$
A=Q\Lambda Q^T=
\begin{bmatrix}c&-s\\s&c\end{bmatrix}
\begin{bmatrix}1&0\\0&-1\end{bmatrix}
\begin{bmatrix}c&s\\-s&c\end{bmatrix}
=
\begin{bmatrix}\cos2\theta&\sin2\theta\\ \sin2\theta&-\cos2\theta\end{bmatrix}.
$$

**这是一面镜子**：沿 $\theta$ 线的分量不动（$\lambda=1$）、垂直分量翻转（$\lambda=-1$）。检验：$(1,0)\mapsto(\cos2\theta,\sin2\theta)$——反射把辐角 $0$ 送到 $2\theta$，角度翻倍正是镜面反射的定义（LA_10 §2 的反射家族 $R=2P-I$ 在此具体化）。**谱定理是双向的**：知道 $Q,\Lambda$ 能造出 $S$；知道 $S$ 的谱信息能反推全部矩阵性质。

### 5.2 Worked 6.4 B：正弦与余弦——DST 与 DCT 的诞生

LA_10 的 Worked 6.3 B 见过 $-1,2,-1$ 三对角家族（$-K$）。现在看正号版 $A_3$ 与它的自由端兄弟 $B_4$：

$$
A_3=\begin{bmatrix}2&-1&0\\-1&2&-1\\0&-1&2\end{bmatrix},
\qquad
B_4=\begin{bmatrix}1&-1&0&0\\-1&2&-1&0\\0&-1&2&-1\\0&0&-1&1\end{bmatrix}.
$$

$-1,2,-1$ 是**二阶差分**——二阶导数的离散化身（LA_06 §3.4 已预告）。二阶导数的特征函数是 $\sin,\cos$，所以：

| 矩阵 | 边界 | 特征值 | 特征向量 |
|------|------|--------|---------|
| $A_3$（两端固定） | Dirichlet | $2-\sqrt2,\ 2,\ 2+\sqrt2$ | **离散正弦**（sine 曲线上的采样点） |
| $B_4$（两端自由） | Neumann | $0,\ 2-\sqrt2,\ 2,\ 2+\sqrt2$ | **离散余弦**（cos 在半格点 $\pi/8,3\pi/8,5\pi/8,7\pi/8$ 采样） |

验算 $A_3$：和 $=6$ = trace ✓、积 $=4$ = $\det$ ✓（$B_4$ 呢？trace 仍是 $6$，但 $\det=0$——多出的 $\lambda=0$ 配特征向量 $(1,1,1,1)$：自由端允许全体平移，能量为零，§8 半正定的标本）。$B_4$ 的特征向量矩阵就是 **DCT**（Discrete Cosine Transform）——JPEG 图像压缩的标准选型正是 $B_8$。三对角对称矩阵的小世界里藏着整个数字信号处理：**固定边界出正弦、自由边界出余弦**，而两者的统一理论（Fourier）在 LA_14。

---

## 6. 正定矩阵：五重判据（§6.5 核心）

### 6.1 从 2×2 开始：不想算特征值

**目标**：判定对称 $S$ 的 $\lambda$ 是否全正，但**不真的去算**（特征值计算是 $O(n^3)$ 的苦役，而下面的判据只是消元——快得多）。$2\times2$：

$$
S=\begin{bmatrix}a&b\\b&c\end{bmatrix}:\qquad
\lambda_1,\lambda_2>0\iff a>0\ \text{且}\ ac-b^2>0.
$$

三连例（教材）：

| 矩阵 | $a$ | $\det=ac-b^2$ | 判定 | 特征值 |
|------|-----|---------------|------|--------|
| $S_1=\begin{bmatrix}1&2\\2&1\end{bmatrix}$ | $1>0$ | $1-4=-3<0$ | ❌ 不定 | $3,\ -1$（trace 2、det $-3$） |
| $S_2=\begin{bmatrix}1&2\\2&6\end{bmatrix}$ | $1>0$ | $6-4=2>0$ | ✅ 正定 | 皆正（trace 7、det 2） |
| $S_3=-S_2=\begin{bmatrix}-1&-2\\-2&-6\end{bmatrix}$ | $-1<0$ | $2>0$ | ❌ 负定 | 皆负 |

$S_3$ 是最重要的警示：**$\det>0$ 单独不算数**（$S_3$ 的行列式也是 $+2$！）——必须从 $a>0$ 开始逐级往上验。**主元版**判据同源：第一主元 $a$，第二主元 $c-\frac{b^2}{a}=\frac{ac-b^2}{a}$——**每个主元 = 相邻两个左上角行列式之比**（LA_04 的公式在此复活），$a>0$ 且 $ac-b^2>0$ 恰好让两个主元全正。

### 6.2 五重判据（本篇的中心定理）

> **定理**：对称 $S$ 满足以下任一条，则五条全满足——
>
> 1. 全部 $n$ 个**主元**为正；
> 2. 全部 $n$ 个**左上角行列式**为正（$1\times1,\ 2\times2,\dots,n\times n$）；
> 3. 全部 $n$ 个**特征值**为正；
> 4. **能量** $x^TSx>0$ 对一切 $x\neq\mathbf0$；
> 5. $S=A^TA$，其中 $A$ **列无关**。

这就是教材所说 ties together the whole course 的一条定理：主元（Ch 2 消元）、行列式（Ch 5）、特征值（Ch 6 前三节）、能量与 $A^TA$（Ch 4 正交/最小二乘）——**四大主线在一个定义上会师**。1↔2 靠主元=行列式之比；2↔3 是 2×2 论证的逐级推广（各级 det = 特征值的部分积）；3↔4、4↔5 见 §7。

### 6.3 一个 3×3 的全检验：I + J₃

$$
S=\begin{bmatrix}2&1&1\\1&2&1\\1&1&2\end{bmatrix}=I+J_3
\qquad(J_3=\text{全 }1\text{ 矩阵},\ \text{LA 10 的 } 5I-J_4\ \text{近亲}).
$$

| 判据 | 数值 | 结论 |
|------|------|------|
| 左上角行列式 | $2,\ 3,\ 4$ | 全正 ✓ |
| 主元（= 行列式之比） | $2,\ 3/2,\ 4/3$ | 全正 ✓ |
| 特征值（$J_3$ 的 $3$ 配 $e=(1,1,1)$，其余 $1$） | $1,\ 1,\ 4$ | 全正 ✓ |

三个加练（LA_10 的移位技巧：$S-cI$ 平移全部特征值）：

- $S-I=J_3$：特征值 $0,0,3$——**半正定**（边缘情形，§8）；
- $S-2I$：特征值 $-1,-1,2$——**不定**（indefinite）；
- $I+J_3$ 本身：正定。

同一个矩阵家族一次演示三种定性。**任何一条判据都是决定性的**——实战里通常挑最快的（消元得主元）。

---

## 7. 能量视角：xᵀSx 与三种平方和

### 7.1 能量定义与两条免费定理

从 $Sx=\lambda x$ 两边左乘 $x^T$：$x^TSx=\lambda\|x\|^2$——特征向量上的能量是 $\lambda$ 倍长度平方。但**真正的定义走得更远**：

> **定义**：$S$ 正定 $\iff x^TSx>0$ 对**一切** $x\neq\mathbf 0$（不只特征向量）。物理与工程里 $x^TSx$（或 $\tfrac12x^TSx$）是系统的**能量**（energy）——正能量才允许稳定的平衡。Strang 的立场：这是五重判据里**最根本**的一条。

$2\times2$ 展开看清结构（四个矩阵元给四项）：

$$
x^TSx=\begin{bmatrix}x&y\end{bmatrix}
\begin{bmatrix}a&b\\b&c\end{bmatrix}
\begin{bmatrix}x\\y\end{bmatrix}
=ax^2+2bxy+cy^2.
$$

对角元 $a,c$ 出纯平方项、一对 $b$ 出交叉项 $2bxy$。能量视角立刻送两条定理：

- **$S+T$ 正定**（若 $S,T$ 都正定）：$x^T(S+T)x=x^TSx+x^TTx>0$——**能量直接相加**。主元与特征值在矩阵相加时全乱了套，能量不会；这就是能量定义的统治力（凸优化里凸函数相加仍凸的根源）；
- **$S=A^TA$ 正定**（若 $A$ 列无关）：括号的力量，
$$
x^TA^TAx=(Ax)^T(Ax)=\|Ax\|^2>0
\qquad(x\neq\mathbf0\Rightarrow Ax\neq\mathbf0\ \text{即列无关}).
$$

**LA_07 的 $A^TA$ 定理升级了**：列无关不只保证 $A^TA$ 可逆——它保证 $A^TA$ **正定**（可逆只是 $\lambda\neq0$，正定是 $\lambda>0$）。

### 7.2 同一个 S 的三个 A：三种平方和

取 §5 的 $A_3$（$-1,2,-1$ 三对角）作为 $S$——消元给出主元 $2,\ 3/2,\ 4/3$、乘数 $-\tfrac12,\ 0,\ -\tfrac23$：

$$
S=LDL^T=
\begin{bmatrix}1&0&0\\-\tfrac12&1&0\\0&-\tfrac23&1\end{bmatrix}
\begin{bmatrix}2&0&0\\0&\tfrac32&0\\0&0&\tfrac43\end{bmatrix}
\begin{bmatrix}1&-\tfrac12&0\\0&1&-\tfrac23\\0&0&1\end{bmatrix}.
$$

教材给出**三个**不同的 $A$ 让 $S=A^TA$，对应能量 $x^TSx=2x_1^2-2x_1x_2+2x_2^2-2x_2x_3+2x_3^2$ 的**三种平方和分解**：

| $A$ | 形状与身份 | $\|Ax\|^2$（= $x^TSx$） |
|-----|-----------|------------------------|
| $A_1=\begin{bmatrix}1&0&0\\-1&1&0\\0&-1&1\\0&0&1\end{bmatrix}$ | $4\times3$ **一阶差分**（列无关） | $x_1^2+(x_2-x_1)^2+(x_3-x_2)^2+x_3^2$ |
| $A_2=\sqrt D\,L^T$ | $3\times3$ 上三角 = **Cholesky 因子** | $2(x_1-\tfrac12x_2)^2+\tfrac32(x_2-\tfrac23x_3)^2+\tfrac43x_3^2$ |
| $A_3=Q\sqrt\Lambda\,Q^T$ | $3\times3$ 对称 = **对称平方根** | $\lambda_1(q_1^Tx)^2+\lambda_2(q_2^Tx)^2+\lambda_3(q_3^Tx)^2$ |

三种分解三种口味：$A_1$ 的能量是**相邻差分**（弹簧链的势能——相邻节点位移差的平方）；$A_2$ 的系数是**主元**（逐项配方，LA_04 的 complete the square）；$A_3$ 的系数是**特征值**（沿正交特征方向的分解，§8 椭圆的正轴版本）。三种平方和恒等——一个正定矩阵，三种"看见"正定的方式。

**Cholesky 分解**正式登场：

$$
\boxed{\ S=C^TC\ }\qquad(C=\sqrt D\,L^T\ \text{上三角，对角元}=\sqrt{\text{主元}}),
$$

MATLAB 命令 `chol(S)`。LA_08 的纽带在此闭环：$A^TA=R^TR$（Gram-Schmidt 作用在 $A$ 上 = 对 $A^TA$ 做 Cholesky）。Strang 的一句话总结分工：**长方形 $A_1$ 是我们构造 $S$ 的方式，三角 $A_2$ 是我们拆解 $S$ 的方式**（build vs break）。

### 7.3 边角的力量：b 不能太大

把 $S$ 的 $(1,3),(3,1)$ 零元改成 $b$：$T=\begin{bmatrix}2&-1&b\\-1&2&-1\\b&-1&2\end{bmatrix}$。左上角行列式 $2,\ 3$ 纹丝不动，第三级呢？

$$
\det T=4+2b-2b^2=(1+b)(4-2b)>0
\iff -1<b<2.
$$

角落的一个元素就能摧毁正定性——**正定是全局性质，不是对角线的私人事务**。（$b=-1$ 的循环版 $T$ 正是 §8 半正定的主角。）

---

## 8. 半正定、椭圆与极小值

### 8.1 半正定：站在边缘上

**正半定**（positive semidefinite，PSD）：把 $>$ 放宽为 $\ge$——$\lambda_i\ge0$、主元 $\ge0$、行列式 $\ge0$、能量 $x^TSx\ge0$。边缘上发生什么：最小特征值为 $0$、$\det=0$、对应的特征向量上**能量恰为零**。两个标本：

$$
S=\begin{bmatrix}1&2\\2&4\end{bmatrix}
\quad(\lambda=5,0;\ \text{左上角行列式 }1,0;\ \text{秩 }1),
\qquad
T=\begin{bmatrix}2&-1&-1\\-1&2&-1\\-1&-1&2\end{bmatrix}
\quad(\lambda=0,3,3).
$$

- $S=A^TA$ 仍成立，但 $A$ 的列**相关**：$A=\begin{bmatrix}1&0\\2&0\end{bmatrix}$，第二列是零——零空间方向上能量为零。把 $4$ 加大任何一个 $\varepsilon>0$（$S=\begin{bmatrix}1&2\\2&4+\varepsilon\end{bmatrix}$，$\det=\varepsilon>0$）就跳回正定——**半正定是正定的边界**；
- 循环 $T$（$\S7.3$ 的 $b=-1$）：特征向量 $(1,1,1)$ 上 $Tx=\mathbf 0$、能量为零——**全体平移不消耗能量**（自由弹簧链的刚体模式，§5 的 $B_4$ 同款）；$T=A^TA$ 的 $A$ 只能是**循环差分** $A=\begin{bmatrix}1&-1&0\\0&1&-1\\-1&0&1\end{bmatrix}$，$(1,1,1)$ 落在 $N(A)$ 里。

### 8.2 椭圆与主轴定理

**Example 2**：斜着的椭圆 $5x^2+8xy+5y^2=1$。左边的二次型配对称矩阵：

$$
5x^2+8xy+5y^2=\begin{bmatrix}x&y\end{bmatrix}
\begin{bmatrix}5&4\\4&5\end{bmatrix}
\begin{bmatrix}x\\y\end{bmatrix},
\qquad
S=\begin{bmatrix}5&4\\4&5\end{bmatrix}
$$

（交叉项 $8xy$ 平摊给 $(1,2)$ 与 $(2,1)$ 各一半——**永远先把二次型写成对称形式**。）$S$ 的谱：$\lambda=9,1$，特征向量 $(1,1)/\sqrt2,\ (1,-1)/\sqrt2$。代入 $S=Q\Lambda Q^T$，定义新坐标 $X=\frac{x+y}{\sqrt2},\ Y=\frac{x-y}{\sqrt2}$：

$$
x^TSx=9X^2+Y^2=1
\qquad\Longrightarrow\qquad
\text{半轴长}=\frac{1}{\sqrt{\lambda_1}}=\frac13\ \text{（短轴）},\quad
\frac{1}{\sqrt{\lambda_2}}=1\ \text{（长轴）}.
$$

**主轴定理的几何**：椭圆的**轴指向特征向量方向**、**半轴长 $=1/\sqrt\lambda$**——所以**大特征值配短轴**（$9$ 配 $1/3$）。斜椭圆（$S$ 的世界）转正（$\Lambda$ 的世界）的旋转矩阵正是 $Q$。三点注意：

- 平方和的系数是**特征值**（9 与 1），不是主元（5 与 9/5）——两种分解别混（§7 表格的第三行 vs 第二行）；
- 一个 $\lambda<0$：$9X^2-Y^2=1$，椭圆变**双曲线**；两个都负（$-I$）：$-x^2-y^2=1$ 无点；$S=I$：正圆。**特征值的符号决定二次曲面的类型**；
- $n\times n$ 正定 $S$：$x^TSx=1$ 是 $\mathbb R^n$ 的**椭球**（ellipsoid），轴沿特征向量——LA_12 的 PCA 就是拿它切数据。

### 8.3 极小值判定：微积分的矩阵化

一元微积分：$f'(x)=0$ 且 $f''(x)>0$ ⟹ 极小。多元版本把二阶导数攒成矩阵：

$$
\text{梯度归零：}\ \frac{\partial F}{\partial x}=\frac{\partial F}{\partial y}=0,
\qquad
\text{二阶导数矩阵（Hessian）：}\
S=\begin{bmatrix}F_{xx}&F_{xy}\\F_{yx}&F_{yy}\end{bmatrix}
\ \text{（对称——混合偏导相等）}
$$

$$
F\ \text{在梯度零点处取极小}\iff S\ \text{正定}.
$$

道理：梯度零点附近 $F\approx$ 常数 $+\ ax^2+2bxy+cy^2$（$F$ 的二阶导数恰是 $2a,2b,2b,2c$）——**极小值问题就是二次型正定性问题**。三个示范：

| 函数 | 梯度零点 | Hessian | 定性 |
|------|---------|---------|------|
| $F_2=x^3+xy-x$ | $(0,1)$（$F_x=3x^2+y-1,\ F_y=x$） | $\begin{bmatrix}0&1\\1&0\end{bmatrix}$，$\lambda=\pm1$ | **鞍点**（saddle） |
| $z=4x^2+12xy+cy^2$ | $(0,0)$ | $\begin{bmatrix}4&6\\6&c\end{bmatrix}$，$\det=4c-36$ | $c>9$ 碗、$c<9$ 鞍、$c=9$ 抛物谷 |
| $f=x^2+4xy+3y^2$ | $(0,0)$ | $\begin{bmatrix}1&2\\2&3\end{bmatrix}$，$\det=-1$ | 鞍：$f=(x+2y)^2-y^2$，点 $(0,1)$ 处 $f=-1<0$ |

第三行特别扎心：**系数全正不等于正定**——交叉项 $4xy$ 可以压倒一切。$c=9$ 的边界值也值得看一眼：$z=4x^2+12xy+9y^2=(2x+3y)^2$——半正定，图形是**抛物形山谷**（沿直线 $2x+3y=0$ 全为零、不升不降），介于碗与鞍之间。深度学习在高维损失面上"找最小值难、找鞍点易"的困境，矩阵语言就是：随机 Hessian 的负特征值个数随维度增长（§12）。

---

## 9. Schur 补与 -1,2,-1 家族的谱（Worked 6.5 B/C）

### 9.1 Worked 6.5 B：块矩阵的正定性

LA_03 的 Schur 补回来了。对称块矩阵 $M=\begin{bmatrix}A&B\\B^T&C\end{bmatrix}$（$A$ 可逆）何时正定？对块做消元——第二行块减去 $B^TA^{-1}$ 倍第一行块：

$$
\begin{bmatrix}I&0\\-B^TA^{-1}&I\end{bmatrix}
\begin{bmatrix}A&B\\B^T&C\end{bmatrix}
\begin{bmatrix}I&-A^{-1}B\\0&I\end{bmatrix}
=
\begin{bmatrix}A&0\\0&C-B^TA^{-1}B\end{bmatrix}.
$$

$$
\boxed{\ M\ \text{正定}\iff A\ \text{正定且}\ S=C-B^TA^{-1}B\ \text{正定}\ }
$$

（$S$ 就是 **Schur 补**——LA_03 的 $A=LU$ 分块版；两侧乘的是互为转置的块三角阵，合同变换保正定性。）工程里 $M$ 常是刚度矩阵：$A$ 是保留的自由度、$C$ 是要消去的——**消元从未离开**，只是搬进了块世界。

### 9.2 Worked 6.5 C：猜出 -1,2,-1 的全部特征对

**Strang 自称的最爱**：$n\times n$ 三对角 $-1,2,-1$ 矩阵 $S$ 的特征值是什么？直接解 $\det(S-\lambda I)=0$ 是灾难。**猜**——猜的线索藏在微分方程里（§5 已铺）：$S$ 是二阶导数的离散版，而

$$
\frac{d^2y}{dx^2}=\lambda y,\qquad y(0)=y(1)=0
\qquad\Longrightarrow\qquad
y=\sin k\pi x,\quad \lambda=-k^2\pi^2.
$$

在 $n$ 个等分点 $x=h,2h,\dots,nh$（$h=\frac{1}{n+1}$，端点 $0$ 与 $1$ 是固定的墙）上采样正弦，得到 $S$ 的特征向量；特征值离散化为

$$
\boxed{\ \lambda_k=2-2\cos k\pi h,\qquad
x_k=(\sin k\pi h,\ \sin 2k\pi h,\ \dots,\ \sin nk\pi h),\qquad h=\tfrac{1}{n+1}\ }
$$

**验证 $n=3$**（$h=\tfrac14$）：$k=1,2,3$ 给 $\lambda=2-\sqrt2,\ 2,\ 2+\sqrt2$——与 §5 的 $A_3$ 完全一致 ✓；$x_1\propto(\sin\frac\pi4,\sin\frac\pi2,\sin\frac{3\pi}4)=(1,\sqrt2,1)$——与 LA_10 Worked 6.3 B 的特征向量 $(1,\sqrt2,1)$ 完全一致 ✓。**对所有 $n$**：$\lambda_k=2-2\cos\frac{k\pi}{n+1}>0$（$k\le n$ 时夹在 $(0,4)$ 内）——**整个 $-1,2,-1$ 家族全部正定**，而循环版（$b=-1$ 的 $T$）多出 $\lambda_0=0$ 掉到半正定。一行公式收割一整个矩阵家族；而"猜"的合法性来自 $\sin$ 是二阶差分的天然特征向量——**DST 的谱定理版证明**（§5）。

---

## 10. 例子与反例速查

| 命题 | 真假 | 例子 / 反例 |
|------|------|------------|
| 对称矩阵的特征值必为实数 | ✅ | §3.1 共轭两连击；$\begin{bmatrix}1&3\\3&1\end{bmatrix}$ 的 $4,-2$ |
| 对称矩阵重根必短缺特征向量 | ❌ | $S=I$（$\lambda=1,1$）或 $I+J_3$（$\lambda=1,1,4$）——Schur 保票 |
| 非对称矩阵可以有复特征值 | ✅ | 旋转矩阵 $\lambda=\cos\theta\pm i\sin\theta$，特征向量 $(1,\mp i)$ |
| 主元符号 = 特征值符号（无需对称） | ❌ | $B=\begin{bmatrix}1&-2\\3&-4\end{bmatrix}$：主元 $1,2$ 全正、$\lambda=-1,-2$ 全负 |
| 对角元全正 ⟹ 正定 | ❌ | $\begin{bmatrix}1&2\\2&1\end{bmatrix}$：对角 $1,1$，$\lambda=3,-1$ |
| $\det>0$ ⟹ 正定 | ❌ | $-I$（$\det=1$）或 $\begin{bmatrix}-1&-2\\-2&-6\end{bmatrix}$（$\det=2$，负定） |
| trace $>0$ ⟹ 正定 | ❌ | $\begin{bmatrix}1&2\\2&1\end{bmatrix}$：trace 2，$\det=-3$ |
| $a>0,\ c>0,\ a+c>2b$ ⟹ 二次型正定 | ❌ | $\begin{bmatrix}1&5\\5&24\end{bmatrix}$：$a+c=25>10$，但 $\det=24-25=-1$，$\lambda$ 一正一负 |
| 正定 ⟹ 可逆 | ✅ | $\lambda_i>0\Rightarrow\det=\prod\lambda_i>0$（反之不成立） |
| 正定的投影矩阵只有 $I$ | ✅ | 投影的 $\lambda\in\{0,1\}$，全正 ⟹ 全是 1 ⟹ $P=I$ |
| $S$ 正定 ⟹ $S^{-1}$ 正定 | ✅ | $S^{-1}=Q\Lambda^{-1}Q^T$，$\lambda^{-1}>0$ |
| $S,T$ 正定 ⟹ $S+T$ 正定 | ✅ | 能量相加：$x^T(S+T)x=x^TSx+x^TTx>0$ |
| $S,T$ 正定 ⟹ $ST$ 正定 | ❌ | $ST$ 一般**不对称**（但 $\lambda$ 仍全正，习题 8） |
| 对角阵对角全正 ⟹ 正定 | ✅ | $x^T\,\mathrm{diag}(d_i)\,x=\sum d_ix_i^2>0$ |
| 对称 + 正交的矩阵满足 $S^2=I$ | ✅ | $S^T=S^{-1}$ 与 $S^T=S$ 合并；$\lambda=\pm1$（反射家族） |
| 全部 $\lambda=2$ 的对称阵是 $2I$ | ✅ | $S=Q(2I)Q^T=2I$——正交相似下唯一（一般矩阵不成立） |
| $S$ 正定 ⟹ 每个 $s_{ii}\ge\lambda_{\min}$ | ✅ | $s_{ii}=e_i^TSe_i\ge\lambda_{\min}\|e_i\|^2$（Rayleigh 下界）；对角元是 0-1 向量的能量 |
| 特征值夹在对角元的最小最大之间 | ✅ | 上表的上界版本：$s_{ii}\le\lambda_{\max}$（LA_10 Gershgorin 的精化版） |

---

## 11. 易混淆辨析

| 概念对 | 辨析 |
|--------|------|
| **$A=X\Lambda X^{-1}$ vs $S=Q\Lambda Q^T$** | 一般对角化（$X$ 只需可逆、$\lambda$ 可复）vs 对称对角化（$Q$ 正交、$\lambda$ 全实、永不出错）。$Q^T$ 比 $X^{-1}$ 便宜又稳定——数值线性代数偏爱对称的理由 |
| **谱定理 vs 主轴定理** | 同一定理的两个名字：数学里叫 spectral theorem（矩阵分解视角），几何/物理里叫 principal axis theorem（椭圆转正视角） |
| **五重判据用哪个** | 主元：最快（一次消元）；行列式：$n$ 个数字逐级验；特征值：最贵，别用来判正定；能量：证明与理论的首选；$A^TA$：构造系（协方差、刚度、核矩阵都是它） |
| **主元 vs 特征值** | 数值不同（$2,\tfrac32,\tfrac43$ vs $2-\sqrt2,2,2+\sqrt2$）、**符号个数相同**、乘积同为 $\det$。别把平方和的系数写成主元——那是 Cholesky 版（系数=主元），谱版的系数是 $\lambda$ |
| **正定 vs 半正定** | $>$ vs $\ge$：边缘上 $\det=0$、$\lambda_{\min}=0$、零空间方向能量为零。半正定 + 微扰 $\varepsilon I$ 跳回正定（$S+\varepsilon I$ 特征值全加 $\varepsilon$） |
| **不定 vs 负定** | $\lambda$ 有正有负（双曲线、鞍点）vs 全负（$-S$ 正定；无点图）。$\det<0$ ⟹ 不定；$\det>0$ 要看 $a$ 的符号才能分辨正定/负定 |
| **$A^TA$ vs $A^TA$ 正定** | $A$ 任意：$A^TA$ 恒**半正定**（$\|Ax\|^2\ge0$）；$A$ 列无关：升级**正定**。LA_07 定理的完整版 |
| **对称 vs 反对称 vs 正交**（三家族谱） | $S^T=S$：$\lambda$ 全实；$K^T=-K$：$\lambda$ 全纯虚；$Q^TQ=I$：$\lvert\lambda\rvert=1$。三者都有一套正交（复）特征向量——LA_10 §4 的表在 Schur 光照下全部落实 |
| **相似 vs 合同** | 相似 $B=C^{-1}SC$ 保**特征值**；合同 $B=A^TSA$ 保**特征值符号的个数**（惯性定理，Law of Inertia）但数值可变。正定性在合同下不变——Schur 补证明用的正是合同 |
| **Hessian 正定 vs 函数凸** | 一点的 Hessian 正定 ⟹ 该点是局部极小；**处处**半正定 ⟹ 凸函数。二阶信息从点到全局的升级 |

---

## 12. ML / 应用连接

- **协方差矩阵 = 半正定**：$\Sigma=\frac1mA_{\mathrm{centered}}^TA_{\mathrm{centered}}$ 是 $A^TA$ 型——$x^T\Sigma x$ 是投影方向的方差，恒非负。**PCA 就是把谱定理作用在 $\Sigma$ 上**：特征向量 = 主成分方向、特征值 = 方差、白化 = 沿 $q_i$ 缩放 $1/\sqrt{\lambda_i}$（本篇 $\sqrt\Lambda$ 的用武之地）——LA_12 正片；
- **Hessian 与深度学习的损失面**：极小值判定的矩阵版直接进入优化——Newton 法每步解 $H p=-\nabla f$，要求 $H$ 正定（否则步进鞍点/最大值方向），于是有阻尼（$H+\varepsilon I$）、Gauss–Newton、trust region 三种工程补丁。高维随机神经网络的 Hessian 以**鞍点为主**（负特征值个数随过参数化下降）——为什么 SGD 能一路下山找到好极小，谱理论给了第一层解释；
- **Mahalanobis 距离与多元高斯**：$d^2=x^T\Sigma^{-1}x$ 是加权范数，$\Sigma^{-1}$ 正定 ⟹ 等值面 $x^T\Sigma^{-1}x=c$ 是**椭球**（§8 的 $n$ 维版）——多元高斯密度 $\exp(-\tfrac12x^T\Sigma^{-1}x)$ 的等值线正是它；均值离群点检测、高斯分类器的几何全在这；
- **核矩阵与 GP**：核方法 $K_{ij}=k(x_i,x_j)$ 的 Gram 矩阵 = $\Phi^T\Phi$ 型（$\phi(x_i)$ 为特征行）——**半正定是核函数合法的充要条件**（Mercer 定理）；高斯过程回归每步算 $\text{chol}(K+\sigma^2I)$——本篇 Cholesky 的日常出场；
- **图拉普拉斯 = 半正定**：LA_06 的关联矩阵 $A$ 组合出本篇主角 $L=A^TA$（等价于度矩阵减邻接矩阵）：$x^TLx=\sum_{\text{边}(i,j)}(x_i-x_j)^2$——正是 **$A_1$ 型能量**（相邻差分平方和）非负，故 $L$ 半正定；连通图 ⟹ 仅 $\lambda_1=0$（常向量，全体等势）。谱聚类的 Fiedler value $\lambda_2$、GNN 的消息传递平滑假设，全建立在这块地基上。
- **Cholesky：正定系统的专用快车道**：对称正定系统不走 $LU$（对称世界主元恒同号，无需选主元！）——$C^TC$ 分解减半计算量、保对称、数值稳。从 GP 回归到拟牛顿法（BFGS 维持 $H\approx C^TC$）到内点法解 KKT 系统，全是它的地盘；LA_15 的共轭梯度法**只对正定矩阵定义**；
- **Rayleigh 商与谱分析**：$\lambda_{\max}=\max x^TSx/x^Tx$ 给出对称矩阵的算子范数、幂法的收敛对象、Oja 规则学主成分的目标；神经网络权重谱（$\sigma_{\max}$、谱归一化）的估计与正则化，都在反复使用这条极值刻画（习题 9 是它的两行证明）。

---

## 13. 习题精选（Problem Set 6.4–6.5 选摘）

**题 1**（6.4 #3，对称-反对称分裂）：把任何方阵 $A$ 写成对称 $S$ 加反对称 $N$（$S^T=S$、$N^T=-N$）。
*提示*：$S=\frac12(A+A^T)$、$N=\frac12(A-A^T)$——LA_04 已用过一次的分裂，现在看清它为什么唯一：对称与反对称子空间互为正交补（矩阵内积下），维数 $\frac{n^2+n}2+\frac{n^2-n}2=n^2$ 恰好凑满（LA_06 的直和）。

**题 2**（6.4 #10）：$A^3=\mathbf 0$ 则特征值必须是什么？举一个 $A\neq\mathbf 0$ 的例子；再证明：若 $A$ 对称且 $A^3=\mathbf 0$，则 $A=\mathbf 0$。
*提示*：$\lambda^3=0\Rightarrow\lambda$ 全零（幂零矩阵，LA_10）。非对称例：$\begin{bmatrix}0&1\\0&0\end{bmatrix}$。对称时 $A=Q\Lambda Q^T=Q\cdot\mathbf 0\cdot Q^T=\mathbf 0$——**谱定理把幂零与零矩阵划上等号**（可对角化 + 全零谱 ⟹ 零矩阵；短缺特征向量的 Jordan 世界才会非零幂零）。

**题 3**（6.4 #12，假证明找茬）：有人说每个实矩阵特征值全实——$Ax=\lambda x$ 两边左乘 $x^T$ 得 $\lambda=\frac{x^TAx}{x^Tx}$，分子分母都是实数。找出漏洞。
*提示*：$x$ 可能是**复**向量——$x^TAx$ 未必实、$x^Tx$ 甚至是**零**。测试：$90°$ 旋转矩阵 $\lambda=i$、$x=(i,1)$：$x^Tx=i^2+1=0$——分母直接消失。**复世界的内积必须带共轭**（$\bar x^Tx$），这正是 §3.1 证明里用 $\bar x^T$ 的原因、也是 Ch 9 Hermitian 的入口。

**题 4**（6.4 #27+35，对称 × 正交）：哪些矩阵既对称又正交？证明它们满足 $S^2=I$，特征值只能是什么？写出全部 $2\times2$ 解。
*提示*：$S^T=S=S^{-1}$ ⟹ $S^2=I$；$\lambda=\pm1$（LA_10 复习：正交给 $\lvert\lambda\rvert=1$，对称给实）。$S=Q\,\mathrm{diag}(\pm1)\,Q^T$ 取遍全部解；$2\times2$：$Q$ 是旋转，得 $\begin{bmatrix}\cos\theta&\sin\theta\\ \sin\theta&-\cos\theta\end{bmatrix}$——§5 的反射全家福（镜面角度 $\theta/2$ 遍取）。

**题 5**（6.4 #18，SVD 的前奏）：$A$ 是 $2\times2$。分块矩阵 $S=\begin{bmatrix}0&A^T\\A&0\end{bmatrix}$ 对称。证明：若 $Az=\sigma z$，则 $\pm\sigma$ 都是 $S$ 的特征值（特征向量 $(z,\pm z)$ 造出来），且 $\sigma^2$ 是 $A^TA$ 的特征值。取 $A=I$ 求出 $S$ 的全部特征对。
*提示*：$S\begin{bmatrix}z\\ z\end{bmatrix}=\begin{bmatrix}A^Tz\\ Az\end{bmatrix}$——要等于 $\sigma\begin{bmatrix}z\\z\end{bmatrix}$ 需 $Az=\sigma z$ 且 $A^Tz=\sigma z$（对称 $A$ 即可）。一般情形用 $(y,\pm z)$ 与 $A^Ty=\sigma y$（$z$ 配 $A$、$y$ 配 $A^T$）。$A=I$：$\lambda=1,1,-1,-1$。**$S$ 的特征值就是 $\pm$奇异值**——LA_12 SVD 的第一块砖在此预埋。

**题 6**（6.5 #20，四条快问快答）：给出快速理由：(a) 正定必可逆；(b) 正定的投影矩阵只有 $I$；(c) 对角全正的对角阵正定；(d) $\det>0$ 的对称阵未必正定。
*提示*：(a) $\lambda>0$ 无零；(b) 投影的 $\lambda\in\{0,1\}$；(c) $\sum d_ix_i^2>0$；(d) $-I$。四条各一行——正定世界的"常识清单"。

**题 7**（6.5 #31，碗-鞍边界）：$z=4x^2+12xy+cy^2$ 对哪些 $c$ 是碗、哪些 $c$ 是鞍？边界值 $c=9$ 时图形是什么？
*提示*：$S=\begin{bmatrix}4&6\\6&c\end{bmatrix}$，$\det=4c-36$。$c>9$ 碗（正定）、$c<9$ 鞍（不定）、$c=9$：$z=(2x+3y)^2$——**抛物形山谷**，沿直线 $2x+3y=0$ 整条谷底高度为零（半正定），既非极小也无鞍。

**题 8**（6.5 #33）：$S,T$ 对称正定时 $ST$ 未必对称——证明 $ST$ 的特征值仍然全正。
*提示*：$STx=\lambda x$ 两边左乘 $(Tx)^T$：$(Tx)^TS(Tx)=\lambda\, x^TT^Tx=\lambda\|Tx\|^2$。左边是正定能量（$Tx\neq\mathbf0$ 因 $T$ 可逆）、$\|Tx\|^2>0$，故 $\lambda>0$。**非对称矩阵也可以有全正谱**——但注意它的二选一：要么配对称化 $\tfrac12(ST+TS)$ 用能量，要么别当正定用。

**题 9**（6.5 #36，Rayleigh 商）：$S$ 正定、$\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_n$。(a) $\lambda_1I-S$ 的特征值是什么？是半正定吗？(b) 推出 $\lambda_1x^Tx\ge x^TSx$；(c) 结论：$x^TSx/x^Tx$ 的最大值是多少？
*提示*：(a) $\lambda_1-\lambda_i\ge0$，半正定；(b) $x^T(\lambda_1I-S)x\ge0$ 直接展开；(c) 最大值 $=\lambda_1$，在 $x=q_1$ 处取到。这就是 §4 末尾 Rayleigh 刻画的证明——**不用展开、不用微积分，一个移位阵搞定**。

**题 10**（6.5 #37，能量配方）：$S=\begin{bmatrix}a&a&a\\a&a+c&a-c\\a&a-c&a+c\end{bmatrix}$。对哪些 $a,c$ 正定？半正定呢？
*提示*：配方能量 $x^TSx=a(x_1+x_2+x_3)^2+c(x_2-x_3)^2$（先验证再使用：$S=aJ_3+c(e_2-e_3)(e_2-e_3)^T$）。两项平方非负 ⟹ $a\ge0,c\ge0$ 时半正定；$a>0,c>0$ 时若能量为零则两项全零 ⟹ $x=\mathbf 0$，正定。**配方 = 显式分解**：$S=BB^T$，其中 $B=\left[\ \sqrt a\,\mathbf 1\ \ \sqrt c\,(e_2-e_3)\ \right]$ 是 $3\times2$——列无关（$a>0,c>0$）恰对应正定。

---

## 14. 逻辑推导概念图

下面把本篇的逻辑骨架画成一棵树：顶部是出发点（实对称假设与正定定义），箭头是推理方向，每一步只使用它上游的结论。特别值得盯住"实对称"这条假设——它在 λ 全实、特征向量正交、Schur 收口、符号配对四条线上都是 load-bearing 的。

```text
出发点：实对称假设 S^T = S
│
├─→ λ 全实（共轭两连击：(λ−λ̄)·x̄^Tx = 0，长度平方恒正逼出 λ = λ̄）
│
├─→ 不同 λ 的特征向量正交（x^TSy = y^TSx，唯一一步用了 S^T = S）
│
├─→ Schur 分解 A = QTQ^T（每个方阵都能三角化）
│        └─→ 对称时 T 上三角且对称 ⟹ 必为对角阵
│                 └─→ 谱定理 S = QΛQ^T：λ 全实、q 标准正交、重根也不短缺
│                          ├─→ S = λ_1q_1q_1^T + ... + λ_nq_nq_n^T（正交投影加权和）
│                          ├─→ 主轴定理：椭圆轴沿 q_i、半轴 1/√λ_i（大 λ 配短轴）
│                          └─→ Rayleigh 商：λ_1 = max x^TSx / x^Tx
│
├─→ 符号配对（LDL^T 中把 L 连续推向 I）：正 λ 个数 = 正主元个数
│
└─→ 出发点：正定定义 x^TSx > 0（对一切 x ≠ 0）
         ├─→ 五重判据等价：主元全正 / 左上角行列式全正 / λ 全正 / 能量 / S = A^TA 列无关
         │        ├─→ Cholesky S = C^TC（平方和的系数是主元）
         │        └─→ 放宽为 ≥ 得半正定：边缘上 det = 0、零空间方向能量为零
         ├─→ 能量相加：S、T 正定 ⟹ S+T 正定（主元与 λ 在加法下全乱，能量不乱）
         └─→ Hessian 正定 ⟹ 梯度零点是极小（一元 f''>0 的矩阵版）
```

| 出发点 | 买到什么 | 撤掉它失去什么 |
|---|---|---|
| $S^T=S$（实对称） | $\lambda$ 全实、谱定理 $S=Q\Lambda Q^T$（重根不短缺）、特征值与主元符号配对 | $B=\begin{bmatrix}1&-2\\3&-4\end{bmatrix}$ 主元全正 $\lambda$ 却全负；非对称重根可短缺（LA_10 的 $\begin{bmatrix}5&1\\0&5\end{bmatrix}$）；$\lambda$ 可为复（旋转矩阵） |
| $x^TSx>0$ 对**一切** $x\neq\mathbf 0$（能量定义） | $S+T$ 正定、$S=A^TA$ 正定、Hessian 极小判定 | 只看局部信息会被骗：$\begin{bmatrix}1&2\\2&1\end{bmatrix}$ 对角全正却 $\lambda=3,-1$；$-S_2$ 的 $\det=+2$ 却负定 |
| $A$ 列无关（$S=A^TA$ 的那个 $A$） | $A^TA$ 正定：$x^TA^TAx=\|Ax\|^2>0$（LA_07 定理升级） | 列相关只剩半正定：$A=\begin{bmatrix}1&0\\2&0\end{bmatrix}$ 第二列是零，循环 $T$ 的 $(1,1,1)$ 方向能量恰为零 |
| 左上角行列式从 $a>0$ 逐级往上验 | 五重判据之一；$I+J_3$ 一次全检（$2,\ 3,\ 4$ 全正） | 跳级就翻车：$S_3$ 的 $\det=+2$ 却负定；角落一个 $b=2$ 就摧毁正定性（$\det T=(1+b)(4-2b)$） |

---

## 15. 小结与下一篇

Chapter 6 五节至此走完一条完整的弧线，本篇收口的两件套：

| 对象 | 一句话 | 接口 |
|------|--------|------|
| 谱定理 | $S=Q\Lambda Q^T$：实 $\lambda$、正交 $q$、永可对角化 | $\lambda$ 与主元符号配对；$S=\sum\lambda_iq_iq_i^T$ |
| 五重判据 | 主元 / 左上角行列式 / 特征值 / 能量 $x^TSx$ / $A^TA$（列无关） | 验一即全；Cholesky $S=C^TC$；椭圆半轴 $1/\sqrt{\lambda}$ |
| 三副分解 | $LDL^T$（主元）、$C^TC$（Cholesky）、$Q\Lambda Q^T$（谱） | 同一矩阵三套平方和（§7 表） |
| 几何与应用 | 主轴定理：斜椭圆转正；Hessian 正定 ⟹ 极小 | 半正定 = 边缘（$\det=0$）；$-1,2,-1$ 家族 $\lambda_k=2-2\cos k\pi h$ |

一路走来的伏笔回收清单：$A^TA$ 从 LA_07 的可逆性定理升级为正定性；$LDL^T$ 从 LA_04 的记录格式升级为符号配对的证明引擎；LA_08 的 $A^TA=R^TR$ 升级为 Cholesky 的标准形；LA_10 的 $-K$ 三对角与 5I−J₄ 家族在本篇拿到闭式谱。

但对称只是特权阶层。**一般矩阵**（长方形、非对称）没有 $Q\Lambda Q^T$——它们的权利法案要等全书最伟大的分解：把 $\S7$ 的 $A^TA$ 思想推到极致，每个矩阵 $A$ 都能造出对称正半定的 $A^TA$ 与 $AA^T$，两者的特征向量拼出 $A$ 自己的**奇异向量**，特征值的平方根是**奇异值**：

$$
A=U\Sigma V^T
$$

谱定理 + 四个基本子空间 + 伪逆，三条线在 SVD 汇合——LA 12 正片。

---

*下一篇：[线性代数 12：SVD 与 PCA](../LA_12_svd_pca/)*
