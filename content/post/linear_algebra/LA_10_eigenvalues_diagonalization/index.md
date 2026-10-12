---
title: "线性代数 10: 特征值与对角化"
date: 2026-11-10
math: true
tags:
  - Notes
  - LinearAlgebra
categories:
  - LinearAlgebra
description: "Strang 第 5 版 §6.1-6.3 精读——Ax=λx 与特征方程 det(A−λI)=0、迹与行列式两个快速检验、复特征值与旋转矩阵、对角化 A=XΛX⁻¹ 与相似矩阵、Fibonacci 与 Lucas 闭式、代数重数与几何重数、du/dt=Au 的纯指数解与三步法、2×2 稳定性判据、矩阵指数 e^{At}=Xe^{Λt}X⁻¹ 与 te^{λt}"
---

# 线性代数 10：特征值与对角化

> 📖 本篇覆盖 Strang《Introduction to Linear Algebra》第 5 版 §6.1–6.3。前置：[LA 09](../LA_09_determinants/)——行列式篇结尾留下的那座桥 det(A−λI)=0，本篇正式过桥。

---

## 1. 这一篇解决什么问题？

全书在此换挡。前半本是 $Ax=b$：**平衡**与**稳态**——解一个方程，得到一个静止的答案。后半本关于**变化**：连续时间 $du/dt=Au$、离散时间 $u_{k+1}=Au_k$。这些方程**不能靠消元解决**——§3 会亲眼看到消元改变特征值。

核心思想一句话：**避开矩阵 $A$ 的全部复杂性，找那些不被 $A$ 改变方向的向量**。$A$ 乘一般向量会把它拧到别的方向；但某些例外向量 $x$ 满足 $Ax=\lambda x$——乘完只剩一个数的效果。**数比向量好算**：$A$ 的第 100 次幂作用在 $x$ 上就是 $\lambda^{100}x$，一行写完。

开场例（教材第一页）：$A=\begin{bmatrix}.8&.3\\.2&.7\end{bmatrix}$ 的 $A^{100}$ 列几乎是 $(.6,.4)$ 的重复：

$$
A^{100}\approx\begin{bmatrix}.6000&.6000\\.4000&.4000\end{bmatrix},
$$

这不是乘 100 次矩阵算出来的，而是用了两个特征值 $\lambda=1$ 和 $\lambda=\tfrac12$。本篇三步走：

| 节 | 问题 | 工具 |
|---|------|------|
| §6.1 | 特征值/特征向量**怎么求** | $\det(A-\lambda I)=0$（LA 09 的行列式在此兑现） |
| §6.2 | 矩阵**幂**怎么算 | 对角化 $A=X\Lambda X^{-1}$、Fibonacci 闭式 |
| §6.3 | **微分**方程怎么解 | $e^{\lambda t}$、稳定性、矩阵指数 $e^{At}$ |

---

## 2. Ax=λx：定义、三大家族与求解三步（§6.1 上）

### 2.1 定义

$$
\boxed{\ Ax=\lambda x\ (x\neq\mathbf 0)\ }
$$

**特征向量**（eigenvector）$x$：被 $A$ 乘后**不改变方向**的向量（可以拉伸、压缩、反转、不动）。**特征值**（eigenvalue）$\lambda$：拉伸的倍数——$\lambda=2$、$\tfrac12$、$-1$、$1$ 各有几何；$\lambda=0$ 也完全合法，此时 $Ax=\mathbf 0$ 说明 $x$ 住在零空间（LA 05 的 $N(A)$）。极端情形 $A=I$：**每个**向量都是特征向量，全部 $\lambda=1$。

### 2.2 三大家族：Markov、投影、反射

| 家族 | 例 | 特征值 | 特征向量的身份 |
|------|-----|--------|---------------|
| Markov（列和 1） | $\begin{bmatrix}.8&.3\\.2&.7\end{bmatrix}$ | $1,\ \tfrac12$ | $(.6,.4)$ 稳态、$(1,-1)$ 衰减模态 |
| 投影 $P$ | $\begin{bmatrix}.5&.5\\.5&.5\end{bmatrix}$ | $1,\ 0$ | $(1,1)$ 列空间、$(1,-1)$ 零空间 |
| 反射 $R$ | $\begin{bmatrix}0&1\\1&0\end{bmatrix}$ | $1,\ -1$ | $(1,1)$ 镜面上、$(1,-1)$ 镜面法向 |

**Markov 例全程**（本篇的基准矩阵）：从 $\det(A-\lambda I)=0$ 出发，

$$
\det\begin{bmatrix}.8-\lambda&.3\\.2&.7-\lambda\end{bmatrix}
=\lambda^2-\tfrac32\lambda+\tfrac12=(\lambda-1)(\lambda-\tfrac12),
$$

故 $\lambda_1=1$、$\lambda_2=\tfrac12$。分别解 $(A-\lambda I)x=\mathbf 0$ 得 $x_1=(.6,.4)$（验证 $Ax_1=x_1$ ✓）与 $x_2=(1,-1)$（验证 $Ax_2=(.5,-.5)=\tfrac12 x_2$ ✓）。第一列 $(.8,.2)=x_1+0.2\,x_2$，于是 $A$ 乘 99 次：

$$
A^{99}\bigl(x_1+0.2\,x_2\bigr)=x_1+0.2\,(\tfrac12)^{99}x_2
\quad\Longrightarrow\quad A^{100}\ \text{第一列}\approx(.6,.4).
$$

$0.2\cdot(1/2)^{99}$ 小到 30 位小数之外。**$x_1$ 是稳态（steady state），$x_2$ 是衰减模态（decaying mode）**——幂越高，衰减模态越透明。这个 $A$ 是 Markov 矩阵：列和为 1 强制 $\lambda=1$（§10.3 的 Google PageRank 用同一条性质）。

**投影例**的三重身份一次看全：列和 1 ⟹ $\lambda=1$；奇异 ⟹ $\lambda=0$；对称 ⟹ 特征向量垂直。投影只保留列空间、摧毁零空间——特征值 $\{1,0\}$ 正是这份"保留/摧毁"名单。

**反射例**：$R=2P-I$。矩阵减 $I$，特征值全体减 1、特征向量不动——所以 $R$ 的 $\{1,-1\}$ 与 $P$ 的 $\{1,0\}$ 恰好错位 1。**教训：没有负元素的矩阵照样可以有负特征值**（$R$ 全是 0 和 1，却有 $\lambda=-1$）。

### 2.3 特征方程与求解三步

把 $\lambda x$ 移到左边：$(A-\lambda I)x=\mathbf 0$。有非零解 ⟺ $A-\lambda I$ 奇异 ⟺ 行列式为零：

$$
\boxed{\ \det(A-\lambda I)=0\ }
$$

这个**特征多项式**只含 $\lambda$ 不含 $x$；$n\times n$ 矩阵给出 $n$ 次多项式、$n$ 个根（可重复、可为复数）。

**Example 4**（奇异矩阵）：$A=\begin{bmatrix}1&2\\2&4\end{bmatrix}$（$\det=0$，故 $\lambda=0$ 必在名单上）：

$$
\det(A-\lambda I)=(1-\lambda)(4-\lambda)-4=\lambda^2-5\lambda=\lambda(\lambda-5),
$$

$\lambda_1=0$（特征向量 $(2,-1)\in N(A)$）、$\lambda_2=5$（特征向量 $(1,2)$，验证 $A(1,2)=(5,10)=5(1,2)$ ✓）。教材的强调：**$\lambda=0$ 毫不特殊**——它是特征值当且仅当 $A$ 奇异，仅此而已。

**求解三步**（全章的操作纲领）：

| 步 | 动作 | 产物 |
|---|------|------|
| 1 | 算 $\det(A-\lambda I)$（对角线减 $\lambda$） | $\lambda$ 的 $n$ 次多项式 |
| 2 | 解 $\det(A-\lambda I)=0$ | $n$ 个特征值 |
| 3 | 对每个 $\lambda$ 解 $(A-\lambda I)x=\mathbf 0$ | 特征向量（一整条线） |

**2×2 捷径**（PS 6.1 #11 的妙观察）：$A-\lambda I$ 奇异时两行都是某个 $(a,b)$ 的倍数，特征向量直接取 $(b,-a)$。特征向量本来就不唯一——任何非零倍数都行，MATLAB 的 `eig(A)` 统一除以长度取单位向量。

**警告**：两个特征值相等时，可能只剩**一条**特征向量线（无法凑满 $n$ 个无关向量）——但也可能管够（$A=I$：$\lambda$ 全是 1，特征向量遍地都是）。重根是"可能短缺"，不是"必然短缺"。§7 专门处理这件事。

---

## 3. 特征方程、迹与行列式（§6.1 中）

**坏消息先行：消元不保特征值。** 行交换、行相加都会改变 $\lambda$：

$$
A=\begin{bmatrix}1&3\\2&6\end{bmatrix}\ \text{有}\ \lambda=0,7
\qquad\longrightarrow\qquad
U=\begin{bmatrix}1&3\\0&0\end{bmatrix}\ \text{有}\ \lambda=0,1.
$$

（$U$ 三角，$\lambda$ 就是对角线；行 2 减 2·行 1 把 $\lambda=7$ 改成了 $\lambda=1$。）三角矩阵的特征值恒在对角线上——但那是**它自己的** $\lambda$，不是原矩阵的。LA 02–04 苦心经营的消元机器，在这里第一次失灵。

**好消息：两个快速检验。**

$$
\boxed{\ \lambda_1\lambda_2\cdots\lambda_n=\det A\ },
\qquad
\boxed{\ \lambda_1+\lambda_2+\cdots+\lambda_n=\mathrm{trace}\ A=a_{11}+\cdots+a_{nn}\ }
$$

（证明在 §5 与习题：前者取 $\det(X\Lambda X^{-1})$，后者用 $\mathrm{tr}(XY)=\mathrm{tr}(YX)$。）这两个检验**不能替代**求解，但计算出错时它们几乎总能报警。

**2×2 的好消息加倍**：trace 和 det 完全决定 2×2 的特征值。trace $=3$、det $=2$ ⟹ $\lambda=1,2$，下面三个矩阵同谱（都是 $\det(A-\lambda I)=\lambda^2-3\lambda+2$）：

$$
\begin{bmatrix}1&9\\0&2\end{bmatrix},
\qquad
\begin{bmatrix}3&1\\-2&0\end{bmatrix},
\qquad
\begin{bmatrix}7&-3\\10&-4\end{bmatrix}.
$$

**Worked 6.1 A**（特征值全家福）：$A=\begin{bmatrix}2&-1\\-1&2\end{bmatrix}$（LA 04 的 $K$ 矩阵 2 阶版！）：

$$
\det(A-\lambda I)=\lambda^2-4\lambda+3=(\lambda-1)(\lambda-3)=0
\quad\Longrightarrow\quad \lambda_1=1,\ x_1=(1,1);\quad \lambda_2=3,\ x_2=(1,-1).
$$

检验：trace $2+2=1+3$ ✓、det $3=1\cdot3$ ✓。$A^2$、$A^{-1}$、$A+4I$ **共享同一组特征向量**，特征值按函数走：

| 矩阵 | 特征值 | 验证 |
|------|--------|------|
| $A$ | $1,\ 3$ | — |
| $A^2=\begin{bmatrix}5&-4\\-4&5\end{bmatrix}$ | $1,\ 9$ | trace $10=1+9$、det $9$ ✓ |
| $A^{-1}$ | $1,\ \tfrac13$ | 倒数 |
| $A+4I$ | $5,\ 7$ | 平移 |

一般规律：$Ax=\lambda x\Rightarrow A^2x=\lambda^2x$、$A^{-1}x=\lambda^{-1}x$、$(A+cI)x=(\lambda+c)x$——证明全部是"对 $Ax=\lambda x$ 再做一次运算"（习题 9 的三连）。

---

## 4. 复特征值、特殊矩阵家族与两个陷阱（§6.1 下）

### 4.1 复特征值：旋转矩阵

**Example 5**：$90°$ 旋转 $Q=\begin{bmatrix}0&-1\\1&0\end{bmatrix}$ 没有实特征向量（旋转后没有任何实向量保持方向），但

$$
\det(Q-\lambda I)=\lambda^2+1=0
\quad\Longrightarrow\quad \lambda_1=i,\ \lambda_2=-i.
$$

第二个更妙的论证：$Q^2=-I$。平方矩阵平方特征值，故 $\lambda^2=-1$，$\lambda=\pm i$。复特征向量也现成：$Q(1,-i)=(i,1)=i(1,-i)$ ✓、$Q(1,i)=(-i,1)=-i(1,i)$ ✓。一般旋转角 $\theta$ 的矩阵（习题 #14）特征值为

$$
\lambda=\cos\theta\pm i\sin\theta=e^{\pm i\theta}.
$$

### 4.2 特殊矩阵 = 复平面上的特殊数（§6.4 的预告）

| 矩阵家族 | 条件 | 复数类比 | 特征值 |
|----------|------|---------|--------|
| 对称 | $S^T=S$ | 实数 | $\lambda$ 全**实** |
| 反对称 | $A^T=-A$ | 纯虚数 | $\lambda$ 全**纯虚** |
| 正交 | $Q^TQ=I$ | 单位圆 $|\lambda|=1$ | $|\lambda|=1$ |

这三条不止是类比，是 §6.4（LA_11）要证的定理；这些家族的特征向量还互相垂直。本篇先记账。

### 4.3 陷阱一：A+B 与 AB 的特征值

第一个直觉错误："AB 的特征值 = $\lambda(A)\cdot\lambda(B)$"。假证明长这样：

$$
ABx=A(\beta x)=\beta(Ax)=\beta\lambda x.\qquad(\text{假!})
$$

错在哪？**$A$ 的特征向量一般不是 $B$ 的特征向量**——第一步 $Bx=\beta x$ 就未必成立。反例（两个零特征值矩阵乘出 1）：

$$
A=\begin{bmatrix}0&1\\0&0\end{bmatrix},\quad
B=\begin{bmatrix}0&0\\1&0\end{bmatrix}:\quad
\lambda(A)=\lambda(B)=\{0,0\},
$$

$$
AB=\begin{bmatrix}1&0\\0&0\end{bmatrix}\ (\lambda=1,0),
\qquad
A+B=\begin{bmatrix}0&1\\1&0\end{bmatrix}\ (\lambda=1,-1).
$$

$\lambda(A+B)\neq\lambda(A)+\lambda(B)$（这里 $0+0\neq1+(-1)$……至少和为 0 对上了 trace）。假证明指出的**正确版本**：若 $A,B$ 真的共享 $n$ 个无关特征向量，则特征值可乘、可加。共享特征向量的判据是量子力学的核心方程：

$$
\boxed{\ A,B\ \text{共享}\ n\ \text{个无关特征向量}\iff AB=BA\ }
$$

**Heisenberg 不确定性原理**：位置矩阵 $P$ 与动量矩阵 $Q$ 不对易——$QP-PQ=I$（无限维矩阵才可能；有限维时左边迹为零、右边迹为 $n$，矛盾——习题 6 的一行证明）。于是 $P$ 与 $Q$ 无法同时对角化：**位置测得越准，动量就越测不准**，$\|Px\|\,\|Qx\|\ge\tfrac12\|x\|^2$（PS #36 用 Schwarz 不等式推出）。

### 4.4 陷阱二：消元（已见）与估计器 Gershgorin

特征值既然脆弱，能**估计**吗？**Gershgorin 圆盘定理**（Worked 6.1 B）：每个特征值落在至少一个以对角元 $a_{ii}$ 为圆心、以 $R_i=\sum_{j\neq i}|a_{ij}|$（该行其余元素绝对值之和）为半径的圆盘内：

$$
|\lambda-a_{ii}|\le R_i\quad\text{对某个 }i\text{ 成立}.
$$

证明一行：$\lambda$ 是特征值 ⟹ $A-\lambda I$ 奇异 ⟹ $A-\lambda I$ 不可能行对角占优（LA 02 §2.5）⟹ 至少一行 $|a_{ii}-\lambda|\le R_i$。2×2 例：圆心 $a,d$、半径 $|b|,|c|$ 的两个圆盘；3×3 例 $\begin{bmatrix}d_1&1&2\\2&d_2&1\\1&2&d_3\end{bmatrix}$ 三个圆盘半径全为 3。复平面上的圆——$\lambda$ 本来就可能复。

**Worked 6.1 C**（对称 3×3 完整演示）：$S=\begin{bmatrix}1&-1&0\\-1&2&-1\\0&-1&1\end{bmatrix}$。

- 行和为零 ⟹ $S(1,1,1)=\mathbf 0$ ⟹ $\lambda_1=0$ 免费到手；
- 算 $\det(S-\lambda I)=(1-\lambda)\bigl[(2-\lambda)(1-\lambda)-2\bigr]=(1-\lambda)(-\lambda)(3-\lambda)$；
- $\lambda=0,1,3$（trace $1+2+1=4=0+1+3$ ✓），特征向量 $(1,1,1),\ (1,0,-1),\ (1,-2,1)$——**两两垂直**（对称的礼物，LA_11 证明）。

教材的务实收尾：3×3 手算已是极限，再大就用 `eig(A)`（[X,E] 输出单位特征向量），永远别碰行列式。MATLAB 的 `eigshow` 演示值得一玩：单位圆上的 $x$ 转圈、$Ax$ 跟着转，两者**平行的瞬间**就是特征向量。

---

## 5. 对角化 A=XΛX⁻¹ 与相似矩阵（§6.2 上）

### 5.1 一行推导

设 $A$ 有 $n$ 个**线性无关**特征向量 $x_1,\dots,x_n$，排进**特征向量矩阵** $X$ 的列。逐列看 $AX$ 与 $X\Lambda$：

$$
AX=A[\,x_1\cdots x_n\,]=[\,Ax_1\cdots Ax_n\,]=[\,\lambda_1x_1\cdots\lambda_nx_n\,]=X
\underbrace{\begin{bmatrix}\lambda_1&&\\&\ddots&\\&&\lambda_n\end{bmatrix}}_{\Lambda}.
$$

$\Lambda$（大 lambda）在右边的意思是"$\lambda_i$ 只乘第 $i$ 列"。$X$ 可逆（列无关），于是

$$
\boxed{\ X^{-1}AX=\Lambda,\qquad A=X\Lambda X^{-1}\ }
$$

**Example 1**（三角开场）：$A=\begin{bmatrix}1&5\\0&6\end{bmatrix}$（$\lambda$ 在对角线上：1、6）：

$$
X=\begin{bmatrix}1&1\\0&1\end{bmatrix},\quad
\Lambda=\begin{bmatrix}1&0\\0&6\end{bmatrix},\quad
X^{-1}=\begin{bmatrix}1&-1\\0&1\end{bmatrix}.
$$

$A^2=X\Lambda^2X^{-1}$：$\Lambda^2=\mathrm{diag}(1,36)$，乘回来 $A^2=\begin{bmatrix}1&35\\0&36\end{bmatrix}$（直接乘也验得）。$k=0$ 得 $A^0=I$，$k=-1$ 得 $A^{-1}$——**幂的运算变成对角线的运算**，因为 $X\Lambda X^{-1}X\Lambda X^{-1}$ 中间那对 $X^{-1}X$ 自动相消：

$$
\boxed{\ A^k=X\Lambda^kX^{-1}\ }
$$

### 5.2 四个要点

| # | 要点 | 内容 |
|---|------|------|
| 1 | **无重根 ⟹ 自动可对角化** | 不同 $\lambda$ 的特征向量必无关（下面证明） |
| 2 | 特征向量可乘非零常数 | $A(cx)=\lambda(cx)$；软件统一输出单位向量 |
| 3 | $X$ 的列序 = $\Lambda$ 的对角序 | 交换 $X$ 的列，$\Lambda$ 对应重排 |
| 4 | 重根可能短缺 | 只有 $n$ 个无关特征向量才能填满 $X$（§7 正面处理） |

**定理（不同 $\lambda$ ⟹ 无关）的完整证明**——教材最值得抄下来的两行。设 $c_1x_1+c_2x_2=\mathbf 0$。乘 $A$ 得 $c_1\lambda_1x_1+c_2\lambda_2x_2=\mathbf 0$；乘 $\lambda_2$ 得 $c_1\lambda_2x_1+c_2\lambda_2x_2=\mathbf 0$。**两式相减**：

$$
(\lambda_1-\lambda_2)c_1x_1=\mathbf 0
\quad\Longrightarrow\quad c_1=0
\quad(\lambda_1\neq\lambda_2,\ x_1\neq\mathbf 0),
$$

同理 $c_2=0$。推广到 $j$ 个：逐次用 $\lambda_j,\lambda_{j-1},\dots$ 相减剥掉后排，最后剩下 $(\lambda_1-\lambda_2)\cdots(\lambda_1-\lambda_j)c_1x_1=\mathbf 0$，逼迫每个 $c_i=0$。∎

### 5.3 可逆 ≠ 可对角化（必须刻在脑门上）

两件事管两样东西，毫无关联：

| 性质 | 看什么 | 判据 |
|------|--------|------|
| **可逆性** | 特征**值** | $\lambda=0$ 出现与否（$\det A=\prod\lambda_i$） |
| **可对角化** | 特征**向量** | 能否凑满 $n$ 个无关向量 |

四个象限都有居民：$\begin{bmatrix}1&1\\0&1\end{bmatrix}$ 可逆但不可对角化（$\lambda=1,1$ 只有一条特征向量线）；$\begin{bmatrix}1&0\\0&0\end{bmatrix}$ 不可逆但可对角化。还有一个安慰条款：**每个特征值至少有一个特征向量**——$A-\lambda I$ 奇异，零空间维数 $\ge1$。若解 $(A-\lambda I)x=\mathbf 0$ 只解出 $x=\mathbf 0$，那 $\lambda$ 根本不是特征值，回去查 $\det(A-\lambda I)=0$ 的根。

### 5.4 相似矩阵：同一个 Λ 的一家人

固定 $\Lambda$（或任何矩阵 $C$），让可逆矩阵 $B$ 任意跑：

$$
A=BCB^{-1}\quad\text{与}\quad C\ \text{称为**相似**（similar）},
$$

（对角化是其特例 $B=X$、$C=\Lambda$。）**相似矩阵共享特征值**，证明只有一行：

$$
Cx=\lambda x\quad\Longrightarrow\quad (BCB^{-1})(Bx)=BCx=B\lambda x=\lambda(Bx).
$$

特征向量换了人（$Bx$），特征值原封不动。三个家庭的规模感：

- $C=I$：家族只有 $BIB^{-1}=I$ 一个成员——$\lambda$ 全为 1 且可对角化的矩阵只有 $I$；
- $C=\begin{bmatrix}1&1\\0&1\end{bmatrix}$（$\lambda=1,1$ 短缺）：整个家族 $A=\begin{bmatrix}1-rs&r^2\\-s^2&1+rs\end{bmatrix}$（参数 $r,s$ 不全为零）——trace 恒 2、det 恒 1、永远只有一条特征向量线（Jordan 形，§8.3/LA_13 再会）；
- $\Lambda=\mathrm{diag}(1,0)$：家族 = 全体满足 $A^2=A$ 的矩阵（幂等阵）——$\begin{bmatrix}1&0\\0&0\end{bmatrix}$、$\begin{bmatrix}.5&.5\\.5&.5\end{bmatrix}$、以及一切 $\begin{bmatrix}a&b\\c&1-a\end{bmatrix}$（$a-a^2=bc$）；对称的成员就是投影矩阵（LA 07 全家福）。

---

## 6. 矩阵幂与 Fibonacci：三步法（§6.2 中）

### 6.1 Fibonacci 数列的线性代数化

$0,1,1,2,3,5,8,13,\dots$，规则 $F_{k+2}=F_{k+1}+F_k$。自然界出场率惊人：梨树每 3 圈长 8 个生长点、柳树 5 与 13、向日葵冠军 144 圈 233 粒种子——正是 $F_{12}$ 与 $F_{13}$。

标量两步规则 → 向量一步规则：$u_k=(F_{k+1},F_k)$，则

$$
u_{k+1}=\begin{bmatrix}F_{k+2}\\F_{k+1}\end{bmatrix}
=\underbrace{\begin{bmatrix}1&1\\1&0\end{bmatrix}}_{A}u_k,
\qquad
u_0=(F_1,F_0)=(1,0),\qquad u_{100}=(F_{101},F_{100}).
$$

求 $F_{100}$ 不必乘 100 次：$A$ 的特征方程 $\lambda^2-\lambda-1=0$（把矩阵的规则抄给 $\lambda$）给出

$$
\lambda_1=\frac{1+\sqrt5}{2}\approx1.618\ (\text{黄金比}),\qquad
\lambda_2=\frac{1-\sqrt5}{2}\approx-0.618,
$$

特征向量 $x_1=(\lambda_1,1)$、$x_2=(\lambda_2,1)$（companion 矩阵的标配）。三步走：

**第一步**（分解 $u_0$）：$u_0=\dfrac{x_1-x_2}{\lambda_1-\lambda_2}$，即 $c_1=\tfrac1{\sqrt5},\ c_2=-\tfrac1{\sqrt5}$（分母 $\lambda_1-\lambda_2=\sqrt5$）。

**第二步**（乘幂）：每个特征向量只被自己的 $\lambda^k$ 乘。

**第三步**（合并）：取第二个分量（$x_1,x_2$ 的第二分量都是 1），$\lambda_2^{100}\approx0$ 消失：

$$
\boxed{\ F_{100}=\frac{\lambda_1^{100}-\lambda_2^{100}}{\sqrt5}
=\ \text{最接近}\ \frac{\lambda_1^{100}}{\sqrt5}\ \text{的整数}\ (\approx3.54\times10^{20})\ }
$$

每个 $F_k$ 都是整数，而公式里全是无理数——取整后严丝合缝。$F_{101}/F_{100}\to\lambda_1$：黄金比。

### 6.2 三步法（差分方程的一般解）

$$
u_{k+1}=Au_k
\qquad\Longrightarrow\qquad
\boxed{\ u_k=A^ku_0=X\Lambda^kX^{-1}u_0=c_1\lambda_1^kx_1+\cdots+c_n\lambda_n^kx_n\ }
$$

| 步 | 矩阵语言 | 做什么 |
|---|---------|--------|
| 1 | $c=X^{-1}u_0$ | 把 $u_0$ 写成特征向量的组合 |
| 2 | $\Lambda^k$ | 每个分量独立乘 $\lambda_i^k$ |
| 3 | $X(\Lambda^kc)$ | 加回组合——这就是 $u_k$ |

**"Follow the eigenvectors"**——特征向量各走各的指数通道，永不相扰。

**Example 3**（更快的 Fibonacci，规则改 $F_{k+2}=F_{k+1}+2F_k$）：$A=\begin{bmatrix}2&1\\1&0\end{bmatrix}$，$\lambda=2,-1$，$x_1=(2,1)$、$x_2=(-1,1)$。$u_0=(1,0)=\tfrac13(x_1-x_2)$，故

$$
F_k=\frac{2^k-(-1)^k}{3}:\quad 0,\ 1,\ 1,\ 3,\ 5,\ 11,\ 21,\dots
$$

（$F_4=(16-1)/3=5$ ✓。$\lambda=2$ 主导，增长快于 Fibonacci。）

**Worked 6.2 A（Lucas 数）**：同规则、新起点 $L_1=1,\ L_2=3$，接下来 $4,7,11,18$。同一个 $A$，第一步换成 $u_1=(3,1)=\lambda_1x_1+\lambda_2x_2$（系数恰好是特征值——验证：第一分量 $\lambda_1^2+\lambda_2^2=(\lambda_1+\lambda_2)^2-2\lambda_1\lambda_2=1+2=3$ ✓）。于是

$$
\boxed{\ L_{100}=\lambda_1^{100}+\lambda_2^{100}\ }
$$

Lucas 起步快、终身领先，渐近倍数恰是 $\sqrt5$（$L_k/F_k\to\sqrt5$）。

**Markov 矩阵的极限**（回到 §2 基准例）：$A^k=X\Lambda^kX^{-1}$，$(\tfrac12)^k\to0$ 熄灭，$\lambda=1$ 长存：

$$
A^\infty=\begin{bmatrix}.6&1\\.4&-1\end{bmatrix}
\begin{bmatrix}1&0\\0&0\end{bmatrix}
\begin{bmatrix}1&1\\.4&-.6\end{bmatrix}
=\begin{bmatrix}.6&.6\\.4&.4\end{bmatrix}
$$

两列都是稳态 $x_1$。一般判据：$A^k\to$ 零矩阵 $\iff$ 所有 $|\lambda_i|<1$（**模长**，不是实部——差分世界看模、微分世界看实部，§9 的对偶）。

---

## 7. 不可对角化：代数重数与几何重数（§6.2 下）

特征值的"重复"有两种数法， shortage 就藏在这条缝里：

| 记号 | 名字 | 定义 | 性质 |
|------|------|------|------|
| **AM** | 代数重数（algebraic multiplicity） | $\lambda$ 作为 $\det(A-\lambda I)=0$ 根的重数 | 计数来自多项式 |
| **GM** | 几何重数（geometric multiplicity） | $\lambda$ 的无关特征向量个数 $=\dim N(A-\lambda I)$ | 计数来自零空间 |

$$
\boxed{\ GM\le AM\ (\text{每个特征值恒成立})\ }
\qquad
A\ \text{可对角化}\iff\text{每个}\ \lambda\ \text{都}\ GM=AM.
$$

**标准病灶**：$A=\begin{bmatrix}0&1\\0&0\end{bmatrix}$。$\det(A-\lambda I)=\lambda^2$（AM $=2$），但 $Ax=\mathbf 0$ 只解出 $x=c(1,0)$ 一条线（GM $=1$）。"本该"有两个特征向量，只发了一个。

**三个同病相怜的 5,5**（$\det(A-\lambda I)=(\lambda-5)^2$、trace 10、det 25，但 $\mathrm{rank}(A-5I)=1$ → GM $=1$）：

$$
\begin{bmatrix}5&1\\0&5\end{bmatrix},
\qquad
\begin{bmatrix}6&-1\\1&4\end{bmatrix},
\qquad
\begin{bmatrix}5&2\\0&5\end{bmatrix}.
$$

教材建议：这两个矩阵是检验一切"特征向量命题"的试金石——很多真假题的假就假在它们身上。

**一个安慰定理**（PS #25）：$A^2=A$ ⟹ 必可对角化。$\lambda$ 只能是 0 或 1；$\lambda=1$ 的特征向量撑满列空间 $C(A)$、$\lambda=0$ 的撑满零空间 $N(A)$，维数 $r+(n-r)=n$ 恰好凑满——**幂等阵永远有完整的特征向量班底**（投影矩阵是其在 LA 07 的化身）。

**Worked 6.2 B**（全 1 矩阵的减法）：$A=5I-J_4$（$J_4$ 是全 1 矩阵）。$J_4$ 秩一（LA 03 的 $uv^T$、LA 06 的 $A=CR$ 标本）⟹ $\lambda(J_4)=4,0,0,0$（trace $=4$ 送出 4）。减法把每个特征值从 5 里扣：

$$
\lambda(A)=1,\ 5,\ 5,\ 5
\qquad\Longrightarrow\qquad
\det A=1\cdot5\cdot5\cdot5=125.
$$

$\lambda=1$ 的特征向量是 $(1,1,1,1)$；其余特征向量与之垂直（$A$ 对称）。最漂亮的对角化矩阵是 **Hadamard**（LA 08 §5.3 的 $H_4$）：$X=H_4/2$，正交且对称，$X^{-1}=X$。逆矩阵是秩一更新（用 $J^2=4J$ 验证 $(5I-J)\cdot\tfrac15(I+J)=\tfrac15(5I+4J-J)=I$ ✓）：

$$
A^{-1}=\tfrac15\bigl(I+J_4\bigr),
$$

$\lambda(A^{-1})=1,\ \tfrac15,\ \tfrac15,\ \tfrac15$。**图论彩蛋**：$5I-J_4$ 正是完全图 $K_5$ 的约化拉普拉斯矩阵，$\det=125=5^3$ 恰好数出 $K_5$ 的生成树个数（Cayley 公式 $n^{n-2}$；矩阵—树定理，§10.1/LA_16 正片）。换成 6 个节点：$6I-J_5$ 的特征值 $1,6,6,6,6$，$\det=6^4=1296=K_6$ 的生成树数。

---

## 8. 微分方程 du/dt=Au：纯指数解、二阶方程与圆周运动（§6.3 上）

### 8.1 纯指数解

只处理**线性、常系数**方程（$A$ 不随 $t$、不随 $u$ 变——这两类分别是时变与非线性方程，不归线性代数管）。猜解 $u=e^{\lambda t}x$，代回 $du/dt=Au$：左边 $\lambda e^{\lambda t}x$、右边 $e^{\lambda t}Ax$，**$e^{\lambda t}$ 约掉**，剩 $\lambda x=Ax$——又是老朋友。

$$
\boxed{\ Ax=\lambda x\quad\Longrightarrow\quad u(t)=e^{\lambda t}x\ \text{是纯指数解}\ }
$$

$\lambda>0$ 增长、$\lambda<0$ 衰减；复数 $\lambda=r+is$ 时实部 $r$ 定生灭、虚部 $s$ 出振荡（$e^{ist}$ 是转圈因子）。

### 8.2 Example 1：耦合与解耦

$$
\frac{du}{dt}=\begin{bmatrix}0&1\\1&0\end{bmatrix}u,\qquad
u(0)=(4,2).
\qquad\text{分量版：}\ \frac{dy}{dt}=z,\ \frac{dz}{dt}=y\ (\text{耦合!}).
$$

笨办法是盯着方程找组合 $y+z$（增长）、$y-z$（衰减）——聪明办法是让特征向量代劳：$\lambda=\pm1$、$x_1=(1,1)$、$x_2=(1,-1)$，恰好就是这两个组合。完整解 $u=Ce^tx_1+De^{-t}x_2$，用初始值定常数：$C(1,1)+D(1,-1)=(4,2)$ 给 $C=3,\ D=1$：

$$
u(t)=3e^t\begin{bmatrix}1\\1\end{bmatrix}+e^{-t}\begin{bmatrix}1\\-1\end{bmatrix}.
$$

### 8.3 三步法（微分版，与 §6 完全同构）

$$
\boxed{\ u(t)=e^{At}u(0)=Xe^{\Lambda t}X^{-1}u_0=c_1e^{\lambda_1t}x_1+\cdots+c_ne^{\lambda_nt}x_n\ }
$$

| 差分版（§6） | 微分版（本节） | 唯一的区别 |
|--------------|---------------|-----------|
| $\lambda_i^k$ | $e^{\lambda_it}$ | 幂 → 指数 |

**Example 2**：$A=\begin{bmatrix}1&1&1\\0&2&1\\0&0&3\end{bmatrix}$（三角，$\lambda=1,2,3$），特征向量 $x_1=(1,0,0)$、$x_2=(1,1,0)$、$x_3=(1,1,1)$。第一步解 $Xc=u(0)$：

$$
u(0)=(9,7,4)=2x_1+3x_2+4x_3
\quad\Longrightarrow\quad
u(t)=2e^{t}x_1+3e^{2t}x_2+4e^{3t}x_3.
$$

### 8.4 二阶方程与 companion 矩阵

力学里最重要的方程 $my''+by'+ky=0$（质量—阻尼—弹簧）。代 $y=e^{\lambda t}$：每求导一次掉出一个 $\lambda$，剩下 $m\lambda^2+b\lambda+k=0$。线性代数的处理是**升维**：把 $(y,y')$ 打包成状态向量 $u$，

$$
\frac{d}{dt}\begin{bmatrix}y\\y'\end{bmatrix}
=\begin{bmatrix}y'\\-ky-by'\end{bmatrix}
=\underbrace{\begin{bmatrix}0&1\\-k&-b\end{bmatrix}}_{\text{companion 矩阵}}\begin{bmatrix}y\\y'\end{bmatrix},
$$

$$
\det\begin{bmatrix}-\lambda&1\\-k&-b-\lambda\end{bmatrix}=\lambda^2+b\lambda+k\quad(\text{同一个二次式!}),\qquad
\text{特征向量 } (1,\lambda_i).
$$

**Worked 6.3 A**：$y''+4y'+3y=0$。$(\lambda+1)(\lambda+3)=0$，$\lambda=-1,-3$，$y=c_1e^{-t}+c_2e^{-3t}\to0$（稳定）。companion 形式 $A=\begin{bmatrix}0&1\\-3&-4\end{bmatrix}$ 给出同样的特征方程与特征向量 $(1,-1),(1,-3)$。两条路必得同一个答案——它只能是同一个答案。

### 8.5 Example 3：圆周运动（无阻尼振荡）

$y''+y=0$（$m=1,k=1,b=0$）：$\lambda^2+1=0$，$\lambda=\pm i$。状态形式 $A=\begin{bmatrix}0&1\\-1&0\end{bmatrix}$（反对称！），$u(0)=(1,0)$：

$$
u(t)=\tfrac12 e^{it}\begin{bmatrix}1\\-i\end{bmatrix}+\tfrac12 e^{-it}\begin{bmatrix}1\\i\end{bmatrix}
=\begin{bmatrix}\cos t\\-\sin t\end{bmatrix}.
$$

（Euler 公式 $e^{\pm it}=\cos t\pm i\sin t$ 把复数对撞回实数。）$u$ 沿单位圆转圈：$\cos^2t+\sin^2t=1$，**半径永不变**——无阻尼 = 能量守恒。

### 8.6 Worked 6.3 B：热方程与波方程（−K 矩阵）

$$
A=\begin{bmatrix}-2&1&0\\1&-2&1\\0&1&-2\end{bmatrix}=-K
\qquad(K=\mathrm{tridiag}(-1,2,-1),\ \text{LA 04/05/09 的三对角老朋友、路径图的拉普拉斯}).
$$

$$
\det(A-\lambda I)=(-2-\lambda)\bigl[(-2-\lambda)^2-2\bigr]=0
\quad\Longrightarrow\quad
\lambda=-2,\ -2\pm\sqrt2
$$

（全负；trace $-6=-2+(-2+\sqrt2)+(-2-\sqrt2)$ ✓）。特征向量 $x_1=(1,0,-1)$、$x_2=(1,-\sqrt2,1)$、$x_3=(1,\sqrt2,1)$ 两两垂直（对称）。初始 $u(0)=(0,2\sqrt2,0)=x_3-x_2$ 干净利落。**同一个 $A$、两种时间演化**：

| 方程 | 物理直觉 | 解的形态 |
|------|---------|---------|
| $u'=Au$（热方程 $\partial u/\partial t=\partial^2u/\partial x^2$ 的离散版） | 中间盒子的热量向两边扩散、边界冻结在 0 | $u(t)=e^{\lambda_3t}x_3-e^{\lambda_2t}x_2$，全负 $\lambda$ ⟹ **衰减**（负定矩阵，LA_11 正式命名） |
| $u''=Au$（波方程 $\partial^2u/\partial t^2=\partial^2u/\partial x^2$） | 中间盒子被弹开、波向两边传播 | $\omega^2=-\lambda$ 给频率，解是 $2\cos\omega t\,x_i$ 的组合——**振荡** |

### 8.7 数值格式：连续看实部，离散看模长

在屏幕上画圆要把微分改差分。对 $y''=-y$ 用 $\bigl(Y_{n+1}-2Y_n+Y_{n-1}\bigr)/\Delta t^2$ 近似 $y''$，三种取法对应三个一步矩阵：

| 格式 | 一步矩阵 | 特征值 | 轨迹 |
|------|---------|--------|------|
| Forward（前向 Euler） | $\begin{bmatrix}1&\Delta t\\-\Delta t&1\end{bmatrix}$ | $1\pm i\Delta t$，模长 $>1$ | **螺旋外爆** |
| Backward（后向 Euler） | 上面的逆 | 模长 $<1$ | 螺旋内缩 |
| Centered（leapfrog 蛙跳） | 中心差分 | 模长 $=1$（$\Delta t<2$ 时） | **贴住圆** |

判据换轴了：连续系统 $e^{\lambda t}$ 看**实部**（§9），离散系统 $\lambda^k$ 看**模长**。leapfrog 是分子动力学的主力格式——计算科学的日常：一个微分方程身后跟着一族差分方程，有的稳定有的爆炸。LA_15（数值篇）会再遇到它们。

---

## 9. 稳定性判据与矩阵指数 e^{At}（§6.3 下）

### 9.1 谁衰减？2×2 的完整判据

解是纯指数解的组合，于是**稳定性 = 所有 $\lambda$ 实部为负**（$e^{\lambda t}=e^{rt}e^{ist}$，$|e^{ist}|=1$ 无功无过，$e^{rt}$ 定生死）。哪些矩阵过关？2×2 有完整答案（$T$=trace、$D$=det）：

$$
\boxed{\ A\ \text{稳定}\iff T<0\ \text{且}\ D>0\ }
$$

**双向证明**（教材全程）：$\lambda$ 全负实 ⟹ 和（$T$）为负、积（$D$）为正。反向：$D=\lambda_1\lambda_2>0$ ⟹ 两 $\lambda$ 同号；$T<0$ ⟹ 同负。复根情形 $\lambda=r\pm is$（$\lambda$ 必共轭，否则 $T,D$ 不实）：$D=(r+is)(r-is)=r^2+s^2>0$ **自动成立**，$T=2r<0$ 恰好管住实部。∎

$(T,D)$ 平面上抛物线 $T^2=4D$ 分开实根（线下）与复根（线上），稳定区是**左上角**：

| 例 | $T$ | $D$ | 判决 |
|----|-----|-----|------|
| $\begin{bmatrix}0&1\\-1&-2\end{bmatrix}$ | $-2$ | $1$ | 稳定（$\lambda=-1,-1$，实重根） |
| $\begin{bmatrix}0&4\\5&-6\end{bmatrix}$ | $-6$ | $-20$ | 不稳定（$D<0$：一正一负） |
| $\begin{bmatrix}0&1\\-1&0\end{bmatrix}$ | $0$ | $1$ | 中性（$\lambda=\pm i$：永转圈，§8.5） |

### 9.2 矩阵指数：级数定义

把 $e^x=1+x+\tfrac{x^2}{2!}+\cdots$ 里的 $x$ 换成矩阵 $At$：

$$
\boxed{\ e^{At}=I+At+\tfrac{(At)^2}{2!}+\tfrac{(At)^3}{3!}+\cdots\ }
$$

三件保证：级数**永远收敛**（阶乘碾压矩阵幂）；逐项求导得 $\dfrac{d}{dt}e^{At}=Ae^{At}$，故 $u(t)=e^{At}u(0)$ 解一切 $u'=Au$——**特征向量短缺也照算**（这是它超过三步法的地方）；特征值逐个变 $e^{\lambda t}$（把级数作用在 $x$ 上，$x$ 的系数自动聚成 $e^{\lambda t}$）。

### 9.3 对角化版：e^{At} = Xe^{Λt}X⁻¹

可对角化时把 $A=X\Lambda X^{-1}$ 塞进级数，**中间的 $X^{-1}X$ 对对相消**：

$$
e^{At}=X\Bigl[I+\Lambda t+\tfrac{(\Lambda t)^2}{2!}+\cdots\Bigr]X^{-1}
=\boxed{\ Xe^{\Lambda t}X^{-1}\ },\qquad
e^{\Lambda t}=\mathrm{diag}\bigl(e^{\lambda_1t},\dots,e^{\lambda_nt}\bigr).
$$

与三步法逐字对应：$Xc=u(0)$、乘 $e^{\lambda_it}$、合并。**两条鸿沟上的桥都通了**：$A^{-1}b$ 解 $Ax=b$，$e^{At}u(0)$ 解 $u'=Au$——都是"不求也知道答案长什么样"的短公式。

### 9.4 三个 Examples：短缺、旋转与三角

**Example 4（短缺时 te^{λt} 自动长出来）**：$y''-2y'+y=0$ 有重根 $(\lambda-1)^2=0$。状态矩阵 $A=\begin{bmatrix}0&1\\-1&2\end{bmatrix}$，$\lambda=1,1$ 只有特征向量 $(1,1)$ 一条线——对角化失败，但矩阵指数不慌：

$$
e^{At}=e^t\,e^{(A-I)t}=e^t\bigl[I+(A-I)t\bigr],
\qquad (A-I)^2=\mathbf 0\ (\text{级数两项截断!}),
$$

$$
e^{At}=e^t\begin{bmatrix}1-t&t\\-t&1+t\end{bmatrix},
\qquad
y(t)=e^t\bigl[(1-t)\,y(0)+t\,y'(0)\bigr].
$$

微分方程课上"再找一个 $te^t$"的口诀，在线代里是 $(A-I)^2=0$ 的自动产物。

**Example 5（旋转与正交）**：$A=\begin{bmatrix}0&1\\-1&0\end{bmatrix}$。$A^4=I$，幂按 $A,A^2,A^3,A^4=A,\dots$ 循环，右上角轮流 $1,0,-1,0$——级数里长出的正是 $\cos$ 与 $\sin$ 的级数：

$$
e^{At}=\begin{bmatrix}\cos t&\sin t\\-\sin t&\cos t\end{bmatrix}.
$$

三条规则（本例全部现形）：① $e^{At}$ 永远可逆，逆是 $e^{-At}$；② $e^{At}$ 的特征值是 $e^{\lambda t}$（此处 $e^{\pm it}$，模 1）；③ **$A$ 反对称 ⟹ $e^{At}$ 正交**（$e^{-At}=(e^{At})^T$）⟹ $\|u(t)\|=\|u(0)\|$：能量守恒、中性稳定、纯振荡。

**Example 6（三角）**：$A=\begin{bmatrix}1&1\\0&2\end{bmatrix}$，$\lambda=1,2$、$x_1=(1,0)$、$x_2=(1,1)$，$u(0)=(1,1)=x_1+x_2$：

$$
u(t)=e^t\begin{bmatrix}1\\0\end{bmatrix}+e^{2t}\begin{bmatrix}1\\1\end{bmatrix},
\qquad
e^{At}=\begin{bmatrix}e^t&e^{2t}-e^t\\0&e^{2t}\end{bmatrix}.
$$

**Worked 6.3 C（幂零：级数变多项式）**：$da/dt=0,\ db/dt=a,\ dc/dt=2b,\ dz/dt=3c$。系数矩阵严格下三角、幂零（$A^4=\mathbf 0$，对角线每平方下移一格），级数四项封顶：

$$
e^{At}=\begin{bmatrix}1&0&0&0\\ t&1&0&0\\ t^2&2t&1&0\\ t^3&3t^2&3t&1\end{bmatrix},
\qquad
\begin{aligned}
a(t)&=a(0)\\
b(t)&=t\,a(0)+b(0)\\
c(t)&=t^2a(0)+2t\,b(0)+c(0)\\
z(t)&=t^3a(0)+3t^2b(0)+3t\,c(0)+z(0)
\end{aligned}
$$

逐个积分与矩阵指数给出**同一个** $e^{At}$——必然如此。最后一条性质收尾：$e^{As}\cdot e^{At}=e^{A(s+t)}$ 恒成立（同一 $A$ 的指数可自由合并）；但 $e^{A+B}\neq e^A e^B$（除非 $AB=BA$——交叉项 $AB+BA$ 对不上，和 §4.3 的陷阱同源）。

---

## 10. 例子与反例速查

| 命题 | 真假 | 例子 / 反例 |
|------|------|------------|
| 消元保持特征值 | ❌ | $\begin{bmatrix}1&3\\2&6\end{bmatrix}$（$\lambda=0,7$）消元到 $\begin{bmatrix}1&3\\0&0\end{bmatrix}$（$\lambda=0,1$） |
| 三角矩阵的 $\lambda$ 在对角线上 | ✅ | $\det(A-\lambda I)=(a_{11}-\lambda)\cdots(a_{nn}-\lambda)$ |
| 无负元素 ⟹ 无负特征值 | ❌ | $R=\begin{bmatrix}0&1\\1&0\end{bmatrix}$ 全非负，$\lambda=-1$ |
| 实矩阵的特征值必为实数 | ❌ | 旋转 $Q$：$\lambda=\pm i$；一般 $\cos\theta\pm i\sin\theta$ |
| $\lambda(A+B)=\lambda(A)+\lambda(B)$ | ❌ | 两个幂零阵相加得 $\lambda=\pm1$（§4.3 反例） |
| $\lambda(AB)=\lambda(A)\lambda(B)$ | ❌ | 两个零谱阵相乘出 $\lambda=1$；但 AB 与 BA 的**非零**特征值相同（PS 6.1 #6/#39） |
| $\det(A-\lambda_1I)$ 的列沿 $x_2$ 方向 | ✅ | PS 6.1 #11：其零空间是 $x_1$ 的线，列空间（2×2）只剩 $x_2$ |
| 不同 $\lambda$ 的特征向量必无关 | ✅ | §5.2 两行证明（乘 $A$、乘 $\lambda_j$、相减） |
| 重根 ⟹ 不可对角化 | ❌ | $A=I$、$5I-J_4$（$\lambda=5$ 三重，GM=3）；短缺只是**可能** |
| 可逆 ⟺ 可对角化 | ❌ | $\begin{bmatrix}1&1\\0&1\end{bmatrix}$ 可逆不可对角化；$\begin{bmatrix}1&0\\0&0\end{bmatrix}$ 反之 |
| 每个特征值至少有一个特征向量 | ✅ | $A-\lambda I$ 奇异 ⟹ $\dim N(A-\lambda I)\ge1$ |
| $A^k\to\mathbf 0\iff$ 所有 $\lvert\lambda\rvert<1$ | ✅ | 差分世界看**模长**；$\lvert\lambda\rvert=1$ 不衰减（旋转、$A=I$） |
| $A^2=A$ ⟹ 可对角化 | ✅ | $\lambda\in\{0,1\}$；$C(A)$ 与 $N(A)$ 恰好凑满 $n$ 个特征向量 |
| $A$ 与 $A^T$ 特征值相同 | ✅ | $\det(A-\lambda I)=\det(A^T-\lambda I)$；但特征向量**不同**（PS #21） |
| 相似矩阵特征值相同 | ✅ | $(BCB^{-1})(Bx)=\lambda(Bx)$；特征向量不同 |
| $e^{A+B}=e^Ae^B$ | ❌ | 除非 $AB=BA$；$e^{As}e^{At}=e^{A(s+t)}$ 恒真（同一个 $A$） |
| $\det A=125$ 与生成树无关 | ❌ | $5I-J_4$ 是 $K_5$ 约化拉普拉斯，$125=5^3$ 恰数出 $K_5$ 的生成树 |

---

## 11. 易混淆辨析

| 概念对 | 辨析 |
|--------|------|
| **先求 $\lambda$ 还是先求 $x$** | $\lambda$ 先（$\det(A-\lambda I)=0$，只含 $\lambda$），$x$ 后（对每个 $\lambda$ 解 $(A-\lambda I)x=\mathbf 0$）。顺序反了寸步难行 |
| **$\det(A-\lambda I)$ vs $\det(\lambda I-A)$** | 同一根的两种写法，差 $(-1)^n$ 倍；教材用前者。别在 3 阶时把符号吵混——检验靠 trace 与 det |
| **trace 检验 vs det 检验** | 和与积。2×2 时二者合起来**完全决定** $\lambda$；$n\times n$ 时只是报警器（错解大概率过不了检验） |
| **AM vs GM** | 代数重数数根、几何重数数向量；GM ≤ AM 恒成立；短缺（GM<AM）⟹ 不可对角化 |
| **可逆 vs 可对角化** | 可逆看**值**（$\lambda=0$?）、可对角化看**向量**（够 $n$ 个吗）。四个象限全有居民 |
| **差分稳定 vs 微分稳定** | $u_{k+1}=Au_k$：所有 $\lvert\lambda\rvert<1$；$u'=Au$：所有 $\mathrm{Re}\,\lambda<0$。§8.7 前向 Euler 把衰减的微分方程变成发散的差分方程——判据换轴的现场事故 |
| **$A^k$ vs $e^{At}$** | 同一三步法、不同增长因子：$\lambda^k$ vs $e^{\lambda t}$。对角化失效时 $A^k$ 无捷径，$e^{At}$ 还有级数兜底（Example 4） |
| **相似 vs 相等** | 相似共享 $\lambda$ 不共享 $x$；$A=X\Lambda X^{-1}$ 说"$A$ 在特征向量这组基下**就是** $\Lambda$"——LA 13（基变换）的正题 |
| **companion 矩阵的特征向量** | $(1,\lambda_i)$ 是模板：第一行说 $y'=y'$，第二行说 $y''=-ky-by'$。二阶方程的两个初始条件 $(y(0),y'(0))$ 恰是二维状态向量的坐标 |
| **$e^{At}$ 的三条铁律** | 可逆（$e^{-At}$）；特征值 $e^{\lambda t}$；反对称 ⟹ 正交保长度。第三条是"能量守恒"的矩阵语言 |

---

## 12. ML / 应用连接

- **PageRank 与幂法（power iteration）**：Markov 矩阵 $\lambda_1=1$ 的特征向量就是稳态分布。Google 实际算法是幂法：任取 $u_0$ 反复乘 $A$，$u_k=c_1(1)^kx_1+c_2\lambda_2^kx_2+\cdots$ 中只剩 $\lambda_1$ 项活下来——收敛速率由 $\lvert\lambda_2/\lambda_1\rvert$ 决定（这也是推荐系统、谱聚类取 top-$k$ 特征向量的通用武器）；
- **RNN 梯度消失/爆炸的精确诊断**：沿时间反向传播 = 重复乘同一个权重矩阵（$u_{k+1}=Au_k$ 的转置版）。谱半径 $<1$ 梯度消失、$>1$ 爆炸——本篇 $A^k\to0\iff\lvert\lambda\rvert<1$ 就是那句工程结论的数学本体。LSTM/GRU 的门控、梯度裁剪、正交初始化（$\lvert\lambda\rvert=1$ 的中性区，§9 Example 5 的保长度）全是围绕这条判据的工程对策；
- **图拉普拉斯与谱聚类**：Worked 6.3 B 的 $-K$ 是路径图的拉普拉斯、Worked 6.2 B 的 $5I-J$ 是完全图的拉普拉斯——**图的全部谱理论就是本篇**。第二小特征值 $\lambda_2$（Fiedler value）度量图连通性、指引谱聚类把图切成两半；GNN 的消息传递算子谱也在同一条轴上（LA_16 展开）；
- **Neural ODE 与连续深度**：$du/dt=f_\theta(u,t)$ 的线性版 $u'=Au$ 解是 $e^{At}u(0)$——把残差网络的离散层 $u_{k+1}=u_k+f(u_k)$ 加密到时间连续的极限，就是 Neural ODE；训练用的 adjoint 方法是"反向传播的连续化"，而稳定性分析（何时流映射收缩）用的正是本篇的谱判据；
- **数值积分器与训练稳定性同源**：§8.7 前向 Euler 螺旋外爆 = 学习率过大时梯度发散的同款现象——都是"把稳定连续系统离散化后判据从实部跳到模长"的代价。leapfrog/symplectic 格式保能量（反对称 ⟹ $e^{At}$ 正交保范数）的思想，在对抗训练的正交约束、RNN 的正交循环权重里反复还魂；
- **快速幂**：算 $F_{100}$ 不必乘 100 次，同理算 $A^k$ 可以 $O(\log k)$ 次矩阵乘（平方—折叠）；RSA 的模幂、Fibonacci heap、AVL 树高度界的 $\varphi^k$ 增长，全是本篇的直系亲属；
- **别用特征多项式求根**：5 次以上无根式解（Galois），数值上展开 $\det(A-\lambda I)$ 又严重病态。生产代码走 **QR 算法**（LA_04 的 Householder × 迭代），`np.linalg.eig` 的黑箱内部就是它——LA_15 数值篇开箱。与 LA_09 的"别用 det 判奇异"并列为本系列两条数值军规。

---

## 13. 习题精选（Problem Set 6.1–6.3 选摘）

**题 1**（Worked 6.1 A + PS 6.1 #2–4）：$A=\begin{bmatrix}2&-1\\-1&2\end{bmatrix}$。求 $A$、$A^2$、$A^{-1}$、$A+4I$ 的特征值与特征向量，并验证 trace 与 det。若把 $A$ 换成 $A+I$，特征向量变吗？特征值呢？
*提示*：$\lambda=1,3$、$x=(1,1),(1,-1)$ 全家共享；$A^2$ 得 $1,9$（$A^2=\begin{bmatrix}5&-4\\-4&5\end{bmatrix}$）、$A^{-1}$ 得 $1,\tfrac13$、$A+4I$ 得 $5,7$。**特征向量不动，特征值按函数走**——这就是对角化的全部好处。

**题 2**（6.1 #19，由 $\lambda$ 反推）：3×3 矩阵 $B$ 已知 $\lambda=0,1,2$。能确定：(a) rank？(b) $\det(B^TB)$？(c) $B^TB$ 的特征值？(d) $(B^2+I)^{-1}$ 的特征值？
*提示*：(a) $\lambda=0$ 只出现一次 ⟹ $\dim N(B)=1$ ⟹ rank $=2$；(b) $\det(B^TB)=\det B^T\det B=0$；(c) **不能**——$B^TB$ 的谱不由 $B$ 的谱决定（对称化的信息在 $\lambda$ 之外）；(d) 函数 $1/(\lambda^2+1)$ 逐个代入：$1,\ \tfrac12,\ \tfrac15$。

**题 3**（6.1 #10，Markov 极限）：$A=\begin{bmatrix}.6&.2\\.4&.8\end{bmatrix}$。求 $A$ 与 $A^\infty$ 的特征值，解释为什么 $A^{100}\approx A^\infty$。
*提示*：trace $1.4$、det $0.4$ ⟹ $\lambda=1,\ 0.4$。$\lambda=1$ 的特征向量解 $(A-I)x=0$ 得 $(1,2)/3$，故 $A^\infty=\begin{bmatrix}1/3&1/3\\2/3&2/3\end{bmatrix}$；$0.4^{100}\approx0$。

**题 4**（6.1 #27，全 1 家族）：求 4×4 全 1 矩阵 $A$ 与 $C=J-I$（对角 0、其余 1）的秩和特征值。
*提示*：$A$ 秩一：$\lambda=4,0,0,0$；$C=J-I$：$\lambda=4-1=3$ 与 $0-1=-1$（三重），$\det C=3\cdot(-1)^3=-3$。**秩一 ⟹ 至多一个非零 $\lambda$，且它就是 trace**（LA 06 的 $uv^T$）。

**题 5**（6.2 #18，闭式幂）：对 $A=\begin{bmatrix}2&-1\\-1&2\end{bmatrix}$ 证明 $A^k=\dfrac12\begin{bmatrix}3^k+1&3^k-1\\3^k-1&3^k+1\end{bmatrix}$。
*提示*：$X=\begin{bmatrix}1&1\\1&-1\end{bmatrix}$，$X^{-1}=\tfrac12X$；$A^k=X\,\mathrm{diag}(3^k,1)\,X^{-1}$。$k=1$ 验算 $\tfrac12\begin{bmatrix}4&2\\2&4\end{bmatrix}=A$ ✓。

**题 6**（6.2 #20–22，trace 三连）：(a) 由 $A=X\Lambda X^{-1}$ 证 $\det A=\lambda_1\cdots\lambda_n$；(b) 证 $\mathrm{tr}(XY)=\mathrm{tr}(YX)$，取 $Y=\Lambda X^{-1}$ 推出 $\mathrm{tr}\,A=\mathrm{tr}\,\Lambda$；(c) 证明有限维矩阵不可能 $AB-BA=I$。
*提示*：(a) $\det(X\Lambda X^{-1})=\det X\det\Lambda\det X^{-1}$；(b) 对角元逐个对账 $\sum_i\sum_j x_{ij}y_{ji}=\sum_j\sum_i y_{ji}x_{ij}$；(c) 左边迹为零（(b) 的推论）、右边迹为 $n$。**Heisenberg 的 $QP-PQ=I$ 只活在无限维**。

**题 7**（6.2 #33，Lax 的旋转幂）：$B=\begin{bmatrix}3&2\\-5&-3\end{bmatrix}$、$C=\begin{bmatrix}5&7\\-3&-4\end{bmatrix}$。用特征值 $e^{i\theta}$ 证明 $B^4=I$、$C^3=-I$，并用直接乘法验证。
*提示*：$B$：trace 0、det 1 ⟹ $\lambda=\pm i=e^{\pm i\pi/2}$ ⟹ $B^4$ 特征值全 1，且 $B$ 可对角化 ⟹ $B^4=I$（直接乘 $B^2=-I$ ✓）。$C$：trace 1、det 1 ⟹ $\lambda^2-\lambda+1=0$，$\lambda=e^{\pm i\pi/3}$ ⟹ $C^3=-I$（直接乘 ✓）。**谱在单位圆上 ⟹ 幂有周期**——$C^{1024}=C^{3\cdot341+1}=-C$。

**题 8**（6.3 #1–2，两条路）：解 $du/dt=\begin{bmatrix}4&3\\0&1\end{bmatrix}u$，$u(0)=(5,-2)$。(a) 特征值法；(b) 逐分量回代（先解 $z$ 后解 $y$）。
*提示*：(a) 三角 ⟹ $\lambda=4,1$；$x_1=(1,0)$、$x_2=(1,-1)$；$u(0)=3x_1+2x_2$，$u(t)=3e^{4t}(1,0)+2e^t(1,-1)$。(b) $z' = z$ 给 $z=-2e^t$；代入 $y'=4y+3z$ 解一阶线性方程，得 $y=3e^{4t}+2e^t$——同一答案。**上三角系统的"先解后回代"就是 LA 02 回代的微分版**。

---

## 14. 逻辑推导概念图

下面把本篇的逻辑骨架画成一棵树：顶部是出发点（定义与假设），箭头是推理方向，每一步只使用它上游的结论。读者可以沿箭头反向核对——某个定理到底用了哪条假设、抽掉哪一条会塌。

```text
出发点：定义 Ax = λx（x ≠ 0）
│
├─→ (A−λI)x = 0 有非零解 ⟺ A−λI 奇异
│        └─→ 特征方程 det(A−λI) = 0（LA 09 的行列式在此兑现）
│                 ├─→ n 次多项式、n 个根（可重复、可为复数：旋转矩阵 λ = ±i）
│                 ├─→ 三角矩阵的 λ 恒在对角线上
│                 └─→ 两个快速检验：Σλ = trace A、Πλ = det A
│
├─→ 对 Ax = λx 再做一次运算：A^2、A^{-1}、A+cI 共享特征向量，λ 按函数走
│
├─→ 猜解 u = e^{λt}x 代回 du/dt = Au，e^{λt} 约掉
│        └─→ 纯指数解：实部定生灭、虚部出振荡
│
├─→ 不同 λ ⟹ 特征向量必无关（乘 A、乘 λ、两式相减的两行证明）
│        └─→ 无重根 ⟹ 自动可对角化
│
└─→ 出发点：A 有 n 个线性无关特征向量（可对角化条件）
         ├─→ 对角化 A = XΛX^{-1}（X 列无关 ⟹ 可逆）
         │        ├─→ A^k = XΛ^kX^{-1}：幂变成对角线的运算
         │        │        └─→ Fibonacci / Lucas 闭式、Markov 极限 A^∞
         │        └─→ u(t) = Xe^{Λt}X^{-1}u(0)：微分版三步法
         │                 └─→ 稳定 ⟺ 全部 Re λ < 0（2×2 判据：T<0 且 D>0）
         └─→ 短缺（GM < AM）时三步法失灵
                  └─→ 矩阵指数 e^{At} 级数兜底：(A−I)^2 = 0 自动长出 te^{λt}
```

| 出发点 | 买到什么 | 撤掉它失去什么 |
|---|---|---|
| 定义 $Ax=\lambda x$（被 $A$ 乘后不改变方向的向量） | 特征方程 $\det(A-\lambda I)=0$、求解三步、trace 与 det 两个快速检验 | 消元救不了你：$\begin{bmatrix}1&3\\2&6\end{bmatrix}$（$\lambda=0,7$）消元到 $\begin{bmatrix}1&3\\0&0\end{bmatrix}$（$\lambda=0,1$）——LA 02–04 的机器在此失灵 |
| $A$ 有 $n$ 个线性无关特征向量 | $A=X\Lambda X^{-1}$、$A^k=X\Lambda^kX^{-1}$、差分与微分两套三步法 | $\begin{bmatrix}1&1\\0&1\end{bmatrix}$ 可逆却不可对角化：$A^k$ 无捷径，只剩 $e^{At}$ 级数兜底 |
| $\lambda$ 互不相同 | 不同 $\lambda$ 的特征向量必无关，无重根自动可对角化 | 重根是"可能短缺"而非"必然短缺"：$A=I$、$5I-J_4$（$\lambda=5$ 三重、GM$=$3）照样管够，$\begin{bmatrix}5&1\\0&5\end{bmatrix}$ 却只剩一条线——对称才是保票（LA_11） |
| 方程线性、常系数（$A$ 不随 $t$、不随 $u$ 变） | 纯指数解 $u=e^{\lambda t}x$：$e^{\lambda t}$ 约掉，回到 $Ax=\lambda x$ | 时变与非线性方程不归线性代数管——猜解的前提整个消失 |
| $A$ 实对称（本篇记账，LA_11 兑现） | 特征向量自动两两垂直（Worked 6.1 C、Worked 6.3 B）；反对称 ⟹ $e^{At}$ 正交保长度 | 非对称可能短缺（$\begin{bmatrix}6&-1\\1&4\end{bmatrix}$ 的 5,5 病灶）、$\lambda$ 可为复（旋转矩阵 $\pm i$） |

---

## 15. 小结与下一篇

| 对象 | 公式 | 一句话 |
|------|------|--------|
| 特征方程 | $\det(A-\lambda I)=0$ | LA 09 的行列式换来的 $n$ 个根 |
| 两个检验 | $\prod\lambda=\det A$、$\sum\lambda=\mathrm{tr}A$ | 算错的报警器；2×2 的完全决定者 |
| 对角化 | $A=X\Lambda X^{-1}$ | $n$ 个无关特征向量 ⟺ 可对角化；无重根 ⟹ 自动 |
| 矩阵幂 | $A^k=X\Lambda^kX^{-1}$ | Fibonacci：$F_k=(\lambda_1^k-\lambda_2^k)/\sqrt5$ |
| 重数 | $GM\le AM$ | 短缺 ⟹ 不可对角化（$te^{\lambda t}$ 兜底） |
| 微分方程 | $u(t)=Xe^{\Lambda t}X^{-1}u(0)$ | 三步法；稳定 ⟺ 全部 $\mathrm{Re}\,\lambda<0$ |
| 矩阵指数 | $e^{At}=\sum(At)^k/k!$ | 永远收敛；短缺也有解；反对称 ⟹ 正交 |

本篇反复撞见同一堵墙：**对称矩阵总是格外乖**——Worked 6.1 C 与 Worked 6.3 B 的特征向量自动垂直、Example 5 的反对称给出正交的 $e^{At}$、$\begin{bmatrix}6&-1\\1&4\end{bmatrix}$ 提醒非对称可能短缺。下一篇正面回答：**实对称矩阵的特征值全是实数、特征向量总能选成标准正交基**——于是对角化升级为 $A=Q\Lambda Q^T$（正交矩阵进出、不用求逆），再往前一步就是正定矩阵（$\lambda$ 全正）与它的五重判据——机器学习里 $A^TA$、协方差、Hessian、核矩阵的公共老家。

---

*下一篇：[线性代数 11：对称矩阵与正定矩阵](../LA_11_symmetric_positive_definite/)*
