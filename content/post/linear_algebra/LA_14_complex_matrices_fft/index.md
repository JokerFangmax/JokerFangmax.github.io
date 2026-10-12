---
title: "线性代数 14: 复矩阵、Hermitian 与 FFT"
date: 2026-11-14
math: true
tags:
  - Notes
  - LinearAlgebra
categories:
  - LinearAlgebra
description: "Strang 第 5 版 §9.1-9.3 精读——复数与 n 次单位根（Euler 公式、z^n=1 的单位圆根）、实矩阵的复特征值成对共轭、共轭转置 A^H 与复内积、Hermitian 矩阵的实特征值与正交特征向量、酉矩阵与复谱定理 S=UΛU^H、Fourier 矩阵 F_n 的正交性、循环矩阵对角化（周期 -1,2,-1 矩阵特征值 0,2,4,2）、FFT 分治因子分解与 ½n log₂ n 计数、bit 反转与快速卷积"
---

# 线性代数 14：复矩阵、Hermitian 与 FFT

> 📖 本篇覆盖 Strang《Introduction to Linear Algebra》第 5 版 §9.1–9.3。前置：[LA 13](../LA_13_linear_transformations/)（循环矩阵被 Fourier 基对角化的预告）。本篇是复数世界的完整入场券：全系列用惯的转置 $T$ 在此升级为共轭转置 $H$，而压轴的 FFT 被 Strang 称为"上个世纪最有价值的数值算法"。

---

## 1. 这一篇解决什么问题？

[LA 10](../LA_10_eigenvalues_diagonalization/) 早已撞见这堵墙：$90°$ 旋转矩阵 $\begin{bmatrix}0&-1\\1&0\end{bmatrix}$ 没有实特征值——特征方程 $\lambda^2+1=0$ 在 $\mathbb R$ 里无解，而在 $\mathbb C$ 里答案是 $\lambda=\pm i$，特征向量 $(1,\pm i)$。Strang 开章明义：**即使矩阵是实的，特征值与特征向量也常常是复的**——完整的线性代数必须包含复数（"每个 $n\times n$ 方阵有 $n$ 个特征值"这句话本身就是代数基本定理，没有复数它不成立）。

本篇三步走：

1. **复数速成**（§9.1）：从 $i^2=-1$ 到 Euler 公式 $e^{i\theta}=\cos\theta+i\sin\theta$，核心道具是 $n$ 次单位根 $1,w,w^2,\dots,w^{n-1}$——离散 Fourier 变换的全部原料；
2. **实 → 复词典**（§9.2）：把全系列的 $T$ 换成 $H$（共轭转置），对称 → Hermitian、正交 → 酉——**实特征值与正交特征向量这两大好性质原封不动地保留**；
3. **Fourier 矩阵与 FFT**（§9.3）：全书最重要的复矩阵 $F_n$。[LA 13](../LA_13_linear_transformations/) 埋下的"循环矩阵被 $F$ 对角化"在此完整兑现，而 FFT 把乘 $F_n$ 的成本从 $n^2$ 次乘法砍到 $\tfrac12 n\log_2 n$。

一句话主线（Strang 自评 unforgettable words）：

> 转置任何向量或矩阵时，把每个元素换成它的共轭（$i$ 变成 $-i$）。

前篇伏笔兑现表：

| 来源 | 埋下的话 | 本篇兑现 |
|------|---------|---------|
| LA 10 复特征值 | 旋转矩阵 $\lambda=\pm i$、特征向量 $(1,\pm i)$ | §3 共轭成对定理 |
| LA 12 §5 正常矩阵 | $A^TA=AA^T$ 的矩阵"最像对称" | §6 正常 ⟺ 可酉对角化 |
| LA 13 循环矩阵 | $P$ 特征向量 $=F$ 列、卷积定理预告 | §8–9 全套兑现 |
| [LA 08](../LA_08_least_squares_gram_schmidt/) Hadamard | $H_{2n}$ 递推与 FFT 分治同构 | §9.4 |
| LA 11 DST/DCT | 正弦族是实数世界的 Fourier | §7 复指数统一版 |

---

## 2. 复数速成：单位圆、Euler 与 n 次单位根（§9.1）

### 2.1 四则与共轭

复数 $z=a+ib$：加法像平面向量（实部虚部分开加），乘法只需一条新规则 $i^2=-1$：

$$
(3+2i)+(3+2i)=6+4i,\qquad
(3+2i)(1-i)=3-3i+2i-2i^2=5-i,
$$

$$
(1+i)^2=1+i+i+i^2=2i\qquad(\text{角度 }45°\ \text{翻倍成 }90°).
$$

**共轭**（conjugate）$\bar z=a-ib$ 是关于实轴的镜像。共轭与加、乘完全相容（和的共轭 = 共轭的和，积同理），于是有一个立刻的重要推论——实矩阵的复特征值**成对出现**（§3）。而 $z$ 与 $\bar z$ 自己碰到一起时，虚部必然抵消：

$$
z+\bar z=2a\ (\text{实}),\qquad
z\bar z=a^2+b^2=|z|^2\ (\text{实}),\qquad
(3+2i)(3-2i)=13.
$$

**除法**的诀窍是分子分母同乘 $\bar z$，把分母变成实数：

$$
\frac{1}{3+2i}=\frac{3-2i}{(3+2i)(3-2i)}=\frac{3-2i}{13}.
$$

单位圆上的数 $|z|=1$ 有最漂亮的特例：$z\bar z=1$，即 $1/z=\bar z$。

### 2.2 极坐标与 Euler 公式：模相乘、角相加

绝对值 $|z|=\sqrt{a^2+b^2}$（也叫模，记 $r$）给出大小；角度 $\theta$ 给出方向（$\tan\theta=b/a$）。三角恒等式 $a=r\cos\theta,\ b=r\sin\theta$ 加上 Euler 公式给出极坐标形态：

$$
\boxed{\ e^{i\theta}=\cos\theta+i\sin\theta\ }
\qquad\Longrightarrow\qquad
z=a+ib=r\cos\theta+ir\sin\theta=re^{i\theta}.
$$

（Euler 公式本身可从幂级数看见：$e^x=1+x+\frac{x^2}2+\cdots$ 里代入 $x=i\theta$，偶次幂拼出 $\cos\theta$、奇次幂拼出 $\sin\theta$。）极坐标下**乘幂都变简单**：模取幂、角翻倍——二倍角公式就是 $(\cos\theta+i\sin\theta)^2$ 展开后实部 $\cos^2\theta-\sin^2\theta=\cos2\theta$、虚部 $2\sin\theta\cos\theta=\sin2\theta$。Strang 的现场演示：

$$
(1+i)^8:\quad r=\sqrt2\Rightarrow r^8=16,\qquad \theta=\tfrac\pi4\Rightarrow 8\theta=2\pi
\quad\Longrightarrow\quad (1+i)^8=16.
$$

### 2.3 n 次单位根：DFT 的全部原料

方程 $z^n=1$ 恰有 $n$ 个解（代数基本定理），等距分布在单位圆上。头号主角：

$$
\boxed{\ w=e^{2\pi i/n}\ }
\qquad\Longrightarrow\qquad
1,\ w,\ w^2,\ \dots,\ w^{n-1}\ \text{都满足}\ z^n=1.
$$

$n=8$ 的全景（$w_8=(1+i)/\sqrt2$，间隔 $45°$）：

| 幂 | 角度 | 直角坐标 |
|---|------|---------|
| $w^0$ | $0°$ | $1$ |
| $w$ | $45°$ | $(1+i)/\sqrt2$ |
| $w^2$ | $90°$ | $i$ |
| $w^3$ | $135°$ | $(-1+i)/\sqrt2$ |
| $w^4$ | $180°$ | $-1$ |
| $w^5$ | $225°$ | $(-1-i)/\sqrt2$ |
| $w^6$ | $270°$ | $-i$ |
| $w^7=w^{-1}=\bar w$ | $315°$ | $(1-i)/\sqrt2$ |

两个必记事实（§7 正交性证明的引擎）：

- **$w^m=-1$ 当 $n=2m$**：半圈就是 $-1$（$w^{n/2}=e^{\pi i}$）。FFT 的奇偶分裂全靠它；
- **全部单位根之和为零**：$S=1+w+\cdots+w^{n-1}$。给 $S$ 整体乘一个 $w$（每个数沿圆转一格）：$wS=w+w^2+\cdots+w^n=S$（每项后挪一位、末项 $w^n$ 绕回 $1$）。转一格后不动的和只能是零——$S=0$。（代数版：等比求和 $S=(w^n-1)/(w-1)=0$。）

Strang 的预告：**这 $n$ 个单位根是信号处理的关键数字**——离散 Fourier 变换用的就是 $w=e^{2\pi i/n}$ 与它的幂。

---

## 3. 实矩阵的复特征值：成对共轭（§9.1 下）

**定理（共轭特征值对）**：$A$ 是**实**矩阵且 $Ax=\lambda x$，则 $A\bar x=\bar\lambda\bar x$。

**证明**：对 $Ax=\lambda x$ 两边取共轭。矩阵是实的（$\bar A=A$），纯量的共轭逐个取：$A\bar x=\bar\lambda\bar x$。∎

复特征值永远**成对出现** $\lambda,\bar\lambda$，各自带共轭的特征向量 $x,\bar x$——LA 10 旋转矩阵的 $\pm i$ 与 $(1,\pm i)$ 是最小标本。

### 3.1 标本：旋转-缩放矩阵 [[a,b],[-b,a]]

$$
A=\begin{bmatrix}a&b\\-b&a\end{bmatrix},\qquad
\det(A-\lambda I)=(a-\lambda)^2+b^2=0
\quad\Longrightarrow\quad
\boxed{\ \lambda=a\pm bi\ },
$$

特征向量：对 $\lambda=a+bi$ 解 $(A-\lambda I)x=\mathbf 0$ 得 $x=(1,\ i)$（验证：$A(1,i)=(a+bi,\ -b+ai)=(a+bi)\,(1,i)$ ✓）。

这个矩阵的几何身份：$a=r\cos\theta,\ b=r\sin\theta$ 时 $A=r\,R_{-\theta}$（缩放 $r$ 倍 + 旋转 $-\theta$），特征值恰是极坐标 $re^{\mp i\theta}=a\pm bi$——**复数乘法 = 旋转 + 缩放**，§2.2 的极坐标在此对号入座。特例 $a=0,b=-1$：$90°$ 旋转、$\lambda=\pm i$。LA 12 §5 那个扰动实验（$\pm\tfrac1{10}\pm\tfrac i{10}$）的复数对，就是这类矩阵在数据里的样子。

### 3.2 何时出现复特征值：一个判据

实 $2\times2$ 矩阵 $\begin{bmatrix}a&b\\c&d\end{bmatrix}$ 的特征值由二次公式给出，判别式

$$
(a+d)^2-4(ad-bc)=(a-d)^2+4bc.
$$

| 情形 | 特征值 |
|------|--------|
| $(a-d)^2+4bc\ge0$ | 两个实数 |
| $(a-d)^2+4bc<0$ | 共轭复数对（$\Leftrightarrow$ 迹的平方 < 4 倍行列式） |
| $bc>0$ | 必为实数（上式恒正）——$b,c$ 同号的矩阵躲开了复数 |

---

## 4. 共轭转置 A^H：长度与内积的复数修正（§9.2 上）

### 4.1 z^T z 的失败

实向量的长度平方 $\|x\|^2=x_1^2+\cdots+x_n^2$ 直接搬到复数会塌方：对 $z=(1,\ i)$，

$$
z^Tz=1^2+i^2=0
$$
——非零向量的"长度"为零，还有些向量会算出复数长度。问题出在平方：我们想要的其实是 $|z_j|^2=a_j^2+b_j^2$，即 $z_j\bar z_j$。修正只需要一个动作——**转置时顺便共轭**：

$$
z^H:=\bar z^T,\qquad
\boxed{\ \|z\|^2=z^Hz=|z_1|^2+\cdots+|z_n|^2\ }
$$

修正后：$\|(1,i)\|^2=1+1=2$；$\|(1+i,\ 1-i)\|^2=2+2=4$。只有零向量长度为零，一切恢复正常。MATLAB 的转置命令 `'` 自动做共轭（$z'$ 就是 $z^H$）——工程师的世界里 $H$ 是默认值。

### 4.2 复内积：次序第一次变得重要

**定义**：$u,v$（实或复）的内积是 $u^Hv=\bar u_1v_1+\cdots+\bar u_nv_n$。

复数世界里 $u^Hv\ne v^Hu$：两者互为**共轭**（$v^Hu=\overline{u^Hv}$）。实数世界里两边一样，所以从没操心过次序；现在是"为更大好处忍受的小不便"。（谁取共轭是自由约定——Strang 共轭第一个，有的书共轭第二个——但整本书必须统一。）

**Example 1**（出乎意料的垂直）：$u=(1,\ i),\ v=(i,\ 1)$：

$$
u^Hv=\begin{bmatrix}1&-i\end{bmatrix}\begin{bmatrix}i\\1\end{bmatrix}=i-i=0.
$$

这两个向量看上去毫无垂直的理由，但 $u^Hv=0$ 就是复数世界的正交。同样 $(1,i)\perp(1,-i)$：$u^Hv=1+(-i)(-i)=1-1=0$ ✓。

### 4.3 A^H 与实 → 复词典

$A^H=\bar A^T$（共轭转置，也叫伴随 adjoint）。它存在的理由与 LA 04 的 $(Ax)^Ty=x^T(A^Ty)$ 完全平行：

$$
(Au)^Hv=u^H(A^Hv),\qquad (AB)^H=B^HA^H.
$$

全系列的地基词典（章首原表，值得整张抄走）：

| 实数世界 $\mathbb R^n$ | 复数世界 $\mathbb C^n$ |
|---|---|
| 长度 $\|x\|^2=x_1^2+\cdots+x_n^2$ | $\|z\|^2=\|z_1\|^2+\cdots+\|z_n\|^2$ |
| 转置 $(A^T)_{ij}=A_{ji}$ | 共轭转置 $(A^H)_{ij}=\bar A_{ji}$ |
| 内积 $x^Ty$ | 内积 $u^Hv$（次序有讲究） |
| $(Ax)^Ty=x^T(A^Ty)$ | $(Au)^Hv=u^H(A^Hv)$ |
| 正交 $x^Ty=0$ | 正交 $u^Hv=0$ |
| 对称 $S=S^T$，$S=Q\Lambda Q^T$ | Hermitian $S=S^H$，$S=U\Lambda U^H$ |
| 反对称 $K^T=-K$ | 反 Hermitian $K^H=-K$ |
| 正交 $Q^TQ=I$ | 酉 $U^HU=I$ |
| $\|Qx\|=\|x\|$ | $\|Uz\|=\|z\|$ |

（MATLAB 的 `eig` 求复特征值、`'` 做共轭转置——本节就是这些约定的数学出处。）

---

## 5. Hermitian 矩阵：好性质原封保留（§9.2 中）

### 5.1 定义与主例

**Hermitian**：$S=S^H$，逐元素条件 $S_{ij}=\overline{S_{ji}}$——**对角线必须实**（$S_{ii}=\bar S_{ii}$），对角线两侧互为共轭。实对称矩阵是特例（共轭不起作用）。主例：

$$
S=\begin{bmatrix}2&3-3i\\3+3i&5\end{bmatrix}
\qquad(\text{对角 }2,5\ \text{实；反对角互为共轭}).
$$

### 5.2 三大性质（每条证明都只有几行）

**性质 1（能量是实的）**：对任意复向量 $z$，数 $z^HSz$ 是实数。

*证明*：$z^HSz$ 是 $1\times1$，取共轭转置就是取共轭：$(z^HSz)^H=z^HS^H(z^H)^H=z^HSz$。等于自己共轭的数是实数。∎

展开主例看一眼：对角项 $2|z_1|^2+5|z_2|^2$ 显然实；非对角两项 $(3-3i)\bar z_1z_2+(3+3i)z_1\bar z_2$ 互为共轭、相加虚部抵消——整串必实。

**性质 2（特征值实）**：$Sz=\lambda z$ 两边左乘 $z^H$：$z^HSz=\lambda z^Hz$，于是

$$
\lambda=\frac{z^HSz}{z^Hz}
$$

分子实（性质 1）、分母实且正（长度平方）——$\lambda$ 实。∎

主例验证：$\det(S-\lambda I)=(2-\lambda)(5-\lambda)-|3-3i|^2=\lambda^2-7\lambda+10-18=(\lambda-8)(\lambda+1)$，$\lambda=8,\ -1$ ✓（迹 $7=2+5$ ✓、行列式 $-8$ ✓、$|3-3i|^2=9+9=18$ ✓）。

**性质 3（特征向量正交）**：$Sz=\lambda z,\ Sy=\mu y$，$\lambda\ne\mu$，则 $y^Hz=0$。

*证明*：$Sz=\lambda z$ 左乘 $y^H$；$Sy=\mu y$ 取共轭转置（用 $S^H=S$）得 $y^HS=\mu y^H$，右乘 $z$。两式左端相同：$\lambda\,y^Hz=\mu\,y^Hz$，$\lambda\ne\mu$ 迫使 $y^Hz=0$。∎

### 5.3 完整对角化：S = QΛQ^H

主例的两个单位特征向量（解 $(S-8I)z=0$ 与 $(S+I)y=0$）：

$$
\lambda=8:\ z=\begin{bmatrix}1\\1+i\end{bmatrix},\qquad
\lambda=-1:\ y=\begin{bmatrix}1-i\\-1\end{bmatrix},
$$

$$
y^Hz=(1+i)\cdot1+(-1)(1+i)=0\ ✓,\qquad
\|z\|^2=1+2=3=\|y\|^2.
$$

单位化后排进列，得对角化矩阵 $S=Q\Lambda Q^H$，$\Lambda=\mathrm{diag}(8,-1)$：

$$
Q=\frac1{\sqrt3}\begin{bmatrix}1&1-i\\1+i&-1\end{bmatrix}.
$$

**这个 $Q$ 是个彩蛋：它同时是 Hermitian（$Q^H=Q$）和酉矩阵**——下一节的主角提前登场。展开验证对角化成立：

$$
S=\frac{8\,zz^H-yy^H}{3}
=\frac13\left(\begin{bmatrix}8&8(1-i)\\8(1+i)&16\end{bmatrix}-\begin{bmatrix}2&-1+i\\-1-i&1\end{bmatrix}\right)
=\begin{bmatrix}2&3-3i\\3+3i&5\end{bmatrix}\ ✓.
$$

---

## 6. 酉矩阵与复谱定理（§9.2 下）

### 6.1 酉矩阵：复世界的正交矩阵

**酉**（unitary）：列正交规范的复方阵，即 $U^HU=I$，于是 $U^{-1}=U^H$。它保长度也保内积（LA 08 正交矩阵 $\|Qx\|=\|x\|$ 的复版）：

$$
\|Uz\|^2=z^HU^HUz=z^Hz=\|z\|^2.
$$

**特征值全在单位圆上**：$Uz=\lambda z$ 时 $\|z\|=\|Uz\|=|\lambda|\,\|z\|$，非零 $z$ 迫使 $|\lambda|=1$。∎

立刻套用 §5.3 的彩蛋：$Q$ 既 Hermitian（$\lambda$ 实）又酉（$|\lambda|=1$），实数里模为 1 的只有 $\pm1$，而 $\mathrm{tr}\,Q=0$——**$Q$ 的特征值恰是 $1$ 与 $-1$**。再由 $Q^H=Q$ 与 $U^{-1}=U^H$ 得 $Q^2=I$（照两次镜子复原，LA 08 反射矩阵的复数亲戚）。

**复谱定理**：$S$ Hermitian $\Longrightarrow$ $S=U\Lambda U^H$，$U$ 酉、$\Lambda$ 实对角。LA 11 实谱定理 $S=Q\Lambda Q^T$ 一字不差地搬家。

### 6.2 Example 3：F₃ 与"忘记共轭"陷阱

$w=e^{2\pi i/3}=-\tfrac12+\tfrac{\sqrt3}2 i$（$w^3=1$，$w^4=w$）。三阶 Fourier 矩阵：

$$
F_3=\begin{bmatrix}1&1&1\\1&w&w^2\\1&w^2&w\end{bmatrix}.
$$

它**对称**（$F=F^T$）但**不 Hermitian**（共轭会把 $w$ 换成 $w^2\ne w$）——对称与 Hermitian 在复世界分家了。它酉吗？检验列 2 与列 3 的内积，**忘记共轭**会算出灾难：

$$
\text{忘记共轭：}\ \tfrac13(1+w\cdot w^2+w^2\cdot w)=\tfrac13(1+1+1)=1\ne0\ (\text{错！})
$$

$$
\text{正确（}H\text{）：}\ \tfrac13(1+\bar w\,w^2+\bar w^2\,w)=\tfrac13(1+w+w^2)=0\ ✓.
$$

（$\bar w=w^2$、$\overline{w^2}=w$，最后一步用了 §2.3 的单位根求和。）结论：$F_3$ 是酉矩阵（除以 $\sqrt3$ 后）。列 1 ⊥ 列 2 同理：$1+w+w^2=0$——三个单位根首尾相接构成正三角形，重心在原点。

### 6.3 反 Hermitian 与正常矩阵：全家福

| 家族 | 定义 | 特征值 | 例子 |
|------|------|--------|------|
| Hermitian | $S^H=S$ | 全实 | $\begin{bmatrix}2&3-3i\\3+3i&5\end{bmatrix}$ |
| 反 Hermitian | $K^H=-K$ | 全纯虚 | $K=iS$（$S$ Hermitian）；实反对称是特例 |
| 酉 | $U^HU=I$ | 全在单位圆 $\|\lambda\|=1$ | $F_3/\sqrt3$ |

实反对称矩阵（LA 04 一瞥、$K^T=-K$）特征值纯虚，现在两行就证：$Kx=\lambda x$ 左乘 $x^H$ 得 $x^HKx=\lambda x^Hx$；再对 $x^HKx$ 取共轭转置（$K^H=-K$）：$x^HK^Hx=\bar\lambda x^Hx$，即 $-x^HKx=\bar\lambda x^Hx$。两式对照：$-\lambda=\bar\lambda$——纯虚。∎

**正常矩阵**（normal，PS 9.2 #28）：$N^HN=NN^H$。上面的三个家族全部正常，而正常性恰好刻画"可酉对角化"：

$$
N=U\Lambda U^H\ (\text{酉对角化})\ \Longleftrightarrow\ N^HN=NN^H.
$$

LA 12 §5 说过"$A^TA=AA^T$ 的矩阵最像对称"——那正是实正常矩阵：酉对角化 $N=U\Lambda U^H$ 时 $N^HN=U|\Lambda|^2U^H=NN^H$；不正常（如 $\begin{bmatrix}1&1\\0&1\end{bmatrix}$）就没有正交特征向量组。Hermitian、酉都是正常矩阵在"特征值摆哪儿"上的两个特例（实轴上 / 单位圆上）。

---

## 7. Fourier 矩阵 F_n（§9.3 上）

### 7.1 定义与 F_4

$n$ 阶 **Fourier 矩阵**：$w=e^{2\pi i/n}$，第 $(j,k)$ 元是 $w^{jk}$（行列编号从 0 到 $n-1$——第 0 行第 0 列全为 1）：

$$
F_n=\begin{bmatrix}
1&1&1&\cdots&1\\
1&w&w^2&\cdots&w^{n-1}\\
1&w^2&w^4&\cdots&w^{2(n-1)}\\
\vdots&&&\ddots&\vdots\\
1&w^{n-1}&w^{2(n-1)}&\cdots&w^{(n-1)^2}
\end{bmatrix},
\qquad
F_4\ (w=i)=\begin{bmatrix}
1&1&1&1\\
1&i&-1&-i\\
1&-1&1&-1\\
1&-i&-1&i
\end{bmatrix}.
$$

（$F_4$ 的指数全部 mod 4：$w^4=1,\ w^6=w^2,\ w^9=w$——下标的世界是 $\mathbb Z/n$，书中玩笑"在这个世界里 $2+2=0$"。）

$F_n$ 对称（$F=F^T$）但不 Hermitian（对角线 $1,w,w^4,\dots$ 非实）——§6.2 的 $F_3$ 是最小标本。

### 7.2 正交性与逆：F^H F = nI

列的长度平方：每列模平方和 $1+1+\cdots+1=n$；不同列 $j\ne k$ 的内积（记得共轭！$\overline{w^{jk}}=w^{-jk}$）：

$$
(\text{列}j)^H(\text{列}k)=\sum_{r=0}^{n-1}w^{r(k-j)}=\frac{w^{n(k-j)}-1}{w^{k-j}-1}=0
$$

（分子 $w^{n(k-j)}=1$，分母非零——又是单位根求和）。于是

$$
\boxed{\ F_n^HF_n=nI\ }
\qquad\Longrightarrow\qquad
F_n^{-1}=\frac{\bar F_n}{n}=\frac{F_n^H}{n}.
$$

**逆变换 = 把 $i$ 换成 $-i$ 再除 $n$**。酉化版本 $F_n/\sqrt n$ 所有奇异值全为 1（条件数 1，数值模范公民——LA 15 的预告）。

**约定警示**：Strang 的 $F$ 用 $w=e^{2\pi i/n}$、$y=Fc$ 从**频率空间**到**物理空间**；MATLAB 与 NumPy 的 `fft` 用共轭约定 $\bar w=e^{-2\pi i/n}$、`fft(y)` 从物理到频率。两套约定数学等价，读代码时先看约定。

### 7.3 乘 F 的含义：有限 Fourier 级数与插值

$y=F_nc$ 的分量：输入 $n$ 个系数 $c_0,\dots,c_{n-1}$，输出

$$
y_j=\sum_{k=0}^{n-1}c_k\,w^{jk}=\sum_{k=0}^{n-1}c_k\,e^{ikx_j},
\qquad x_j=\frac{2\pi j}{n}
$$

——在 $n$ 个等距点 $x_j$ 上求值的**有限 Fourier 级数**。$y_0$ 是系数和，$y_1$ 是在 $x=\frac\pi2$（$n=4$）处的值，依此类推；下一个点 $x=2\pi$ 绕回 $y_0$（$e^{2\pi i}=1$）——一切以 $n$ 为周期。

另一个视角（教材的 Important note）：**DFT = 插值**。找多项式 $p(z)=c_0+c_1z+\cdots+c_{n-1}z^{n-1}$ 匹配 $n$ 个函数值 $f_0,\dots,f_{n-1}$，插值点取 $n$ 个单位根 $z=1,w,\dots,w^{n-1}$——$F_n$ 正是这批特殊点上的 **Vandermonde 矩阵**（LA 08 §3.3 多项式拟合里的老相识，换了复数的插值点）。周期函数 $f(x)$ 经 $z=e^{i\theta}$ 生活在单位圆上，DFT 就是圆上的多项式插值。

---

## 8. 循环矩阵被 F 对角化：兑现 LA_13（§9.3 中）

### 8.1 P 的特征向量就是 F 的列

[LA 13](../LA_13_linear_transformations/) §9 立起的框架：循环置换矩阵

$$
P=\begin{bmatrix}0&1&0&0\\0&0&1&0\\0&0&0&1\\1&0&0&0\end{bmatrix},
\qquad P^4=I
\ \Longrightarrow\
\lambda^4=1
\ \Longrightarrow\
\lambda=1,\ i,\ -1,\ -i,
$$

特征向量 $(1,\lambda,\lambda^2,\lambda^3)$——**把这些列拼起来恰好是 $F_4$**（列 $\lambda=1,i,-1,-i$ 即 $w^0,w^1,w^2,w^3$ 的幂）。于是 $PF=F\Lambda$，即

$$
\boxed{\ P=F\Lambda F^{-1}\ }
\qquad(\Lambda=\mathrm{diag}(1,i,-1,-i)).
$$

### 8.2 一切循环矩阵同时被对角化

**循环矩阵**（circulant）每行都是上一行的循环右移。关键：每个循环矩阵都是 $P$ 的多项式，而多项式与对角化可交换（PS 9.3 #13）：

$$
C=c_0I+c_1P+c_2P^2+c_3P^3
\qquad\Longrightarrow\qquad
C=F\,\underbrace{\left(c_0I+c_1\Lambda+c_2\Lambda^2+c_3\Lambda^3\right)}_{E\ \text{对角！}}\,F^{-1}.
$$

特征值逐个读出（$\lambda_k^P$ 是 $P$ 的第 $k$ 个特征值）：$\lambda_k=c_0+c_1\lambda_k^P+c_2(\lambda_k^P)^2+c_3(\lambda_k^P)^3$。两个标本：

| 循环矩阵 | $c$ | 特征值 | 验证 |
|---|---|---|---|
| $C=2I+5P$（PS 9.2 #10） | $(2,5,0,0)$ | $7,\ 2+5i,\ -3,\ 2-5i$ | 和 $=8=$ 迹 $2+2+2+2$ ✓ |
| 周期 $\begin{bmatrix}2&-1&0&-1\\-1&2&-1&0\\0&-1&2&-1\\-1&0&-1&2\end{bmatrix}$ | $(2,-1,0,-1)$ | $0,\ 2,\ 4,\ 2$ | 和 $=8=$ 迹 ✓ |

第二行值得细看：$\lambda_k=2-\lambda_k^P-(\lambda_k^P)^3$，代入 $1,i,-1,-i$ 依次得 $0,\ 2,\ 4,\ 2$；统一公式 $\lambda_k=2-2\cos\frac{2\pi k}4$。这是 **LA 11 固定端点 $-1,2,-1$ 矩阵的周期版**——那边特征向量是正弦 $\sin\frac{jk\pi}{n+1}$（端点钉死），这边是复指数（首尾相接）；那边最小特征值 $2-2\cos\frac\pi{n+1}>0$，这边 $\lambda_0=0$ 且 $(1,1,1,1)$ 在零空间——**周期边界允许常数解**（[LA 06](../LA_06_independence_basis_dimension/) 图关联矩阵的常数向量又一次出场；离散拉普拉斯算子的两种边界条件）。

### 8.3 卷积定理：矩阵乘法变逐点乘法

$C$ 乘向量 $x$，按行看是"系数 $c$ 与 $x$ 的**循环卷积**"。对角化把整个计算改写成三步：

$$
Cx=F\,\big(E\,(F^{-1}x)\big)
\qquad\big(E=\mathrm{diag}(\lambda_0,\dots,\lambda_{n-1})\big).
$$

**空间域的卷积 = 频率域的逐点乘法**——卷积定理（LA 13 预告的兑现）。直接乘 $C$ 用 $n^2$ 次乘法；右边两次变换 + 一次对角乘，用 FFT 只需约 $n\log_2 n+n$ 次（§9.4 计数）——这正是信号处理里"快速卷积"与多项式快速乘法的内核。

---

## 9. FFT：把 n² 砍成 ½n log₂ n（§9.3 核心）

### 9.1 问题与钥匙

计算 $y=F_nc$ 是稠密矩阵乘向量：$n^2$ 次乘法。$F$ 没有零元，看起来省无可省——但 $w^{jk}$ 这个**特殊模式**允许把 $F$ 分解成一串稀疏矩阵，分解本身"造出"大量零。钥匙：**连接 $F_n$ 与半尺寸的 $F_{n/2}$**（$n$ 取 2 的幂）。

### 9.2 F_4 的完整因子分解（可逐列验算）

$$
F_4=
\underbrace{\begin{bmatrix}1&0&1&0\\0&1&0&i\\1&0&-1&0\\0&1&0&-i\end{bmatrix}}_{\left[\begin{smallmatrix}I&D\\I&-D\end{smallmatrix}\right],\ D=\mathrm{diag}(1,i)}
\underbrace{\begin{bmatrix}1&1&0&0\\1&-1&0&0\\0&0&1&1\\0&0&1&-1\end{bmatrix}}_{\mathrm{diag}(F_2,F_2)}
\underbrace{\begin{bmatrix}1&0&0&0\\0&0&1&0\\0&1&0&0\\0&0&0&1\end{bmatrix}}_{P:\ \text{偶前奇后}}
$$

三个因子各司其职（从右往左读）：

1. **置换 $P$**：把 $c=(c_0,c_1,c_2,c_3)$ 重排成（偶，奇）$=(c_0,c_2\,\vert\,c_1,c_3)$；
2. **半尺寸变换**：上下两块各做 $F_2$——半尺寸矩阵自动改用 $w^2$（4 阶的根 $i$ 平方后 $=-1$，正是 2 阶的根！教材特意强调：$F_{n/2}$ 的根是 $w_{n/2}=w^2$）；
3. **合并** $\left[\begin{smallmatrix}I&D\\I&-D\end{smallmatrix}\right]$：把两个半尺寸输出 $u,v$ 拼成 $y=(u_0+v_0,\ u_1+iv_1,\ u_0-v_0,\ u_1-iv_1)$。

现场验证 $c=(0,1,0,1)$：$Pc=(0,0,1,1)$；上半 $F_2(0,0)=(0,0)$，下半 $F_2(1,1)=(2,0)$；合并 $y=(2,\ 0,\ -2,\ 0)$。直接乘 $F_4$ 验证（下标从 0 起：$c$ 的第 1、3 分量非零，结果 = 第 1 列 + 第 3 列 $=(2,\ i-i,\ -2,\ -i+i)$）✓。

### 9.3 蝴蝶公式：一行推导

一般步：$m=n/2$，$c'$= 偶下标、$c''$= 奇下标，半尺寸输出 $y'=F_mc'$、$y''=F_mc''$（$F_m$ 用 $w^2=w_{m}$）。按奇偶拆开求和：

$$
y_j=\sum_{k=0}^{n-1}w^{jk}c_k
=\sum_{r}(w^2)^{jr}c_{2r}\;+\;w^j\sum_{r}(w^2)^{jr}c_{2r+1}
=y'_j+w^j y''_j .
$$

对 $j\ge m$：$w^{mk}=e^{\pi ik}=(-1)^k$——偶项不变、奇项变号，且 $(w^2)$ 的指数以 $m$ 为周期：

$$
\boxed{\ y_j=y'_j+w^j y''_j,\qquad y_{j+m}=y'_j-w^j y''_j,\qquad j=0,\dots,m-1\ }
$$

（$w^m=-1$ 保证了那对正负号。）流程图上每个"合并"画成蝴蝶形（$y''$ 分别乘 $1,i$ 与 $-1,-i$）——**蝶形结**（butterfly）是 FFT 的名字来源。

### 9.4 递归、计数与 bit 反转

同样的分解对 $F_{1024}$ 里的每个 $F_{512}$ 再来一次，$F_{512}$ 里的 $F_{256}$ 再来……**递归**（Cooley–Tukey 1965；高斯 1805 年已预见到）：

$$
\text{层数 } \ell=\log_2 n,\qquad
\text{每层 } n/2 \text{ 次乘法（对角 } D \text{ 的旋转因子）}
\qquad\Longrightarrow\qquad
\boxed{\ \tfrac12 n\log_2 n\ }.
$$

$n=1024=2^{10}$ 的账：

| | 乘法次数 |
|---|---|
| 直接乘 $F_{1024}$ | $1024^2=1{,}048{,}576\approx$ 一百万 |
| FFT | $\tfrac12\cdot1024\cdot10=5120\approx$ 五千 |
| 加速 | **约 200 倍** |

$n$ 再翻倍差距更悬殊（$n=2^{20}$ 时相差十万倍量级）——"FFT 革命了信号处理"的定量注脚。

两个收尾事实：

- **bit 反转**（bit reversal）：$\ell$ 层"偶前奇后"置换的净效果 = 把下标写成二进制后**逆序**。$n=4$：$00,01,10,11\to00,10,01,11$ 即 $0,2,1,3$；$n=8$：$0,4,2,6,1,5,3,7$。所以 FFT 输入按 bit 反序进场、$\log_2 n$ 层蝴蝶、输出自然顺序；
- **Hadamard 同构**（兑现 [LA 08](../LA_08_least_squares_gram_schmidt/) 的预告）：快速 Hadamard 变换的递推 $H_{2n}=\begin{bmatrix}H_n&H_n\\H_n&-H_n\end{bmatrix}$ 与 FFT 的分治**同构**——合并矩阵 $\left[\begin{smallmatrix}I&I\\I&-I\end{smallmatrix}\right]$ 就是 $H_2$ 张量积，唯一的区别是 FFT 多了对角旋转因子 $D$。小波变换的快速算法（Mallat 级联）也是同一族蝴蝶。

---

## 10. 例子与反例速查

| 命题 | 真假 | 例子 / 反例 |
|------|------|------------|
| $z^Tz$ 是复向量的长度平方 | ❌ | $(1,i)$：$1^2+i^2=0$，非零向量长度归零；须用 $z^Hz$ |
| $u^Hv=v^Hu$ | ❌ | $u=(1,i),\ v=(i,0)$：$u^Hv=i$，$v^Hu=-i$（互为共轭） |
| 实矩阵的特征值必实 | ❌ | $90°$ 旋转 $\lambda=\pm i$；共轭成对才是定理 |
| Hermitian 矩阵的特征值必实 | ✅ | $\lambda=z^HSz/z^Hz$，分子实、分母正 |
| 实反对称矩阵的特征值是纯虚数 | ✅ | $-\lambda=\bar\lambda$；$\begin{bmatrix}0&1\\-1&0\end{bmatrix}$ 的 $\pm i$ |
| 对称复矩阵必有实特征值 | ❌ | $F_3$ 对称但非 Hermitian——"对称"在复世界不够格 |
| Hermitian 矩阵对角线必实 | ✅ | $S_{ii}=\bar S_{ii}$；$F_3$ 对角 $(1,w,w)$ 非实故非 Hermitian |
| 酉矩阵的特征值模为 1 | ✅ | $\|z\|=\|Uz\|=|\lambda|\|z\|$；$F_3/\sqrt3$ |
| 酉矩阵的行列式是实数 | ❌ | $\mathrm{diag}(i,1)$ 酉，$\det=i$；只有 $|\det U|=1$ 保证 |
| 既 Hermitian 又酉 ⟹ 特征值 $\pm1$ | ✅ | §5.3 的 $Q$：实 + 模 1 + 迹 0 ⟹ 恰 $1,-1$，且 $Q^2=I$ |
| $1,w,\dots,w^{n-1}$ 之和为零 | ✅ | 旋转不动点论证 $wS=S\Rightarrow S=0$（$n\ge2$） |
| $z^n=1$ 只有根 $z=1$ | ❌ | $n$ 个根等距分布单位圆；$z^8=1$ 有 8 个 |
| 单位圆上 $1/z=\bar z$ | ✅ | $z\bar z=1$；$1/e^{i\theta}=e^{-i\theta}$ |
| $F_n$ 的列不用共轭也算正交 | ❌ | $F_3$ 列 2·列 3 裸点积 $=\tfrac13(1+1+1)=1\ne0$；共轭后才是 0 |
| 循环矩阵互相可交换 | ✅ | 都是 $P$ 的多项式，多项式可交换 |
| 周期 $-1,2,-1$ 矩阵正定 | ❌ | $\lambda_0=0$，$(1,1,1,1)$ 在零空间（固定端点版才正定，LA 11） |

---

## 11. 易混淆辨析

| 概念对 | 辨析 |
|--------|------|
| **$A^T$ vs $A^H$** | 转置只翻行列；共轭转置还把 $i$ 换 $-i$。实矩阵两者相同——复世界必须用 $H$，否则连长度都算不对 |
| **对称 $S^T=S$ vs Hermitian $S^H=S$** | Hermitian 要求 $S_{ij}=\bar S_{ji}$（对角线实）。$F_3$ 对称但不 Hermitian；只有 Hermitian 享受实特征值 + 正交特征向量 |
| **正交 vs 酉** | 同一个定义 $Q^HQ=I$，元素全实叫正交、含复数叫酉。正交 ⊂ 酉；酉的行列式模 1 但可复 |
| **$\bar z$ vs $1/z$** | 一般 $1/z=\bar z/\|z\|^2$；只在单位圆上 $1/z=\bar z$（DFT 的根全住单位圆——所以"逆=共轭"贯穿 §7–9） |
| **Hermitian vs 反 Hermitian** | $K=iS$：Hermitian 乘 $i$ 变反 Hermitian（特征值实 → 纯虚）。实世界里"对称 vs 反对称"的复数升级 |
| **正常 vs Hermitian** | 正常 $N^HN=NN^H$ ⟺ 可酉对角化，是最大家族；Hermitian（特征值在实轴）、酉（在单位圆）、反 Hermitian（在虚轴）都是它的特例 |
| **$w$ vs $\bar w$（两套约定）** | Strang 的 $F$ 用 $e^{2\pi i/n}$（频率→物理，$y=Fc$）；MATLAB/NumPy 的 `fft` 用 $e^{-2\pi i/n}$（物理→频率）。互为共轭转置，结果只差行序 |
| **$F_n$ vs $F_n/\sqrt n$ vs $F_n^{-1}$** | $F^HF=nI$：只有除以 $\sqrt n$ 才是酉；逆是 $\bar F/n$（共轭 + 除 $n$）。三个版本在同一页混用是头号事故源 |
| **DFT vs FFT** | DFT 是**变换**（乘 $F_n$，数学对象）；FFT 是计算它的**算法**（分治因子分解，工程成就）。FFT ≠ 另一种变换 |
| **复特征值对 vs 两个独立特征值** | 实矩阵的 $\lambda,\bar\lambda$ 携带同一份信息（特征向量也共轭）——只算一个即可；复矩阵则没有这种配对 |
| **有限 Fourier 级数 vs Fourier 级数** | 本篇是 $n$ 项级数在 $n$ 个点上求值（有限、离散、代数完胜）；无穷级数与收敛性是 §10.5 / LA 16 的函数版 |

---

## 12. ML / 应用连接

- **FFT 无处不在**：MP3/AAC 音频编码、JPEG（DCT 可由 FFT 计算）、OFDM（WiFi/5G 把数据调制在单位根 $w^k$ 上——本节的 $F_n$ 列就是 OFDM 的子载波）、雷达、MRI 重建。任何一个"实时处理信号"的系统背后都是 $\tfrac12n\log_2 n$ 这行计数；
- **卷积定理与 CNN**：循环填充的卷积算子是循环矩阵 ⟹ 在 $F$ 基下是对角的——**卷积 = 频域逐点乘**。大卷积核（如语音/医学影像）用 FFT 卷积 $O(n\log n)$ 打败直接卷积 $O(n^2)$；反过来这也解释了 CNN 的频谱视角：学到的卷积核的特征响应就是 §8.2 的 $\lambda_k$（频率响应曲线）；
- **频域特征工程**：Transformer 的正弦位置编码是 $e^{ikx}$ 的多尺度叠加（本篇的单位根语言）；NeRF 的 Fourier features 用高频正弦对抗谱偏差（低频先学的归纳偏置）；去噪 = 保留低频系数、清掉高频（LA 12 图像压缩的频域表亲）；
- **RoPE 旋转位置编码**：把注意力内积乘上单位复数 $e^{im\theta}$（即 2×2 旋转），正是 §6.1 "$\|Uz\|=\|z\|$、内积不变"的现场应用——**旋转是保持内积的最一般操作**，位置编码只想平移注意力分数而不想扭曲它；
- **量子计算与量子 ML**：量子态住在 $\mathbb C^n$（归一化 $\|z\|=1$），量子门是酉矩阵（演化保范数），可观测量是 Hermitian（测量值必实）。本节的整张词汇表——内积、酉、Hermitian、谱定理——就是量子力学的线性代数底座；
- **复值网络**：无线通信、音频、EEG 的原生数据是复数（幅度+相位）；复数域的 batchnorm 协方差是 Hermitian 正定矩阵，复数域的"正交初始化"是酉矩阵——本节定义的直接移植；
- **图信号处理**：图拉普拉斯的特征基称为图 Fourier 基（LA 06 关联矩阵 → LA 16 正片）；循环图上的图卷积退化为本篇的循环卷积——谱 GNN 是卷积定理在图上的推广。工程提醒：`np.fft.fft` 用 $\bar w$ 约定且正变换不除 $n$——与教材公式对数时先核约定与归一化。

---

## 13. 习题精选（Problem Set 9.1–9.3 选摘）

**题 1**（9.1 #1–2）：计算 $(2+i)(2-i)$、$(2+i)^2$、$\dfrac1{2+i}$，并在复平面上标出 $(2+i)^2$。模与角各是多少？
*提示*：乘积 $=5$、平方 $=3+4i$、倒数 $=(2-i)/5$。$|2+i|=\sqrt5$、角 $\approx26.57°$；平方后模 $5$、角翻倍 $\approx53.13°$（$\tan^{-1}(4/3)$）——**极坐标一口报出答案**。

**题 2**（9.1 #16，作者最爱）：$z=\sin\theta+i\cos\theta$。求 $|z|$ 与角（小心！），并计算 $z\cdot(\cos\theta+i\sin\theta)$。
*提示*：$|z|=1$，但角是 $\frac\pi2-\theta$ 而不是 $\theta$（实部是 $\sin$）。乘积：$(\sin\theta+i\cos\theta)(\cos\theta+i\sin\theta)=\sin\theta\cos\theta+i\sin^2\theta+i\cos^2\theta-\cos\theta\sin\theta=i$——两个模 1 的数相乘，落在虚轴上 ✓。

**题 3**（9.1 #11）：求 $A=\begin{bmatrix}a&b\\-b&a\end{bmatrix}$ 的特征值与特征向量；解释它为什么是"旋转-缩放"；实矩阵的共轭特征值对在哪里现身？
*提示*：$\lambda=a\pm bi$ 各配 $x=(1,\ \pm i)$（验证 $A(1,i)=(a+bi)\,(1,i)$）；$A=rR_{-\theta}$（$a=r\cos\theta,\ b=r\sin\theta$），$\lambda=re^{\mp i\theta}$。$A$ 实 ⟹ $(\lambda,x)$ 与 $(\bar\lambda,\bar x)$ 成对（§3 定理的直接检验）。

**题 4**（9.1 #14）：证明实反对称矩阵（$K^T=-K$）的特征值是纯虚数。
*提示*：$Kx=\lambda x$ 左乘 $x^H$ 得 $x^HKx=\lambda x^Hx$；对左边取共轭转置（$K^H=-K$）得 $-x^HKx=\bar\lambda x^Hx$。对照两式：$\bar\lambda=-\lambda$。LA 04 的"反对称"在此拿到复数判决书。

**题 5**（9.2 #1）：$u=(1+i,\ 1-i,\ 1+2i)$、$v=(i,\ i,\ i)$。求 $\|u\|,\ \|v\|,\ u^Hv,\ v^Hu$。
*提示*：$\|u\|^2=2+2+5=9$、$\|v\|^2=3$。$u^Hv=(1-i)i+(1+i)i+(1-2i)i=(1+i)+(i-1)+(i+2)=2+3i$；$v^Hu=2-3i$——**换序 = 取共轭**的现场标本。

**题 6**（9.2 #5）：证明 $A^HA$ 恒 Hermitian；列无关时它还是正定的。这是哪条老定理的复数版？
*提示*：$(A^HA)^H=A^H(A^H)^H=A^HA$；$z^HA^HAz=\|Az\|^2>0\ (z\ne\mathbf 0)$ 当 $N(A)=\{\mathbf 0\}$。[LA 07](../LA_07_orthogonality_projections/) 的 $A^TA$ 可逆定理（$\|Ax\|^2=0\Rightarrow Ax=\mathbf 0$）一字不差地升级为 $H$ 版——最小二乘正规方程从此在复数世界照常工作。

**题 7**（9.2 #8 + #10）：对 4 阶循环置换 $P$ 计算 $P^2,\ P^3,\ P^{100}$ 与全部特征值；再求循环矩阵 $C=2I+5P$ 的特征值。
*提示*：$P^4=I$ ⟹ $P^{100}=P^{0}=I$（$100=25\times4$）。$\lambda=1,i,-1,-i$（$P$ 的特征向量是 $F_4$ 的列）。$C$ 的特征值 $=2+5\lambda=7,\ 2+5i,\ -3,\ 2-5i$，和 $8=$ 迹 ✓——**同一组特征向量，对角元逐个加常数**。

**题 8**（9.2 #24）：$u^Hu=1$ 时证明 $I-2uu^H$ 既 Hermitian 又酉。它是什么变换？
*提示*：$(uu^H)^H=uu^H$ ⟹ Hermitian；$(I-2uu^H)^H(I-2uu^H)=I-4uu^H+4u(u^Hu)u^H=I$。复 Householder 反射（LA 08 的 $I-2uu^T$ 复数版）——$uu^H$ 是投到 $u$ 线上的秩一投影。

**题 9**（9.3 #5 + #7）：对 $F_4$：(a) 证明 $Fv=w,\ Fw=4v$（$v=(1,0,0,0),\ w=(1,1,1,1)$）；(b) 用 FFT 三步算 $c=(1,0,1,0)$ 与 $c=(0,1,0,1)$ 的变换。
*提示*：(a) $Fv$ = 第一列 $=(1,1,1,1)$；$Fw$ = 行和 $=(4,\ 1+i-1-i,\ 1-1+1-1,\ 1-i-1+i)=(4,0,0,0)$。(b) $c=(1,0,1,0)$：$Pc=(1,1,0,0)$，$u=F_2(1,1)=(2,0),\ v'=(0,0)$，$y=(2,0,2,0)$；$c=(0,1,0,1)$：$y=(2,0,-2,0)$——纯偶/纯奇的频率一目了然（角频率 $0$ 与 $\pi$）。

**题 10**（9.3 #14 + #17）：(a) 求周期矩阵 $C=\begin{bmatrix}2&-1&0&-1\\-1&2&-1&0\\0&-1&2&-1\\-1&0&-1&2\end{bmatrix}$ 的特征值（用 $C=2I-P-P^3$ 与 $P$ 的特征值）；(b) 写出 $0,\dots,7$ 的 bit 反转顺序。
*提示*：(a) $\lambda_k=2-\lambda_k^P-(\lambda_k^P)^3$：$k=0,1,2,3$ 依次 $0,\ 2,\ 4,\ 2$（一般式 $2-2\cos\frac{2\pi k}4$）——与 LA 11 固定端点公式 $\lambda_k=2-2\cos\frac{k\pi}{n+1}$ 并排对比，一个差 $\cos$ 的周期。(b) 二进制 $000,\dots,111$ 逐个逆序：$0,4,2,6,1,5,3,7$。

---

## 14. 逻辑推导概念图

全篇的逻辑可以压成一棵树：顶部是出发点（$i^2=-1$ 与共轭转置 $A^H$ 这两组定义），箭头是推理方向，每一步只用上游结论。读者可沿箭头反向核对"哪个定理用了哪条假设"。

```text
出发点 A：i² = -1 ──→ Euler 公式 e^{iθ} = cosθ + i sinθ
    └──→ n 次单位根 w = e^{2πi/n}（n 个根等距分布单位圆）
              ├──→ w^m = -1（当 n = 2m，半圈 = -1）
              └──→ 1 + w + ... + w^{n-1} = 0（转一格不动 ⟹ 和为零）

出发点 B：转置时顺便共轭（A^H = conjugate-transpose，内积 u^H v）
    ├──→ 长度 ‖z‖² = z^H z（z^T z 对 (1,i) 算出 0，被迫修正）
    ├──→ Hermitian S = S^H ──→ z^H S z 实 ──→ λ 实 ──→ 特征向量正交
    │                                        └──→ 复谱定理 S = UΛU^H
    ├──→ 酉 U^H U = I ──→ ‖Uz‖ = ‖z‖ ──→ |λ| = 1（特征值在单位圆上）
    └──→ 正常 N^H N = NN^H ⟺ 可酉对角化（上面全家都是特例）

A + B 汇合
    ├──→ F^H F = nI（每个非对角列内积都是"单位根之和 = 0"）
    │        ├──→ F⁻¹ = F^H/n（逆 = 共轭 + 除 n）；F_n/√n 条件数 1
    │        └──→ P = FΛF⁻¹ ──→ 一切循环矩阵同时被对角化
    │                              └──→ 卷积定理：卷积 = 频域逐点乘
    └──→ w^m = -1 + 奇偶拆分 ──→ 蝴蝶公式 y_j = y'_j + w^j y''_j
                                    └──→ 递归 log₂ n 层 ⟹ ½ n log₂ n（FFT）
```

| 出发点 | 买到什么 | 撤掉它失去什么 |
|--------|---------|---------------|
| 转置时顺便共轭（$\|z\|^2=z^Hz$、$u^Hv$） | 复世界长度恒正、内积与正交可用；Hermitian 三大性质的地基 | $(1,i)$ 的长度算成 $1^2+i^2=0$；$F_3$ 列 2·列 3 裸点积 $=1\ne0$——正交性整个塌掉（§4.1、§6.2） |
| $A$ 是实矩阵（$\bar A=A$） | 共轭特征值对：$Ax=\lambda x\Rightarrow A\bar x=\bar\lambda\bar x$ | 复矩阵没有这种配对，"$\lambda,\bar\lambda$ 各带共轭特征向量、只算一半"的福利消失（§3） |
| $S=S^H$（对角线实 + 两侧互为共轭） | $z^HSz$ 实 ⟹ $\lambda$ 实、特征向量正交、复谱定理 $S=U\Lambda U^H$ | 只剩对称 $F_3=F_3^T$：对角线 $(1,w,w)$ 非实、$\lambda$ 可以是复数——"对称"在复世界不够格（§6.2、§10 速查） |
| $w^m=-1$（$n=2m$） | 蝴蝶公式里 $y_j$ 与 $y_{j+m}$ 的那对正负号 | 奇偶分裂后两半输出接不回来，FFT 递归断裂，退回 $n^2$ 次乘法（§9.3） |
| 循环矩阵 $=P$ 的多项式（与 $P$ 共享特征向量） | $F^{-1}CF=\Lambda$ 一网打尽；卷积 = 频域逐点乘，快速卷积 $O(n\log n)$ | 各矩阵对角化各凭本事，"同一组 $F$ 列通吃"与多项式可交换性全没了（§8.2–8.3） |
| 正常性 $N^HN=NN^H$ | 可酉对角化的精确刻画，Hermitian/酉/反 Hermitian 统一为"特征值摆哪儿"的特例 | $\begin{bmatrix}1&1\\0&1\end{bmatrix}$ 不正常，没有正交特征向量组，谱定理不进门（§6.3） |

---

## 15. 小结与下一篇

本篇五件套，一句话各表：

| 概念 | 一句话 | 接口 |
|------|--------|------|
| 共轭转置 $A^H$ | 转置时顺手共轭——长度、内积、正交的全部修正 | 实 → 复词典（§4.3 整表） |
| Hermitian $S=S^H$ | 复世界的"对称"：$z^HSz$ 实、$\lambda$ 实、特征向量正交 | 复谱定理 $S=U\Lambda U^H$ |
| 酉 $U^HU=I$ | 复世界的"正交"：$\|Uz\|=\|z\|$、$\|\lambda\|=1$ | 正常矩阵 $N^HN=NN^H$ 统一全家 |
| Fourier 矩阵 $F_n$ | $(F_n)_{jk}=w^{jk}$，$F^HF=nI$，对角化一切循环矩阵 | 卷积定理：卷积 = 频域逐点乘 |
| FFT | $F_n=\left[\begin{smallmatrix}I&D\\I&-D\end{smallmatrix}\right]\mathrm{diag}(F_{n/2})P$ 递归 | $\tfrac12n\log_2 n$、蝴蝶、bit 反转 |

第 9 章以"一个矩阵乘一个向量"收束全书最漂亮的应用，也顺手配齐了最后一件武器：**酉矩阵是数值世界的模范公民**——$F_n/\sqrt n$ 所有奇异值全为 1、条件数为 1，乘它不放大任何误差。可普通的矩阵呢？消元在浮点运算里会发生什么、$\|\delta x\|/\|x\|$ 最坏会被放大多少倍？LA 12 的 $\sigma_1/\sigma_n$ 在下一章正式上岗：条件数、消元的稳定性、$Ax=b$ 的迭代解法——数值线性代数。

---

*下一篇：[线性代数 15：数值线性代数](../LA_15_numerical_linear_algebra/)*
