---
title: "泛函分析 13: 谱论入门——算子的\"特征值\""
date: 2026-10-09
math: true
tags:
  - Notes
  - FunctionalAnalysis
categories:
  - FunctionalAnalysis
description: "第 5 章开篇：积分方程 (λ−T)x=y 的可解性问题如何逼出'谱'的概念。有限维世界 C = 点谱 ∪ 预解集干净二分；无穷维世界 λ−A 不可逆有三种病理（不单射=点谱 σp / 单射值域稠密不满=连续谱 σc / 值域不稠密=剩余谱 σr）——乘法算子（纯剩余谱）、对角算子（点谱+连续谱）、Volterra 算子（谱={0} 半径 0）三个样本完整计算。四大定理兑现篇：闭图像定理白送 R(λ,A) 有界、Neumann 级数（B(X) 中的几何级数）证预解集是开集、预解式解析 + Liouville 定理证 Gelfand–Mazur（'没有谱的算子不存在'——HB 分离最后收账）、一致有界性原理在 Gelfand 谱半径公式 r(A)=lim‖Aⁿ‖^{1/n} 中兑现"
---

# 泛函分析 13：谱论入门——算子的"特征值"

> 📖 本系列基于步尚全《泛函分析基础》整理，保留教材定理编号。本篇覆盖 §5.1（基本概念及例子）——第 5 章开篇。
> 前篇：[泛函分析 12：逼近论——泛函分析的落地](../FA_12_approximation_theory/)。
> 编号已对照教材 PDF 核实（§5.1 全部正文 + 习题 5 题 1–13）。

---

## 1. 这一篇解决什么问题？

教材开篇就给谱论的问题原型——**积分方程**。$k\in C([a,b]^2)$，积分算子 $(Tx)(s)=\int_a^bk(s,t)x(t)\,dt$，求解

$$
(\lambda-T)x(s)=y(s),\qquad s\in[a,b].
$$

三个递进的问题：**对什么样的 $y$ 有解？有解时何时唯一？对怎样的 $\lambda$，任取 $y$ 都有唯一解？**——最后这个问句就是谱论的标准问题：

> 给定算子 $A$，**哪些 $\lambda$ 使 $\lambda-A$ "可逆"（一一映射且逆自动连续）？**

有限维时这是老朋友：$A$ 是 $n$ 阶方阵，$\lambda-A$ 不可逆 $\iff\det(\lambda-A)=0$ $\iff$ $\lambda$ 是特征值。复平面被切成干净的两块：

$$
\mathbb{C}=\sigma_p(A)\cup\rho(A)\qquad(\text{特征值} \mid \text{正则值})。
$$

无穷维时**这张二分表会漏人**：$\lambda-A$ 可以单射而不满射（方程解唯一但不是每个 $y$ 都有解），可以满射而逆不连续……而且——本篇最大惊喜——**"没有谱的算子不存在**"（定理 5.1.4 Gelfand–Mazur）：任何复 Banach 空间上的有界算子都有谱点。想造一个"处处可逆"的算子？复分析（Liouville 定理）不答应。

本篇还是**第 4 章四大定理的集中兑现篇**（比 FA 12 更直接）：

| 兑现场景 | 用的定理 | 在哪 |
|---|---|---|
| $\lambda\in\rho(A)$ 时 $R(\lambda,A)=(\lambda-A)^{-1}$ **自动有界** | 闭图像定理（FA 11） | §2 |
| $\rho(A)$ 是**开集**（谱对参数稳定） | Neumann 级数（B(X) 完备性） | §3.3 |
| 谱**非空**、有界、闭 | Liouville + Hahn-Banach（FA 09） | §3.5 |
| 谱半径公式 $r(A)=\lim\|A^n\|^{1/n}$ 的关键一步 | **一致有界性原理**（FA 10） | §3.6 |

**为什么必须复空间**：谱论的引擎是复变函数（解析性、Liouville）；实空间上谱论"远没有复空间完美"（教材原话）——例如实空间上旋转矩阵 $\begin{pmatrix}0&-1\\1&0\end{pmatrix}$ 没有实特征值，但复化后谱 $=\{\pm i\}$ 立刻现身。**谱论天然生活在复平面上**。

---

## 2. 核心定义与直觉

### 2.1 四分法：λ−A 不可逆的三种病理

**定义 5.1.1**　$X$ 非零复 Banach 空间，$D(A)\subset X$ 线性子空间，$A:D(A)\to X$ **闭**线性算子，$\lambda\in\mathbb{C}$。记 $\lambda-A:=\lambda I_X-A$。

- **特征值**（本征值）：存在 $x_0\in D(A)\setminus\{0\}$ 使 $Ax_0=\lambda x_0$；特征向量全体为**特征空间**；全体特征值之集为**点谱** $\sigma_p(A)$。
- **正则值**：$\lambda-A:D(A)\to X$ 为一一映射；全体正则值之集为**预解集** $\rho(A)$。

**定义 5.1.2**　设 $\lambda-A$ 为单射且不为满射（$\lambda\notin\sigma_p(A)\cup\rho(A)$）：

- **连续谱点**：$\overline{R(\lambda-A)}=X$（值域**稠密**但不满）——全体为 $\sigma_c(A)$；
- **剩余谱点**：$\overline{R(\lambda-A)}\ne X$（值域**不稠密**）——全体为 $\sigma_r(A)$；
- **谱** $\sigma(A)=\sigma_p(A)\cup\sigma_c(A)\cup\sigma_r(A)$（不交并），$\mathbb{C}=\rho(A)\cup\sigma(A)$（不交并）。

**四分法读表**（检查 $\lambda-A$ 的三道关卡）：

| 关卡 | 通过 | 不通过 |
|---|---|---|
| 第一关：单射？（方程解唯一） | ❌ 不单射 ⟹ $\lambda\in\sigma_p(A)$ | ✅ 单射 ↓ |
| 第二关：满射？（方程解存在） | ✅ 满射 ⟹ $\lambda\in\rho(A)$ | ❌ 不满射 ↓ |
| 第三关：值域稠密？ | ✅ 稠密 ⟹ $\lambda\in\sigma_c(A)$ | ❌ 不稠密 ⟹ $\lambda\in\sigma_r(A)$ |

三点直觉：

1. **有限维自动停在二分**：$\dim X<\infty$ 时单射 ⟹ 满射（秩—零化度），$\sigma_c,\sigma_r$ 无处容身。**连续谱和剩余谱是无穷维的特产**——"解方程"在无穷维里有了矩阵世界没有的病。
2. **连续谱的"连续"**：值域稠密 ⟹ 任取 $y$，有 $y_\varepsilon\to y$ 使 $(\lambda-A)x_\varepsilon=y_\varepsilon$ 可解——**方程可以近似解到任意精度，但精确解可以不存在**。"差一点点可解"的谱点。
3. **剩余谱的"剩余"**：值域连稠密都不够——存在泛函 $f\ne0$ 零化整个值域（$f\in{}^\perp R(\lambda-A)$），方程不仅无解，还**结构性地无解**（整个方向都够不着）。例 5.1.1 会给出现场。

### 2.2 为什么限定闭算子：闭图像定理的第一次兑现

教材明确：只讨论**闭**线性算子的谱论（一般算子谱论复杂且应用少）。闭性不是装饰，它买来两件硬货：

**(1) $\lambda-A$ 自动闭**：$x_n\to x$、$(\lambda-A)x_n\to y$ 时 $Ax_n\to y-\lambda x$，$A$ 闭给 $x\in D(A)$、$Ax=y-\lambda x$，即 $(\lambda-A)x=y$——对账完成。

**(2) 预解式自动有界**：$\lambda\in\rho(A)$ 时 $(\lambda-A)^{-1}:X\to X$ 存在、线性，且

$$
(x,y)\in G_{\lambda-A}\iff(y,x)\in G_{(\lambda-A)^{-1}}
$$

——**图像翻转仍是闭集**，而定义域是全空间 $X$（Banach），**闭图像定理（FA 11）一步到位**：$(\lambda-A)^{-1}\in B(X)$。这是 FA 11 "免检通道"的教科书级应用：正则值的定义只要求"一一映射"（纯代数），**逆的连续性是 Banach + 闭性白送的**。定义 $A$ 的**预解式**

$$
R(\lambda,A)=(\lambda-A)^{-1},\qquad\lambda\in\rho(A)\in B(X).
$$

（呼应 FA 11 题 33：有界逆 ⟺ 值域闭——本篇的 $\rho$ 判据"满射"比"值域闭"更强，差的正是单射侧的信息。）

### 2.3 向量值解析函数：给算子值函数做复分析

**定义 5.1.3**　$\Omega\subset\mathbb{C}$ 开，$f:\Omega\to X$。连续性照抄；**可导**：存在 $a\in X$ 使 $\lim_{t\to t_0}\big\|\frac{f(t)-f(t_0)}{t-t_0}-a\big\|=0$，记 $f'(t_0)=a$；处处可导称**解析**。

关键桥梁：$f$ 解析 ⟹ 对一切 $\varphi\in X'$，**纯量函数** $\varphi\circ f$ 解析——**向量值函数的复分析全部经对偶泛函压缩成经典复分析**（FA 09 "对偶观测者"姿态的又一次出场）。Gelfand–Mazur 的证明就靠这句话进门。

---

## 3. 主要定理与证明

### 3.1 三个样本算子：四分法的完整现场

**例 5.1.1（乘法算子——纯剩余谱）**　$A:C[0,1]\to C[0,1]$，$(Ax)(t)=t\,x(t)$。$\|A\|=1$（$x=1$ 取等）。逐层检查：

- **点谱空**：$(\lambda-t)x(t)=0$ 逐点成立 ⟹ $x$ 至多在一点非零 ⟹ 连续性逼 $x=0$。$\sigma_p(A)=\varnothing$。
- **$\lambda\notin[0,1]$ 正则**：$x(t)=\frac{y(t)}{\lambda-t}$ 连续，$(\lambda-A)^{-1}$ 显式存在有界。$\rho(A)\supset\mathbb{C}\setminus[0,1]$。
- **$\lambda\in[0,1]$ 剩余谱**：$y\in R(\lambda-A)$ 必满足 $y(\lambda)=0$；取极限知闭包也不含"在 $\lambda$ 点非零"的函数（常函数 1 就进不来）。值域**不稠密** ⟹ $\lambda\in\sigma_r(A)$。

$$
\rho(A)=\mathbb{C}\setminus[0,1],\qquad\sigma(A)=\sigma_r(A)=[0,1],\qquad\sigma_p=\sigma_c=\varnothing.
$$

**读法**：$A$ 没有特征向量（连续函数"躲不开"变系数），但 $[0,1]$ 上每个 $\lambda$ 都让方程**结构性无解**——值域被"在 $\lambda$ 处取零"这个闭条件拦住。谱 $=$ 系数函数 $t$ 的值域：乘法算子的谱就是"乘的那个函数"的像。**$\sigma(A)\subset[0,1]$ 还是实轴上的区间——实谱，这将是 FA 14 自伴算子的预告片**（本例的 $A$ 恰是自伴的）。

**例 5.1.2（对角算子——点谱 + 连续谱）**　$\alpha_n\searrow 0$，$A\in B(\ell^2)$：$A(x_1,x_2,\dots)=(\alpha_1x_1,\alpha_2x_2,\dots)$，$\|A\|=\alpha_1$。

- **每个 $\alpha_n$ 是特征值**：$Ae_n=\alpha_ne_n$。$\sigma_p(A)\supset\{\alpha_n\}$。
- **$0$ 是连续谱**：$A$ 单射；$A$ 不满射（若满射，开映射定理给 $A^{-1}$ 有界，但 $\|A^{-1}\|\ge\frac{1}{\alpha_n}\to\infty$ 矛盾——**FA 11 上场**）；但值域稠密：有限支撑列 $y^{(n_0)}$ 的原像仍是有限支撑列，而有限支撑列稠密（FA 08 的老事实）。$0\in\sigma_c(A)$。
- **其余全正则**：$\lambda\notin\{\alpha_n\}\cup\{0\}$ 时 $\inf_n|\lambda-\alpha_n|>0$（$\alpha_n\to0$ 且 $\lambda\ne0$ 把"距离零点"堵死），逐坐标除以 $\lambda-\alpha_n$ 仍在 $\ell^2$。

$$
\sigma_p(A)=\{\alpha_n\},\quad\sigma_c(A)=\{0\},\quad\sigma_r(A)=\varnothing,\quad\rho(A)=\mathbb{C}\setminus(\{\alpha_n\}\cup\{0\}).
$$

**读法**：对角算子是"无穷维矩阵"最直接的化身，谱 $=$ 对角元集合 $\cup$ 其聚点。**聚点 $0$ 处的谱型是"连续"**——特征值序列的极限点自己不当特征值，方程 $(0-A)x=y$ 可近似解（有限支撑）但 $y=(\frac1n)$ 这种就无精确解。这个例子是 FA 14 紧算子谱论（定理 5.2.4：非零谱全是特征值、$0$ 是唯一可能聚点）的预告片。

**例 5.1.4（Volterra 算子——谱半径为零）**　$(Ax)(t)=\int_0^tx(s)\,ds$（FA 06 题 20 的老朋友，$\|A\|\le1$）。

- **点谱空**：$Ax=\lambda x$ 时若 $\lambda=0$ 则 $x=0$；若 $\lambda\ne0$ 求导得 $\lambda x'=x$、$x(0)=0$ ⟹ $x(t)=Ce^{t/\lambda}=0$。
- **$0$ 是剩余谱**：$R(A)\subset\{y:y(0)=0\}$（闭真子空间，不稠密）。
- **谱只有 $\{0\}$**：计算迭代核（Fubini 累次积分）

$$
|(A^nx)(t)|\le\|x\|_\infty\frac{t^n}{n!}\le\frac{\|x\|_\infty}{n!}
\qquad\Longrightarrow\qquad
\|A^n\|^{1/n}\le\frac{1}{(n!)^{1/n}}\to0.
$$

Gelfand 谱半径公式（定理 5.1.5）给 $r(A)=0$——**谱全部挤在 $\{0\}$**（谱非空 ⟹ $\sigma(A)=\sigma_r(A)=\{0\}$）。

**读法**：$\|A\|=1$ 但 $r(A)=0$——**范数与谱半径可以差任意远**。Volterra 算子"每次积分抹平一档"，迭代 $n$ 次误差衰减 $1/n!$（快于任何几何衰减）；这是"光滑化算子"的谱论签名。$r(A)=0$ 的算子叫**拟幂零**（quasinilpotent）——谱论里的"温和派"。

### 3.2 定理 5.1.1：预解式等式（谱论的代数引擎）

> **定理 5.1.1（预解式等式）**　$A$ 闭线性算子，$\lambda,\mu\in\rho(A)$。则
> $$
> R(\lambda,A)-R(\mu,A)=(\mu-\lambda)R(\lambda,A)R(\mu,A).
> $$

**证明（完整——一页纯代数）**　$\mu\in\rho(A)$ 给 $(\mu-A)R(\mu,A)=I_X$。于是

$$
R(\lambda,A)=R(\lambda,A)\,I_X=R(\lambda,A)(\mu-A)R(\mu,A)
=R(\lambda,A)\big[(\lambda-A)+(\mu-\lambda)\big]R(\mu,A)
$$
$$
=\big[I_X+(\mu-\lambda)R(\lambda,A)\big]R(\mu,A)=R(\mu,A)+(\mu-\lambda)R(\lambda,A)R(\mu,A).\;\blacksquare
$$

**读法**：把 $R(\lambda,A)(\lambda-A)=I_X$ 里的 $\lambda-A$ 换成 $(\mu-A)+(\mu-\lambda)$——**"换底"一次，等式落地**。它是数值分析里恒等式 $\frac1{\lambda-a}-\frac1{\mu-a}=\frac{\mu-\lambda}{(\lambda-a)(\mu-a)}$ 的算子版（把 $a$ 换成 $A$、除法换成求逆），但注意**乘法不可交换，插入位置必须精确**。后续所有证明（连续性、可导性）全靠它做差——**谱论的"分部积分"**。

### 3.3 引理 5.1.1 + 定理 5.1.2：Neumann 级数与预解集开性

> **引理 5.1.1（J. von Neumann）**　$X$ 复 Banach，$A\in B(X)$，$\|A\|<1$。则 $1-A$ 在 $B(X)$ 中可逆，且 $\|R(1,A)\|\le\frac{1}{1-\|A\|}$。

**证明（完整——几何级数的算子版）**　部分和 $S_n=1+A+\cdots+A^n$。$\|S_{n+m}-S_n\|\le\sum_{k>n}\|A\|^k\le\frac{\|A\|^{n+1}}{1-\|A\|}\to0$——$\{S_n\}$ 是 $B(X)$ 中的 Cauchy 列。**$X$ Banach ⟹ $B(X)$ Banach（FA 06 定理 2.4.5）**，故 $S_n\to S\in B(X)$。乘法连续性给

$$
S(1-A)=\lim_nS_n(1-A)=\lim_n(1-A^{n+1})=I_X
$$

（$\|A^{n+1}\|\le\|A\|^{n+1}\to0$），同理 $(1-A)S=I_X$。范数估计沿部分和取极限。∎

**读法**：这就是**几何级数 $\frac{1}{1-a}=\sum a^n$ 的非交换版**，收敛的账单由 $B(X)$ 的完备性支付（FA 04 "绝对收敛 ⟹ 收敛"需要 Banach——这里是"范数级数收敛 ⟹ 算子级数收敛"，同一个机制）。**"‖A‖<1 ⟹ 1−A 可逆"是全篇最常用的一块砖**，马上连续用两次。

> **定理 5.1.2**　$A$ 闭线性算子。则 $\rho(A)$ 是 $\mathbb{C}$ 的**开集**。

**证明（完整）**　设 $\lambda_0\in\rho(A)$。对任意 $\lambda\in\mathbb{C}$，**关键分解**（式 5.2）：

$$
\lambda-A=\big(\lambda_0-A\big)+(\lambda-\lambda_0)=\Big(1+(\lambda-\lambda_0)R(\lambda_0,A)\Big)(\lambda_0-A).
$$

（验证：展开时 $(\lambda-\lambda_0)R(\lambda_0,A)(\lambda_0-A)=(\lambda-\lambda_0)I_X$。）当 $|\lambda-\lambda_0|\cdot\|R(\lambda_0,A)\|<1$ 时，von Neumann 引理给第一个因子 $1+(\lambda-\lambda_0)R(\lambda_0,A)$ 可逆（它是 $1-B$ 形状，$\|B\|<1$）；第二个因子 $\lambda_0-A$ 可逆（$\lambda_0\in\rho(A)$）。**两个可逆算子的复合可逆** ⟹ $\lambda\in\rho(A)$。即开圆盘 $B\big(\lambda_0,\frac{1}{\|R(\lambda_0,A)\|}\big)\subset\rho(A)$。∎

**读法（两条）**：

1. **谱对参数稳定**：$\rho$ 开 ⟹ $\sigma=\mathbb{C}\setminus\rho$ **闭**。正则值经不起扰动但也不会被瞬间摧毁——半径 $\frac{1}{\|R(\lambda_0,A)\|}$ 精确量化"离谱多远才安全"：**预解式越大（逆越病态），安全半径越小**。数值线性代数的"谱摄动敏感性"（条件数越大特征值越怕扰动）在这一个圆盘里已经全息。
2. **分解式 (5.2) 是"相对扰动"技巧**：把 $\lambda$ 相对 $\lambda_0$ 的偏移写成乘法因子——预解式等式供弹药、Neumann 引理供可逆性、分解式供组装。三件套在定理 5.1.3（解析性）里原班人马再演一场。

### 3.4 定理 5.1.3：预解式是解析函数（谱论的分析引擎）

> **定理 5.1.3**　闭算子 $A$ 的预解式 $\rho(A)\to B(X)$，$\lambda\mapsto R(\lambda,A)$ 为解析函数，且 $R'(\lambda,A)=-R(\lambda,A)^2$。

**证明（完整）**　固定 $\lambda_0\in\rho(A)$，取 $|\lambda-\lambda_0|<\frac{1}{2\|R(\lambda_0,A)\|}$。

**有界性**：由分解 (5.2) 取逆（注意 $(ST)^{-1}=T^{-1}S^{-1}$ 反序）：

$$
R(\lambda,A)=R(\lambda_0,A)\big(1+(\lambda-\lambda_0)R(\lambda_0,A)\big)^{-1},
$$

Neumann 级数估计给 $\|(1+(\lambda-\lambda_0)R(\lambda_0,A))^{-1}\|\le\frac{1}{1-\frac12}=2$，故 $\|R(\lambda,A)\|\le2\|R(\lambda_0,A)\|$（式 5.3——**预解式局部一致有界**）。

**连续性**：预解式等式给 $R(\lambda,A)-R(\lambda_0,A)=(\lambda_0-\lambda)R(\lambda_0,A)R(\lambda,A)$，于是 $\|R(\lambda,A)-R(\lambda_0,A)\|\le2\|R(\lambda_0,A)\|^2|\lambda-\lambda_0|\to0$。

**可导性**：差商用预解式等式改写

$$
\frac{R(\lambda,A)-R(\lambda_0,A)}{\lambda-\lambda_0}=-R(\lambda,A)R(\lambda_0,A)\ \longrightarrow\ -R(\lambda_0,A)^2,
$$

（极限号由刚证的连续性 + 乘法连续性兑现）。∎

**读法**：$\frac{d}{d\lambda}\frac{1}{\lambda-a}=-\frac{1}{(\lambda-a)^2}$ 的算子版——**预解式是"算子值的有理函数"**，在 $\rho(A)$ 上全纯、在谱点处"有极点"（严格说极点仅对孤立谱点成立，FA 14 紧算子的非零谱点就是极点）。本篇的两大定理（Gelfand–Mazur、Gelfand 公式）都是"**预解式解析 ⟹ 压缩到纯量复分析**"这一次跳水的产物。

### 3.5 定理 5.1.4：Gelfand–Mazur——没有谱的算子不存在

> **定理 5.1.4（Gelfand–Mazur）**　$X$ 复 Banach，$A\in B(X)$。则 $\sigma(A)\ne\varnothing$。

**证明（完整——Liouville 定理的算子版首秀）**　$A=0$ 时 $0\in\sigma(A)$（$X\ne\{0\}$）。设 $A\ne0$，反设 $\sigma(A)=\varnothing$，则 $\rho(A)=\mathbb{C}$，预解式**整平面解析**。

**第一步（远处衰减）**：$|\lambda|>2\|A\|$ 时 $\big\|\frac{A}{\lambda}\big\|<\frac12$，Neumann 引理给

$$
\lambda-A=\lambda\Big(1-\frac{A}{\lambda}\Big)\ \text{可逆},\qquad
\|R(\lambda,A)\|\le\frac{1}{|\lambda|}\cdot\frac{1}{1-\frac12}=\frac{2}{|\lambda|}\le\frac{1}{\|A\|}. \tag{5.4}
$$

**第二步（压缩到纯量）**：任取 $\varphi\in B(X)'$，$\lambda\mapsto\varphi(R(\lambda,A))$ 是**整函数**（定义 5.1.3 的对偶桥梁），且由式 (5.4) 它**有界**（$\bar B(0,2\|A\|)$ 上连续故有界，圆外被 $\frac{\|\varphi\|}{\|A\|}$ 控制）。**Liouville 定理**：有界整函数必常——$\varphi(R(\lambda,A))=c_\varphi$ 对一切 $\lambda$。

**第三步（HB 分离收账）**：若 $\lambda\ne\mu$，预解式等式给 $R(\lambda,A)-R(\mu,A)=(\mu-\lambda)R(\lambda,A)R(\mu,A)$；两个可逆算子的复合可逆故 $\ne0$，于是 $R(\lambda,A)\ne R(\mu,A)$。**Hahn-Banach（FA 09 定理 4.1.4）**：存在 $\varphi\in B(X)'$ 分离它们，$\varphi(R(\lambda,A))\ne\varphi(R(\mu,A))$——与第二步"对每个 $\varphi$ 都恒常"矛盾。∎

**品味**：证明是一次三级的跳水——**算子值函数 →（对偶泛函）→ 纯量整函数 →（Liouville）→ 常函数 →（HB 分离）→ 矛盾**。每一级都在复习前章：解析性的定义靠对偶观测（FA 09），分离靠 HB（FA 09），远处衰减靠 Neumann 级数（本篇）。结论本身极具哲理：**复数域上"处处可逆"的非零算子是幻影**——有限维早已如此（特征多项式总有根，代数基本定理），Gelfand–Mazur 是它在无穷维的接棒，接棒的手是复分析而非代数。

**推论（引理 5.1.2 + 定义 5.1.4）**：$|\lambda|>\|A\|$ 时 $\lambda\in\rho(A)$（Neumann），故

$$
\sigma(A)\subset\{\lambda:|\lambda|\le\|A\|\}\ \text{有界闭（紧！）},\qquad r(A):=\max_{\lambda\in\sigma(A)}|\lambda|\ \text{（谱半径）存在}.
$$

**谱是复平面上的紧集**——这是"谱"配得上"谱"这个名字的原因：它是算子携带的一个**有限的可观测对象**（紧集可以用有限信息逼近），而不是散落全平面的幽灵。$\max$ 能取到（紧集上连续函数取最值——FA 02 的紧性红利），又一次 **sup→max**（FA 09 收官表的续篇：这次靠谱紧性）。

### 3.6 定理 5.1.5：Gelfand 谱半径公式（UBP 兑现）

> **定理 5.1.5（I. M. Gelfand）**　$A\in B(X)$。则极限存在且
> $$
> r(A)=\lim_{n\to\infty}\|A^n\|^{1/n}.
> $$

**证明（结构完整——三段）**　记 $r=\inf_{n\ge1}\|A^n\|^{1/n}$。

**第一段（极限存在 = inf）**：对任意 $\varepsilon>0$ 取 $n_0$ 使 $\|A^{n_0}\|^{1/n_0}<r+\varepsilon$。任给 $n$，分解 $n=kn_0+m$（$0\le m<n_0$），次可乘性给

$$
\|A^n\|^{1/n}\le\Big(\|A^{n_0}\|^{1/n_0}\Big)^{\frac{kn_0}{n}}\cdot\|A\|^{\frac{m}{n}}\ \longrightarrow\ \|A^{n_0}\|^{1/n_0}<r+\varepsilon\qquad(n\to\infty),
$$

令 $\varepsilon\to0^+$ 得 $\lim\|A^n\|^{1/n}=r=\inf_n\|A^n\|^{1/n}$。（**下确界的近似优化 + 整除分解**——和 FA 06 算子范数"试验向量逼近"同款思想：inf 被一个子列近似达到，其余指标被次可乘性摊平。）

**第二段（$r(A)\le r$：Neumann 级数再就业）**：$|\lambda|>r$ 时取 $\varepsilon$ 使 $\|A^n\|^{1/n}\le r+\varepsilon<|\lambda|$，级数 $\sum_k\frac{A^k}{\lambda^{k+1}}$ 在 $B(X)$ 中绝对收敛（几何控制），直接验证 $(\lambda-A)S=S(\lambda-A)=I$，故 $\{|\lambda|>r\}\subset\rho(A)$，即 $r(A)\le r$，且

$$
R(\lambda,A)=\sum_{k\ge0}\frac{A^k}{\lambda^{k+1}}\qquad(|\lambda|>r).
$$

**第三段（$r\le r(A)$：解析延拓 + UBP 收账）**：任取 $\varphi\in B(X)'$，在 $\{|\lambda|>r\}$ 上 $\varphi(R(\lambda,A))=\sum_k\frac{\varphi(A^k)}{\lambda^{k+1}}$ 是 **Laurent 展开**；但左边在更大的区域 $\{|\lambda|>r(A)\}$ 上解析（定理 5.1.3），由 Laurent 展开的唯一性，**展开式在更大区域仍成立**——特别地，$\sum_k\frac{\varphi(A^k)}{(r(A)+\varepsilon)^{k+1}}$ 对每个 $\varphi$ 收敛，即

$$
\sup_{k\ge0}\Big|\,J(A^k)(\varphi)\,\Big|\cdot\frac{1}{(r(A)+\varepsilon)^{k+1}}<\infty\qquad(\forall\varphi\in B(X)').
$$

**一致有界性原理（FA 10 定理 4.2.2）**登场：$B(X)'$ 是 Banach 空间，逐 $\varphi$ 有界 ⟹ 一致有界，$\|A^k\|\le C(r(A)+\varepsilon)^{k+1}$，开 $k$ 次方取极限给 $r\le r(A)+\varepsilon$，再令 $\varepsilon\to0$。∎

**品味**：第三段是全篇最深的机关——**"每个观测者看到的有界"经 UBP 升级为"一致有界"**，正是 FA 10 反例表里"丢线性才分家"的那座桥。而 Laurent 展开的唯一性充当"解析延拓的指纹"：同一列系数 $\varphi(A^k)$ 在小圆外和大圆外必须讲同一个故事，故事讲得下去就逼出衰减率。**Gelfand 公式的实用价值**：$\|A^n\|^{1/n}$ 可以用幂迭代数值逼近（只需矩阵乘法，不需要找特征值）——PageRank、Markov 收敛率估计的实际算法。

**例 5.1.3（有限维对账）**　$A\in B(\mathbb{C}^n)$：$\sigma(A)=\{\lambda_1,\dots,\lambda_n\}$（特征多项式的根），$r(A)=\max_i|\lambda_i|$——一切与线性代数对上，且 $\|A^n\|^{1/n}\to\max|\lambda_i|$（即便 $\|A\|\gg r(A)$，如严重非正规矩阵）。

---

## 4. 例子与反例速查

| 命题 | 真假 | 例子/反例 |
|------|------|----------|
| $\mathbb{C}=\sigma_p\cup\rho$ 干净二分 | ✅ 仅有限维 | 单射⟹满射（秩—零化度）；无穷维 $\sigma_c,\sigma_r$ 登场 |
| 存在谱为空的有界算子 | ❌ | Gelfand–Mazur（复 Banach 上 $\sigma\ne\varnothing$；Liouville 拒绝） |
| $\sigma_p(A)=\varnothing$ 但 $\sigma(A)$ 不可数 | ✅ | 乘法算子 $(Ax)(t)=tx$：$\sigma=\sigma_r=[0,1]$ |
| $\sigma_c\ne\varnothing$ 的算子 | ✅ | 对角算子 $\alpha_n\searrow0$：$0\in\sigma_c$（值域稠密不满） |
| $\sigma(A)=\{0\}$ 且 $A\ne0$ | ✅ | Volterra：$r(A)=0$（$\|A^n\|\le1/n!$）——拟幂零 |
| $\|A\|=1$ 但 $r(A)=0$ | ✅ | 同上——范数与谱半径可差任意远 |
| $\lambda\in\rho(A)$ ⟹ $(\lambda-A)^{-1}$ 自动有界 | ✅ | 闭图像定理（闭算子 + 定义域全空间 Banach） |
| $\rho(A)$ 是开集、$\sigma(A)$ 是紧集 | ✅ | 定理 5.1.2 + 引理 5.1.2；谱半径 $\max$ 取得到（紧性红利） |
| 谱只依赖算子，不依赖所在空间 | ❌ | 习题 5/6：右移算子在 $\ell^\infty$ 上 $\sigma_p=\{|\lambda|\le1\}$，在 $\ell^2$ 上 $|\lambda|=1$ 不是特征值 |
| $\lambda$ 是 $A^m$ 的特征值 ⟹ $\lambda$ 的某 $m$ 次根是 $A$ 的特征值 | ✅ | 题 7（谱映射定理 $p(\sigma(A))=\sigma(p(A))$ 的点谱版雏形） |
| 幂等算子（$A^2=A\ne0,I$）的谱 | $\sigma=\{0,1\}$ | 题 8：$A$ 是投影型，$R(A)=\{x:Ax=x\}$ 分离 0 与 1 |
| $|\lambda|>\|A\|$ ⟹ $\lambda\in\rho(A)$ | ✅ | Neumann 级数 $\sum A^n/\lambda^{n+1}$——谱的先验定位 |
| $|\lambda|>\|A\|$ 是 $\rho$ 的最优范围 | ❌ | Volterra $\|A\|=1$ 但 $\rho=\mathbb{C}\setminus\{0\}$——范数界远非紧 |
| 特征值互异的特征向量线性无关 | ✅ | 引理 5.2.2（§5.2 预告，归纳 + "作用 A 再减"经典招） |

---

## 5. 易混淆辨析

| 概念对 | 辨析 |
|--------|------|
| **特征值 vs 谱点** | 特征值：**不单射**（有非零核）；谱点：**不可逆**（三种病理任一）。有限维两者重合（不单射⟹不可逆），无穷维 $\sigma\setminus\sigma_p$ 可以占满整个谱（乘法算子：无特征值但谱是整条线段）。**"求特征值"只是谱论的三分之一**。 |
| **连续谱 vs 剩余谱** | 都是"单射不满射"，分界在**值域稠不稠**：稠密（$\sigma_c$）= 方程可近似解到任意精度但精确解可缺；不稠密（$\sigma_r$）= 存在泛函整体零化值域、方程结构性无解。记忆：**连续谱"差一点"，剩余谱"差一个方向"**。乘法算子（$\sigma_r$）vs 对角算子的聚点（$\sigma_c$）是一对标准样本。 |
| **$\rho(A)$ 开 vs $\sigma(A)$ 紧** | 同一硬币两面（$\sigma=\mathbb{C}\setminus\rho$），但来源不同：开性是 Neumann 级数买的（局部扰动不破坏可逆），有界性也是 Neumann 买的（$\|\lambda\|>\|A\|$ 必正则），闭性是开性的余集。**谱的"紧"让 max 取到、让谱半径良定义**——sup→max 故事线（FA 06❌/FA 08✅/FA 09✅）的谱论续集。 |
| **$r(A)$ vs $\|A\|$** | 永远 $r(A)\le\|A\|$（Neumann），可以严格差任意远（Volterra：$1$ vs $0$）。$\|A\|$ 是"一次作用的放大率"，$r(A)$ 是"**长期迭代**的渐近放大率" $\lim\|A^n\|^{1/n}$——差值全部来自**非正规性/循环抵消**。FA 14：自伴（正规）算子恰好 $r(A)=\|A\|$，差值是非正规性的谱论度量。 |
| **预解式 $R(\lambda,A)$ vs $(\lambda-A)^{-1}$ 的记号** | 同一个东西；$R(\lambda,A)$ 强调"**$\lambda$ 的函数**"（解析性是核心属性）。两个福利公式：$R(\lambda,A)-R(\mu,A)=(\mu-\lambda)R(\lambda,A)R(\mu,A)$（代数）；$R'(\lambda,A)=-R(\lambda,A)^2$（分析）。**预解式把一族逆算子组织成一个解析函数**——谱论的全部复分析工具由此进门。 |
| **谱依赖算子 vs 依赖"算子+空间"** | 同一形式的不同空间上谱可以巨变：右移 $S$ 在 $\ell^2$ 上 $\sigma_p=\varnothing$（$Sx=\lambda x$ 递推逼 $x=0$，$|\lambda|\le1$ 时也在 $\ell^2$ 容不下），在 $\ell^\infty$ 上 $\sigma_p=\{|\lambda|\le1\}$（特征向量 $x=(1,\lambda,\lambda^2,\dots)$ 有界！）。**谱是"算子+空间"的联合属性**——与 FA 11"无界是算子+范数配对属性"同款教训。 |
| **闭算子谱论 vs 一般算子谱论** | 教材只做闭算子的原因：闭 ⟹ $\lambda-A$ 闭 ⟹ 图像翻转闭 ⟹ 闭图像定理白送 $R(\lambda,A)$ 有界。**没有闭性，"正则值"定义里的逆可以无界**，整个理论的地基（预解式 ∈ B(X)）塌掉。微分算子天然闭而不有界（FA 11 例 4.4.2）——无界算子谱论全靠"闭"字撑着（量子力学的可观测量全部住在这一页）。 |

---

## 6. ML 连接

### 6.1 谱半径 = 迭代系统的稳定性判据（RNN / Echo State）

递归系统的核心问题"$x_{n+1}=Wx_n+\cdots$ 会不会爆炸"的答案是谱半径：迭代 $n$ 步的放大率是 $\|W^n\|^{1/n}\to r(W)$（Gelfand 公式），**$r(W)<1$ ⟹ 几何收敛、$r(W)>1$ ⟹ 沿主导方向指数爆炸**。 Echo State Property（储备池计算的可训条件）的经典充分条件 $\|W\|<1$ 只是 $r(W)<1$ 的保守替代——本篇 Volterra 例子说明两者可差任意远：**范数判据"每次都缩"，谱半径判据"长期净缩"**，允许局部放大、只要循环抵消。梯度爆炸/消失（RNN 长程依赖的本质困难）、动力系统稳定域、Markov 链收敛率（$|\lambda_2|<1$ 的谱隙）——全是 $r(\cdot)$ 的应用侧写；幂迭代数值求 $r(W)$ 不需要谱分解（PageRank 的原始动机）。

### 6.2 Neumann 级数 = 残差网络与隐式模型的展开

$\|A\|<1$ 时 $(1-A)^{-1}=\sum_{n\ge0}A^n$——"求逆"被"无穷次自作用"取代。ML 里的三个化身：**残差流** $x_{t+1}=x_t+f(x_t)$ 的不动点视图中，$(I-\lambda J)^{-1}$ 型算子反复出现（implicit models / DEQ 的不动点迭代收敛性 = 相应算子压缩）；**图神经网络的消息传递** $H^{(k+1)}=\sigma(\hat AH^{(k)}W)$ 展开后是 $\hat A$ 的幂级数——**GNN 的感受野就是 Neumann 级数的截断长度**，$r(\hat A)<1$（如重归一化 $D^{-1/2}AD^{-1/2}$）是过平滑/爆炸的分界；**迭代正则化**（Landweber 迭代 $x_{k+1}=x_k+\alpha A^*(y-Ax_k)$）的收敛域分析同样是 Neumann 型级数。本篇给的定量礼物是收敛半径与 $\|R(\lambda,A)\|$ 的精确换算——**"离谱多远，展开就多稳"**。

### 6.3 预解式解析性：模型关于超参数的平滑依赖

$\lambda\mapsto R(\lambda,A)$ 解析且 $R'=-R^2$——**"逆算子"对参数的依赖不仅连续而且光滑**，且有显式微分方程。这是"**训练好的模型对超参数/数据扰动的敏感性**"的数学原型：正则化参数 $\lambda$ 扫描（regularization path）时，解 $\hat\beta(\lambda)=(X^TX+\lambda I)^{-1}X^Ty$ 是 $\lambda$ 的解析函数——路径平滑意味着可用连续优化/热启动策略；预解式在谱点附近的无界增长（$\|R\|\to\infty$）则是"临界超参数"的预警。定理 5.1.2 的稳定半径 $\frac{1}{\|R(\lambda_0,A)\|}$ 直接翻译成数值实践中**条件数越大越怕扰动**的定量版本——FA 11 §6.1 条件数故事在"参数轴"上的细化。

---

## 7. 逻辑推导概念图

下面这棵树把全篇的推导链压缩成一张图：顶部是出发点（闭算子假设、特征值/正则值的定义、四分法的三道关卡），缩进与箭头是推理方向，每一步只使用上游已经证得的结论。读者可沿树自上而下核对"哪个定理用了哪条假设"：闭性只服务预解式的有界性，完备性只服务 Neumann 级数，复结构只服务 Gelfand–Mazur 与 Gelfand 公式——三份前提各管一段，互不越界。

```text
出发点（公理与定义）
├── X 非零复 Banach 空间，A:D(A)→X 闭线性算子（定义 5.1.1）
│   ├── A 闭 ⟹ λ-A 闭 ⟹ 图像翻转仍是闭集
│   │   └── 闭图像定理（FA 11）→ λ∈ρ(A) 时 R(λ,A) 自动有界
│   └── 预解式等式（定理 5.1.1）做差
│       └── R(λ,A) 连续且可导，R'=-R^2（定理 5.1.3：预解式解析）
│
├── B(X) 完备（X Banach ⟹ B(X) Banach，FA 06 定理 2.4.5）
│   └── Neumann 级数（引理 5.1.1）：‖A‖<1 ⟹ 1-A 可逆
│       ├── 定理 5.1.2：ρ(A) 开（相对扰动分解）⟹ σ(A) 闭
│       └── 引理 5.1.2：|λ|>‖A‖ ⟹ λ∈ρ(A) ⟹ σ(A) 有界
│
├── 定义 5.1.2 的三道关卡（λ-A 不可逆的病理分型）
│   ├── 不单射 → σ_p（特征值）
│   ├── 单射不满射、值域稠密 → σ_c（连续谱）
│   └── 单射不满射、值域不稠密 → σ_r（剩余谱）
│
▼ 三路合流：谱的结构定理
├── 预解式解析 + 对偶压缩 φ∘f（定义 5.1.3）+ Liouville + HB 分离
│   └── Gelfand–Mazur（定理 5.1.4）：σ(A)≠∅——没有谱的算子不存在
├── σ 非空 + 有界 + 闭 ⟹ σ(A) 紧，谱半径 r(A)=max|λ| 存在
└── Laurent 展开 + 一致有界性原理（FA 10 定理 4.2.2）
    └── Gelfand 谱半径公式（定理 5.1.5）：r(A)=lim ‖A^n‖^{1/n}
```

| 出发点 | 买到什么 | 撤掉它失去什么 |
|---|---|---|
| A 是闭算子（教材的讨论范围） | λ-A 闭 → 图像翻转闭 → 闭图像定理白送 $R(\lambda,A)\in B(X)$（§2.2） | 正则值的定义只要求一一映射，逆可以无界——预解式 $\in B(X)$ 这块地基塌掉，解析性、谱半径公式全部悬空（§5 末行辨析） |
| X 是复 Banach 空间 | Gelfand–Mazur：$\sigma(A)\ne\varnothing$；谱半径良定义（§3.5） | 实空间上旋转矩阵 $\begin{pmatrix}0&-1\\1&0\end{pmatrix}$ 没有实特征值，复化后谱 $=\{\pm i\}$ 才现身——"处处可逆"的幻影排不掉，谱论"远没有复空间完美"（§1） |
| B(X) 的完备性 | Neumann 级数收敛：$\rho(A)$ 开、$|\lambda|>\|A\|$ 必正则（§3.3） | 部分和 $S_n$ 无处收敛，"‖A‖<1 ⟹ 1−A 可逆"这块全篇最常用的砖失效，谱的有界性先验估计一并消失 |
| 定义 5.1.2 的稠密性分界 | $\sigma_c$ 与 $\sigma_r$ 分家，四分法读表（§2.1） | 乘法算子（$\sigma_r=[0,1]$，结构性无解）与对角算子聚点 $0$（$\sigma_c$，可近似解）无法区分谱型——"差一点"与"差一个方向"混作一团（§4 速查） |

---

## 8. 习题精选（教材习题 5 选摘）

**题 1**：$v\in C[0,1]$ 固定，$(Ax)(t)=v(t)x(t)$。求 $\sigma(A)$。
*提示*：完全复制例 5.1.1 的三层检查，答案 $\sigma(A)=v([0,1])$（**连续函数的紧像集**，自动闭）。$\lambda\notin v([0,1])$：距离 $\inf_t|\lambda-v(t)|>0$（紧性！），$\frac{1}{\lambda-v(t)}$ 连续 ⟹ $\lambda\in\rho(A)$。$\lambda\in v([0,1])$：记 $E_\lambda=\{t:v(t)=\lambda\}$（闭非空）。任取 $y\in R(\lambda-A)$ 有 $y|_{E_\lambda}=0$，一致极限保持逐点条件 ⟹ 闭包也如此 ⟹ 值域不稠密、常函数 1 进不来。再看单射侧：$(\lambda-v)x=0$ ⟹ $x$ 在 $E_\lambda$ 之外为零；若 $E_\lambda$ **无内点**，则 $E_\lambda$ 之外稠密，连续性逼 $x=0$（单射，$\lambda\in\sigma_r(A)$）；若 $E_\lambda$ **有内点**（$v$ 在某段上恒等于 $\lambda$），则在 $\operatorname{int}E_\lambda$ 内造连续凸包、边界归零的函数即得特征向量（$\lambda\in\sigma_p(A)$）。**完整判据**：$\lambda\in\sigma_p\iff v$ 在一个非退化区间上恒取 $\lambda$；否则 $\lambda\in\sigma_r$。例 5.1.1（$v(t)=t$，每个 $E_\lambda$ 是单点）落在"全剩余谱"一侧。**乘法算子的谱 = 符号的像**：这是谱理论里"算子 = 广义函数"的第一课（FA 14 的乘法谱定理将是它的终章）。

**题 4**：$T_n,T\in B(X)$，$\|T_n-T\|\to0$，$\lambda_0\in\rho(T)$。求证 $n$ 足够大时 $\lambda_0\in\rho(T_n)$，且 $R(\lambda_0,T_n)\to R(\lambda_0,T)$。
*提示*：扰动分解（定理 5.1.2 证明的翻版）：
$$
\lambda_0-T_n=\big(1-(T_n-T)R(\lambda_0,T)\big)(\lambda_0-T)
$$
（展开时 $(T_n-T)R(\lambda_0,T)(\lambda_0-T)=T_n-T$）。$\|(T_n-T)R(\lambda_0,T)\|\le\|T_n-T\|\,\|R(\lambda_0,T)\|\to0$，Neumann 引理收工（$n$ 大时第一因子可逆）。收敛性用**第二预解式恒等式**（题 9）：
$$
R(\lambda_0,T_n)-R(\lambda_0,T)=R(\lambda_0,T_n)\big(T-T_n\big)R(\lambda_0,T),
$$
两边取范数 $\le\|R(\lambda_0,T_n)\|\,\|T_n-T\|\,\|R(\lambda_0,T)\|\to0$（$\|R(\lambda_0,T_n)\|$ 在 $n$ 大时被 Neumann 估计一致控制）。**谱的上半连续性**：正则值经范数小扰动不丢——数值上"算子摄动 ⟹ 正则值保住"的定量保证，也是摄动理论（Kato）的第一块砖。

**题 5 + 题 6（移位算子的两个房间）**：右移 $S(x_1,x_2,\dots)=(x_2,x_3,\dots)$。（5）在 $\ell^\infty$ 上：$|\lambda|>1$ 时 $\lambda\in\rho(S)$——解 $(\lambda-S)x=y$ 即 $\lambda x_n-x_{n+1}=y_n$，**尾部几何求和** $x_n=\sum_{j\ge0}\frac{y_{n+j}}{\lambda^{j+1}}$ 直接验证满足方程且有界（$|x_n|\le\frac{\|y\|_\infty}{|\lambda|-1}$）；$|\lambda|\le1$ 时 $x=(1,\lambda,\lambda^2,\dots)\in\ell^\infty$ 是特征向量（$Sx=\lambda x$ 逐项验证），且特征空间**恰为一维** $\operatorname{span}\{(1,\lambda,\lambda^2,\dots)\}$（$Sx=\lambda x$ 递推 $x_{n+1}=\lambda x_n$ 把整个 $x$ 锁定在首坐标上）。（6）在 $\ell^2$ 上 $|\lambda|=1$：同样递推给 $x_n=\lambda^{n-1}x_1$，$\sum|x_n|^2=\infty$——**$\notin\ell^2$，不是特征值**（事实上 $\ell^2$ 右移 $\sigma_p=\varnothing$、$\sigma=\{|\lambda|\le1\}$ 全是连续谱型）。**同一形式、两个空间、两份谱型**：$\ell^\infty$ 房"有界即可容纳几何列"，$\ell^2$ 房"平方可和把单位圆上的几何列拒之门外"——谱是算子+空间的联合属性（辨析 §5 第六行的实锤）。

**题 7**：$\lambda$ 是 $A^m$ 的特征值。求证 $\lambda$ 的某个 $m$ 次方根是 $A$ 的特征值。
*提示*：设 $A^mx=\lambda x$（$x\ne0$）。若 $\lambda=0$：取最小的 $j\ge1$ 使 $A^jx=0$，则 $A(A^{j-1}x)=0$、$A^{j-1}x\ne0$——$0$ 已是特征值。若 $\lambda\ne0$：设 $\mu_1,\dots,\mu_m$ 为 $\lambda$ 的 $m$ 个互异 $m$ 次方根，**多项式恒等式** $\prod_{j=1}^m(t-\mu_j)=t^m-\lambda$ 代入 $t=A$（多项式演算合法，因子两两交换）：
$$
0=(A^m-\lambda I)x=\prod_{j=1}^m(A-\mu_jI)\,x .
$$
若每个 $A-\mu_jI$ 都单射，则**交换的单射算子复合仍单射**（归纳：$A-\mu_1I$ 单射，把 $(\prod_{j\ge2})x\mapsto0$ 逼回 $(\prod_{j\ge2})x=0$，层层剥离），最终逼 $x=0$——矛盾。故某个 $A-\mu_jI$ 不单射，$\mu_j\in\sigma_p(A)$。∎ **谱映射的种子**：$\sigma(A^m)=\sigma(A)^m$（取 $m$ 次幂），一般版是题 10 的 $p(\sigma(A))=\sigma(p(A))$——"对算子做多项式 = 对谱做同样的多项式"，谱把算子代数压缩成数的代数。

**题 8**：$A^2=A\ne0,I$。求证 $\sigma(A)=\{0,1\}$。
*提示*：幂等 ⟹ $X=R(A)\oplus N(A)$ 直和分解（$x=Ax+(x-Ax)$，两块分别落在 $R(A)$、$N(A)$，交为零：$x=Ax\wedge Ax=0$ ⟹ $x=0$）。$A\ne0$ 给 $R(A)\ne\{0\}$（有 $x$ 使 $Ax\ne0$），$A\ne I$ 给 $N(A)\ne\{0\}$——**两个特征空间都非平凡**：$Ax=x$（$x\in R(A)$）给 $1\in\sigma_p$；$Ax=0$ 给 $0\in\sigma_p$。其余 $\lambda$：$(\lambda-A)^{-1}$ 在分解上逐块写显式（$R(A)$ 块上作用 $\frac{1}{\lambda-1}$、$N(A)$ 块上 $\frac{1}{\lambda}$）⟹ $\lambda\in\rho(A)$。**投影算子的谱就是它的"开关值"** $\{0,1\}$——FA 07 的正交投影（$P^2=P$）在此获得谱论身份；深度学习里的 gating/mask 算子（幂等线性层）谱型同款。

---

*下一篇：[泛函分析 14：紧算子与自伴算子的谱论](../FA_14_compact_self_adjoint_spectral_theory/)*
