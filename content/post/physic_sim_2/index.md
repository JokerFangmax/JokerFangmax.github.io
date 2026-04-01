---
title: "Physically-based Simulation Week 2"
date: 2026-04-01
math: true
tags:
  - Notes
  - Physics
  - Simulation
categories:
  - Simulation
description: "物理仿真第二周课堂笔记"
---

# Physically-Based Simulation 笔记：Rigid Body Simulation 与 Articulated-Body Simulation

> 依据课程课件《Rigid-Body Simulation》和《Articulated-Body Simulation》整理。
> 本文目标不是逐页抄写 PPT，而是把课件中的核心概念、公式、建模逻辑与实现流程串起来，整理成适合博客复习的结构化笔记。

---

## 1. 这两讲在讲什么？

这两份课件实际上围绕一个主线展开：

1. **先理解单个刚体怎么运动**：
   - 如何描述刚体的位置与朝向；
   - 如何定义线速度、角速度；
   - 如何从受力推出加速度与角加速度；
   - 如何把这些连续动力学方程变成数值仿真中的时间推进过程。

2. **再理解多个刚体如何通过关节与接触连接起来**：
   - 关节会带来几何约束；
   - 约束可以写成方程，也可以近似成“柔顺的弹簧约束”；
   - 接触会带来非穿透条件，进一步形成互补约束；
   - 摩擦会让接触问题更复杂，需要额外建模与数值求解。

所以，如果把整个内容压缩成一句话：

> **Rigid Bodies I** 负责建立“刚体运动学 + 刚体动力学 + 约束力”的基本语言；  
> **Rigid Bodies II** 则把这些内容装进“时间步进 + 关节约束 + 碰撞接触 + 摩擦求解”的完整仿真框架中。

---

## 2. 刚体运动的描述方式

### 2.1 世界坐标系与刚体坐标系

一个刚体的运动，不是描述刚体上每个粒子都各自怎么动，而是用一个整体刚性变换描述。

对于任意时刻 $t$，刚体的状态可以由两部分给出：

- **平移**：质心位置 $\mathbf{x}(t) \in \mathbb{R}^3$
- **旋转**：朝向矩阵 $\mathbf{R}(t) \in SO(3)$

其中：

- 世界坐标系是静止参考系；
- 刚体坐标系（body frame）固定在刚体上；
- $\mathbf{R}(t)$ 把刚体局部坐标变换到世界坐标；
- $\mathbf{x}(t)$ 通常取刚体某个参考点的位置，最常见的是**质心**。

因此，刚体上某个粒子在刚体局部坐标系中的固定位置记为 $\mathbf{b}_i$，那么它在世界坐标中的位置是：

$
\mathbf{x}_i(t) = \mathbf{x}(t) + \mathbf{R}(t)\mathbf{b}_i
$

这个式子非常重要，它表达了一个事实：

- 局部坐标 $\mathbf{b}_i$ 是不变的；
- 刚体所有点之所以运动，只是因为整体在平移和旋转。

---

### 2.2 为什么 body frame 通常放在质心？

课件中指出，我们可以调整 body frame，使得

$
\sum_i m_i \mathbf{b}_i = \mathbf{0}
$

这意味着：body frame 的原点选在质心上。

这样做的好处是：

$
\sum_i m_i \mathbf{x}_i(t) = m\mathbf{x}(t)
$

也就是说，$\mathbf{x}(t)$ 直接就是整个刚体的质心轨迹。

这会让后面动力学方程显著简化，因为：

- 平移动力学天然围绕质心最简单；
- 转动动力学中的惯性张量也更自然。

---

## 3. 刚体运动学：速度与加速度

---

### 3.1 线速度

质心的线速度定义最直接：

$
\mathbf{v}(t) = \dot{\mathbf{x}}(t)
$

线加速度则是：

$
\mathbf{a}(t) = \ddot{\mathbf{x}}(t)
$

---

### 3.2 角速度与旋转矩阵的导数

一个关键问题是：旋转矩阵 $\mathbf{R}(t)$ 的导数 $\dot{\mathbf{R}}(t)$ 应该如何理解？

因为 $\mathbf{R}(t)\in SO(3)$，它不是任意矩阵，而必须满足：

$
\mathbf{R}\mathbf{R}^T = \mathbf{I}
$

对时间求导可得：

$
\dot{\mathbf{R}}\mathbf{R}^T + \mathbf{R}\dot{\mathbf{R}}^T = 0
$

因此 $\dot{\mathbf{R}}\mathbf{R}^T$ 是一个**反对称矩阵**。任意三维反对称矩阵都可以对应某个向量 $\boldsymbol{\omega}$，于是有：

$
\dot{\mathbf{R}} = [\boldsymbol{\omega}]\mathbf{R}
$

其中 $[\boldsymbol{\omega}]$ 是由角速度向量构成的叉乘矩阵，满足：

$
[\boldsymbol{\omega}]\mathbf{u} = \boldsymbol{\omega}\times \mathbf{u}
$

所以常写成：

$
\dot{\mathbf{R}} = \boldsymbol{\omega} \times \mathbf{R}
$

这说明：

> 角速度不是“欧拉角变化率”，而是描述旋转矩阵瞬时变化的更本质量。

---

### 3.3 刚体上任意点的速度

由

$
\mathbf{x}_i = \mathbf{x} + \mathbf{R}\mathbf{b}_i
$

求导：

$
\mathbf{v}_i = \mathbf{v} + \dot{\mathbf{R}}\mathbf{b}_i
$

代入 $\dot{\mathbf{R}} = [\boldsymbol{\omega}]\mathbf{R}$：

$
\mathbf{v}_i = \mathbf{v} + \boldsymbol{\omega}\times (\mathbf{x}_i - \mathbf{x})
$

这条公式特别值得记住：

> 刚体上某点的速度 = 质心线速度 + 绕质心旋转产生的附加速度。

直观理解：

- 离质心越远，旋转项越大；
- 如果点就在质心，旋转项为 0；
- 如果角速度为 0，那么所有点只做整体平移。

---

### 3.4 刚体上任意点的加速度

继续对速度求导：

$
\mathbf{a}_i = \mathbf{a} + \dot{\boldsymbol{\omega}}\times (\mathbf{x}_i - \mathbf{x}) + \boldsymbol{\omega}\times\big(\boldsymbol{\omega}\times (\mathbf{x}_i - \mathbf{x})\big)
$

其中三项分别表示：

1. 质心的平动加速度；
2. 角加速度带来的切向加速度；
3. 离心/向心相关项。

这说明刚体上一点的加速度并不只是整体平移的结果，而是平移与转动共同决定的。

---

## 4. 刚体动力学：Newton–Euler 方程

课件接下来从受力出发，希望得到：

- 线加速度 $\mathbf{a}$
- 角加速度 $\dot{\boldsymbol{\omega}}$

---

### 4.1 平动：牛顿第二定律

对整个刚体，所有外力求和得到：

$
\sum_i \mathbf{f}_i = m\mathbf{a}
$

这和单个质点的牛顿第二定律形式一致，只不过这里的 $m$ 是整体质量，$\mathbf{a}$ 是质心加速度。

---

### 4.2 转动：欧拉方程

关于质心取矩：

$
\sum_i (\mathbf{x}_i - \mathbf{x}) \times \mathbf{f}_i = \mathbf{I}\dot{\boldsymbol{\omega}} + \boldsymbol{\omega}\times \mathbf{I}\boldsymbol{\omega}
$

左边是**外力矩**，右边是**角动量变化率**。

这个公式很关键。和初等力学里常见的

$
\boldsymbol{\tau} = I\alpha
$

相比，它多出来了一个非线性项：

$
\boldsymbol{\omega}\times \mathbf{I}\boldsymbol{\omega}
$

原因是：

- 在三维中，惯性张量是矩阵，不是标量；
- 转轴方向本身也可能变化；
- 因此角动量 $\mathbf{I}\boldsymbol{\omega}$ 与角速度 $\boldsymbol{\omega}$ 不一定同方向。

所以三维刚体旋转一般比二维复杂得多。

---

### 4.3 惯性张量是什么？

课件中给出惯性张量：

$
\mathbf{I} \triangleq \sum_i -m_i (\mathbf{x}_i - \mathbf{x})^{*2}
$

这里本质上是惯性矩阵的标准表达。更重要的是理解它的意义：

> 惯性张量描述了质量分布对转动“抗拒程度”的方向性。

它不是一个单纯数字，而是一个 **3×3 矩阵**，因为：

- 绕不同轴转，难易程度不同；
- 物体形状不对称时，不同方向会耦合。

---

### 4.4 世界系惯性张量与物体系惯性张量

如果刚体局部坐标系中的惯性张量是 $\mathbf{I}_b$，那么世界系下：

$
\mathbf{I} = \mathbf{R}\mathbf{I}_b\mathbf{R}^T
$

这非常重要，因为实际仿真中通常：

- 预先在 body frame 下计算 $\mathbf{I}_b$；
- 每一帧用当前朝向 $\mathbf{R}$ 转到世界系。

所以：

- **body frame** 中的惯性张量通常是常量；
- **world frame** 中的惯性张量会随物体旋转而变化。

---

## 5. 从连续动力学到数值仿真

课件把刚体动力学抽象成一个 ODE：

$
\dot{\mathbf{q}} = \mathbf{F}(\mathbf{q}; t)
$

其中状态变量可以写作：

$
\mathbf{q} = (\mathbf{x}, \mathbf{R}, \mathbf{v}, \boldsymbol{\omega})
$

于是一个刚体仿真器在每个时间步要做的事就是：

1. 读取当前状态 $(\mathbf{x}, \mathbf{R}, \mathbf{v}, \boldsymbol{\omega})$；
2. 统计外力与外力矩；
3. 计算当前世界系惯性张量 $\mathbf{I}$；
4. 解 Newton–Euler，得到 $\mathbf{a}, \dot{\boldsymbol{\omega}}$；
5. 更新速度与位置；
6. 更新角速度与旋转矩阵；
7. 进入下一步。

换句话说，刚体仿真本质上就是：

> **用数值积分器求解刚体状态随时间变化的 ODE。**

这里理论上可以套用前面课程里提到的：

- Forward Euler
- RK2
- RK4
- 以及更多高阶方法

但在实际刚体仿真里，仅仅“会积分”还不够，因为还有约束、碰撞、接触、摩擦等问题。

---

## 6. 关节系统与最大坐标（Maximal Coordinates）

当系统中有多个刚体通过关节连接时，例如：

- cart-pole
- 机械臂
- articulated character

就需要加入约束。

---

### 6.1 铰接体的建模思路

课件采用的是 **maximal coordinates** 视角：

- 每个 link 都保留完整刚体状态；
- 每个刚体都有自己的 $(\mathbf{x}_i, \mathbf{R}_i)$；
- 关节不直接减少自由度，而是通过约束方程把相邻刚体关联起来。

也就是说：

- 一个自由刚体有 6 个自由度；
- 多个刚体拼起来先得到很多自由度；
- 再用约束 $\phi(\mathbf{q})=0$ 把不允许的运动“消掉”。

这和机器人学中常见的最小坐标方法不同。最大坐标的优点通常是：

- 建模统一；
- 易于处理接触与拓扑变化；
- 对复杂约束系统更通用。

但代价是：

- 变量更多；
- 必须显式处理约束力。

---

### 6.2 关节约束的形式

课件写成：

$
\phi_i(\mathbf{q}_{parent}, \mathbf{q}_{child}) = 0
$

这表示：

- 每个关节都会给出一个几何约束函数；
- 当且仅当该约束为零时，关节是“满足”的。

例如：

- 铰链关节要求两个连接点重合，并且某些轴对齐；
- 平移关节要求只允许沿某一方向相对运动；
- 摆（pendulum）约束要求杆长固定。

因此，多刚体系统的核心之一是：

> 把“关节的几何要求”写成可微的约束函数 $\phi$。

---

## 7. 柔顺约束（Compliance Constraints）

这是两份课件里非常重要的一部分。

---

### 7.1 为什么不用硬约束，而引入柔顺约束？

理想情况下，我们希望：

$
\phi(\mathbf{q}) = 0
$

严格成立。

但数值上，硬约束往往带来：

- 难解的代数方程；
- 很强的刚性（stiffness）；
- 数值不稳定或条件数恶化。

因此，课件采取一个常见思路：

> 不把约束当作绝对不可违背的硬条件，而把它近似成一个很硬的弹簧。

对应势能：

$
V = \frac{1}{2}k\,\phi(\mathbf{q})\cdot \phi(\mathbf{q})
$

其中 $k$ 是刚度。

当 $k$ 很大时：

- 违反约束会产生很大能量；
- 系统就会“尽量”把 $\phi$ 拉回 0；
- 效果上接近硬约束。

---

### 7.2 由势能导出约束力

课件通过虚位移（virtual displacement）来推导约束力。

定义：

$
\delta V = -\mathbf{f}\cdot \delta \mathbf{q}
$

又因为

$
\lambda = k\phi(\mathbf{q})
$

于是最终会得到：

- 线性约束力：
  $
  -\mathbf{J}_v^T\lambda
  $
- 角向约束力矩：
  $
  -\mathbf{J}_\omega^T\lambda
  $

其中：

- $\mathbf{J}_v$ 是约束对平移变量的 Jacobian；
- $\mathbf{J}_\omega$ 是约束对旋转变量的 Jacobian；
- $\lambda$ 像是“约束强度”或“拉格朗日乘子样”的量。

因此刚体动力学方程变成：

$
\mathbf{f} - \mathbf{J}_v^T\lambda = m\mathbf{a}
$
$
\boldsymbol{\tau} - \mathbf{J}_\omega^T\lambda = \mathbf{I}\dot{\boldsymbol{\omega}} + \boldsymbol{\omega}\times \mathbf{I}\boldsymbol{\omega}
$

这组式子表达了一个核心思想：

> 约束不是直接“把状态投影回去”，而是通过额外的约束力/力矩影响动力学演化。

---

### 7.3 3D 摆的 Jacobian 例子

课件给出一个很经典的约束：

$
\phi(\mathbf{q}) = \mathbf{x} + \mathbf{R}\mathbf{b} - \mathbf{c}
$

含义是：

- 刚体上的某个局部点 $\mathbf{b}$，经过刚体变换后应与世界中的固定点 $\mathbf{c}$ 重合；
- 这正是球铰或摆式连接的最基本形式。

对应 Jacobian：

$
\mathbf{J} = [\mathbf{I}, -[\mathbf{R}\mathbf{b}]]
$

它的意义极强：

$
\mathbf{J}
\begin{bmatrix}
\mathbf{v} \\
\boldsymbol{\omega}
\end{bmatrix}
= \mathbf{v} + \boldsymbol{\omega}\times (\mathbf{R}\mathbf{b})
$

这正是该连接点在世界中的速度。

所以 Jacobian 的本质可以理解为：

> **把 generalized velocity（线速度 + 角速度）映射成关节点的实际速度。**

而 $\mathbf{J}^T\lambda$ 则反过来把接触/约束点上的力映射回刚体的 generalized force（力 + 力矩）。

这是后面接触与摩擦建模的基础。

---

## 8. Articulated-Body Simulation：从约束力到时间步进

到了第二份课件，重点从“刚体动力学公式”走向“真正可执行的时间步进算法”。

---

### 8.1 单个带约束 link 的离散方程

课件先考虑一个带单个约束 $\phi$ 的 link。

连续形式大致是：

- 平动：$m\mathbf{a}$ 相关
- 转动：$\mathbf{I}\dot{\boldsymbol{\omega}}$ 相关
- 再加上约束力项 $\mathbf{J}^T\lambda$

时间离散后，把步长记为 $h$，得到类似形式：

$
\mathbf{M}\dot{\mathbf{q}}^{n+1} + \mathbf{J}^T \mu = \mathbf{b}
$

其中：

- $\mathbf{M}$ 是广义质量矩阵；
- $\dot{\mathbf{q}}^{n+1}$ 是新速度；
- $\mu = h\lambda$ 是时间离散后的约束冲量形式；
- $\mathbf{b}$ 则包含上一时刻速度、外力、以及陀螺项等。

这一步的本质是：

> 不再直接求加速度，而是直接求“下一步速度”。

这在接触动力学里非常常见，因为速度级的约束更适合处理冲量与非穿透。

---

### 8.2 为什么要做隐式约束离散？

课件说明：

如果直接用显式 $\lambda = k\phi(\mathbf{q})$，虽然简单，但稳定性差。

于是采用隐式离散，近似：

$
\mu = hk\phi(\mathbf{q} + h\dot{\mathbf{q}}^{n+1})
$

再线性化为：

$
\mathbf{J}\dot{\mathbf{q}}^{n+1} - \frac{c}{h^2}\mu = -\frac{\phi}{h}
$

其中引入了 **compliance**：

$
c = k^{-1}
$

于是整体线性系统变成：

$
\begin{bmatrix}
\mathbf{M} & \mathbf{J}^T \\
\mathbf{J} & -\frac{c}{h^2}\mathbf{I}
\end{bmatrix}
\begin{bmatrix}
\dot{\mathbf{q}}^{n+1} \\
\mu
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{b} \\
-\frac{\phi}{h}
\end{bmatrix}
$

---

### 8.3 数值刚性（stiffness）与 compliance reformulation

如果用非常大的刚度 $k$ 来逼近硬约束，那么原系统中会出现非常大的量，造成：

- 条件数变差；
- 线性系统更难解；
- 数值不稳定。

因此课件把第二行整体除以 $k$，引入柔顺度 $c=1/k$，得到更好条件数的系统。

这一步特别重要，因为它不只是一个符号替换，而是一个数值线性代数意义上的改写：

> **把“非常硬的弹簧”写成“非常小的柔顺度”，可以让方程更好解。**

这也是现代很多基于约束的仿真器常见的思路。

---

### 8.4 多 link 系统的扩展

对于多个刚体组成的 articulated body：

- 质量矩阵 $\mathbf{M}$ 通常按 link 组成 block diagonal；
- 约束 Jacobian $\mathbf{J}$ 按关节拼接；
- 每个约束只涉及相关的两个 link，因此 $\mathbf{J}$ 通常很稀疏。

总流程是：

1. 遍历所有 link，组装 $\mathbf{M}$ 和 $\mathbf{b}$；
2. 遍历所有约束，计算 $\phi$ 与 $\mathbf{J}$；
3. 解线性系统得到新速度；
4. 再根据新速度更新位置与旋转。

这就是关节系统时间步进的主干。

---

## 9. 碰撞检测与接触建模

仅有关节还不够。仿真中最棘手的一类现象之一是：

- 刚体碰撞
- 接触
- 非穿透
- 释放
- 摩擦

第二份课件接下来就开始讲这个。

---

### 9.1 离散碰撞检测

课件先用一个最简单例子：

- 一个球撞击平面 $z=0$

若球心为 $\mathbf{c}$，半径为 $r$：

- 若 $c_z > r$：没有碰撞；
- 若 $c_z \le r$：发生碰撞，需要加入接触力。

这一步叫 **discrete collision detection**。

它的作用是：

> 找出哪些接触在当前时间步需要被纳入动力学求解。

---

### 9.2 接触点与局部接触坐标系

检测到碰撞后，需要构建：

- 接触点 $\mathbf{p}$
- 接触局部坐标系 $\mathbf{R}_p$

其中局部坐标系的第三轴（通常记 z 轴）沿接触法线方向。

这样做的目的，是把接触问题拆成：

- **法向** 分量：决定非穿透；
- **切向** 分量：决定摩擦。

这比在世界坐标系中直接处理要自然得多。

---

## 10. 接触 Jacobian：连接“刚体速度”和“接触点速度”

设接触点为 $\mathbf{p}$，其世界系速度可以写成：

$
\mathbf{v}_p = \mathbf{J}_p
\begin{bmatrix}
\mathbf{v} \\
\boldsymbol{\omega}
\end{bmatrix}
= \mathbf{J}_p\dot{\mathbf{q}}
$

再把它转到接触局部坐标系：

$
\mathbf{v}_p^{local} = \mathbf{R}_p^T \mathbf{J}_p\dot{\mathbf{q}} = \mathbf{J}^*\dot{\mathbf{q}}
$

类似地，接触点处的局部接触力 $\boldsymbol{\lambda}_p^{local}$ 通过

$
(\mathbf{J}^*)^T \boldsymbol{\lambda}_p^{local}
$

映射回刚体的广义力。

这再次体现 Jacobian 的双重角色：

- $\mathbf{J}\dot{\mathbf{q}}$：速度前向映射；
- $\mathbf{J}^T\lambda$：力反向映射。

这是接触动力学最核心的线性代数结构之一。

---

## 11. 无摩擦接触与互补条件（LCP）

---

### 11.1 为什么接触不是普通等式约束？

对于关节，我们希望始终满足 $\phi=0$。

但对于接触，不是这样。

因为接触有两种可能：

1. **真的压在一起**：存在法向接触力，法向相对速度为 0；
2. **已经分离/正在离开**：接触力为 0，但法向速度可以大于等于 0。

所以接触不是一个单纯的等式，而是一组**互补条件**。

---

### 11.2 无摩擦接触的法向条件

只看法向量，记：

- $\lambda_n$：法向接触力
- $v_n$：接触点法向相对速度

则接触 obey：

$
0 \le \lambda_n \perp v_n \ge 0
$

这等价于三个条件：

$
\lambda_n \ge 0,
\quad v_n \ge 0,
\quad \lambda_n v_n = 0
$

含义分别是：

1. **接触力不能为负**：物体不能“吸住”地面；
2. **法向速度不能朝内**：不允许继续穿透；
3. **二者不能同时严格正**：
   - 如果有接触力，则说明在贴住，故 $v_n=0$；
   - 如果在离开，则 $\lambda_n=0$。

这就是线性互补问题（LCP）的典型形式。

---

### 11.3 两种情况的物理理解

课件把它拆成两个 case：

#### 情况 A：No contact

$
\lambda_n = 0, \quad v_n \ge 0
$

表示：

- 没有法向支撑力；
- 物体正在离开或至少不再往里穿。

#### 情况 B：Contact

$
\lambda_n \ge 0, \quad v_n = 0
$

表示：

- 接触激活；
- 需要一个正的支撑力使法向速度被压到 0。

这正是“地面只能推，不能拉”的数学表达。

---

## 12. 多接触点为什么难？

如果系统中有 $m$ 个接触点，每个点都有两种情况：

- 接触
- 不接触

那么总组合数是：

$
2^m
$

这意味着暴力枚举会是指数复杂度。

接触点多时显然不可行。

这也是刚体接触仿真难点之一：

> 你不仅要解方程，还要同时决定“哪些接触激活、哪些接触释放”。

---

## 13. Blocked Gauss-Seidel（BGS）求解接触 LCP

为避免指数枚举，课件给出一种迭代式近似方法：**Blocked Gauss-Seidel**。

思路是：

1. 先给每个接触点随便指定一个状态，比如都设为 no contact；
2. 依次处理每个接触点：
   - 暂时冻结其他接触点的状态；
   - 只对当前这个点尝试它的两种 case；
   - 选出满足条件的状态；
3. 扫完一轮后继续循环；
4. 直到收敛，或者达到最大迭代次数。

它的优点是：

- 每次只看局部，复杂度低很多；
- 实现相对直接；
- 在很多工程场景中够用。

但缺点也很明显：

- 不一定保证收敛；
- 结果可能依赖更新顺序；
- 强耦合接触下可能振荡或变慢。

所以它属于一种工程上很实用、但理论上未必最理想的方法。

---

## 14. 摩擦建模

如果只考虑法向接触，物体只能“顶住”，不能阻碍切向滑动。

于是需要摩擦。

---

### 14.1 接触局部坐标下的力与速度分解

在局部接触系里，把接触力与速度分解为：

- 切向：$\boldsymbol{\lambda}_t, \mathbf{v}_t$
- 法向：$\lambda_n, v_n$

这样就可以把摩擦写成标准的库仑摩擦形式。

---

### 14.2 三种物理情况

课件给出三种情况：

#### 1. 无接触

$
\boldsymbol{\lambda}_t = 0, \quad \lambda_n = 0, \quad v_n \ge 0
$

没有接触，自然没有摩擦。

#### 2. 接触但静摩擦（static friction）

$
\|\boldsymbol{\lambda}_t\| \le \mu \lambda_n,
\quad \lambda_n > 0,
\quad \mathbf{v}_t = 0,
\quad v_n = 0
$

含义：

- 接触点贴住地面；
- 切向相对速度为 0；
- 摩擦力大小尚未达到极限，只要不超过 $\mu\lambda_n$ 就能“锁住”。

#### 3. 接触且滑动（sliding friction）

$
\|\boldsymbol{\lambda}_t\| = \mu \lambda_n,
\quad \lambda_n > 0,
\quad v_n = 0
$

并且摩擦方向与滑动方向相反。

这正是经典库仑摩擦模型：

> 静摩擦负责“尽量不动”；一旦必须滑，就以最大摩擦圆锥边界的大小反向阻碍运动。

---

### 14.3 Boxed LCP 近似

真正的库仑摩擦会形成一个圆锥约束，数值上较难处理。

课件中将其近似成 boxed LCP：

对某个切向分量，例如 $\lambda_x$：

- 静摩擦时：
  $
  -\mu\lambda_n \le \lambda_x \le \mu\lambda_n, \quad v_x = 0
  $
- 滑动时可能卡在上下边界：
  $
  \lambda_x = -\mu\lambda_n, \quad v_x \ge 0
  $
  或
  $
  \lambda_x = \mu\lambda_n, \quad v_x \le 0
  $

这种做法本质上是：

- 用盒子替代圆锥；
- 牺牲一部分物理精度；
- 换取更简单的数值求解。

这是很多实时物理引擎里常见的工程近似。

---

## 15. 两份课件串起来后的整体算法框架

现在可以把两讲的内容合在一起，看完整刚体/铰接体仿真一帧是怎么跑的。

---

### 15.1 不带接触的 articulated-body time stepping

1. 从所有 link 收集：
   - 位置 $\mathbf{x}, \mathbf{R}$
   - 速度 $\mathbf{v}, \boldsymbol{\omega}$
2. 组装所有 link 的广义质量矩阵 $\mathbf{M}$ 与右端项 $\mathbf{b}$
3. 遍历所有关节，计算：
   - 约束值 $\phi$
   - Jacobian $\mathbf{J}$
4. 解 compliance constraint 线性系统，得到新速度 $\dot{\mathbf{q}}^{n+1}$
5. 用新速度更新各个刚体的：
   - 位置 $\mathbf{x}^{n+1}$
   - 姿态 $\mathbf{R}^{n+1}$

---

### 15.2 带接触的 time stepping

在上面流程中再插入接触模块：

1. 收集各刚体状态；
2. 组装 $\mathbf{M}$ 和 $\mathbf{b}$；
3. 组装关节约束 $\phi, \mathbf{J}$；
4. 进行碰撞检测，找到所有接触点；
5. 为每个接触点建立局部接触坐标系，组装接触 Jacobian $\mathbf{J}^*$；
6. 用 BGS 等方法求解接触 LCP / boxed LCP；
7. 得到所有 link 的新速度；
8. 更新位置和旋转。

所以整个系统里存在两类约束：

- **关节约束**：通常是 equality-like 的几何约束；
- **接触约束**：通常是 complementarity 的不等式约束。

前者更偏“结构连接”，后者更偏“环境交互”。

---

## 16. 这两讲最值得真正理解的几个核心概念

---

### 16.1 刚体不是很多粒子分别算，而是“质心 + 朝向”

这是刚体仿真的第一性原理。

你并不需要追踪每个粒子的自由度，只需追踪：

- $\mathbf{x}(t)$
- $\mathbf{R}(t)$

其他点的位置速度都能从它们推出来。

---

### 16.2 Jacobian 是整个课程的桥梁

无论是：

- 关节约束
- 接触点速度
- 约束力回传
- 摩擦建模

本质上都离不开 Jacobian：

- $\mathbf{J}\dot{\mathbf{q}}$：把 generalized velocity 映射到某个点/约束空间；
- $\mathbf{J}^T\lambda$：把约束空间里的力映射回 generalized force。

所以可以把 Jacobian 理解成：

> **连接“系统自由度”和“局部几何行为”的接口。**

---

### 16.3 约束不只是“限制位置”，而是通过力影响动力学

从初学者视角，最容易误解的一点是：

- 以为约束就是“直接把物体拉回合法位置”

但在动力学仿真里，更自然的做法通常是：

- 约束先决定一个力/冲量；
- 再由这个力/冲量改变速度；
- 最终影响位置。

这就是为什么会有：

$
\mathbf{J}^T\lambda
$

这样的项直接进入动力学方程。

---

### 16.4 接触的本质是“有力时不离开，离开时无力”

这就是互补条件：

$
0 \le \lambda_n \perp v_n \ge 0
$

这一句几乎可以看作接触动力学的灵魂公式。

它同时编码了：

- 不可穿透；
- 不可吸引；
- 接触与分离之间的切换逻辑。

---

### 16.5 数值仿真里很多困难不是物理本身，而是“可解性”

这两讲反复在处理这个问题：

- 刚性太强怎么办？→ compliance
- 接触 case 太多怎么办？→ BGS
- 摩擦圆锥太难怎么办？→ boxed approximation

这说明真实仿真中常常不是“公式不会写”，而是：

> **公式虽然有了，但要把它写成稳定、可算、速度可接受的数值算法。**

---

## 17. 复习时建议重点掌握的公式清单

### 17.1 刚体点的位置、速度、加速度

$
\mathbf{x}_i = \mathbf{x} + \mathbf{R}\mathbf{b}_i
$

$
\mathbf{v}_i = \mathbf{v} + \boldsymbol{\omega}\times (\mathbf{x}_i - \mathbf{x})
$

$
\mathbf{a}_i = \mathbf{a} + \dot{\boldsymbol{\omega}}\times (\mathbf{x}_i - \mathbf{x}) + \boldsymbol{\omega}\times \big(\boldsymbol{\omega}\times (\mathbf{x}_i - \mathbf{x})\big)
$

### 17.2 旋转矩阵导数

$
\dot{\mathbf{R}} = [\boldsymbol{\omega}]\mathbf{R}
$

### 17.3 Newton–Euler 方程

$
\sum_i \mathbf{f}_i = m\mathbf{a}
$

$
\sum_i (\mathbf{x}_i - \mathbf{x})\times \mathbf{f}_i = \mathbf{I}\dot{\boldsymbol{\omega}} + \boldsymbol{\omega}\times \mathbf{I}\boldsymbol{\omega}
$

### 17.4 世界系惯性张量

$
\mathbf{I} = \mathbf{R}\mathbf{I}_b\mathbf{R}^T
$

### 17.5 柔顺约束势能

$
V = \frac{1}{2}k\,\phi(\mathbf{q})\cdot \phi(\mathbf{q})
$

$
\lambda = k\phi(\mathbf{q})
$

### 17.6 3D 摆的 Jacobian

$
\phi(\mathbf{q}) = \mathbf{x} + \mathbf{R}\mathbf{b} - \mathbf{c}
$

$
\mathbf{J} = [\mathbf{I}, -[\mathbf{R}\mathbf{b}]]
$

### 17.7 Compliance constraint 的离散线性系统

$
\begin{bmatrix}
\mathbf{M} & \mathbf{J}^T \\
\mathbf{J} & -\frac{c}{h^2}\mathbf{I}
\end{bmatrix}
\begin{bmatrix}
\dot{\mathbf{q}}^{n+1} \\
\mu
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{b} \\
-\frac{\phi}{h}
\end{bmatrix}
$

### 17.8 无摩擦接触互补条件

$
0 \le \lambda_n \perp v_n \ge 0
$

### 17.9 静摩擦与动摩擦

静摩擦：

$
\|\boldsymbol{\lambda}_t\| \le \mu \lambda_n,
\quad \mathbf{v}_t = 0
$

滑动摩擦：

$
\|\boldsymbol{\lambda}_t\| = \mu \lambda_n,
\quad \text{方向与 } \mathbf{v}_t \text{ 相反}
$

---

## 18. 一段适合放在博客开头/结尾的总结

刚体仿真的核心，在于把物理、几何和数值计算统一起来：

- **物理** 决定了受力如何产生平动与转动；
- **几何** 决定了关节、接触、法线、局部坐标系与 Jacobian；
- **数值方法** 决定了这些方程如何在离散时间步中稳定求解。

从单个刚体到多刚体系统，再到碰撞、接触与摩擦，本质上都是在不断回答同一个问题：

> 在满足几何约束与物理规律的前提下，下一时刻的速度和位置应该是什么？

而这两讲给出的答案框架是：

1. 用 $(\mathbf{x}, \mathbf{R}, \mathbf{v}, \boldsymbol{\omega})$ 表示刚体状态；
2. 用 Newton–Euler 建立动力学方程；
3. 用 Jacobian 表示关节与接触对速度/力的映射；
4. 用 compliance constraint 处理几何约束；
5. 用 LCP/boxed LCP 处理接触与摩擦；
6. 用时间步进方法把一切串成仿真器。

如果后面继续学习：

- penalty method
- impulse-based contact
- reduced coordinates
- articulated-body algorithm
- XPBD / projective dynamics
- differentiable simulation

那么这两讲里的大部分概念都会反复出现，只是形式会越来越系统化。

---

## 19. 可继续补充的博客方向

如果之后你还要把这一章扩展成更完整的博客系列，可以继续加：

1. **单独写一篇：SO(3)、叉乘矩阵与角速度的关系**
2. **单独写一篇：Newton–Euler 方程详细推导**
3. **单独写一篇：Jacobian 为什么既能传速度又能传力**
4. **单独写一篇：LCP 与互补约束的直观解释**
5. **单独写一篇：静摩擦、动摩擦与摩擦圆锥近似**
6. **单独写一篇：maximal coordinates vs minimal coordinates**

这样整套 PBS 笔记会非常完整。
