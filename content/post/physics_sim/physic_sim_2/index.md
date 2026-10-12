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

故:

$
\dot{\mathbf{R}}\mathbf{R}^T = -(\dot{\mathbf{R}}\mathbf{R}^T)^T
$

因此 $\dot{\mathbf{R}}\mathbf{R}^T$ 是一个**反对称矩阵**。这里可以稍微展开一下：

三维反对称矩阵 $\mathbf{A}$ 满足 $\mathbf{A}^T=-\mathbf{A}$，所以它的对角线元素一定为 $0$，而非对角线元素成相反数。也就是说，它一定可以写成

$
\mathbf{A}=
\begin{bmatrix}
0 & -a_3 & a_2\\
a_3 & 0 & -a_1\\
-a_2 & a_1 & 0
\end{bmatrix}
$

这样的形式。这个矩阵实际上只由 $a_1,a_2,a_3$ 三个独立参数决定，所以我们自然可以把它们收集成一个向量

$
\mathbf{a}=
\begin{bmatrix}
a_1\\
a_2\\
a_3
\end{bmatrix}
$

并记

$
[\mathbf{a}] =
\begin{bmatrix}
0 & -a_3 & a_2\\
a_3 & 0 & -a_1\\
-a_2 & a_1 & 0
\end{bmatrix}
$

这个记号叫做向量 $\mathbf{a}$ 的**叉乘矩阵**（也常叫 hat map）。它的意义是：对任意向量 $\mathbf{u}$，矩阵乘法 $[\mathbf{a}]\mathbf{u}$ 的结果，恰好等于向量叉乘 $\mathbf{a}\times\mathbf{u}$。

因此，任意三维反对称矩阵都可以唯一对应某个向量。在这里，我们就把

$
\dot{\mathbf{R}}\mathbf{R}^T = [\boldsymbol{\omega}]
$

其中 $\boldsymbol{\omega}$ 就是角速度向量，于是有：

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

这个式子也常写成

$
\boldsymbol{\tau} = \mathbf{I}\dot{\boldsymbol{\omega}} + \boldsymbol{\omega}\times (\mathbf{I}\boldsymbol{\omega})
$

下面补一下它的推导。

先从最基本的角动量定理出发。在惯性系里，

$
\boldsymbol{\tau} = \frac{d\mathbf{L}}{dt}
$

其中 $\mathbf{L}$ 是刚体关于质心的角动量。

对于刚体，如果我们在**刚体坐标系**下描述它，那么惯性张量 $\mathbf{I}$ 是常数，因此

$
\mathbf{L} = \mathbf{I}\boldsymbol{\omega}
$

这一步要注意：这个式子最方便的理解方式是在物体系里，因为物体系跟着刚体一起转，此时各质点坐标不变，所以 $\mathbf{I}$ 不随时间变化。

但角动量定理里的 $\dfrac{d\mathbf{L}}{dt}$ 是在**惯性系**里求导，而我们刚才写 $\mathbf{L}=\mathbf{I}\boldsymbol{\omega}$ 又更适合在**物体系**里看，所以先要搞清楚一个更基础的问题：

> 同一个几何向量，用物体系坐标表示和用世界系坐标表示时，它们的时间导数之间是什么关系？

设某个几何向量在世界系和物体系中的坐标分别为 $\mathbf{a}^{world}$ 和 $\mathbf{a}^{body}$，则

$
\mathbf{a}^{world} = \mathbf{R}\mathbf{a}^{body}
$

这里的 $\mathbf{R}$ 表示刚体姿态从物体系到世界系的变换。它之所以属于 $SO(3)$，不是额外假设，而是因为刚体运动必须保持长度和夹角不变，所以对任意向量 $\mathbf{u},\mathbf{v}$，都有

$
(\mathbf{R}\mathbf{u})\cdot(\mathbf{R}\mathbf{v}) = \mathbf{u}\cdot\mathbf{v}
$

因此 $\mathbf{R}^T\mathbf{R}=\mathbf{I}$。再加上这里描述的是纯旋转而不是镜像翻转，所以 $\det(\mathbf{R})=1$，于是 $\mathbf{R}\in SO(3)$。这也就是为什么前面 3.2 节可以从 $\mathbf{R}\in SO(3)$ 推出角速度矩阵。

现在对

$
\mathbf{a}^{world} = \mathbf{R}\mathbf{a}^{body}
$

两边求导：

$
\frac{d\mathbf{a}^{world}}{dt}=
\dot{\mathbf{R}}\mathbf{a}^{body} + \mathbf{R}\frac{d\mathbf{a}^{body}}{dt}
$

再利用前面 3.2 节得到的

$
\dot{\mathbf{R}} = [\boldsymbol{\omega}]\mathbf{R}
$

可得

$
\dot{\mathbf{R}}\mathbf{a}^{body}=
[\boldsymbol{\omega}]\mathbf{R}\mathbf{a}^{body}=
[\boldsymbol{\omega}]\mathbf{a}^{world}=
\boldsymbol{\omega}\times \mathbf{a}^{world}
$

所以最原始、最不容易混淆的形式其实是

$
\frac{d\mathbf{a}^{world}}{dt}=
\boldsymbol{\omega}\times \mathbf{a}^{world}
+
\mathbf{R}\frac{d\mathbf{a}^{body}}{dt}
$

这时右边第二项里的 $\mathbf{R}$ **并没有消失**。它的作用是：把“在物体系里算出来的导数坐标”重新变回世界系坐标，好让右边两项都和左边一样，在同一个坐标系里相加。

很多教材会把它简写成

$
\left(\frac{d\mathbf{a}}{dt}\right)_{world}=
\left(\frac{d\mathbf{a}}{dt}\right)_{body}
+
\boldsymbol{\omega}\times \mathbf{a}
$

这只是记号上的简化。这里

$
\left(\frac{d\mathbf{a}}{dt}\right)_{body}
$

严格来说指的是“先在物体系里求导，再把结果看成同一个几何向量带回世界系后的表示”。所以这不是单纯的分量导数，而是一个已经做过坐标转换的几何量。也正因如此，初看时会让人误以为右边少了一个 $\mathbf{R}$。

如果想记得更稳，可以先记上面那个带 $\mathbf{R}$ 的版本；后面在概念熟了以后，再把它简写成教材常用形式。

顺便看一下它和 3.3 节的关系。3.3 节中，对刚体上固定点有

$
\mathbf{x}_i - \mathbf{x} = \mathbf{R}\mathbf{b}_i
$

其中 $\mathbf{b}_i$ 是该点在物体系中的固定坐标，所以

$
\frac{d\mathbf{b}_i}{dt} = 0
$

代入上面的带 $\mathbf{R}$ 版本，就得到

$
\frac{d}{dt}(\mathbf{x}_i - \mathbf{x})=
\boldsymbol{\omega}\times (\mathbf{x}_i - \mathbf{x})
+
\mathbf{R}\cdot 0=
\boldsymbol{\omega}\times (\mathbf{x}_i - \mathbf{x})
$

也就是

$
\mathbf{v}_i - \mathbf{v}=
\boldsymbol{\omega}\times (\mathbf{x}_i-\mathbf{x})
$

所以 3.3 节并不是额外的新内容，而正是这个求导公式在“该向量在物体系中固定不变”这一特殊情形下的直接应用。

把 $\mathbf{a}$ 换成角动量 $\mathbf{L}$，就有

$
\left(\frac{d\mathbf{L}}{dt}\right)_{world}=
\left(\frac{d\mathbf{L}}{dt}\right)_{body}
+
\boldsymbol{\omega}\times \mathbf{L}
$

又因为在物体系里 $\mathbf{I}$ 是常数，

$
\left(\frac{d\mathbf{L}}{dt}\right)_{body}=
\left(\frac{d}{dt}(\mathbf{I}\boldsymbol{\omega})\right)_{body}=
\mathbf{I}\dot{\boldsymbol{\omega}}
$

所以

$
\boldsymbol{\tau}=
\left(\frac{d\mathbf{L}}{dt}\right)_{world}=
\mathbf{I}\dot{\boldsymbol{\omega}} + \boldsymbol{\omega}\times \mathbf{L}=
\mathbf{I}\dot{\boldsymbol{\omega}} + \boldsymbol{\omega}\times (\mathbf{I}\boldsymbol{\omega})
$

这就得到了 Euler 方程。

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

可以把新增项

$
\boldsymbol{\omega}\times (\mathbf{I}\boldsymbol{\omega})
$

理解成：虽然在物体系里只看到 $\mathbf{I}\dot{\boldsymbol{\omega}}$，但由于整个坐标系本身也在以 $\boldsymbol{\omega}$ 转动，所以角动量向量在惯性系里还会额外“被带着转”，这就是这个叉乘项的来源。

所以三维刚体旋转一般比二维复杂得多。

---

### 4.3 惯性张量是什么？

课件中给出惯性张量：

$
\mathbf{I} \triangleq \sum_i -m_i [\mathbf{x}_i - \mathbf{x}]^2
$

这里本质上是惯性矩阵的标准表达，其中`[x_i-x]`是前面定义的矩阵表示。更重要的是理解它的意义：

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

这个式子最好也补一下推导，不然会像是直接背结论。

在物体系里，刚体上第 $i$ 个质点相对质心的位置记为 $\mathbf{b}_i$。变换到世界系后，

$
\mathbf{x}_i - \mathbf{x} = \mathbf{R}\mathbf{b}_i
$

而惯性张量的定义是

$
\mathbf{I} = \sum_i -m_i[\mathbf{x}_i-\mathbf{x}]^2
$

所以关键就在于，如何把 $[\mathbf{x}_i-\mathbf{x}]$ 改写成 $\mathbf{b}_i$ 的形式。

这里要用到叉乘矩阵在旋转下的一个性质：

$
[\mathbf{R}\mathbf{u}] = \mathbf{R}[\mathbf{u}]\mathbf{R}^T
$

为什么这条式子成立？因为对任意向量 $\mathbf{v}$，

$
[\mathbf{R}\mathbf{u}]\mathbf{v}=
(\mathbf{R}\mathbf{u})\times \mathbf{v}
$

另一方面，由于旋转矩阵会保持叉乘结构，

$
(\mathbf{R}\mathbf{u})\times (\mathbf{R}\mathbf{w})=
\mathbf{R}(\mathbf{u}\times \mathbf{w})
$

令 $\mathbf{w}=\mathbf{R}^T\mathbf{v}$，则 $\mathbf{v}=\mathbf{R}\mathbf{w}$，于是

$
(\mathbf{R}\mathbf{u})\times \mathbf{v}=
(\mathbf{R}\mathbf{u})\times (\mathbf{R}\mathbf{w})=
\mathbf{R}(\mathbf{u}\times \mathbf{w})=
\mathbf{R}[\mathbf{u}]\mathbf{w}=
\mathbf{R}[\mathbf{u}]\mathbf{R}^T\mathbf{v}
$

由于这对任意 $\mathbf{v}$ 都成立，所以

$
[\mathbf{R}\mathbf{u}] = \mathbf{R}[\mathbf{u}]\mathbf{R}^T
$

把 $\mathbf{u}$ 换成 $\mathbf{b}_i$，就得到

$
[\mathbf{x}_i-\mathbf{x}]=
[\mathbf{R}\mathbf{b}_i]=
\mathbf{R}[\mathbf{b}_i]\mathbf{R}^T
$

于是

$
[\mathbf{x}_i-\mathbf{x}]^2=
(\mathbf{R}[\mathbf{b}_i]\mathbf{R}^T)(\mathbf{R}[\mathbf{b}_i]\mathbf{R}^T)=
\mathbf{R}[\mathbf{b}_i]^2\mathbf{R}^T
$

代回惯性张量定义：

$
\mathbf{I}=
\sum_i -m_i[\mathbf{x}_i-\mathbf{x}]^2=
\sum_i -m_i\mathbf{R}[\mathbf{b}_i]^2\mathbf{R}^T=
\mathbf{R}\left(\sum_i -m_i[\mathbf{b}_i]^2\right)\mathbf{R}^T
$

而括号里的东西正是物体系惯性张量 $\mathbf{I}_b$，因此

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
7. 进入下一时间步。

换句话说，刚体仿真本质上就是：

> **用数值积分器求解刚体状态随时间变化的 ODE。**

这里理论上可以套用前面课程里提到的：

- Forward Euler
- RK2
- RK4
- 以及更多高阶方法

但在实际刚体仿真里，仅仅“会积分”还不够，因为还有约束、碰撞、接触、摩擦等问题。

---

## 6. Articulated Bodies
### 6.1 从单刚体到多刚体

前面课程已经讲过：一个**单刚体**可以用它的 body frame 来描述，其状态通常写成

$
q_i=(x_i, R_i),
$

其中：

- $x_i$：第 $i$ 个刚体参考点的位置；
- $R_i$：第 $i$ 个刚体的朝向；
- 在 3D 中，一个自由刚体总共有 6 个自由度：3 个平移 + 3 个转动。

但现实中的机器人、机械臂、cart-pole、人体骨架都不是“单个自由刚体”，而是由多个刚体部件组成，并通过关节连接。课件把这种系统称为 **articulated body**。

---

### 6.2 Link 与 Joint

#### 6.2.1 Link

**Link** 指一个刚体部件。比如在 cart-pole 中：

- cart 是一个 link；
- pole 是另一个 link。

每个 link 单独看都可以当成一个自由刚体。

#### 6.2.2 Joint

**Joint** 指两个 link 之间的连接结构，它的作用不是“凭空赋予运动”，而是**限制相对运动**。

课件举了两个最常见的例子：

- **Prismatic joint（移动副）**：只允许沿某一根轴平移；
- **Revolute joint（转动副）**：只允许绕某一根轴旋转。

所以理解关节最好的方式就是：

> 先假设两个刚体彼此完全自由，
> 再由 joint 去“删掉”不允许的相对自由度。

---

### 6.3 自由度（DoF）的角度看关节

两个 3D 刚体之间的相对运动，如果完全自由，一共有 6 个自由度：

- 3 个相对平移自由度；
- 3 个相对转动自由度。

关节本质上就是“保留部分自由度，去掉其余自由度”。

#### 6.3.1 常见关节对应的自由度

- **Revolute joint**：保留绕一根轴的旋转 → 1 DoF
- **Prismatic joint**：保留沿一根轴的平移 → 1 DoF
- **Universal joint**：保留绕两根轴的旋转 → 2 DoF

因此，关节约束的数量通常等于：

$
\text{约束数} = 6 - \text{允许的相对自由度数}.
$

例如：

- revolute joint：约束 5 个相对自由度；
- prismatic joint：也约束 5 个相对自由度。

---

### 6.4 为什么要引入 Joint Frames

课件在第 32–34 页提出：为了更方便地写关节约束，不直接只盯着刚体整体位姿，而是**在关节两侧各建立一个局部坐标系（joint frame）**。

设：

- 父刚体（parent link）上有一个 joint frame；
- 子刚体（child link）上也有一个 joint frame。

那么关节约束就可以理解为：

> 这两个 joint frame 在世界坐标里必须满足某种相对关系。

这种做法的好处是：

1. **统一**：不同关节都可以转化为“两个 frame 如何对齐”的问题；
2. **局部**：关节是局部结构，用局部 frame 表达最自然；
3. **模块化**：换一个关节类型，只需要改这两个 frame 之间的相对关系。

---

### 6.5 Pendulum 例子：约束是怎么来的

课件用一个 pendulum（摆）说明 joint frame 的思想。摆杆如果被当作一个 2D 刚体，它的状态可写成

$
q=(x,y,\theta).
$

如果它完全自由，那么 $(x,y,\theta)$ 可以任意取值；但实际上摆杆的一端被铰接在世界中的一个固定点上，因此这些变量不是独立的。

设：

- 刚体局部坐标系中，连接点位置为 $b$；
- 世界坐标中，支点位置为 $c$。

则该连接点在世界中的位置为

$
x + R(\theta)b.
$

由于这个点必须和支点重合，所以必须满足

$
\phi(q)=x+R(\theta)b-c=0.
$

这就是一个标准的 **holonomic constraint**：它直接把构型变量 $q$ 约束住。

#### 6.5.1 这个式子的物理意义

它不是在说“这个点速度要怎样”，而是在说：

- **任何时刻**，摆杆上的那个连接点都必须在支点位置；
- 因而 $x,y,\theta$ 三者之间必须满足这个几何关系。

如果把它展开到二维，通常会得到两个标量约束：

$
\begin{cases}
x + (R(\theta)b)_x - c_x = 0, \\
y + (R(\theta)b)_y - c_y = 0.
\end{cases}
$

这样原本二维自由刚体的 3 个自由度就被约束掉 2 个，只剩 1 个自由度，也就是摆角。

---

### 6.6 Prismatic Joint：具体公式怎么写
#### 6.6.1 几何含义

Prismatic joint 只允许子刚体相对父刚体：

- 沿某一根指定轴平移；
- 不允许横向偏移；
- 不允许相对转动。

#### 6.6.2 2D 情况下的推导

先考虑二维。设父 joint frame 的坐标轴为 $(e_1,e_2)$，其中 $e_1$ 是允许滑动的方向。

设子 frame 原点相对父 frame 原点的位置向量为

$
r = p_c - p_p.
$

如果这是一个 prismatic joint，那么：

1. 允许沿 $e_1$ 方向移动；
2. 不允许沿 $e_2$ 方向移动；
3. 子 frame 与父 frame 的朝向必须一致。

于是约束可写成：

#### （1）横向位移为 0

$
e_2^T (p_c-p_p)=0.
$

#### （2）相对角度为 0

二维里设父、子刚体朝向分别为 $\theta_p, \theta_c$，则

$
\theta_c-\theta_p=0.
$

这两个约束总共去掉了 2 个自由度，而二维两个刚体的相对运动原本有 3 个自由度，因此还剩 1 个自由度，也就是沿滑轨方向的平移。

#### 6.6.3 3D 情况下的理解

3D 里两个刚体相对运动有 6 个自由度，而 prismatic joint 只保留 1 个平移自由度，所以需要施加 5 个约束：

- 2 个横向平移约束；
- 3 个相对转动约束。

也就是说：

- 子 frame 原点必须落在父 frame 的滑动轴上；
- 两个 frame 的方向必须对齐。

---

### 6.7 Maximal Coordinates：为什么“先自由再约束”

第 36 页给出了本节最核心的建模方式：**maximal coordinates**。课件写道：

- 每个 link 分配一个 body frame，记为 $q_i=(x_i,R_i)$；
- 对每个 joint，在父子两侧建立 joint frames；
- 对每个关节写一个 holonomic constraint:
  $
  \phi_i(q_p,q_c)=0.
  $

#### 6.7.1 什么叫 maximal

所谓 maximal，不是说表示最省变量，而是说：

> 先给每个刚体都保留完整的自由刚体变量，
> 再通过约束去掉不允许的运动。

例如 3D 中，一个 link 的状态可以写成

$
q_i=(x_i,R_i).
$

如果有 $n$ 个 link，就会有 $n$ 套这样的变量。它们在一开始都被当成“自由刚体”，这就是“最大坐标”的含义。

#### 6.7.2 为什么要这么做

这样做的优点是：

1. **统一**：所有 link 都用同一种变量表示；
2. **灵活**：关节只需额外添加约束；
3. **适合碰撞/接触**：因为每个刚体都有独立位姿，做碰撞检测很自然。

缺点是：

- 变量多；
- 需要额外求解约束力或拉格朗日乘子；
- 数值处理更复杂。

这也是后续课程为什么要讲 compliance constraints 和带约束时间步进。

---

### 6.8 Holonomic Constraint 到底是什么

#### 6.8.1 定义

**Holonomic constraint（完整约束）** 指的是：

> 可以直接写成“构型变量之间的显式方程”的约束。

标准形式是

$
\phi(q,t)=0,
$

若不显含时间，则写作

$
\phi(q)=0.
$

其中 $q$ 是系统的配置变量，如位置、角度、旋转等。

#### 6.8.2 在本节里的含义

本节所有关节约束本质上都属于 holonomic constraint，因为它们都是在说：

- 某些点必须重合；
- 某些轴必须对齐；
- 某些相对位姿必须满足固定关系。

这些都能直接写成 $q$ 的函数等于 0。

例如：

- 摆的支点重合：
  $
  x+Rb-c=0;
  $
- 一般关节：
  $
  \phi_i(q_p,q_c)=0.
  $

#### 6.8.3 与 nonholonomic constraint 的区别

Holonomic constraint 限制的是**位置/构型**；

Nonholonomic constraint 往往限制的是**速度**，典型形式是

$
A(q)\dot q = 0,
$

它不一定能积分成某个 $\phi(q)=0$ 的位置约束。

例如“小车不能侧滑”通常属于 nonholonomic constraint，因为它限制的是速度方向，而不是简单的构型方程。

---

### 6.9 Maximal Coordinates 与 Reduced Coordinates 的对比

这一节课采用的是 maximal coordinates，但你最好顺便知道它和 reduced coordinates 的区别。

#### 6.9.1 Reduced coordinates

Reduced coordinates 的思想是：

> 不给每个 link 都保留完整位姿，
> 而是直接用系统真正的独立自由度来表示。

例如双摆可以直接用

$
q=(\theta_1,\theta_2)
$

来描述，而不需要为每根杆都写一个 $(x,R)$。

#### 6.9.2 两者对比

##### Maximal coordinates

- 每个刚体都有完整位姿变量；
- 关节作为约束额外加入；
- 变量更多，但建模统一；
- 对接触和碰撞更方便。

##### Reduced coordinates

- 只保留独立自由度；
- 自动满足关节约束；
- 变量更少，效率可能更高；
- 但复杂拓扑、碰撞和关节切换时往往更麻烦。

#### 6.9.3 为什么这门课先讲 maximal coordinates

因为它更直接反映“物理系统 = 自由刚体 + 约束”的本质，也更便于后续统一处理：

- joint constraints
- compliance constraints
- contacts
- collisions

---

#### 6.10 这一节的最终总结

这一节内容可以压缩成一句话：

> **多刚体系统 = 多个自由刚体 + 关节几何约束。**

更具体地说：

1. 每个 link 都先被当成一个自由刚体，状态为 $q_i=(x_i,R_i)$；
2. 每个 joint 在父、子刚体上各定义一个 joint frame；
3. 要求这两个 joint frame 满足某种几何关系；
4. 这些几何关系写成
   $
   \phi_i(q_p,q_c)=0,
   $
   它们就是 **holonomic constraints**；
5. 这样就把 articulated body 统一写成了一个带约束的多刚体系统。

如果后面继续往动力学走，那么这些 $\phi_i=0$ 还会进一步通过 Jacobian 和拉格朗日乘子，转化成约束力进入时间步进算法中。

---

## 7. Compliance Constraints：从几何约束到约束力

### 7.1 为什么要从“几何约束”走向“力学约束”

上一章已经讲过：在 maximal coordinates 里，每个 link 都先被当成自由刚体，然后每个关节都写成一个 holonomic constraint：

$
\phi(q_p,q_c)=0.
$

这个式子本质上只是一个**几何关系**，它告诉我们：

- 哪些位置必须重合
- 哪些方向必须对齐
- 哪些相对运动被禁止

但是如果我们要真正做动力学仿真，只知道“它应该满足约束”还不够。  
我们还必须回答：

> 当刚体偏离这个约束时，系统应该产生什么力，把它拉回去？

所以这一章要解决的问题是：

> **如何从关节约束 $\phi(q)=0$ 推出约束力？**

---

### 7.2 从硬约束到 Compliance Constraint

课件第 37–38 页提出了一种很重要的思路：  
先不要把 $\phi(q)=0$ 当成“绝对刚性”的硬约束，而是把它看成一个**弹簧能量最小化问题**。

也就是说，我们不直接说：

$
\phi(q)=0
$

而是定义一个势能：

$
V(q)=\frac12 k\,\phi(q)\cdot\phi(q),
$

其中 $k>0$ 是一个很大的刚度参数。

---

#### 7.2.1 这个势能是什么意思

这个式子的物理直觉和一维弹簧完全一样：

- 如果 $\phi(q)=0$，说明约束完全满足，势能最小，为 0；
- 如果 $\phi(q)\neq 0$，说明系统偏离了理想约束，势能增加；
- 偏离越大，势能越大；
- 因此系统会产生“恢复力”，把它往 $\phi=0$ 拉回去。

所以你可以把 compliance constraint 理解成：

> **用一个“很硬的弹簧”去近似理想关节约束。**

---

#### 7.2.2 为什么不直接用硬约束

因为直接解硬约束通常需要：

- 拉格朗日乘子
- 精确保持 $\phi(q)=0$
- 更复杂的带约束动力学系统

而把它改成势能后，约束就变成了一种普通的内力来源：

- 和弹簧力一样
- 可以直接并入动力学方程
- 实现更直接，概念上也更顺

当然，这只是一个近似；当 $k$ 足够大时，它会逼近硬约束。

---

### 补充：微分 $d$ 与变分 $\delta$ 的区别

在力学、变分法和连续体推导里，经常同时看到 $d$ 和 $\delta$，它们形式相似，但语境不同。

#### 1. $d$：微分，强调变量本身的微小变化
例如
$
df = \frac{df}{dx}dx
$
这里的 $dx$ 表示自变量发生了一个微小改变量。  
如果变量随时间演化，还常写成
$
dx = \dot x\,dt
$
因此，$d$ 更常用于：

- 普通微积分
- 全微分、链式法则
- 时间演化中的真实小变化

---

#### 2. $\delta$：变分，强调“试探性扰动”
例如
$
\delta V = \nabla V \cdot \delta x
$
这里的 $\delta x$ 不一定表示系统真的会这样运动，而是：

> 在当前状态附近，人为施加一个无穷小的允许扰动，研究量的一阶变化。

因此，$\delta$ 更常用于：

- 虚位移（virtual displacement）
- 虚功原理
- 势能变分
- 变分法与泛函分析
- 对矩阵、函数、轨迹做扰动分析

---

#### 3. 一个直观例子

若粒子当前真实速度沿 $x$ 方向，则经过小时间 $dt$ 的真实位移是

$
dx = \dot x\,dt
$

这属于微分语境。

但如果我们为了分析系统，额外取一个 $y$ 方向的小扰动

$
\delta x = (0,\varepsilon)
$

哪怕系统当前并不会真的往 $y$ 方向运动，这仍然是一个合法的“试探性扰动”，属于变分语境。

---

#### 4. 一句话总结

可以把它简单记成：

- $d$：变量真的改了一点时的微分
- $\delta$：人为选一个无穷小扰动时的变分

更严格地说：

- $d$ 更强调“实际变量变化”
- $\delta$ 更强调“假想扰动 / 虚位移 / 变分分析”

---

#### 5. 为什么虚功原理里用 $\delta$

虚功原理关心的是：

> 对任意允许的虚位移 $\delta x$，系统的总虚功如何变化

这里的 $\delta x$ 不是系统真实随时间前进得到的位移，而是一个用于测试平衡、约束和力学关系的“试探方向”。  
因此在虚功、势能驻值、约束推导里，通常写 $\delta$ 而不是 $d$。

### 7.3 势能如何产生力：虚功观点

课件第 39 页开始进入关键推导：  
已知势能 $V(q)$，怎样求出它对应的约束力？

这里使用的是 **虚功（virtual work）** 的思想。

设刚体发生一个虚位移 $\delta q$，则势能变化 $\delta V$ 应满足：

$
\delta V \equiv - f \cdot \delta q.
$

这个式子非常重要，它说的是：

> 保守力做的虚功，等于势能变化的负号。

一维里大家熟悉：

$
f=-\frac{dV}{dx}.
$

这里则是在刚体位姿空间里的更一般版本。

---

## 7.4 先推导平动力：线性约束力怎么来

课件第 40 页先只考虑平移虚位移 $\delta x$。

---

### 7.4.1 势能写法

我们有

$
V(q)=\frac12 k\,\phi(q)\cdot\phi(q).
$

定义

$
\lambda := k\phi(q).
$

注意这里的 $\lambda$ 暂时不是拉格朗日乘子，而更像是“弹簧响应量”，只是记号上长得类似。

于是对 $x$ 的虚位移，势能变分可写成：

$
\delta V=
\left(\frac{\partial \phi}{\partial x}\right)^T \lambda \cdot \delta x.
$

课件把

$
J_v := \frac{\partial \phi}{\partial x}
$

看成线性部分的 Jacobian。于是：

$
\delta V = J_v^T\lambda \cdot \delta x.
$

根据虚功定义

$
\delta V = -f\cdot \delta x,
$

可知约束产生的线力为

$
f_{constraint} = -J_v^T\lambda.
$

---

### 7.4.2 直觉解释

这个式子和很多优化/机器人学里的 Jacobian transpose force 很像。

直观上：

- $\phi(q)$ 描述“约束误差”
- $\lambda=k\phi$ 描述“恢复强度”
- $J_v$ 描述“位置变化会如何改变约束误差”
- 所以 $J_v^T\lambda$ 就把“约束空间的误差力”映射回了世界中的实际力

因此

$
-J_v^T\lambda
$

就是作用在刚体平动上的约束恢复力。

---

## 7.5 再推导转动力矩：为什么会出现 $J_\omega$

只考虑平移还不够，因为刚体还有旋转。  
课件第 41–44 页就是在推导：  
当刚体发生一个微小旋转时，势能会如何变化，从而得到约束力矩。

---

### 7.5.1 刚体旋转的虚位移怎么表示

课件采用：

$
\delta R = \delta r \times R
$

这里：

- $\delta r\in \mathbb{R}^3$ 是一个无穷小旋转向量
- $\delta r\times R$ 表示由该小旋转引起的旋转矩阵变化

这和前面讲刚体运动学时

$
\dot R = \omega \times R
$

是同一种思想：  
角速度或小旋转向量都可以通过叉乘矩阵来作用在旋转矩阵上。

---

### 7.5.2 势能对旋转的变分

课件把与旋转有关的那部分整理成。(这里因为 $R$ 是矩阵，所以“标量对矩阵求导”后面要配双点积 `:`)

$
\delta V=
\sum_i \left(\frac{\partial \phi_i}{\partial R}\lambda_i\right) : \delta R= 
\sum_i \left(\frac{\partial \phi_i}{\partial R}\lambda_i\right) :([\delta r]R)
$

这里课件中间其实做了一个重要变形：先把右边的 $R$ 挪到左边，方便后面把式子改写成“某个向量与 $\delta r$ 的内积”。

要用到的恒等式是

$
A:(BC) = (AC^T):B
$

令

$
A=\frac{\partial \phi_i}{\partial R}\lambda_i,\qquad
B=[\delta r],\qquad
C=R
$

则

$
\left(\frac{\partial \phi_i}{\partial R}\lambda_i\right):([\delta r]R)=
\left(\left(\frac{\partial \phi_i}{\partial R}\lambda_i\right)R^T\right):[\delta r]
$

所以旋转部分的势能变分可以先写成

$
\delta V_{rot}=
\sum_i \left(\left(\frac{\partial \phi_i}{\partial R}\lambda_i\right)R^T\right):[\delta r]
$

接下来再把“矩阵和叉乘矩阵的双点积”变成普通向量内积。也就是说，后面真正要做的是把每一项都改写成:

$
\text{某个 3D 向量}\cdot \delta r
$

---

## 7.6 为什么课件要单独推导那个矩阵恒等式

课件第 42–43 页中间插入的，其实不是一个可有可无的小技巧，而是把 7.5.2 最后一行公式继续往下推的关键。

前面我们已经得到了

$
\delta V_{rot}=
\sum_i \left(\left(\frac{\partial \phi_i}{\partial R}\lambda_i\right)R^T\right):[\delta r]
$

现在的问题是：右边还是“矩阵 : 矩阵”的形式，但我们最终想要的是像

$
\delta V_{rot} = (\text{某个力矩向量})\cdot \delta r
$

这样的普通向量内积。为此，课件才会单独推一个矩阵恒等式。

对任意矩阵 $A\in\mathbb{R}^{3\times 3}$ 和向量 $b\in\mathbb{R}^3$，有

$
A:[b]=
(A_{32}-A_{23},\,A_{13}-A_{31},\,A_{21}-A_{12})\cdot b
$

也就是说，只有 $A$ 的反对称部分会对 $A:[b]$ 有贡献。则上式也可写成

$
A:[b] = [A^T-A]\cdot b
$

这个恒等式可以直接验证。因为

$
[b]=
\begin{bmatrix}
0 & -b_3 & b_2\\
b_3 & 0 & -b_1\\
-b_2 & b_1 & 0
\end{bmatrix}
$

于是

$
\begin{aligned}
A:[b]
&= \sum_{m,n} A_{mn}[b]_{mn} \\
&= A_{12}(-b_3)+A_{13}b_2+A_{21}b_3+A_{23}(-b_1)+A_{31}(-b_2)+A_{32}b_1 \\
&= (A_{32}-A_{23})b_1+(A_{13}-A_{31})b_2+(A_{21}-A_{12})b_3
\end{aligned}
$

这正是上面的向量内积形式。

它的作用其实很明确：

> **把旋转矩阵的变分形式，转成一个普通 3D 向量的力矩形式。**

因为最终我们希望得到的是：

- 力 $f\in\mathbb{R}^3$
- 力矩 $\tau\in\mathbb{R}^3$

而不是某种抽象的矩阵变分对象。

现在把它套回 7.5.2。令

$
M_i :=
\left(\frac{\partial \phi_i}{\partial R}\lambda_i\right)R^T
$

则

$
M_i:[\delta r]=
(M_i^T-M_i)\cdot \delta r
$

因此

$
\delta V_{rot}=
\sum_i (M_i^T-M_i)\cdot \delta r
$

再把 $\lambda_i$ 提出来：

$
M_i^T-M_i=
\lambda_i\left(
R\left(\frac{\partial \phi_i}{\partial R}\right)^T-
\left(\frac{\partial \phi_i}{\partial R}\right)R^T
\right)
$

所以

$
\delta V_{rot}=
\sum_i
\lambda_i\,
\left(
R\left(\frac{\partial \phi_i}{\partial R}\right)^T-
\left(\frac{\partial \phi_i}{\partial R}\right)R^T
\right)\cdot \delta r
$

这说明对于每个标量约束 $\phi_i$，它对应的旋转 Jacobian 行向量可以定义为

$
(J_\omega)_i^T:=
\left(
R\left(\frac{\partial \phi_i}{\partial R}\right)^T-
\left(\frac{\partial \phi_i}{\partial R}\right)R^T
\right)
$

于是总的旋转势能变分就写成

$
\delta V_{rot}=
J_\omega^T\lambda \cdot \delta r
$

这才是 7.5.2 最后一行更准确的写法。前面写成 $: [\delta r]$ 只是推导过程中的中间形态，还没有真正化成力矩向量。

于是按照虚功定义

$
\delta V = -\tau \cdot \delta r,
$

就得到约束产生的力矩：

$
\tau_{constraint} = -J_\omega^T \lambda.
$

### 7.6.1 对比总结
和线性情况完全平行：
- $J_v$ 把平移变化映射到约束误差变化；
- $J_\omega$ 把旋转变化映射到约束误差变化；
- 所以它们的转置再乘上 $\lambda$，就得到约束对平动和转动施加的广义力。

于是，约束的广义力可以统一写成：

$
\text{generalized constraint force}=-
\begin{bmatrix}
J_v^T \\
J_\omega^T
\end{bmatrix}
\lambda.
$

---

## 7.7 约束力最终如何进入 Newton–Euler 方程

课件第 44 页给出本章最重要的结果之一。

原本自由刚体的 Newton–Euler 方程是：

$
f = ma,
\qquad
\tau = I\dot\omega + \omega \times I\omega.
$

加入约束以后，约束会额外施加：

- 线力 $-J_v^T\lambda$
- 力矩 $-J_\omega^T\lambda$

所以动力学方程变成：

$
f - J_v^T\lambda = ma,
$

$
\tau - J_\omega^T\lambda = I\dot\omega + \omega \times I\omega.
$

这两个式子非常关键，因为它告诉你：

> 几何约束并没有“神秘地消失”，而是作为额外的广义力进入了刚体动力学。

---

## 7.8 这一段背后的真正结构: ***核心步骤总结***

到这里，其实整章已经可以总结成一条清晰链路：

### 第一步：先有几何约束
$
\phi(q)=0
$

### 第二步：把它变成弹簧能量
$
V(q)=\frac12 k\,\phi(q)\cdot\phi(q)
$

### 第三步：定义响应量
$
\lambda = k\phi(q)
$

### 第四步：对位姿求变分
得到：
- 平移对应 $J_v$
- 旋转对应 $J_\omega$

### 第五步：通过虚功得到广义力
$
f_c = -J_v^T\lambda,\qquad
\tau_c = -J_\omega^T\lambda
$

### 第六步：把它们加进 Newton–Euler
$
f + f_c = ma,\qquad
\tau + \tau_c = I\dot\omega + \omega\times I\omega
$

这就是 compliance constraints 的完整逻辑。

---

## 7.9 3D Pendulum 例子：把抽象公式落地

课件第 45–51 页给了一个 3D pendulum 的具体例子。这里是整章最值得真正吃透的地方，因为前面的所有抽象符号都在这里变成具体几何量。

---

### 7.9.1 问题设置

课件考虑的约束是：

$
\phi(q)=x+Rb-c.
$

其中：

- $x$：刚体参考点在世界坐标中的位置
- $R$：刚体朝向
- $b$：连接点在刚体局部坐标中的位置
- $c$：世界中的固定支点

这个式子的意思是：

> 刚体上的局部点 $b$，经过当前刚体位姿变换后，必须与世界固定点 $c$ 重合。

如果 $\phi(q)=0$，就说明这个点正好固定在支点上，于是这个刚体就像一个摆一样绕这个点转动。

---

### 7.9.2 为什么这是一个 3 维约束

因为

$
\phi(q)\in\mathbb{R}^3.
$

也就是说：

- x 坐标要对齐
- y 坐标要对齐
- z 坐标要对齐

所以它其实是 3 个标量约束打包写成一个向量约束。

这很合理：  
3D 里“某个点必须固定在某处”就是三个方向都不能偏。

---

## 7.10 这个例子里的 $J_v$ 是什么

课件第 45 页直接给出：

$
J_v = \frac{\partial \phi}{\partial x} = I.
$

为什么这么简单？

因为

$
\phi(q)=x+Rb-c
$

对 $x$ 求导，就是单位矩阵(向量对向量求导)：

$
\frac{\partial \phi}{\partial x}=I.
$

直觉上也很好懂：

- 如果整体位置 $x$ 往某方向动一点
- 约束误差 $\phi$ 就完全同样地往那个方向变一点

所以线性 Jacobian 就是恒等映射。

---

## 7.11 这个例子里的 $J_\omega$ 为什么是 $-[Rb]$

课件第 46–49 页其实是在用更直观的方式推导这一点。

---

### 7.11.1 先把势能变分写成沿真实运动的一小步

前面 7.4 到 7.6 是从一般变分公式出发推导 $J_v,J_\omega$。到了这个具体例子，其实还可以更直观：直接看刚体在一个极小时间 $\delta t$ 内真实走了一小步时，势能如何变化。

仍然从

$
V(q)=\frac12 k\,\phi(q)\cdot\phi(q),\qquad
\lambda = k\phi(q)
$

出发。沿着系统当前运动方向前进一个极小时间 $\delta t$，有

$
\delta V=
\dot V\,\delta t=
k\phi\cdot\dot\phi\,\delta t=
\lambda\cdot\dot\phi\,\delta t
$

所以关键就变成：这个例子里的 $\dot\phi$ 是什么。

注意约束函数

$
\phi(q)=x+Rb-c
$

对时间求导：

$
\dot\phi = \dot x + \dot R b
$

又因为

$
\dot x = v,\qquad
\dot R = \omega \times R
$

所以

$
\dot\phi = v + \omega \times (Rb)
$

这其实就是连接点 $x+Rb$ 的世界速度。也就是说: 约束误差的变化率，正好等于“被约束那个点”此刻想要运动的速度。

---

### 7.11.2 再把它拆成平移项和转动项

把上面的 $\dot\phi$ 代回去：

$
\delta V=
\lambda\cdot\dot\phi\,\delta t=
\lambda\cdot v\,\delta t+
\lambda\cdot(\omega\times (Rb))\,\delta t
$

先看第一项。因为

$
\delta x = v\,\delta t
$

所以

$
\lambda\cdot v\,\delta t=
\lambda\cdot \delta x
$

这说明

$
J_v^T\lambda = \lambda
$

也就是

$
J_v = I
$

这和 7.10 的结果一致。

再看第二项。利用标量三重积恒等式

$
\lambda\cdot(\omega\times (Rb))=
((Rb)\times \lambda)\cdot \omega
$

而根据叉乘矩阵定义，

$
(Rb)\times \lambda = [Rb]\lambda
$

再记

$
\delta r = \omega\,\delta t
$

于是

$
\lambda\cdot(\omega\times (Rb))\,\delta t=
([Rb]\lambda)\cdot \delta r
$

综合起来，

$
\delta V=
\lambda\cdot \delta x+
([Rb]\lambda)\cdot \delta r
$

和虚功写法

$
\delta V = J_v^T\lambda\cdot \delta x + J_\omega^T\lambda\cdot \delta r
$

逐项比较，就得到

$
J_v^T\lambda = \lambda,\qquad
J_\omega^T\lambda = [Rb]\lambda
$

因此

$
J_v = I,\qquad
J_\omega^T = [Rb],\qquad
J_\omega = -[Rb]
$

最后一个等号利用了叉乘矩阵的反对称性：

$
[Rb]^T = -[Rb]
$

所以这个具体例子里的 Jacobian 可以写成

$
J=
\begin{bmatrix}
J_v & J_\omega
\end{bmatrix}=
\begin{bmatrix}
I & -[Rb]
\end{bmatrix}
$

这就是课件第 48–49 页的结论。

---

## 7.12 为什么 $J^T\lambda$ 恰好等于“力 + 力矩”

课件第 50 页继续说明：

$
J^T\lambda=
\begin{bmatrix}
\lambda\\
Rb\times \lambda
\end{bmatrix}.
$

这说明：

- 上半部分是施加在刚体上的线力 $\lambda$
- 下半部分是该力相对于刚体参考点产生的力矩 $Rb\times\lambda$

这和你在经典力学里的常识完全一致：

> 一个作用在点 $Rb$ 处的力 $\lambda$，  
> 对刚体参考点会产生力矩
> $\tau = (Rb)\times \lambda$

所以 $J^T\lambda$ 不只是一个抽象代数表达，它正是“点力诱导的广义力”。

---

## 7.13 最终检查：它真的是一个弹簧恢复力吗

课件第 51 页做了一个一致性检查。既然

$
\lambda = k\phi,
$

那约束施加的力就是

$
-J_v^T\lambda = -\lambda = -k\phi.
$

而约束施加的力矩就是

$
-J_\omega^T\lambda = Rb \times (-\lambda) = Rb\times(-k\phi).
$

这恰好和一个弹簧施加在连接点上的恢复力完全一致：  
如果连接点偏离支点，就有一个指向回去的弹簧力，并且该点力对刚体中心产生力矩。

这一步非常重要，因为它说明前面那套 Jacobian / 虚功推导并不是“形式上凑出来的”，而是和直觉完全一致。

---

## 7.14 这一章真正想让你掌握什么

如果不想陷在公式细节里，我建议你把本章记成下面这四句话。

### 1. 几何约束可以先软化成弹簧能量
$
V(q)=\frac12 k\phi(q)\cdot\phi(q)
$

### 2. 力来自势能的负梯度
平移给出线力，旋转给出力矩。

### 3. Jacobian 是“约束误差对位姿变化的敏感度”
- $J_v$：对平移的敏感度
- $J_\omega$：对旋转的敏感度

### 4. $J^T\lambda$ 就是广义约束力
它会自然进入 Newton–Euler 方程。

---

## 7.15 最终总结

本章完成了从**关节几何约束**到**动力学约束力**的转换。

具体来说：

1. 对关节约束
   $
   \phi(q)=0
   $
   不直接硬性求解，而是先定义 compliance energy
   $
   V(q)=\frac12 k\phi(q)\cdot\phi(q)
   $

2. 定义
   $
   \lambda = k\phi(q)
   $

3. 用虚功原理分别对平移与旋转求变分，得到
   $
   f_c=-J_v^T\lambda,\qquad
   \tau_c=-J_\omega^T\lambda
   $

4. 将其加入 Newton–Euler 方程：
   $
   f - J_v^T\lambda = ma
   $
   $
   \tau - J_\omega^T\lambda = I\dot\omega + \omega\times I\omega
   $

5. 在 3D pendulum 例子中，
   $
   \phi(q)=x+Rb-c,
   \qquad
   J=[I,\,-[Rb]]
   $
   并且 $J^T\lambda$ 恰好对应“作用在连接点上的点力及其诱导力矩”。

所以这一章最本质的一句话是：

> **约束不是凭空让刚体 obey 某种几何关系，而是通过约束势能产生恢复力，再由这些恢复力把刚体拉回到满足约束的状态。**

---

## 8. 关节约束下的时间步进（Articulated-Body Time Stepping）
### 8.1 这一章在解决什么问题

上一章中，我们已经知道：

- 每个 link 都有状态：位置 $x$、朝向 $R$、线速度 $v$、角速度 $\omega$
- 关节约束可以写成某种几何条件 $\phi(q)=0$
- 几何约束会通过 Jacobian 和约束力进入动力学

但如果真的要模拟一个 articulated body，我们需要在**每一个时间步**完成下面几件事：

1. 已知当前状态 $(x,R,v,\omega)$
2. 根据外力、外力矩和约束
3. 求出下一时刻的速度 $v',\omega'$
4. 再用新速度更新位置与旋转

所以这里的核心不是再推导“连续时间方程”，而是：

> **如何把带关节约束的刚体动力学整理成一个可数值求解的离散系统。**

---

### 8.2 从单个 link 开始：连续形式

课件第 3 页先考虑一个最简单情形：**一个 link + 一个 unary constraint $\phi$**。

其连续形式写成：

$
m I_{3\times 3}\, a = f - J^T \lambda
$

$
I \dot\omega + \omega \times I\omega = \tau - J_\omega^T \lambda
$

课件把它们合并成一个更紧凑的广义形式时，严格来说更合适的写法应该是：

$
M \dot \nu + J^T \lambda = \text{external and inertial terms}
$

把质量矩阵和广义加速度明确写开，就是

$
\begin{pmatrix}
mI_{3\times 3} & \\
& I
\end{pmatrix}
\begin{pmatrix}
a\\
\dot\omega
\end{pmatrix}
+
J^T\lambda=
\begin{pmatrix}
f\\
\tau-\omega\times I\omega
\end{pmatrix}
$

其中

$
\nu \sim \begin{bmatrix} v \\ \omega \end{bmatrix}
$

表示把线速度和角速度拼起来得到的**广义速度**。所以左边这一项本质上是“质量矩阵乘以广义加速度”。

如果坚持用传统记号，把广义速度写成 $\dot q$，那么这里就应写成

$
M \ddot q + J^T \lambda = \text{external and inertial terms}
$

这里：

- $\nu$（也就是传统记号下的 $\dot q$）是广义速度
- $M$ 是质量-转动惯量矩阵
- $J$ 是约束 Jacobian
- $\lambda$ 是约束响应量

这一步的重点不是公式形式，而是你要意识到：

> **线速度和角速度可以拼成一个统一的广义速度向量，平动和转动方程也就能拼成一个统一的块矩阵方程。**

---

### 8.3 时间离散：为什么未知量换成了“步末速度”

课件第 4 页开始做 temporal discretization。设时间步长为 $h$，并用上标/引号表示步末量。

这里最核心的离散思想是：

- 不直接先求加速度 $a,\dot\omega$
- 而是直接把方程改写成**新速度 $v',\omega'$** 的方程

这很常见，因为数值模拟里我们真正需要推进的是：

- 当前速度 $\to$ 下一步速度
- 当前位置 $\to$ 下一步位置

于是，离散后方程被整理成

$
M \nu' + J^T \mu = b
$

把它按平移/转动两部分展开，课件写成

$
\begin{pmatrix}
mI_{3\times 3} & \\
& I
\end{pmatrix}
\begin{pmatrix}
v^+\\
\omega^+
\end{pmatrix}
+
hJ^T\lambda=
\begin{pmatrix}
hf+mv\\
h(\tau-\omega\times I\omega)+I\omega
\end{pmatrix}
$

其中课件定义了

$
\mu = h\lambda.
$

这里：

- $\nu'$ 表示步末广义速度
- $b$ 收集了当前速度、外力、外力矩以及陀螺项等已知量
- $\mu$ 是把约束量乘上时间步长后的版本

---

### 8.4 为什么要引入 $\mu=h\lambda$

这一步表面上只是换记号，其实很自然。

因为离散方程里，外力通常会以“冲量”形式出现：

$
h f,\quad h \tau
$

所以约束项也很自然地按同样尺度处理，写成

$
\mu = h\lambda
$

这样一来，约束项就不再像连续时间里那样是“瞬时力”，而更像：

> **在一个时间步上累计产生的约束冲量**

这在接触和碰撞里尤其合理，因为碰撞响应本来就经常以冲量形式来理解。

---

### 8.5 仅有动力学方程还不够：为什么还需要第二行

课件第 5 页说得很直接：  
我们还需要另一个方程去确定 $\mu=h\lambda$。

因为当前只有

$
M \dot q' + J^T \mu = b
$

这里未知量有两个：

- 步末速度 $\dot q'$
- 约束响应 $\mu$

仅凭这一行不够解。

---

### 8.6 约束如何离散：显式 vs 隐式

如果完全照搬上一讲的 compliance 形式，可以写

$
\lambda = k\phi(q)
$

但课件指出：  
**直接显式使用这个式子，数值上不够稳定；更稳定的是把约束也做隐式离散。** 

于是大意上会写成：

$
\mu = hk\,\phi(q + h\dot q')
$

再对 $\phi$ 做一阶线性化：

$
\phi(q+h\dot q') \approx \phi(q) + h J(q)\dot q'
$

代入并整理后，课件给出第二个方程：

$
kJ(q)\dot q' - \frac{1}{h^2}\mu = -\frac{k}{h}\phi
$

这一步非常重要，它说明：

> **新速度会影响新约束误差，所以约束方程不能只看旧状态，而要让步末速度隐式地进入其中。**

这正是“更稳定”的来源。

---

### 8.7 合起来的块线性系统

课件第 6 页把两部分拼起来，得到一个平衡的线性系统：

$
\begin{bmatrix}
M & J^T \\
kJ & -\frac{1}{h^2}I
\end{bmatrix}
\begin{bmatrix}
\dot q'\\
\mu
\end{bmatrix}=
\begin{bmatrix}
b\\
-\frac{k}{h}\phi
\end{bmatrix}
$

这是本章最关键的方程之一。

---

### 8.8 每一块分别代表什么

这个系统建议你这样记。

#### 第一行：动力学平衡
$
M\dot q' + J^T\mu = b
$

表示：

- 如果没有约束，广义速度本来应该由 $b$ 给出
- 但因为有关节约束，需要加上 $J^T\mu$ 这一项来修正

所以第一行是在说：

> **约束冲量会改变速度。**

#### 第二行：约束一致性
$
kJ\dot q' - \frac{1}{h^2}\mu = -\frac{k}{h}\phi
$

表示：

- 如果当前约束有误差 $\phi\neq 0$
- 那么需要通过速度修正和约束响应来逐步把它拉回去

所以第二行是在说：

> **速度必须朝着减小约束误差的方向变化。**

---

### 8.9 数值刚性（numerical stiffness）为什么会出现

课件第 7 页指出：如果想让约束逼近“硬约束”，就需要很大的 $k$。  
但这会导致第二行里的系数

- $kJ$
- $-k/h$

都很大，从而造成线性系统病态。

直观上很好理解：

- $k$ 越大，弹簧越硬
- 系统越接近“必须严格满足约束”
- 但数值求解就越容易变得不稳定、条件数变差

这就是所谓的 numerical stiffness。

---

### 8.10 compliance $c=1/k$：为什么它能改善条件数

课件第 8 页做了一个关键重参数化：  
把第二行整体除以 $k$，并引入

$
c = \frac{1}{k}
$

得到更好条件的系统：

$
\begin{bmatrix}
M & J^T\\
J & -\frac{c}{h^2}I
\end{bmatrix}
\begin{bmatrix}
\dot q'\\
\mu
\end{bmatrix}=
\begin{bmatrix}
b\\
-\frac{\phi}{h}
\end{bmatrix}
$

这就是课件说的 **compliance constraints**。

---

### 8.11 为什么这个形式更好

因为现在你不再直接操控一个“很大很大的刚度 $k$”：

- $k\to \infty$ 对应 $c\to 0$
- 小的 compliance 更适合数值上表达“接近刚性”
- 同时矩阵系数尺度不再那么夸张

所以从实践上看：

> **用 compliance $c$ 来参数化约束软硬程度，比直接用刚度 $k$ 更稳定也更方便。**

---

### 8.12 多个 link 时怎么扩展

课件第 9–10 页说明扩展到多 link 非常直接。

#### 8.12.1 质量矩阵变成块对角

如果有多个 link，则每个 link 都有自己的

- 平动质量块
- 转动惯量块

所以整体质量矩阵 $M$ 是一个块对角矩阵：

$
M=
\begin{bmatrix}
M_1 & & \\
& M_2 & \\
& & M_3
\end{bmatrix}
$

因为在 maximal coordinates 下，每个刚体自己的惯性是独立的。

---

#### 8.12.2 约束 Jacobian 按连接关系拼接

例如 link 1 和 link 2 有约束 $\phi_{12}$，  
link 1 和 link 3 有约束 $\phi_{13}$，

则整体 Jacobian 会按“约束作用在哪些 link 上”来拼。

所以你可以把 $J$ 理解成：

> **每一行对应一个约束，每一列对应某个 link 的广义速度分量。**

如果某个约束不作用在某个 link 上，那一块就是 0。

---

### 8.13 这套时间步进算法的实际流程

课件第 11 页给出了很简洁的总结。

更详细地说，一个时间步可以这样做：

1. 从所有 links 收集当前 $x,R,v,\omega$
2. 遍历所有 links，组装整体 $M$ 和 $b$
3. 遍历所有 joints / constraints，计算 $\phi$ 与 $J$
4. 解上面的 compliance 线性系统，得到步末速度 $\dot q'$
5. 再用 $v',\omega'$ 去更新新的位置 $x'$ 与朝向 $R'$

这就完成了**无碰撞、只有关节约束**的 articulated-body 时间步进。

---

### 8.14 本章总结

本章的关键思想可以压缩成一句话：

> **把关节约束离散成一个与速度耦合的线性系统，在每个时间步同时解出步末速度和约束响应。**

核心公式是 compliance 形式：

$
\begin{bmatrix}
M & J^T\\
J & -\frac{c}{h^2}I
\end{bmatrix}
\begin{bmatrix}
\dot q'\\
\mu
\end{bmatrix}=
\begin{bmatrix}
b\\
-\frac{\phi}{h}
\end{bmatrix}
$

它奠定了后面处理**碰撞、接触和摩擦**的统一框架。

---

## 9. 离散碰撞检测（Discrete Collision Detection）
### 9.1 为什么碰撞检测必须放在前面

前一章的时间步进只处理了：

- 刚体本身的惯性
- 外力
- 关节约束

但如果物体撞到了地面，系统里会突然多出一种新的约束：

> **物体不能继续穿透接触面。**

这个约束不是固定的 joint constraint，而是：

- 只有在接触发生时才出现
- 其位置、法向、接触点都随几何关系变化

因此必须先做**碰撞检测**，找出接触几何信息，后面才谈得上建立接触力学模型。

---

### 9.2 课件为什么先从“球撞平面”讲起

课件第 13–14 页选了最简单的例子：  
一个球与平面 $z=0$ 接触。

这样做的好处是：

- 几何关系最直观
- 接触法向清晰
- 容易把重点放在“接触约束如何写”上，而不是复杂几何算法

---

### 9.3 离散碰撞检测在这里做什么

设球心为 $c$，半径为 $r$。

那么：

- 如果球心高度 $c_z > r$，说明球底部还在地面上方，没有接触
- 如果 $c_z \le r$，说明球已经碰到或穿入平面，需要处理接触

课件第 14 页就是这条判断逻辑。

这说明这里的碰撞检测是**离散的（discrete）**：

> 直接看当前时刻的几何位置，判断是否重叠/穿透。

它不考虑连续时间内是否“刚好擦过”某处，这属于更高级的 continuous collision detection 范畴。

---

### 9.4 检测到接触后，我们到底需要什么信息

课件第 15 页指出，碰撞检测阶段要输出两个关键东西：

1. 接触点 $p$
2. 接触局部坐标系 $R_c$

这是后续所有接触动力学的输入。

---

### 9.5 接触点 $p$ 为什么重要

因为接触力不是作用在刚体参考点上，而是作用在**接触点**上。

所以如果我们想知道接触力会对刚体产生什么效果，就必须知道：

- 它作用在哪里
- 它相对于刚体中心或 body frame 的位置是什么

这和前面 3D pendulum 例子里“点力诱导广义力”是同一个结构。

---

### 9.6 为什么还要构造一个局部接触坐标系 $R_c$

因为接触问题最自然的分解方式是：

- 法向方向：防止穿透
- 切向方向：处理摩擦

而在世界坐标里，接触法向和切向不一定对齐全局坐标轴。  
所以我们引入一个**接触局部 frame**，使其第三列（或 $z$ 轴）正好就是接触法向。

这样一来：

- 局部 $z$ 方向 = 法向
- 局部 $x,y$ 方向 = 两个切向

后面写接触力和摩擦约束都会非常方便。

---

### 9.7 球-平面例子里如何构造 $p$ 和 $R_c$

对于球与平面：

- 法向 $n$ 就是平面法向（对地面来说通常是 $(0,0,1)$）
- 接触点可取为
  $
  p = c - rn
  $

也就是从球心沿法向往下走一个半径，到达球表面与平面的接触点。

课件还给出一种构造局部 frame 的方法：

- 取 $n$ 作为第三轴
- 再构造两个与 $n$ 正交的单位向量，作为切向轴

这里要稍微分清两件事：

- 在这个**球-平面特例**里，如果球心 $c$、半径 $r$ 和接触法向 $n$ 都已经确定，那么
  $
  p=c-rn
  $
  其实是唯一确定的；
- 真正**不唯一**的是局部接触坐标系 $R_c$ 的选法。因为只要第三轴固定为法向 $n$，那么前两个切向轴仍然可以在切平面内绕 $n$ 再旋转一个任意角度。

所以这里更准确地说，应该是：

> 在球-平面这个例子里，接触点 $p$ 由 $p=c-rn$ 唯一确定；  
> 但接触局部 frame $R_c$ 并不唯一，因为切向基的选取有自由度。

这很重要，因为物理上真正关键的是：

- 法向要正确
- 切平面张成的二维子空间要正确

至于切向坐标轴在切平面内具体怎么转一个角度，通常不影响法向非穿透约束；摩擦里虽然会影响切向分量的具体坐标表示，但只要两个切向轴始终张成同一个切平面，就仍然是合法的选择。

只有在更一般的接触建模里，例如面接触、线接触，或者把一片接触区域压缩成一个“代表接触点”时，才会出现“$p$ 的选取本身也不唯一”的情况。

即：
**$p$ 和 $R_c$ 的选法不是唯一的**。

---

### 9.8 本章总结

碰撞检测这一步的意义不是直接算力，而是：

> **把“是否接触”这个几何问题，转成后续接触动力学所需的局部几何数据。**

最关键的输出是：

- 接触点 $p$
- 接触法向 / 局部接触坐标系 $R_c$

后面所有接触 Jacobian、互补条件、摩擦模型，都会围绕这两样数据展开。

---

## 10. Contact Jacobian：从接触点速度到广义速度
### 10.1 为什么接触问题也要引入 Jacobian

前面关节约束已经让我们很熟悉一个套路：

- 先写几何量（约束、点位置、点速度）
- 再把它们表示成刚体广义速度的线性函数
- 最后用 Jacobian 的转置把力映射回广义力

接触问题完全一样。

只不过这次不再是 joint constraint，而是：

> **接触点在法向上不能往里穿。**

要表达这个条件，我们需要先知道：

- 接触点在世界中的速度是什么
- 这个速度在接触局部坐标里沿法向和切向的分量是什么

这正是 Contact Jacobian 的作用。

---

### 10.2 接触点速度的世界坐标表达

课件第 17 页回顾了：  
设 $J_p$ 是刚体上点 $p$ 的 Jacobian，那么接触点速度满足：

$
v_p = J_p
\begin{bmatrix}
v\\
\omega
\end{bmatrix}
= J_p \dot q
$

其中 $\dot q$ 是刚体广义速度。

如果你联系上一讲的 3D pendulum，这其实就是同一件事：

$
v_p = v + \omega \times (p-x)
$

只不过现在我们把它抽象成 Jacobian 形式 $J_p\dot q$。

---

### 10.3 为什么还要变到局部接触坐标系里

世界坐标下的速度虽然也能看，但接触约束最关心的是：

- 法向速度 $v_n$
- 切向速度 $v_t$

而这两个量最自然是在**接触局部坐标系**里定义。

于是课件第 17 页把接触点速度转到局部 frame：

$
v_p^B = {R_c}^T J_p \dot q
$

并定义

$
J_c := {R_c}^T J_p
$

于是

$
v_p^B = J_c \dot q
$

这就是 Contact Jacobian。

---

### 10.4 $J_c$ 的物理意义

这个式子非常值得记住：

$
v_p^B = J_c\dot q
$

表示：

> **刚体的广义速度 $\dot q=(v,\omega)$，决定了接触点在局部接触坐标系中的速度。**

而局部坐标系的好处是：

- 第三个分量直接就是法向速度
- 前两个分量直接就是切向速度

所以一旦有了 $J_c$，后面的非穿透和摩擦条件都能直接写在速度分量上。

---

### 10.5 接触力为什么也要转到局部坐标

课件第 18 页又做了一个完全平行的变换。

设局部接触力为

$
\lambda_p^B = {R_c}^T \lambda_p
$

那么其诱导的广义力为

$
J_p^T \lambda_p=
J_p^T R_c \lambda_p^B=
{J_c}^T \lambda_p^B
$

这一步特别关键，因为它表明：

> **同一个 Contact Jacobian，既负责把广义速度映射成接触点速度，也负责把接触点力映射回广义力。**

这与机器人学和刚体动力学里的标准 Jacobian-transpose 力映射完全一致。

---

### 10.6 速度映射与力映射是一对对偶关系

建议你把这两条背下来：

#### 速度映射
$
v_p^B = J_c\dot q
$

#### 力映射
$
Q_{contact} = {J_c}^T \lambda_p^B
$

这里 $Q_{contact}$ 是作用在刚体广义坐标上的力/力矩。

这就是为什么 Jacobian 这么核心：  
它在速度空间和力空间之间提供了桥梁。

---

### 10.7 接触动力学的块系统从哪来

课件第 19 页开始把接触加进动力学。

对一个无关节、单刚体的简单情形，块系统写成：

$
\begin{bmatrix}
M & {J_c}^T\\
J_c & 0
\end{bmatrix}
\begin{bmatrix}
\dot q'\\
\lambda_p^B
\end{bmatrix}=
\begin{bmatrix}
b\\
v_p^B
\end{bmatrix}
$

这个式子表面看像上一章的关节约束系统，但这里还没有真正写完，因为：

- $\lambda_p^B$ 未知
- $v_p^B$ 也未知
- 接触有特殊的单边性质：只能推不能拉

所以不能像关节那样简单写成等式约束，后面必须加互补条件。

---

### 10.8 本章总结

接触 Jacobian 的核心作用是：

1. 把刚体广义速度映射成接触点局部速度
   $
   v_p^B = J_c\dot q
   $
2. 把局部接触力映射成刚体广义力
   $
   Q_{contact} = {J_c}^T\lambda_p^B
   $

这就把接触问题成功接进了前面的矩阵动力学框架，为下一章写非穿透互补条件做好了准备。

---

## 11. 非穿透接触与线性互补约束（LCP）
### 11.1 为什么接触不是普通等式约束

关节约束通常是双边约束：

$
\phi(q)=0
$

意思是系统必须始终满足，往正负两边偏都不允许。

但接触不是这样。  
对于地面接触，真正的物理条件是：

- 物体不能往地里穿
- 但如果它想离开地面，是可以离开的
- 地面只能“推”物体，不能“拉”住物体

所以接触是**单边约束（unilateral constraint）**，而不是双边约束。

---

### 11.2 摩擦忽略时，接触力长什么样

课件第 20 页先考虑 frictionless contact。此时局部接触力只保留法向分量：

$
\lambda_p^B = (0,0,\lambda_n)
$

局部接触速度则写成：

$
v_p^B = (?, ?, v_n)
$

因为在无摩擦情形下：

- 切向速度不受约束
- 唯一关键的是法向速度 $v_n$

---

### 11.3 两种情况：接触 vs 分离

课件第 21–22 页把接触状态拆成两种互斥情况。

#### 情况 A：物体保持接触
此时地面需要提供非负法向力来阻止继续穿透，因此：

$
\lambda_n \ge 0,\qquad v_n=0
$

解释：

- $\lambda_n\ge 0$：地面只能向上推，不能向下拉
- $v_n=0$：接触点法向速度为 0，不能再继续往里走

---

#### 情况 B：物体脱离接触
如果物体本来就有离开地面的趋势，那么地面不该继续施力：

$
\lambda_n = 0,\qquad v_n \ge 0
$

解释：

- $\lambda_n=0$：既然分离了，就没有接触力
- $v_n\ge 0$：法向速度朝离开方向或刚好为 0

---

### 11.4 为什么这两种情况可以合成一个互补条件

课件这里其实是在说：接触问题不是单独只求一个 $\lambda_n$，而是把**动力学未知量**和**接触未知量**一起放进一个块矩阵系统里。单个法向接触时，可以写成

$
\begin{pmatrix}
M & J_c^T\\
J_c & 0
\end{pmatrix}
\begin{pmatrix}
\dot q^{+}\\
0\\
0\\
\lambda_n
\end{pmatrix}=
\begin{pmatrix}
b\\
?\\
?\\
v_n
\end{pmatrix}
$

这个公式的意思是：

- 上半部分是时间步进后的动力学方程；
- 下半部分是在接触局部坐标系里施加的接触速度条件；
- 这个例子里只保留法向接触力 $\lambda_n$，切向两个分量先设为 $0$；
- 右下角的 $v_n$ 就是接触点在法向上的相对速度。

也就是说，求解器实际上是在同时决定：

- 步末广义速度 $\dot q^+$
- 法向接触力（或冲量）$\lambda_n$

而它们之间正是通过 $J_c$ 耦合起来的。

课件第 23 页把两种情况合并成：

$
0 \le \lambda_n \perp v_n \ge 0
$

即：

$
\lambda_n \ge 0,\qquad v_n \ge 0,\qquad \lambda_n v_n = 0.
$

这里的“$\perp$”表示互补（complementary）。

这个条件非常重要，建议这样理解：

- $\lambda_n$ 和 $v_n$ 都不能为负
- 但它们不能同时严格为正
- 要么靠力顶住（$\lambda_n>0, v_n=0$）
- 要么已经离开（$\lambda_n=0, v_n>0$）
- 也允许边界情况（$\lambda_n=0,v_n=0$）

---

### 11.5 为什么它叫 LCP

因为接触动力学里：

- $\lambda_n$ 通过线性系统决定 $v_n$
- 但 $\lambda_n$ 和 $v_n$ 之间还要满足上面的互补关系

于是问题整体就是：

> 在线性关系之上，附加互补条件

这就是 **Linear Complementarity Problem**。

它不是普通线性方程组，因为未知量之间存在“要么这个活跃，要么那个活跃”的逻辑结构。

---

### 11.6 单接触点时怎么求解

课件第 24–25 页给出最朴素的方法：  
直接枚举两种情况。

#### Case 1：假设无接触
设

$
\lambda_n=0
$

先解动力学，看结果是否满足

$
v_n\ge 0
$

如果是，说明这个假设自洽。

---

#### Case 2：假设有接触
设

$
v_n=0
$

再解系统，看算出的接触力是否满足

$
\lambda_n\ge 0
$

如果是，也自洽。

---

### 11.7 为什么至少在理想情况下总能有解

课件问了一个很自然的问题：会不会两种情况都失败？

在标准良性情形下，接触力学模型就是为了反映物理世界的基本单边条件，通常应该至少有一个物理解。  
但更复杂的多接触、摩擦和数值近似下，求解可能困难、可能不唯一、也可能需要迭代近似算法。

这也正是下一章要讨论的重点。

---

### 11.8 多个接触点时会发生什么

课件第 26–28 页扩展到多刚体、多接触点。

这里先要回答一个很自然的问题：为什么会说“枚举所有接触状态”？

原因不是我们主观上想枚举，而是**每个接触点都自带一个二选一的互补逻辑**：

- 要么该点处于 active 状态：$\lambda_n>0,\ v_n=0$
- 要么该点处于 inactive 状态：$\lambda_n=0,\ v_n>0$

也就是说，对每个接触点，你一开始都不知道它到底应该被归到哪一种情况里。为了找到真正自洽的物理解，最朴素的办法就是：

1. 先假设一组接触状态；
2. 在这个假设下解线性系统；
3. 再检查解出来的 $\lambda_n$ 和 $v_n$ 是否真的满足互补条件。

所以“枚举所有接触状态”的本质，是在枚举所有可能的 active set。

单接触点时，这只意味着检查两种情况，所以还很容易；但多个接触点时，组合数会迅速爆炸。

如果有 $m$ 个接触点，那么每个接触点都有两种状态：

- 接触
- 不接触

于是总组合数就是：

$
2^m
$

课件第 28 页明确写出其复杂度是指数级。

这说明：

> **枚举所有接触状态在多个接触点时几乎不可行。**

不过这里要注意：课件讲“枚举”主要是为了帮助你理解互补条件为什么会带来组合爆炸，并不意味着实际工程里真的暴力尝试全部状态。实际求解里通常会改用：

- active-set 方法
- LCP 求解器
- Projected/Blocked Gauss–Seidel 一类迭代法

因此必须采用更高效的近似或迭代算法。

---

### 11.9 本章总结

无摩擦接触的关键不是“接触力怎么写”，而是：

> **接触力和法向速度之间存在单边互补关系。**

核心条件是：

$
0\le \lambda_n \perp v_n \ge 0
$

这比普通等式约束更复杂，也解释了为什么接触求解通常比关节约束更难。

---

## 12. 多接触点求解：Blocked Gauss–Seidel（BGS）
### 12.1 为什么需要 BGS

上一章已经看到：

- 单接触点时，可以直接枚举“接触 / 不接触”两种情况
- 多接触点时，组合数是 $2^m$

这在稍大一点的系统里就完全不可接受。

所以我们希望要一种方法：

- 不同时枚举所有接触点的所有状态
- 而是“局部地”更新每个接触点
- 逐步逼近一个全局自洽的解

这就是 BGS 的思想。

---

### 12.2 BGS 的基本想法

课件第 29 页先给出初始化：  
为所有碰撞先指定一个当前状态选择，比如都设成“无接触”。

例如对于 3 个接触点，初始状态可以写成：

$
(N,N,N)
$

这里 $N$ 表示 no contact，$Y$ 表示 in contact。

---

### 12.3 一次只更新一个接触点

课件第 30–31 页说明 BGS 的核心步骤：  
更新第一个接触点时，把其他接触点的状态都暂时冻结，只在当前点尝试两种情况。

例如当前状态是

$
(N,N,N)
$

更新第一个点时：

- 假设它是 $N$，看看是否自洽
- 假设它是 $Y$，看看是否更合适

选定后可能变成

$
(Y,N,N)
$

然后再去更新第二个点、第三个点。

---

### 12.4 为什么它叫 “Gauss–Seidel”

因为它和线性代数里经典的 Gauss–Seidel 迭代思路很像：

- 当前更新某个分量时
- 其他分量先固定
- 用最新值立刻参与后续更新

所以这里本质上是在做：

> **逐个接触点的坐标更新 / block coordinate update**

只不过每个 block 不是一个实数，而是一个接触状态选择以及相应的小系统求解。

---

### 12.5 “Blocked” 体现在哪

因为单个接触点更新时，处理的不是一个简单标量，而是一小块相关变量：

- 这个点的接触力
- 这个点的法向速度
- 该点对应的局部线性系统

所以它被称为 **Blocked** Gauss–Seidel，而不是最简单的标量版本。

---

### 12.6 BGS 的优点

它最大的优点是：

- 不再需要全局枚举 $2^m$ 种状态
- 每次只处理一个接触点，代价低得多
- 实现简单，工程上常用

换句话说，它是一个典型的：

> **用迭代近似换取可计算性** 的方法。

---

### 12.7 BGS 的缺点：是否保证收敛？

课件第 32 页明确反问：  
**Does BGS guarantee convergence?** 

潜台词就是：**不一定。**

原因直观上也好理解：

- 一个接触点的状态会影响别的接触点
- 更新一个点时做出的局部选择，可能破坏另一个点的自洽性
- 所以整体可能震荡、循环，或者只得到近似解

因此 BGS 更像是：

- 一个实践中常用的近似迭代法
- 在很多场景下好用
- 但不是无条件严格收敛的万能算法

---

### 12.8 带接触的整体时间步进流程

课件第 33 页总结了“时间步进 + 接触”的完整流程。

更详细地说：

1. 收集所有 links 的当前 $x,R,v,\omega$
2. 组装整体 $M$ 和 $b$
3. 组装 joints 对应的 $\phi$ 与 $J$
4. 做碰撞检测，得到所有接触点的 $p$ 与局部 frame $R_c$
5. 组装接触 Jacobian $J_c$
6. 用 BGS 近似求解 LCP，得到新的步末速度 $\dot q'$
7. 用 $v',\omega'$ 更新 $x',R'$

这样，关节约束与接触约束就在同一个时间步框架中结合了起来。

---

### 12.9 本章总结

BGS 的核心思想可以概括为：

> **多接触点问题太大，不能全局穷举；那就逐个接触点做局部状态更新，并迭代到收敛或最大步数。**

它不是最严格的理论方法，但在接触动力学里非常典型，也很符合“工程可用优先”的思想。

补充说明，关于 BGS 最容易困惑的两点是：

1. **当前点刚更新得自洽，后面别的点一变，它会不会又不自洽？**

会，而且这正是 BGS 必须反复 sweep 多轮的原因。  
更新某个接触点时，我们只是在“其他接触点当前先固定”的前提下，让这个点局部满足自己的互补条件。等后面别的接触点更新后，耦合关系变了，这个点当然可能又被带偏。因此 BGS 的目标从来不是“一次更新永久正确”，而是通过反复来回修正，逐步逼近整体自洽。

2. **如果当前点看起来两种情况都不自洽怎么办？**

这通常不表示“这个点彻底无解”，更常见的解释是：

- 当前其他接触点的取值还不够好，所以这个点所在的局部背景本身就还不一致；
- 你看到的是某一轮迭代中的暂时矛盾，而不是最终结果；
- 继续 sweep 几轮后，随着其他点被修正，这个点往往也会回到更合理的状态。

从算法观点看，BGS 不是在每一步就构造一个已经全局正确的 active set；它只是在做局部修正，并希望这些局部修正在迭代中逐渐兼容。

所以理解 BGS 时，最好把它想成：

> **每一步只保证“当前点对当前背景局部自洽”，最终才追求“所有点同时近似自洽”。**

---

## 13. 摩擦接触：静摩擦、滑动摩擦与库仑锥
### 13.1 为什么摩擦比法向接触复杂得多

在前面无摩擦模型里，我们只关心：

- 法向接触力 $\lambda_n$
- 法向速度 $v_n$

这已经足以阻止穿透。

但真实接触里，物体在接触面上还可能：

- 静止不滑
- 沿切向滑动
- 滑动方向不断变化

这就需要引入切向力和切向速度。

---

### 13.2 局部接触坐标中的力和速度分解

课件第 35 页把局部接触力与速度分解成：

$
\lambda_p^B = (\lambda_t,\lambda_n)
$

$
v_p^B = (v_t,v_n)
$

其中：

- $\lambda_t\in \mathbb R^2$：切向摩擦力
- $\lambda_n\in \mathbb R$：法向力
- $v_t\in \mathbb R^2$：切向速度
- $v_n\in \mathbb R$：法向速度 

所以在局部接触坐标里，接触问题自然被分成：

- 1 个法向问题
- 2 个切向问题

---

### 13.3 Case 1：无接触

如果没有接触，那么当然没有摩擦力：

$
\lambda_t=0,\qquad \lambda_n=0,\qquad v_n\ge 0
$

这和前面无摩擦时完全一致。

---

### 13.4 Case 2：接触但静摩擦（sticking）

课件第 36 页给出第二种情况：接触但不滑。

其条件是：

$
\|\lambda_t\|\le \mu \lambda_n,\qquad \lambda_n>0
$

$
v_t=0,\qquad v_n=0
$

解释如下：

- $v_n=0$：法向上保持接触
- $v_t=0$：切向上也没有相对滑动
- $\|\lambda_t\|\le \mu\lambda_n$：静摩擦力大小不能超过库仑极限

这说明静摩擦不是一个固定大小的力，而是：

> **在某个范围内自适应地取值，只要足以阻止滑动即可。**

---

### 13.5 Case 3：接触且滑动（sliding）

课件第 37 页给出第三种情况：发生滑动。

此时：

$
\|\lambda_t\| = \mu \lambda_n,\qquad \lambda_n>0
$

而且摩擦力方向与滑动方向相反：

$
\frac{v_t}{\|v_t\|}=
-\frac{\lambda_t}{\|\lambda_t\|}
$

同时仍有

$
v_n=0
$

这正是经典库仑摩擦：

- 一旦开始滑，摩擦力大小达到极限
- 方向永远反着切向速度

---

### 13.6 为什么静摩擦和滑动摩擦本质不同

建议你特别记住这一点：

#### 静摩擦
- 目标：让 $v_t=0$
- 力大小可变
- 只要不超过 $\mu \lambda_n$

#### 滑动摩擦
- 目标：阻碍滑动，但阻止不了
- 力大小固定在边界 $\mu\lambda_n$
- 方向与速度相反

所以摩擦并不是一个统一的线性关系，而是一个**分段、非光滑、非线性**模型。

这也是为什么真正精确求解摩擦接触要比无摩擦接触难得多。

---

### 13.7 库仑摩擦锥（Coulomb Friction Cone）的直观理解

从条件

$
\|\lambda_t\|\le \mu \lambda_n
$

可以看出：  
对于给定法向力 $\lambda_n$，允许的切向摩擦力形成一个圆盘；在三维力空间里，这些圆盘沿法向叠起来，就是一个圆锥。

所以：

- 锥内：静摩擦可能区域
- 锥面：滑动摩擦极限区域

这就是库仑摩擦锥。

---

### 13.8 本章总结

摩擦接触的核心不是再多加一个力，而是：

> **切向力与切向速度之间有一套比法向互补更复杂的分段关系。**

三种状态分别是：

1. 无接触：$\lambda_t=\lambda_n=0$
2. 接触静止：$\| \lambda_t\|\le \mu\lambda_n,\ v_t=0$
3. 接触滑动：$\| \lambda_t\|=\mu\lambda_n,\ \lambda_t$ 与 $v_t$ 反向

它们共同构成了接触动力学中最经典的库仑摩擦模型。

---

## 14. Boxed LCP 近似与工程实现视角
### 14.1 为什么要近似摩擦锥

上一章的库仑摩擦锥条件虽然物理上很自然，但从求解角度看很麻烦：

- $\|\lambda_t\|\le \mu\lambda_n$ 是圆锥约束
- 滑动时还要额外满足方向相反
- 这会带来非线性、非光滑问题

而前面我们的求解器思路（LCP / BGS）更擅长处理：

- 线性关系
- 一维互补或区间约束

所以工程上常常做一个近似：  
把圆锥换成一个盒子（box）。

---

### 14.2 Boxed LCP 是什么

课件第 38 页以单个切向分量 $\lambda_x, v_x$ 为例说明 boxed LCP。

对某个切向方向，考虑：

#### Case 1：静摩擦
$
-\mu\lambda_n \le \lambda_x \le \mu\lambda_n,\qquad v_x=0
$

#### Case 2.1：向一个方向滑动
$
\lambda_x=-\mu\lambda_n,\qquad v_x\ge 0
$

#### Case 2.2：向另一个方向滑动
$
\lambda_x=\mu\lambda_n,\qquad v_x\le 0
$

这就把原本二维切向上的圆盘约束，拆成了两个独立坐标轴上的“区间 + 互补”约束。

这里特别容易误解的一点是：

> Boxed LCP 不是在说“如果 $x$ 方向速度大于 0，那么真实摩擦力就等于 $\mu\lambda_n$ 并完全沿 $x$ 方向作用”。

更准确地说，它是在对**切向摩擦力的各个坐标分量**分别施加一个区间约束。

也就是说，对单个分量 $\lambda_x$，boxed 近似只要求：

- 不滑时，$\lambda_x$ 落在区间 $[-\mu\lambda_n,\mu\lambda_n]$ 内；
- 若沿 $+x$ 方向滑动，则该分量饱和到 $-\mu\lambda_n$；
- 若沿 $-x$ 方向滑动，则该分量饱和到 $+\mu\lambda_n$。

所以它控制的是“摩擦力在某个切向轴上的分量”，而不是严格意义下整个切向摩擦向量的真实投影关系。

真正的库仑摩擦要求的是整个切向向量满足

$
\|\lambda_t\|\le \mu\lambda_n
$

滑动时还要满足

$
\lambda_t = -\mu\lambda_n\frac{v_t}{\|v_t\|}
$

也就是：

- 限制的是**整个向量的模长**
- 方向反着**总切向速度方向**

而 boxed LCP 把这个“圆盘/圆锥约束”近似成了“每个坐标分量各自被夹在一个区间里”。因此它更容易求解，但不再是对真实摩擦锥的精确表达。

---

### 14.3 为什么叫 “boxed”

因为对于每个切向分量 $\lambda_x$，它被限制在一个区间里：

$
[-\mu\lambda_n,\ \mu\lambda_n]
$

多个切向分量组合起来，就形成了一个盒子，而不再是圆盘/圆锥。

所以这相当于：

- 把圆形摩擦锥截面近似成正方形
- 把圆锥近似成一个棱锥 / 盒状约束

这会牺牲一部分旋转对称性，但大大简化求解。

例如在二维切向平面里，真实库仑摩擦允许的是一个半径为 $\mu\lambda_n$ 的圆盘；  
而 boxed 近似允许的是边长为 $2\mu\lambda_n$ 的正方形。

所以 boxed 近似的含义不是“把真实摩擦力精确分解到 $x,y$ 方向”，而是：

> **用一个更容易处理的正方形可行域，去近似原本的圆形可行域。**

---

### 14.4 这种近似的优缺点

#### 优点
- 更容易接入 LCP / projected Gauss–Seidel / BGS 等算法
- 每个切向分量能独立裁剪到区间里
- 工程实现简单

#### 缺点
- 不再精确等价于真实库仑圆锥
- 摩擦方向和大小会带有网格轴/局部坐标轴依赖
- 在某些方向上会偏强或偏弱

所以它是一种典型的：

> **为了可解性而采用的工程近似**

---

### 14.5 Gripper 示例想说明什么

课件最后两页给出一个 gripper 代码示例。

它的意义不是再引入新理论，而是告诉你：

- 前面这些东西不是“纯数学推导”
- 它们真的可以落到代码里
- 一个简单的抓取器场景里就同时需要：
  - 刚体动力学
  - 关节约束
  - 接触检测
  - 非穿透
  - 摩擦

也就是说，这整份课件其实是在给你搭一个最基础的刚体仿真器逻辑框架。

---

### 14.6 从实现角度看这一讲的统一主线

如果把 `rigid-bodies-ii.pdf` 整体压缩成一条工程主线，就是：

1. 用块矩阵系统处理关节约束时间步进
2. 在每步开头做碰撞检测
3. 对每个接触点构造局部 frame 和 contact Jacobian
4. 用互补条件表达非穿透
5. 用 BGS 近似求解多接触点
6. 再加入库仑摩擦或 boxed LCP 近似
7. 更新速度与位姿，进入下一步

这样，基础的 articulated rigid-body simulator 就成形了。

---

### 14.7 本章总结

boxed LCP 的本质是：

> **把真实的库仑摩擦锥约束换成更容易求解的坐标轴区间约束。**

它不是最精确的物理模型，但在很多工程仿真器中都非常常见，因为：

- 足够实用
- 数值上更容易实现
- 与 Gauss–Seidel 类方法兼容性好

这也体现了 physics-based simulation 的一个典型特点：

> 理论上最精确的模型，未必是数值上最可用的模型；  
> 实际仿真常常是在物理真实性、稳定性和计算效率之间做折中。

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
