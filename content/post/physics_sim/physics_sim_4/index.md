---
title: "Physically-based Simulation Week 5"
date: 2026-05-04
math: true
tags:
  - Notes
  - Physics
  - Simulation
categories:
  - Simulation
description: "物理仿真流体课堂笔记"
---

# Fluid Simulation I — Lecture Notes

> 本文根据课件 **Fluid Simulation** 整理，目标是把原本偏课堂讲义风格的 slides，改写成适合直接放到 blog 上的 lecture notes。整篇不按页逐条翻译，而是围绕一条更容易理解的主线展开：**流体的运动学表示 → 不可压缩约束 → Euler 方程 → MAC 网格离散化 → Stable Fluids 时间推进 → 投影法 → 不规则边界与 level set 表示**。

---

## 1. 这份课件到底在讲什么？

这份课件表面上是在讲“fluid simulation”，但它真正想建立的是一个非常经典的流体模拟框架：

1. 先从**连续体运动学**出发，明确流体是什么；
2. 然后写出**不可压缩无粘流体**的控制方程，也就是 Euler equations；
3. 接着把这些连续方程离散到 **MAC grid** 上；
4. 再通过 **advection + force + projection** 的 splitting scheme 做时间推进；
5. 最后扩展到**自由表面和不规则边界**，并用 **level set** 来表示流体区域。

如果用一句话总结，就是：

> **这份课件想教你如何从连续流体方程，一步一步走到一个真正可实现的稳定流体求解器。**

---

## 2. 流体的运动学：Lagrangian 视角

课件一开始先从 **Lagrangian representation** 讲起。这是一个非常自然的出发点，因为它最符合“跟着流体粒子看世界”的直觉。

设流体的材料坐标（material space）中的点记为 $X$，它在世界坐标（world space）中的位置记为

$
\mathbf{x} = \phi(X,t).
$

这里：

- $X$ 表示“这是哪一个材料点”；
- $\phi(X,t)$ 表示这个材料点在时间 $t$ 时跑到了哪里。

这就是所谓的 **deformation map**：它把材料空间映射到世界空间。

### 2.1 为什么流体也能像固体那样写成形变映射？

很多人第一次看这里会疑惑：流体不是会“流动”吗，为什么还写得像弹性体？

关键在于，这里只是做**运动学描述**，也就是：

- 只关心“粒子从哪到哪”；
- 还没有讨论“它为什么这么动”。

从这个角度，不管是点集、刚体还是流体，都可以先写成“材料点随时间运动”的形式。区别只在于它们满足的约束不同。

课件第 4 页专门做了一个对比：

- **point set**：没有约束；
- **fluid**：满足不可压缩约束；
- **rigid body**：满足刚体等距约束。

这张对比图的意义很大，因为它说明：

> **流体不是“没有约束的点集”，它和刚体一样，也是一类带约束的连续体。**

---

## 3. 不可压缩约束：为什么 $\det F = 1$？

对于不可压缩流体，课件给出的约束是：

$
\det F(X,t) = 1,
$

其中 $F = \nabla_X \phi(X,t)$ 是形变梯度。

这个条件的物理意义是：

> **任意一个微小流体体积在运动过程中体积不变。**

因为 $\det F$ 描述的正是局部体积缩放因子。

- 如果 $\det F > 1$，局部体积膨胀；
- 如果 $\det F < 1$，局部体积压缩；
- 如果 $\det F = 1$，局部体积保持不变。

所以，不可压缩的意思并不是“流体不会变形”，而是：

- 它当然可以拉伸、扭曲、翻卷；
- 但任何局部小体积都不能被压缩或膨胀。

这一点非常重要，因为很多人会把“不可压缩”误解成“形状不变”。其实真正不变的是**体积**，不是形状。

---

## 4. 速度约束：从 $\det F = 1$ 推到散度为零

课件第 5–7 页开始问：

> 如果位置满足不可压缩约束，那么速度场到底要满足什么条件？

这个问题很关键，因为数值模拟时我们真正迭代更新的往往不是 $\phi$，而是速度场 $\mathbf{u}$ 或 $\mathbf{v}$。

从 Lagrangian 角度，如果对

$
\det F(X,t) = 1
$

对时间求导，就能得到速度所满足的约束。课件给出的结论是：

$
\mathrm{tr}(\nabla_X \mathbf{v}\, F^{-1}) = 0.
$

这在 Lagrangian 坐标下是速度的不可压缩条件。

继续求时间导数，还可以得到加速度也必须满足相应的约束。

### 4.1 为什么这里要先推速度约束？

因为后面无论写 Euler 方程还是做数值离散，核心未知量通常都是速度场。如果只知道“位置必须满足体积不变”，但不知道“速度该怎么限制”，那就无法真正推进时间。

所以这一部分的作用是建立桥梁：

- 位置上的体积保持；
- 变成速度上的零散度约束；
- 再进入 Eulerian PDE 表述。

---

## 5. Eulerian 视角：为什么流体模拟更常用它？

课件第 9 页之后切换到了 **Eulerian representation**。

这对应的是另一种观察方式：

- **Lagrangian**：跟着每个流体粒子走；
- **Eulerian**：固定看空间中的某个位置，问这个位置此刻流过什么。

这两种视角描述的是同一个物理量，但适合的问题不同。

### 5.1 Lagrangian vs. Eulerian

课件第 10–12 页用变形映射 $\phi$ 和其逆映射 $\psi$ 说明：

- Lagrangian view：$q^L(X,t)$，表示材料点 $X$ 携带的量；
- Eulerian view：$q^E(x,t)$，表示空间位置 $x$ 处观测到的量。

两者可以通过 push-forward / pull-back 互相转换：

$
q^L(X,t) = q^E(\phi(X,t), t),
$

$
q^E(x,t) = q^L(\psi(x,t), t).
$

### 5.2 为什么数值流体更偏好 Eulerian？

因为在网格法里，我们通常把速度、压力等量存放在固定空间网格上。这样：

- 容易做有限差分；
- 容易处理偏微分方程；
- 容易和边界条件结合。

所以，虽然 Lagrangian 视角更直观，但后面真正做离散求解时，主要还是 Eulerian 视角。

---

## 6. Material Derivative：为什么会出现 $\partial_t + u\cdot\nabla$？

这一部分是整份课件最重要的基础概念之一。

课件第 13 页推导了 **material derivative**：

$
\frac{Dq}{Dt} = \frac{\partial q}{\partial t} + \mathbf{u}\cdot\nabla q.
$

它表示的不是“固定在空间某点看到的变化率”，而是：

> **跟着某个流体粒子一起移动时，这个粒子携带的物理量变化有多快。**

这里有两部分：

- $\partial q/\partial t$：固定在空间某一点观察到的局部时间变化；
- $\mathbf{u}\cdot\nabla q$：因为粒子在空间中移动，穿过了有空间梯度的场，因此额外带来的变化。

### 6.1 为什么这个公式很自然？

你可以把它理解成链式法则：

- 粒子位置本身在变；
- 场在空间中也有分布；
- 所以跟着粒子看，变化既来自“时间本身”，也来自“位置移动”。

课件第 14–16 页用一个一维温度传输例子说明这一点：

- 初始温度场是 $T = 10x$；
- 如果流体以常速 $c$ 向右平移；
- 那么 Eulerian 温度场虽然会变化，但每个流体粒子携带的温度却保持不变。fileciteturn8file0

这就是 advection 的本质：

> **物理量被流体“带着走”。**

---

## 7. Eulerian 下的不可压缩条件：$\nabla\cdot u = 0$

课件第 18 页把 Lagrangian 的不可压缩约束转成了 Eulerian 形式，得到经典结果：

$
\nabla \cdot \mathbf{u} = 0.
$

这就是我们在流体模拟中最常看到的不可压缩条件。fileciteturn8file0

### 7.1 这个式子的直观意义是什么？

散度 $\nabla\cdot \mathbf{u}$ 衡量的是一个小体积周围的“净流出率”：

- 如果流出比流入多，体积会变小或膨胀异常；
- 如果流入比流出多，局部会挤压堆积；
- 只有净流量为零，局部体积才能保持不变。

所以

$
\nabla\cdot \mathbf{u}=0
$

就是不可压缩流体最核心的速度约束。

课件第 19 页还给了一个离散直觉图：一个网格单元四个边上的流量流入流出应当平衡。fileciteturn8file0

这张图非常重要，因为后面 projection 的离散方程，本质上就是把这条“净流量为零”的要求施加到每个流体网格单元上。

---

## 8. Dynamics：Euler Equations 是怎么来的？

课件第 20–21 页正式写出了不可压缩无粘流体的 **Euler equations**：

$
\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla \mathbf{u} + \frac{1}{\rho}\nabla p = \mathbf{g},
$

同时满足约束：

$
\nabla\cdot\mathbf{u}=0.
$

如果有自由表面，还会有边界条件：

$
p=0.
$

### 8.1 每一项分别表示什么？

这个方程虽然看起来短，但其实把整个流体动力学都压缩进去了：

- $\partial_t \mathbf{u}$：局部随时间变化；
- $\mathbf{u}\cdot\nabla \mathbf{u}$：对流项，表示速度场被自己搬运；
- $\frac{1}{\rho}\nabla p$：压力梯度带来的力；
- $\mathbf{g}$：外力，例如重力。

换句话说，流体速度的变化来自三类机制：

1. 自己把自己 advect；
2. 外力推动；
3. 压力强制它恢复不可压缩。

### 8.2 为什么 $\nabla p$ 可以被叫做压力力？

课件第 22 页专门从物理角度解释压力。其核心直觉是：

- 如果某个小体积左边压力大、右边压力小；
- 那么就会受到一个从高压到低压的净力；
- 这个力的密度正比于 $-\nabla p$。fileciteturn8file0

所以 pressure 并不是额外附会的数学量，而是：

> **用来强制不可压缩约束成立的反作用力。**

课件附录最后也强调了这一点：pressure is the Lagrange multiplier of the incompressibility constraint。fileciteturn8file0

这句话非常重要，因为它解释了压力在数值求解中的角色：

- 我们并不是先知道压力，再算速度；
- 而是先有一个不满足约束的速度；
- 再通过解一个压力泊松方程，找到恰好能把速度“投影回零散度空间”的压力。

---

## 9. Navier–Stokes 与边界条件：这门课故意省略了什么？

课件第 23 页顺便提到，如果加上黏性项，就得到著名的 Navier–Stokes 方程：

$
\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla \mathbf{u} + \frac{1}{\rho}\nabla p = \mathbf{g} + \nu \Delta \mathbf{u}.
$

不过课件说明，在图形学里这个项常被忽略，因为数值耗散已经会带来类似的平滑作用。fileciteturn8file0

这说明本讲选择的是一个经典的“graphics-style stable fluids”框架：

- 追求稳定、简单、可实现；
- 不追求高度物理真实性。

### 9.1 边界条件

课件第 24–25 页还提到边界条件。除自由表面的 $p=0$ 外，还有典型的 no-separation boundary：

$
\mathbf{u}\cdot\mathbf{n}=\mathbf{u}_{\text{solid}}\cdot\mathbf{n}.
$

这表示流体在法向上不能穿过固体边界；但若不再加切向约束，则对应的是 **free-slip** 边界。fileciteturn8file0

也就是说：

- 法向速度要匹配固体；
- 切向速度则可以自由滑动。

这也是图形学里很常见的边界处理方式。

---

## 10. 为什么要用 MAC Grid？

从第 26 页开始，课件进入空间离散化。这里最核心的结构就是 **Marker-and-Cell (MAC) grid**。

### 10.1 离散化的目标

课件第 29 页把目标说得很清楚：我们要离散的是两个场：

- 速度场 $\mathbf{u}$；
- 压力场 $p$；

以及它们相关的微分算子：

- 散度 $\nabla\cdot\mathbf{u}$；
- 压力梯度 $\nabla p$。fileciteturn8file0

问题在于：

> 如果你把所有量都放在同一个网格位置上，很多离散算子会变得不自然，甚至数值上不稳定。

所以 MAC grid 的核心思想是：

- **压力存储在 cell center**；
- **速度分量存储在边/面中心**。

### 10.2 Staggered Grid 的具体存储方式

课件第 30–39 页详细说明了二维 MAC grid：

- $u$（x 方向速度）存储在竖直边中心；
- $v$（y 方向速度）存储在水平边中心；
- $p$ 存储在网格单元中心。fileciteturn8file0

这种 staggered arrangement 的好处非常大：

1. **散度在 cell center 上自然计算**
   因为 cell 左右两侧正好有 $u$，上下两侧正好有 $v$。

2. **压力梯度在边中心上自然计算**
   因为边两侧正好各有一个压力值，可以做中心差分。

3. **减少压力-速度解耦问题**
   如果压力和速度都放在 cell center，容易出现 checkerboard 等伪解。

所以 MAC grid 不是一个“随便选的存储方式”，而是专门为不可压缩流体离散设计的结构。

### 10.3 MAC grid 上如何求值？

课件第 31–37 页还说明：

- 在任意位置求 $u(x)$、$v(x)$、$p(x)$ 时，用双线性插值；
- 在 cell center 求速度散度时，用中心差分；
- 在 edge center 求压力梯度时，也用中心差分。fileciteturn8file0

这一步是后面 advection 和 projection 的基础，因为回溯点一般不会刚好落在网格采样点上，所以必须支持任意位置插值。

---

## 11. Time Stepping：Stable Fluids 的 splitting scheme

课件第 40–43 页开始讲时间推进。这里采用的是经典 **operator splitting** 思想。

把 Euler 方程改写成：

$
\frac{\partial \mathbf{u}}{\partial t} = -\mathbf{u}\cdot\nabla \mathbf{u} + \mathbf{g} - \frac{1}{\rho}\nabla p.
$

然后把右边拆成三类“力”：

1. **Advection**：$-\mathbf{u}\cdot\nabla\mathbf{u}$
2. **External force**：$\mathbf{g}$
3. **Projection / pressure**：$-\frac{1}{\rho}\nabla p$

于是一个时间步被拆成三步：

1. Advect velocity
2. Apply external forces
3. Project velocity to incompressible field

课件强调，这样做的动机是：

> 每个子问题都比原始耦合 PDE 更容易求解；当时间步长 $h \to 0$ 时，这种 sequential splitting 不会破坏正确性和收敛性。fileciteturn8file0

### 11.1 为什么 splitting 这么常见？

因为完整的流体方程耦合很强：

- advection 本身就非线性；
- 外力项简单但要和别的项耦合；
- 压力项又要负责 enforcing incompressibility。

如果硬要一步同时解所有项，会很麻烦；而 splitting 之后：

- advection 可以单独处理；
- force 直接显式更新；
- projection 则变成一个椭圆型泊松方程。

这正是 Stable Fluids 成为经典入门框架的原因：

> **它把一个复杂流体方程拆成几个各自可控、各自稳定的子问题。**

---

## 12. Semi-Lagrangian Advection：为什么它又稳又便宜？

课件第 44–55 页专门讲 advection，这里使用的是 **semi-Lagrangian advection**。

### 12.1 为什么不用普通显式积分？

课件第 46–47 页先说明：

- 显式时间积分很便宜，但不稳定；
- 隐式积分稳定，但昂贵；
- semi-Lagrangian 是一个专门针对 advection 设计的折中方案：**便宜且无条件稳定**。fileciteturn8file0

### 12.2 核心思路：沿特征线反向追踪

对于 advection 方程

$
\frac{Dq}{Dt}=0,
$

它的物理意思是：

> 粒子携带的量 $q$ 沿着流线保持不变。

如果正向理解，就是：粒子把自己的 $q$ 搬到新位置去。

但 semi-Lagrangian 采用的是反向思路：

> **为了计算新时刻某个网格点 $x$ 上的值，不去问“谁会流到哪”，而去问“现在在 $x$ 的流体，是从哪里来的”。**

于是算法变成：

1. 对于新网格位置 $x$，用当前速度场回溯：
   $
   x_b = x - h\,u(x)
   $
2. 在旧场上查询 $q(x_b)$；
3. 把这个值复制到新场中的 $x$。

这就是 semi-Lagrangian 的本质。

### 12.3 为什么它无条件稳定？

课件第 51 页给出一个很漂亮的性质：更新后的场范数不会超过原来的场范数，因此它是 unconditionally stable。fileciteturn8file0

直观理解是：

- 新值是从旧场里插值得到；
- 插值本质上不会凭空放大极值；
- 所以不会像显式迎风差分那样轻易炸掉。

当然，它虽然稳定，但并不代表精确。课件第 53 页明确提醒：

- 时间步 $h$ 越大，回溯误差越大；
- 稳定不等于高精度；
- 经验上常要求回溯位移别超过几个网格尺寸。fileciteturn8file0

这也是 semi-Lagrangian 的经典代价：

> **它用数值耗散换来了稳定性。**

### 12.4 用在速度场上会有什么特别之处？

课件第 52 页提到，把 $q$ 换成速度 $u$ 之后，其实会涉及 velocity transport 的几何问题，并引用了 covector fluids 作为更严格的处理。fileciteturn8file0

不过在本讲的 stable fluids 框架里，主要思路还是：

- 在 MAC grid 上，对每个速度采样点回溯；
- 用插值得到旧场中的速度；
- 把对应分量拷回新网格。

这已经能工作，而且实现简单。

### 12.5 边界之外怎么办？

课件第 54 页说明，如果回溯点跑出了流体区域，一个基本策略是：

- 取离流体域最近的点；
- 用那个点的速度。fileciteturn8file0

更高级的方法可以做 extrapolation，但本讲先不展开。

### 12.6 Semi-Lagrangian 的完整流程

课件第 55 页给出了 advection 子步骤的完整算法摘要：

- 输入：MAC grid 上的速度场和时间步长；
- 对每个存储速度分量的位置：
  - 插值得到当前位置速度；
  - 回溯到旧位置；
  - 若越界则投影回流体域；
  - 再插值得到旧位置速度，并把需要的分量写到新网格。fileciteturn8file0

这一页实际上已经非常接近可以直接写代码的伪代码了。

---

## 13. Applying Forces：最简单的一步

课件第 56–57 页讲外力项：

$
\frac{\partial \mathbf{u}}{\partial t} = \mathbf{g}.
$

因为 $g$ 是常量源项，所以离散后直接得到：

$
\mathbf{u} \leftarrow \mathbf{u} + h\mathbf{g}.
$

这一步非常简单，但意义上要分清：

- advection 是“速度场自己搬运自己”；
- force 是“外部世界在推速度场”。

在很多图形学场景里，最常见的外力就是 gravity。

---

## 14. Projection：为什么压力求解是整个算法的核心？

从第 58 页开始，课件进入 projection，也就是整个 stable fluids 里最核心的一步。

它要解决的问题是：

> advection 和 external forces 之后得到的速度场，通常已经不再满足 $\nabla\cdot u = 0$。如何把它修正回不可压缩空间？

这一步对应的子问题是：

$
\frac{\partial \mathbf{u}}{\partial t} = -\frac{1}{\rho}\nabla p,
\qquad
\text{s.t. } \nabla\cdot\mathbf{u}=0
$

也就是说：

- 通过施加某个压力梯度；
- 恰好把速度场修正到零散度；
- 同时满足边界条件。

### 14.1 为什么叫 projection？

因为从线性代数角度，这一步本质上是在做：

- 给定一个任意速度场；
- 把它投影到“散度为零的子空间”。

压力就是这个投影操作所对应的拉格朗日乘子。

这也是为什么很多流体模拟器里，**解压力泊松方程**是整个计算成本最高的一步。

---

## 15. Projection 的离散推导：先把新速度写成压力的函数

课件第 60–65 页先处理边界与速度更新公式。

假设我们已经有 advection + force 后的速度，想得到投影后的 $u^{n+1}$。

对 MAC grid 上的边速度，课件给出类似如下离散式：

$
u^{n+1}_{i+1/2,j} = u_{i+1/2,j} - \frac{h}{\rho\Delta x}(p_{i+1,j} - p_{i,j}).
$

对 $v$ 分量同理。fileciteturn8file0

这一步的关键思想是：

> **投影后的速度不是直接求，而是先表示成压力未知量的函数。**

于是接下来只要利用不可压缩约束，就能把问题化成只含压力的方程。

### 15.1 特殊边界情况

课件特别区分了两种边界：

#### 流体-空气边界

若相邻单元是空气，则采用自由表面条件 $p=0$。于是边上的压力差中，空气侧压力直接取零。fileciteturn8file0

#### 流体-固体边界

若相邻单元是固体，我们没有那个固体单元的压力值，但我们知道边界速度必须满足固体速度条件。因此可以“虚构”一个 ghost pressure，使更新后的边速度达到目标边界速度。fileciteturn8file0

这一步的本质是：

- 不规则区域外的量虽然没有真实定义；
- 但我们可以通过边界条件反推出一个“等效值”；
- 让离散方程仍然成立。

---

## 16. 施加不可压缩约束：得到压力泊松方程

课件第 66–72 页是 projection 的核心推导。

既然新速度已经写成压力的函数，那么把它代入离散散度约束：

$
\nabla\cdot \mathbf{u}^{n+1} = 0
$

就得到一个只关于压力的线性系统。

在一般流体单元情况下，最终得到的离散形式本质上就是一个泊松方程：

$
\frac{h}{\rho \Delta x^2}(4p_{i,j} - p_{i+1,j} - p_{i-1,j} - p_{i,j+1} - p_{i,j-1})
= -\frac{1}{\Delta x}(u_{i+1/2,j}-u_{i-1/2,j}+v_{i,j+1/2}-v_{i,j-1/2}).
$

课件第 69 页明确指出，这正是

$
\frac{h}{\rho}\Delta p = \nabla\cdot \mathbf{u}
$

的离散化。fileciteturn8file0

### 16.1 为什么右边是原速度场的散度？

因为 projection 的目的，就是消除当前速度中的散度。右边若为零，说明本来就不可压缩；右边越大，说明当前局部“流入流出不平衡”越严重，需要更大的压力修正。

### 16.2 为什么不同边界会改系数？

课件第 70–71 页说明：

- 若邻居是空气，直接把空气侧压力替换为 0；
- 若邻居是固体，则由于边界速度条件，系数结构会发生变化。fileciteturn8file0

这说明压力矩阵的具体形式依赖于：

- 哪些单元是 fluid；
- 哪些邻居是 air；
- 哪些邻居是 solid。

所以 projection 其实不只是“解一个普通五点 Laplacian”，而是：

> **在边界条件约束下解一个变系数的离散泊松问题。**

---

## 17. 线性系统求解：为什么要用 PCG？

课件第 72–73 页把压力求解写成矩阵形式：

$
A p = b.
$

并指出：

- $A$ 是稀疏的；
- 是对称的；
- 在合适边界条件下是正定的。fileciteturn8file0

这种系统最适合用迭代法求解，而课件推荐的是 **Preconditioned Conjugate Gradient (PCG)**。

### 17.1 为什么不是直接高斯消元？

因为真实流体网格很大：

- 2D 已经可能有几十万未知量；
- 3D 会更多；
- 稀疏矩阵直接做稠密消元非常浪费。

而 CG / PCG 只需要矩阵-向量乘法，因此：

- 更适合稀疏系统；
- 更省内存；
- 在 Poisson 型问题上很常见。

课件还提到更高级的方法，例如 MICCG(0) 和 multigrid。fileciteturn8file0

这也说明：

> **流体模拟里最大的数值瓶颈，往往不是 advection，而是椭圆方程求解。**

---

## 18. Stable Fluids 的完整时间推进流程

课件第 74 页给出了完整的稳定流体算法摘要。可以概括为：

1. 根据 CFL 风格约束确定当前子步长 $h$；
2. Advect velocity；
3. Apply external force；
4. Project velocity。fileciteturn8file0

这个流程非常经典，也是很多入门流体模拟代码的骨架。

如果把它翻译成更容易博客读者理解的话，可以写成：

> 每一步先让流体按当前速度把自己搬运一下，再加上重力等外力，最后通过求压力把速度场“压回”不可压缩状态。

这三个步骤循环往复，就能得到一个稳定、看起来合理的流体动画。

---

## 19. Appendix：Euler Equation 从哪里来？

课件后面的附录第 80–93 页给出了 Euler 方程的变分推导，使用的是：

- D’Alembert principle；
- Hamilton’s principle；
- Lagrange multiplier。

这部分的重点不一定是每步公式，而是最后的物理意义：

> **压力是不可压缩约束的拉格朗日乘子。**

课件从流体的作用量出发：

$
A = \int \int_\Omega \left( \frac12 \rho |u|^2 + \rho g\cdot x \right) dX dt,
$

并加上约束 $\det F = 1$，引入乘子 $p(X,t)$。最后经过变分与分部积分，得到

$
\rho \dot{u} + \nabla_X p \cdot F^{-T} = \rho g,
$

再切换到 Eulerian 视角，变成我们熟悉的 Euler equations。fileciteturn8file0

这部分的价值是：它让你明白 pressure 并不是凭空加进去的，而是数学上自然出现的约束反作用力。

---

## 20. 不规则边界：为什么轴对齐网格不够？

从第 94 页开始，课件进入 irregular boundary conditions。

前面的 projection 推导默认边界与网格轴对齐，因此：

- fluid / solid 划分很整齐；
- 边界恰好落在网格边上；
- 压力和速度离散比较简单。

但现实中，自由表面和固体边界往往都是不规则的，例如玻璃杯中的水。课件第 95–96 页用一张杯中水的图来提醒：

> 真实边界往往不是网格对齐的，必须引入更一般的几何表示。fileciteturn8file0

于是问题变成：

- 怎么表示这条不断变化的自由表面？
- 怎么在投影步骤中处理不规则边界？

---

## 21. Level Set：为什么它适合表示自由表面？

课件第 97–109 页引入了 **level set** 表示。

定义一个标量函数 $\phi(x)$，使得：

- $\phi(x)=0$：边界；
- $\phi(x)<0$：流体内部；
- $\phi(x)>0$：流体外部。fileciteturn8file0

这就把“复杂曲面边界”变成了“一个标量场的零水平集”。

### 21.1 Signed Distance Function

课件进一步指出，一个很好的选择是 **signed distance function (SDF)**：

$
\phi(x) = \pm \min_{y\in \partial\Omega} |x-y|.
$

它的优点是：

- 数值上直观；
- 查询一个点离边界多远很方便；
- 法向方向可由 $\nabla\phi$ 给出；
- 适合做投影、重建和 advect。

课件第 100 页还指出 SDF 满足 Eikonal equation：

$
|\nabla \phi| = 1
$

（除了奇异点外）。fileciteturn8file0

### 21.2 Level Set 能支持哪些操作？

课件第 109 页总结得很好：level set 支持

- 从 mesh 转成 SDF；
- 从 SDF 重建 mesh；
- 查询点到表面的 signed distance；
- 把点投影到最近表面点；
- 随速度场 advect。fileciteturn8file0

这正是它在自由表面流体里如此常见的原因。

### 21.3 Advecting a Level Set

课件第 108 页指出，level set 本身也像其他 quantity 一样被流体携带：

$
\frac{D\phi}{Dt} = 0.
$

也就是说，你可以用和 advection 类似的方法来推进自由表面。

这就让整个系统形成闭环：

- 速度场推动 level set；
- level set 定义流体区域；
- 流体区域又影响下一步 projection。

---

## 22. Irregular Boundary Projection：哪两件事需要修？

课件第 111 页说得很明确：不规则边界下，projection 主要要修两件事：

1. 如何处理自由表面条件 $p=0$；
2. 如何处理固体边界速度条件 $u\cdot n = u_{solid}\cdot n$。fileciteturn8file0

### 22.1 Free surface：引入 $\theta$

在规则网格下，压力梯度是用 $\Delta x$ 近似的；但如果边界落在单元内部，真实距离不再是整整一个 $\Delta x$，而是 $\theta \Delta x$，其中 $\theta \in [0,1]$ 表示边界在边上的相对位置。课件第 112–113 页用这个想法修正了自由表面处的压力梯度。fileciteturn8file0

这说明：

> irregular boundary 的关键不是“重新发明方程”，而是“修正离散几何比例”。

### 22.2 Solid velocity：引入 ghost velocity

对于不规则固体边界，课件第 114–115 页说明，需要构造一个 **ghost velocity**，把流体侧速度外推到边界另一侧，从而在散度计算中正确反映固体速度约束。fileciteturn8file0

这和前面 axis-aligned 情况中的“ghost pressure”思想是同类问题：

- 边界另一侧的量没有真实物理定义；
- 但为了保持离散公式统一，需要构造一个等效值；
- 使边界条件在离散层面成立。

---

## 23. 最终完整算法：自由表面 Stable Fluids

课件最后第 116 页给出了自由表面版本的完整时间推进流程：

1. Extrapolate velocity field；
2. Advect the level set，并可选择 reinitialize；
3. Advect the velocity field；
4. Apply external forces；
5. Project the velocity。fileciteturn8file0

这和前面的“基本 stable fluids”相比，多了两步关键几何处理：

- velocity extrapolation；
- level set advection / reinitialization。

这说明当自由表面进入系统后，求解器不再只是“解速度”，还必须同步维护流体区域的几何表示。

---

## 24. 这份课件的真正主线

如果把整份课件压缩成一条非常清晰的知识链，大概是这样的：

### 24.1 第一步：先从连续体运动学理解流体

- 用 $x=\phi(X,t)$ 表示粒子运动；
- 用 $\det F = 1$ 表示不可压缩；
- 再推到速度约束。

### 24.2 第二步：切换到 Eulerian PDE 形式

- 引入 material derivative；
- 得到 $\nabla\cdot u = 0$；
- 写出 Euler equations。

### 24.3 第三步：把方程离散到 MAC grid 上

- pressure 放 cell center；
- velocity components 放 edge/face center；
- 这样散度和梯度离散都自然。

### 24.4 第四步：用 splitting 构造稳定时间推进

- advection；
- forces；
- projection。

### 24.5 第五步：projection 的本质是解压力泊松方程

- 先把新速度写成压力函数；
- 再用零散度约束推导出 $Ap=b$；
- 用 PCG 等方法求解。

### 24.6 第六步：真实自由表面需要几何表示

- 用 level set 表示流体区域；
- 用 irregular boundary correction 修正 projection；
- 得到更真实的自由表面流体算法。

所以，这份课件真正想教会你的不是“一个技巧”，而是：

> **如何把不可压缩流体的连续理论，一步一步转成一个真正可运行的数值模拟框架。**

---

## 25. 最值得记住的几个核心结论

### 25.1 不可压缩流体最核心的约束是 $\nabla\cdot u = 0$

这不是附加修饰，而是整个压力求解和投影法的中心。

### 25.2 Material derivative 是流体 PDE 的基础语言

$\partial_t + u\cdot\nabla$ 表示“跟着粒子走”的变化率，是 advection 的本质来源。

### 25.3 MAC grid 不是随便选的离散格式

它专门为不可压缩流体设计，使 pressure gradient 和 velocity divergence 的离散都很自然。

### 25.4 Stable Fluids 的核心不是“逼真”，而是“稳定且易实现”

semi-Lagrangian advection 牺牲了一些精度，换来了强稳定性。

### 25.5 Projection 是整个算法的数学核心

它通过解一个泊松方程，把速度场重新拉回不可压缩空间。

### 25.6 自由表面问题本质上是“流体动力学 + 边界几何表示”的耦合

level set 让我们能够处理时间变化的复杂边界，而不仅仅是规则盒子中的流体。

---

## 26. 一句话总结

如果把这份 lecture notes 压缩成一句话，可以写成：

> **Fluid Simulation I 的核心，是从不可压缩无粘流体的连续方程出发，借助 MAC 网格、semi-Lagrangian advection 和 pressure projection，把流体动力学转化为一个稳定、可实现，并且能逐步扩展到自由表面与不规则边界的数值模拟框架。**

