可以把 **SMPL、LBS、糖纸伪影（candy-wrapper artifact）** 看成一条完整的技术链：

> **SMPL 负责定义“这个人长什么样、摆什么姿势” → 骨架给出各关节的刚体变换 → LBS 把骨架变换传播到人体表面 → 但 LBS 对旋转做线性平均会产生非刚性缩放/剪切 → 极端扭转时就出现糖纸伪影。**

真正做相关研究时，最重要的不是背 SMPL 公式，而是理解其中有三层完全不同的东西：**统计形状模型、运动学骨架、蒙皮变形算子**。SMPL 把这三者组合成了一个可微分的人体生成函数。

![Image](https://images.openai.com/static-rsc-4/TGKzUBudYDej726qafCnrj3U7HHZBCVwNp1AK4yDN4-5o0n6TGjAf4865Ebzn5cabHvgDGzVxCg5vN5aKmIndtsLOuo4M-T3aqbwxPnANqa49ItLMzZKYUfq2XYrqaRKLjSQpEILbc0A1Oz_Hom_Exs0m4FSZm8IGwpX88x0pWVUriwdC7d3l1p4LJAOc0fT?purpose=fullsize)

![Image](https://caterpillarstudygroup.github.io/GAMES105_mdbook/assets/07-14.png)

经典 SMPL 是 Loper 等人在 2015 年提出的。它将人体表示为一个固定拓扑的三角网格，并学习 shape blend shapes、pose blend shapes、joint regressor 和 skinning weights。经典模型有 6890 个顶点、13776 个三角面、24 个运动学关节。([TU Eindhoven Files][1])

---

# 一、先建立一个最重要的整体认识：SMPL 到底是什么？

很多人第一次学 SMPL 会把它理解成：

> 输入姿势 → 输出人体 mesh。

这太粗糙。

更准确地说，SMPL 是一个参数化函数：

$$
M(\beta,\theta)
\longrightarrow
V\in \mathbb R^{6890\times3}
$$

其中：

$$
\beta = \text{人体身份/体型参数}
$$

$$
\theta = \text{人体姿势参数}
$$

最终得到所有人体顶点。

论文中的核心形式可以写成：

$$
\boxed{
M(\beta,\theta)
=
W\left(
T_P(\beta,\theta),
J(\beta),
\theta,
\mathcal W
\right)
}
$$

而：

$$
\boxed{
T_P(\beta,\theta)
=
\bar T+B_S(\beta)+B_P(\theta)
}
$$

这里其实出现了 SMPL 最重要的几个组件：

| 符号              | 含义                    |
| --------------- | --------------------- |
| \(\bar T\)      | 平均人体模板                |
| \(B_S(\beta)\)  | 身份/体型带来的形变            |
| \(B_P(\theta)\) | 姿势导致的局部形变修正           |
| \(J(\beta)\)    | 当前体型对应的骨架关节位置         |
| \(\mathcal W\)  | skinning weights      |
| \(W(\cdot)\)    | Linear Blend Skinning |

所以整个流程不是简单的：

$$
\text{pose}\rightarrow mesh
$$

而是：

$$
\beta
\rightarrow
\text{shaped body}
$$

同时：

$$
\beta
\rightarrow
\text{skeleton}
$$

然后：

$$
\theta
\rightarrow
\text{pose corrective deformation}
$$

最后：

$$
\text{skeleton transforms}
+
\text{skinning weights}
\rightarrow
\text{posed mesh}
$$

这几个步骤分清楚以后，SMPL 基本就通了。

---

# 二、为什么 SMPL 必须有一个固定 template？

SMPL 首先定义一个 canonical/rest pose 的人体：

$$
\bar T
$$

一般就是熟悉的近似 T-pose。

它由：

$$
N=6890
$$

个顶点组成：

$$
\bar T=
\begin{bmatrix}
x_1&y_1&z_1\\
x_2&y_2&z_2\\
\vdots\\
x_{6890}&y_{6890}&z_{6890}
\end{bmatrix}
$$

一个特别重要的性质是：

> **所有 SMPL 人体都具有完全相同的 topology 和 vertex correspondence。**

比如：

```text
vertex 3500
```

在所有 SMPL 人体上大致都对应同一个语义区域。

因此不同人：

$$
V^{person1}_{3500}
$$

和：

$$
V^{person2}_{3500}
$$

有明确的 correspondence。

这也是 SMPL 对很多研究特别方便的原因。

你可以做：

$$
V_A-V_B
$$

因为顶点一一对应。

---

# 三、Shape：SMPL 怎么表示“胖瘦高矮”？

SMPL 不直接用：

```text
height = 180
weight = 75
waist = 80
```

这种显式人体测量。

它使用一个低维 latent：

$$
\beta=
[\beta_1,\beta_2,\dots,\beta_B]
$$

经典研究中非常常见的是：

$$
B=10
$$

不过现在官方模型可以提供更多 shape components，官方当前下载版本可达到 300 个 shape components。([SMPL][2])

Shape deformation 是：

$$
\boxed{
B_S(\beta)
=
\sum_{n=1}^{B}\beta_n S_n
}
$$

其中：

$$
S_n\in\mathbb R^{6890\times3}
$$

就是第 \(n\) 个 shape basis。

于是：

$$
T_S(\beta)
=
\bar T+B_S(\beta)
$$

---

## 一个非常好的理解方式

假设只有两个 shape basis：

$$
S_1=\text{高矮变化方向}
$$

$$
S_2=\text{胖瘦变化方向}
$$

那么：

$$
\beta_1=2
$$

就会沿着 \(S_1\) 方向移动所有人体顶点：

$$
v_i'
=
v_i+\beta_1S_{1,i}
$$

实际 SMPL 的 PCA basis 没有这么干净。

也就是说：

> \(\beta_1\) 不严格等于“胖瘦”，\(\beta_2\) 也不严格等于“高矮”。

它们是从人体扫描数据学习出来的主要变化方向。

---

# 四、为什么 Skeleton 也必须跟着 \(\beta\) 变化？

这是一个非常关键、但经常被初学者忽略的问题。

一个 1.5 米的人和一个 2 米的人：

```text
hip → knee
```

距离显然不一样。

所以不可能所有人共享同一套固定 joint 坐标。

SMPL 从 shaped mesh 中回归 joints：

$$
\boxed{
J(\beta)
=
\mathcal J
\left(
\bar T+B_S(\beta)
\right)
}
$$

其中：

$$
\mathcal J
$$

是一个 joint regressor。

经典 SMPL 中：

$$
\mathcal J\in\mathbb R^{24\times6890}
$$

所以每一个 joint 本质上都是若干 mesh vertices 的加权组合。公开实现通常就是一个 \(24\times6890\) 的 `J_regressor`。([GitHub][3])

例如非常简化地：

$$
J_{\text{knee}}
=
0.2v_{100}
+
0.3v_{101}
+
0.5v_{102}
$$

当然真实 regressor 会涉及更多顶点。

一个研究中非常重要的认识是：

> **SMPL joints 不等于某几个 mesh vertex。**

joint 通常位于身体内部。

---

# 五、Pose：\(\theta\) 到底是什么？

经典 SMPL 有：

$$
24
$$

个运动学 joint。

每个 joint 通常用一个三维 axis-angle 表示：

$$
\omega_k\in\mathbb R^3
$$

所以：

$$
\theta\in\mathbb R^{24\times3}
=
\mathbb R^{72}
$$

例如：

$$
\omega=
\begin{bmatrix}
0.3\\
0\\
0
\end{bmatrix}
$$

表示：

* rotation axis：x 轴
* rotation angle：0.3 rad

因为：

$$
\|\omega\|=0.3
$$

而：

$$
\frac{\omega}{\|\omega\|}
$$

就是旋转轴。

然后通过 exponential map / Rodrigues formula：

$$
R_k
=
\exp([\omega_k]_\times)
$$

得到：

$$
R_k\in SO(3)
$$

即真正的 rotation matrix。

SMPL 原始论文的 Rodrigues 公式后来还有官方 errata，正确形式中二阶项是：

$$
(1-\cos\|\omega\|)
$$

而不是单纯的 \(\cos\|\omega\|\)。([TU Eindhoven Files][4])

---

# 六、24 个 joint 为什么经常又看到“23 joints”？

这个地方很容易把人搞晕。

实际上可以这样理解：

$$
24=1+23
$$

其中：

```text
1 个 root/pelvis
+
23 个相对于 parent 的 articulated joints
```

root rotation：

$$
R_0
$$

通常表示整个人的 global orientation。

其余：

$$
R_1,\dots,R_{23}
$$

表示身体内部 articulation。

所以你会看到：

$$
\theta\in R^{72}
$$

但 pose corrective feature 却是：

$$
23\times9=207
$$

这两个数字并不矛盾。

---

# 七、为什么“骨头转了”，还不够？

现在进入理解 SMPL 的一个关键地方。

假设你只有一个普通 skeleton + LBS。

手臂弯曲：

$$
0^\circ\rightarrow90^\circ
$$

人体真实的肘部并不是简单的：

> 一堆顶点随着两根骨头转。

真实人体还会出现：

* 肌肉挤压
* 皮肤拉伸
* 肘窝凹陷
* 关节附近体积变化
* 肩膀形状改变

纯 LBS 无法表示这些。

因此 SMPL 引入：

$$
\boxed{
B_P(\theta)
}
$$

即 **pose-dependent blend shapes**。

它可以理解为：

> “在真正做 LBS 之前，先根据当前姿势把 mesh 修一下。”

---

# 八、Pose Blend Shapes 是 SMPL 非常漂亮的一部分

经典 SMPL 使用：

$$
R_k(\theta)-I
$$

作为 pose feature。

把 23 个关节的 rotation matrix 展平：

$$
R_k-I
\in \mathbb R^{3\times3}
$$

所以每个 joint：

$$
9
$$

维。

总共：

$$
23\times9=207
$$

得到：

$$
r(\theta)\in\mathbb R^{207}
$$

然后：

$$
\boxed{
B_P(\theta)
=
P\,r(\theta)
}
$$

这里：

$$
P\in\mathbb R^{20670\times207}
$$

因为：

$$
6890\times3=20670
$$

公开 SMPL 文件里的 `posedirs` 因此通常就是：

$$
6890\times3\times207
$$

这可以在实际 SMPL 实现的数据结构中看到。([GitHub][5])

---

# 九、为什么不用原始 axis-angle，而要用 \(R-I\)？

这是很值得研究者理解的一点。

假设：

$$
\theta=0
$$

也就是 rest pose。

那么：

$$
R=I
$$

因此：

$$
R-I=0
$$

所以自然得到：

$$
B_P(0)=0
$$

非常漂亮。

也就是说：

> canonical pose 下完全不需要 pose corrective。

另一方面，rotation matrix 的元素本身已经包含：

$$
\sin\theta,\cos\theta
$$

这样的非线性关系。

所以虽然：

$$
B_P
$$

对 rotation matrix entries 是线性的：

$$
B_P=P\,r
$$

但它对原始 angle：

$$
\theta
$$

并不是简单线性的。

这比：

$$
B_P(\theta)=P\theta
$$

表达能力要强很多。

SMPL 的一个重要设计正是 pose blend shapes 对 rotation-matrix elements 线性，这也是原论文强调的特点。([TU Eindhoven Files][1])

---

# 十、所以到真正蒙皮之前，人体已经变成什么了？

现在有：

$$
\bar T
$$

加上 identity deformation：

$$
B_S(\beta)
$$

再加：

$$
B_P(\theta)
$$

得到：

$$
\boxed{
T_P(\beta,\theta)
=
\bar T+
B_S(\beta)+
B_P(\theta)
}
$$

这叫 posed template。

注意：

> **这里还没有真的把手臂旋转到 90°。**

它只是：

> 为接下来的骨骼 deformation 做好局部几何 correction。

然后才轮到 LBS。

这件事很重要：

$$
\boxed{
\text{Pose corrective}
\quad\text{发生在}\quad
\text{LBS之前}
}
$$

而不是 LBS 后再修。

---

# 十一、现在进入 LBS：蒙皮到底是在解决什么问题？

假设 skeleton 已经有：

```text
pelvis
spine
shoulder
elbow
wrist
...
```

如果一个 mesh vertex：

$$
v_i
$$

位于前臂中间，它可能主要受 elbow/wrist bone 控制。

如果在上臂中间：

可能主要受：

```text
shoulder
elbow
```

控制。

于是每个 vertex 有：

$$
w_{ik}
$$

表示：

> vertex \(i\) 被 joint/bone \(k\) 影响多少。

通常：

$$
w_{ik}\ge0
$$

以及：

$$
\boxed{
\sum_k w_{ik}=1
}
$$

SMPL 的 weight matrix：

$$
\mathcal W\in\mathbb R^{6890\times24}
$$

公开模型也是这个维度。([GitHub][5])

---

# 十二、如果只有一个 bone influence，非常简单

假设：

$$
w_{i,k}=1
$$

那么：

$$
v_i'=A_kv_i
$$

其中：

$$
A_k\in SE(3)
$$

是骨头的 rigid transform。

齐次坐标写成：

$$
\tilde v=
\begin{bmatrix}
x\\y\\z\\1
\end{bmatrix}
$$

那么：

$$
A=
\begin{bmatrix}
R&t\\
0&1
\end{bmatrix}
$$

所以：

$$
\tilde v'=A\tilde v
$$

这就是普通刚体运动：

$$
v'=Rv+t
$$

完全不会让物体发生缩放或剪切。

---

# 十三、问题来了：关节附近不能只属于一根骨头

假设 elbow 附近：

$$
w_{\text{upper arm}}=0.5
$$

$$
w_{\text{forearm}}=0.5
$$

如果突然从：

```text
100% upper arm
```

切到：

```text
100% forearm
```

mesh 会出现 discontinuity。

所以要 smooth blending：

$$
\boxed{
v_i'
=
\sum_k
w_{ik}A_k\tilde v_i
}
$$

这就是：

# Linear Blend Skinning

简称：

$$
\boxed{LBS}
$$

也常叫：

```text
skeletal subspace deformation
```

---

# 十四、“Linear” 到底 linear 在哪里？

因为：

$$
v_i'
=
\sum_k
w_{ik}
A_k
v_i
$$

利用矩阵分配律：

$$
v_i'
=
\left(
\sum_k
w_{ik}A_k
\right)
v_i
$$

也就是：

$$
\boxed{
A_i^{blend}
=
\sum_k
w_{ik}A_k
}
$$

然后：

$$
v_i'
=
A_i^{blend}v_i
$$

所以它其实在做一件非常直接的事：

> **把多个 transformation matrix 直接做 weighted average。**

也正因为这样：

* 实现极其简单
* GPU 极快
* 完全可微
* 所有游戏引擎都很好支持

SMPL 选择这种传统 graphics pipeline，就是为了兼容工业 animation/rendering pipeline。([TU Eindhoven Files][1])

但也正是这个：

$$
\sum w_kA_k
$$

埋下了糖纸伪影的问题。

---

# 十五、先补一个非常关键的东西：骨头不是绕世界原点旋转

假设 elbow 在：

$$
J=
\begin{bmatrix}
1\\2\\3
\end{bmatrix}
$$

你想让手臂绕 elbow 转。

正确操作不是：

$$
Rv
$$

而是：

$$
\boxed{
R(v-J)+J
}
$$

展开：

$$
Rv-RJ+J
$$

所以：

$$
v'=Rv+(J-RJ)
$$

相当于 translation：

$$
t=J-RJ
$$

因此骨骼 transform 本身已经把“rotation center 是 joint”编码进去了。

---

# 十六、SMPL 的骨架是一个 kinematic tree

比如概念上：

```text
pelvis
├── left hip
│   └── left knee
│       └── left ankle
├── right hip
│   └── right knee
│       └── right ankle
└── spine
    └── chest
        ├── left shoulder
        │   └── elbow
        │       └── wrist
        └── right shoulder
            └── elbow
                └── wrist
```

因此 elbow rotation 是：

> 相对于 shoulder coordinate frame。

wrist rotation：

> 相对于 elbow coordinate frame。

所以全局 transform 是递归得到：

$$
G_k
=
G_{\text{parent}(k)}
G_k^{local}
$$

其中：

$$
G_k^{local}
=
\begin{bmatrix}
R_k&
J_k-J_{\text{parent}(k)}
\\
0&1
\end{bmatrix}
$$

---

# 十七、这里还有一个经常被忽略的 bind pose inverse

如果你直接：

$$
G_kv
$$

会有问题。

因为：

$$
G_k
$$

同时包含：

```text
骨架本来就在这个位置
+
当前 pose rotation
```

但 mesh 本来就是按照 rest skeleton 建出来的。

所以必须去掉 bind pose transform：

$$
\boxed{
A_k
=
G_k(\theta,J)
G_k(0,J)^{-1}
}
$$

于是 rest pose：

$$
\theta=0
$$

时：

$$
A_k
=
G_k(0)G_k(0)^{-1}
=
I
$$

于是：

$$
v_i'=v_i
$$

这就是经典 skeletal animation 里的：

# inverse bind matrix

理解 LBS 时，这一步非常重要。

---

# 十八、现在终于可以看完整 SMPL forward 了

经典 forward pipeline 可以压缩成：

$$
\boxed{
v_{\text{shaped}}
=
\bar T+B_S(\beta)
}
$$

然后：

$$
\boxed{
J
=
J_{\text{regressor}}
v_{\text{shaped}}
}
$$

姿势转 rotation matrices：

$$
\boxed{
R_k
=
Rodrigues(\theta_k)
}
$$

构造：

$$
\boxed{
p=
vec(R_1-I,\ldots,R_{23}-I)
}
$$

再做 pose correction：

$$
\boxed{
v_{\text{posed}}
=
v_{\text{shaped}}
+
Pp
}
$$

接着 forward kinematics：

$$
\boxed{
A_1,\ldots,A_{24}
=
FK(R,J)
}
$$

最后：

$$
\boxed{
v_i
=
\sum_{k=1}^{24}
w_{ik}
A_k
\tilde v_{\text{posed},i}
}
$$

这就是你在 PyTorch SMPL implementation 中看到的完整数学意义。

经典实现的 tensor 规模通常就是：

$$
v_{\rm template}:6890\times3
$$

$$
shapedirs:6890\times3\times10
$$

$$
posedirs:6890\times3\times207
$$

$$
J_{\rm regressor}:24\times6890
$$

$$
weights:6890\times24
$$

$$
faces:13776\times3
$$

这些都可以在公开实现的数据结构中直接验证。([GitHub][5])

---

# 十九、现在我们进入核心：为什么 LBS 会产生糖纸伪影？

这是我建议你真正推一遍的部分。

根本原因只有一句话：

$$
\boxed{
SO(3)\text{ 不是线性空间}
}
$$

换成人话就是：

> 两个合法的 rotation matrix 做普通线性平均，结果通常不是 rotation matrix。

假设：

$$
R_1,R_2\in SO(3)
$$

那么：

$$
R_1^TR_1=I
$$

$$
R_2^TR_2=I
$$

并且：

$$
\det R_1=\det R_2=1
$$

但：

$$
R=
0.5R_1+0.5R_2
$$

一般：

$$
R^TR\neq I
$$

并且：

$$
\det R\neq1
$$

于是：

$$
R
$$

不再是 rigid rotation。

它可能偷偷包含：

* scaling
* shearing
* volume collapse

这就是所有问题的根源。

---

# 二十、直接用数学把 candy-wrapper 算出来

这比看十张示意图都重要。

假设有一根圆柱形手臂：

```text
──────────────
```

沿 x 轴。

我们只考虑横截面的：

$$
(y,z)
$$

现在圆柱两端的 bones 发生 twist。

假设两个骨骼分别旋转：

$$
+\alpha
$$

和：

$$
-\alpha
$$

在 yz 平面：

$$
R(+\alpha)=
\begin{bmatrix}
\cos\alpha&-\sin\alpha\\
\sin\alpha&\cos\alpha
\end{bmatrix}
$$

另一根骨头：

$$
R(-\alpha)=
\begin{bmatrix}
\cos\alpha&\sin\alpha\\
-\sin\alpha&\cos\alpha
\end{bmatrix}
$$

现在中间 vertex：

$$
w_1=w_2=0.5
$$

LBS 做：

$$
R_{\text{blend}}
=
\frac12R(+\alpha)
+
\frac12R(-\alpha)
$$

代进去：

$$
R_{\text{blend}}
=
\begin{bmatrix}
\cos\alpha&0\\
0&\cos\alpha
\end{bmatrix}
$$

也就是：

$$
\boxed{
R_{\text{blend}}
=
\cos\alpha I
}
$$

注意！

这已经：

# 不是 rotation

而变成：

# scaling

---

# 二十一、糖纸伪影的恐怖地方现在非常明显

原本圆柱半径：

$$
r
$$

现在：

$$
r'
=
r\cos\alpha
$$

而两个骨骼之间的 relative rotation 是：

$$
\phi=2\alpha
$$

因此：

$$
\boxed{
r'
=
r\cos\frac{\phi}{2}
}
$$

这就是 candy wrapper artifact 的数学核心。

比如：

|   两骨骼相对 twist |     中间截面半径 |
| ------------: | ---------: |
|   \(0^\circ\) |  \(1.00r\) |
|  \(60^\circ\) | \(0.866r\) |
|  \(90^\circ\) | \(0.707r\) |
| \(120^\circ\) |   \(0.5r\) |
| \(180^\circ\) |     \(0r\) |

到：

$$
180^\circ
$$

时：

$$
\cos90^\circ=0
$$

于是整个横截面：

$$
(y,z)
\rightarrow
(0,0)
$$

圆柱直接被压成一条线。

这就是：

# Candy-wrapper artifact

像糖果包装纸两端被拧：

```text
=====> <=====
```

中间越来越细，最后夹成一点。

相关 skinning 文献正是把这个现象归因于刚体 transformation 的线性混合不再保持 rigid transformation。([用户计算机科学中心][6])

---

# 二十二、一个更直观的 180° 例子

假设：

$$
R_1=I
$$

另一根骨头绕 x 轴旋转：

$$
180^\circ
$$

于是：

$$
R_2=
\begin{bmatrix}
1&0&0\\
0&-1&0\\
0&0&-1
\end{bmatrix}
$$

权重：

$$
w_1=w_2=0.5
$$

所以：

$$
R_{\text{blend}}
=
\frac12(I+R_2)
$$

得到：

$$
R_{\text{blend}}
=
\begin{bmatrix}
1&0&0\\
0&0&0\\
0&0&0
\end{bmatrix}
$$

因此：

$$
\begin{bmatrix}
x\\y\\z
\end{bmatrix}
\rightarrow
\begin{bmatrix}
x\\0\\0
\end{bmatrix}
$$

整个圆柱：

# 直接塌缩到 x 轴。

而且 determinant：

$$
\det R_{\text{blend}}=0
$$

说明这个 transform 甚至：

> 不可逆。

所以 LBS 极端情况下实际上会制造一个 singular transformation。

---

# 二十三、这不是简单的“权重没调好”

这个认识特别重要。

很多第一次碰 LBS artifact 的人会想：

> 那是不是 skinning weights 学得不好？

不是。

即使：

$$
w_1=w_2=0.5
$$

是最合理的 smooth interpolation，问题仍然存在。

问题本质上在：

$$
\boxed{
w_1R_1+w_2R_2
}
$$

这个操作本身。

rotation 所在的：

$$
SO(3)
$$

是一个 curved manifold。

你不能像普通 vector 那样直接：

$$
0.5R_1+0.5R_2
$$

然后期待它仍然属于：

$$
SO(3)
$$

---

# 二十四、可以把它理解成“弦”和“圆弧”的区别

假设二维单位圆上两个点：

$$
p_1
$$

和：

$$
p_2
$$

如果你线性 interpolate：

$$
p(t)
=
(1-t)p_1+tp_2
$$

得到的是：

# chord

即圆的弦。

它会钻进圆里面。

而正确 rotation interpolation 应该走：

# arc

即圆弧。

所以：

$$
\|p(t)\|<1
$$

也就是出现：

# shrinkage。

LBS 其实就是在 transformation space 里面走“弦”。

这就是为什么 twist 会收缩。

---

# 二十五、这也解释了 elbow 为什么容易瘪

糖纸伪影最典型的是：

# twist

但 LBS 还有另一个经典问题：

# volume loss around bends

例如 elbow：

```text
upper arm
   \
    \
     forearm
```

elbow 附近一个 vertex 同时受：

$$
A_{\text{upper}}
$$

和：

$$
A_{\text{forearm}}
$$

影响。

LBS 得到：

$$
0.5A_1v+0.5A_2v
$$

实际上是两个位置之间的：

# Euclidean straight-line average

而真实皮肤应该：

* 绕 joint 滑动
* 保持一定 volume
* 产生肌肉挤压

所以 mesh 往 joint center 缩。

于是出现：

```text
正常手臂：

===========
      \====

LBS：

======<
       \=====
```

关节附近被 pinch。

SMPL 原论文也指出传统 LBS 会在关节产生典型不真实形变，包括所谓 taffy、bowtie 等效应，因此 SMPL 学习 pose-dependent corrective blend shapes 来补偿这些系统性误差。([TU Eindhoven Files][1])

---

# 二十六、那 Dual Quaternion Skinning 为什么能解决 candy wrapper？

这时候 DQS 的动机就非常自然了。

LBS：

$$
A_{\text{blend}}
=
\sum_kw_kA_k
$$

直接平均 matrix。

但 rigid transformation 属于：

$$
SE(3)
$$

我们真正希望的是：

> interpolate rigid transformations，同时保持结果仍然是 rigid transformation。

Dual Quaternion Skinning 用：

$$
\hat q
=
q_r+\epsilon q_d
$$

表示 rigid transform。

其中：

$$
q_r
$$

是 rotation quaternion。

$$
q_d
$$

编码 translation。

对于：

$$
(R,t)
$$

可以构造：

$$
q_d
=
\frac12
t_q q_r
$$

其中：

$$
t_q=(0,t_x,t_y,t_z)
$$

是 pure quaternion。

---

# 二十七、DQS 怎么 blend？

大致做：

$$
\tilde q
=
\sum_k w_k\hat q_k
$$

再进行归一化：

$$
\boxed{
\hat q
=
\frac{\tilde q}{\|\tilde q\|}
}
$$

然后用：

$$
\hat q
$$

对 vertex 做 rigid transformation。

关键就在：

> normalize 后得到的仍然对应一个 rigid transformation。

不会出现 LBS 那种：

$$
R^TR\neq I
$$

造成的 artificial scale。

因此 candy-wrapper 会极大缓解甚至消失。

Kavan 等提出的 dual-quaternion skinning 正是为了用接近 LBS 的实时开销，避免其 transformation blending 伪影。([用户计算机科学中心][7])

---

# 二十八、但 DQS 不是完美答案

非常重要。

DQS 解决了：

$$
\text{twist shrinkage}
$$

但很容易产生另一种问题：

# bulging

特别是 elbow/knee bending。

为什么？

因为 DQS 太努力保持：

# rigid / volume

但真实人体并不是刚体。

例如 elbow 90°：

真实人体应该发生：

* skin compression
* muscle deformation
* soft tissue movement

如果 rigid interpolation 太强，就可能产生：

```text
elbow 变胖
```

的 bulge。

相关研究比较中也明确展示了 LBS 的 candy-wrapper artifact 与 DQS bending 时的 bulging artifact。([用户计算机科学中心][8])

所以不是：

$$
DQS>LBS
$$

这么简单。

---

# 二十九、这也解释了一个看似矛盾的问题

你可能会问：

> 如果 DQS 比 LBS 几何上合理，为什么 SMPL 还使用 LBS？

这其实特别重要。

因为 SMPL 不是：

```text
裸 LBS
```

而是：

$$
\boxed{
\text{LBS}
+
\text{learned pose corrective blend shapes}
}
$$

也就是说：

> SMPL 知道 LBS 会犯什么错，然后学习 \(B_P(\theta)\) 提前修正 mesh。

例如 LBS 会让 elbow 收缩。

那么：

$$
B_P(\theta)
$$

可以提前把 elbow：

```text
撑胖一点
```

然后经过 LBS：

```text
撑胖
+
LBS收缩
≈
真实人体
```

所以：

$$
B_P
$$

某种意义上就是：

# learned inverse compensation

这其实是 SMPL 很漂亮的地方。

---

# 三十、甚至官方 SMPL 最终主要发布的是 LBS 版本

原论文其实也研究过 dual-quaternion blend skinning 版本。

但是官方 FAQ 明确说明，公开提供的是 LBS SMPL，因为在他们的模型中，**SMPL-LBS 的整体精度比相应 SMPL-DQBS 版本更好，而且 LBS 更简单、更标准**。([SMPL][9])

所以不要得到一个错误结论：

> DQS 数学上更“正确”，所以 SMPL 换成 DQS 一定更好。

并不是。

整个：

$$
B_P
$$

是在某一种 deformation model 下学习出来的。

你突然把：

$$
LBS\rightarrow DQS
$$

等于改变整个 forward model。

原本学习出来 compensating LBS error 的 pose correctives 不一定还适用。

---

# 三十一、现在你可以重新理解 SMPL 的名字

SMPL：

# Skinned Multi-Person Linear Model

里面的每个词其实都有意义。

**Skinned**：人体 mesh 通过 skeletal skinning 运动。

**Multi-Person**：同一个模型通过：

$$
\beta
$$

表示大量不同人体 identity。

**Linear**：很多核心 deformation components 都设计成线性结构：

$$
B_S(\beta)
$$

对 \(\beta\) 线性，

$$
B_P
$$

对 rotation-matrix elements 线性，

LBS：

$$
\sum_kw_kA_k
$$

也是线性 blend。

但注意：

> **整个 SMPL \(M(\beta,\theta)\) 并不是关于 \(\beta,\theta\) 的线性函数。**

因为里面还有：

$$
Rodrigues(\theta)
$$

以及：

$$
FK
$$

以及 shape-dependent joints 等非线性运算。

---

# 三十二、几个做研究时尤其需要牢牢记住的区别

我把最容易混淆的概念压成这张表：

| 概念                     | 它解决的问题                         | 是否物理模型  |
| ---------------------- | ------------------------------ | ------- |
| \(\beta\)              | 人与人之间的 shape variation         | 否       |
| \(\theta\)             | articulated pose               | 否       |
| Joint regressor        | 根据 body shape 找 skeleton       | 否       |
| FK                     | 根据 joint rotations 计算骨架变换      | 否       |
| Skinning weights       | 顶点受哪些 bones 控制                 | 否       |
| LBS                    | skeleton → surface deformation | 否       |
| Pose blend shape       | 修正姿势导致的 surface deformation    | 否       |
| DQS                    | 更合理地插值 rigid transforms        | 否       |
| Soft-tissue simulation | 肌肉/脂肪动态                        | 才逐渐接近物理 |
| Cloth simulation       | 衣服受力/碰撞/褶皱                     | 是物理问题   |

这对你现在做人-衣服、human motion、simulation 一类研究尤其重要：

$$
\boxed{\text{SMPL 本质上是 kinematic/statistical model，不是 physics model}}
$$

---

# 三十三、SMPL 并不知道“力”

这一点一定要非常明确。

如果：

$$
\theta_{\text{elbow}}=90^\circ
$$

SMPL 只知道：

> elbow 转 90°。

它不知道：

* muscle force
* external force
* gravity
* contact
* pressure
* collision force
* inertia

所以两个完全不同的物理场景：

```text
自己抬手
```

和：

```text
别人把你的手推到同样角度
```

只要：

$$
(\beta,\theta)
$$

一样，

标准 SMPL mesh 就基本一样。

这是参数人体模型与真实人体动力学之间非常根本的 gap。

---

# 三十四、SMPL 也没有真正的“皮肤层”

虽然名字叫 skinning，但这里的 skin：

不是 biological skin。

Skinning 在 graphics 中只是：

> 如何让 surface mesh 跟着 skeleton deformation。

SMPL mesh 其实就是一个表面：

$$
\mathcal M=(V,F)
$$

并没有：

```text
skin layer
fat layer
muscle
bone geometry
fascia
```

这样的显式结构。

所以不要把：

# skinning

和：

# skin simulation

搞混。

完全不是一回事。

---

# 三十五、这对你研究服装尤其重要

假设你把一件紧身衣直接绑定到 SMPL skeleton，然后 LBS：

$$
V_{\text{garment}}'
=
LBS(
V_{\text{garment}},
\theta,
W_{\text{garment}}
)
$$

它可以：

* 跟身体动
* 保持大致位置
* 做简单动画

但它不能表示：

* 衣服自然垂下
* cloth inertia
* fold
* flutter
* sliding
* cloth-body friction
* contact
* garment removal
* garment falling onto ground

因为 LBS 的逻辑是：

$$
\boxed{
\text{vertex movement}
=
\text{weighted bone movement}
}
$$

而真实衣服应该是：

$$
\boxed{
M\ddot x
=
f_{\rm gravity}
+
f_{\rm stretch}
+
f_{\rm bending}
+
f_{\rm collision}
+
f_{\rm friction}
+\cdots
}
$$

这完全是另一套问题。

所以对于你现在想研究的“**可以真正穿脱的服装**”，这一点尤其关键：

> garment 一旦高度依赖 SMPL/LBS skinning，本质上就仍然是“贴在人身上的壳”。

你想让衣服：

$$
\text{穿上}\rightarrow\text{脱掉}\rightarrow\text{摊平}
$$

那么 garment state 必须能够脱离：

$$
\theta_{\text{human}}
$$

独立存在。

这时候 SMPL 最合适的角色应该是：

$$
\boxed{\text{人体几何/碰撞体/动作条件}}
$$

而不是：

$$
\boxed{\text{衣服运动本身的 deformation model}}
$$

---

# 三十六、如果你准备真正做相关研究，我建议你把下面这个计算图刻在脑子里

$$
\boxed{
\beta
}
$$

↓ shape blend shapes

$$
v_{\rm shaped}
=
\bar T+B_S(\beta)
$$

↙　　　　　　　　　↘

joint regression　　　pose correction

$$
J(\beta)
$$

　　　　　　　　　

$$
v_{\rm posed}
=
v_{\rm shaped}+B_P(\theta)
$$

↓　　　　　　　　　　　↓

FK(\(\theta,J\))　　　　　　mesh

↓

$$
A_1,\dots,A_{24}
$$

↘　　　　　　　　　　　↙

$$
\boxed{
V
=
LBS(v_{\rm posed},A,W)
}
$$

最终：

$$
V\in\mathbb R^{6890\times3}
$$

---

# 三十七、最后给你一个判断自己是否“真的懂 SMPL/LBS”的标准

如果看到下面这个式子：

$$
v_i
=
\sum_k
w_{ik}
G_k(\theta,J)
G_k(0,J)^{-1}
\left[
\bar v_i+
B_{S,i}(\beta)+
B_{P,i}(\theta)
\right]
$$

你应该能马上在脑子里读成：

> 先从平均人体出发，根据 \(\beta\) 改成人 \(A\) 的体型；再根据姿势加入 elbow/shoulder/hip 等区域的 pose corrective；根据这个人的体型回归出骨架 joint；将每个 local joint rotation 沿 kinematic tree 累乘成 global transform；乘 inverse bind transform 得到“相对 rest pose”的骨骼运动；最后每个 surface vertex 根据自己的 skinning weights 混合多个 bone transforms。

而当你看到：

$$
\sum_kw_kR_k
$$

时应该立刻警觉：

$$
\boxed{
\sum_kw_kR_k\notin SO(3)
}
$$

因此它可能引入：

$$
\text{scale/shear}
$$

从而产生：

$$
\boxed{\text{candy-wrapper / volume-loss artifacts}}
$$

如果这两件事你完全吃透，那么你已经不是“会用 SMPL”，而是开始真正理解 **parametric human model + skeletal deformation** 这套体系了。

下一步如果要继续深入，我建议直接进入 **“从代码和 tensor 角度手推一次 SMPL forward”**：我可以拿一个具体的 `β: [B,10]`、`θ:[B,72]`，从 `shapedirs → J_regressor → Rodrigues → posedirs → FK → LBS`，逐行把每个 tensor 的 shape、矩阵乘法和几何意义全部推出来；这一遍做完，你以后看 SMPL、SMPL-X、MANO、STAR 乃至各种 garment/body deformation 代码都会快很多。

[1]: https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf?utm_source=chatgpt.com "SMPL: A Skinned Multi-Person Linear Model"
[2]: https://smpl.is.tue.mpg.de/?utm_source=chatgpt.com "SMPL"
[3]: https://github.com/autocyz/smpl_understand?utm_source=chatgpt.com "GitHub - autocyz/smpl_understand: understand about SMPLmodel（http://smpl.is.tue.mpg.de/downloads） · GitHub"
[4]: https://files.is.tue.mpg.de/black/papers/SMPL2015_errata.pdf?utm_source=chatgpt.com "Errata for SMPL: A Skinned Multi-Person Linear Model"
[5]: https://github.com/chongyi-zheng/SMPLpp/blob/master/SMPL%2B%2B/scripts/preprocess.py?utm_source=chatgpt.com "SMPLpp/SMPL++/scripts/preprocess.py at master · chongyi-zheng/SMPLpp · GitHub"
[6]: https://users.cs.utah.edu/~ladislav/thesis/LKthesisLowresPrint.pdf?utm_source=chatgpt.com "Czech Technical University in Prague"
[7]: https://users.cs.utah.edu/~ladislav/kavan07skinning/kavan07skinning.html?utm_source=chatgpt.com "Skinning with Dual Quaternions"
[8]: https://users.cs.utah.edu/~ladislav/kavan12elasticity/kavan12elasticity.pdf?utm_source=chatgpt.com "ACM SIGGRAPH Asia 2012, Singapore, November 29 – December 1, 2012"
[9]: https://smpl.is.tue.mpg.de/faq.html?utm_source=chatgpt.com "SMPL"
