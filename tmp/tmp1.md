可以把这件事拆成三层来理解：**“我想怎么转” → 用 3 个数表示 → 变成真正能乘在 3D 点上的旋转矩阵。**

### 1. 什么是轴角向量（axis-angle vector）

三维空间中的一次旋转，本质上只需要两件事：

- **绕哪根轴转**
- **转多少角度**

假设单位旋转轴是

\[
\mathbf u =
\begin{bmatrix}
u_x\\u_y\\u_z
\end{bmatrix},
\qquad \|\mathbf u\|=1
\]

转角是 \(\theta\)，那么轴角向量定义成：

\[
\boxed{\boldsymbol\omega=\theta\mathbf u}
\]

所以它是一个普通的三维向量：

\[
\boldsymbol\omega =
\begin{bmatrix}
\omega_x\\
\omega_y\\
\omega_z
\end{bmatrix}
\]

但它同时把“轴”和“角度”压缩在一起。

因为

\[
\|\boldsymbol\omega\|
=
\|\theta\mathbf u\|
=
|\theta|\|\mathbf u\|
=
|\theta|
\]

所以：

\[
\boxed{
\theta=\|\boldsymbol\omega\|
}
\]

而旋转轴：

\[
\boxed{
\mathbf u=\frac{\boldsymbol\omega}
{\|\boldsymbol\omega\|}
}
\]

例如：

\[
\boldsymbol\omega=
\begin{bmatrix}
0.3\\0\\0
\end{bmatrix}
\]

那么

\[
\theta=\sqrt{0.3^2}=0.3
\]

并且

\[
\mathbf u=
\frac{(0.3,0,0)}{0.3}
=
(1,0,0)
\]

所以它表示：

\[
\boxed{\text{绕 x 轴旋转 }0.3\text{ rad}}
\]

也就是大约：

\[
0.3\times\frac{180}{\pi}
\approx17.2^\circ
\]

这就是 SMPL 为什么每个关节只需要 **3 个 pose 参数**。不是直接存一个 \(3\times3\) 矩阵的 9 个数，而是存：

\[
(\omega_x,\omega_y,\omega_z)
\]

---

## 2. 但轴角向量不能直接拿来旋转点

假如有一个点：

\[
\mathbf p=
\begin{bmatrix}
x\\y\\z
\end{bmatrix}
\]

真正做旋转通常需要：

\[
\mathbf p'=R\mathbf p
\]

其中

\[
R\in SO(3)
\]

是一个 \(3\times3\) 旋转矩阵。

例如绕 \(x\) 轴旋转 \(\theta\)：

\[
R_x(\theta)=
\begin{bmatrix}
1&0&0\\
0&\cos\theta&-\sin\theta\\
0&\sin\theta&\cos\theta
\end{bmatrix}
\]

所以问题变成：

> 已知任意轴角向量 \(\omega\)，怎么得到对应的旋转矩阵 \(R\)？

这就是 **Rodrigues 公式 / 指数映射**解决的问题。

---

# 3. 先理解 \([\omega]_\times\) 是什么

给定向量：

\[
\boldsymbol\omega=
\begin{bmatrix}
\omega_x\\
\omega_y\\
\omega_z
\end{bmatrix}
\]

可以构造一个反对称矩阵：

\[
\boxed{
[\boldsymbol\omega]_\times =
\begin{bmatrix}
0&-\omega_z&\omega_y\\
\omega_z&0&-\omega_x\\
-\omega_y&\omega_x&0
\end{bmatrix}
}
\]

这个矩阵最重要的性质是：

\[
[\boldsymbol\omega]_\times\mathbf v
=
\boldsymbol\omega\times\mathbf v
\]

也就是说：

> 矩阵乘法 \([\omega]_\times v\)，等价于三维向量叉积 \(\omega\times v\)。

比如

\[
\omega=(1,0,0)
\]

那么：

\[
[\omega]_\times=
\begin{bmatrix}
0&0&0\\
0&0&-1\\
0&1&0
\end{bmatrix}
\]

拿它乘：

\[
v=(0,1,0)
\]

得到：

\[
[\omega]_\times v=
\begin{bmatrix}
0\\0\\1
\end{bmatrix}
\]

正好就是：

\[
(1,0,0)\times(0,1,0)
=
(0,0,1)
\]

---

# 4. 为什么这个叉积矩阵会和旋转有关系？

这是一个非常关键的直觉。

考虑一个点 \(\mathbf p\)，绕单位轴 \(\mathbf u\) 旋转一个**极小角度** \(d\theta\)。

它的微小位移方向近似为：

\[
d\mathbf p
=
(\mathbf u\times\mathbf p)d\theta
\]

又因为：

\[
\mathbf u\times\mathbf p
=
[\mathbf u]_\times\mathbf p
\]

所以：

\[
\frac{d\mathbf p}{d\theta}
=
[\mathbf u]_\times\mathbf p
\]

这其实就是一个微分方程。

它的解是什么？

\[
\boxed{
\mathbf p(\theta)
=
e^{\theta[\mathbf u]_\times}\mathbf p(0)
}
\]

因此旋转矩阵就是：

\[
\boxed{
R=e^{\theta[\mathbf u]_\times}
}
\]

而因为：

\[
\boldsymbol\omega=\theta\mathbf u
\]

所以：

\[
\boxed{
R=e^{[\boldsymbol\omega]_\times}
}
\]

这就是所谓的：

\[
\boxed{\text{指数映射 Exponential Map}}
\]

---

# 5. 这里的 \(e^{[\omega]_\times}\) 是什么意思？

这里不是普通数字的：

\[
e^x
\]

而是**矩阵指数**。

数学上定义：

\[
e^A
=
I+A+\frac{A^2}{2!}
+\frac{A^3}{3!}
+\cdots
\]

所以：

\[
e^{[\omega]_\times}
=
I+[\omega]_\times
+\frac{[\omega]_\times^2}{2!}
+\frac{[\omega]_\times^3}{3!}
+\cdots
\]

看起来很恐怖。

但是因为三维叉积矩阵具有非常特殊的代数性质，这个无穷级数可以被化简。

最后就得到 **Rodrigues 公式**。

---

# 6. Rodrigues 公式

先定义：

\[
\theta=\|\omega\|
\]

以及单位旋转轴：

\[
\mathbf u=\frac{\omega}{\theta}
\]

那么：

\[
\boxed{
R
=
I
+
\sin\theta[\mathbf u]_\times
+
(1-\cos\theta)[\mathbf u]_\times^2
}
\]

这就是最经典的 Rodrigues rotation formula。

注意这里的：

\[
\boxed{1-\cos\theta}
\]

正是你原文提到的那个二阶项。

---

## 如果直接使用 \(\omega\)，公式会长这样

因为：

\[
[\mathbf u]_\times
=
\frac{[\omega]_\times}{\theta}
\]

代进去：

\[
\boxed{
R
=
I
+
\frac{\sin\theta}{\theta}
[\omega]_\times
+
\frac{1-\cos\theta}{\theta^2}
[\omega]_\times^2
}
\]

其中：

\[
\theta=\|\omega\|
\]

这两个公式是**完全等价的**。

也就是说一定要注意你看到的论文到底使用的是：

\[
[\mathbf u]_\times
\]

还是

\[
[\omega]_\times
\]

否则很容易觉得系数“不一样”。

---

# 7. 用你给的例子真正算一次

假设 SMPL 某个关节：

\[
\omega=
\begin{bmatrix}
0.3\\0\\0
\end{bmatrix}
\]

### 第一步：角度

\[
\theta=\|\omega\|=0.3
\]

### 第二步：旋转轴

\[
u=
\begin{bmatrix}
1\\0\\0
\end{bmatrix}
\]

所以：

\[
[u]_\times=
\begin{bmatrix}
0&0&0\\
0&0&-1\\
0&1&0
\end{bmatrix}
\]

平方：

\[
[u]_\times^2
=
\begin{bmatrix}
0&0&0\\
0&-1&0\\
0&0&-1
\end{bmatrix}
\]

代入：

\[
R
=
I+
\sin0.3[u]_\times
+
(1-\cos0.3)[u]_\times^2
\]

因为：

\[
\sin0.3\approx0.2955
\]

\[
\cos0.3\approx0.9553
\]

所以：

\[
R\approx
\begin{bmatrix}
1&0&0\\
0&0.9553&-0.2955\\
0&0.2955&0.9553
\end{bmatrix}
\]

你会发现这恰好就是熟悉的：

\[
\boxed{
R_x(0.3)=
\begin{bmatrix}
1&0&0\\
0&\cos0.3&-\sin0.3\\
0&\sin0.3&\cos0.3
\end{bmatrix}
}
\]

所以 Rodrigues 公式并没有发明一种“新旋转”。

它只是告诉你：

> **对于任意旋转轴 \(u\)，如何统一地构造旋转矩阵。**

---

# 8. Rodrigues 公式的几何意义其实更容易理解

对于一个向量 \(p\)，绕单位轴 \(u\) 转 \(\theta\)，可以把 \(p\) 分成：

\[
p=p_\parallel+p_\perp
\]

其中：

- \(p_\parallel\)：沿旋转轴的部分
- \(p_\perp\)：垂直旋转轴的部分

沿轴的部分不会改变。

垂直部分则在平面里旋转。

最终可以写成：

\[
\boxed{
p'
=
p\cos\theta
+
(u\times p)\sin\theta
+
u(u^\top p)(1-\cos\theta)
}
\]

这也是 Rodrigues 公式的另一个常见形式。

三个部分可以理解为：

\[
\underbrace{p\cos\theta}_{原方向}
+
\underbrace{(u\times p)\sin\theta}_{旋转方向}
+
\underbrace{u(u^\top p)(1-\cos\theta)}_{把轴向分量补回来}
\]

它和矩阵形式：

\[
R
=
I+\sin\theta[u]_\times
+(1-\cos\theta)[u]_\times^2
\]

本质完全相同。

---

# 9. “指数映射”和“Rodrigues 公式”到底是什么关系？

这个地方在看 SMPL / 机器人 / SLAM 文献时特别容易混乱。

可以简单记成：

\[
\boxed{
\text{指数映射是数学定义，Rodrigues 是它在 }SO(3)\text{ 上的闭式解}
}
\]

也就是：

\[
R
=
\underbrace{\exp([\omega]_\times)}_{\text{指数映射}}
\]

计算出来之后：

\[
=
\underbrace{
I+\frac{\sin\theta}{\theta}[\omega]_\times
+\frac{1-\cos\theta}{\theta^2}[\omega]_\times^2
}_{\text{Rodrigues 闭式公式}}
\]

所以在代码里看到：

```python
R = batch_rodrigues(theta)
```

本质就是：

\[
\omega
\xrightarrow{\text{Rodrigues}}
R
\]

---

# 10. 为什么叫从 \(\mathfrak{so}(3)\) 到 \(SO(3)\)？

如果你继续读 SMPL、机器人学或 SLAM，很快会看到这两个符号：

\[
SO(3)
\]

和：

\[
\mathfrak{so}(3)
\]

这是理解指数映射的核心。

### \(SO(3)\)

是真正的三维旋转矩阵集合：

\[
SO(3)=
\{R\in\mathbb R^{3\times3}
\mid
R^\top R=I,\det R=1
\}
\]

一个合法旋转矩阵必须满足：

\[
R^\top R=I
\]

而且：

\[
\det R=1
\]

---

### \(\mathfrak{so}(3)\)

则是所有 \(3\times3\) 反对称矩阵：

\[
\mathfrak{so}(3)
=
\left\{
A\mid A^\top=-A
\right\}
\]

例如：

\[
[\omega]_\times=
\begin{bmatrix}
0&-\omega_z&\omega_y\\
\omega_z&0&-\omega_x\\
-\omega_y&\omega_x&0
\end{bmatrix}
\]

就是：

\[
[\omega]_\times\in\mathfrak{so}(3)
\]

而指数映射：

\[
\boxed{
\exp:\mathfrak{so}(3)\rightarrow SO(3)
}
\]

完成了：

\[
\boxed{
\omega
\rightarrow
[\omega]_\times
\rightarrow
\exp([\omega]_\times)
\rightarrow
R
}
\]

所以在 SMPL 里可以画成：

\[
\boxed{
\underbrace{
(\omega_x,\omega_y,\omega_z)
}_{3个pose参数}
\rightarrow
\underbrace{
[\omega]_\times
}_{反对称矩阵}
\rightarrow
\underbrace{
R
}_{3\times3旋转矩阵}
}
\]

---

# 11. 放回 SMPL 中，你现在应该这样理解

假设 SMPL 有一个 elbow 关节，它的 pose 参数是：

\[
\theta_k=
(\omega_x,\omega_y,\omega_z)
\]

例如：

\[
(0,0,1.2)
\]

代表大约绕 \(z\) 轴旋转 \(1.2\) rad。

SMPL 不会直接拿这三个数字去做 LBS。

而是先：

\[
\omega_k
\overset{\mathrm{Rodrigues}}{\longrightarrow}
R_k
\]

得到：

\[
R_k\in\mathbb R^{3\times3}
\]

然后根据 kinematic tree 计算这个关节的全局变换：

\[
G_k
\]

最后才进入你之前看到的 LBS：

\[
v_i'
=
\sum_k
w_{ik}G_k
\begin{bmatrix}
v_i\\1
\end{bmatrix}
\]

所以整个逻辑其实是：

\[
\boxed{
\text{SMPL pose参数}
\rightarrow
\text{轴角}
\rightarrow
\text{旋转矩阵}
\rightarrow
\text{骨骼变换}
\rightarrow
\text{LBS}
\rightarrow
\text{人体mesh}
}
\]

这条链弄清楚之后，后面理解 **SMPL 的 pose blend shapes 为什么用 \(R_k-I\)**、以及 **LBS 糖纸伪影为什么会产生**就会容易很多。