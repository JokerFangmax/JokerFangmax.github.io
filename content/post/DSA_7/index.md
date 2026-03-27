---
title: "DSA 7: Search Tree App"
date: 2026-03-27
math: true
tags:
  - Notes
  - DSA
categories:
  - DSA
description: "DSA CHAP7"
---

# 数据结构与算法笔记（七）：搜索树应用

> 本文基于课程第七章“搜索树应用”整理，内容主要围绕区间查询（range query）、kd-tree、多层搜索树（multi-level search tree）、range tree、interval tree、segment tree 这几类经典结构展开。我会在课件基础上补充这些结构背后的设计动机、查询逻辑、复杂度分析与相互关系，整理成适合直接发布到博客的 Markdown 笔记。主要依据你上传的课件内容：fileciteturn9file0

---

## 1. 为什么 BST 会自然走向“搜索树应用”

前一章的二叉搜索树解决的是：

- 单关键码
- 动态查找、插入、删除
- 以“有序性”换取更高效的单点查询

但在很多实际问题里，我们关心的并不只是：

- 某个 key 是否存在

而是：

- 一个区间里有多少对象
- 一个矩形窗口里有哪些点
- 一个查询点落在哪些区间里

也就是说，查询目标从 **point query** 扩展到了 **range query / stabbing query**。

这一章的核心主线可以概括为：

1. 从一维区间查询出发，理解“输出敏感（output-sensitive）”思想  
2. 由 1D 推广到 2D，看到普通二分查找为何失效  
3. 学习用 **kd-tree** 递归切空间  
4. 学习用 **multi-level search tree / range tree** 分层索引  
5. 学习针对区间 stabbing query 的 **interval tree** 与 **segment tree**

这其实就是：

> 从“给有序集合建索引”发展到“给几何对象和区间对象建索引”。

---

## 2. 一维区间查询（1D Range Query）

## 2.1 问题定义

课件第 2 页给出 1D range query 的基本形式：fileciteturn9file0

给定一组固定点集：

\[
P = \{p_1, p_2, \dots, p_n\}
\]

这些点都位于 x 轴上。  
对任意区间：

\[
I = [x_1, x_2]
\]

我们希望支持两种查询：

1. **COUNTING**：有多少点落在区间内？
2. **REPORTING**：把区间内所有点枚举出来

并且强调一个在线场景：

- 点集 `P` 固定
- 区间 `I` 会被反复、随机地查询

因此目标是：

> 先对 `P` 做预处理，再让后续查询尽可能快。

---

## 2.2 暴力解法

课件第 3 页给出最直接的思路：fileciteturn9file0

对每个点 `p`，测试：

\[
p \in [x_1, x_2]
\]

于是每次查询时间都是：

\[
O(n)
\]

如果只是看 reporting，这看起来似乎也不算太差，因为最坏情况下区间里本来就可能有 `Θ(n)` 个点要输出。

但课件马上提出一个更细的问题：

> 如果我们把“枚举输出的时间”单独算作输出成本，只关心“定位这些点”的搜索成本，能否更快？

这就引出了 **output-sensitive** 的分析方式。

---

## 3. 一维区间查询的改进：排序 + 二分搜索

## 3.1 预处理思路

课件第 4 页提出：fileciteturn9file0

1. 把所有点按坐标升序排序，存入有序向量
2. 再额外加入一个哨兵：

\[
p[0] = -\infty
\]

于是查询区间 `[x_1, x_2]` 时，可先做：

\[
t = search(x_2) = \max \{ i \mid p[i] \le x_2 \}
\]

也就是找到“不大于 `x_2` 的最后一个点”。

这个操作可在：

\[
O(\log n)
\]

时间内完成。

---

## 3.2 为什么从右端点开始反向枚举

课件给出的 reporting 思路是：fileciteturn9file0

- 从 `p[t]` 开始向左回扫
- 直到越过区间左边界 `x_1`
- 停在某个 `p[s]`

于是输出点数：

\[
r = t - s
\]

查询总时间就是：

\[
O(\log n + r)
\]

这里的 `r` 就是输出规模（output size）。

---

## 3.3 Counting 为什么只需 `O(log n)`

课件第 5 页进一步指出：fileciteturn9file0

如果我们不需要真正把点枚举出来，只是想计数，那么：

- `t` 可二分得到
- `s` 也可二分得到

于是：

\[
	ext{count} = t - s
\]

两次二分后就能完成 counting，因此复杂度是：

\[
O(\log n)
\]

这与实际输出规模 `r` 无关。

---

## 3.4 输出敏感（Output-Sensitive）

课件第 5 页把这个结论总结成：

- **enumerating query**：`O(r + log n)`
- **counting query**：`O(log n)` fileciteturn9file0

这就是一个经典的 output-sensitive 结果：

> 查询时间不仅与输入规模 `n` 有关，还与输出规模 `r` 有关。

这在几何数据结构中非常常见。因为很多时候：

- 搜索部分可以做得很快
- 但最终结果本来就有很多项需要输出

因此 “`O(log n + r)`” 已经是非常理想的复杂度形式。

---

## 4. 从 1D 到 2D：为什么普通二分查找失效

## 4.1 平面区间查询（Planar Range Query）

课件第 7 页把问题推广到二维：fileciteturn9file0

给定平面点集：

\[
P = \{p_1, p_2, \dots, p_n\}
\]

查询一个轴对齐矩形区域：

\[
R = [x_1, x_2] 	imes [y_1, y_2]
\]

仍有两类任务：

1. `|R \cap P|`：counting
2. `R \cap P`：reporting

---

## 4.2 为什么 1D 的有序向量方法不够了

在 1D 中，所有点只按 x 坐标排一次序即可。  
但在 2D 中，一个点同时有：

- x 坐标
- y 坐标

若只按 x 排序，那么区间 `[x_1, x_2]` 内的候选点还需要再按 y 过滤。  
这就失去了“直接靠一维有序性锁定答案”的简单结构。

课件第 7 页直接指出：

> Binary search doesn't help this kind of query. fileciteturn9file0

也就是说：

- 单一维度排序不够表达二维矩形约束
- 必须引入更复杂的几何组织方式

---

## 4.3 Inclusion-Exclusion 的预处理思路

课件第 8～10 页给出一种二维 counting 思路：fileciteturn9file0

对每个点 `(x, y)` 预处理：

\[
n(x,y) = |([0,x] 	imes [0,y]) \cap P|
\]

也就是“被 `(x,y)` 支配（dominated）”的点数。  
这样对任意矩形：

\[
R = [x_1, x_2] 	imes [y_1, y_2]
\]

可由容斥公式得到：

\[
|R \cap P|
=
n(x_2,y_2) - n(x_1,y_2) - n(x_2,y_1) + n(x_1,y_1)
\]

课件第 10 页清楚展示了这个 Inclusion-Exclusion Principle。fileciteturn9file0

---

## 4.4 为什么这个方法不够好

课件第 11 页指出，它虽然能把每次查询做到 `O(log n)` 级别，但预处理与存储要达到：

\[
\Theta(n^2)
\]

而且高维时会更糟。fileciteturn9file0

所以我们需要一种：

- 查询快
- 存储别太爆炸
- 更可扩展到高维

的结构。  
这就引出了后面的 **kd-tree** 和 **多层搜索树**。

---

## 5. kd-tree：把空间递归切开

## 5.1 1D 情况：Complete BST 视角

课件第 13～17 页先重新审视 1D range search，把它写成一棵平衡 BST 的形式。fileciteturn9file0

其中：

- 每个节点保存某个关键分割值
- `search(x)` 返回“不大于 x 的最大 key”
- 查询 `[x_1, x_2]` 时，可先找到：
  - `search(x_1)`
  - `search(x_2)`
- 再从二者的 LCA（最低公共祖先）出发，确定一批需要完整报告的子树

课件第 15 页特别强调：

> 区间结果可以表示为 **`O(log n)` 个互不相交子树的并**。 fileciteturn9file0

这一思想非常关键，因为它在后面的 kd-tree 和 range tree 中都会反复出现。

---

## 5.2 2D kd-tree 的核心思想

课件第 19 页给出二维 kd-tree 的出发点：fileciteturn9file0

- 从整个平面开始作为一个大区域
- 在偶数层按垂直方向切
- 在奇数层按水平方向切
- 递归地把子区域再切分下去

因此 kd-tree 本质上是：

> **对空间做分而治之（divide-and-conquer）递归切分。**

每个内部节点对应一条切分线，每个叶子对应一个最终子区域。

---

## 5.3 为什么要尽量按中位数切

课件第 20 页指出，为了让结构有效：

- 每次切分都应尽量均衡（at median）
- 区域边界要约定开闭方式，避免重复归属 fileciteturn9file0

按中位数切的目的显然是：

> 保证树高为 `O(log n)`。

否则若每次切分极不均衡，kd-tree 也会退化。

---

## 5.4 kd-tree 的构建

课件第 24 页给出递归伪代码：fileciteturn9file0

```text
buildKdTree(P, d):
    if P = {p}: return CreateLeaf(p)
    root = CreateKdNode()
    root->splitDirection = Even(d) ? VERTICAL : HORIZONTAL
    root->splitLine = FindMedian(root->splitDirection, P)
    (P1, P2) = Divide(P, root->splitDirection, root->splitLine)
    root->lc = buildKdTree(P1, d + 1)
    root->rc = buildKdTree(P2, d + 1)
    return root
```

这段代码清楚体现了 kd-tree 的三个要素：

1. **切分方向**：随深度交替
2. **切分位置**：取中位数
3. **递归构造左右子树**

---

## 5.5 Canonical Subsets

课件第 27 页指出：fileciteturn9file0

> kd-tree 中每个节点对应：
> - 平面中的一个矩形子区域
> - 以及该区域内所包含的点集

这个点集就叫 **canonical subset（规范子集）**。

并且：

- 同一深度的节点区域互不相交
- 它们的并覆盖整个平面
- 内部节点的区域 = 两个孩子区域的并

这意味着 kd-tree 不只是“几何切分树”，同时还是一种“点集分层组织树”。

---

## 6. kd-tree 查询：Accepted / Rejected / Recursion

## 6.1 查询框架

课件第 30 页给出二维 kd-tree 查询伪代码：fileciteturn9file0

```text
kdSearch(v, R):
    if v is leaf:
        if point(v) inside R: report(v)
        return

    if region(lc(v)) ⊆ R:
        reportSubtree(lc(v))
    else if region(lc(v)) ∩ R ≠ ∅:
        kdSearch(lc(v), R)

    if region(rc(v)) ⊆ R:
        reportSubtree(rc(v))
    else if region(rc(v)) ∩ R ≠ ∅:
        kdSearch(rc(v), R)
```

这个框架极其重要，因为它把每个节点分成三类：

1. **accepted**：整个子区域都在查询矩形内，整棵子树可直接报告
2. **rejected**：整个子区域与查询矩形不相交，可直接剪枝
3. **recursion**：部分相交，必须继续递归下探

课件第 17 页的 “hot knives through a chocolate cake” 图，就是在可视化这三类节点。fileciteturn9file0

---

## 6.2 为什么这叫“热刀切巧克力”

因为 kd-tree 把空间切成很多矩形块，像一层层切开的巧克力蛋糕；  
而查询矩形就像两把“热刀”从上往下切过去：

- 完全落在刀之间的块：accepted
- 完全落在外面的块：rejected
- 被刀边穿过的块：recursion

这幅图特别适合建立直觉：  
真正需要递归深入的，只是那些被边界切穿的区域，而不是所有区域。

---

## 6.3 kd-tree 的复杂度

课件第 35～38 页给出分析：fileciteturn9file0

### 6.3.1 预处理

若每层都在点集中找中位数并切分，则总预处理满足：

\[
T(n) = 2T(n/2) + O(n) = O(n \log n)
\]

### 6.3.2 存储

树高是 `O(log n)`，总节点数是 `O(n)`，因此存储是：

\[
O(n)
\]

### 6.3.3 查询

查询时间可写成：

\[
	ext{Query Time} = 	ext{Search} + 	ext{Report}
\]

在二维情况下，课件给出的结论是典型的：

\[
O(\sqrt{n} + r)
\]

或等价地，把搜索代价理解为与“递归节点数”相关。  
其中 `r` 为输出规模。课件第 37 页强调搜索部分取决于递归调用数 `Q(n)`。fileciteturn9file0

---

## 6.4 kd-tree 的高维推广

课件第 38 页指出：

- 在 `k` 维空间中，可轮流按第 1、2、…、k 个坐标切分
- 得到更一般的 kd-tree fileciteturn9file0

但高维下，查询复杂度会迅速恶化。  
这体现了几何数据结构里很经典的“维度灾难（curse of dimensionality）”。

---

## 7. Multi-Level Search Tree：把二维查询拆成两个一维查询

## 7.1 基本想法

课件第 40 页给出一个极自然的思路：fileciteturn9file0

二维矩形查询：

\[
[x_1, x_2] 	imes [y_1, y_2]
\]

似乎可以拆成：

1. 先做 x-query：找出所有 x 落在 `[x_1, x_2]` 的点
2. 再在这些候选里做 y-query：筛掉不在 `[y_1, y_2]` 的点

也就是：

> 2D Range Query = x-Query + y-Query

---

## 7.2 为什么简单串联会失败

课件第 41 页给出 worst case：fileciteturn9file0

- x-query 返回了几乎全部点
- y-query 最后把它们几乎全部剔除
- 最终输出规模 `r = 0`
- 却已经花了 `Ω(n)` 时间

这说明“先筛一维、再线性扫另一维”并不够高效。  
我们需要的是：

> **在所有相关子集上同时拥有第二维的快速索引。**

---

## 7.3 Tree of Trees

课件第 42 页提出 multi-level search tree 的核心结构：fileciteturn9file0

1. 先建立一个按 x 组织的 1D BBST（x-tree）
2. 对 x-tree 中每个节点 `v`，再建立一个按 y 坐标组织的 BBST（y-tree）
3. 其中这个 y-tree 存放的正是与节点 `v` 对应 canonical subset 中的点

于是整棵结构就成了：

> **x-tree of y-trees**

这就是 multi-level search tree（MLST）。

---

## 7.4 查询逻辑

课件第 45 页给出二维查询算法：fileciteturn9file0

1. 在 x-tree 上做 1D range query，找出所有与 `[x_1, x_2]` 对应的 canonical subsets  
   这些集合数量是 `O(log n)`。

2. 对于每个 canonical subset，对应地进入其 y-tree 中再做一次 1D y-range query。

因此二维查询被改写成：

> 2D Range Query = x-Query * y-Queries

注意这里不是“先筛再扫”，而是“在每个规范子集上都有自己的 y 索引”。

---

## 7.5 复杂度

课件第 46～48 页给出二维 MLST 的复杂度：fileciteturn9file0

### 7.5.1 空间

因为每个输入点会出现在多个 y-tree 中，二维 2-level search tree 的空间是：

\[
O(n \log n)
\]

### 7.5.2 预处理

构建时间同样是：

\[
O(n \log n)
\]

### 7.5.3 查询

- x-range query 找到 `O(log n)` 个 canonical subsets
- 每个 canonical subset 做一次 y-range query，需要 `O(log n)` 时间
- 因此总查询时间是：

\[
O(\log^2 n + r)
\]

这比 kd-tree 的 `O(\sqrt{n}+r)` 在二维时通常更优。

---

## 8. Range Tree：用 Fractional Cascading 再降一个 `log`

## 8.1 一个关键观察：y-search 都在查“同一个 y 区间”

课件第 50～53 页指出，二维 MLST 还可以继续优化。fileciteturn9file0

在 2D 查询中：

- x-tree 上会访问多个节点
- 每个节点都要在自己的 y-list / y-tree 上查同一个 `[y_1, y_2]`

也就是说：

> 我们在重复地查不同的表，但 key 却是同一个。

这就产生了“coherence（相干性）”。

---

## 8.2 从 BBST of BBST 到 BBST of Lists

课件第 50 页指出：

> 由于每个 y-search 本身就是 1D 查询，不再需要进一步递归  
> 因此没必要把 canonical subset 存成 y-BBST，排好序的 y-list 就够了。 fileciteturn9file0

于是结构从：

- `BBST<BBST<T>>`

变成：

- `BBST<List<T>>`

这一步已经减少了一些常数复杂度。

---

## 8.3 Fractional Cascading 的思想

课件第 52～54 页说明：fileciteturn9file0

- 对于父节点 `v` 的 y-list `A_v`
- 以及其左右孩子的 y-lists `A_L, A_R`
- 在 `A_v` 中每个元素处额外存两个指针：
  - 指向 `A_L` 中“不大于它的最大元素”
  - 指向 `A_R` 中“不大于它的最大元素”

于是：

1. 在根节点的 y-list 中做一次 `O(log n)` 搜索
2. 之后往下走到孩子时，不必重新二分
3. 只需沿着这些“级联指针”在 `O(1)` 时间更新当前位置

这项技术就叫：

> **Fractional Cascading**

---

## 8.4 Range Tree 的复杂度

课件第 56 页给出结论：fileciteturn9file0

带 fractional cascading 的 MLST 就叫 **range tree**。

对于二维情况：

- 空间：`O(n log n)`
- 构建：`O(n log n)`
- 查询：从 `O(log^2 n + r)` 降到

\[
O(\log n + r)
\]

这几乎达到了二维正交范围 reporting 的理想级别。

---

## 8.5 更高维情况

课件第 57 页指出：

- fractional cascading 只能应用到最后一级
- 在 `d` 维中，query time 一般变成：

\[
O(\log^{d-1} n + r)
\]

同时空间和构建时间也会随维度继续乘上对数因子。fileciteturn9file0

这再次说明：

> range tree 在低维几何查询中很强，但高维下代价仍会迅速上升。

---

## 9. Interval Tree：解决 stabbing query

## 9.1 问题定义

课件第 59 页引入另一个经典问题：fileciteturn9file0

给定一组区间：

\[
I = \{[l_i, r_i]\}
\]

以及一个查询点 `q_x`，要求找出：

> 所有包含 `q_x` 的区间

这类问题叫：

> **Stabbing Query**

与 range query 不同，这里输入对象不是点，而是区间；查询对象是一个点。

---

## 9.2 Interval Tree 的划分思想

课件第 60～63 页给出 interval tree 的构建思路：fileciteturn9file0

1. 取所有端点的中位数作为 `x_mid(v)`
2. 把所有区间分成三类：
   - `S_left`：完全在中位数左边
   - `S_right`：完全在中位数右边
   - `S_mid`：跨越中位数的区间
3. 对 `S_left` 与 `S_right` 递归建树
4. 在当前节点上存放 `S_mid`

这形成了一棵深度约为 `O(log n)` 的树。

---

## 9.3 Associative Lists

课件第 63 页指出，对每个节点 `v`，要把 `S_mid(v)` 再排成两张表：fileciteturn9file0

- `L_left`：按左端点排序
- `L_right`：按右端点排序

为什么要两张表？

因为查询点 `q_x` 相对 `x_mid(v)` 有两种位置关系：

- 若 `q_x < x_mid(v)`，则要快速筛出左端点不超过 `q_x` 的区间
- 若 `q_x > x_mid(v)`，则要快速筛出右端点不小于 `q_x` 的区间

这两种筛法需要不同排序方向，因此需要两张关联表。

---

## 9.4 查询算法

课件第 66 页给出伪代码：fileciteturn9file0

```text
queryIntervalTree(v, qx):
    if !v: return
    if qx < xmid(v):
        report all segments of Smid(v) containing qx
        queryIntervalTree(lc(v), qx)
    else if xmid(v) < qx:
        report all segments of Smid(v) containing qx
        queryIntervalTree(rc(v), qx)
    else:
        report all segments of Smid(v)
```

核心点在于：

- 查询路径只会沿一条根到叶的路径走下去
- 每层访问一个节点
- 并在该节点的 `S_mid` 中报告那些跨越查询点的区间

---

## 9.5 复杂度

课件第 67 页总结：fileciteturn9file0

- 查询最多访问 `O(log n)` 个节点
- 在每个节点上，报告部分总计加起来就是输出规模 `r`

因此总查询复杂度为：

\[
O(r + \log n)
\]

这是 stabbing query 的一个非常漂亮的 output-sensitive 结果。

---

## 10. Segment Tree：基于离散化与规范覆盖

## 10.1 从 elementary intervals 出发

课件第 69～70 页首先把所有区间端点排序：

\[
p_1 < p_2 < \cdots < p_m
\]

然后由它们定义出 `m+1` 个 **elementary intervals（基本区间）**。fileciteturn9file0

在同一个 elementary interval 里，任意 stabbing query 的输出都是相同的。  
这说明，如果只看 stabbing query：

> 真正重要的不是整个实数轴，而是端点诱导出来的这些离散区间。

---

## 10.2 为什么不能直接给每个 elementary interval 存答案

课件第 70～74 页分析了这种朴素方案：fileciteturn9file0

- 用有序向量存所有 elementary intervals
- 二分找到查询点落在哪个 EI 中
- 直接返回其存储的输出集

问题在于：

- 一个原始区间可能跨越 `Ω(n)` 个 EI
- 若在每个 EI 中都显式存它
- 总空间最坏达到：

\[
\Omega(n^2)
\]

因此必须想办法压缩表示。

---

## 10.3 Canonical Subsets with Greedy Merging

课件第 75～78 页给出关键优化：fileciteturn9file0

1. 用一棵平衡 BST 建在所有 elementary intervals 上
2. 每个节点 `v` 对应一个区间范围 `R(v)`
3. 对于每个原始区间 `s`，不是把它复制到所有叶子
4. 而是尽量贪心地把它存到那些满足：

\[
R(v) \subseteq s
\]

的最大规范节点上

这就类似于用若干个 **canonical subsets** 来覆盖原区间。

---

## 10.4 `InsertSegment(v, s)` 的逻辑

课件第 78 页给出伪代码：fileciteturn9file0

```text
InsertSegment(v, s):
    if R(v) ⊆ s:
        store s at v and return
    if R(lc(v)) ∩ s ≠ ∅:
        InsertSegment(lc(v), s)
    if R(rc(v)) ∩ s ≠ ∅:
        InsertSegment(rc(v), s)
```

课件特别指出：

> 每一层至多访问 4 个节点（2 store + 2 recurse）  
> 因而每个区间插入总共只需 `O(log n)` 时间。 fileciteturn9file0

---

## 10.5 为什么总空间是 `O(n log n)`

每个区间最多被分解到 `O(log n)` 个规范节点上；  
共有 `n` 个区间，因此总存储空间是：

\[
O(n \log n)
\]

这比朴素方案的 `Ω(n^2)` 好得多。

---

## 10.6 Segment Tree 的查询

课件第 79～80 页给出查询算法：fileciteturn9file0

```text
Query(v, qx):
    report all intervals in Int(v)
    if v is a leaf: return
    if qx ∈ R(lc(v)):
        Query(lc(v), qx)
    else:
        Query(rc(v), qx)
```

注意这里与 interval tree 不同：

- 查询只沿一条根到叶的路径走下去
- 因为查询点只会落在某个孩子区间中
- 但沿途所有节点存的 `Int(v)` 都要报告

---

## 10.7 查询复杂度

课件第 80～81 页总结：fileciteturn9file0

- 每层只访问一个节点
- 总共访问 `O(log n)` 个节点
- 每个节点报告其规范子集中的若干区间
- 所有报告总和就是输出规模 `r`

因此查询时间为：

\[
O(r + \log n)
\]

构建与存储都是：

\[
O(n \log n)
\]

这正是 segment tree 处理 1D stabbing query 的经典复杂度。

---

## 11. 本章几种结构的关系与对比

这一章结构很多，最容易混。这里做一个主线梳理。

### 11.1 kd-tree

适合：

- 点集
- 低维正交 range query

思路：

- 递归切空间
- 每个节点对应一个矩形区域与 canonical subset

特点：

- 空间 `O(n)`
- 构建 `O(n log n)`
- 二维查询通常可做到 `O(√n + r)` 量级

---

### 11.2 Multi-Level Search Tree

适合：

- 把多维 range query 拆成多次一维搜索

思路：

- 第一维建一棵树
- 每个节点再为第二维建一棵树
- 形成 tree of trees

二维复杂度：

- 空间 `O(n log n)`
- 构建 `O(n log n)`
- 查询 `O(log^2 n + r)`

---

### 11.3 Range Tree

适合：

- 低维正交范围查询，尤其是二维

思路：

- 本质仍是 MLST
- 但把下层树改成排序表
- 再用 fractional cascading 把重复搜索级联起来

二维复杂度：

- 空间 `O(n log n)`
- 构建 `O(n log n)`
- 查询优化到 `O(log n + r)`

---

### 11.4 Interval Tree

适合：

- 区间集合
- stabbing query（点 stabbing 区间）

思路：

- 按中位数把区间分成 left / mid / right
- `S_mid` 存在当前节点
- 左右递归继续分

复杂度：

- 查询 `O(r + log n)`

---

### 11.5 Segment Tree

适合：

- 区间集合
- stabbing query
- 以及后续各种区间覆盖 / 区间更新问题的基础框架

思路：

- 先对端点离散化
- 再用平衡树表示 elementary intervals
- 把区间贪心存进 `O(log n)` 个 canonical nodes

复杂度：

- 空间 `O(n log n)`
- 构建 `O(n log n)`
- 查询 `O(r + log n)`

---

## 12. 这一章真正想传达什么

这一章虽然标题叫 “BST Application”，但它真正讲的并不只是“BST 的几个应用题”。

更深层的主线其实是：

### 12.1 一维有序搜索为什么成功

在 1D 中，排序 + 二分已经足够强大，因为顺序性天然与空间是一致的。

### 12.2 多维问题为什么困难

一旦进入 2D / kD：

- 单一总序失效
- 一个对象在多个维度上同时受约束
- 必须引入空间划分或多级索引

### 12.3 Canonical Subset 是核心思想

无论是：

- 1D balanced BST 的 range search
- kd-tree
- MLST / range tree
- segment tree

都反复出现一个核心概念：

> **把查询答案表示成若干 canonical subsets 的并。**

这是一种极其重要的设计范式：

- 先预处理出一批“规范子集”
- 查询时只需定位少量相关子集
- 再统一报告它们

### 12.4 Output-sensitive 是几何搜索的自然目标

因为 reporting query 中：

- 若答案本来就有 `r` 项
- 任意算法至少要花 `Ω(r)` 时间输出它们

因此最理想的复杂度形式往往就是：

\[
O(	ext{search cost} + r)
\]

例如：

- 1D range query：`O(log n + r)`
- interval tree：`O(log n + r)`
- segment tree：`O(log n + r)`
- range tree：二维可做到 `O(log n + r)`

---

## 13. 本章主线总结

### 13.1 从 BST 出发，搜索问题从“单点”推广到“区间 / 矩形 / stabbing”

这使得数据结构目标从：

- 查一个 key

扩展为：

- 查一整个子集

### 13.2 1D 范围查询相对简单

排序 + 二分即可做到：

- counting：`O(log n)`
- reporting：`O(log n + r)`

### 13.3 2D 及更高维需要更复杂的索引组织

- kd-tree：递归切空间
- MLST：树中再套树
- range tree：再配 fractional cascading 降一个 `log`

### 13.4 stabbing query 对应另一类结构

- interval tree：按中位数分 interval
- segment tree：按 elementary intervals 离散化再做 canonical cover

### 13.5 几何搜索的核心语言是“canonical subsets + output sensitivity”

这两个思想贯穿整章。

---

## 14. 一句话总结

> 这一章真正重要的，不只是记住 kd-tree、range tree、interval tree、segment tree 这些名字，而是理解：当查询对象从单点扩展为区间、矩形或区间集合时，搜索树的任务就不再是“找到一个节点”，而是要把答案组织成少量 canonical subsets，并以 `O(search + output)` 的方式高效返回。
