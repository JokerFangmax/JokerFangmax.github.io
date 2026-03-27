---
title: "DSA 11: The Application of Graph"
date: 2026-03-27
math: true
tags:
  - Notes
  - DSA
categories:
  - DSA
description: "DSA CHAP11"
---

# 数据结构与算法笔记（十一）：图应用

> 本文基于课程第十一章“图应用”整理。内容主要围绕四条主线展开：  
> 第一条是**双连通分量（BCC）与关节点**，说明 DFS 时间戳和回边信息如何揭示图的脆弱结构；  
> 第二条是**优先级搜索（PFS）统一框架**，说明 BFS、Dijkstra、Prim 等算法其实都可以看成“按某种优先级不断扩展一棵树”；  
> 第三条是**最短路径问题**，包括 Dijkstra 的思想、最短路径树以及 Floyd–Warshall 的动态规划；  
> 第四条是**最小支撑树（MST）**，包括 Prim、Kruskal 与并查集。  
> 这一章表面上像是若干图算法的拼盘，但真正的主线其实非常清楚：  
> **先用 DFS 理解图的结构，再用“优先级选点”统一看待一大类图优化算法。**  
> 主要依据你上传的课件内容：fileciteturn13file0

---

## 1. 从“图遍历”到“图应用”

上一章我们已经学过图的两类基础遍历：

- **BFS**：按层推进，天然适合无权最短路、层次距离
- **DFS**：一路深入后回溯，天然适合判环、边分类、拓扑排序

这一章的重要变化是：

> 我们不再满足于“把图走一遍”，而是希望借助遍历留下的结构信息，去解决更具体的问题。

课件把这些应用分成几个部分：

1. 双连通分量（BCC）
2. 优先级搜索（PFS）
3. Dijkstra 最短路径
4. Prim 最小支撑树
5. Kruskal 最小支撑树与并查集
6. Floyd–Warshall 全源最短路 fileciteturn13file0

乍看之下这些算法彼此相差很大，但实际上可归纳成两类思路：

### 1.1 结构分析型：DFS 留下的时间戳和回边信息

这类问题关注的是图的“结构脆弱性”或“分块结构”，例如：

- 哪个顶点一删就会把图切开？
- 图里哪些部分是“紧密粘在一起”的？
- 某个点是不是 articulation point（关节点）？

这就是 **BCC / articulation point** 问题。

### 1.2 最优化型：优先级驱动地扩展一棵树

这类问题关注的是：

- 下一步应该优先扩展哪个顶点？
- 优先级数应如何定义？
- 如何不断维护一棵“越来越好”的树？

这就是：

- Dijkstra
- Prim
- 以及它们共享的 **PFS（Priority First Search）** 框架

---

# 第一部分：双连通分量（BCC）与关节点

## 2. 什么是关节点与双连通分量

### 2.1 关节点（articulation point）

课件第 2 页给出定义：fileciteturn13file0

在无向图中，若删除某个顶点 `v` 之后，原图的连通分量数增加，则称 `v` 为：

> **关节点（articulation point / cut-vertex）**

直观理解：

- 这个点像图中的“交通枢纽”或“桥头堡”
- 一旦它消失，原本连在一起的若干块会被拆开

因此关节点衡量的是：

> **图对单点失效的脆弱程度**

---

### 2.2 双连通图（bi-connected graph）

课件第 2 页进一步定义：fileciteturn13file0

若一张图**没有关节点**，则称其为：

> **双连通图（bi-connected graph）**

意思是：

- 任意删去一个顶点
- 图仍保持连通

可以把它理解成“更稳固的连通”。

---

### 2.3 双连通分量（BCC）

课件第 2 页还给出：

> **极大的双连通子图**，称作双连通分量（Bi-Connected Component, BCC）。 fileciteturn13file0

这里“极大”表示：

- 不能再往里添加额外顶点 / 边而仍保持双连通

所以一张一般的无向图，可以被若干个 BCC 拼接起来；  
拼接点往往正是关节点。

于是我们可以把 BCC 看作：

> **图中“局部足够牢固”的块结构**

---

## 3. 蛮力做法为什么太慢

课件第 3 页先给出最直接思路：fileciteturn13file0

对每个顶点 `v`：

1. 暂时删除 `v`
2. 再检查剩余图是否连通
3. 若不连通，则 `v` 是关节点

由于一次连通性检查可用 BFS/DFS 在：

\[
O(n + e)
\]

时间内完成，而要对每个顶点都试一次，所以总时间是：

\[
O(n \cdot (n + e))
\]

课件明确指出：

> 太慢！fileciteturn13file0

而且即便只找出了关节点，也还没有真正求出各个 BCC。

这说明必须利用 DFS 遍历时顺手积累的信息，一次性把结构问题解决掉。

---

## 4. 关节点的 DFS 判定准则

课件第 4～5 页给出了判定关节点的核心直觉。fileciteturn13file0

## 4.1 DFS 树中的叶节点绝不可能是关节点

课件第 3 页已先提示这一点：fileciteturn13file0

若某顶点在 DFS 树中是叶子，则删去它：

- 不会把其后代子树切断（因为本来没有后代）
- 它也不是连接不同 DFS 子树的关键接口

因此 DFS 树叶节点不可能是 articulation point。

这提供了第一个很好的局部直觉：

> 是否是关节点，和 DFS 树中的子树结构密切相关。

---

## 4.2 根节点的判定条件

课件第 4 页指出：fileciteturn13file0

DFS 树根节点 `r` 成为关节点，当且仅当它至少有 **两棵子树**。

原因很直观：

- 若根只有一棵子树，则整张图都在这一棵 DFS 子树里
- 删去根后，剩余顶点仍可在这棵子树内部互相连通
- 但若根有两棵或更多子树，它们之间没有通过根以外的更高祖先可联通的机会
- 删除根就会把这些子树拆开

所以：

> **根节点的特殊判定条件：子树数至少 2。**

---

## 4.3 一般内部节点的判定条件

课件第 4 页进一步给出内部节点 `v` 的条件：fileciteturn13file0

若存在某个孩子 `u`，使得：

- `subtree(u)` 不能通过任何 BACKWARD 边
- 连接到 `v` 的任何真祖先 `a`

那么 `v` 就是关节点。

这句话可以翻译成更直观的说法：

> 若 `u` 这棵子树往上“爬”不到 `v` 以上的更高位置，那么它离开 `v` 就无路可走了。

于是删去 `v` 后：

- `u` 所在子树会和上方 / 旁侧图断开
- 因此 `v` 必是关节点

---

## 5. Highest Connected Ancestor：判定关节点的关键量

课件第 5 页引入了一个非常核心的量：fileciteturn13file0

\[
hca(v)
\]

表示：

> `subtree(v)` 经由后向边所能抵达的**最高祖先**

这相当于 Tarjan 风格 low-link 思想的一个版本。  
课件这里把它命名为 **Highest Connected Ancestor**。

---

### 5.1 为什么 dTime 越小，祖先越高

由 DFS 的括号引理和发现顺序可知：

- DFS 中越早发现的祖先，通常越“高”
- 因此比较 `dTime` 的大小即可判断谁更靠上

课件第 5 页明确写道：fileciteturn13file0

> `dTime` 越小的祖先，辈份越高

于是“最高祖先”就可以转化为：

> **能连到的最小 `dTime`**

---

### 5.2 更新规则一：遇到后向边

若 DFS 过程中发现一条后向边 `(v, u)`，则有：

\[
hca(v) = \min(hca(v), dTime(u))
\]

因为：

- 这说明 `v` 的子树可以通过该后向边爬到祖先 `u`
- 而 `u` 的 `dTime` 越小，说明越高

---

### 5.3 更新规则二：孩子回溯返回后

当 `DFS(u)` 完成并返回 `v` 时：

- 若 `u` 的子树还能通过后向边爬到比 `v` 更高的祖先
- 即：

\[
hca(u) < dTime(v)
\]

则 `v` 自己显然也可以借助 `u` 这棵子树做到这一点，于是：

\[
hca(v) = \min(hca(v), hca(u))
\]

---

### 5.4 何时判定 `v` 为关节点

课件第 5 页给出最终判据：fileciteturn13file0

若某个孩子 `u` 返回时满足：

\[
hca(u) \ge dTime(v)
\]

则说明：

- `u` 这棵子树最早只能回到 `v`
- 根本碰不到 `v` 的任何真祖先
- 所以删掉 `v` 后，这棵子树会被切开

于是 `v` 就是关节点，且：

> `{v} + subtree(u)` 构成一个 BCC

这就是整个 BCC 算法的核心判据。

---

## 6. BCC 算法：基于 DFS + 栈

课件第 7～9 页给出了代码框架。fileciteturn13file0

## 6.1 用 `fTime` 暂存 `hca`

课件第 7 页写了一个宏：fileciteturn13file0

```cpp
#define hca(x) ( fTime(x) ) //利用此处闲置的fTime
```

也就是说，在这个算法阶段：

- `fTime` 暂时不再表示 DFS 完成时间
- 而是复用来存 `hca`

这是一个很常见的工程技巧：

> 若某字段在当前算法中暂时不用，就可以复用为别的辅助量，节省额外存储。

---

## 6.2 栈 `S` 的作用

课件第 7 页中算法签名为：fileciteturn13file0

```cpp
void Graph<Tv, Te>::BCC( Rank v, int & clock, Stack<Rank> & S )
```

并在入口就做：

```cpp
hca(v) = dTime(v) = ++clock;
status(v) = DISCOVERED;
S.push(v);
```

这说明栈 `S` 始终保存：

> 当前 DFS 路径上、尚未被归入某个完整 BCC 的顶点

当某个 BCC 被确认时，就从栈顶弹出一段顶点。

---

## 6.3 遇到未发现孩子 `u`

课件第 8 页给出 `UNDISCOVERED` 分支：fileciteturn13file0

```cpp
parent(u) = v; 
type(v, u) = TREE;
BCC(u, clock, S);

if ( hca(u) < dTime(v) )
    hca(v) = min(hca(v), hca(u));
else
    while ( u != S.pop() );
```

这段代码正对应前面的判定逻辑：

### 情况一：`hca(u) < dTime(v)`

说明 `u` 子树可绕回 `v` 的真祖先，  
所以 `v` 还不能在这里截断成一个新 BCC。

### 情况二：`hca(u) >= dTime(v)`

说明 `v` 是关节点，  
并且从栈顶到 `u` 为止的这一批顶点，加上 `v`，恰好形成一个 BCC。

于是通过：

```cpp
while ( u != S.pop() );
```

把该 BCC 中除 `v` 外的所有顶点弹出。

---

## 6.4 遇到后向边

课件第 9 页给出 `DISCOVERED` 分支：fileciteturn13file0

```cpp
type(v, u) = BACKWARD;
if ( u != parent(v) )
    hca(v) = min( hca(v), dTime(u) );
```

注意这里专门排除了：

```cpp
u == parent(v)
```

因为在无向图中，DFS 树边是双向可见的：

- 从 `v` 看到 `parent(v)` 这条边并不算真正的后向边
- 否则每条树边都会错误地被当成一个回边

所以必须排除它。

---

## 6.5 复杂度

课件第 10 页总结：fileciteturn13file0

- 运行时间与常规 DFS 相同：

\[
O(n + e)
\]

- 辅助栈操作总成本也不过如此
- 除原图本身外，只需：
  - 一个 `O(e)` 规模的边栈 / 顶点栈（依具体实现）
  - 一个 `O(n)` 的递归调用栈

这说明 BCC 虽然功能很强，但复杂度仍然保持在线性级别，是非常高效的。

---

## 7. BCC 的算法直觉总结

到这里，可以把 BCC 的核心思想浓缩成一句话：

> **DFS 一边向下探测，一边问：这棵子树是否还找得到一条“逃回更高祖先”的路？**

- 如果能逃回去，说明它和更高部分仍然黏在一起，暂时不能切开
- 如果逃不回去，说明当前节点就是唯一咽喉点，形成新的 BCC 边界

这是一种非常典型的“DFS + low-link / hca”思维，在后面的强连通分量、桥、割点等问题里会一再出现。

---

# 第二部分：优先级搜索 PFS

## 8. 为什么 BFS、Dijkstra、Prim 可以统一起来

课件第 18 页开宗明义：fileciteturn13file0

> 各种遍历算法的区别，仅在于选取顶点进行访问的次序。

也就是说：

- BFS 的区别在于：优先扩展“最早被发现”的顶点
- DFS 的区别在于：优先扩展“最近被发现”的顶点
- Dijkstra 的区别在于：优先扩展当前距离源点最短的顶点
- Prim 的区别在于：优先扩展跨越当前割的最短边对应顶点

于是它们都可抽象为：

> **每个尚未加入树的顶点都带一个 priority，  
> 每一步选择优先级最高（或数值最小）的顶点加入树。**

---

## 8.1 Bag 视角

课件第 18 页指出：fileciteturn13file0

- 不同算法，本质上对应不同的顶点选取策略
- 不同选取策略，又取决于“存放与提供顶点”的数据结构 Bag

这其实已经把图遍历抽象成了一个更一般的框架：

- 顶点集合存在一个候选池里
- 每轮从中取一个“当前最合适的”
- 加入遍历树
- 再更新其它顶点优先级

若 Bag 用：

- 队列 → BFS
- 栈 → DFS
- 最小优先队列 → Dijkstra / Prim

就会得到不同算法。

---

## 8.2 PFS 统一框架

课件第 19～20 页给出统一模板：fileciteturn13file0

```cpp
template <typename Tv, typename Te>
template <typename PU>
void Graph<Tv, Te>::pfs( Rank s, PU prioUpdater ) {
    priority(s) = 0;
    status(s) = VISITED;
    parent(s) = -1;
    while (1) {
        for ( Rank w = firstNbr(s); -1 < w; w = nextNbr(s, w) )
            prioUpdater(this, s, w);

        for ( int shortest = INT_MAX, w = 0; w < n; w++ )
            if ( UNDISCOVERED == status(w) )
                if ( shortest > priority(w) ) {
                    shortest = priority(w);
                    s = w;
                }

        if ( VISITED == status(s) ) break;
        status(s) = VISITED;
        type( parent(s), s ) = TREE;
    }
}
```

这段代码非常值得仔细理解。

---

## 8.3 PFS 每轮到底做了什么

每一轮循环做两件事：

### 第一步：基于当前顶点 `s` 更新其邻居优先级

即：

```cpp
prioUpdater(this, s, w)
```

它的意义取决于具体算法策略。

### 第二步：在所有尚未加入树的顶点中，选出优先级最高者

然后把该顶点加入当前遍历树。

于是整棵 PFS 树是：

- 从源点 `s` 出发
- 每步挑一个当前最“值得”加入的顶点
- 再基于它修正邻居的估计值

---

## 8.4 复杂度

课件第 21 页分析：fileciteturn13file0

若直接按模板实现：

- 前一内循环：
  - 邻接矩阵下为 `O(n^2)`
  - 邻接表下为 `O(n+e)`
- 后一循环每次都要在线性扫描中选最优顶点，总计：

\[
O(n^2)
\]

因此总复杂度是：

\[
O(n^2)
\]

课件还指出：

> 若改用优先级队列，则可改进为

\[
O((e+n)\log n)
\]

虽然对稠密图未必划算，但对稀疏图是很大改进。fileciteturn13file0

---

# 第三部分：Dijkstra 单源最短路径

## 9. 最短路径问题分类

课件第 23～24 页先给出问题背景：fileciteturn13file0

在带权有向图中，希望找到：

- 从 `u` 到 `v` 的最短路径及其长度

典型应用包括：

- 旅游中的最省成本路线
- 路由器最快转发
- 机器人路径规划

---

## 9.1 SSSP 与 APSP

课件第 24 页区分两类问题：fileciteturn13file0

### 单源最短路径（SSSP）

给定一个源点 `s`，求它到其余所有顶点的最短路径。  
代表算法：

- **Dijkstra（1959）**

### 全源最短路径（APSP）

求任意两点间最短路径。  
代表算法：

- **Floyd–Warshall（1962）**

---

## 9.2 无权图为什么 BFS 就够了

课件第 24 页也特别指出：fileciteturn13file0

- 若图是无权（或等权）图，最短路径可直接用 BFS
- 因为 BFS 已经保证最少边数的层次最短路

所以 Dijkstra 真正处理的是：

> **带非负权的单源最短路径**

---

## 10. Dijkstra 的核心性质

## 10.1 最短路径的前缀仍是最短路径

课件第 27 页给出最重要的结构性质：fileciteturn13file0

> 任一最短路径的前缀，也是一条最短路径。

这听起来简单，但它正是 Dijkstra 的理论基础。

因为这说明：

- 若某个顶点 `v` 已经以最短方式被确定
- 那么从 `v` 再向外延伸，才有意义
- 否则基于非最优前缀做扩展，整体不可能最优

---

## 10.2 为什么要求边权非负

课件第 28 页明确强调：fileciteturn13file0

- 若存在负权环，则最短路径根本无定义
- 即便没有负权环，只要有负权边，Dijkstra 也可能失效

原因在于 Dijkstra 的贪心假设是：

> 当前看起来最短的那个顶点，以后不会再被更短路径改写

而负权边可能让“以后绕一圈再回来”反而更便宜，这会破坏贪心正确性。

因此：

> **Dijkstra 只适用于边权非负的图。**

---

## 11. 最短路径树（SPT）

课件第 29 页指出：fileciteturn13file0

所有从源点 `s` 出发的最短路径之并：

- 连通
- 无环

因此构成一棵树，叫：

> **Shortest Path Tree, SPT**

这和 BFS 树很像，但权重条件不同。

---

### 11.1 SPT 不等于 MST

课件第 30 页特别提醒：fileciteturn13file0

\[
SPT 
e MST
\]

这一点非常容易混淆。

- **SPT** 关注：从源点 `s` 到每个点 individually 最短
- **MST** 关注：整棵支撑树边权总和最小

它们优化目标完全不同，所以一般不会相同。

---

## 12. Dijkstra 的贪心过程

课件第 32～35 页用“减而治之、选取、更新”的图示讲解了 Dijkstra 的核心过程。fileciteturn13file0

可以把它概括成两步循环：

### 12.1 选取（extract-min）

从所有尚未加入 SPT 的顶点中，选出当前估计距离最小者 `s`。  
此时可断言：

> `s` 的最短距离已经最终确定。

### 12.2 更新（relaxation）

枚举 `s` 的每个邻接顶点 `w`，尝试通过 `s` 去改进它的最短距离估计：

\[
dist(w) \leftarrow \min(dist(w), dist(s) + weight(s,w))
\]

这就叫：

> **松弛（relaxation）**

---

## 13. Dijkstra 如何嵌入 PFS 框架

课件第 42～43 页指出，Dijkstra 与 PFS 完全兼容：fileciteturn13file0

- PFS 中 `priority(v)` 就表示当前源点到 `v` 的最短距离估计
- 每次选优先级最高者，等价于选最小 `priority`
- `prioUpdater` 则实现标准松弛

---

### 13.1 Dijkstra 的 `prioUpdater`

课件第 43 页给出：fileciteturn13file0

```cpp
template <typename Tv, typename Te> struct DijkPU {
    virtual void operator()( Graph<Tv, Te>* g, Rank s, Rank w ) {
        if ( UNDISCOVERED != g->status(w) ) return;
        if ( g->priority(w) > g->priority(s) + g->weight(s, w) ) {
            g->priority(w) = g->priority(s) + g->weight(s, w);
            g->parent(w) = s;
        }
    }
};
```

这段代码非常标准：

- 若 `w` 还未确定
- 且通过 `s` 到 `w` 更短
- 就更新 `priority(w)` 与 `parent(w)`

---

## 14. Dijkstra 的复杂度与实现取舍

课件沿用 PFS 的复杂度分析：fileciteturn13file0

### 14.1 朴素实现

- 邻接矩阵 + 顺序扫描最小 `priority`
- 总复杂度：

\[
O(n^2)
\]

这在稠密图下是合理的，甚至常常更直接。

### 14.2 优先队列实现

若图较稀疏，则可用最小堆维护尚未确定的顶点：

\[
O((n+e)\log n)
\]

这在工程上更常见。

---

# 第四部分：最小支撑树 MST

## 15. MST 的问题背景

课件第 45～46 页给出最小支撑树的定义与应用：fileciteturn13file0

给定连通带权网络：

\[
N = (V, E)
\]

希望找到一棵支撑树 `T = (V, F)`，满足：

- 覆盖所有顶点
- 本身是一棵树
- 边权总和最小

这就是：

> **Minimum Spanning Tree, MST**

典型应用：

- 电信网络设计
- 电路布线
- 各种近似算法的基础结构
- 几何近邻结构等

---

## 15.1 边权可以为负吗

课件第 47 页讨论了一个容易忽略的问题：fileciteturn13file0

MST 中边权：

- 可以为零
- 甚至可以为负

因为所有支撑树边数都固定为：

\[
|V|-1
\]

所以统一平移所有边权，不会改变谁是最优。  
这与最短路完全不同——最短路一旦有负环就会出问题，而 MST 不会。

---

## 16. Prim 算法：极短跨边

## 16.1 Cut Property

课件第 50 页给出 Prim 正确性的核心依据：fileciteturn13file0

设 `(U, V \ U)` 是网络的一个割。  
若边 `uv` 是该割上的一条**极短跨边**（minimum crossing edge），则：

> 必存在一棵 MST 包含 `uv`

这就是经典的 **割性质（Cut Property）**。

直观理解：

- 割把顶点分成两边
- 若想把两边重新连起来，总得选一条跨过去的边
- 最短那条边一定是“安全的”

---

## 16.2 Prim 的递增构造

课件第 52 页总结 Prim 思路：fileciteturn13file0

1. 任选一个起点
2. 当前已有树顶点集记为 `V_k`
3. 把 `(V_k, V\V_k)` 视为一个割
4. 从所有跨边中选取最短者
5. 把其对应新顶点加入树
6. 重复直到包含全部顶点

所以 Prim 本质是：

> **始终把当前树向外“长”一步，每次选最短跨边。**

---

## 16.3 Prim 如何嵌入 PFS

课件第 61～62 页说明 Prim 与 PFS 的统一：fileciteturn13file0

- `priority(w)` 表示当前割中连接 `w` 的最短跨边权重
- 每次选 `priority` 最小的未收录顶点
- `prioUpdater` 在扫描新加入顶点 `s` 的邻边时，尝试更新这些权重估计

---

### 16.4 Prim 的 `prioUpdater`

课件第 62 页给出：fileciteturn13file0

```cpp
template <typename Tv, typename Te> struct PrimPU {
    virtual void operator()( Graph<Tv, Te>* g, Rank s, Rank w ) {
        if ( UNDISCOVERED != g->status(w) ) return;
        if ( g->priority(w) > g->weight(s, w) ) {
            g->priority(w) = g->weight(s, w);
            g->parent(w) = s;
        }
    }
};
```

与 Dijkstra 对比可见：

- Dijkstra：更新的是 `priority(s) + weight(s,w)`
- Prim：只看边自身权重 `weight(s,w)`

这正反映了两者目标不同：

- Dijkstra 关心源点到 `w` 的整条路径长度
- Prim 只关心把 `w` 接入当前树所需的最短跨边

---

## 17. Kruskal 算法：从小到大挑边

## 17.1 贪心原则

课件第 64～65 页给出 Kruskal 的思路：fileciteturn13file0

1. 把所有边按权重从小到大排序
2. 初始化时，每个顶点各自是一棵独立树，形成森林
3. 按顺序考察每条边 `e=(u,v)`
4. 若 `u,v` 属于不同树，则这条边安全，加入森林
5. 同时把两棵树合并
6. 直到加入了 `n-1` 条边

这就是：

> **能加就加，但绝不形成环。**

---

## 17.2 为什么 Kruskal 是对的

课件第 66 页指出：fileciteturn13file0

当 Kruskal 选择一条边 `e=(u,v)` 来合并两棵树时：

- 把当前其中一棵树 `T` 与其余部分 `V\T` 看成一个割
- 边 `e` 正是该割当前能选到的最短跨边

因此由割性质可知，这条边是安全的。  
课件也提醒：要做严格证明，仍然需要归纳论证“当前森林始终是某棵 MST 的子图”。

---

## 18. 并查集：Kruskal 的数据结构核心

课件第 67～75 页把 Kruskal 与并查集联系起来：fileciteturn13file0

## 18.1 Union-Find 问题

课件第 68 页定义：

- `Find(x)`：找到 `x` 所属等价类 / 连通块代表
- `Union(x, y)`：合并 `x` 与 `y` 所属等价类

Kruskal 正好需要它来回答：

- 某条边两端是否已在同一连通块中？
- 若不在，合并两个连通块

于是：

> **Kruskal = 排序 + 并查集**

---

## 18.2 Quick-Find

课件第 69 页先给出 `Quick-Find`：fileciteturn13file0

```python
group[k] 记录元素所属子集编号
find(k): return group[k]
union(i,j): 扫描全部元素，把 jGroup 改成 iGroup
```

特点：

- `find` 很快：`O(1)`
- `union` 很慢：最坏 `O(n)`

---

## 18.3 Quick-Union

课件第 72～73 页进一步改成 parent forest：fileciteturn13file0

- 每个集合是一棵代表树
- `find(x)` 沿 parent 指针找到根
- `union(x,y)` 把一棵树根挂到另一棵树根下

这样：

- `union` 更自然
- 但若树太高，`find` 会变慢

---

## 18.4 加权合并 + 路径压缩

课件第 74～75 页给出两大经典优化：fileciteturn13file0

### 按秩 / 按规模合并（weighting / rank）

总是把较小树挂到较大树下，避免高度失控。

### 路径压缩（path compression）

在执行 `find(x)` 时，把路径上所有节点直接压到根上，后续查找更快。

这两者结合后，并查集的均摊复杂度几乎可看作常数。  
因此 Kruskal 总复杂度主要由“边排序”主导：

\[
O(e \log e)
\]

通常等价写成：

\[
O(e \log n)
\]

---

# 第五部分：Floyd–Warshall 全源最短路径

## 19. 为什么还需要 Floyd–Warshall

课件第 77 页提出问题：fileciteturn13file0

若要找全源最短路 APSP，一个直观想法是：

- 对每个源点都运行一次 Dijkstra

若采用 `O(n^2)` 版 Dijkstra，则总成本为：

\[
n 	imes O(n^2) = O(n^3)
\]

这已经是一个有效方案。

那为什么还需要 Floyd–Warshall？

课件自己回答：

- 形式简单
- 算法紧凑
- 便于实现
- 允许负权边（只要没有负权环） fileciteturn13file0

所以 FW 的优势不一定在渐进复杂度，而在：

> **动态规划表达非常优雅，且适用范围更宽。**

---

## 19.1 图矩阵到最短路矩阵

课件第 77 页还提示了一个关键观察：fileciteturn13file0

> 图矩阵 ~ 最短路径矩阵

也就是说：

- 邻接矩阵描述的是一步边权
- Floyd–Warshall 最终计算的是任意两点之间的最短距离矩阵

这是一个从“原始边关系”到“全局可达最优关系”的闭包过程。

---

## 20. Floyd–Warshall 的动态规划定义

课件第 78 页给出核心定义：fileciteturn13file0

把所有顶点编号为：

\[
1,2,\dots,n
\]

定义：

\[
dist_k(u,v)
\]

表示：

> 只允许使用前 `k` 个顶点作为中转点时，从 `u` 到 `v` 的最短路径长度

这就是 Floyd–Warshall 最本质的状态定义。

---

### 20.1 为什么这样定义状态非常自然

因为任意最短路径只有两种可能：

1. **不经过第 `k` 个顶点**
2. **至少经过一次第 `k` 个顶点**

若不经过，则答案就是：

\[
dist_{k-1}(u,v)
\]

若经过，则可分解成：

\[
dist_{k-1}(u,k) + dist_{k-1}(k,v)
\]

于是立刻得到递推。

---

## 21. 从暴力递归到动态规划

课件第 79 页先写出一个递归式：fileciteturn13file0

```cpp
dist(u, v, k):
    if k < 1 return w(u, v)
    u2v = dist(u, v, k-1)
    for each x ...
        u2x2v = dist(u, x, k-1) + dist(x, v, k-1)
        u2v = min(u2v, u2x2v)
```

课件紧接着指出：

> 存在大量重复递归调用，怎么办？

答案当然是：

> **记忆化 / 动态规划**

---

## 22. Floyd–Warshall 的三重循环

课件第 80 页给出最终形式：fileciteturn13file0

```cpp
for k in range(0, n)
    for u in range(0, n)
        for v in range(0, n)
            A[u][v] = min( A[u][v], A[u][k] + A[k][v] )
```

这就是最经典的 Floyd–Warshall。

---

## 22.1 三重循环的含义

### 最外层 `k`

表示：

- 当前允许使用前 `k` 个顶点作为中转点

### 中间层 `u`

枚举源点

### 最内层 `v`

枚举终点

于是每次更新都在尝试：

> 要不要让路径 `u -> v` 经过 `k` 这个新开放的中转点？

---

## 22.2 复杂度

三重循环显然给出总时间：

\[
O(n^3)
\]

空间上若只维护距离矩阵：

\[
O(n^2)
\]

这与“跑 `n` 次 Dijkstra”在稠密图上的量级是同阶的，  
但形式上更紧凑统一。

---

## 22.3 Floyd–Warshall 的优点与适用场景

可以把 FW 的优势概括为：

1. 代码极简
2. 适合邻接矩阵
3. 可直接处理负权边（无负环）
4. 适合中小规模稠密图
5. 可方便求图的“中心点”等全局性质

课件第 77 页就提到“图的中心点”：fileciteturn13file0

- 对每个顶点 `s`，定义它的半径为：
  - 所有顶点到 `s` 的最大距离
- 半径最小者即图中心

这类问题显然需要全源最短路矩阵。

---

# 第六部分：本章整体串联

## 23. 这一章真正想传达什么

这一章内容不少，但不是杂乱无章。  
把主线捋顺后，会发现它其实非常统一。

---

## 23.1 第一条主线：DFS 不只是遍历工具，也是结构分析工具

通过 `dTime` 与 `hca / low-link`，DFS 能回答：

- 哪些点是 articulation point
- 图如何分解成 BCC
- 哪些局部结构是脆弱的

所以 DFS 不只是“走一遍图”，更是在给图做结构诊断。

---

## 23.2 第二条主线：很多图算法都可以看成 PFS

PFS 告诉我们：

> 图上的很多“树生长算法”都只是在回答同一个问题：
> 
> **下一步该把哪个顶点加入当前树？**

不同算法只是在“优先级定义”上不同：

- BFS：按层次 / 队列顺序
- Dijkstra：按当前最短路估计
- Prim：按当前最短跨边估计

这是一种非常统一的抽象视角。

---

## 23.3 第三条主线：MST 与最短路虽然都在“选边”，但目标完全不同

- Dijkstra / SPT：
  - 从源点到每个点 individually 最短
- Prim / Kruskal / MST：
  - 整棵支撑树总权重最小

它们都生成树，但生成标准不同。  
课件专门提醒 `SPT ≠ MST`，就是要防止混淆。 fileciteturn13file0

---

## 23.4 第四条主线：局部贪心 + 全局安全性

这一章几乎所有经典算法都在体现一种模式：

### Dijkstra

- 当前最短的未确定顶点是安全的

### Prim

- 当前割上的极短跨边是安全的

### Kruskal

- 当前最短且不成环的边是安全的

### Floyd–Warshall

- 当前开放的新中转点，可用 DP 安全地增量纳入

也就是说，这一章实质上也是：

> **图算法中的贪心与动态规划范式入门。**

---

## 24. 本章主线总结

### 24.1 双连通分量问题说明 DFS 时间戳很有力量

通过 `dTime + hca/low-link`，  
一次 DFS 就能在线性时间内识别：

- articulation point
- BCC 边界

### 24.2 PFS 是统一框架

它把一类“不断生长树”的图算法统一为：

- 维护 priority
- 每轮取最优顶点
- 再更新其邻接点优先级

### 24.3 Dijkstra 与 Prim 只是 priority 含义不同

- Dijkstra：当前最短路估计
- Prim：当前最短跨边估计

代码形态几乎完全一样，只差 `prioUpdater`

### 24.4 Kruskal 则是“按边排序”的另一种 MST 思路

- Prim：按顶点扩张
- Kruskal：按边合并森林

其核心数据结构是并查集。

### 24.5 Floyd–Warshall 展示了图算法中的动态规划

- 通过“允许的中转点集合”逐步扩张状态
- 用三重循环完成 APSP

---

## 25. 一句话总结

> 这一章真正重要的，不只是记住 BCC、Dijkstra、Prim、Kruskal、Floyd–Warshall 这些算法名字，而是理解：DFS 能揭示图的脆弱结构，PFS 能统一一大类优先扩张算法，而最短路与最小支撑树问题则展示了图算法中贪心与动态规划两种最核心的设计思想。
