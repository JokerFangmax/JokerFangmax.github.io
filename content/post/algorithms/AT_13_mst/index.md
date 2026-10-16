---
title: "AT 13: 最小生成树——Kruskal、Prim 与 PFS 统一视角"
date: 2026-12-06
math: true
tags:
  - Notes
  - 算法模板
  - 图论
  - 最小生成树
  - 并查集
categories:
  - 算法模板
description: "最小生成树双模板：Kruskal（排序 + 并查集，稀疏图首选）与 Prim（朴素 O(n²)，稠密/完全图首选），以及 PFS 统一视角——BFS、Dijkstra、Prim 是同一个框架换优先级函数，Dijkstra 与 Prim 的差别只有一行。样题：洛谷 P3366 模板、LC 1584 连接所有点的最小费用，附题目清单"
---

# AT 13：最小生成树——Kruskal、Prim 与 PFS 统一视角

> 本篇是 DSA 刷题模板系列（[AT 00 导读](../AT_00_reading_guide/)）的第 13 篇。核心思想一句话：**稀疏图 Kruskal（排序 + 并查集），稠密图/完全图 Prim 朴素版 $O(n^2)$，堆优化 Prim 几乎永远不写**。
> 理论后盾：[DSA 10 §5](../../DSA/DSA_10/)（支撑树与 MST 概念）、[DSA 11 §15–18](../../DSA/DSA_11/)（割性质、Prim、Kruskal、并查集）。存图来自 [AT 11](../AT_11_graph_bfs_dfs/)。

---

## 1. 这个模板解决什么问题

**最小生成树（Minimum Spanning Tree, MST）**：无向带权连通图里，选 $n-1$ 条边把所有顶点连起来，使边权总和最小。与最短路的区别一句话讲清（DSA 11 §23.3 的主线）：**最短路关心"每个点到源点各走多远"，MST 关心"整体骨架的总造价"**——两者的贪心都是"局部最短 + 全局安全"，但安全性的依据不同。

| 场景 | 选什么 | 复杂度 | 一句话理由 |
|------|--------|--------|-----------|
| 一般稀疏图（$m$ 与 $n$ 同阶） | Kruskal | $O(m \log m)$ | 排序是瓶颈，并查集近乎常数 |
| 稠密图 / 完全图（$m \approx n^2$） | Prim 朴素 | $O(n^2)$ | 不用存边表，邻接矩阵/现算距离即可 |
| 要输出"割边结构" / 边按权分批处理 | Kruskal | — | 边排序的天然顺序本身有用 |
| 堆优化 Prim | 一般不写 | $O(m \log n)$ | 与堆优化 Dijkstra 同款复杂度，还更难写对 |

两个算法的正确性都靠**割性质（Cut Property）**（DSA 11 §16.1）：把顶点分成两半，跨过这个割的最短边必然属于某棵 MST——"局部最短的跨边是安全的"。

---

## 2. 代码模板

### 2.1 Kruskal：排序 + 并查集

```cpp
#include <bits/stdc++.h>
using namespace std;

struct DSU {                                   // 并查集：路径压缩 + 按秩合并
    vector<int> f, r;
    DSU(int n) : f(n + 1), r(n + 1, 0) { iota(f.begin(), f.end(), 0); }
    int find(int x) { return f[x] == x ? x : f[x] = find(f[x]); }
    bool unite(int a, int b) {                 // 返回 false 表示本来就在同一集合
        a = find(a); b = find(b);
        if (a == b) return false;
        if (r[a] < r[b]) swap(a, b);
        f[b] = a;
        if (r[a] == r[b]) r[a]++;
        return true;
    }
};

struct Edge { int u, v, w; };

long long kruskal(int n, vector<Edge>& edges) {   // 返回 MST 总权；不连通返回 -1
    sort(edges.begin(), edges.end(),
         [](const Edge& a, const Edge& b){ return a.w < b.w; });
    DSU dsu(n);
    long long total = 0;
    int cnt = 0;                               // 已选边数
    for (auto& e : edges) {
        if (dsu.unite(e.u, e.v)) {             // 两端不同集合 → 这条边安全
            total += e.w;
            if (++cnt == n - 1) break;         // 提前收工
        }
    }
    return cnt == n - 1 ? total : -1;          // 收不齐 = 图不连通
}
```

框架就是 DSA 11 §17 的四句口诀：**边按权排序 → 能加就加 → 加了就并 → 凑满 $n-1$ 条收工**。`unite` 返回 `false`（两端已连通）意味着这条边会成环，跳过它。并查集细节见 [AT 10（并查集）](../AT_10_union_find/)。

### 2.2 Prim 朴素版（稠密图 / 完全图首选）

```cpp
long long prim_n2(int n, vector<vector<long long>>& g) {   // g: 邻接矩阵，无边 = INF
    vector<long long> low(n + 1, LLONG_MAX);   // low[v]: v 接入树的最短跨边
    vector<bool> in(n + 1, false);             // 是否已进树
    long long total = 0;
    low[1] = 0;                                // 从顶点 1 长起（任选）
    for (int i = 1; i <= n; i++) {             // 长出 n 个顶点
        int u = 0;                             // 未进树顶点中 low 最小者
        for (int v = 1; v <= n; v++)
            if (!in[v] && low[v] < low[u]) u = v;    // low[0] = LLONG_MAX 作哨兵
        if (low[u] == LLONG_MAX) return -1;    // 剩下的点连不上（不连通）
        in[u] = true;
        total += low[u];
        for (int v = 1; v <= n; v++)           // 用新顶点更新所有跨边
            low[v] = min(low[v], g[u][v]);
    }
    return total;
}
```

`low[v]` 的语义是**当前割 $(树内, 树外)$ 上连接 $v$ 的最短跨边**——每次选最小的 `low` 进树，正是"极短跨边"贪心。完全图（如 LC 1584 的平面点集）不需要邻接矩阵，`g[u][v]` 现算即可。

### 2.3 PFS 统一视角：三个算法一个框架

DSA 11 §8 的核心观点，值得在刷题层面兑现一遍：**BFS、Dijkstra、Prim 都是"每轮从候选池里挑当前最值得的顶点进树"**，区别只在"最值得"怎么定义、候选池用什么结构：

| 算法 | 候选池（Bag） | 优先级含义 | prioUpdater（更新邻居） |
|------|--------------|-----------|------------------------|
| BFS | 队列 | 先发现者优先（层数） | $prio(w) \gets prio(s) + 1$ |
| DFS | 栈 | 后发现者优先 | $prio(w) \gets$ 更晚的时间戳 |
| Dijkstra | 小根堆 | $prio = $ 到源点距离 | $prio(w) \gets \min(prio(w),\; prio(s) + w(s,w))$ |
| Prim | 小根堆（朴素版换线性扫描） | $prio = $ 最短跨边 | $prio(w) \gets \min(prio(w),\; w(s,w))$ |

看最后两行：**Dijkstra 与 Prim 的 prioUpdater 只差一个 `prio(s) +`**——Dijkstra 累积整条路径的长度（关心"到源点多远"），Prim 只看这一条边本身（关心"接进树多贵"）。这就是 §2.2 的 `low[v] = min(low[v], g[u][v])` 与 Dijkstra 的 `dist[v] = dist[u] + w` 的全部区别。背模板时把这两行对照着背，永远不会张冠李戴。

---

## 3. 常见变形与坑

| 变形 | 一句话做法 |
|------|-----------|
| 图不连通 → 最小生成森林 | Kruskal 天然支持：跑完取所有"加进来的边"，各连通分量各得一棵树 |
| 次小生成树 | 先求 MST，再枚举非树边替换环上最大边 |
| 最大边最小化（瓶颈生成树） | MST 就是瓶颈最小的生成树，Kruskal 途中第一条连通两端的边即答案 |
| "删除边权和最大"类题 | 总权 − MST 权 = 可删去的最大权（见清单 P2820） |
| 完全图 + 点坐标 | Prim 朴素 $O(n^2)$，边不落盘、现算距离 |
| 边权相等的多解 | 模板不区分同权边，题目若要求字典序再特判 |

坑清单：

1. **忘了图可能不连通**：Kruskal 数边数（$n-1$ 才成功），Prim 检查 `low[u] == INF`——洛谷 P3366 对不连通输出 `orphan`，不判直接 WA；
2. **边权可以为负**：MST 的贪心不依赖"非负"假设（DSA 11 §15.1），负权照跑——别把 Dijkstra 的限制搬过来；
3. **重边与自环**：Kruskal 自然处理（自环 `unite` 必返 `false`）；邻接矩阵存重边记得取 `min`；
4. **MST ≠ 最短路树**：从同一点出发，MST 上两点的路径**不是**最短路（DSA 11 §11.1 有反例）——求"各点到源最短路"请回 [AT 12](../AT_12_shortest_path/)；
5. **无向图 Kruskal 的边只存一条**（它不像遍历要双向走），但并查集合并是双向的；
6. **多测清空**：`DSU`、边表、`low/in` 每组数据都要重建。

---

## 4. 样题精讲

### 4.1 洛谷 P3366.【模板】最小生成树

> 题目：[https://www.luogu.com.cn/problem/P3366](https://www.luogu.com.cn/problem/P3366)。$n$ 个顶点 $m$ 条边的无向图，输出最小生成树的边权和；若图不连通输出 `orphan`。

模板题，§2.1 抄全加上 `main`：

```cpp
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> f, r;
    DSU(int n) : f(n + 1), r(n + 1, 0) { iota(f.begin(), f.end(), 0); }
    int find(int x) { return f[x] == x ? x : f[x] = find(f[x]); }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (r[a] < r[b]) swap(a, b);
        f[b] = a;
        if (r[a] == r[b]) r[a]++;
        return true;
    }
};
struct Edge { int u, v, w; };

int main() {
    int n, m;
    scanf("%d%d", &n, &m);
    vector<Edge> edges(m);
    for (auto& e : edges) scanf("%d%d%d", &e.u, &e.v, &e.w);
    sort(edges.begin(), edges.end(),
         [](const Edge& a, const Edge& b){ return a.w < b.w; });
    DSU dsu(n);
    long long total = 0;
    int cnt = 0;
    for (auto& e : edges) {
        if (dsu.unite(e.u, e.v)) {
            total += e.w;
            if (++cnt == n - 1) break;
        }
    }
    if (cnt == n - 1) printf("%lld\n", total);
    else printf("orphan\n");
    return 0;
}
```

复杂度 $O(m \log m)$。注意结构化绑定与 `auto& e : edges` 配合时 `e.u` 直接可读——比下标循环更不易写错边端点。

### 4.2 LC 1584. 连接所有点的最小费用（完全图 → Prim 朴素）

> 题目：[https://leetcode.cn/problems/min-cost-to-connect-all-points/](https://leetcode.cn/problems/min-cost-to-connect-all-points/)（中等）。平面上 $n$ 个点，连接两点的费用是**曼哈顿距离** $|x_i - x_j| + |y_i - y_j|$，求把所有点连通的最小总费用。

**思路**：这是 $n$ 个点的完全图（$n^2$ 条边）——**完全图上的 MST 首选 Prim 朴素版**：边不落盘，`low[v]` 的更新直接现算曼哈顿距离，省掉 $O(n^2)$ 的边表与排序：

```cpp
class Solution {
public:
    int minCostConnectPoints(vector<vector<int>>& points) {
        int n = points.size();
        vector<int> low(n, INT_MAX);           // low[v]: v 接入树的最短跨边
        vector<bool> in(n, false);
        int total = 0;
        low[0] = 0;
        for (int i = 0; i < n; i++) {          // 长出 n 个点
            int u = -1;
            for (int v = 0; v < n; v++)
                if (!in[v] && (u == -1 || low[v] < low[u])) u = v;
            in[u] = true;
            total += low[u];
            for (int v = 0; v < n; v++) {      // 现算完全边权更新跨边
                if (in[v]) continue;
                int w = abs(points[u][0] - points[v][0])
                      + abs(points[u][1] - points[v][1]);
                low[v] = min(low[v], w);
            }
        }
        return total;
    }
};
```

复杂度 $O(n^2)$、空间 $O(n)$。**选型判断是本题考点**：如果条件反射地建边表跑 Kruskal，$n = 1000$ 时要建百万条边、排序百万级——能过但难看；`low[v] = min(low[v], w)` 这一行就是 §2.3 表里 Prim 的 prioUpdater。

---

## 5. 题目清单

| 题号 | 题名 | 平台/难度 | 考的变形 | 链接 |
|------|------|----------|---------|------|
| P3366 | 【模板】最小生成树 | 洛谷 普及+ | Kruskal 模板 + 不连通判定 | [题目](https://www.luogu.com.cn/problem/P3366) |
| P1546 | [USACO3.1] 最短网络 Agri-Net | 洛谷 普及+ | 邻接矩阵输入，Prim 小数据 | [题目](https://www.luogu.com.cn/problem/P1546) |
| P2820 | 局域网 | 洛谷 普及+ | 总权 − MST = 最大可删边权 | [题目](https://www.luogu.com.cn/problem/P2820) |
| LC 1584 | 连接所有点的最小费用 | LeetCode 中等 | 完全图 Prim $O(n^2)$ | [题目](https://leetcode.cn/problems/min-cost-to-connect-all-points/) |
| LC 743 | 网络延迟时间 | LeetCode 中等 | 对照题：最短路 vs MST 的选型 | [题目](https://leetcode.cn/problems/network-delay-time/) |

---

## 6. 与其他篇的联系

- Kruskal 的并查集完整模板（带权并查集、扩展域）在 [AT 10（并查集）](../AT_10_union_find/)；
- "Bag 换结构 = 换算法"的 PFS 统一视角出自 [DSA 11 §8](../../DSA/DSA_11/)；Dijkstra 模板见 [AT 12（最短路）](../AT_12_shortest_path/)，两篇的 prioUpdater 对照着背；
- Prim 朴素版的"线性扫描选最小"与朴素 Dijkstra（[AT 12 §2.2](../AT_12_shortest_path/)）结构完全同构——同一份代码骨架，改一行更新式就能在两个算法间切换；
- MST 之上的进阶（次小生成树、Kruskal 重构树）以本篇和 [AT 10](../AT_10_union_find/) 为地基。

---

*下一篇：[AT 14：关节点与双连通分量](../AT_14_articulation_bcc/)——DFS 时间戳的第一次深水区。*
