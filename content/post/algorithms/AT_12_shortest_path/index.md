---
title: "AT 12: 最短路——Dijkstra、SPFA 与 Floyd 的选型与模板"
date: 2026-12-05
math: true
tags:
  - Notes
  - 算法模板
  - 图论
  - 最短路
  - Dijkstra
  - Floyd
categories:
  - 算法模板
description: "最短路三件套的选型表与模板：堆优化 Dijkstra O(m log m)（含懒惰删除）、朴素 Dijkstra O(n²)、SPFA 与负环判定、Floyd 的 DP 语义（中转点 k 在最外层）。样题：洛谷 P4779 模板、LC 743 网络延迟时间、P1119 灾后重建（分阶段 Floyd），附题目清单"
---

# AT 12：最短路——Dijkstra、SPFA 与 Floyd 的选型与模板

> 本篇是 DSA 刷题模板系列（[AT 00 导读](../AT_00_reading_guide/)）的第 12 篇。核心思想一句话：**先背选型表，再背模板——非负单源用堆优化 Dijkstra，有负权或要判负环用 SPFA，多源或 $n \le 500$ 用 Floyd**。
> 理论后盾：[DSA 11：图的应用](../../DSA/DSA_11/)（问题分类 §9、Dijkstra 贪心与松弛 §10–14、Floyd 的 DP 定义 §19–22）。存图与遍历框架在 [AT 11](../AT_11_graph_bfs_dfs/)。

---

## 1. 这个模板解决什么问题

最短路是图论题里考得最多的一族。写题时第一件事不是敲模板，是**选型**：

| 场景 | 选什么 | 复杂度 | 一句话理由 |
|------|--------|--------|-----------|
| 单源、边权非负、稀疏图 | 堆优化 Dijkstra | $O(m \log m)$ | 洛谷默认出题复杂度 |
| 单源、边权非负、稠密图（$m \approx n^2$） | 朴素 Dijkstra | $O(n^2)$ | 堆版在稠密图上反而退化到 $O(n^2 \log n)$ |
| 有负权边 | SPFA（Bellman–Ford 队列优化） | 平均 $O(km)$，最坏 $O(nm)$ | Dijkstra 的贪心在负权下不成立 |
| 判负环 | SPFA + 计数法 | 同上 | 松弛次数 $\ge n$ 即有负环 |
| 多源最短路 / $n \le 500$ | Floyd | $O(n^3)$ | 三重循环写完收工，还能顺带传闭包 |
| 无权图 | BFS（见 [AT 11](../AT_11_graph_bfs_dfs/)） | $O(n+m)$ | 权全是 1，别杀鸡用牛刀 |

一个高频误区：**负权 ≠ 必须弃用 Dijkstra**。DSA 11 §10.2 讲清了根源——Dijkstra 依赖"已确定顶点的距离不会再被更新"，负边会破坏这个单调性。

---

## 2. 代码模板

### 2.1 堆优化 Dijkstra（默认模板，懒惰删除）

```cpp
#include <bits/stdc++.h>
using namespace std;
const long long INF = 1e18;

int n, m;
vector<vector<pair<int,int>>> adj;             // adj[u] = {（邻居, 边权）}

vector<long long> dijkstra(int s) {
    vector<long long> dist(n + 1, INF);
    priority_queue<pair<long long,int>,
                   vector<pair<long long,int>>,
                   greater<>> pq;              // 小根堆，按 dist 升序
    dist[s] = 0;
    pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;             // 懒惰删除：过期的堆条目直接跳过
        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {       // 松弛（relaxation）
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});         // 不删旧条目，插入新的
            }
        }
    }
    return dist;                               // dist[v] == INF 即不可达
}
```

两个设计决定值得点破：

1. **懒惰删除（lazy deletion）**：C++ 的 `priority_queue` 不支持 decrease-key，做法是"每次松弛成功就压一个新条目，弹出时发现 `d > dist[u]` 就丢弃"。同一顶点在堆里可有多个副本，但每个副本至多被弹出处理一次，总代价仍是 $O(m \log m)$。
2. **松弛（relaxation）**：`dist[u] + w < dist[v]` 时更新——DSA 11 §12.2 的标准术语，所有最短路算法的核心动作都是它，区别只在"以什么顺序对哪些边做"。

### 2.2 朴素 Dijkstra（稠密图）

```cpp
vector<long long> dijkstra_n2(int s) {
    vector<long long> dist(n + 1, INF);
    vector<bool> done(n + 1, false);
    dist[s] = 0;
    for (int i = 1; i <= n; i++) {             // 选 n 次
        int u = 0;                             // 在未确定顶点中线性找最小
        for (int v = 1; v <= n; v++)
            if (!done[v] && dist[v] < dist[u]) u = v;   // dist[0]=INF 作哨兵
        done[u] = true;
        for (auto [v, w] : adj[u])             // 用 u 松弛其所有出边
            dist[v] = min(dist[v], dist[u] + w);
    }
    return dist;
}
```

这其实就是 DSA 11 §8.2 PFS 框架的朴素实现："更新邻居优先级 → 线性扫描选最小"。**边数接近 $n^2$ 时它比堆版快**（堆版要做 $O(m) = O(n^2)$ 次插入）。

### 2.3 SPFA 与负环判定

```cpp
bool spfa(int s, vector<long long>& dist) {    // 返回 false 表示存在负环
    dist.assign(n + 1, INF);
    vector<int> cnt(n + 1, 0);                 // 每条边成功松弛的累计次数（对顶点计数）
    vector<bool> inq(n + 1, false);
    queue<int> q;
    dist[s] = 0; inq[s] = true; q.push(s);
    while (!q.empty()) {
        int u = q.front(); q.pop(); inq[u] = false;
        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                if (!inq[v]) { q.push(v); inq[v] = true; }
                if (++cnt[v] >= n) return false;   // 一条边被松弛 n 次 ⇒ 负环
            }
        }
    }
    return true;
}
```

**判环原理**：无负环时最短路至多 $n-1$ 条边（更长的会被简单路径替代）；某个顶点被松弛成功 $n$ 次，说明路径长度已突破这个上界，唯一解释是走了负环。**坑**：SPFA 最坏 $O(nm)$，没负权就别用——不少题目专门卡 SPFA。

### 2.4 Floyd：三重循环与它的 DP 语义

```cpp
// g 为邻接矩阵：g[u][v] = 边权，无边 = INF，g[u][u] = 0
for (int k = 1; k <= n; k++)                   // 中转点 k 必须在最外层！
    for (int u = 1; u <= n; u++)
        for (int v = 1; v <= n; v++)
            if (g[u][k] + g[k][v] < g[u][v])
                g[u][v] = min(g[u][v], g[u][k] + g[k][v]);
```

这不是"枚举三个顶点"的暴力——DSA 11 §20 给出了正确读法：$g_k[u][v]$ 表示**只允许用前 $k$ 个顶点做中转**时 $u \to v$ 的最短路，则

$$
g_k(u,v) = \min\{\, g_{k-1}(u,v),\; g_{k-1}(u,k) + g_{k-1}(k,v) \,\}
$$

路径要么不经过第 $k$ 个中转点，要么经过它一次（拆成两段）。**$k$ 在最外层**是滚动数组对 $k$ 维的枚举序——写到内层就把"前 $k$ 个中转点"的阶段性破坏了，答案会错（不是精度问题，是语义错误）。

---

## 3. 常见变形与坑

| 变形 | 一句话做法 |
|------|-----------|
| 输出具体路径 | 松弛时记 `pre[v] = u`，终点沿 `pre` 回溯再反转 |
| 最短路计数 | 松弛时 `<` 则替换、`==` 则累加（P1144） |
| 分层图 / 状态扩充 | 把"顶点 × 状态"当新顶点建图（LC 787 的 $K$ 站中转） |
| 建反向图求"到终点"的最短路 | 所有边反向后再从终点跑一遍单源 |
| 传送/换乘类建图 | 拆点或加虚点，把非线性代价转成普通边 |

坑清单：

1. **`INF` 用 `INT_MAX` 再加边权会溢出**——用 `long long` 且 `INF` 取 $10^{18}$，或先判 `dist[u] == INF` 再松弛；
2. **懒惰删除那行 `if (d > dist[u]) continue;` 忘写**——复杂度退化、答案还可能被旧条目污染（配合下面第 3 条）；
3. **无向图的最短路要加双向边**，且两条边都要参与松弛；
4. **Floyd 的 `k` 写到内层**——语义错误，小样例可能碰巧对；
5. **多源/起点不固定**：别跑 $n$ 次 Dijkstra，先想 Floyd 或超级源点（虚点连所有源点、边权 0）；
6. **判完负环别忘了"负环存在 ⇒ 最短路可能无意义"**（$-\infty$），题目通常只问存在性。

---

## 4. 样题精讲

### 4.1 洛谷 P4779.【模板】单源最短路径（标准版）

> 题目：[https://www.luogu.com.cn/problem/P4779](https://www.luogu.com.cn/problem/P4779)。$n$ 个顶点 $m$ 条有向边的非负权图，给定源点 $s$，输出 $s$ 到每个顶点的最短距离。

模板题，把 §2.1 抄全即可：

```cpp
#include <bits/stdc++.h>
using namespace std;
const long long INF = 1e18;

int n, m, s;
vector<vector<pair<int,int>>> adj;

int main() {
    scanf("%d%d%d", &n, &m, &s);
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v, w;
        scanf("%d%d%d", &u, &v, &w);
        adj[u].push_back({v, w});              // 有向图：只加一条
    }
    vector<long long> dist(n + 1, INF);
    priority_queue<pair<long long,int>,
                   vector<pair<long long,int>>,
                   greater<>> pq;
    dist[s] = 0;
    pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto [v, w] : adj[u])
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
    }
    for (int i = 1; i <= n; i++)
        printf("%lld%c", dist[i] == INF ? (long long)(1e9) : dist[i], " \n"[i == n]);
    return 0;
}
```

复杂度 $O((n + m)\log m)$。（提示：洛谷模板题常约定"不可达输出某个固定值"，提交前对照题面确认，别照抄本文的输出分支。）

### 4.2 LC 743. 网络延迟时间（堆优化 Dijkstra 套用）

> 题目：[https://leetcode.cn/problems/network-delay-time/](https://leetcode.cn/problems/network-delay-time/)（中等）。$n$ 个节点的有向图，边 `times[i] = (u, v, w)` 表示信号从 $u$ 传到 $v$ 耗时 $w$。从节点 $k$ 发出信号，问多久**所有**节点都收到；收不齐返回 $-1$。

**思路**：信号沿最短路传播，答案 = $\max_v \mathrm{dist}(k, v)$；有节点不可达返回 $-1$。模板一行不改地套：

```cpp
class Solution {
public:
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        vector<vector<pair<int,int>>> adj(n + 1);
        for (auto& t : times)
            adj[t[0]].push_back({t[1], t[2]});
        const int INF = 1e9;
        vector<int> dist(n + 1, INF);
        priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
        dist[k] = 0;
        pq.push({0, k});
        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > dist[u]) continue;
            for (auto [v, w] : adj[u])
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    pq.push({dist[v], v});
                }
        }
        int ans = 0;
        for (int v = 1; v <= n; v++) {
            if (dist[v] == INF) return -1;
            ans = max(ans, dist[v]);
        }
        return ans;
    }
};
```

复杂度 $O(m \log m)$。**LeetCode 上最短路几乎全是这道的原型**：把"最大到达时间"换成别的聚合（判断可达、路径存在性），模板不动。

### 4.3 洛谷 P1119. 灾后重建（分阶段 Floyd）

> 题目：[https://www.luogu.com.cn/problem/P1119](https://www.luogu.com.cn/problem/P1119)。$n$ 个村庄各有修复时间 $t_i$；多组询问 $(x, y, t)$：在第 $t$ 天，只经过**已修复**村庄（$t_i \le t$）时 $x$ 到 $y$ 的最短路。

**思路**：这题是 Floyd 的 DP 语义（"前 $k$ 个中转点"）的直接兑现——**把村庄按修复时间排序，一个村庄修复 = 允许它当中转点 = Floyd 外层循环推进一格**。询问按 $t$ 排序离线处理：

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d%d", &n, &m);
    vector<int> t(n);
    for (auto& x : t) scanf("%d", &x);
    vector<vector<long long>> g(n, vector<long long>(n, 1e18));
    for (int i = 0; i < n; i++) g[i][i] = 0;
    for (int i = 0; i < m; i++) {
        int u, v, w;
        scanf("%d%d%d", &u, &v, &w);
        g[u][v] = g[v][u] = min(g[u][v], (long long)w);
    }
    int q;
    scanf("%d", &q);
    vector<array<int,3>> qs(q);                // (t, x, y)
    for (auto& p : qs) scanf("%d%d%d", &p[1], &p[2], &p[0]);
    vector<int> idx(q);
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(),
         [&](int a, int b){ return qs[a][0] < qs[b][0]; });

    vector<int> order(n);                      // 村庄按修复时间排序
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(),
         [&](int a, int b){ return t[a] < t[b]; });

    int ptr = 0;                               // 已开放的中转点数
    vector<long long> ans(q, -1);
    for (int qi : idx) {
        auto [T, x, y] = qs[qi];
        while (ptr < n && t[order[ptr]] <= T) {   // 新修复的村庄逐一开放为中转点
            int k = order[ptr++];
            for (int u = 0; u < n; u++)
                for (int v = 0; v < n; v++)
                    g[u][v] = min(g[u][v], g[u][k] + g[k][v]);
        }
        if (t[x] > T || t[y] > T) ans[qi] = -1;              // 端点未修复
        else ans[qi] = (g[x][y] >= 1e18 ? -1 : g[x][y]);
    }
    for (int i = 0; i < q; i++) printf("%lld\n", ans[i]);
    return 0;
}
```

复杂度 $O(n^3 + q \log q)$。Floyd 的外层枚举被"时间"赋予了物理含义——**DP 的阶段性本来就是算法的一部分**，这是本篇最值得带走的一题。

---

## 5. 题目清单

| 题号 | 题名 | 平台/难度 | 考的变形 | 链接 |
|------|------|----------|---------|------|
| P4779 | 【模板】单源最短路径（标准版） | 洛谷 普及+ | 堆优化 Dijkstra | [题目](https://www.luogu.com.cn/problem/P4779) |
| P3371 | 【模板】单源最短路径（弱化版） | 洛谷 普及- | 朴素 Dijkstra / SPFA 皆可过 | [题目](https://www.luogu.com.cn/problem/P3371) |
| P3385 | 【模板】负环 | 洛谷 提高 | SPFA 判负环 | [题目](https://www.luogu.com.cn/problem/P3385) |
| P1119 | 灾后重建 | 洛谷 提高- | 分阶段 Floyd | [题目](https://www.luogu.com.cn/problem/P1119) |
| P1144 | 最短路计数 | 洛谷 提高- | Dijkstra + 计数 | [题目](https://www.luogu.com.cn/problem/P1144) |
| LC 743 | 网络延迟时间 | LeetCode 中等 | Dijkstra 原型题 | [题目](https://leetcode.cn/problems/network-delay-time/) |
| LC 787 | K 站中转内最便宜的航班 | LeetCode 中等 | 分层图 / Bellman–Ford | [题目](https://leetcode.cn/problems/cheapest-flights-within-k-stops/) |

---

## 6. 与其他篇的联系

- 存图与遍历来自 [AT 11](../AT_11_graph_bfs_dfs/)；堆优化 Dijkstra 的 `priority_queue` 用法（比较器、懒惰删除的思想）详见 [AT 09（堆）](../AT_09_heap/)；
- Dijkstra、Prim、BFS 在 [DSA 11 §8](../../DSA/DSA_11/) 是同一个 PFS 框架换不同优先级函数——这个视角在 [AT 13（最小生成树）](../AT_13_mst/)会变成"两行代码之差"的对照表；
- 判负环的 SPFA、差分约束等进阶题从这里出发；网格图上的"多源 BFS"（LC 994）在 [AT 11](../AT_11_graph_bfs_dfs/)，是 $w=1$ 特例下的"多源最短路"。

---

*下一篇：[AT 13：最小生成树](../AT_13_mst/)——Kruskal、Prim 与 PFS 统一视角。*
