---
title: "AT 11: 图的存储与遍历——邻接表、BFS/DFS 与拓扑排序"
date: 2026-12-04
math: true
tags:
  - Notes
  - 算法模板
  - 图论
  - BFS
  - DFS
  - 拓扑排序
categories:
  - 算法模板
description: "图论题的地基模板：vector 邻接表与链式前向星两种存图、BFS（入队时标记）与 DFS（递归/显式栈）、网格 flood fill 的 dx/dy 写法、Kahn 与 DFS 两种拓扑排序。样题：LC 200 岛屿数量、洛谷 P1443 马的遍历、P4017 最大食物链计数，附题目清单"
---

# AT 11：图的存储与遍历——邻接表、BFS/DFS 与拓扑排序

> 本篇是 DSA 刷题模板系列（[AT 00 导读](../AT_00_reading_guide/)）的第 11 篇。核心思想一句话：**图论题先写对存图，把遍历框架背熟，剩下的只是往框架里填"访问时干什么"**。
> 理论后盾：[DSA 10：图](../../DSA/DSA_10/)（表示方法 §7/§12、BFS §13–16、DFS §17–21、拓扑排序 §22–24）。本篇不重复推导，只给"怎么写"。

---

## 1. 这个模板解决什么问题

所有图论题的第一步都是**存图**，第二步是**遍历**（或以遍历为骨架的搜索）。这两个写错，后面的算法再对也白搭：

| 需求 | 选什么 | 一句话理由 |
|------|--------|-----------|
| 一般稀疏图（$m$ 与 $n$ 同阶或略大） | vector 邻接表 | 空间 $O(n+m)$，遍历邻居总代价 $O(n+m)$ |
| 内存卡得死 / 卡 vector 的毒瘤题 | 链式前向星 | 数组模拟链表，无 vector 重建开销 |
| $n \le 3000$ 左右的稠密图 | 邻接矩阵 | 判边存在 $O(1)$，$O(n^2)$ 空间可承受 |
| 网格图（棋盘、地图） | 不存图，直接 `dx/dy` | 网格本身就是隐式图 |

DSA 10 §12.4 有完整的选型讨论；刷题里 **90% 的图用 vector 邻接表**，网格题用 `dx/dy`，其余两类是特例。

---

## 2. 代码模板

### 2.1 存图：vector 邻接表（竞赛默认）

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Edge {
    int to;                      // 边指向的顶点
    long long w;                 // 边权（无权图可去掉）
};
int n, m;                        // 顶点数、边数
const int N = 1e5 + 5;           // 顶点数上限（按题目数据范围改）
vector<Edge> adj[N];             // adj[u]: u 的所有出边

void addEdge(int u, int v, long long w) {
    adj[u].push_back({v, w});
    adj[v].push_back({u, w});    // 无向图加这行；有向图删掉
}
```

要点：**无向图 = 两条有向边**，这是所有无向图算法（BFS/DFS/Dijkstra/Prim……）的标准前提；顶点编号统一用 `0..n-1` 或 `1..n` 都行，但全题只能用一套。

### 2.2 BFS：队列 + 距离数组

```cpp
vector<int> bfs(int s) {                       // s 为源点
    vector<int> dist(n, -1);                   // -1 兼作"未访问"标记
    queue<int> q;
    dist[s] = 0;
    q.push(s);                                 // 标记必须发生在"入队时"，不是出队时！
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (auto [v, w] : adj[u])
            if (dist[v] == -1) {               // 第一次被发现
                dist[v] = dist[u] + 1;         // 无权图：层数即最短路
                q.push(v);
            }
    }
    return dist;                               // dist[v] == -1 即不可达
}
```

DSA 10 §14.2 讲了顶点的三种状态（`UNDISCOVERED / DISCOVERED / VISITED`）；刷题里用一个 `dist` 数组就够——`dist != -1` 等价于"已发现"。**坑 1（最高频）**：标记时机必须是入队时。若等出队才标记，同一个顶点会被塞进队列多次，轻则超时，重则答案错。

<details>
<summary><b>为什么"入队时标记"是对的——从 DSA 10 的状态机看</b></summary>

BFS 的不变量是"队列里至多包含每个顶点的一个副本"。入队时立刻把状态从 `UNDISCOVERED` 置为 `DISCOVERED`，后续再扫到该顶点的其他入边时，检查 `dist[v] == -1` 不成立，就不会重复入队。若改为出队时才标记，顶点在被处理前一直保持 `UNDISCOVERED`，期间它可能被多个邻居各自入队一次——队列里的重复副本会指数级放大搜索量。教材版（DSA 10 §14.1）用独立 `status()` 数组表达同一件事，刷题版把它折叠进 `dist` 的 `-1` 哨兵，语义完全等价。

</details>

### 2.3 DFS：递归版与显式栈版

```cpp
// 递归版（默认写法，注意开全局避免爆栈之外的问题）
bool vis[N];
void dfs(int u) {
    vis[u] = true;
    // 此处做"进入 u 时"的事（如计数、染色）
    for (auto [v, w] : adj[u])
        if (!vis[v]) dfs(v);
    // 此处做"离开 u 时"的事（如拓扑序、Tarjan 回溯）
}
```

递归深度可达 $n$，$n \ge 10^6$ 时可能爆栈，改显式栈：

```cpp
void dfs_iter(int s) {
    stack<int> st;
    st.push(s);
    while (!st.empty()) {
        int u = st.top(); st.pop();
        if (vis[u]) continue;
        vis[u] = true;
        for (auto [v, w] : adj[u])
            if (!vis[v]) st.push(v);
    }
}
```

**坑 2**：显式栈版"离开时机"（对应 DSA 10 §18.2 的 `fTime`）不会自然出现——栈里只压了顶点，没压"回边"事件。需要 `fTime` 的算法（拓扑排序、Tarjan）**直接用递归版**，别改造。

### 2.4 网格 flood fill：dx/dy 数组

```cpp
const int dx[] = {-1, 1, 0, 0};               // 上、下、左、右
const int dy[] = {0, 0, -1, 1};

void fill(int x, int y) {                     // 把 (x,y) 所在连通块全部标记
    vis[x][y] = true;
    for (int i = 0; i < 4; i++) {
        int nx = x + dx[i], ny = y + dy[i];
        if (nx < 0 || nx >= n || ny < 0 || ny >= m) continue;   // 越界先判
        if (vis[nx][ny] || grid[nx][ny] != 目标标记) continue;
        fill(nx, ny);
    }
}
```

把二维坐标当顶点、相邻格当边——网格就是一张隐式图，不需要显式存图。八连通把数组换成 8 个方向即可。

### 2.5 拓扑排序：Kahn（入度版，默认）

```cpp
bool toposort(vector<int>& order) {           // 返回 false 表示有环
    vector<int> indeg(n, 0);
    for (int u = 0; u < n; u++)
        for (auto [v, w] : adj[u]) indeg[v]++;
    queue<int> q;
    for (int u = 0; u < n; u++)
        if (indeg[u] == 0) q.push(u);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        order.push_back(u);
        for (auto [v, w] : adj[u])
            if (--indeg[v] == 0) q.push(v);
    }
    return (int)order.size() == n;            // 没收齐 = 剩下的顶点成环
}
```

这正是 DSA 10 §23 的零入度算法：**每删一个零入度顶点，把它指向的边一并删掉；最后收不齐顶点说明图有环**。"排序 + 判环"一体是它的隐藏福利。

<details>
<summary><b>DFS 版拓扑排序：fTime 逆序</b></summary>

DSA 10 §24 的结论：**按 DFS 完成时刻 `fTime` 从大到小输出，就是拓扑序**。实现上把顶点在"离开时"压栈，最后整体逆序：

```cpp
void dfs(int u) {
    vis[u] = true;
    for (auto [v, w] : adj[u])
        if (!vis[v]) dfs(v);
    order.push_back(u);                       // 离开时记录
}
// 调用后 reverse(order.begin(), order.end())
```

为什么对：对 DAG 中任意边 $u \to v$，无论 DFS 先访问谁，$v$ 的 `fTime` 一定早于 $u$（要么 $v$ 是 $u$ 的后代先完成，要么 $v$ 整棵子树先完成）。逆序后 $u$ 排在 $v$ 前。刷题里 Kahn 是默认（好写、顺带判环），DFS 版的价值在于它是 Tarjan 系算法（[AT 14](../AT_14_articulation_bcc/)）的思维雏形。

</details>

---

## 3. 常见变形与坑

| 变形 | 一句话做法 |
|------|-----------|
| 连通分量计数 | 外层循环每个未访问顶点，各起一次 DFS/BFS，计数器即分量数 |
| 无权最短路 | BFS 的 `dist` 直接就是答案，别用 Dijkstra |
| 多源 BFS | 所有源点**一起**入队（初始 `dist=0`），其余不变 |
| 网格走马/特殊步 | 只改 `dx/dy` 数组，框架不动 |
| 拓扑序上 DP | Kahn 的 `order` 天然是 DP 的计算序（见 §4 样题三） |

坑清单：

1. **入队/访问标记时机**：见 §2.2，图论 WA 第一名；
2. **无向图忘加反向边**：调试半天发现图"只能单向走"；
3. **显式栈做不了 Tarjan**：需要 `fTime` 的算法一律递归；
4. **网格越界**：`dx/dy` 移动后必须先判边界再访问，顺序反了直接 RE；
5. **重边与自环**：邻接表天然容忍重边；Kahn 遇自环会使该点入度永不归零，恰好被"判环"正确处理；
6. **多测不清空**：数组开全局时，每组数据前 `memset` / 重新 `assign`，`adj` 要逐项 `clear()`。

---

## 4. 样题精讲

### 4.1 LC 200. 岛屿数量（网格 flood fill 计数）

> 题目：[https://leetcode.cn/problems/number-of-islands/](https://leetcode.cn/problems/number-of-islands/)（中等）。`'1'` 是陆地、`'0'` 是水，数出岛屿（四连通陆地块）的数量。

**思路**：陆地格是顶点、四邻接是边，岛屿数 = 连通分量数。外层扫描每个格子，遇到没访问过的 `'1'` 就答案 +1，并把整块连通区域 flood fill 标记掉。

```cpp
class Solution {
    int n, m;
    void fill(vector<vector<char>>& g, int x, int y) {
        if (x < 0 || x >= n || y < 0 || y >= m) return;
        if (g[x][y] != '1') return;
        g[x][y] = '0';                        // 直接改原网格当访问标记，省 O(nm) 空间
        fill(g, x - 1, y); fill(g, x + 1, y);
        fill(g, x, y - 1); fill(g, x, y + 1);
    }
public:
    int numIslands(vector<vector<char>>& grid) {
        n = grid.size(); m = grid[0].size();
        int ans = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                if (grid[i][j] == '1') { ans++; fill(grid, i, j); }
        return ans;
    }
};
```

复杂度 $O(nm)$：每格至多被标记一次。这是"隐式图 + 分量计数"的原型题，完整题解（含并查集做法）见 [LeetCode 200 题解](../../LeetCode/LeetCode_200/)。

### 4.2 洛谷 P1443. 马的遍历（BFS 求步数）

> 题目：[https://www.luogu.com.cn/problem/P1443](https://www.luogu.com.cn/problem/P1443)。$n \times m$ 棋盘上有一匹马，从给定起点出发，输出它到每个格子最少走几步，到不了输出 $-1$。

**思路**：马的 8 个跳跃方向就是 `dx/dy`，每步代价相同 → **无权最短路 → BFS**，一步一入队，层数即步数。这是"只改 `dx/dy`、框架不动"的标准示范：

```cpp
#include <bits/stdc++.h>
using namespace std;
const int dx[] = {-2,-2,-1,-1, 1, 1, 2, 2};
const int dy[] = {-1, 1,-2, 2,-2, 2,-1, 1};

int n, m, dist[405][405];

int main() {
    int sx, sy;
    scanf("%d%d%d%d", &n, &m, &sx, &sy);
    memset(dist, -1, sizeof dist);
    queue<pair<int,int>> q;
    dist[sx][sy] = 0;
    q.push({sx, sy});                         // 入队时标记
    while (!q.empty()) {
        auto [x, y] = q.front(); q.pop();
        for (int i = 0; i < 8; i++) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 1 || nx > n || ny < 1 || ny > m) continue;
            if (dist[nx][ny] != -1) continue;
            dist[nx][ny] = dist[x][y] + 1;
            q.push({nx, ny});
        }
    }
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) printf("%-5d", dist[i][j]);  // 注意题面的场宽要求
        printf("\n");
    }
    return 0;
}
```

复杂度 $O(nm)$。

### 4.3 洛谷 P4017. 最大食物链计数（Kahn + 拓扑序 DP）

> 题目：[https://www.luogu.com.cn/problem/P4017](https://www.luogu.com.cn/problem/P4017)。给定食物网（边 $u \to v$ 表示 $u$ 被 $v$ 吃），统计从生产者（入度 0，不吃任何东西）到顶级消费者（出度 0，不被任何东西吃）的路径条数，对 $80112002$ 取模。

**思路**：食物网是 DAG，"路径计数"在拓扑序上做 DP——Kahn 每弹出一个顶点，它的所有前驱都已算完：

```cpp
#include <bits/stdc++.h>
using namespace std;
const int MOD = 80112002;

int main() {
    int n, m;
    scanf("%d%d", &n, &m);
    vector<vector<int>> adj(n + 1);
    vector<int> indeg(n + 1, 0), outdeg(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int a, b;
        scanf("%d%d", &a, &b);                // a 被 b 吃 → 边 a→b
        adj[a].push_back(b);
        indeg[b]++; outdeg[a]++;
    }
    vector<long long> ways(n + 1, 0);
    queue<int> q;
    for (int u = 1; u <= n; u++)
        if (indeg[u] == 0) { ways[u] = 1; q.push(u); }
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) {
            ways[v] = (ways[v] + ways[u]) % MOD;
            if (--indeg[v] == 0) q.push(v);
        }
    }
    long long ans = 0;
    for (int u = 1; u <= n; u++)
        if (outdeg[u] == 0) ans = (ans + ways[u]) % MOD;
    printf("%lld\n", ans);
    return 0;
}
```

复杂度 $O(n + m)$。**为什么 Kahn 的弹出序就是合法 DP 序**：顶点 $v$ 入队的前提是所有前驱都弹过队、都把贡献累进 `ways[v]` 了。这个"拓扑序 DP"套路是 DAG 上一切计数/最值问题的通用骨架。

---

## 5. 题目清单

| 题号 | 题名 | 平台/难度 | 考的变形 | 链接 |
|------|------|----------|---------|------|
| LC 200 | 岛屿数量 | LeetCode 中等 | 网格分量计数 | [题解](../../LeetCode/LeetCode_200/) |
| LC 994 | 腐烂的橘子 | LeetCode 中等 | 多源 BFS | [题解](../../LeetCode/LeetCode_994/) |
| LC 207 | 课程表 | LeetCode 中等 | 拓扑排序判环 | [题解](../../LeetCode/LeetCode_207/) |
| LC 210 | 课程表 II | LeetCode 中等 | 输出拓扑序 | [题目](https://leetcode.cn/problems/course-schedule-ii/) |
| LC 133 | 克隆图 | LeetCode 中等 | 遍历 + 哈希建新图 | [题目](https://leetcode.cn/problems/clone-graph/) |
| P1443 | 马的遍历 | 洛谷 普及- | BFS 步数 + 场宽输出 | [题目](https://www.luogu.com.cn/problem/P1443) |
| P1596 | [USACO10OCT] Lake Counting S | 洛谷 普及- | flood fill 分量计数 | [题目](https://www.luogu.com.cn/problem/P1596) |
| P1451 | 求细胞数量 | 洛谷 普及- | 连通块计数（字符图） | [题目](https://www.luogu.com.cn/problem/P1451) |
| P4017 | 最大食物链计数 | 洛谷 普及+ | Kahn + 拓扑序 DP | [题目](https://www.luogu.com.cn/problem/P4017) |

---

## 6. 与其他篇的联系

- 本篇的存图与遍历是 [AT 12（最短路）](../AT_12_shortest_path/)、[AT 13（最小生成树）](../AT_13_mst/)、[AT 14（关节点与 BCC）](../AT_14_articulation_bcc/) 的公共地基，后三篇直接复用 `addEdge`；
- BFS 的队列、DFS 的栈分别回到 [AT 06（队列）](../AT_06_queue_monotonic_queue/)与 [AT 04（栈）](../AT_04_stack/)；堆优化 Dijkstra 的优先队列见 [AT 09（堆）](../AT_09_heap/)；
- 理论：图的表示与 BFS/DFS 状态机在 [DSA 10](../../DSA/DSA_10/)；拓扑排序与"PFS 统一框架"在 [DSA 11](../../DSA/DSA_11/)，其中 PFS 视角（BFS/DFS 只是换优先级函数）会在 [AT 13](../AT_13_mst/) 里兑现成一行之差的 Prim 与 Dijkstra。

---

*下一篇：[AT 12：最短路](../AT_12_shortest_path/)——Dijkstra、SPFA、Floyd 的选型与模板。*
