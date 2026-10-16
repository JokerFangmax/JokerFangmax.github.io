---
title: "AT 14: 关节点与双连通分量——Tarjan 割点/桥/点双"
date: 2026-12-07
math: true
tags:
  - Notes
  - 算法模板
  - 图论
  - Tarjan
categories:
  - 算法模板
description: "一遍 DFS 同时求割点、桥与点双连通分量：dfn/low 的语义辨析（low 只经回边取对方 dfn、经树边从孩子继承）、割点的根/非根双重判定、重边用边编号判父、点双弹栈时关节点不弹出。附 P3388/P1656/P8435 完整代码与 LC 1192"
---

# AT 14：关节点与双连通分量——Tarjan 割点/桥/点双

> 本篇是 AT 系列第 14 篇。核心思想一句话：**DFS 往下探的同时问每棵子树——你找得到一条绕回更高祖先的"逃生通道"吗？找不到，当前节点就是咽喉。** 理论推导（判定准则的完整论证）在 [DSA 11 §2–7](../../DSA/DSA_11/)，本篇只留"怎么写对"的最小说明。

---

## 1. 这个模板解决什么问题

无向图的**脆弱性分析**——删哪个点/哪条边会让图断开：

| 问题 | 定义 | 判定核心 |
|------|------|---------|
| 割点（关节点） | 删除后连通分量数增加的顶点 | 存在孩子 $u$：$\text{low}[u] \ge \text{dfn}[v]$（根节点特殊） |
| 桥（割边） | 删除后连通分量数增加的边 | 树边 $(v,u)$：$\text{low}[u] > \text{dfn}[v]$ |
| 点双连通分量（点双/BCC） | 极大的无割点连通子图 | 关节点切开的"块"，DFS + 栈当场分割 |

暴力是对每个点删掉再判连通，$O(n(n+m))$；Tarjan 的做法是一遍 DFS 顺手积累信息，$O(n+m)$。

## 2. 代码模板

### 2.1 dfn / low 的语义（最大的坑在这里）

设 DFS 中每个点 $v$ 有：

- $\text{dfn}[v]$：$v$ 的**时间戳**（被发现的次序，从 1 编号）；
- $\text{low}[v]$：$v$ 的子树**绕过树边**能触及的**最小时间戳**。

`low` 的更新只有两条规则，方向绝不能混：

| 场景 | 更新 | 用什么更新 |
|------|------|-----------|
| 走树边 $(v,u)$，孩子回溯后 | $\text{low}[v] \gets \min(\text{low}[v],\ \text{low}[u])$ | 孩子的 **low** |
| 遇到回边 $(v,u)$（$u$ 已访问且非父边） | $\text{low}[v] \gets \min(\text{low}[v],\ \text{dfn}[u])$ | 对方的 **dfn** |

**为什么回边用 dfn 而不用 low**：回边 $(v,u)$ 说明 $v$ 能直接摸到 $u$；但 $u$ 的 low 描述的是"$u$ 的子树"能到哪，与 $v$ 无关——把别人的子树能力算到自己头上，会把"绕得回祖先"误判成真，漏报割点/桥。这是本模板第一大坑。

另一个易混点：不同教材对 low 初值不同（有的教材让 $\text{low}[v]$ 初值为 $+\infty$、点双判定比较的是"能否触及 $v$ 的真祖先"）。本篇与绝大多数竞赛模板一致：**初值 $\text{low}[v]=\text{dfn}[v]$，"$\ge$" 判定即对应"触及不了真祖先"**。抄模板时整套抄，别混抄两种定义。

### 2.2 主模板：一遍 DFS 求割点 + 桥

```cpp
int timer_ = 0;
vector<int> dfn, low;
vector<bool> isCut;
vector<pair<int,int>> bridges;

void tarjan(int u, int root, int inEdge) {   // root：本次 DFS 的起点；inEdge：进入 u 的边编号
    dfn[u] = low[u] = ++timer_;
    int children = 0;                     // u 作为 DFS 树根时的孩子数
    for (auto [v, id] : g[u]) {
        if (!dfn[v]) {                    // 树边
            children++;
            tarjan(v, root, id);
            low[u] = min(low[u], low[v]);           // 规则一：继承孩子的 low
            if (u != root && low[v] >= dfn[u]) isCut[u] = true;  // 非根割点
            if (low[v] >  dfn[u]) bridges.push_back({min(u,v), max(u,v)});  // 桥
        } else if (id != inEdge) {        // 回边（父边被边编号排除）
            low[u] = min(low[u], dfn[v]);            // 规则二：只取对方 dfn
        }
    }
    if (u == root && children >= 2) isCut[u] = true; // 根：两棵以上子树才是割点
}
```

（图不连通时外层对每个未访问点 `tarjan(v, v, -1)`。）

**关键不变量**：

1. **父边判断用"边编号"而不是"父节点"**——两点间有重边时，第二条边是合法回边（它本身就是一条绕回父亲的逃生通道），若按"跳过父节点"写，重边会被误判成桥；
2. 根节点没有祖先，"回不到真祖先"的判定失效，必须换成**孩子数 $\ge 2$**；
3. 割点判定的比较对象是孩子 `low[v]` 与自己的 `dfn[u]`，两端都是**下标意义上的时间戳**。

### 2.3 点双：加一个顶点栈

在 2.2 基础上加栈即可（完整可提交版见 §4.3）：

```cpp
// tarjan(u) 入口处：stk.push_back(u);
// 树边回溯、发现 low[v] >= dfn[u] 时：
bcc.push_back({});
int w;
do { w = stk.back(); stk.pop_back(); bcc.back().push_back(w); } while (w != v);
bcc.back().push_back(u);   // 关节点 u 属于这个 BCC，但留在栈上（它还属于别的 BCC）
```

弹到**孩子 $v$** 为止、关节点 $u$ **补进去但不弹出**——因为一个关节点同时属于多个 BCC（[DSA 11 §6.2](../../DSA/DSA_11/) 的栈语义：保存"尚未归入完整 BCC 的顶点"）。

## 3. 常见变形与坑

| 变形 | 一句话做法 |
|------|-----------|
| 只求割点 | 邻接表可以不存边编号（无重边干扰父边判断时） |
| 只求桥 | `low[v] > dfn[u]`，重边必须按边编号判父 |
| 点双 BCC | 顶点栈 + 弹到孩子为止，关节点不弹出 |
| 边双 BCC | 删掉所有桥后每个连通分量即一个边双：求桥 + [并查集](../AT_10_union_find/)缩点，或栈存边编号弹到桥边为止 |
| 缩点建图 | 点双缩点成 "BCC—关节点" 的树（圆方树：圆点=原点，方点=BCC） |
| 有向图版本 | 强连通分量（SCC）另一套模板，low 语义相近但判定不同，别混用 |

坑清单（按翻车频率排序）：

1. **回边用 `low[v]` 更新**——必须用 `dfn[v]`（§2.1）；
2. **重边当桥**——父边判断必须用边编号（§2.2）；
3. **忘根节点特判**——根只看孩子数是否 $\ge 2$；
4. **图不连通**——外层循环对所有未访问点各起一次 DFS；
5. **点双把关节点弹出**——关节点属于多个 BCC，弹出后后续 BCC 会缺它；
6. **孤立点约定**——某些题（如 P8435）专门考"孤立点是否自成一点双"，以题面样例为准核对；
7. **自环**——自环对割点/桥无贡献，建边时跳过 `a == b` 即可。

## 4. 样题精讲

### 4.1 [洛谷 P3388 【模板】割点（割顶）](https://www.luogu.com.cn/problem/P3388)

$n$ 点 $m$ 边无向图求割点：第一行输出个数，第二行按编号升序输出。主模板直接套——邻接表无需边编号（割点判定不受重边影响，重边不改变"子树绕不绕得回祖先"），输出排序即可：

```cpp
#include <bits/stdc++.h>
using namespace std;
int n, m, timer_ = 0;
vector<vector<int>> g;
vector<int> dfn, low;
vector<bool> isCut;

void tarjan(int u, int root) {
    dfn[u] = low[u] = ++timer_;
    int children = 0;
    for (int v : g[u]) {
        if (!dfn[v]) {
            children++;
            tarjan(v, root);
            low[u] = min(low[u], low[v]);
            if (u != root && low[v] >= dfn[u]) isCut[u] = true;
        } else {
            low[u] = min(low[u], dfn[v]);
        }
    }
    if (u == root && children >= 2) isCut[u] = true;
}

int main() {
    scanf("%d%d", &n, &m);
    g.assign(n + 1, {});
    dfn.assign(n + 1, 0); low.assign(n + 1, 0);
    isCut.assign(n + 1, false);
    for (int i = 0; i < m; i++) {
        int a, b; scanf("%d%d", &a, &b);
        g[a].push_back(b); g[b].push_back(a);
    }
    for (int v = 1; v <= n; v++)
        if (!dfn[v]) tarjan(v, v);
    vector<int> ans;
    for (int v = 1; v <= n; v++) if (isCut[v]) ans.push_back(v);
    printf("%d\n", (int)ans.size());
    for (int i = 0; i < (int)ans.size(); i++)
        printf("%d%c", ans[i], i + 1 == (int)ans.size() ? '\n' : ' ');
    return 0;
}
```

复杂度 $O(n+m)$。自测（两个三角形用一条边相连的"蝴蝶图"）：割点为桥两端的 2 和 5——输出与手推一致。

### 4.2 [洛谷 P1656 炸铁路](https://www.luogu.com.cn/problem/P1656)

$n$ 个城市（$1 \le n \le 150$，$1 \le m \le 5000$，**保证无重边**）求所有 key road（桥），按 $a$ 从小到大、$a$ 相同按 $b$ 从小到大输出（$a<b$）。模板 + 排序输出：

```cpp
#include <bits/stdc++.h>
using namespace std;
int n, m, timer_ = 0;
vector<vector<pair<int,int>>> g;
vector<int> dfn, low;
vector<pair<int,int>> bridges;

void tarjan(int u, int inEdge) {
    dfn[u] = low[u] = ++timer_;
    for (auto [v, id] : g[u]) {
        if (!dfn[v]) {
            tarjan(v, id);
            low[u] = min(low[u], low[v]);
            if (low[v] > dfn[u]) bridges.push_back({min(u,v), max(u,v)});
        } else if (id != inEdge) {
            low[u] = min(low[u], dfn[v]);
        }
    }
}

int main() {
    scanf("%d%d", &n, &m);
    g.assign(n + 1, {});
    dfn.assign(n + 1, 0); low.assign(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int a, b; scanf("%d%d", &a, &b);
        g[a].push_back({b, i}); g[b].push_back({a, i});
    }
    for (int v = 1; v <= n; v++) if (!dfn[v]) tarjan(v, -1);
    sort(bridges.begin(), bridges.end());
    for (auto& [a, b] : bridges) printf("%d %d\n", a, b);
    return 0;
}
```

本题无重边，但边编号写法照样安全——**"按边编号判父"是普适写法，重边场景零成本切换**。

### 4.3 [洛谷 P8435 【模板】点双连通分量](https://www.luogu.com.cn/problem/P8435)

$n \le 5\times10^5$、$m \le 2\times10^6$，输出点双个数及每个点双（**任意顺序可过**，题面明说）。两处细节：自环跳过（对点双无贡献）；孤立点本题按"自成一点双"的约定处理（题面温馨提示样例五专门考这个，提交前拿样例核对）：

```cpp
#include <bits/stdc++.h>
using namespace std;
int n, m, timer_ = 0;
vector<vector<pair<int,int>>> g;
vector<int> dfn, low, stk;
vector<vector<int>> bcc;

void tarjan(int u, int inEdge) {
    dfn[u] = low[u] = ++timer_;
    stk.push_back(u);
    for (auto [v, id] : g[u]) {
        if (!dfn[v]) {
            tarjan(v, id);
            low[u] = min(low[u], low[v]);
            if (low[v] >= dfn[u]) {           // u 是该 BCC 中最后被弹出的点
                bcc.push_back({});
                int w;
                do { w = stk.back(); stk.pop_back(); bcc.back().push_back(w); } while (w != v);
                bcc.back().push_back(u);      // 关节点 u 属于该 BCC，但不弹出
            }
        } else if (id != inEdge) {
            low[u] = min(low[u], dfn[v]);
        }
    }
}

int main() {
    scanf("%d%d", &n, &m);
    g.assign(n + 1, {});
    dfn.assign(n + 1, 0); low.assign(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int a, b; scanf("%d%d", &a, &b);
        if (a == b) continue;                 // 自环不参与点双
        g[a].push_back({b, i}); g[b].push_back({a, i});
    }
    for (int v = 1; v <= n; v++)
        if (!dfn[v]) tarjan(v, -1);
    // 孤立点：按题面约定各成一个点双（本题约定如此，以样例为准）
    for (int v = 1; v <= n; v++)
        if (g[v].empty()) bcc.push_back({v});
    printf("%d\n", (int)bcc.size());
    for (auto& c : bcc) {
        printf("%d ", (int)c.size());
        for (int i = 0; i < (int)c.size(); i++)
            printf("%d%c", c[i], i + 1 == (int)c.size() ? '\n' : ' ');
    }
    return 0;
}
```

蝴蝶图自测（边 1-2、三角形 {2,3,4}、桥 2-5、三角形 {5,6,7}）：正确输出 4 个点双——$\{2,3,4\}$、$\{5,6,7\}$ 两个三角形，桥 2-5 两端成 $\{2,5\}$，悬边 1-2 成 $\{1,2\}$，与手推一致。

## 5. 题目清单

| 题号 | 题名 | 平台 | 考的变形 |
|------|------|------|---------|
| [P3388](https://www.luogu.com.cn/problem/P3388) | 【模板】割点（割顶） | 洛谷 | 割点 |
| [P1656](https://www.luogu.com.cn/problem/P1656) | 炸铁路 | 洛谷 | 桥 + 排序输出 |
| [P8435](https://www.luogu.com.cn/problem/P8435) | 【模板】点双连通分量 | 洛谷 | 点双 + 孤立点/自环 |
| [P8436](https://www.luogu.com.cn/problem/P8436) | 【模板】边双连通分量 | 洛谷 | 边双（桥 + 缩点思路） |
| [LC 1192](https://leetcode.cn/problems/critical-connections-in-a-network/) | 查找集群内的关键连接 | LeetCode | 桥（无重边，`low[v] > dfn[u]`） |

LeetCode 上这类题稀缺（图论重灾区是遍历与拓扑，见 [AT 11](../AT_11_graph_bfs_dfs/)）——割点/桥/BCC 的主战场在洛谷与其他 OJ，建议按 P3388 → P1656 → P8435 → P8436 顺序刷。

## 6. 与其他篇的联系

- 理论后盾：[DSA 11 §2–7](../../DSA/DSA_11/)——hca（Highest Connected Ancestor）就是本篇的 low，判定准则的完整论证在那边；
- [AT 11（图的存储与遍历）](../AT_11_graph_bfs_dfs/)：邻接表、DFS 骨架是本篇的地基；
- [AT 10（并查集）](../AT_10_union_find/)：边双缩点 = 求桥 + 并查集合并；
- 全系列地图与 STL 选型速查：[AT 00](../AT_00_reading_guide/)。

下一篇 [AT 15](../AT_15_segment_tree/) 换赛道：从图论的 DFS 时间轴跳到区间上的分治树。
