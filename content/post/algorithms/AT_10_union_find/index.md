---
title: "AT 10: 并查集——把等价关系维护到近乎 O(1)"
date: 2026-12-03
math: true
tags:
  - Notes
  - 算法模板
  - 并查集
categories:
  - 算法模板
description: "并查集模板全家桶：路径压缩+按秩合并的基础版与带统计版、带权并查集（权值随路径压缩累加，食物链为标杆）、扩展域（敌人拆虚拟节点，团伙为标杆）。样题：洛谷 P3367 模板、P1892 团伙、P2024 食物链、LC 684 冗余连接"
---

# AT 10：并查集——把等价关系维护到近乎 $O(1)$

> 本篇是 [DSA 刷题模板系列](../AT_00_reading_guide/)第 10 篇。核心思想一句话：**并查集只回答"是否同属一个集合"，不回答集合长什么样——把动态等价关系的维护降到均摊近常数。**
> 理论后盾：[DSA 11 §18](../../DSA/DSA_11/)——从 Quick-Find / Quick-Union 到加权合并 + 路径压缩的演进逻辑。本篇直接给最终形态，再往上有两个变体（带权、扩展域），是刷题从"会模板"到"会建模"的分水岭。

---

## 1. 这个模板解决什么问题

凡是"**用等价关系把元素分组，随时查询/合并**"，就是并查集：

| 场景 | 等价关系 | 典型题 |
|------|---------|--------|
| 连通分量计数 | 连通 | LC 547 省份数量、P1536 村村通 |
| 判环 / 冗余边 | "在同一棵树里" | LC 684 冗余连接 |
| Kruskal MST | 加入边时不成环 | [AT 13](../AT_13_mst/) |
| 传递关系分类 | 同类 / 敌友 / 吃与被吃 | P1892 团伙、P2024 食物链 |

它的对手是"每次 BFS/DFS 全图重算"——$O(q \cdot (n+m))$ 的做法在 $10^5$ 次询问下必死，并查集把每次合并/查询做到 $\alpha(n)$（反阿克曼函数，$n \le 10^9$ 时不超过 4，视为常数）。

**为什么两个优化缺一不可**（DSA 11 §18.4 的结论）：只路径压缩，树可以暂时很高；只按秩合并，单次 `find` 仍可能走长路。两者结合均摊 $O(\alpha(n))$。刷题实现上路径压缩必写（一行），按秩/按大小合并也顺手写上（三行），不省这个。

---

## 2. 代码模板

### 2.1 基础版：路径压缩 + 按大小合并

```cpp
struct DSU {
    vector<int> fa, sz;                  // fa: 父指针（根指向自己）；sz: 仅根有效，存集合大小
    DSU(int n) : fa(n + 1), sz(n + 1, 1) { iota(fa.begin(), fa.end(), 0); }
    int find(int x) {
        return fa[x] == x ? x : fa[x] = find(fa[x]);    // 路径压缩：父直接指向根
    }
    bool same(int x, int y) { return find(x) == find(y); }
    bool unite(int x, int y) {           // 返回"是否真的合并了"——判环时直接用
        x = find(x); y = find(y);
        if (x == y) return false;
        if (sz[x] < sz[y]) swap(x, y);   // 小树挂大树
        fa[y] = x; sz[x] += sz[y];
        return true;
    }
};
```

三个要点：

1. **`find` 的递归一行版**是精髓：`fa[x] = find(fa[x])` 在回溯时把路径上每个节点直接挂到根——先递归再改写，顺序不能反；
2. **`unite` 返回 `bool`**：合并失败（本来同集合）就是"成环"信号，LC 684 一行判定；统计连通块数时"每成功合并一次减一"也靠它；
3. **`sz` 只在根处有效**，中间节点的 `sz` 是过期的历史值，别读。

### 2.2 带权并查集：给每条父子边挂一个权

维护 $d_x$ = $x$ 到**其父节点**的关系权（加法意义下可累积，如距离、类别差）。`find` 时路径压缩，权值沿途累加：

```cpp
int find(int x) {
    if (fa[x] == x) return x;
    int root = find(fa[x]);              // 先把父链压完，fa[x] 的 d 已更新完毕
    d[x] = (d[x] + d[fa[x]]) % 3;        // 新权 = 到旧父的权 + 旧父到根的权
    fa[x] = root;
    return root;
}
```

**最容易错的一行**：必须**先递归、再累加**——`d[fa[x]]` 在递归返回后已经是"旧父到根"的全权值，先累加后递归就用了半新半旧的值。合并两棵树时按"当前这条新关系"解方程定根的权（见 §4.3 食物链）。

### 2.3 扩展域：把"关系"编码进节点编号

"敌人的敌人是朋友"这类**多类互斥关系**，不必上带权版：把每个元素 $x$ 拆成多个虚拟节点——$x$ 本体与 $x+n$（敌人域）。规则翻译成四次合并：

```cpp
// 朋友 x-y：同类，两个域各自合并
unite(x, y); unite(x + n, y + n);
// 敌人 x-y：我的敌人是你的朋友 → 本体连对方的敌人域
unite(x, y + n); unite(x + n, y);
```

如果 $x$ 与 $x+n$ 被并进同一集合，说明出现"自己是自己的敌人"的矛盾。域的个数 = 类别数（两类拆 2 域，三类可拆 3 域或改用带权版）。

---

## 3. 常见变形与坑

| 变形 | 一句话做法 |
|------|-----------|
| 连通块计数 | 初始答案 $=n$，每次 `unite` 成功减一；或最后数 `find(i)==i` 的个数 |
| 判环/找冗余边 | 逐条边 `unite`，第一次失败的边即所求（LC 684） |
| 集合大小查询 | `sz[find(x)]`（须用按大小合并版） |
| 带关系类别（同类/捕食） | 带权并查集，权 mod 类别数（P2024） |
| 敌友传递 | 扩展域，$x$ 与 $x+n$（P1892） |
| 删点/删边 | 并查集**不支持删除**——离线倒序做，删除变撤销=合并（经典套路） |
| 边权最小化连通（MST） | 排序 + 逐边 `unite`，即 Kruskal（[AT 13](../AT_13_mst/)） |

坑清单：

- **`find` 不写路径压缩**：链式数据直接 $O(n)$ 单次，TLE 得很冤；
- **带权版先改 `fa[x]` 再算权**：`d[fa[x]]` 就取错了，§2.2 的顺序是铁律；
- **`unite` 两边不先 `find`**：比较的是两个内部节点，逻辑直接错；
- **数组开小**：扩展域要开 $2n$，带权版看数据范围（P2024 要 $5 \times 10^4 + 5$）；
- **想在线删除**：并查集没有删除，换"倒序离线"或线段树分治。

---

## 4. 样题精讲

### 4.1 洛谷 P3367 【模板】并查集——模板验收

> [P3367 【模板】并查集](https://www.luogu.com.cn/problem/P3367)（普及/提高-）：操作 `M x y` 合并、`Q x y` 查询是否同集合。

直接套 §2.1，完整可提交程序：

```cpp
#include <cstdio>
#include <numeric>
#include <vector>
using namespace std;

struct DSU {
    vector<int> fa, sz;
    DSU(int n) : fa(n + 1), sz(n + 1, 1) { iota(fa.begin(), fa.end(), 0); }
    int find(int x) { return fa[x] == x ? x : fa[x] = find(fa[x]); }
    bool same(int x, int y) { return find(x) == find(y); }
    void unite(int x, int y) {
        x = find(x); y = find(y);
        if (x == y) return;
        if (sz[x] < sz[y]) swap(x, y);
        fa[y] = x; sz[x] += sz[y];
    }
};

int main() {
    int n, m; scanf("%d %d", &n, &m);
    DSU d(n);
    while (m--) {
        char op[3]; int x, y;
        scanf("%s %d %d", op, &x, &y);
        if (op[0] == 'M') d.unite(x, y);
        else puts(d.same(x, y) ? "Y" : "N");
    }
    return 0;
}
```

### 4.2 LC 684 冗余连接——`unite` 返回值就是判环器

> [LC 684. 冗余连接](https://leetcode.cn/problems/redundant-connection/)（中等）：树加一条多余边，返回成环的那条（多条则返回最后出现的）。

**怎么套模板**：逐条边尝试 `unite`——第一次"合并失败"（两端已连通）的边就是环上最后出现的那条，直接返回：

```cpp
class Solution {
    vector<int> fa;
    int find(int x) { return fa[x] == x ? x : fa[x] = find(fa[x]); }
public:
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        fa.resize(n + 1);
        iota(fa.begin(), fa.end(), 0);
        for (auto& e : edges) {
            int a = find(e[0]), b = find(e[1]);
            if (a == b) return e;          // 第一次成环的边即答案
            fa[a] = b;
        }
        return {};
    }
};
```

$O(n\alpha(n))$。同款思路：LC 547 省份数量（连通块计数，初始答案 $n$、成功合并减一）。比 DFS 找环省一个数量级的代码量——**判环题先想并查集**。

### 4.3 洛谷 P2024 食物链——带权并查集的标杆

> [P2024 [NOI2001] 食物链](https://www.luogu.com.cn/problem/P2024)（普及+/提高）：$n$ 只动物三类关系（A 吃 B，B 吃 C，C 吃 A），$k$ 句话，判假话数。话分两种："$x$ 与 $y$ 同类"、"$x$ 吃 $y$"；编号超出 $n$ 或自相矛盾都算假话。

**建模**：给每条父子边挂权 $d_x \in \{0,1,2\}$——$d_x = 0$ 表示 $x$ 与父同类，$1$ 表示 $x$ 吃父，$2$ 表示 $x$ 被父吃。关系可加（模 3）：$x$ 对 $y$ 的关系 $= d_x - d_y \pmod 3$（沿树换根的差值）。

于是两句话的判定只是一次模减法：

| 话 | 已知同根时为真当且仅当 |
|----|----------------------|
| $x$ 同类 $y$ | $(d_x - d_y) \bmod 3 = 0$ |
| $x$ 吃 $y$ | $(d_x - d_y) \bmod 3 = 1$ |

不同根则该话无法证伪，为真——同时**合并两树**：挂 $fa[r_x] = r_y$，新边的权 $d_{r_x}$ 由"要求 $d_x + d_{r_x} \equiv d_y + r$"解出。

```cpp
#include <cstdio>
const int N = 50005;
int fa[N], d[N];                     // d[x]: x 到父的关系（0 同类 / 1 吃父 / 2 被父吃）

int find(int x) {
    if (fa[x] == x) return x;
    int root = find(fa[x]);          // 先递归：d[fa[x]] 已是"旧父到根"的全权值
    d[x] = (d[x] + d[fa[x]]) % 3;    // 再累加（顺序不可换）
    fa[x] = root;
    return root;
}

int main() {
    int n, k; scanf("%d %d", &n, &k);
    for (int i = 1; i <= n; i++) fa[i] = i, d[i] = 0;
    int ans = 0;
    while (k--) {
        int op, x, y; scanf("%d %d %d", &op, &x, &y);
        if (x > n || y > n) { ans++; continue; }          // 编号越界：假话
        int rx = find(x), ry = find(y);
        if (op == 1) {                                     // 断言：x y 同类
            if (rx == ry && d[x] != d[y]) ans++;           // 同根但类别差非 0
            else if (rx != ry) {                           // 无法证伪 → 合并
                fa[rx] = ry;
                d[rx] = ((d[y] - d[x]) % 3 + 3) % 3;       // 解方程 d[x]+d[rx] ≡ d[y]
            }
        } else {                                           // 断言：x 吃 y
            if (x == y) { ans++; continue; }               // 自己吃自己：假话
            if (rx == ry && (d[x] - d[y] + 3) % 3 != 1) ans++;
            else if (rx != ry) {
                fa[rx] = ry;
                d[rx] = ((d[y] - d[x] + 1) % 3 + 3) % 3;   // 要求 d[x]+d[rx] ≡ d[y]+1
            }
        }
    }
    printf("%d\n", ans);
    return 0;
}
```

两次 `((...) % 3 + 3) % 3` 是 C++ 负数取模的标准处理——`%` 对负数返回负值，忘了 `+3` 是这题最常见 WA。验证：官方样例（含"101 超界、自己吃自己、1 与 3 由传递性不同类"三个假话）输出 3，实测一致。

### 4.4 洛谷 P1892 团伙——扩展域的标杆

> [P1892 [BalticOI 2003] 团伙](https://www.luogu.com.cn/problem/P1892)（普及+/提高）：$n$ 人，朋友的朋友是朋友，敌人的敌人是朋友，给 $m$ 条 F/E 关系，求最多团体数。

**建模**：§2.3 的扩展域，$x$ 与 $x+n$ 互为敌人域。朋友：两域各自合并；敌人：交叉合并。最后数 $1..n$ 里不同根的个数：

```cpp
#include <cstdio>
const int N = 2005;
int fa[N * 2];                        // 扩展域：x 与 x+n
int find(int x) { return fa[x] == x ? x : fa[x] = find(fa[x]); }
void unite(int x, int y) { fa[find(x)] = find(y); }

int main() {
    int n, m; scanf("%d %d", &n, &m);
    for (int i = 1; i <= 2 * n; i++) fa[i] = i;
    while (m--) {
        char op[3]; int x, y;
        scanf("%s %d %d", op, &x, &y);
        if (op[0] == 'F') { unite(x, y); unite(x + n, y + n); }        // 朋友
        else              { unite(x, y + n); unite(x + n, y); }        // 敌人
    }
    static bool vis[N * 2] = {false};
    int ans = 0;
    for (int i = 1; i <= n; i++) {     // 只数本体节点里的不同根
        int r = find(i);
        if (!vis[r]) { vis[r] = true; ans++; }
    }
    printf("%d\n", ans);
    return 0;
}
```

**为什么必须交叉合并**：`unite(x, y+n)` 说的是"我的本体和你的敌人同伙"——由"敌人的敌人是朋友"反向推出"朋友的敌人是敌人"，两次交叉合并恰好把这两条传递规则全部编码。手玩样例：9 人 7 条关系（含 E 1 3、E 7 9 两条敌对）得 4 组，与程序一致。

---

## 5. 题目清单

| 题号 | 题名 | 难度 | 考的变形 |
|------|------|------|---------|
| [洛谷 P3367](https://www.luogu.com.cn/problem/P3367) | 【模板】并查集 | 普及/提高- | 基础版 |
| [洛谷 P1551](https://www.luogu.com.cn/problem/P1551) | 亲戚 | 普及/提高- | 基础版（同根判定） |
| [洛谷 P1536](https://www.luogu.com.cn/problem/P1536) | 村村通 | 普及- | 连通块计数 |
| [洛谷 P1892](https://www.luogu.com.cn/problem/P1892) | [BalticOI 2003] 团伙 | 普及+/提高 | 扩展域 |
| [洛谷 P2024](https://www.luogu.com.cn/problem/P2024) | [NOI2001] 食物链 | 普及+/提高 | 带权并查集 |
| [LC 547](https://leetcode.cn/problems/number-of-provinces/) | 省份数量 | 中等 | 连通块计数 |
| [LC 684](https://leetcode.cn/problems/redundant-connection/) | 冗余连接 | 中等 | 判环（unite 返回值） |

进阶方向：带撤销的并查集（离线倒序）、线段树分治、可撤销并查集（按秩合并不路径压缩，支持 rollback）——出现在省选及以上，本篇不展开。

---

## 6. 与其他篇的联系

- 理论：[DSA 11 §18](../../DSA/DSA_11/)——Quick-Find → Quick-Union → 双优化的演进；§18 的结论"Kruskal = 排序 + 并查集"在 [AT 13 最小生成树](../AT_13_mst/)兑现；
- [AT 11 图的存储与遍历](../AT_11_graph_bfs_dfs/)：连通分量也能 BFS/DFS 数，但边动态到来/多次查询时并查集完胜；
- [AT 13 最小生成树](../AT_13_mst/)：Kruskal 的骨架就是本文 `unite` 的返回值；
- [AT 14 关节点与双连通分量](../AT_14_articulation_bcc/)：边双连通分量缩点用并查集是常见巧解，Tarjan 系列也会反复和它配合；
- LC 547 这类"数连通块"在网格图上另有 flood fill 写法（[AT 11](../AT_11_graph_bfs_dfs/)），点图用并查集更顺手。

---

*下一篇 [AT 11：图的存储与遍历](../AT_11_graph_bfs_dfs/)——邻接表、BFS/DFS 与拓扑排序。*
