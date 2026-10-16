---
title: "AT 16: kd-tree 与平面最近点对——空间分治双武器"
date: 2026-12-09
math: true
tags:
  - Notes
  - 算法模板
  - 计算几何
  - 分治
categories:
  - 算法模板
description: "平面最近点对的两套解法对照：经典分治（中位切半 + 2d 窄带 + y 序双指针，O(n log n) 有理论保证）与 kd-tree（nth_element 轮维建树 + 超矩形剪枝，随机数据表现好但最坏退化）。附 P1429 双解法完整代码、实测性能与两个真实踩坑（nth_element 中位数落点、重合点跳过自己）"
---

# AT 16：kd-tree 与平面最近点对——空间分治双武器

> 本篇是 AT 系列第 16 篇。核心思想一句话：**把平面递归切成小块，任何查询只跟"离得近的块"认真计较。** 理论（canonical subsets、Accepted/Rejected/Recursion 三分类、复杂度分析）在 [DSA 7 §5–6](../../DSA/DSA_7/)，本篇给两套可提交的武器和它们的适用边界。

---

## 1. 这个模板解决什么问题

| 问题 | 定义 | 解法 |
|------|------|------|
| 平面最近点对 | $n$ 个点中距离最小的一对 | 经典分治，$O(n\log n)$ **有理论保证**（比赛稳过） |
| 最近邻查询 | 给定查询点，找库中离它最近的点 | kd-tree，单次均摊 $O(\log n)$（随机数据）、最坏 $O(n)$ |
| $k$ 近邻 / 矩形区域统计 | 半径内邻居、矩形内点数 | kd-tree 剪枝框架同一套 |

两者是一对互补武器：**最近点对要"全体点两两负责"→ 用有保证的分治；在线查询/动态加点 → 用 kd-tree**。同一个模板题（P1429）双解法对照最能看清差别，见 §4。

## 2. 代码模板一：平面最近点对分治

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
struct Pt { ll x, y; };
bool byX(const Pt&a, const Pt&b){ return a.x<b.x || (a.x==b.x && a.y<b.y); }
bool byY(const Pt&a, const Pt&b){ return a.y<b.y; }

double solve(vector<Pt>& p, int l, int r, vector<Pt>& buf) {
    if (r - l <= 3) {                              // 暴力收尾（≤4 个点）
        double best = 1e30;
        for (int i = l; i < r; i++) for (int j = i+1; j < r; j++)
            best = min(best, hypot((double)(p[i].x-p[j].x), (double)(p[i].y-p[j].y)));
        sort(p.begin()+l, p.begin()+r, byY);
        return best;
    }
    int mid = (l + r) >> 1; ll midx = p[mid].x;
    double d = min(solve(p, l, mid, buf), solve(p, mid, r, buf));
    // 按 y 归并左右两半（等价于全程只排一次 y）
    merge(p.begin()+l, p.begin()+mid, p.begin()+mid, p.begin()+r, buf.begin()+l, byY);
    copy(buf.begin()+l, buf.begin()+r, p.begin()+l);
    // 收集距中线 < d 的点进窄带
    vector<Pt> band;
    for (int i = l; i < r; i++) if (abs(p[i].x - midx) < d) band.push_back(p[i]);
    // 窄带内按 y 双指针：每个点只需检查后面最多 ~6 个点
    double best = d;
    for (int i = 0; i < (int)band.size(); i++)
        for (int j = i+1; j < (int)band.size() && band[j].y - band[i].y < best; j++)
            best = min(best, hypot((double)(band[i].x-band[j].x), (double)(band[i].y-band[j].y)));
    return best;
}
```

**三个关键设计**：

1. **按 $x$ 切、按 $y$ 合**——递归前整体按 $x$ 排序（`main` 里一次），回溯时用 `merge` 把两半按 $y$ 归并，每层 $O(n)$，总 $O(n\log n)$；避免窄带阶段重新排序；
2. **窄带宽度 $d$**——跨中线的点对必须两侧各距中线 $< d$，带外点对不可能更优；
3. **带内双指针的常数保证**——带内按 $y$ 排序后，与某点距离 $< d$ 的后续点全部落在 $2d\times d$ 的矩形里，由鸽笼原理这种点至多常数个，内层 `while` 摊还 $O(1)$。

## 3. 代码模板二：kd-tree 最近邻

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int n;
struct Node { ll x[2]; int lc, rc; ll lo[2], hi[2]; } t[200005];
int cnt = 0;
double best = 1e30;

int build(int l, int r, int d) {                 // 对 [l,r) 按第 d 维建树
    if (l >= r) return 0;
    int mid = (l + r) >> 1;
    nth_element(t+l, t+mid, t+r, [&](const Node&a, const Node&b){ return a.x[d] < b.x[d]; });
    int p = mid; int nd = d ^ 1;                 // 中位数恰好落在 t[mid]，节点就是它
    t[p].lc = build(l, mid, nd); t[p].rc = build(mid+1, r, nd);
    t[p].lo[0]=t[p].hi[0]=t[p].x[0]; t[p].lo[1]=t[p].hi[1]=t[p].x[1];
    for (int c : {t[p].lc, t[p].rc}) if (c) {    // 子树矩形 = 自己 ± 两孩子的矩形
        for (int k = 0; k < 2; k++) {
            t[p].lo[k]=min(t[p].lo[k], t[c].lo[k]);
            t[p].hi[k]=max(t[p].hi[k], t[c].hi[k]);
        }
    }
    return p;
}
double sq(double v){ return v*v; }
double distToRect(int p, ll x, ll y) {           // 点到该子树超矩形的最小距离（剪枝下界）
    double dx = 0, dy = 0;
    if (x < t[p].lo[0]) dx = t[p].lo[0]-x; else if (x > t[p].hi[0]) dx = x-t[p].hi[0];
    if (y < t[p].lo[1]) dy = t[p].lo[1]-y; else if (y > t[p].hi[1]) dy = y-t[p].hi[1];
    return sqrt(sq(dx)+sq(dy));
}
void query(int p, int self) {                    // self：查询点自己的节点下标
    ll x = t[self].x[0], y = t[self].x[1];
    if (!p) return;
    if (p != self)                               // 只跳过自己这个节点（坐标重合的其他点不算）
        best = min(best, sqrt(sq((double)(t[p].x[0]-x))+sq((double)(t[p].x[1]-y))));
    double dl = t[p].lc ? distToRect(t[p].lc, x, y) : 1e30;
    double dr = t[p].rc ? distToRect(t[p].rc, x, y) : 1e30;
    if (dl < dr) {                               // 先搜更近的孩子
        if (dl < best) query(t[p].lc, self);
        if (dr < best) query(t[p].rc, self);
    } else {
        if (dr < best) query(t[p].rc, self);
        if (dl < best) query(t[p].lc, self);
    }
}
```

这就是 [DSA 7 §6.1](../../DSA/DSA_7/) 三分类的距离版：`distToRect >= best` → **rejected**（整块剪掉）；`p != self` 的当前点 → 直接比较；否则 **recursion**。`distToRect` 是孩子子树内**任何点**到查询点距离的下界——下界都超不过当前最优，整块再见。

`main` 里建树后对每个点 `query(root, i)`，全程维护全局 `best`（只减不增，天然合法剪枝）。**注意查询点就是树里的节点**——kd-tree 求最近点对不需要额外建索引。

## 4. 样题精讲：[洛谷 P1429 平面最近点对（加强版）](https://www.luogu.com.cn/problem/P1429)

$n$ 个点（$2 \le n \le 2\times10^5$，整数坐标）求最近点对距离，保留 4 位小数。双解法对照：

| | 分治 | kd-tree |
|---|------|---------|
| 实测耗时（$n=2\times10^5$ 随机数据） | 0.07 s | 0.07 s |
| 最坏复杂度 | $O(n\log n)$ 保证 | $O(n^2)$ 退化（网格化/对抗数据可卡） |
| 代码风险点 | 窄带边界 | off-by-one、重复点（见坑 1/2） |
| 可扩展性 | 只解决本次问题 | 支持在线查询、动态加点（替罪羊式重构） |

**结论**：模板赛（就是求一次最近点对）优先分治——有复杂度保证，不怕构造数据；kd-tree 留给"后续还要查询/加点"的场景。

两个实测踩坑（4000 组随机对拍抓出来的，如实记录）：

1. **`nth_element` 的中位数落点**——`nth_element(t+l, t+mid, t+r)` 把中位数放在**下标 `mid`** 处。首版把节点写成 `p = mid + 1`，该元素同时落在右子树区间里，会被后续递归挪走，树结构整体错乱（对拍 26 组后现形）。口诀：**nth 放哪，节点就是哪**；
2. **重合点跳过自己**——首版用"坐标相等则跳过"来排除查询点自己，但输入里**两个不同点可以坐标完全相同**（真实最近距离为 0），会被一起跳掉。正解是把查询点按**节点下标**传入，只跳 `p == self` 那一个节点。

## 5. 常见变形与坑

| 变形 | 一句话做法 |
|------|-----------|
| $k$ 维最近邻 | 维度取模轮流切（`d = (d+1) % k`），矩形距离逐维累加 |
| $k$ 近邻（KNN） | 递归回溯框架同上，维护大小为 $k$ 的候选集，剪枝下界与"当前第 $k$ 近"比 |
| 矩形区域计数/求和 | 节点存子树和，查询 = Accepted 整块累加 + Rejected 剪掉 + Recursion 下探（DSA 7 §6.1 原版） |
| 动态加点 | 朴素重构（不平衡就整棵重建）或替罪羊树式（失衡比例阈值 $\alpha$ 触发重构） |
| 最近点对带删除/在线 | 一般改用 kd-tree 或 set 启发式合并，分治不适用 |

坑清单：

1. `nth_element` 中位数落点（§4 坑 1）；
2. 重合点按节点下标跳过，不按坐标（§4 坑 2）；
3. **先近后远**的搜索顺序别写反——先搜近的孩子能让 `best` 更快变小、远侧剪得更狠；
4. 剪枝比较用 `distToRect < best`（可能更优才进），写成 `<=` 只是多搜一点不致错，写反方向必 TLE；
5. kd-tree 最坏退化是**结构性质**（DSA 7 §5.3 讲过为什么切中位数：保树高 $O(\log n)$；但对抗性数据仍能让查询路径退化），比赛中先估计数据性质再选武器。

## 6. 题目清单

| 题号 | 题名 | 平台 | 考的变形 |
|------|------|------|---------|
| [P1429](https://www.luogu.com.cn/problem/P1429) | 平面最近点对（加强版） | 洛谷 | 分治（主解）/ kd-tree（对照） |
| [P4148](https://www.luogu.com.cn/problem/P4148) | 简单题 | 洛谷 | kd-tree 动态加点 + 矩形区域和（替罪羊式重构的经典载体） |

LeetCode 上这类题稀缺（LC 几乎不考计算几何离线分治），主战场在洛谷与其他 OJ——按 P1429 → P4148 顺序刷即可。

## 7. 与其他篇的联系

- 理论后盾：[DSA 7 §5–6](../../DSA/DSA_7/)——kd-tree 的构建、三分类查询、$O(\sqrt n + r)$ 复杂度分析；
- [AT 15（线段树）](../AT_15_segment_tree/)：同出 DSA 7 章——线段树是"一维区间"的分治树，kd-tree 是"二维平面"的分治树，剪枝思想（规范节点整块处理）一脉相承；
- [AT 02（二分查找）](../AT_02_binary_search/)：`nth_element` 与快速选择同族，分治取中位数的思维方式共通；
- 全系列地图：[AT 00](../AT_00_reading_guide/)。
