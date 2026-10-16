---
title: "AT 15: 线段树与树状数组——区间修改与 lazy 下推"
date: 2026-12-08
math: true
tags:
  - Notes
  - 算法模板
  - 线段树
  - 树状数组
  - 数据结构
categories:
  - 算法模板
description: "线段树从单点改到区间加 lazy 下推再到乘加双标记（下推顺序先乘后加）的递进模板，树状数组单点加/区间加点查的差分技巧，附 P3372/P3373/P3374/P3368 完整代码与 LC 307/315"
---

# AT 15：线段树与树状数组——区间修改与 lazy 下推

> 本篇是 AT 系列第 15 篇。核心思想一句话：**一次区间修改影响的节点有 $O(n)$ 个，但其中绝大多数是"整段被覆盖"的——对它们只记账不下推，等查询真正路过时再结账。** 这就是懒标记（lazy tag）。理论原型在 [DSA 7 §10](../../DSA/DSA_7/) 的规范覆盖（canonical subsets），本篇给可提交的武器。

---

## 1. 这个模板解决什么问题

| 需求 | 前缀和 | 树状数组 | 线段树 |
|------|--------|---------|--------|
| 区间求和（静态） | ✅ $O(n)$ 预处理 + $O(1)$ 查询 | ✅ | ✅ |
| 单点修改 + 区间查询 | ❌（改一次要 $O(n)$ 重算） | ✅ $O(\log n)$ | ✅ $O(\log n)$ |
| 区间修改 + 区间查询 | ❌ | 🔶 差分只支持点查 | ✅ $O(\log n)$（lazy） |
| 区间最值 / 不可减信息 | ❌ | ❌（只管可减信息） | ✅ |
| 代码量 / 常数 | 最小 | 最小 | 中等 |

选型结论（速记）：**能树状数组就树状数组（短、常数小），要区间改区间查或最值再上线段树**。

线段树把序列挂在二叉树上：节点 $p$ 管区间 $[l,r]$，左孩子管 $[l,\text{mid}]$、右孩子管 $[\text{mid}+1,r]$。区间 $[q_l,q_r]$ 与节点区间的关系只有三种——完全覆盖（打标记走人）/ 完全不相交（返回）/ 部分相交（下推标记后递归孩子）。**每个区间最多被拆成 $O(\log n)$ 个规范节点**，这正是 [DSA 7 §10.4](../../DSA/DSA_7/) 的 `InsertSegment`：贪心地把大区间存进"整段被覆盖"的最大节点。那边的 canonical subset 就是 lazy 标记的理论名字，那边的"查询沿根到叶路径累加"就是本篇的 pushdown。

## 2. 代码模板：区间加 + 区间求和

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int n, m;
vector<ll> sum_, tag;

void pushup(int p) { sum_[p] = sum_[p<<1] + sum_[p<<1|1]; }   // 用孩子重算自己
void build(int p, int l, int r, vector<ll>& a) {
    if (l == r) { sum_[p] = a[l]; return; }
    int mid = (l + r) >> 1;
    build(p<<1, l, mid, a); build(p<<1|1, mid+1, r, a);
    pushup(p);
}
void pushdown(int p, int l, int r) {            // 把 p 的懒标记结账给孩子
    if (tag[p] == 0) return;
    int mid = (l + r) >> 1, lc = p<<1, rc = p<<1|1;
    tag[lc] += tag[p]; tag[rc] += tag[p];       // 孩子也先记账
    sum_[lc] += tag[p] * (mid - l + 1);         // 孩子的和 = 原和 + 标记 × 孩子长度
    sum_[rc] += tag[p] * (r - mid);
    tag[p] = 0;
}
void update(int p, int l, int r, int ql, int qr, ll k) {
    if (ql <= l && r <= qr) {                   // 完全覆盖：记账直接返回
        sum_[p] += k * (r - l + 1); tag[p] += k; return;
    }
    pushdown(p, l, r);                          // 要碰孩子了，先结清旧账
    int mid = (l + r) >> 1;
    if (ql <= mid) update(p<<1, l, mid, ql, qr, k);
    if (qr > mid)  update(p<<1|1, mid+1, r, ql, qr, k);
    pushup(p);
}
ll query(int p, int l, int r, int ql, int qr) {
    if (ql <= l && r <= qr) return sum_[p];
    pushdown(p, l, r);
    int mid = (l + r) >> 1; ll res = 0;
    if (ql <= mid) res += query(p<<1, l, mid, ql, qr);
    if (qr > mid)  res += query(p<<1|1, mid+1, r, ql, qr);
    return res;
}
```

**命名约定**：`pushup`（自底向上重算）、`pushdown`（自顶向下结账）——四个函数一个结构，换信息只改 `pushup`/`pushdown`/打标记三处。

**核心不变量**：`sum_[p]` 永远等于节点 $p$ 区间的**真实和**——打标记时当场把自己的 `sum_` 算对（`sum_[p] += k * len`）；但**两个孩子还欠着这笔账**（`tag[p]` 就是欠条）。所以递归进孩子前必须 `pushdown` 结清欠条；而孩子被改过后返回的路上必须 `pushup`，让祖先的 `sum_` 重新对账。

<details><summary>变体：单点修改版与区间最值版（点击展开）</summary>

单点修改更简单——一路下推到叶子、改完一路 pushup，无懒标记：

```cpp
void update(int p, int l, int r, int x, ll k) {     // a[x] = k（或 += k）
    if (l == r) { sum_[p] += k; return; }
    int mid = (l + r) >> 1;
    x <= mid ? update(p<<1, l, mid, x, k) : update(p<<1|1, mid+1, r, x, k);
    pushup(p);
}
```

区间最值（max）版：`pushup` 改 `max`、`pushdown` 改"孩子 max 与父亲比较"（区间加对 max 的作用是平移，标记仍是加法）；`query` 改两半取 max。P3372 的姊妹题只需这三处替换。
</details>

## 3. 常见变形与坑

| 变形 | 一句话做法 |
|------|-----------|
| 单点加 + 区间和 | 上一节的单点版，或干脆树状数组（§4.3） |
| 区间加 + 区间和 | 本篇主模板 |
| 区间乘 + 区间加 | 双标记，值域变换 $v \mapsto m\cdot v + a$，**下推先乘后加**（§4.2） |
| 区间赋值（覆盖） | 标记换成"赋值标记"，pushdown 直接覆盖（注意与加法标记混合时的优先级） |
| 区间最值 | pushup/pushdown/query 三处改 max，标记仍可为加法 |
| 动态开点 / 权值线段树 | 节点按需创建，值域 $10^{18}$ 也能建 |
| 树上路径 / 子树 | 树剖（重链）或 DFS 序拍平后套普通线段树 |

坑清单：

1. **数组开 $4n$**——递归中点的取整方式让满二叉布局需要 $4n$，开 $2n$ 会偶发 RE；
2. **`long long`**——区间和最坏 $n \cdot \max(a) \cdot m$ 次累加，`int` 必炸；
3. **忘 pushdown 就递归孩子**——读到的是"没结账"的旧值（最隐蔽的 WA）；
4. **双标记下推顺序**——孩子标记的复合必须是"先乘后加"：$\text{tag}_{child}' = m\cdot\text{tag}_{child} + a$，写反了乘法会把父层加法标记一起放大；
5. **build 忘了在叶子赋值** / **update 打完标记忘 pushup**；
6. `mid = (l+r)>>1` 配 `ql <= mid` / `qr > mid` 的边界是配套的，别改成 `mid-1` 风格的一半。

## 4. 样题精讲

### 4.1 [洛谷 P3372 【模板】线段树 1](https://www.luogu.com.cn/problem/P3372)

区间加 + 区间求和，就是 §2 模板原样（主函数读入 `n, m`、建树、循环处理操作，完整程序见 §2 拼上标准 `main` 即可，官方样例输出 `11 8 20` 已实测通过）。复杂度 $O((n+m)\log n)$。

### 4.2 [洛谷 P3373 【模板】线段树 2](https://www.luogu.com.cn/problem/P3373)

区间乘 + 区间加 + 区间和对 $p$ 取模——双标记的试金石。关键设计：**把两种标记统一成一个仿射变换 $v \mapsto m\cdot v + a$**，打乘法标记就是 $(m, 0)$、打加法标记就是 $(1, k)$，复合有统一公式：

```cpp
void apply_tag(int p, int len, ll m, ll a) {     // 复合标记 v -> m*v + a
    sum_[p] = (sum_[p] * m + a * len) % MOD;
    mul[p] = (mul[p] * m) % MOD;
    add[p] = (add[p] * m + a) % MOD;             // 先乘后加，顺序写反全错
}
void pushdown(int p, int l, int r) {
    int mid = (l + r) >> 1;
    apply_tag(p<<1,   mid - l + 1, mul[p], add[p]);
    apply_tag(p<<1|1, r - mid,     mul[p], add[p]);
    mul[p] = 1; add[p] = 0;
}
```

为什么 `add[p] = add[p] * m + a`：孩子原有的账是"先乘 $\text{mul}_{child}$ 再加 $\text{add}_{child}$"，父亲新来的账 $v\mapsto mv+a$ 要套在外面——$(m\cdot v + a)$ 与旧账复合时，旧 add 也被 $m$ 放大。这就是坑 4 的机制版。主循环三个操作分别是 `update(..., k, 0)`（乘）、`update(..., 1, k)`（加）、`query`。官方样例（$p=38$）输出 `3 15` 已实测通过。

### 4.3 [洛谷 P3374 / P3368：树状数组与差分](https://www.luogu.com.cn/problem/P3374)

树状数组（Binary Indexed Tree，BIT）`tr[x]` 管结尾为 $x$、长度 $\text{lowbit}(x)$ 的一段和，两个方向各一个循环：

```cpp
int lowbit(int x) { return x & (-x); }
void update(int x, ll k) { for (; x <= n; x += lowbit(x)) tr[x] += k; }
ll prefix(int x) { ll s = 0; for (; x > 0; x -= lowbit(x)) s += tr[x]; return s; }
ll query(int l, int r) { return prefix(r) - prefix(l - 1); }   // 区间和 = 两前缀差
```

- **P3374（单点加 + 区间和）**：上四行 + 建树时逐个 `update`，完事。官方样例 `14 5 17` 实测通过。
- **P3368（区间加 + 单点查）**：维护**差分数组** $d$（$d_i = a_i - a_{i-1}$），则 $a_x = \sum_{i\le x} d_i$（前缀和查询），区间 $[l,r]$ 加 $k$ 就是 `update(l, k)` + `update(r+1, -k)`。官方样例 `6 3 2` 实测通过。差分思想与 [前缀和](../prefix_sum/) 一文同源——树状数组就是"可修改的前缀和"。

**BIT 的边界**：区间加 + **区间和** 也能做（二阶差分，$O(2\log n)$ 每操作），但要维护 $\sum i\cdot d_i$，代码开始比线段树还绕；超过单点改/区间加点查就别硬撑，直接线段树。

### 4.4 [LC 307. 区域和检索 - 数组可修改](https://leetcode.cn/problems/range-sum-query-mutable/)

`class Solution` 形式的单点覆盖 + 区间和。树状数组版（下标平移到 1 起）：

```cpp
class NumArray {
    int n;
    vector<long long> tr;
    int lowbit(int x) { return x & (-x); }
    void add(int x, long long k) { for (; x <= n; x += lowbit(x)) tr[x] += k; }
    long long prefix(int x) { long long s = 0; for (; x > 0; x -= lowbit(x)) s += tr[x]; return s; }
public:
    NumArray(vector<int>& nums) : n(nums.size()), tr(n + 1, 0) {
        for (int i = 0; i < n; i++) add(i + 1, nums[i]);
    }
    void update(int index, int val) {                 // 单点覆盖 = 撤旧值 + 加新值
        long long old = prefix(index + 1) - prefix(index);
        add(index + 1, val - old);
    }
    int sumRange(int left, int right) {
        return (int)(prefix(right + 1) - prefix(left));
    }
};
```

`update` 要先算出旧值再补差——BIT 天生只支持"加增量"，覆盖语义要自己翻译。样例 `9 8 8` 实测通过。

## 5. 题目清单

| 题号 | 题名 | 平台 | 考的变形 |
|------|------|------|---------|
| [P3372](https://www.luogu.com.cn/problem/P3372) | 【模板】线段树 1 | 洛谷 | 区间加 + 区间和 |
| [P3373](https://www.luogu.com.cn/problem/P3373) | 【模板】线段树 2 | 洛谷 | 乘加双标记 + 取模 |
| [P3374](https://www.luogu.com.cn/problem/P3374) | 【模板】树状数组 1 | 洛谷 | 单点加 + 区间和 |
| [P3368](https://www.luogu.com.cn/problem/P3368) | 【模板】树状数组 2 | 洛谷 | 区间加 + 单点查（差分） |
| [LC 307](https://leetcode.cn/problems/range-sum-query-mutable/) | 区域和检索 - 数组可修改 | LeetCode | BIT 单点覆盖 + 区间和 |
| [LC 315](https://leetcode.cn/problems/count-of-smaller-numbers-after-self/) | 计算右侧小于当前元素的个数 | LeetCode | 权值线段树 / BIT 离散化 + 倒序统计 |

LC 315 提示：线段树不只挂"下标轴"，还能挂**值域轴**（权值线段树）——倒序扫，每个数查询值域 $[0, a_i)$ 内已出现个数，再插入 $a_i$。

## 6. 与其他篇的联系

- 理论后盾：[DSA 7 §9–10](../../DSA/DSA_7/)——elementary intervals、规范覆盖与 $O(n\log n)$ 空间的完整论证；§9 的 Interval Tree（stabbing query）是"存区间、查点"的对偶结构，刷题中出场率远低于本篇，概念回顾即可；
- [前缀和 / 差分](../prefix_sum/)：无修改时的最优解，树状数组差分技巧的直接源头；
- [AT 16（kd-tree）](../AT_16_kdtree/)：同出 DSA 7 章，把"区间分治"从一维搬到二维平面；
- 全系列地图：[AT 00](../AT_00_reading_guide/)。
