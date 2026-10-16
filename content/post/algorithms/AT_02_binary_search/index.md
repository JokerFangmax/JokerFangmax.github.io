---
title: "AT 02: 二分查找与二分答案——半开区间统一模板"
date: 2026-11-25
math: true
tags:
  - Notes
  - 算法模板
  - 二分查找
categories:
  - 算法模板
description: "半开区间 [lo, hi) 统一模板：存在性二分、左边界/右边界二分、旋转数组、二分答案（答案单调则可二分），样题：LC 34/33/153/69/875/1011/410 与洛谷 P1873/P2440/P2678"
---

# AT 02：二分查找与二分答案——半开区间统一模板

> 核心思想一句话：**把二分写成不变量驱动的半开区间 `[lo, hi)`——"区间里还有未判定元素"当循环条件、"判定后缩边界而不是停在中点"，一个骨架覆盖存在性、左右边界、二分答案全部形态。**
> 理论后盾：[DSA 2 §15–19](../../DSA/DSA_2/)——二分版本 A（三分支）、B/C（改进返回值语义）、Fibonacci 与插值查找（刷题几乎不用，折叠块给速记）。

---

## 1. 这个模板解决什么问题

二分在刷题里有两种截然不同的身份，骨架却相同：

| 身份 | 搜索空间 | 问的问题 | 代表题 |
|------|---------|---------|--------|
| **二分查找** | 有序数组 | 目标在哪个下标 | LC 704 / 34 / 33 / 153 |
| **二分答案** | 单调的答案区间 | 满足条件的最大/最小值 | LC 875 / 1011 / 410、洛谷 P1873 |

二分答案的合法性只有一条判据：**答案区间关于判定条件单调**。例如"吃的速度越大、耗时越短"（单调），所以"在时限内吃完的最小速度"可二分。识别出这条单调性，题目就从"构造"退化成"枚举 + 快速判定"。

二分写法的死循环、差一（off-by-one）错误出名地多——根治办法不是背三套模板，是**盯住一个不变量**：

> 循环不变量：答案（或目标位置）始终在 $[lo, hi]$ 内；每步判定后区间严格变小。

---

## 2. 代码模板

### 2.1 骨架：左边界二分（lower_bound 语义）

```cpp
// 返回第一个 >= x 的下标；全小于 x 时返回 n
int lowerBound(vector<int>& a, int x) {
    int lo = 0, hi = (int)a.size();     // 半开 [lo, hi)：hi 从 n 起步
    while (lo < hi) {                   // 区间非空
        int mi = lo + (hi - lo) / 2;    // 防溢出写法
        if (a[mi] >= x) hi = mi;        // [mid, hi) 全部 >= x，mid 可能是答案，留下
        else            lo = mi + 1;    // [lo, mid] 全部 < x，排除
    }
    return lo;                          // lo == hi，收敛点
}
```

三个必须刻进肌肉记忆的约定：

1. **`hi` 初值是 `n` 而不是 `n-1`**——半开区间，右边界是"越界位置"，天然覆盖"目标比谁都大"的返回；
2. **分支只有缩边界，没有 `return mi`**——这是 DSA 2 §18 版本 C 的精髓：不停在中点，把相等也当成"可能还有更左的"继续缩，循环自然收敛到边界；
3. **`lo = mi + 1` / `hi = mi`**——一个排除 mid 一个保留 mid，两种缩法都严格减小区间，死循环不可能发生。

### 2.2 三个常用变体（一行改语义）

```cpp
// 右边界：最后一个 <= x 的位置（upper_bound 语义减一）
int lastLE(vector<int>& a, int x) {
    int lo = 0, hi = (int)a.size();
    while (lo < hi) {
        int mi = lo + (hi - lo) / 2;
        if (a[mi] <= x) lo = mi + 1;    // 翻转比较方向
        else            hi = mi;
    }
    return lo - 1;                      // 返回 lo-1（DSA 版本 C 的返回约定）
}

// 存在性：找具体某个值（版本 A 三分支，只用于"只要找到"）
int find(vector<int>& a, int x) {
    int lo = 0, hi = (int)a.size();
    while (lo < hi) {
        int mi = lo + (hi - lo) / 2;
        if      (a[mi] < x)  lo = mi + 1;
        else if (a[mi] > x)  hi = mi;
        else                 return mi;
    }
    return -1;
}

// 二分答案：在答案区间 [lo, hi] 上找"最大的 ok 值"
// check(v) 单调：v 越大越容易/越难满足
long long bsAnswer(long long lo, long long hi) {
    while (lo < hi) {                   // 留一个口：答案在 [lo, hi]
        long long mid = lo + (hi - lo + 1) / 2;   // 注意 +1：向上取整
        if (check(mid)) lo = mid;       // mid 可行，往右探
        else            hi = mid - 1;
    }
    return lo;
}
```

**二分答案的 `+1` 是防死循环的关键**：当 `lo = mid`（保留 mid）时，若 `lo + 1 == hi` 且 mid 向下取整等于 lo，区间不再缩小——所以保留左侧时 mid 必须**向上取整**。反过来 `hi = mid`（§2.1）时必须向下取整。一句话：**谁保留 mid，mid 就偏向谁**。

实际刷题更省心的等价写法：答案区间闭包 `[lo, hi]`，循环条件 `lo <= hi`，命中即记录 `ans = mid` 并缩边——不优雅但不会死循环，折叠块给出对照。

<details><summary>闭区间 [lo, hi] + ans 记录版（防手生备胎）</summary>

```cpp
// 求满足 check 的最大值；check 单调（越大越难满足）
long long lo = minV, hi = maxV, ans = -1;
while (lo <= hi) {
    long long mid = (lo + hi) / 2;
    if (check(mid)) { ans = mid; lo = mid + 1; }  // 可行，记下并找更大
    else            hi = mid - 1;
}
// ans == -1 表示全不可行
```

无需关心取整方向，代价是多一个变量、语义不如半开版干净。求最小值时把两条分支交换即可。
</details>

<details><summary>Fibonacci / 插值查找速记（DSA 2 §17/§19）</summary>

两者都只在**比较代价不对称**或**分布均匀可预测**时优化常数：Fibonacci 查找把划分点放在黄金分割处（若左边比较便宜）；插值查找按值分布估位置，期望 $O(\log\log n)$ 但最坏 $O(n)$（分布不均即退化）。刷题中数据不保证均匀，**默认纯二分**即可。
</details>

---

## 3. 常见变形与坑

| 变形 | 一句话做法 |
|------|-----------|
| 第一个/最后一个 ≥ / ≤ / > / < | lowerBound/lastLE 组合（STL：`lower_bound` / `upper_bound`） |
| 有序数组中目标出现次数 | `upper_bound - lower_bound` |
| 旋转有序数组找目标 | 先判 `a[lo] <= a[mid]` 定哪半有序，再看目标在不在这半（LC 33） |
| 旋转数组找最小值 | `a[mid]` 与 `a[hi]` 比，`<` 则最小在 `[mid, hi]` 左侧含 mid（LC 153） |
| 二维有序矩阵找目标 | 每行二分，或从右上角走 $O(m+n)$ 阶梯（[LC 74 题解](../../LeetCode/LeetCode_74/)） |
| 求最小满足 vs 最大满足 | 分支方向互换；记住"最小"用 `hi = mid`（收缩可行域上界） |
| 答案是小数 | 循环固定 100 次或 `while (hi - lo > 1e-7)`，无死循环问题 |

坑：

1. **`mid = (lo + hi) / 2` 溢出**：`lo + hi` 可能超 `int`，答案区间大时（$10^{18}$ 级）必须 `lo + (hi - lo) / 2`；
2. **死循环**：`lo = mid` 配向下取整——记住 §2.2 的"谁保留 mid，mid 偏向谁"；
3. **返回值语义没对齐**：`lowerBound` 返回的是"插入位置"，目标不存在时它也给出合法插入点，别当 `-1` 用；
4. **判空**：`a` 为空时半开模板返回 0（= n = 0），调用方要处理；
5. 二分答案忘了**先证明单调性**——不单调时二分会收敛到错误答案且不报错，最阴险。

---

## 4. 样题精讲

### 4.1 LC 34 查找元素的第一个和最后一个位置：左右边界合体

[题解](../../LeetCode/LeetCode_34/)（Medium）。有序数组找目标区间 `[first, last]`，$O(\log n)$。就是两次 §2 模板：

```cpp
class Solution {
public:
    vector<int> searchRange(vector<int>& a, int target) {
        int first = lowerBound(a, target);            // 第一个 >= target
        if (first == (int)a.size() || a[first] != target)
            return {-1, -1};                          // 不存在
        int last = lowerBound(a, target + 1LL) - 1;   // 第一个 > target 再减一
        return {first, last};
    }
private:
    int lowerBound(vector<int>& a, long long x) {
        int lo = 0, hi = a.size();
        while (lo < hi) {
            int mi = lo + (hi - lo) / 2;
            if (a[mi] >= x) hi = mi; else lo = mi + 1;
        }
        return lo;
    }
};
```

一个小技巧：**"最后一个等于 x" = "第一个 ≥ x+1" 再减一**（`x + 1LL` 防溢出），比单独写右边界版少一个模板。这也正是 `upper_bound(x) - 1` 的语义。

### 4.2 LC 33 搜索旋转排序数组（Medium）：判定哪半有序

[题解](../../LeetCode/LeetCode_33/)。旋转后整体无序，但**从中点切开必有一半是有序的**：

```cpp
class Solution {
public:
    int search(vector<int>& a, int target) {
        int lo = 0, hi = a.size() - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (a[mid] == target) return mid;
            if (a[lo] <= a[mid]) {                    // [lo, mid] 有序
                if (a[lo] <= target && target < a[mid]) hi = mid - 1;
                else                                     lo = mid + 1;
            } else {                                   // [mid, hi] 有序
                if (a[mid] < target && target <= a[hi]) lo = mid + 1;
                else                                     hi = mid - 1;
            }
        }
        return -1;
    }
};
```

二分的前提"有序"被局部化成"有一半必有序"——先判断哪半有序（`a[lo] <= a[mid]`），再用有序半的端点判断目标在不在里面。**判有序必须在判目标之前**，顺序反了就错。每步仍能排除一半，$O(\log n)$。变体：无重复时找旋转点可 LC 153（[题解](../../LeetCode/LeetCode_153/)，`a[mid]` 与 `a[hi]` 比较）；有重复（LC 81）最坏退化为 $O(n)$，只能缩边一格。

### 4.3 洛谷 P1873 砍树：二分答案原型

[P1873](https://www.luogu.com.cn/problem/P1873)。锯片高度 $h$ 越高，砍到的木材越少（单调！），求满足总量 $\ge$ 需求的最大 $h$。二分答案 + 线性判定：

```cpp
#include <bits/stdc++.h>
using namespace std;

int n; long long m;
vector<int> h;

bool ok(int H) {                        // 锯片高度 H 能否凑够 m
    long long sum = 0;
    for (int x : h)
        if (x > H) sum += x - H;        // 高于锯片的部分被砍下
    return sum >= m;
}

int main() {
    scanf("%d %lld", &n, &m);
    h.resize(n);
    for (auto& x : h) scanf("%d", &x);
    int lo = 0, hi = *max_element(h.begin(), h.end());
    while (lo < hi) {                   // 求最大可行 h
        int mid = lo + (hi - lo + 1) / 2;   // 保留 mid，向上取整
        if (ok(mid)) lo = mid;
        else         hi = mid - 1;
    }
    printf("%d\n", lo);
    return 0;
}
```

注意 `ok` 单调递减（$h$ 越大越难满足）且答案取"最大可行"，所以是 `lo = mid` 配向上取整。同构题：[P2440 木材加工](https://www.luogu.com.cn/problem/P2440)（切成 $k$ 段的最大段长）、[P2678 跳石头](https://www.luogu.com.cn/problem/P2678)（最短跳跃距离最大化，判定用贪心移石头）。

### 4.4 LC 875 / 1011 / 410：二分答案三连

三题同一个内核，判定函数不同：

| 题 | 答案域 | 单调性 | 判定 check(v) |
|----|--------|--------|---------------|
| [LC 875 爱吃香蕉的珂珂](https://leetcode.cn/problems/koko-eating-bananas/) | 吃速 $k$ | 越快越快吃完 | $\sum \lceil p_i / k \rceil \le h$ |
| [LC 1011 在 D 天内送达包裹的能力](https://leetcode.cn/problems/capacity-to-ship-packages-within-d-days/) | 载重 $w$ | 越大越快送到 | 贪心装船天数 $\le D$ |
| [LC 410 分割数组的最大值](https://leetcode.cn/problems/split-array-largest-sum/) | 段和上限 $s$ | 上限越松越好分 | 贪心分段数 $\le k$ |

```cpp
// LC 1011 判定：载重 w 时按序贪心装船需要几天
int days(vector<int>& w, long long cap) {
    int d = 1; long long cur = 0;
    for (int x : w) {
        if (x > cap) return INT_MAX;        // 单件超重（防死扣）
        if (cur + x > cap) { d++; cur = 0; }
        cur += x;
    }
    return d;
}
// 主循环：lo = max(单件), hi = sum(全部)，求最小可行 cap（hi = mid 型）
```

识别信号：题目问"**最小的最大 / 最大的最小**"，且放宽答案的约束方向单调——几乎必是二分答案。LC 410 进阶可 DP，但二分答案 $O(n \log \text{sum})$ 完胜。

---

## 5. 题目清单

| 题号 | 题名 | 难度 | 考的变形 | 链接 |
|------|------|------|---------|------|
| LC 704 | 二分查找 | Easy | 三分支存在性 | [题](https://leetcode.cn/problems/binary-search/) |
| LC 35 | 搜索插入位置 | Easy | lowerBound 原型 | [题解](../../LeetCode/LeetCode_35/) |
| LC 34 | 在排序数组中查找元素的第一个和最后一个位置 | Medium | 左右边界 | [题解](../../LeetCode/LeetCode_34/) |
| LC 33 | 搜索旋转排序数组 | Medium | 有序半判定 | [题解](../../LeetCode/LeetCode_33/) |
| LC 153 | 寻找旋转排序数组中的最小值 | Medium | 旋转点 | [题解](../../LeetCode/LeetCode_153/) |
| LC 69 | x 的平方根 | Easy | 二分答案（整数域） | [题](https://leetcode.cn/problems/sqrtx/) |
| LC 74 | 搜索二维矩阵 | Medium | 二维展平/阶梯 | [题解](../../LeetCode/LeetCode_74/) |
| LC 875 | 爱吃香蕉的珂珂 | Medium | 二分答案（最小速度） | [题](https://leetcode.cn/problems/koko-eating-bananas/) |
| LC 1011 | 在 D 天内送达包裹的能力 | Medium | 二分答案（最小载重） | [题](https://leetcode.cn/problems/capacity-to-ship-packages-within-d-days/) |
| LC 410 | 分割数组的最大值 | Hard | 二分答案（最小段和上限） | [题](https://leetcode.cn/problems/split-array-largest-sum/) |
| 洛谷 P1873 | 砍树 | 普及+/提高 | 二分答案原型 | [题](https://www.luogu.com.cn/problem/P1873) |
| 洛谷 P2440 | 木材加工 | 普及+ | 二分答案 | [题](https://www.luogu.com.cn/problem/P2440) |
| 洛谷 P2678 | 跳石头 | 普及+/提高 | 二分答案 + 贪心判定 | [题](https://www.luogu.com.cn/problem/P2678) |

---

## 6. 与其他篇的联系

- 理论：版本 A/B/C 的比较次数分析、查找长度见 [DSA 2 §15–19](../../DSA/DSA_2/)；
- 本系列：二分答案的 `check` 里常套贪心（[AT 21 排序](../AT_21_sorting/)）或计数；"二分 + 哈希"求重复数见 [AT 17](../AT_17_hash_table/)；区间上的二分思维在线段树（[AT 15](../AT_15_segment_tree/)）里升维；二分求 LCP 见字符串哈希（[AT 18](../AT_18_string_hash/)）；
- 既有专题文：[滑动窗口](../sliding_window/)处理"连续段"问题，与二分答案的"划分"问题常可互换（LC 410 两解）。

---

*下一篇：[AT 03：链表](../AT_03_linked_list/)——指针操作全家桶。*
