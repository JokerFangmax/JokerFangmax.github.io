---
title: "LeetCode 35: 搜索插入位置"
date: 2026-12-01
math: true
tags:
  - LeetCode
  - 数组
  - 二分查找
categories:
  - LeetCode
description: "LeetCode 35 搜索插入位置题解（二分查找）"
---

# LeetCode 35. 搜索插入位置

> 题目链接：[https://leetcode.cn/problems/search-insert-position/](https://leetcode.cn/problems/search-insert-position/)  
> 难度：简单  
> 标签：数组、二分查找

---

## 1. 题目描述

给定一个**升序**排列的整数数组 `nums`（无重复元素）和一个目标值 `target`，如果 `target` 在数组中，返回它的下标；如果不在，返回它**按顺序插入后应处的位置**。要求算法时间复杂度为 $O(\log n)$。

**数据范围：**

- $1 \le nums.length \le 10^4$
- $-10^4 \le nums[i] \le 10^4$
- `nums` 为**无重复元素**的升序数组
- $-10^4 \le target \le 10^4$

**示例：**

```
示例一：
输入：nums = [1,3,5,6], target = 5
输出：2

示例二：
输入：nums = [1,3,5,6], target = 2
输出：1        （2 插在 1 和 3 之间）

示例三：
输入：nums = [1,3,5,6], target = 7
输出：4        （比所有元素都大，插到末尾）
```

---

## 2. 思路分析

### 2.1 暴力：从头线性扫描

从左到右找到第一个 `nums[i] >= target` 的位置，返回 $i$；扫到末尾都没找到就返回 $n$。时间 $O(n)$，代码只有三行。

但题目**明确要求** $O(\log n)$，而且暴力完全没有用到"有序 + 无重复"这个前提——有序性才是这道题真正递给我们的武器。$10^4$ 的规模线性扫也能过，可如果把数据放大到 $10^7$（比如真实的词典查询、数据库索引定位），线性扫描就不可接受了。

### 2.2 二分查找：每次砍掉一半

**二分查找 (binary search)** 的核心逻辑：每次取中点，与 `target` 比较，就能确定目标在左半还是右半——**一次比较淘汰一半候选**，$O(\log n)$ 次后只剩一个位置。

关键在于把"在不在数组里"和"该插在哪"**统一成一个判定**。观察"插入位置"的本质：

> 插入位置是**第一个大于等于 $target$ 的元素下标**（若不存在这样的元素，则是 $n$）。

- `target` 在数组中时，这个位置就是它的下标（无重复，所以唯一）；
- `target` 不在数组中时，这个位置恰好是它应该插入的地方。

数学化地说，我们在求最小的 $k$ 使得 $nums[k] \ge target$——这正是 **`lower_bound`** 的定义。两种情况被同一个定义覆盖，所以不需要分开处理。

### 2.3 以 `nums = [1,3,5,6]` 走一遍关键步骤

**找 $target = 2$：**

| 轮次 | left | right | mid | nums[mid] | 与 target 比较 | 动作 |
|------|------|-------|-----|-----------|---------------|------|
| 1 | 0 | 3 | 1 | 3 | $3 > 2$ | 目标在左半，`right = mid - 1 = 0` |
| 2 | 0 | 0 | 0 | 1 | $1 < 2$ | 目标在右半，`left = mid + 1 = 1` |
| — | 1 | 0 | — | — | $left > right$ | 循环结束，返回 $left = 1$ |

**找 $target = 7$：**

| 轮次 | left | right | mid | nums[mid] | 与 target 比较 | 动作 |
|------|------|-------|-----|-----------|---------------|------|
| 1 | 0 | 3 | 1 | 3 | $3 < 7$ | `left = 2` |
| 2 | 2 | 3 | 2 | 5 | $5 < 7$ | `left = 3` |
| 3 | 3 | 3 | 3 | 6 | $6 < 7$ | `left = 4` |
| — | 4 | 3 | — | — | $left > right$ | 循环结束，返回 $left = 4$ |

注意返回值统一是 `left`：循环不变量是"`[0, left)` 内全部 $< target$，`(right, n-1]$ 内全部 $\ge target$"，结束时 $left = right + 1$，`left` 正好卡在两段的分界线上。

---

## 3. 代码实现

```cpp
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int left = 0, right = (int)nums.size() - 1;   // 闭区间 [left, right]
        while (left <= right) {
            int mid = left + (right - left) / 2;      // 防溢出写法
            if (nums[mid] < target) {
                left = mid + 1;    // mid 及其左边都小于 target，分界在右侧
            } else {
                right = mid - 1;   // nums[mid] >= target，mid 可能就是答案，但先收缩
            }
        }
        return left;               // 第一个 >= target 的下标（或 n）
    }
};
```

几个实现细节：

1. **`left + (right - left) / 2` 而不是 `(left + right) / 2`**：本题 $n \le 10^4$ 不会溢出，但这是二分题的肌肉记忆——数据范围一大，`left + right` 可能超过 `int` 上界；
2. **`else` 分支含等号**：`nums[mid] == target` 时也执行 `right = mid - 1`，因为无重复元素，最终 `left` 仍会停在这个值上——把 `>=` 归入"往左收缩"正是 `lower_bound` 语义的体现；
3. **返回 `left` 而不是 `right`**：循环结束时 $left = right + 1$，`right` 停在最后一个小于 $target$ 的位置上，`left` 才是插入点。

---

## 4. 复杂度分析

- **时间复杂度**：$O(\log n)$。每轮循环区间长度至少减半，区间从 $n$ 缩到 0 至多需要 $\lceil \log_2 n \rceil + 1$ 轮。
- **空间复杂度**：$O(1)$。只用了三个整型变量，迭代写法无递归栈。

对比：线性扫描 $O(n)$ → 二分查找 $O(\log n)$。$n = 10^4$ 时大约 14 次比较就能定位；$n = 10^9$ 也只要 30 次——这正是题目强调 $O(\log n)$ 的原因：对数级别的算法对规模几乎免疫。

---

## 5. 常见疑问

**疑问一：为什么 `while (left <= right)` 用 `<=` 而不是 `<`？**

因为我们维护的是**闭区间** $[left, right]$。当 $left = right$ 时区间里还有一个元素没被检查，用 `<` 会在漏查的情况下退出。如果改成左闭右开区间 $[left, right)$，条件相应变成 `while (left < right)`——两种写法都对，但**区间定义必须与循环条件、边界更新自洽**，混着写是二分 90% 的错误来源。

**疑问二：`target` 比所有元素都小或都大时会不会越界？**

不会。比所有元素都小时：第一轮 `nums[mid] >= target`，区间一路收缩到空，`left` 停在 0——插入头部，正确。比所有元素都大时（见 2.3 的 $target = 7$ 演示），`left` 一路右移到 $n$——插入末尾，正确。`left` 的取值范围始终在 $[0, n]$ 内，不需要任何特判。

**疑问三：为什么说这题是 `lower_bound`？**

`std::lower_bound(nums.begin(), nums.end(), target) - nums.begin()` 一行就是答案。理解这层对应关系的价值在于：**"找插入位置"和"找第一个满足条件的下标"是同一件事**。把二分写成"判定 + 收缩"的形式，比死记"查找某个值"的模板通用得多——大量二分变体（第一个/最后一个/右边界）都从这个统一视角推导出来。

---

## 6. 相关题目与模板

**可复用模板**（`lower_bound` 骨架，找第一个 $\ge target$ 的下标）：

```cpp
// 返回第一个满足 nums[k] >= target 的 k（不存在则返回 n）
int lowerBound(vector<int>& nums, int target) {
    int left = 0, right = (int)nums.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return left;
}
```

把 `<` 换成 `<=`（即 `nums[mid] <= target` 时右移），得到的是**最后一个 $\le target$ 的位置再右移一格**，即 `upper_bound` 的等价形式——首尾位置类问题（LeetCode 34）靠这两个方向的组合解决。

**同类题目：**

- [LeetCode 34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)：本题的双倍版，左右边界各二分一次
- [LeetCode 74. 搜索二维矩阵](https://leetcode.cn/problems/search-a-2d-matrix/)：把二维摊平成一维后就是本题
- [LeetCode 278. 第一个错误的版本](https://leetcode.cn/problems/first-bad-version/)：单调判定函数上的 `lower_bound`，抽象层次更高一级
- [LeetCode 704. 二分查找](https://leetcode.cn/problems/binary-search/)：最裸的二分，建议和本题一起写掉

**一句话总结**：搜索插入位置的本质是 `lower_bound`——"在不在"和"插在哪"统一为"找第一个 $\ge target$ 的下标"，一个不变量（左段全小于、右段全不小于）走天下。
