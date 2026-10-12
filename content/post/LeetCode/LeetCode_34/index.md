---
title: "LeetCode 34: 在排序数组中查找元素的第一个和最后一个位置"
date: 2026-12-03
math: true
tags:
  - LeetCode
  - 数组
  - 二分查找
categories:
  - LeetCode
description: "LeetCode 34 在排序数组中查找元素的第一个和最后一个位置题解（二分查找）"
---

# LeetCode 34. 在排序数组中查找元素的第一个和最后一个位置

> 题目链接：[https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)  
> 难度：中等  
> 标签：数组、二分查找

---

## 1. 题目描述

给定一个**非递减顺序**（升序、允许重复）排列的整数数组 `nums` 和一个目标值 `target`，找出 `target` 在数组中的**开始位置和结束位置**。若 `target` 不在数组中，返回 `[-1, -1]`。要求时间复杂度 $O(\log n)$。

**数据范围：**

- $0 \le nums.length \le 10^5$
- $-10^9 \le nums[i] \le 10^9$
- `nums` 为非递减数组
- $-10^9 \le target \le 10^9$

**示例：**

```
示例一：
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]        （两个 8 分别在下标 3 和 4）

示例二：
输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]

示例三：
输入：nums = [], target = 0
输出：[-1,-1]
```

---

## 2. 思路分析

### 2.1 暴力：线性扫两端

从左往右找到第一个等于 `target` 的下标，再从右往左找到最后一个——两次 $O(n)$ 扫描。$n = 10^5$ 时约 $10^5$ 次操作，实际能过，但题目明确要求 $O(\log n)$。

更重要的是：暴力同样浪费了"有序"这个前提。当数组是 `[7,7,7,...,7]`（$10^5$ 个 7）而 `target = 7` 时，线性扫描要把整个数组走完才能确定两端——**重复元素越多，线性做法越慢**；而二分恰恰在重复元素多的时候最快（区间成片坍缩）。

### 2.2 一次二分找中间，再左右爬？

一个诱人的折中方案：先二分找到任意一个 `target`，然后向左爬到头、向右爬到头。

**这个做法是 $O(n)$ 的**。反例就是上面的全 7 数组：二分一步命中，接下来左右爬各花 $O(n)$。只要重复段很长，"先二分再爬"就退化成线性——题目出的正是这种数据。

教训：**找到某个 `target` 不等于找到它的边界**。边界信息必须由二分本身的过程给出，不能靠二次扫描补。

### 2.3 正解：把"找边界"写成两段二分

回想 LeetCode 35 的结论："第一个 $\ge target$ 的下标"由 `lower_bound` 求出。本题要的两个位置恰好是两个对称的量：

| 目标 | 定义 | 二分语义 |
|------|------|---------|
| 开始位置 | 第一个等于 $target$ 的下标 | 第一个 $\ge target$ 的下标（`lower_bound`） |
| 结束位置 | 最后一个等于 $target$ 的下标 | 第一个 $> target$ 的下标（`upper_bound`）再减 1 |

**开始位置**：求 $k_1 = $ 第一个 $nums[k] \ge target$。若 $nums[k_1] = target$，$k_1$ 就是开始位置；若 $nums[k_1] > target$（或 $k_1 = n$），说明 `target` 不存在。

**结束位置**：求 $k_2 = $ 第一个 $nums[k] > target$，则结束位置是 $k_2 - 1$（前提同样是存在性检查通过）。

两段二分的差别只有一个字符：判定条件 `nums[mid] < target`（找左边界）与 `nums[mid] <= target`（找右边界）。**等号归哪边，决定了边界往哪边收缩**。

### 2.4 以 `nums = [5,7,7,8,8,10]` 走一遍关键步骤

**找 $target = 8$ 的左边界（`lower_bound` 语义）：**

| 轮次 | left | right | mid | nums[mid] | 判定 | 动作 |
|------|------|-------|-----|-----------|------|------|
| 1 | 0 | 5 | 2 | 7 | $7 < 8$ | `left = 3` |
| 2 | 3 | 5 | 4 | 8 | $8 \not< 8$ | `right = 3` |
| 3 | 3 | 3 | 3 | 8 | $8 \not< 8$ | `right = 2` |
| — | 3 | 2 | — | — | $left > right$ | 返回 $left = 3$ |

**找右边界（`upper_bound` 语义，条件改为 $\le$）：**

| 轮次 | left | right | mid | nums[mid] | 判定 | 动作 |
|------|------|-------|-----|-----------|------|------|
| 1 | 0 | 5 | 2 | 7 | $7 \le 8$ | `left = 3` |
| 2 | 3 | 5 | 4 | 8 | $8 \le 8$ | `left = 5` |
| 3 | 5 | 5 | 5 | 10 | $10 \not\le 8$ | `right = 4` |
| — | 5 | 4 | — | — | $left > right$ | 返回 $left = 5$，结束位置为 $5 - 1 = 4$ |

答案 $[3, 4]$。注意两段二分在同一数据上走出了**不同的路径**——左边界把等号视为"继续向左收缩"，右边界把等号视为"继续向右收缩"，方向由那一个符号决定。

---

## 3. 代码实现

```cpp
class Solution {
    // 第一个满足 nums[k] >= target 的下标（不存在则返回 n）
    int lowerBound(vector<int>& nums, int target) {
        int left = 0, right = (int)nums.size() - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) left = mid + 1;  // 等号不在这边 -> 向左收缩
            else right = mid - 1;
        }
        return left;
    }

    // 第一个满足 nums[k] > target 的下标（不存在则返回 n）
    int upperBound(vector<int>& nums, int target) {
        int left = 0, right = (int)nums.size() - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] <= target) left = mid + 1; // 等号在这边 -> 向右收缩
            else right = mid - 1;
        }
        return left;
    }

public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int lo = lowerBound(nums, target);       // 开始位置候选
        if (lo == (int)nums.size() || nums[lo] != target)
            return {-1, -1};                     // target 不存在，一次二分即可断定
        int hi = upperBound(nums, target) - 1;   // 结束位置 = 上界减一
        return {lo, hi};
    }
};
```

几个实现细节：

1. **存在性检查只做一次**，放在两次边界二分之间：`lower_bound` 的结果不等于 `target` 就直接返回 `[-1,-1]`，省掉第二次二分。而**不必**担心 `upperBound` 的越界——存在性成立时 `target` 至少出现一次，`upper_bound` 结果必然 $\ge lo + 1$，减 1 不越界；
2. **两个函数唯一区别是 `<` 与 `<=`**。把这两个函数并排写在眼前，比任何文字解释都更能说明"等号归属决定收缩方向"；
3. `lo == nums.size()` 必须放在 `nums[lo]` 访问**之前**短路求值，否则 `target` 大于所有元素时越界。

---

## 4. 复杂度分析

- **时间复杂度**：$O(\log n)$。两段独立二分各 $O(\log n)$，加起来仍是 $O(\log n)$（常数因子为 2）。最坏情形（`target` 占满整个数组）也只有约 $2\log_2 n$ 次比较。
- **空间复杂度**：$O(1)$。迭代二分，只用了常数个变量。

对比：线性扫两端 $O(n)$ → "二分命中再爬"最坏 $O(n)$ → 双边界二分 $O(\log n)$。第二种方案最值得警惕：它"看起来"利用了二分，实际最坏复杂度与暴力相同，是被出题数据精准打击的假优化。

---

## 5. 常见疑问

**疑问一：为什么找到任意一个 `target` 后不能向两边爬？**

再强调一次这个陷阱，因为它违反直觉：二分命中只说明 `target` 存在，不说明它在哪里"开始"和"结束"。重复段的长度可以是 $O(n)$，爬的代价也就是 $O(n)$。正确的思路是反过来——**不要先命中再找边界，而是让"找边界"本身成为二分的终点**：二分的循环变量 `left` 从来不指向"某个 target"，而是指向两类元素的"分界线"。分界线思维是二分边界的正解。

**疑问二：右边界为什么不直接二分"最后一个等于 target 的位置"？**

可以那样写（判定 $nums[mid] > target$ 时收缩，否则记录答案并右移），但"记录答案"式的二分与"返回 left"式的二分是两套不变量，混用容易出错。统一成"求第一个 `> target` 的下标再减 1"的好处是：左边界和右边界共享同一套 `left`/`right` 收缩骨架，只有一处条件不同，正确性论证只需要做一遍。

**疑问三：`nums` 为空数组时代码安全吗？**

安全。$n = 0$ 时 `lowerBound` 的循环体不执行（`left = 0 > right = -1`），直接返回 0；外层检查 `lo == nums.size()` 成立，返回 `[-1,-1]`。空数组是本题数据范围里明确包含的用例（示例三），初始化 `right = n - 1 = -1` 的写法对空数组天然免疫——这也是坚持闭区间写法的一个隐性收益。

---

## 6. 相关题目与模板

**可复用模板**（`lower_bound` / `upper_bound` 成对出现，覆盖一切"有序数组 + 重复元素 + 边界"问题）：

```cpp
// 一对孪生二分：唯一区别是 "<" 还是 "<="
int bound(vector<int>& nums, int target, bool upper) {
    int left = 0, right = (int)nums.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target + upper)          // upper=false: >= 左界
            left = mid + 1;                      // upper=true:  >  右界
        else
            right = mid - 1;
    }
    return left;
}
// 开始位置 = bound(nums, t, false)
// 结束位置 = bound(nums, t, true) - 1
// 出现次数 = bound(nums, t, true) - bound(nums, t, false)
```

注意最后一个公式：**统计有序数组中某值的出现次数**就是两个 bound 相减，$O(\log n)$——这是本题最常被复用的衍生技巧。

**同类题目：**

- [LeetCode 35. 搜索插入位置](https://leetcode.cn/problems/search-insert-position/)：单个 `lower_bound` 的入门版，本题的前置知识
- [LeetCode 278. 第一个错误的版本](https://leetcode.cn/problems/first-bad-version/)：抽象谓词上的左边界二分
- [LeetCode 153. 寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)：边界思想的变体——求"断崖"位置
- [LeetCode 162. 寻找峰值](https://leetcode.cn/problems/find-peak-element/)：无序数组上也能二分，只要能确定"答案在 mid 的哪一侧"

**一句话总结**：左右边界 = `lower_bound` 与 `upper_bound - 1`，两段二分只差一个等号的归属；先命中再爬行是伪装成二分的 $O(n)$，分界线思维才是正解。
