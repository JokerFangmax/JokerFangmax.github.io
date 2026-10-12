---
title: "LeetCode 74: 搜索二维矩阵"
date: 2026-12-02
math: true
tags:
  - LeetCode
  - 数组
  - 二分查找
  - 矩阵
categories:
  - LeetCode
description: "LeetCode 74 搜索二维矩阵题解（二分查找）"
---

# LeetCode 74. 搜索二维矩阵

> 题目链接：[https://leetcode.cn/problems/search-a-2d-matrix/](https://leetcode.cn/problems/search-a-2d-matrix/)  
> 难度：中等  
> 标签：数组、二分查找、矩阵

---

## 1. 题目描述

给你一个 $m \times n$ 的矩阵 `matrix`，满足两条性质：

1. 每行中的整数**从左到右升序**排列；
2. 每行的第一个整数**大于前一行的最后一个整数**。

也就是说，如果逐行拼接矩阵，得到的是一个严格递增的一维数组。判断目标值 `target` 是否在矩阵中。

**数据范围：**

- $m == matrix.length$
- $n == matrix[i].length$
- $1 \le m, n \le 100$
- $-10^4 \le matrix[i][j], target \le 10^4$

**示例：**

```
示例一：
输入：matrix = [[1,3,5,7],
               [10,11,16,20],
               [23,30,34,60]], target = 3
输出：true

示例二：
输入：matrix = [[1,3,5,7],
               [10,11,16,20],
               [23,30,34,60]], target = 13
输出：false
```

---

## 2. 思路分析

### 2.1 暴力：全表扫描 / 按行线性查找

最直接的做法是把 $mn$ 个元素全扫一遍，$O(mn)$；稍加利用"每行有序"，可以逐行用 `lower_bound` 定位，最坏仍是 $O(m \log n)$。$m, n \le 100$ 时最多一万次操作，随便过。

但这两条性质合起来说的其实是一句话：**矩阵逐行展开后是一个严格递增数组**。既然"有序数组上查找"的 $O(\log n)$ 方案就是二分，题目等于在耳边喊"把我拉直了再二分"。

### 2.2 关键一步：一维下标与二维坐标的互换

设展开后的一维下标为 $k \in [0, mn)$，与矩阵元素 $(i, j)$ 的对应关系是：

$$k = i \cdot n + j \quad \Longleftrightarrow \quad i = k / n, \; j = k \% n$$

直观理解：前面完整的 $i$ 行贡献了 $i \cdot n$ 个元素，$j$ 是当前行内的偏移。**不需要真的展开、也不需要额外空间**——用 $k$ 做二分，访问元素时现算坐标即可。于是问题完整归约为一个 $N = mn$ 个元素的"数组"上的标准二分查找。

### 2.3 一次二分还是两次二分？

另一个自然想法：先二分行（拿每行首元素找 `target` 可能所在的行），再二分该行——两次 $O(\log m + \log n)$。这当然对，但"先确定行"这一步容易写出边界 bug（`target` 恰好等于某行首元素、或落在两行首元素之间时行号的归属）。

拉平后一次二分 $O(\log(mn))$，与两次二分 $O(\log m + \log n)$ 在复杂度上等价（$\log m + \log n = \log(mn)$），但代码只有一层循环，边界更少。**能用一层结构表达的就不要拆成两层**。

### 2.4 以示例走一遍关键步骤

`matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]`，$m = 3, n = 4$，找 $target = 13$：

| 轮次 | left | right | mid($k$) | $(i, j)$ | matrix[i][j] | 比较 | 动作 |
|------|------|-------|----------|----------|--------------|------|------|
| 1 | 0 | 11 | 5 | (1,1) | 11 | $11 < 13$ | `left = 6` |
| 2 | 6 | 11 | 8 | (2,0) | 23 | $23 > 13$ | `right = 7` |
| 3 | 6 | 7 | 6 | (1,2) | 16 | $16 > 13$ | `right = 5` |
| — | 6 | 5 | — | — | — | $left > right$ | 循环结束，未找到，返回 false |

换 $target = 3$ 验证：第 1 轮 `mid = 5` 得 11，`right = 4`；第 2 轮 `mid = 2` 得 $(0,2)$ 的 5，`right = 1`；第 3 轮 `mid = 0` 得 $(0,0)$ 的 1，`left = 1`；第 4 轮 `mid = 1` 得 $(0,1)$ 的 3，命中，返回 true。

---

## 3. 代码实现

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size(), n = matrix[0].size();
        int left = 0, right = m * n - 1;        // 想象中的展开数组 [0, mn-1]
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int val = matrix[mid / n][mid % n]; // 一维下标 -> 二维坐标
            if (val == target) return true;
            if (val < target) left = mid + 1;
            else right = mid - 1;
        }
        return false;
    }
};
```

几个实现细节：

1. **`mid / n` 与 `mid % n` 不可写反**。$n$ 是列数——除以列数得到行号，对列数取余得到行内偏移。写成 `mid / m` 是这题最典型的笔误；
2. **判等可以提前返回**。与 LeetCode 35 求"插入位置"不同，本题只问存在性，命中即可返回 `true`，无需继续收缩边界；
3. **`m * n` 用 `int` 足够**（本题 $mn \le 10^4$），但换成大矩阵时注意 `m * n` 本身可能溢出，需要 `long long`。

---

## 4. 复杂度分析

- **时间复杂度**：$O(\log(mn))$。二分区间长度为 $mn$，每轮减半。由 $\log(mn) = \log m + \log n$，与"两次二分"的 $O(\log m + \log n)$ 渐近相同。
- **空间复杂度**：$O(1)$。没有真的展开矩阵，坐标换算即时完成。

对比：全表扫描 $O(mn)$ → 逐行二分 $O(m \log n)$ → 拉平一次二分 $O(\log(mn))$。三种做法对 $m = n = 100$ 分别约 $10^4$、$100 \times 7 = 700$、$14$ 次操作——数量级的差距一目了然。

---

## 5. 常见疑问

**疑问一：这题和 LeetCode 240（搜索二维矩阵 II）有什么区别？**

74 题的条件强得多：**行与行之间也严格递增**，所以整张表展开后全局有序，能拉平二分。240 题只有"每行从左到右升序、每列从上到下升序"，行与行之间没有大小约束（下一行的元素可以比上一行的小），展开后不是有序数组，**不能用本题的做法**——240 的标准解法是从右上角出发的 $O(m + n)$ 阶梯查找。把两题搞混是面试常见翻车点，务必看清"每行首元素是否大于上一行末元素"这一句。

**疑问二：为什么不用先定位行再在行内二分？**

可以，且复杂度同阶（见 2.3）。但它需要"在首元素列上找最后一个 $\le target$ 的行"，这是一个 `upper_bound` 语义的二分，边界比标准查找更容易写错；拉平做法只有一层标准的闭区间二分，代码更短、不变量更简单。两个正确方案里，选**不变量更少**的那个。

**疑问三：行列互换的矩阵会不会让 `mid / n` 出错？**

不会出错的前提是 $n$ 取的是**每行的长度** `matrix[0].size()`。若输入保证是规则矩阵（本题保证），任何 $mid$ 换算出的 $(i, j)$ 都合法。反过来，如果输入可能含空行或不规则行（本题不会，但工程代码可能遇到），`matrix[0].size()` 与后续行的长度不一致时换算就会越界——工程场景里要先校验规则性。

---

## 6. 相关题目与模板

**可复用模板**（"虚拟展开数组"的二分骨架，适用于一切按行/按块拼接后有序的结构）：

```cpp
// N 为虚拟总长，idx2val 把虚拟下标映射回真实元素
int left = 0, right = N - 1;
while (left <= right) {
    int mid = left + (right - left) / 2;
    int val = idx2val(mid);        // 本题: matrix[mid / n][mid % n]
    if (val == target) return true;
    if (val < target) left = mid + 1;
    else right = mid - 1;
}
```

"虚拟展开"的思想不止用于矩阵：有序链表转跳表、分块数组的全局查找、环形数组的下标平移（`idx % n`），本质都是**设计一个从紧凑下标到真实存储的映射函数**。

**同类题目：**

- [LeetCode 35. 搜索插入位置](https://leetcode.cn/problems/search-insert-position/)：一维版基础，`lower_bound` 语义
- [LeetCode 240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/)：条件弱化的孪生题，右上角阶梯查找，**不可**拉平二分
- [LeetCode 378. 有序矩阵中第 K 小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-sorted-matrix/)：行有序但行间无约束，二分"值域"而非下标，是矩阵二分的进阶形态
- [LeetCode 33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)：另一类"不完整有序"上的二分，训练"先判断哪半边有序"的思维

**一句话总结**：看到"逐行拼接后整体有序"，就把二维数组当成一个用 $k / n$、$k \% n$ 寻址的虚拟一维数组——一次标准二分，$O(\log(mn))$，没有第二层循环。
