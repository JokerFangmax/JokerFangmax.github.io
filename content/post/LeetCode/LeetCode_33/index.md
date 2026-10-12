---
title: "LeetCode 33: 搜索旋转排序数组"
date: 2026-12-04
math: true
tags:
  - LeetCode
  - 数组
  - 二分查找
categories:
  - LeetCode
description: "LeetCode 33 搜索旋转排序数组题解（二分查找）"
---

# LeetCode 33. 搜索旋转排序数组

> 题目链接：[https://leetcode.cn/problems/search-in-rotated-sorted-array/](https://leetcode.cn/problems/search-in-rotated-sorted-array/)  
> 难度：中等  
> 标签：数组、二分查找

---

## 1. 题目描述

一个原本**升序且元素互不相同**的数组，在某个未知下标处被"旋转"了。例如 `[0,1,2,4,5,6,7]` 旋转后可能变成 `[4,5,6,7,0,1,2]`。给定旋转后的数组 `nums` 和目标值 `target`，返回 `target` 的下标；不存在返回 `-1`。要求算法时间复杂度为 $O(\log n)$。

**数据范围：**

- $1 \le nums.length \le 5000$
- $-10^4 \le nums[i] \le 10^4$
- `nums` 中的每个值**互不相同**
- `nums` 必定是某个升序数组旋转后的结果

**示例：**

```
示例一：
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4

示例二：
输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1

示例三：
输入：nums = [1], target = 0
输出：-1
```

---

## 2. 思路分析

### 2.1 暴力：线性扫描

一遍 `for` 循环找 `target`，$O(n)$。$n \le 5000$ 轻松通过，但题目再次点名要 $O(\log n)$——暴力也再次无视了题目的结构信息：这个数组虽然整体不是有序的，却**几乎**是有序的（两段各自有序，前一段整体大于后一段）。

"几乎有序"能不能二分？这是本题真正要回答的问题。

### 2.2 二分为什么还能用：一半必然有序

二分的可行性依赖于"比较一次、淘汰一半"。标准有序数组里，`nums[mid]` 与 `target` 的一次比较就能定方向；旋转数组做不到这一点——`target` 可能在 `mid` 左边也可能在右边。

但把 `nums[mid]` 改为和**端点**比较，就能得到一个确定性的结论：

> 无论 `mid` 落在哪里，`[left, mid]` 和 `[mid, right]` **至少有一半是完全有序的**。

原因：旋转点（原数组首元素所在处，也是"断崖"）只有一个，它不可能同时出现在两半里。断崖在哪一半，那一半无序，另一半必然有序。而"哪一半有序"只需要一次比较就能判断：

- 若 $nums[left] \le nums[mid]$：断崖不在 $[left, mid]$ 内（或 $left = mid$），左半**升序**；
- 否则：断崖在左半，右半**升序**。

### 2.3 有序的那一半怎么用

定位到有序的一半后，检查 `target` 是否落在这一半的**值域**内：

- 在，则把搜索区间收缩到这一半——回到标准二分；
- 不在，则 `target` 只可能在另一半——同样收缩区间。

无论哪种情况，区间都减半，$O(\log n)$ 成立。**完整的判定链是三步**：判断哪半有序 → 判断 `target` 是否在有序半边的值域内 → 收缩到正确的半边。

### 2.4 以 `nums = [4,5,6,7,0,1,2]` 走一遍关键步骤

**找 $target = 0$：**

| 轮次 | left | right | mid | nums[mid] | 哪半有序 | target 落在哪 | 动作 |
|------|------|-------|-----|-----------|---------|-------------|------|
| 1 | 0 | 6 | 3 | 7 | $4 \le 7$，左半 $[4,5,6,7]$ 有序 | $0 \notin [4,7]$ | 去右半，`left = 4` |
| 2 | 4 | 6 | 5 | 1 | $0 \le 1$，左半 $[0,1]$ 有序 | $0 \in [0,1]$ | 去左半，`right = 4` |
| 3 | 4 | 4 | 4 | 0 | 命中 | — | 返回 4 |

**找 $target = 3$（不存在）：**

| 轮次 | left | right | mid | nums[mid] | 哪半有序 | target 落在哪 | 动作 |
|------|------|-------|-----|-----------|---------|-------------|------|
| 1 | 0 | 6 | 3 | 7 | 左半 $[4,7]$ 有序 | $3 \notin [4,7]$ | `left = 4` |
| 2 | 4 | 6 | 5 | 1 | 左半 $[0,1]$ 有序 | $3 \notin [0,1]$ | `left = 6` |
| 3 | 6 | 6 | 6 | 2 | 命中检查失败 | — | `right = 5` |
| — | 6 | 5 | — | — | $left > right$ | — | 返回 -1 |

---

## 3. 代码实现

```cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = (int)nums.size() - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;

            if (nums[left] <= nums[mid]) {
                // 左半 [left, mid] 有序（<= 兼容 left == mid 的单元素情形）
                if (nums[left] <= target && target < nums[mid])
                    right = mid - 1;          // target 在左半的值域内
                else
                    left = mid + 1;           // 否则只可能在右半
            } else {
                // 右半 [mid, right] 有序
                if (nums[mid] < target && target <= nums[right])
                    left = mid + 1;           // target 在右半的值域内
                else
                    right = mid - 1;          // 否则只可能在左半
            }
        }
        return -1;
    }
};
```

几个实现细节：

1. **`nums[left] <= nums[mid]` 的等号不可省**。当区间收缩到两个元素时 $mid = left$（下取整），`nums[left]` 与 `nums[mid]` 是同一个元素，比较结果自然相等——去掉等号会把它误判成"右半有序"，接着就可能在错误的半边里收缩。这一行是全题最容易写错的地方；
2. **值域判断的端点开闭要自洽**。左半有序时，`mid` 已经判过等（不相等），所以值域写成 $[nums[left], nums[mid])$ 右开；同理右半写成 $(nums[mid], nums[right]]$ 左开。用闭区间两端判断也可以，但必须与"mid 已排除"的事实对齐，否则可能死循环；
3. **先判 `nums[mid] == target`**。命中即返回，后续的区间讨论就不必考虑 `target` 恰好等于 `nums[mid]` 的分支，值域判断得以放心使用开区间端点。

---

## 4. 复杂度分析

- **时间复杂度**：$O(\log n)$。每轮循环区间严格减半：先做一次等值判断，再做一次"哪半有序"判断和一次值域判断——三次比较都是 $O(1)$，区间长度从 $n$ 到 0 至多 $\lceil \log_2 n \rceil + 1$ 轮。
- **空间复杂度**：$O(1)$。迭代二分，常数变量。

对比：线性扫描 $O(n)$ → 本解法 $O(\log n)$。另有一条"两次二分"的路线：先二分找旋转点（断崖），再在正确的有序段里做标准二分，同为 $O(\log n)$ 但代码分两段；本题的单循环写法把两件事合在了一层判定链里，更紧凑。

---

## 5. 常见疑问

**疑问一：旋转数组不是无序的吗，凭什么还能二分？**

二分的前提不是"整体有序"，而是"**每次比较后能确定淘汰哪一半**"。本题的每次循环里，"哪半有序"和"`target` 是否在有序半边值域内"这两个问题都能用 $O(1)$ 次比较确定地回答，于是每一半都可以被**证明**排除或保留——这个确定性才是二分的灵魂。理解了这一点，LeetCode 162（找峰值，完全无序也能二分）就不神秘了。

**疑问二：`nums[left] <= nums[mid]` 写成 `<` 会挂在什么数据上？**

经典反例：`nums = [3, 1], target = 1`。第 1 轮 $left = 0, right = 1, mid = 0$，`nums[0] == nums[mid] == 3`。用 `<` 判断会走进 else 分支（误以为右半有序），值域判断 $nums[mid] < target \le nums[right]$ 即 $3 < 1 \le 1$ 不成立，于是 `right = mid - 1 = -1`，循环结束返回 -1——**答案是 1 却漏掉了**。这个 bug 的隐蔽之处在于大多数测试数据下行为完全正常，只有区间缩到两个元素、且 `mid == left` 时才暴露。

**疑问三：元素互不相同这个条件重要吗？**

非常重要。若允许重复（LeetCode 81 的设定），`nums[left] == nums[mid] == nums[right]` 时无法判断哪半有序——比如 `[1,0,1,1,1]`，必须额外处理（两边端点都缩一格），最坏退化到 $O(n)$。本题数据范围保证互不相同，`nums[left] <= nums[mid]` 才等价于"左半必然有序"。**读题时留意"互不相同"这类条件，它们往往是解法成立的隐含支点。**

---

## 6. 相关题目与模板

**可复用模板**（"部分有序"上的二分骨架：判有序 → 判值域 → 收缩）：

```cpp
// 适用于：区间 [left, right] 中至少一半有序的结构
while (left <= right) {
    int mid = left + (right - left) / 2;
    if (nums[mid] == target) return mid;
    if (nums[left] <= nums[mid]) {                 // 1. 判断哪半有序
        if (nums[left] <= target && target < nums[mid])
            right = mid - 1;                       // 2. 在有序半边值域内 -> 收缩到它
        else
            left = mid + 1;                        //    否则收缩到另一半
    } else {
        if (nums[mid] < target && target <= nums[right])
            left = mid + 1;
        else
            right = mid - 1;
    }
}
```

方法论总结：**面对"不完全有序"的数组，先问"切一刀之后哪一半是有序的"**。只要能廉价地识别有序的一半并用值域排除它，二分就依然成立——旋转数组只是这个框架最经典的实例。

**同类题目：**

- [LeetCode 81. 搜索旋转排序数组 II](https://leetcode.cn/problems/search-rotated-sorted-array-ii/)：允许重复的版本，体会"互不相同"条件的分量
- [LeetCode 153. 寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)：只找断崖，`nums[mid]` 与 `nums[right]` 一次比较即可
- [LeetCode 154. 寻找旋转排序数组中的最小值 II](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array-ii/)：153 + 重复元素，最坏 $O(n)$ 的根源分析
- [LeetCode 35. 搜索插入位置](https://leetcode.cn/problems/search-insert-position/) 与 [LeetCode 34. 查找首尾位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)：完全有序数组的标准二分，本系列的前置篇
- [LeetCode 162. 寻找峰值](https://leetcode.cn/problems/find-peak-element/)：彻底无序数组上的二分，"确定淘汰一半"思维的极限形态

**一句话总结**：旋转数组二分的心法是"**断崖只有一处，切口两边必有一边有序**"——识别有序的那一半，用值域判断 `target` 在不在里面，三步判定链让 $O(\log n)$ 在残缺的有序性上依然成立。
