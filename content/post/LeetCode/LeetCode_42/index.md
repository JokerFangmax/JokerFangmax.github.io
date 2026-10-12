---
title: "LeetCode 42: 接雨水"
date: 2026-10-09
math: true
tags:
  - LeetCode
  - 数组
  - 双指针
  - 单调栈
  - 动态规划
categories:
  - LeetCode
description: "LeetCode 42 接雨水题解（双指针 + DP + 单调栈）"
---

# LeetCode 42. 接雨水

> 题目链接：[https://leetcode.cn/problems/trapping-rain-water/](https://leetcode.cn/problems/trapping-rain-water/)  
> 难度：困难  
> 标签：数组、双指针、单调栈、动态规划

---

## 1. 题目描述

给定 $n$ 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，**下雨之后能接多少雨水**。

直观图景：柱子像一排墙壁立在地面上，雨水落在缝隙里。某个缝隙能存住水，当且仅当它的**左右两侧都存在比它高的柱子**——水会被左右两边的"高墙"托住，超过较矮那堵墙的部分会从侧面流走。

**数据范围：**

- $n = height.length$
- $1 \le n \le 2 \times 10^4$
- $0 \le height[i] \le 10^5$

**示例：**

```
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6

输入：height = [4,2,0,3,2,5]
输出：9
```

示例 1 的水分布（`#` 为柱子，`~` 为水）：

```
              #
      # ~ ~ ~ # # ~ #
  # ~ # # ~ # # # # # #
0 1 0 2 1 0 1 3 2 1 2 1        ← 共 6 格水
```

---

## 2. 思路分析

### 2.1 暴力：逐列求积水

按列思考是最自然的切入。对第 $i$ 根柱子，它上方的积水高度由**左侧最高墙**与**右侧最高墙**中较矮的一边决定：

$$water_i = \max\bigl(0,\ \min(\text{leftMax}_i, \text{rightMax}_i) - height[i]\bigr)$$

其中 $\text{leftMax}_i = \max_{j \le i} height[j]$（含自身，下同），$\text{rightMax}_i$ 对称定义。暴力做法对每个 $i$ 向两侧各扫一遍求最大值，总时间 $O(n^2)$。$n = 2 \times 10^4$ 时约 $4 \times 10^8$ 次比较，会超时。

瓶颈很明显：$\text{leftMax}_i$、$\text{rightMax}_i$ 被重复计算了 $O(n)$ 遍。

### 2.2 优化一（DP）：预计算两侧最大值

最大值有显然的递推关系：

$$\text{leftMax}_i = \max(\text{leftMax}_{i-1},\ height[i]), \qquad \text{rightMax}_i = \max(\text{rightMax}_{i+1},\ height[i])$$

先正扫一遍求出全部 $\text{leftMax}$，再倒扫一遍求 $\text{rightMax}$，最后 $O(n)$ 汇总。时间 $O(n)$，空间 $O(n)$——这是典型的**动态规划 (dynamic programming)** 预处理思想：用递推消掉重复计算。

### 2.3 优化二（双指针）：发现"不必知道两边的精确值"

DP 版已经很快，但有个更深的观察能把空间也压到 $O(1)$。

关键引理：**接水量只取决于左右最大值中较小的一边，而较小的一边何时确定是可以即时判断的。** 用双指针 $l$、$r$ 从两端相向移动，同时维护 `leftMax`（$[0..l]$ 的最大值）和 `rightMax`（$[r..n-1]$ 的最大值）：

- 若 `leftMax < rightMax`：第 $l$ 根柱子的积水**此刻已可结算**——它的真实 $\text{leftMax}_l$ 就是当前的 `leftMax`（左边扫完了），而它右侧至少存在一堵 `rightMax` 高的墙 $> \text{leftMax}_l$，所以 $\min$ 一侧由左边锁定，与右侧的**精确**最大值无关。结算后 $l$ 右移；
- 反之结算 $r$ 侧，$r$ 左移。

每次结算的都是"短板一侧已完全确定"的那根柱子，两边交替推进，谁的最大值已定型就算谁。这就是**对撞双指针 (two pointers)** 的又一经典应用。

**常见的心结**：结算 $l$ 时用的 `rightMax` 可能比真实的 $\text{rightMax}_l$ 小（右边还没扫完），会不会算少？不会——因为 `rightMax > leftMax` 已经保证 $\min$ 取的是左边，右边再大也改变不了 $\min$。**只需要"够大"的证据，不需要精确值**，这是本解法最值得回味的一步。

### 2.4 优化三（单调栈）：换一种切法——按层横着算

前面的解法都是**竖着**逐列算，单调栈给出**横着**逐层算的视角：水是一层一层接的，每一层都是一个"被两堵墙夹住的凹槽"。

单调栈 (monotonic stack) 维护一个**高度递减**的柱子下标栈。遍历到 $height[i]$ 时：

1. 只要栈顶柱子比 $height[i]$ 矮，说明出现了一道右墙——栈顶与 $height[i]$ 之间可能形成凹槽；
2. 弹出栈顶作为凹槽**底**，则左墙是新的栈顶，右墙是 $height[i]$；
3. 凹槽水量 = $\bigl(\min(\text{左墙}, \text{右墙}) - \text{底}\bigr) \times \text{宽度}$，累加；
4. 重复直到栈顶不低于 $height[i]$，把 $i$ 入栈。

每根柱子至多入栈、出栈各一次，$O(n)$。与 LC 739 每日温度"找下一个更大元素"是同一个骨架，本题只是把"找到更大元素后干的事"从"算距离"换成了"结算凹槽"。

### 2.5 三种解法逐列演示

`height = [4,2,0,3,2,5]`，用 DP 法逐列列表（答案应为 9）：

| $i$ | $h_i$ | leftMax | rightMax | $\min$ | 积水 |
|-----|-------|---------|----------|--------|------|
| 0   | 4     | 4       | 5        | 4      | 0    |
| 1   | 2     | 4       | 5        | 4      | **2** |
| 2   | 0     | 4       | 5        | 4      | **4** |
| 3   | 3     | 4       | 5        | 4      | **1** |
| 4   | 2     | 4       | 5        | 4      | **2** |
| 5   | 5     | 5       | 5        | 5      | 0    |

合计 $2 + 4 + 1 + 2 = 9$。

双指针法走同一例子（初始 `leftMax = rightMax = 0`，$l = 0, r = 5$）：全程 `height[l] < height[5] = 5` 成立，每轮都结算**左端**——$l = 0$ 时 `leftMax` 升到 4、自身积水 0；随后 $l = 1, 2, 3, 4$ 依次积水 $2, 4, 1, 2$，$l$ 到 5 时循环结束。有意思的是 `rightMax` 全程没被用到：右端那堵 5 高的墙只需"存在且高于左端"这一证据，就足以锁定每根左柱的 $\min$，右侧的精确最大值无关紧要——这正是 2.3 引理的直接体现。

---

## 3. 代码实现

```cpp
// 主推: 双指针版（两个 class Solution 二选一提交）
class Solution {
public:
    int trap(vector<int>& height) {
        int n = height.size();
        int l = 0, r = n - 1;
        int leftMax = 0, rightMax = 0, ans = 0;
        while (l < r) {
            if (height[l] < height[r]) {
                // 左边是短板: 左侧最大值已定型，min 一定取 leftMax
                leftMax = max(leftMax, height[l]);
                ans += leftMax - height[l];
                l++;
            } else {
                rightMax = max(rightMax, height[r]);
                ans += rightMax - height[r];
                r--;
            }
        }
        return ans;
    }
};

// 备选: DP 预处理版（更好想、更好写对）
// class Solution {
// public:
//     int trap(vector<int>& height) {
//         int n = height.size();
//         if (n == 0) return 0;
//         vector<int> leftMax(n), rightMax(n);
//         leftMax[0] = height[0];
//         for (int i = 1; i < n; i++)               // 正扫: 左侧最大值
//             leftMax[i] = max(leftMax[i - 1], height[i]);
//         rightMax[n - 1] = height[n - 1];
//         for (int i = n - 2; i >= 0; i--)          // 倒扫: 右侧最大值
//             rightMax[i] = max(rightMax[i + 1], height[i]);
//         int ans = 0;
//         for (int i = 0; i < n; i++)
//             ans += min(leftMax[i], rightMax[i]) - height[i];
//         return ans;
//     }
// };
```

几个实现细节：

1. **双指针版比较的是 `height[l]` 与 `height[r]` 本身**而不是 `leftMax` 与 `rightMax`——两种写法都能对，但比较当前柱高更简洁：`height[l] < height[r]` 蕴含右侧至少有一堵墙高于左端，左侧短板地位成立。
2. **先更新最大值、再算积水**（`leftMax = max(leftMax, height[l])` 在 `ans +=` 之前）。顺序反了会把当前柱子自己的高度漏算进墙里，少算积水。
3. DP 版用 `min(leftMax, rightMax) - height[i]` 时**不需要 $\max(0, \cdot)$ 兜底**：两侧最大值都定义成**含自身**，$\min \ge height[i]$ 恒成立。若定义成不含自身就必须兜底——两种定义别混用。

---

## 4. 复杂度分析

| 解法 | 时间 | 空间 | 思路切法 |
|------|------|------|----------|
| 暴力逐列 | $O(n^2)$ | $O(1)$ | 每列现算两侧最大值 |
| DP 预处理 | $O(n)$ | $O(n)$ | 竖切：预计算 $\text{leftMax}/\text{rightMax}$ |
| **双指针** | $O(n)$ | $O(1)$ | 竖切：短板一侧即时结算 |
| 单调栈 | $O(n)$ | $O(n)$ | 横切：逐层结算凹槽 |

除暴力外，三个解法的时间都是 $O(n)$，差别在空间和思维成本。双指针版是"时间、空间、代码量"三者最优，但正确性论证最精巧；面试中先写 DP 版讲清 $water_i$ 公式，再引申到双指针，是最稳的路线。

---

## 5. 常见疑问

**疑问一：为什么接水公式里 `min(leftMax, rightMax)` 要"含自身"？** 含自身的定义让两件事同时成立：$\min \ge height[i]$（无需兜底负数），且递推式连续。若不含自身，柱子自身是最左/最右高墙时（如 `[5,0,5]` 的中间柱，左右各看 5）公式照样对，但边界列会出现"左侧无墙"的未定义状态，徒增特判。**"墙可以就是自己"是这一族公式最省心的约定**。

**疑问二：单调栈版弹出凹槽时，为什么宽度是 `i - st.top() - 1`？** 弹出的栈顶是凹槽底，新栈顶是左墙，当前 $i$ 是右墙——底已经占掉一格，所以左右墙之间的宽度是 $i - \text{st.top()} - 1$（减 1 去掉底本身）。这个**差一 (off-by-one)** 是单调栈类题最常错的地方；稳妥的办法是画一个 `[3,0,3]` 手推：弹 0 后左墙下标 0、右墙下标 2，宽度 $2 - 0 - 1 = 1$，高 $\min(3,3) - 0 = 3$，水 $3$。

**疑问三：与 LC 11 盛最多水的容器到底什么关系？** 一句话：LC 11 是"两堵墙选最优"，LC 42 是"每根柱各算各的"。LC 11 的短板是**两条所选线**的较小值，LC 42 的短板是**两侧前缀最大值**的较小值；LC 11 双指针靠"排除必败端点"收缩，LC 42 双指针靠"短板一侧信息已完备"结算。两题的 $\min$ 都出现，但语义完全不同，面试里混着讲是最常见的翻车点。

---

## 6. 相关题目与模板

> 📚 延伸阅读：[单调队列模板笔记](../../algorithms/monotonic_queue/)（单调栈的孪生结构，一个管"下一个更大元素"，一个管"滑动窗口最值"）。

**可复用模板**（"两侧最大值"类问题的 DP 预处理骨架）：

```cpp
vector<int> L(n), R(n);
L[0] = a[0];      for (int i = 1;     i < n; i++) L[i] = max(L[i-1], a[i]);
R[n-1] = a[n-1];  for (int i = n - 2; i >= 0; i--) R[i] = max(R[i+1], a[i]);
// 逐位置用 min(L[i], R[i]) 结算
```

这个"正扫一遍、倒扫一遍"的骨架适用于一切"每个位置依赖两侧前缀信息"的问题（本系列 [LeetCode 239](../LeetCode_239/) 的滑动窗口最值是它去掉一维后的近亲）。

**同类题目：**

- [LeetCode 11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)：对撞双指针的入门名题，与本题对照见 5.3
- [LeetCode 84. 柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/)：单调栈正主——本题单调栈维护递减栈找"右墙"，84 题维护递增栈找"两边更矮"，一套思想两个方向
- [LeetCode 739. 每日温度](https://leetcode.cn/problems/daily-temperatures/)：单调栈求"下一个更大元素"的最简版
- [LeetCode 407. 接雨水 II](https://leetcode.cn/problems/trapping-rain-water-ii/)：二维版，最小堆从外圈向内灌水，"短板由包围圈最小值决定"的思想与本题一脉相承

**一句话总结**：接雨水把一个物理直觉（水被两侧较矮的墙托住）提炼成 $\min(\text{leftMax}, \text{rightMax}) - h$ 这一列级公式，剩下的问题只是**如何廉价地拿到两侧最大值**——DP 预计算、双指针即时判断、单调栈换成横切视角，殊途同归 $O(n)$。

---
