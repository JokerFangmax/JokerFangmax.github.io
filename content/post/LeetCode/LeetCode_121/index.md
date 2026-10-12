---
title: "LeetCode 121: 买卖股票的最佳时机"
date: 2026-12-15
math: true
tags:
  - LeetCode
  - 数组
  - 动态规划
categories:
  - LeetCode
description: "LeetCode 121 买卖股票的最佳时机题解（一次遍历维护前缀最小值）"
---

# LeetCode 121. 买卖股票的最佳时机

> 题目链接：[https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)  
> 难度：简单  
> 标签：数组、动态规划

---

## 1. 题目描述

给定一个数组 `prices`，第 `i` 个元素 `prices[i]` 表示某支股票在第 `i` 天的价格。你只能选择**某一天买入**这只股票，并选择**在未来的某一个不同的日子卖出**。设计算法计算所能获得的最大利润；如果不能获利（价格单调递减），返回 `0`。

**数据范围：**

- $1 \le prices.length \le 10^5$
- $0 \le prices[i] \le 10^4$

**示例：**

```
输入：prices = [7,1,5,3,6,4]
输出：5
解释：在第 2 天（价格 = 1）买入，在第 5 天（价格 = 6）卖出，最大利润 = 6 - 1 = 5。
     注意利润不能是 7 - 1 = 6，因为 7 出现在 1 之前，必须先买后卖。

输入：prices = [7,6,4,3,1]
输出：0
解释：价格单调递减，任何交易都会亏损，不交易是最优的。
```

---

## 2. 思路分析

### 2.1 暴力：枚举每一对（买入日, 卖出日）

最直接的做法：枚举买入日 $i$、卖出日 $j$（$i < j$），取 `prices[j] - prices[i]` 的最大值，负数则取 $0$。两层循环共 $\binom{n}{2}$ 对，时间 $O(n^2)$。$n = 10^5$ 时约 $5 \times 10^9$ 次运算，必然超时。

问题出在**同一份信息被反复计算**：对每个卖出日 $j$，最优买入日一定是它前面价格最低的那天，而暴力把所有 $j$ 前面的日子都重新比了一遍。

### 2.2 换个视角：站在卖出日这一侧想

把问题改写成等价形式：

$$\text{ans} = \max_{0 \le j < n}\left(prices[j] - \min_{0 \le i < j} prices[i]\right)$$

也就是**枚举卖出日 $j$**，买入价取前缀最小值 $\min(prices[0..j-1])$。前缀最小值可以从左往右一遍扫出来，于是两层循环坍缩成一层：

- 维护变量 `minPrice`：目前见过的最低价；
- 维护变量 `ans`：目前最大的 `prices[i] - minPrice`。

每到一个新位置，先更新 `ans`（用当前价减去历史最低价），再更新 `minPrice`。顺序保证了"卖出价减去的买入价一定来自更早的某天"，不会出现当天买当天卖、或未来价回填的越界。

### 2.3 用动态规划 (dynamic programming) 的语言再说一遍

定义 $dp[i]$ 为**前 $i + 1$ 天内完成一笔交易能获得的最大利润**，则有两种来源：

1. 卖出发生在第 $i$ 天：利润为 $prices[i] - \min_{k < i} prices[k]$；
2. 卖出发生在更早：利润为 $dp[i-1]$。

$$dp[i] = \max\left(dp[i-1],\; prices[i] - \text{preMin}(i-1)\right), \quad dp[0] = 0$$

由于 $dp[i]$ 只依赖 $dp[i-1]$ 和滚动更新的前缀最小值，两个变量就够了——这正是 2.2 的写法。**这道题也是"股票系列"的状态机入门**：只允许一笔交易，状态机只有"未持有 / 持有"两个状态，复杂版本（多笔、含冷冻期、含手续费）都是在此状态机上扩展的。

### 2.4 手动模拟（以示例 1 走一遍）

`prices = [7,1,5,3,6,4]`：

| 天 $i$ | 价格 | 先更新：$prices[i] - \text{minPrice}$ | $\text{ans}$ | 再更新：$\text{minPrice}$ |
|---|---|---|---|---|
| 0 | 7 | —（首日无买入机会） | 0 | 7 |
| 1 | 1 | $1 - 7 = -6$ | 0 | 1 |
| 2 | 5 | $5 - 1 = 4$ | 4 | 1 |
| 3 | 3 | $3 - 1 = 2$ | 4 | 1 |
| 4 | 6 | $6 - 1 = 5$ | **5** | 1 |
| 5 | 4 | $4 - 1 = 3$ | 5 | 1 |

最终 `ans = 5`，即第 2 天买入、第 5 天卖出。

---

## 3. 代码实现

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        int minPrice = prices[0];   // 前缀最低价：最佳买入候选
        int ans = 0;                // 默认不交易，利润 0
        for (int i = 1; i < n; i++) {
            // 先用"当前价 - 历史最低价"更新答案，保证买入日在卖出日之前
            ans = max(ans, prices[i] - minPrice);
            // 再更新历史最低价（当天买当天卖不允许，所以后更新）
            minPrice = min(minPrice, prices[i]);
        }
        return ans;
    }
};
```

两个实现细节：

1. **先更新 `ans`、再更新 `minPrice`**。若顺序颠倒，当某天是历史最低时会先算出 `prices[i] - prices[i] = 0`——本例中虽不影响正确性（结果恰为 0），但语义上允许了"当天买当天卖"，遇到要求严格前后关系的变体就是 bug。固定"先算利润、后更新最小"的顺序最稳妥。
2. **`minPrice` 初始化为 `prices[0]`，循环从 `i = 1` 开始**。首日没有更早的买入机会，天然不参与利润计算，省掉一次特判。

---

## 4. 复杂度分析

- **时间复杂度**：$O(n)$。单层循环，每个位置只做常数次比较与减法。
- **空间复杂度**：$O(1)$。只用 `minPrice`、`ans` 两个变量（不计输入）。

对比：暴力 $O(n^2)$ → 前缀最小值一次遍历 $O(n)$。由于每个价格至少要看一眼才能确定答案，$O(n)$ 已是理论最优。

---

## 5. 常见疑问

**Q1：能不能"找到最小值再找它后面最大的值"？**

不行。最优解中的买入日**不一定是全局最低价**。反例：`prices = [2, 10, 1, 3]`。全局最低价 $1$ 在末尾附近，它后面只能赚 $3 - 1 = 2$；而最优解是 $2$ 买入、$10$ 卖出，赚 $8$。正确视角是"枚举卖出日，配它前面的最低价"，而不是孤立地找两个极值。同理，"先降后升的 V 形谷底买"也只是局部直觉，单次遍历维护前缀最小值才是普适做法。

**Q2：$dp$ 数组必须开出来吗？**

不必。$dp[i]$ 只依赖 $dp[i-1]$，用 `ans` 一个变量滚动即可；`preMin` 同理。$n = 10^5$ 时省下 $400\text{KB}$ 的数组倒无所谓，但这种"降维"意识在股票系列后续题（如 [LeetCode 309](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/)）里会直接决定代码能不能写清爽。

**Q3：为什么答案初始化为 $0$ 而不是无穷小？**

题目允许"不交易"，利润下界是 $0$。若初始化为无穷小且忘记与 $0$ 取 `max`，单调递减序列会输出负利润。这个"下界截断"本质上是给状态机加了第三个选择：什么都不做。

---

## 6. 相关题目与模板

**可复用方法论**：**"枚举右端点，维护左侧极值"**——本题维护左侧最小值，[LeetCode 907. 子数组的最小值之和](https://leetcode.cn/problems/sum-of-subarray-minimums/)维护左侧更小元素的下标，本质都是把 $O(n^2)$ 的二元枚举拆成 $O(n)$ 的"扫描 + 增量维护"。

**股票系列进阶路线**（状态机逐个扩展）：

- [LeetCode 122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)：不限交易次数，贪心收集所有上坡
- [LeetCode 123. 买卖股票的最佳时机 III](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iii/)：至多两笔，两段前缀/后缀分解或四变量 DP
- [LeetCode 188. 买卖股票的最佳时机 IV](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/)：至多 $k$ 笔，通用状态机 DP
- [LeetCode 309. 买卖股票的最佳时机含冷冻期](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/)：卖出后隔一天才能再买
- [LeetCode 714. 买卖股票的最佳时机含手续费](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)：每笔交易扣固定手续费

**同类题目**：

- [LeetCode 53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)：同为"一次遍历、滚动维护"的经典，可对照体会两者状态设计的差异
- [LeetCode 152. 乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/)：53 的乘法版，需要同时维护最大与最小

**一句话总结**：买卖股票的最佳时机 = 枚举卖出日 × 前缀最小值——把二元问题压到一元扫描，一个变量记历史最低，一个变量记历史最好。
