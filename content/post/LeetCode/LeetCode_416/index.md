---
title: "LeetCode 416: 分割等和子集"
date: 2026-12-27
math: true
tags:
  - LeetCode
  - 数组
  - 动态规划
categories:
  - LeetCode
description: "LeetCode 416 分割等和子集题解（0-1 背包动态规划）"
---

# LeetCode 416. 分割等和子集

> 题目链接：[https://leetcode.cn/problems/partition-equal-subset-sum/](https://leetcode.cn/problems/partition-equal-subset-sum/)  
> 难度：中等  
> 标签：数组、动态规划

---

## 1. 题目描述

给你一个**只包含正整数**的非空数组 `nums`。请判断是否能把数组分割成两个子集，使得两个子集的元素和相等。

注意这里的"分割"是划分成两个**不相交**的子集、合起来覆盖所有元素，每个元素恰好属于其中一边。

**数据范围：**

- $1 \le nums.length \le 200$
- $1 \le nums[i] \le 100$

**示例：**

```
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1,5,5] 和 [11]，两者的和都是 11。

输入：nums = [1,2,3,5]
输出：false
解释：数组总和 11 是奇数，无法平分成两个整数和。
```

---

## 2. 思路分析

### 2.1 从"平分"到"挑出一半"：问题转化

设数组总和为 $S$。若能把数组分成和相等的两部分，每部分的和必须是 $S/2$；$S$ 为奇数直接返回 `false`。于是问题等价转化为：

> **能否从数组中挑出一个子集，其元素和恰好等于 $target = S/2$？**

这是经典的**子集和问题（subset sum）**，也是 0-1 背包（0-1 knapsack）的判定版：每个元素最多选一次，问能否恰好装满容量 $target$ 的背包。转化是双向的——挑出和为 $target$ 的子集后，剩下的元素之和自动也是 $target$。

### 2.2 暴力：枚举子集

每个元素"进左边"或"进右边"二选一，枚举 $2^n$ 种划分，$n = 200$ 时完全不可行。也可以 DFS + 剪枝，但元素值都很小（$\le 100$）这一数据特征提示：**和的取值范围很小**——$S \le 200 \times 100 = 2 \times 10^4$，$target \le 10^4$。状态空间应该开在"和"上，而不是"子集"上。

### 2.3 动态规划：布尔背包

定义 $dp[i][j]$ 为"从前 $i$ 个元素中能否选出和恰好为 $j$ 的子集"。转移看第 $i$ 个元素选不选：

$$dp[i][j] = dp[i-1][j] \lor dp[i-1][j - nums[i]]$$

（后者仅当 $j \ge nums[i]$ 时可用）。边界 $dp[i][0] = true$（什么都不选）。答案是 $dp[n][target]$。

用**一维滚动数组**压缩：$dp[j]$ 只依赖上一行**不超过 $j$** 的位置，把 $i$ 维压掉后，$j$ 必须**从大到小**枚举——保证 $dp[j - nums[i]]$ 用的还是"没考虑第 $i$ 个元素"的旧值（详见第 5 节疑问一）。

以 `nums = [1,5,11,5]` 走一遍，$S = 22$，$target = 11$。只列出关键下标（`T`/`F` 表示 true/false）：

| 处理元素后 | $dp[0]$ | $dp[1]$ | $dp[5]$ | $dp[6]$ | $dp[11]$ | 说明 |
|---|---|---|---|---|---|---|
| 初始 | T | F | F | F | F | 只有空集和为 0 |
| 1  | T | **T** | F | F | F | 可凑出 $\{1\}$ |
| 5  | T | T | **T** | **T** | F | 新增 $5 = 5$、$6 = 1+5$ |
| 11 | T | T | T | T | **T** | $11 = 11$ 直接命中 |
| 5  | T | T | T | T | T | $1+5+5 = 11$ 也可（$[1,5,5]$ 那一侧） |

$dp[11] = true$，返回 `true`。

---

## 3. 代码实现

```cpp
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        if (sum % 2) return false;          // 总和为奇数，无法平分
        int target = sum / 2;

        vector<char> dp(target + 1, 0);     // char 当 bool 用，省内存
        dp[0] = true;                       // 空子集凑出 0

        for (int x : nums) {
            // 0-1 背包：j 从大到小，每个元素最多用一次
            for (int j = target; j >= x; j--)
                dp[j] = dp[j] || dp[j - x];
        }
        return dp[target];
    }
};
```

三个实现细节：

1. **内层 $j$ 倒序枚举，且下界到 $x$ 为止**。倒序是 0-1 背包的生命线（正序会变成完全背包，每个元素可重复使用）；$j < x$ 时 $dp[j-x]$ 越界，这些状态本轮回不更新。
2. **`dp[0] = true` 是唯一初始真值**，语义是"空子集的和为 0"。
3. **提前剪枝**：某轮更新后若 `dp[target]` 已为真可以立即返回；更进一步的剪枝是若 `x > target` 直接 `false`（单个元素超过一半，另一侧装不下）。

---

## 4. 复杂度分析

- **时间复杂度**：$O(n \cdot target)$。本题为 $200 \times 10^4 = 2 \times 10^6$，轻松通过。
- **空间复杂度**：$O(target)$。一维滚动数组，$10^4$ 个字节。

对比：枚举子集 $O(2^n)$ → 01 背包 $O(n \cdot target)$。后者的复杂度与**元素值的规模**（pseudo-polynomial，伪多项式）挂钩，这正是子集和问题是 NPC 问题、但小数值场景下可解的原因。还可以用 `bitset` 把每位转移整体左移 $x$ 位再或起来，时间压到 $O(n \cdot target / w)$（$w = 64$），代码仅两行：

```cpp
bitset<10001> dp;
dp[0] = 1;
for (int x : nums) dp |= dp << x;   // 选或不选，位并行
```

---

## 5. 常见疑问

**疑问一：为什么 $j$ 必须倒序枚举？**

正序时，处理元素 $x$ 的过程中，$dp[j-x]$ 可能已经被**本轮**更新过——相当于"用了 $x$ 之后又用了一次 $x$"，元素被重复选取。用 $x = 2$、`dp` 初始只有 `dp[0]=T` 看差异：

| 枚举方式 | 处理 2 之后 | 实际语义 |
|---|---|---|
| 倒序 $j: 11 \to 2$ | `dp[2]=T`，其余 F | 恰好用一个 2，正确 |
| 正序 $j: 2 \to 11$ | `dp[2]=T, dp[4]=T, dp[6]=T, ...` 全 T | 2 被用了任意多次，变成完全背包 |

一句话记忆：**0-1 背包倒序（防复用），完全背包正序（鼓励复用）**。

**疑问二：为什么不初始化成"前缀和再判断"？**

有人会想：$target$ 固定，我排序后用双指针行不行？不行——子集和不是"选一段连续区间"，排序本身就打乱了原始结构，而本题并不要求连续（若是**连续子数组**的和为 $target$，才轮到前缀和 + 哈希出场，如 [LeetCode 560](https://leetcode.cn/problems/subarray-sum-equals-k/)）。贪心（从大到小往一边装）也有反例：`[3,3,4,2]`，$target = 6$，先装 4 再装 2 凑不齐 6，但 $3+3$ 可以——局部最优无法保证覆盖全局解，必须 DP。

**疑问三：元素改成含 0 或负数还成立吗？**

含 0 无影响（0 进哪边都不改变和，且本题数据 $nums[i] \ge 1$ 不会出现）。含负数则麻烦得多：$target = S/2$ 的转化仍成立，但 DP 的 $j$ 取值范围要覆盖负数段（平移偏移或用哈希表存可达集合），且"和恰好"的语义下负数会来回拉扯，复杂度不再由值域单侧界定。

---

## 6. 相关题目与模板

**可复用模板**（0-1 背包"能否恰好装满"判定）：

```cpp
// 判定：从 nums 中选子集，和是否恰好为 target
vector<char> dp(target + 1, 0);
dp[0] = 1;
for (int x : nums)
    for (int j = target; j >= x; j--)
        dp[j] = dp[j] || dp[j - x];
// dp[target] 即答案；求"最多装多少"把 || 改成 max(dp[j], dp[j-x] + w[j])
```

**0-1 背包三件套对比：**

| 问法 | 状态转移 | 典型题 |
|---|---|---|
| 能否恰好装满 | $dp[j] \lor= dp[j-x]$ | 本题 |
| 最多装多少 / 最值 | $dp[j] = \max(dp[j],\ dp[j-x] + v)$ | [1049. 最后一块石头的重量 II](https://leetcode.cn/problems/last-stone-weight-ii/) |
| 装满的方案数 | $dp[j] += dp[j-x]$ | [494. 目标和](https://leetcode.cn/problems/target-sum/) |

**同类题目：**

- [LeetCode 494. 目标和](https://leetcode.cn/problems/target-sum/)：转化后就是"选子集和为特定值"的**计数**版 0-1 背包
- [LeetCode 1049. 最后一块石头的重量 II](https://leetcode.cn/problems/last-stone-weight-ii/)：把石块分两堆使差最小，与本题同一个转化
- [LeetCode 322. 零钱兑换](https://leetcode.cn/problems/coin-change/)：完全背包版（硬币可重复用），对照体会正序/倒序的区别
- [LeetCode 698. 划分为 k 个相等的子集](https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/)：本题的 $k = 2$ 推广到任意 $k$，状态压缩 + 回溯

**一句话总结**：等和分割的本质是**子集和判定**——总和折半当背包容量，每个元素一件物品，布尔 0-1 背包倒序滚动即可。