---
title: "LeetCode 139: 单词拆分"
date: 2026-12-24
math: true
tags:
  - LeetCode
  - 字符串
  - 动态规划
categories:
  - LeetCode
description: "LeetCode 139 单词拆分题解（字符串动态规划）"
---

# LeetCode 139. 单词拆分

> 题目链接：[https://leetcode.cn/problems/word-break/](https://leetcode.cn/problems/word-break/)  
> 难度：中等  
> 标签：字符串、动态规划

---

## 1. 题目描述

给你一个字符串 `s` 和一个字符串列表 `wordDict` 作为字典。判断 `s` 是否可以被**拆分成一个或多个字典中出现的单词**（字典中的单词可以重复使用）。

拆分时要求把 `s` 切成若干段，每一段都恰好是字典里的某个单词，且拼接结果等于 `s` 本身。

**数据范围：**

- $1 \le s.length \le 300$
- $1 \le wordDict.length \le 1000$
- $1 \le wordDict[i].length \le 20$
- `s` 和 `wordDict[i]` 仅由小写英文字母组成
- `wordDict` 中的字符串**互不相同**

**示例：**

```
输入：s = "leetcode", wordDict = ["leet", "code"]
输出：true
解释："leetcode" = "leet" + "code"。
```

```
输入：s = "applepenapple", wordDict = ["apple", "pen"]
输出：true
解释："apple" + "pen" + "apple"，单词可重复使用。
```

```
输入：s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出：false
```

---

## 2. 思路分析

### 2.1 暴力：从左到右尝试切分

递归：从位置 0 出发，枚举第一段的结束位置，若前缀在字典里就递归处理剩下的后缀；任何一条路径走到底（消耗完整个 `s`）就返回 true。

`"catsandog"` 的前几层分支：切出 `"cats"` 剩 `"andog"`、切出 `"cat"` 剩 `"sandog"`……裸递归最坏是指数级（每段长度都有 20 种可能），而且**同一个「剩余后缀能不能拆」的子问题被不同路径反复求解**——比如 `"sandog"` 既可能从 `"cat"+"sandog"` 到达，也可能从 `"cats"+"andog"` 的后续到达。重叠子问题，该上动态规划 (dynamic programming)。

### 2.2 状态设计与转移方程

关键是**只关心位置，不关心切法**。定义：

$$dp[i] = \text{true} \iff s[0..i)\ \text{可以被拆分成字典中的单词}$$

即「前 $i$ 个字符组成的**前缀**是否可拆」。转移时枚举**最后一个单词**的起点 $j$：

$$dp[i] = \mathrm{OR}_{0 \le j < i}\bigl(dp[j] \wedge s[j..i) \in \text{dict}\bigr)$$

直观读法：前缀 $s[0..i)$ 可拆，当且仅当存在某个分割点 $j$，使得**更短的前缀 $s[0..j)$ 可拆**，且**夹在中间的最后一段 $s[j..i)$ 是字典词**。边界 $dp[0] = \text{true}$（空前缀天然可拆——一个单词都不用）。

这正是「完全背包」的字符串版：零钱兑换是「用硬币凑金额」，本题是「用单词凑前缀」，金额对应下标 $i$，硬币对应子串 $s[j..i)$——不过硬币必须从 $s$ 里切出来，能否使用由字符串内容决定。

### 2.3 剪枝：单词最长 20

转移里 $j$ 从 $i-1$ 往回枚举即可，因为字典词最长 20，$j < i - 20$ 时 $s[j..i)$ 长度超过 20，必然不在字典里，直接停止。这一步把每状态的转移次数从 $O(n)$ 压到 $O(L) = O(20)$，是本题最重要的优化。

### 2.4 逐步填表：s = "catsandog"

字典 `{"cats", "dog", "sand", "and", "cat"}`，下标从 1 开始的 $dp[i]$（前 $i$ 个字符）：

| $i$ | 前缀 | 可行切点 $j$（$dp[j]$ 且 $s[j..i)$ 在字典） | $dp[i]$ |
|-----|-------|----------------------------------------------|---------|
| 0 | "" | —（边界） | true |
| 3 | "cat" | $j{=}0$，`"cat"` | true |
| 4 | "cats" | $j{=}0$，`"cats"` | true |
| 7 | "catsand" | $j{=}3$，`"sand"`；$j{=}4$，`"and"` | true |
| 8 | "catsando" | 无（`"o"`、`"do"`、`"ndo"`… 均不在字典） | false |
| 9 | "catsandog" | $j{=}7$ 时剩 `"og"`；$j{=}8$ 时剩 `"g"`；$dp[8]$ 本身 false | **false** |

答案 $dp[9] = \text{false}$：`"catsandog"` 的两种读法 `"cats"+"and"+"og"` 或 `"cat"+"sand"+"og"` 都卡在尾巴 `"og"` 上，与示例一致。这个小例子也展示了 DP 的价值：`dp[7] = true` 有两条路径达成，暴力递归会把 `"andog"`、`"sandog"` 各算一遍，填表只算一次。

---

## 3. 代码实现

```cpp
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> dict(wordDict.begin(), wordDict.end());
        int n = s.size(), maxLen = 0;
        for (auto& w : wordDict) maxLen = max(maxLen, (int)w.size());

        // dp[i]：前 i 个字符可否拆分
        vector<bool> dp(n + 1, false);
        dp[0] = true;                          // 空前缀可拆

        for (int i = 1; i <= n; i++) {
            // j 从 i-1 往回枚举，最后一段 s[j..i) 最长 maxLen
            for (int j = i - 1; j >= 0 && i - j <= maxLen; j--) {
                if (dp[j] && dict.count(s.substr(j, i - j))) {
                    dp[i] = true;
                    break;                     // 已可行，无需再找
                }
            }
        }
        return dp[n];
    }
};
```

实现细节：

1. **字典转 `unordered_set`**：转移中要做「子串是否在字典」的查询，哈希集合把每次查询降到均摊 $O(L)$（哈希比较子串长度），比在 `vector` 里线性扫快一个量级以上。
2. **`i - j <= maxLen` 剪枝**：最后一段长度超过字典最大词长时直接停，利用了 $wordDict[i].length \le 20$ 的数据范围——读数据范围并转化为剪枝，是做题的好习惯。
3. **`dp[j]` 先于子串查询判断**：`dp[j] == false` 时根本不用取子串，短路求值省下哈希开销。
4. **`dp` 长 $n+1$，下标偏移 1**：$dp[i]$ 对应前 $i$ 个字符，`s.substr(j, i - j)` 即 $s[j..i)$，注意这是左闭右开区间，别和 $dp$ 下标混了。

---

## 4. 复杂度分析

- **时间复杂度**：$O(n \cdot L^2)$，其中 $n = |s|,\ L = 20$ 为最大词长。每个终点 $i$ 回看至多 $L$ 个起点 $j$；每次转移取子串并哈希比较代价 $O(L)$。$n = 300$ 时约 $300 \times 20 \times 20 = 1.2 \times 10^5$ 量级。不加词长剪枝则是 $O(n^2 \cdot L)$，本题规模下也能过，但剪枝体现对范围的理解。
- **空间复杂度**：$O(n + |dict| \cdot L)$。`dp` 数组 $O(n)$，哈希集合存储所有单词共 $O(|dict| \cdot L)$ 个字符。

---

## 5. 常见疑问

**疑问一：为什么 $dp[0]$ 必须是 true？把它当边界初始化成 false 会怎样？**

$dp[0]$ 的含义是「空前缀可拆（零个单词）」，它是所有转移的根基：第一个单词对应的切点是 $j = 0$，若 $dp[0] = false$，则任何 $dp[i]$ 都无法通过「第一段从 0 开始」的路径变 true，全表皆 false。这类「空对象合法」的边界（空串、金额 0、容量 0）是字符串 DP 最常见的坑，初始化前先问自己：**空输入的答案是什么**。

**疑问二：本题和零钱兑换结构这么像，循环顺序有讲究吗？**

求「能否」（布尔 OR）和「最少件数」（min）都对顺序不敏感，内外层可换。但若换成「求拆分方案数」（[LeetCode 140 的计数版变体](https://leetcode.cn/problems/word-break-ii/)），就要区分「单词外层=组合计数」「位置外层=排列计数」了，道理与零钱兑换 518/377 之争完全一致，见本系列 [LeetCode 322 题解](../LeetCode_322/) 第 5 节。

**疑问三：记忆化递归（从 0 出发切前缀）和本文的填表 DP 等价吗？**

等价。记忆化是「自顶向下」：`canBreak(i)` 查 `i` 之后能否拆；填表是「自底向上」：按前缀长度从小到大算 $dp[i]$。两者访问的状态集合相同（都是 $n+1$ 个位置），只是顺序不同。记忆化递归的好处是只会真正访问可达状态，填表的好处是无递归栈、代码更平。本题 $n \le 300$，随便选；若是 $10^5$ 级别的字符串（如 [LeetCode 472](https://leetcode.cn/problems/concatenated-words/) 的变体场景），就要认真考虑剪枝与访问顺序了。

---

## 6. 相关题目与模板

**可复用模板**（字符串「前缀可拆」DP 骨架）：

```cpp
// dp[i]：s[0..i) 能否由字典拼出；maxLen 为字典最大词长
unordered_set<string> dict(wordDict.begin(), wordDict.end());
vector<bool> dp(n + 1, false);
dp[0] = true;
for (int i = 1; i <= n; i++)
    for (int j = max(0, i - maxLen); j < i; j++)   // 或从 i-1 倒序
        if (dp[j] && dict.count(s.substr(j, i - j))) {
            dp[i] = true;
            break;
        }
```

方法论总结：**字符串切分问题的第一反应——把「位置」设为状态（前缀 DP），枚举最后一段的起点**；再拿数据范围（词长上界）做剪枝。它和完全背包同构：物品=单词，容量=前缀长度，转移=「子状态可达 且 新增段合法」。

**同类题目：**

- [LeetCode 140. 单词拆分 II](https://leetcode.cn/problems/word-break-ii/)：从「能否」升级为「列出所有拆分方案」，回溯时复用本题的 dp 做可行性剪枝可避免指数爆炸
- [LeetCode 131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/)：同样按位置 DP，「在字典中」换成「是回文」
- [LeetCode 139 → 472. 连接词](https://leetcode.cn/problems/concatenated-words/)：判断每个词能否由更短的词拼出，本质是批量单词拆分
- [LeetCode 322. 零钱兑换](https://leetcode.cn/problems/coin-change/)：同构的完全背包模型，见本系列 [LeetCode 322 题解](../LeetCode_322/)

**一句话总结**：单词拆分 = 完全背包的字符串版——$dp[i] = \mathrm{OR}_j\,(dp[j] \wedge s[j..i) \in dict)$，空串边界置 true，再用最大词长 20 把转移剪到常数级别。
