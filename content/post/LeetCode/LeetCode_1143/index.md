---
title: "LeetCode 1143: 最长公共子序列"
date: 2026-12-31
math: true
tags:
  - LeetCode
  - 字符串
  - 动态规划
categories:
  - LeetCode
description: "LeetCode 1143 最长公共子序列题解（动态规划）"
---

# LeetCode 1143. 最长公共子序列

> 题目链接：[https://leetcode.cn/problems/longest-common-subsequence/](https://leetcode.cn/problems/longest-common-subsequence/)  
> 难度：中等  
> 标签：字符串、动态规划

---

## 1. 题目描述

给定两个字符串 `text1` 和 `text2`，返回它们**最长公共子序列 (longest common subsequence, LCS)** 的长度；不存在公共子序列时返回 $0$。

**子序列 (subsequence)** 是不改变剩余字符相对顺序、可以删掉若干字符（也可以不删）得到的新字符串——注意**不要求连续**。例如 `"ace"` 是 `"abcde"` 的子序列，但 `"aec"` 不是（`e` 跑到了 `c` 前面，相对顺序变了）。

**数据范围：**

- $1 \le text1.length, text2.length \le 1000$
- `text1` 和 `text2` 仅由小写英文字母组成

**示例：**

```
输入：text1 = "abcde", text2 = "ace"
输出：3
解释：最长公共子序列是 "ace"，长度为 3。

输入：text1 = "abc", text2 = "abc"
输出：3

输入：text1 = "abc", text2 = "def"
输出：0
```

---

## 2. 思路分析

### 2.1 暴力：枚举子序列比对

`text1` 有 $2^{m}$ 个子序列（每个字符选或不选），与 `text2` 的 $2^{n}$ 个逐一比对是否为公共子序列。$m = n = 1000$ 时 $2^{1000}$ 是天文数字，完全不可行。同样的问题结构：**大量子序列彼此共享前缀**，重复计算爆炸。

一个更现实的暴力是递归"双指针"：比较 $text1[i]$ 与 $text2[j]$，相等则匹配并各进一位，不等则试着跳过 $text1[i]$ 或跳过 $text2[j]$——但跳过的两个分支会产生大量重叠子问题（不同的跳过顺序到达同一对 $(i, j)$），指数级。重叠子问题，正是动态规划 (dynamic programming) 的用武之地。

### 2.2 状态定义：两个指针的位置

> $dp[i][j]$ = $text1[0..i-1]$（前 $i$ 个字符）与 $text2[0..j-1]$（前 $j$ 个字符）的 LCS 长度。

用"前缀长度"而不是下标做状态，可以让 $i = 0$ / $j = 0$（空串）自然成为边界，避免负下标。转移按**末尾字符**分类讨论：

**情况 A：$text1[i-1] = text2[j-1]$（末尾字符相同）。** 这个公共字符一定可以放进 LCS 的末尾：

$$dp[i][j] = dp[i-1][j-1] + 1$$

**情况 B：末尾字符不同。** 公共子序列不会同时以 $text1[i-1]$ 和 $text2[j-1]$ 结尾（它们不相等），所以至少有一个"末尾字符"不参与——两种可能取较优：

$$dp[i][j] = \max(dp[i-1][j],\; dp[i][j-1])$$

边界 $dp[i][0] = dp[0][j] = 0$：任何串与空串的 LCS 是空串。

### 2.3 为什么情况 A 不需要再和 max 比较

一个经典疑问：末尾字符相等时，万一"不匹配它们"反而更长呢？可以证明**不会**。若 $text1[i-1] = text2[j-1] = c$，则：

- $dp[i-1][j] \le dp[i-1][j-1] + 1$：给 $text1[0..i-2]$ 与 $text2[0..j-1]$ 的任意 LCS（长度 $\le dp[i-1][j-1] + 1$，多出的一个字符至多贡献 1）末尾接上 $c$，就得到 $text1[0..i-1]$ 与 $text2[0..j-1]$ 的一个公共子序列——所以 $dp[i][j] \ge dp[i-1][j-1] + 1$ 已经不会输给任何跳过方案；

直观理解：**末尾字符相等时，"配对它们"永远不亏**——大不了这个配对用在 LCS 最末端，前面该怎么长还怎么长。这个结论让代码少写一个 `max`，也少一次出错机会。

### 2.4 用示例走一遍 DP 表

`text1 = "abcde"`（行，$i$ 从上到下），`text2 = "ace"`（列，$j$ 从左到右）：

| | "" | a | c | e |
|---|---|---|---|---|
| **""** | 0 | 0 | 0 | 0 |
| **a** | 0 | **1** | 1 | 1 |
| **b** | 0 | 1 | 1 | 1 |
| **c** | 0 | 1 | **2** | 2 |
| **d** | 0 | 1 | 2 | 2 |
| **e** | 0 | 1 | 2 | **3** |

几个关键格子的计算过程：

| 格子 | 末尾字符 | 情况 | 计算 | 结果 |
|---|---|---|---|---|
| $dp[1][1]$ | `a` = `a` | A | $dp[0][0] + 1 = 1$ | 1（LCS 匹配到 `a`） |
| $dp[2][2]$ | `b` ≠ `c` | B | $\max(dp[1][2], dp[2][1]) = \max(1,1)$ | 1（`b` 跳过，值不变） |
| $dp[3][2]$ | `c` = `c` | A | $dp[2][1] + 1 = 2$ | 2（LCS 是 `ac`） |
| $dp[5][3]$ | `e` = `e` | A | $dp[4][2] + 1 = 3$ | 3（LCS 是 `ace`） |

答案 $dp[5][3] = 3$。想还原出子序列本身，可以从右下角回溯：字符相等时左上走并记录，不等时走向值较大的那一侧——本例回溯路径恰好经过 `a`、`c`、`e` 三个相等格，拼出 `"ace"`。

### 2.5 空间优化提示

转移只依赖上一行与本行左侧，可压成两个一维数组（甚至一个数组加一个"左上角暂存变量"），空间 $O(\min(m, n))$。求长度够用；需要还原子序列时保留完整表更省事。

---

## 3. 代码实现

```cpp
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int m = text1.size(), n = text2.size();
        // dp[i][j]: text1 前 i 个字符与 text2 前 j 个字符的 LCS 长度
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1[i - 1] == text2[j - 1])
                    dp[i][j] = dp[i - 1][j - 1] + 1;   // 情况 A：末尾配对
                else
                    dp[i][j] = max(dp[i - 1][j],       // 跳过 text1 的末字符
                                   dp[i][j - 1]);      // 跳过 text2 的末字符
            }
        }
        return dp[m][n];
    }
};
```

几个实现细节：

1. **数组开 $(m+1) \times (n+1)$，第 0 行/列表示空串**，天然是边界 0，不需要单独初始化。
2. **`dp[i][j]` 与"前缀长度"对应，`text1[i-1]` 才是第 $i$ 个字符**——这是本模板最经典的差一来源，写 `text1[i]` 会越界或错位。
3. $1001 \times 1001$ 个 `int` 约 4 MB，可以接受；内存敏感时换滚动数组。

---

## 4. 复杂度分析

- **时间复杂度**：$O(mn)$。$m, n \le 1000$，最多 $10^6$ 次状态计算，每次 $O(1)$ 比较 + 转移。注意 LCS 没有已知的**严格**亚二次算法（能否突破 $O(n^{2-\epsilon})$ 与 SETH 强假设相关，是理论开放问题），$O(mn)$ 已是实践最优。
- **空间复杂度**：$O(mn)$；滚动优化后 $O(\min(m, n))$。

---

## 5. 常见疑问

**Q1：为什么"最长公共子**串**"不能套这个转移？**

子串要求**连续**。`text1 = "abcde"`, `text2 = "abce"` 的 LCS 是 `abce`（长 4），但最长公共子串只是 `abc` 或 `bce`（长 3）。子串版的转移是 $dp[i][j] = dp[i-1][j-1] + 1$（仅当末尾相等，否则**直接归 0**，不能像 LCS 那样从 $dp[i-1][j]$ 延续——延续意味着断开的串被"接上"了，而这在连续意义下不合法）。详见 [LeetCode 718. 最长重复子数组](https://leetcode.cn/problems/maximum-length-of-repeated-subarray/)。一句话：**子序列的状态可以"跨过"不匹配处，子串不行**。

**Q2：情况 A 里写 `dp[i][j] = max(dp[i-1][j-1] + 1, dp[i-1][j], dp[i][j-1])` 会错吗？**

不会错，只是冗余（2.3 节已证明三者中第一个不会更小），多比较不改变结果。真正会错的是反方向偷懒：情况 B 里**只写** `dp[i-1][j-1]` 之类的单项——不匹配时两个末尾字符都可能被跳过，必须取两项 max。

**Q3：为什么不能用双指针一遍扫完？**

子序列不要求连续，双指针在失配时无法确定"该跳过谁"——两个方向都必须试。`"abcde"` 和 `"ace"`：第一字符 `a` 匹配后，`b` 对 `c` 失配，跳过 `b` 是对的；但换 `text2 = "be"` 时跳过 `b` 就全盘皆输。失配的双向不确定性使得任何单向扫描都会丢解，必须用二维状态把两种跳过都覆盖住。

---

## 6. 相关题目与模板

**可复用模板**（双串 DP 通用骨架，编辑距离、回文系列全部套它）：

```cpp
// dp[i][j]: 只与前缀/后缀下标 i, j 相关的指标
vector<vector<int>> dp(m + 1, vector<int>(n + 1, /* 边界值 */));
for (int i = 1; i <= m; i++)
    for (int j = 1; j <= n; j++)
        if (s[i - 1] == t[j - 1]) dp[i][j] = /* 对角线转移 + 1 */;
        else                      dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
```

**LCS 是一堆题的"底座"：**

- [LeetCode 583. 两个字符串的删除操作](https://leetcode.cn/problems/delete-operation-for-two-strings/)：只许删除，最少步数 $= m + n - 2 \cdot LCS$
- [LeetCode 712. 两个字符串的最小 ASCII 删除和](https://leetcode.cn/problems/minimum-ascii-delete-sum-for-two-strings/)：带权版，转移里 `+1` 换成字符的 ASCII 值
- [LeetCode 1035. 不相交的线](https://leetcode.cn/problems/uncrossed-lines/)：换皮 LCS——不相交连线恰好就是公共子序列
- [LeetCode 718. 最长重复子数组](https://leetcode.cn/problems/maximum-length-of-repeated-subarray/)：子串（连续）版，失配归 0
- [LeetCode 516. 最长回文子序列](https://leetcode.cn/problems/longest-palindromic-subsequence/)：$s$ 与自己的逆序串求 LCS（也可区间 DP 直接做）
- [LeetCode 72. 编辑距离](https://leetcode.cn/problems/edit-distance/)：双串 DP 的完全体，见本系列 [../LeetCode_72/](../LeetCode_72/)

**一句话总结**：LCS 是双串 DP 的原点——"末尾相同就配对、不同就各跳各的取 max"，$2^m$ 级的子序列枚举被压缩成一张 $(m+1) \times (n+1)$ 的表；后续的编辑距离、删除操作、不相交的线，全是这张表的变体。
