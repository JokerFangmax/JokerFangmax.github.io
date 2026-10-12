---
title: "LeetCode 72: 编辑距离"
date: 2027-01-01
math: true
tags:
  - LeetCode
  - 字符串
  - 动态规划
categories:
  - LeetCode
description: "LeetCode 72 编辑距离题解（动态规划）"
---

# LeetCode 72. 编辑距离

> 题目链接：[https://leetcode.cn/problems/edit-distance/](https://leetcode.cn/problems/edit-distance/)  
> 难度：中等  
> 标签：字符串、动态规划

---

## 1. 题目描述

给你两个单词 `word1` 和 `word2`，返回把 `word1` 转换成 `word2` 所需的**最少操作数**。每次操作可以三选一：

1. **插入**一个字符；
2. **删除**一个字符；
3. **替换**一个字符。

**数据范围：**

- $0 \le word1.length, word2.length \le 500$
- 两个单词都由小写英文字母组成

**示例：**

```
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse → rorse   （把 h 替换为 r）
rorse → rose    （删除 r）
rose  → ros     （删除 e）

输入：word1 = "intention", word2 = "execution"
输出：5
解释：intention → inention → enention → exention → exection → execution
```

编辑距离 (edit distance) 又称 Levenshtein 距离，是拼写纠错、DNA 序列比对、diff 工具背后的基础算法。

---

## 2. 思路分析

### 2.1 暴力：递归枚举操作

从两个串的开头出发比较 `word1[i]` 与 `word2[j]`：相等则各进一位；不等则分别尝试替换、插入、删除三种操作，取操作数最小的分支。问题在于分支爆炸——最坏每步三分叉，复杂度是指数级。与 [最长公共子序列](../LeetCode_1143/) 一样，不同的操作序列经常殊途同归地到达同一对 $(i, j)$，重叠子问题提示我们上动态规划 (dynamic programming)。

### 2.2 状态定义与转移

> $dp[i][j]$ = 把 `word1` 的前 $i$ 个字符（$word1[0..i-1]$）变成 `word2` 的前 $j$ 个字符（$word2[0..j-1]$）所需的最少操作数。

**边界**（一边为空串，只能整体增删）：

- $dp[i][0] = i$：把长度 $i$ 的串变成空串，删 $i$ 次；
- $dp[0][j] = j$：把空串变成长度 $j$ 的串，插 $j$ 次。

**转移**，仍然按**末尾字符**分类：

**情况 A：$word1[i-1] = word2[j-1]$。** 末尾字符已一致，什么都不用做，剩余问题缩到两边都去掉末尾：

$$dp[i][j] = dp[i-1][j-1]$$

**情况 B：末尾字符不同。** 最后一步操作必然要处理掉 $word1$ 的末尾（三种操作各对应一个来源）：

| 操作 | 直观含义 | 转移来源 |
|---|---|---|
| 替换 | 把 $word1[i-1]$ 改成 $word2[j-1]$，两个末尾同时解决 | $dp[i-1][j-1] + 1$ |
| 删除 | 删掉 $word1[i-1]$，$word2$ 的末尾还没对付 | $dp[i-1][j] + 1$ |
| 插入 | 在 $word1$ 末尾插入 $word2[j-1]$，匹配掉了它，但 $word1$ 的末尾还在 | $dp[i][j-1] + 1$ |

$$dp[i][j] = 1 + \min(dp[i-1][j-1],\; dp[i-1][j],\; dp[i][j-1])$$

**为什么恰好三种、不多不少？** 任何一步操作至多"消耗" $word1$ 或 $word2$ 的一个末尾字符：替换消耗两个（对角线）、删除消耗 $word1$ 的（上方）、插入消耗 $word2$ 的（左方）——三个方向正好覆盖所有可能，且都把问题化小，递推可以封闭进行。

### 2.3 情况 A 里要不要顺手取个 min

一个常见的纠结点：末尾字符相等时，理论上也可以"硬操作"（比如把它删了），万一后面更便宜呢？可以证明不必考虑：

若 $word1[i-1] = word2[j-1]$，把任何"从 $dp[i-1][j-1]$ 出发的方案"末尾追加"让这两个字符对齐"的部分，开销不会超过直接配对（为 0）。更严格地：$dp[i-1][j-1] \le dp[i-1][j] + 1 \le dp[i][j]$ 链在取 min 时自动成立。所以**相等时直接取对角线，既正确又省事**——这与 LCS 里"末尾相等配对不亏"是同一条引理。

### 2.4 用示例走一遍 DP 表

`word1 = "horse"`（行），`word2 = "ros"`（列）：

| | "" | r | o | s |
|---|---|---|---|---|
| **""** | 0 | 1 | 2 | 3 |
| **h** | 1 | 1 | 2 | 3 |
| **o** | 2 | 2 | 1 | 2 |
| **r** | 3 | 2 | 2 | 2 |
| **s** | 4 | 3 | 3 | 2 |
| **e** | 5 | 4 | 4 | **3** |

抽几个代表性格子看计算：

| 格子 | 末尾字符 | 计算 | 结果 |
|---|---|---|---|
| $dp[1][1]$ | `h` ≠ `r` | $1 + \min(0, 1, 1)$ = 1 | 替换 `h`→`r` 最划算 |
| $dp[2][2]$ | `o` = `o` | $dp[1][1] = 1$ | 白送一次对齐 |
| $dp[3][1]$ | `r` = `r` | $dp[2][0] = 2$ | "ho" 变 "r"：删 o、替换 h，共 2 步 |
| $dp[4][3]$ | `s` = `s` | $dp[3][2] = 2$ | "hors" 变 "ros" 用 2 步 |
| $dp[5][3]$ | `e` ≠ `s` | $1 + \min(dp[4][2]=3, dp[4][3]=2, dp[5][2]=4)$ = 3 | 删掉 `e` 即达 |

答案 $dp[5][3] = 3$：从右下角回溯可知方案是删 `e`、删 `r`（替换 `h`→`r` 之后 `rorse` 里多出来的那个 `r`）、替换 `h`→`r`，恰好对应题面给出的三步。

### 2.5 空间优化提示

与 LCS 完全同构：转移只用到上一行和对角线，两个一维数组（或一个数组 + 一个暂存左上角的变量）即可压到 $O(n)$。$500 \times 500$ 的表只有约 1 MB，本题直接开二维最省心。

---

## 3. 代码实现

```cpp
class Solution {
public:
    int minDistance(string word1, string word2) {
        int m = word1.size(), n = word2.size();
        // dp[i][j]: word1 前 i 个字符 → word2 前 j 个字符的最少操作数
        vector<vector<int>> dp(m + 1, vector<int>(n + 1));

        for (int i = 0; i <= m; i++) dp[i][0] = i;   // 边界：全部删除
        for (int j = 0; j <= n; j++) dp[0][j] = j;   // 边界：全部插入

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1[i - 1] == word2[j - 1])
                    dp[i][j] = dp[i - 1][j - 1];     // 末尾相同：白送
                else
                    dp[i][j] = 1 + min({dp[i - 1][j - 1],  // 替换
                                        dp[i - 1][j],      // 删除 word1 末字符
                                        dp[i][j - 1]});    // 插入 word2 末字符
            }
        }
        return dp[m][n];
    }
};
```

几个实现细节：

1. **`min({...})` 是初始化列表重载**，比嵌套两次 `min(min(a,b),c)` 可读性好得多。
2. **边界先填满第 0 行和第 0 列再进主循环**，主循环里就不必再特判 $i$ 或 $j$ 为 0。
3. **空串输入**（$m = 0$ 或 $n = 0$）被边界天然覆盖，直接返回 $n$ 或 $m$，无需特判。

---

## 4. 复杂度分析

- **时间复杂度**：$O(mn)$。$m, n \le 500$，最多 $2.5 \times 10^5$ 个状态、每次 $O(1)$ 转移。与 LCS 一样，$O(mn)$ 是双串编辑距离的实践最优（强次二次算法的存在性与 SETH 相关）。
- **空间复杂度**：$O(mn)$；滚动数组可压至 $O(\min(m, n))$（需要沿较小维开表）。

---

## 5. 常见疑问

**Q1：'插入' 和 '删除' 是不是重复算了？**

看似有六种操作（对 A 插入、对 B 插入、删 A、删 B、A→B 替换、B→A 替换），实际只有三种。原因：**在对 `word1` 末尾插入一个字符 = 让 `word2` 的末尾字符被匹配掉，等价于"删除 `word2` 的末尾"**——本题目标固定是 `word1 → word2`，对 `word2` 的删除只能通过在 `word1` 里插入来实现，对 `word2` 的插入只能通过删 `word1` 来实现，双向替换也只有 `word1 → word2` 一个方向。所以表格里三个方向（对角线/上/左）恰好对应三组等价类，不多不少。

**Q2：能不能用 LCS 反推编辑距离？**

只有当**不允许替换**（或替换代价为 2）时才有公式 $ans = m + n - 2 \cdot LCS$（见 [LeetCode 583](https://leetcode.cn/problems/delete-operation-for-two-strings/)）。本题替换只花 1 步，替换"一步顶删除 + 插入两步"，纯 LCS 公式会高估：`"horse"` 与 `"ros"` 的 LCS 是 `"os"`（长 2，`"ros"` 不是——`horse` 里 `o` 在 `r` 前面，顺序对不上），按公式得 $5 + 3 - 2 \times 2 = 4 > 3$，而真实答案 3 靠一次替换省下了一步。**替换的存在打破了 LCS 抵消的美梦**，必须老老实实列三种操作。

**Q3：为什么先判相等再转移，反过来行不行？**

把相等判断并入 `1 + min(...)`（即相等时也允许花 1 步替换成同一个字符）不影响正确性——取 min 后 $dp[i-1][j-1]$ 本来就是下界。但把相等分支单独写出来，一是省掉三方向的比较，二是**语义清晰**：相等时的转移"不消耗操作"是这道题最重要的结构（操作只发生在失配处），值得在代码里显式表达。

---

## 6. 相关题目与模板

**可复用模板**（双串 DP 骨架 + 三方向转移，比 [LCS 模板](../LeetCode_1143/) 多一个"代价"维度）：

```cpp
// dp[i][j]: 把 s 前 i 个字符变成 t 前 j 个字符的最小总代价
vector<vector<int>> dp(m + 1, vector<int>(n + 1));
for (int i = 0; i <= m; i++) dp[i][0] = i * COST_DEL;
for (int j = 0; j <= n; j++) dp[0][j] = j * COST_INS;
for (int i = 1; i <= m; i++)
    for (int j = 1; j <= n; j++)
        dp[i][j] = (s[i-1] == t[j-1])
            ? dp[i-1][j-1]
            : min({dp[i-1][j-1] + COST_SUB,   // 替换
                   dp[i-1][j]   + COST_DEL,   // 删除
                   dp[i][j-1]   + COST_INS}); // 插入
```

把三个代价设成不同函数，就得到**带权编辑距离**——生物信息学里 DNA 序列比对（不同碱基替换代价不同）用的正是它。

**同类题目：**

- [LeetCode 583. 两个字符串的删除操作](https://leetcode.cn/problems/delete-operation-for-two-strings/)：砍掉插入/替换，只剩删除——退化成 $m + n - 2 \cdot LCS$
- [LeetCode 712. 两个字符串的最小 ASCII 删除和](https://leetcode.cn/problems/minimum-ascii-delete-sum-for-two-strings/)：删除代价从 1 变成字符 ASCII 值
- [LeetCode 1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/)：本系列 [../LeetCode_1143/](../LeetCode_1143/)，编辑距离的"零替换"亲戚
- [LeetCode 161. 相隔为 1 的编辑距离](https://leetcode.cn/problems/one-edit-distance/)：判断 $dp = 1$，可以不建表直接双指针
- [LeetCode 10. 正则表达式匹配](https://leetcode.cn/problems/regular-expression-matching/)：双串 DP 的进阶——转移里出现 `*` 的"零次/多次"分支

**一句话总结**：编辑距离是双串 DP 的完全体——末尾相等白送、失配时"替换走对角、删除走上、插入走左"三路取 min，LCS 是它把替换代价设为无穷大的特例；$O(mn)$ 一张表，撑起了从拼写纠错到 DNA 比对的整个应用家族。
