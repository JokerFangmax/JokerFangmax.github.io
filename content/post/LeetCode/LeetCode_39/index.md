---
title: "LeetCode 39: 组合总和"
date: 2026-11-26
math: true
tags:
  - LeetCode
  - 数组
  - 回溯
categories:
  - LeetCode
description: "LeetCode 39 组合总和题解（回溯剪枝）"
---

# LeetCode 39. 组合总和

> 题目链接：[https://leetcode.cn/problems/combination-sum/](https://leetcode.cn/problems/combination-sum/)  
> 难度：中等  
> 标签：数组、回溯

---

## 1. 题目描述

给定一个**无重复元素**的正整数数组 `candidates` 和一个目标数 `target`，找出 `candidates` 中所有可以使数字和为 `target` 的组合。同一个数字可以被**无限次重复选取**；`candidates` 中的数字互不相同，且答案中不能有重复的组合。组合内的数字可以按任意顺序排列，组合列表的顺序也可以任意。

**数据范围：**

- $2 \le candidates.length \le 40$
- $1 \le candidates[i] \le 40$，且元素互不相同
- $1 \le target \le 40$

**示例：**

```
输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]
解释：2 + 2 + 3 = 7，7 = 7。其余组合都无法凑出 7。

输入：candidates = [2,3,5], target = 8
输出：[[2,2,2,2],[2,3,3],[3,5]]
```

---

## 2. 思路分析

### 2.1 暴力：从"每个数选多少个"想到搜索

"和恰好为 $target$"、数字可重复选，最朴素的念头是枚举每个数字的个数 $c_1, c_2, \dots$，满足 $\sum c_i \cdot a_i = target$。但这样要枚举所有非负整数向量，组合爆炸；更自然的视角是**决策过程**：站在"当前还差 `rest` 要凑"的状态上，每次从 `candidates` 里挑一个数减掉 `rest`，`rest` 减到 $0$ 就收获一个组合，减成负数就失败。

这正是**回溯 (backtracking)** 的形状：路径是已选的数字列表，深度的上界是 $target / \min(candidates) \le 40$。

### 2.2 防重复的关键：只许往前选

直接实现马上会遇到致命问题——`target = 7` 时 `[2,2,3]` 和 `[3,2,2]` 会被当成两个答案。原因是**同一批数字按不同顺序被走了多条路径**。

解决办法是给递归加一个**起始下标 `start`**：第 $i$ 层选了 `candidates[j]`，下一层只允许从 $j$ 开始选（注意是 $j$ 不是 $j+1$，因为元素可以重复用）。这相当于强制路径上的下标**单调不减**：

| 规则 | 允许的路径 | 效果 |
|------|-----------|------|
| 每层都从 $0$ 开始选 | `[2,2,3]`、`[2,3,2]`、`[3,2,2]` 都走到 | 同一组合出现多次 |
| 下一层从当前 $j$ 开始 | 只允许 `[2,2,3]`（下标 0,0,1） | 每个组合恰好一条路径 |

一条路径上选的数字集合确定了，下标单调的走法**只有一种**——把组合与路径一一对应起来，重复自然消失。

### 2.3 剪枝：排序 + 剩余量刹车

候选数组排序后有一个免费的优化：枚举到 `candidates[j] > rest` 时直接 `break`——后面的数更大，全都超了。这让"从 39 凑 40"这类例子几乎不分支。不排序也能做（把 `break` 换成 `continue`），但排序一次 $O(n \log n)$ 换取每层提前止损，稳赚。

汇总两个终止条件：

- `rest == 0`：收获当前 `path`；
- `rest < 0`（或排序后遇到 `c > rest`）：此路不通。

### 2.4 逐步走一遍：candidates = [2,3,6,7], target = 7

先排序（本题已有序）。树中每层从当前 `start` 开始尝试：

```
rest=7
├─ 选 2 → rest=5
│   ├─ 选 2 → rest=3
│   │   ├─ 选 2 → rest=1 → 2>1，break
│   │   ├─ 选 3 → rest=0 ✓ 收 [2,2,3]
│   │   └─ (6,7 均 > 3，break)
│   ├─ 选 3 → rest=2 → 3>2，break
│   └─ (6,7 > 5，break)
├─ 选 3 → rest=4
│   ├─ 选 3 → rest=1 → break
│   └─ (6,7 > 4，break)
├─ 选 6 → rest=1 → break
└─ 选 7 → rest=0 ✓ 收 [7]
```

只有两条路径活到 `rest == 0`，输出 `[[2,2,3],[7]]`，与示例一致。注意第一层选 2 之后第二层仍允许选 2（`start` 不加一），这正是"可重复选取"的体现。

---

## 3. 代码实现

```cpp
class Solution {
    vector<vector<int>> ans;
    vector<int> path;

    void dfs(vector<int>& candidates, int start, int rest) {
        if (rest == 0) {                 // 恰好凑满，收获组合
            ans.push_back(path);
            return;
        }
        for (int j = start; j < (int)candidates.size(); j++) {
            if (candidates[j] > rest)    // 已排序，后面的更大，整枝剪掉
                break;
            path.push_back(candidates[j]);   // 选择
            dfs(candidates, j, rest - candidates[j]);  // start=j：允许重复选
            path.pop_back();                 // 撤销
        }
    }

public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());   // 剪枝的前提
        dfs(candidates, 0, target);
        return ans;
    }
};
```

三个细节：

1. **递归传 `j` 而不是 `j + 1`**——这是本题与"元素不可重复"类组合题的唯一差别，`j+1` 就变成了 LeetCode 40 的去重形态的一半；
2. `rest == 0` 的判断放在循环外，靠的是"每层只选 $\le rest$ 的数"，因此永远不会出现 `rest < 0` 的递归调用；
3. 排序放在入口只做一次；`break` 依赖有序性，若无序只能 `continue`，剪枝力骤减。

---

## 4. 复杂度分析

- **时间复杂度**：最坏 $O(n^{target/\min a_i})$ 级别。搜索树的每个内部节点至多 $n$ 个分支，深度上界是 $target / \min(candidates) \le 40$（全选 $1$ 时最深，但本题 $target \le 40$，实测规模极小）。更精细的上界是答案总数乘以路径长度，均是被题目约束钉死的小常数。
- **空间复杂度**：$O(target / \min a_i)$（不计输出），即递归栈与 `path` 的深度；排序额外 $O(\log n)$ 栈空间。

---

## 5. 常见疑问

**Q1：`start` 防重复的直觉到底是什么？**

一句话：**规定组合内部的枚举顺序，让每个多重集只对应一条路径**。`start` 强制"后选的数在数组中不早于先选的数"，等价于只生成有序组合。它与"先算出所有排列再去重"相比不只是常数优化——后者要付出 $k!$ 倍的路径代价，40 这个规模下可能根本搜不完。

**Q2：为什么可以用 `break` 而不是 `continue`？**

因为排序后候选数组有序，`candidates[j] > rest` 意味着 $j$ 之后的所有数都大于 `rest`，整段区间整体失效，跳出循环即可。若数组无序，超标的数后面仍可能跟小的，只能 `continue` 逐个试探——剪枝从"砍一段"退化为"砍一个"。

**Q3：会不会搜出无限深的路径？**

不会。`candidates[i] \ge 1` 保证了每选一个数 `rest` 至少减 1，路径长度被 $target$ 封顶；又因为遇到 `candidates[j] > rest` 直接剪枝，递归在 `rest` 减到 $0$ 之前每层都严格递减，深度 $\le target$。但如果题目允许 $0$ 存在（本题不允许），`start` 方案会原地打转，需要额外处理。

---

## 6. 相关题目与模板

**可复用模板**（组合枚举 + 可重复选取 + 排序剪枝）：

```cpp
void dfs(int start, int rest) {
    if (rest == 0) { ans.push_back(path); return; }
    for (int j = start; j < n; j++) {
        if (a[j] > rest) break;        // 有序前提下的整枝剪掉
        path.push_back(a[j]);
        dfs(j, rest - a[j]);           // j：可重复；j+1：每个只用一次
        path.pop_back();
    }
}
```

一个极其有用的记忆点：**递归传 `j` 还是 `j+1`，一道组合题的两种形态就分开了**——传 `j` 是"可重复选取"（本题），传 `j+1` 是"每个元素至多选一次"。

**同类题目：**

- [LeetCode 40. 组合总和 II](https://leetcode.cn/problems/combination-sum-ii/)：元素可重复、每个只能用一次，在 `j+1` 之上还要"同层跳过相等元素"
- [LeetCode 77. 组合](https://leetcode.cn/problems/combinations/)：最裸的组合枚举，只传 `j+1`
- [LeetCode 216. 组合总和 III](https://leetcode.cn/problems/combination-sum-iii/)：固定个数的组合枚举
- [LeetCode 322. 零钱兑换](https://leetcode.cn/problems/coin-change/)：只求最少硬币数不求方案——同样的决策结构换成动态规划，对照着看能体会"要方案用回溯、要最值用 DP"

**一句话总结**：组合枚举防重复的核心是 **`start` 参数把路径与组合一一对应**，排序 + `break` 则让每层在"余额不足"时整枝止损。
