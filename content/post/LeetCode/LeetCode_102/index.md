---
title: "LeetCode 102: 二叉树的层序遍历"
date: 2026-11-09
math: true
tags:
  - LeetCode
  - 树
  - 广度优先搜索
categories:
  - LeetCode
description: "LeetCode 102 二叉树的层序遍历题解（广度优先搜索）"
---

# LeetCode 102. 二叉树的层序遍历

> 题目链接：[https://leetcode.cn/problems/binary-tree-level-order-traversal/](https://leetcode.cn/problems/binary-tree-level-order-traversal/)  
> 难度：中等  
> 标签：树、广度优先搜索

---

## 1. 题目描述

给你二叉树的根节点 `root`，返回其节点值的**层序遍历**（level order traversal）：即**逐层地、从左到右**访问所有节点，并按层分组返回。

与前序/中序/后序遍历返回一条扁平序列不同，本题的答案是一个"数组的数组"——同一层的节点值放在同一个子数组里，子数组按从上到下的顺序排列。

**数据范围：**

- 树中节点数目 $0 \le n \le 2000$
- $-1000 \le Node.val \le 1000$

**示例 1：**

```
输入：root = [3,9,20,null,null,15,7]

    3
   / \
  9  20
    /  \
   15   7

输出：[[3],[9,20],[15,7]]
```

**示例 2：**

```
输入：root = [1]
输出：[[1]]
```

**示例 3：**

```
输入：root = []
输出：[]
```

---

## 2. 思路分析

### 2.1 从需求倒推：为什么是队列

"逐层、从左到右"这六个字规定了访问顺序：先访问离根近的，再访问离根远的；同层内先左后右。这正是**广度优先搜索 (BFS)** 的定义。而 BFS 的标配容器是**队列 (queue)**——先进先出的性质恰好保证"先发现的节点先被扩展，其孩子（更深的层）排在所有同层节点之后"。

反过来看，栈或递归为什么不行：递归天然沿一条根到叶的路径走到底（深度优先），访问顺序是"纵向"的；要把纵向的访问重排成横向的层，就得引入额外的深度信息。

### 2.2 核心技巧：入队时"切层"

只用一个队列做 BFS，得到的是一条混合了所有层的流水；要按层分组，必须在**层与层的边界上切一刀**。最优雅的切法是**循环开始前记录本层节点数**：

```
while 队列非空:
    sz = 当前队列长度        // 恰好是本层的节点数
    重复 sz 次: 弹出一个节点，访问它，把非空孩子入队
```

为什么"入队前记录的长度"恰好等于本层节点数？因为队列里永远是"当前层剩下的 + 下一层已发现的"，而下一层的孩子只有处理当前层节点时才会入队——开始处理第 $i$ 层的那一刻，队列里的节点**全部属于第 $i$ 层**，一个不多（第 $i+1$ 层还没有任何节点入队），一个不少（第 $i$ 层节点都已入队）。

### 2.3 以示例 1 走一遍完整流程

树为 `[3,9,20,null,null,15,7]`：

| 轮次 | 层开始时队列 | sz | 弹出并访问 | 本轮孩子入队 | 收进答案 |
|-----|-------------|----|-----------|--------------|----------|
| 1 | [3] | 1 | 3 | 9, 20 | [3] |
| 2 | [9, 20] | 2 | 9（无孩子），20 | 15, 7 | [9, 20] |
| 3 | [15, 7] | 2 | 15（无），7（无） | — | [15, 7] |
| 4 | 空 | — | — | — | 结束 |

最终答案 `[[3],[9,20],[15,7]]`。注意第 2 轮：弹出 9 时 20 还在队里，20 的孩子 15、7 已经陆续入队——但因为 `sz = 2` 提前锁定，15、7 不会被本轮误访问；它们留给第 3 轮。**"先记 sz 再入队"这个顺序保证了层的纯净**。

### 2.4 备选：DFS 也能做——把层号递下去

另一种做法是**深度优先搜索 (DFS)** 带层号递归：`dfs(node, level)` 把 `node->val` 追加到 `ans[level]`（不足则先 push 空数组），再对左右孩子递归 `level + 1`。DFS 的访问顺序虽然是纵向的，但只要携带层号，值就能归进正确的层。

| 维度 | BFS + 队列 | DFS + 层号 |
|------|-----------|-----------|
| 代码量 | 需要手写队列循环 | 递归更短 |
| 空间 | 队列 $O(w)$（最宽层） | 调用栈 $O(h)$ |
| 层序语义 | 天然匹配，结果顺序无需思考 | 靠层号归位，同层顺序恰好也是从左到右 |
| 可扩展 | 模板通吃 103/107/199/429 | 改锯齿形等需重想 |

本题是 BFS 教学题，下面以 BFS 为主实现；DFS 版本的价值在于说明"层序信息 ≠ 必须用队列"。

---

## 3. 代码实现

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        if (root == nullptr) return ans;    // 空树返回 []，而非 [[]]

        queue<TreeNode*> q;
        q.push(root);

        while (!q.empty()) {
            int sz = q.size();              // 关键：先锁死本层节点数

            vector<int> level;
            level.reserve(sz);
            for (int i = 0; i < sz; i++) {  // 恰好处理完一整层
                TreeNode* node = q.front();
                q.pop();

                level.push_back(node->val); // 访问

                // 孩子入队：属于下一层，不影响本轮计数
                if (node->left)  q.push(node->left);
                if (node->right) q.push(node->right);
            }
            ans.push_back(move(level));     // 一层收工
        }
        return ans;
    }
};
```

实现细节：

1. **`sz` 必须在 for 循环之前取好**。若把循环条件写成 `i < q.size()`，本轮入队的孩子会撑大长度，同层与下一层混在一起，层序结构彻底崩坏——这是本题最高频的 WA 原因；
2. **空树返回空数组**。若漏掉开头的判空，循环体一次都不执行，自然返回 `[]`——其实不判空也正确，显式写出只是让意图清楚；
3. **`level.reserve(sz)` 与 `move`**：层大小已知，预留 + 移动语义省去 vector 的反复拷贝，$n = 2000$ 规模下是免费的好习惯；
4. 孩子判空后再入队，队列里就永远不会出现空指针，弹出后无需再判 `node != nullptr`。

---

## 4. 复杂度分析

- **时间复杂度**：$O(n)$。每个节点恰好入队一次、出队一次，每次进出做常数工作（取值、判空、push）。
- **空间复杂度**：$O(w)$，$w$ 为最宽一层的节点数，不计返回值。满二叉树时最后一层约有 $\lceil n/2 \rceil$ 个节点，故最坏 $O(n)$；这是 BFS 与 DFS（栈深 $O(h)$）的空间互补关系：**树越"胖"，BFS 队列越大、DFS 栈越浅；树越"瘦"，正好反过来**。

---

## 5. 常见疑问

**疑问一：把 `for (int i = 0; i < sz; i++)` 写成 `while (!q.empty())` 会怎样？**

功能上仍能遍历所有节点，但**层的信息丢了**：循环会在队列空时才停，答案变成一整条扁平序列 `[[3,9,20,15,7]]`（若内层只 push_back 单值）或干脆无法分组。层序遍历的灵魂不在 BFS 本身，而在"按层切分"——`sz` 就是那把刀。理解了这一点，103（锯齿形）、107（自底向上）、637（层平均值）全都只是在本模板上改一两行。

**疑问二：`q.size()` 返回的类型是 `size_t`（无符号），有什么坑？**

若写成 `for (int i = q.size() - 1; i >= 0; i--)` 之类依赖 `size()` 参与减法的式子，当队列为空时 `q.size() - 1` 会下溢成一个巨大的无符号数而不是 $-1$，循环失控。本题标准写法（先存进 `int sz`）天然避开；但凡见到 `q.size() - 1`、`v.size() - k` 都要立刻警觉"这个容器此刻可能为空吗"。

**疑问三：为什么入队顺序是先左后右，能不能反？**

层序定义要求**同一层内从左到右**，而队列先进先出，所以先入左孩子、再入右孩子。反过来（先右后左）得到的每层是右到左的顺序——恰是 103 锯齿形遍历偶数层的做法。可见"先左后右"不是实现细节，而是题目语义的一部分，改顺序就是在改题。

---

## 6. 相关题目与模板

**可复用模板**（二叉树 BFS 按层处理，一族题的母版）：

```cpp
queue<TreeNode*> q;
if (root) q.push(root);
while (!q.empty()) {
    int sz = q.size();                    // 切层：本层节点数
    for (int i = 0; i < sz; i++) {
        TreeNode* node = q.front(); q.pop();
        // ... 在这里按题意使用 node（收集 / 每层最后一个 / 求和 ...）
        if (node->left)  q.push(node->left);
        if (node->right) q.push(node->right);
    }
    // ... 在这里按层收尾（push 层数组 / 反转 / 记平均值 ...）
}
```

把"收尾"和"使用"两处换掉即可通吃一层题目：

| 题目 | 改动点 |
|------|--------|
| [LeetCode 107. 层序遍历 II](https://leetcode.cn/problems/binary-tree-level-order-traversal-ii/) | 最后 `reverse(ans.begin(), ans.end())` |
| [LeetCode 103. 锯齿形层序遍历](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/) | 偶数层把 `level` 反转，或交替使用左右入队顺序 |
| [LeetCode 199. 右视图](https://leetcode.cn/problems/binary-tree-right-side-view/) | 每层只收 `i == sz - 1` 那个节点 |
| [LeetCode 637. 层平均值](https://leetcode.cn/problems/average-of-levels-in-binary-tree/) | 层内求和除以 `sz` |
| [LeetCode 429. N 叉树的层序遍历](https://leetcode.cn/problems/n-ary-tree-level-order-traversal/) | 入队从"左右两孩子"改成"遍历 children 列表" |

**一句话总结**：层序遍历 = 队列 BFS + 用 `sz` 在队列上切层——这把刀切准了，整个"层序家族"就都是同一个模板的填空题。
