---
title: "LeetCode 104: 二叉树的最大深度"
date: 2026-11-05
math: true
tags:
  - LeetCode
  - 树
  - 深度优先搜索
  - 广度优先搜索
categories:
  - LeetCode
description: "LeetCode 104 二叉树的最大深度题解（深度优先搜索）"
---

# LeetCode 104. 二叉树的最大深度

> 题目链接：[https://leetcode.cn/problems/maximum-depth-of-binary-tree/](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)  
> 难度：简单  
> 标签：树、深度优先搜索、广度优先搜索

---

## 1. 题目描述

给出一棵二叉树的根节点 `root`，返回它的**最大深度**（maximum depth）。

二叉树的最大深度是指**从根节点到最远叶子节点的最长路径上的节点数**。注意官方定义数的是"节点数"而不是"边数"：只有一个根节点的树深度为 $1$，空树深度为 $0$。

**数据范围：**

- 树中节点数目 $0 \le n \le 10^4$
- $-100 \le Node.val \le 100$

**示例 1：**

```
输入：root = [3,9,20,null,null,15,7]

    3
   / \
  9  20
    /  \
   15   7

输出：3
解释：最长路径 3 -> 20 -> 15（或 3 -> 20 -> 7），共 3 个节点。
```

**示例 2：**

```
输入：root = [1,null,2]
输出：2
```

---

## 2. 思路分析

二叉树的递归解法有一个通用框架，值得先说清楚：**遍历（traverse）与分解（divide）**，即"自顶向下传信息"与"自底向上算答案"两种视角。这道题恰好两种视角都能做，对比着学效果最好。

### 2.1 视角一：遍历——自顶向下把深度"传"下去

想象自己沿着树往下走，每往下走一层，身上的"计数器"就加一。走到**叶子节点**（或者说空节点）时，用当前计数器更新全局最大值——这本质是一次**深度优先搜索 (DFS)** 的遍历：

```
void traverse(TreeNode* node, int depth) {
    if (node == nullptr) {
        ans = max(ans, depth);   // 空节点处结算
        return;
    }
    traverse(node->left,  depth + 1);
    traverse(node->right, depth + 1);
}
```

以示例 1 走一遍（`depth` 为到达该节点时的计数）：

| 到达节点 | depth | 说明 |
|---------|-------|------|
| 3       | 1     | 起点 |
| 9       | 2     | 叶子，结算 2 |
| 20      | 2     | |
| 15      | 3     | 叶子，结算 **3** |
| 7       | 3     | 叶子，结算 **3** |

答案为 $3$。这个写法可行，但要维护一个外部变量 `ans`，属于"回溯"的味道，代码状态不够纯粹。

### 2.2 视角二：分解——自底向上把答案"算"上来

换一个问题定义的方式：**整棵树的最大深度，和它的左右子树的最大深度，是什么关系？**

一句话：根节点的深度 = 左右子树深度的较大者 **加一**（那个"一"就是根本身）。写成递推：

$$depth(root) = \max(depth(left), depth(right)) + 1$$

边界条件：空树 $depth = 0$，递归到这里自然返回。

这就是**分解问题（分治）**的思路——先递归拿到子问题的答案，再在"后序位置"（左右子树都处理完之后）合并。它不需要外部变量，函数本身就是答案：

```cpp
int maxDepth(TreeNode* root) {
    if (root == nullptr) return 0;
    return max(maxDepth(root->left), maxDepth(root->right)) + 1;
}
```

还是示例 1，看返回值是怎么"冒上来的"（建议从最深处往上看）：

| 节点 | 左子树返回 | 右子树返回 | 本节点返回 |
|-----|-----------|-----------|-----------|
| 9   | 0         | 0         | 1         |
| 15  | 0         | 0         | 1         |
| 7   | 0         | 0         | 1         |
| 20  | 1         | 1         | 2         |
| 3   | 1         | 2         | **3**     |

### 2.3 视角三：广度优先搜索——一层一层往外扩

第三种想法来自另一个直观定义：**最大深度 = 层数**。用**广度优先搜索 (BFS)** 从根开始一层层扩散，用一个队列维护"当前层"的所有节点，每清空一层，层数加一，队列空了层数就是答案。

| 方法 | 递归/迭代 | 需要额外变量 | 本质 |
|------|----------|-------------|------|
| 遍历 + depth 参数 | 递归 | 全局 `ans` | 前序位置传下去 |
| 分解 + max 合并 | 递归 | 无 | 后序位置算上来 |
| BFS 按层计数 | 迭代 | 队列 + 计数器 | 逐层扩散 |

三种都是 $O(n)$，本题任选。分解版代码最短、最体现树形 DP 的思想，是首选；BFS 版的"按层处理"骨架是下一题层序遍历的核心，值得提前熟悉。

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
    int maxDepth(TreeNode* root) {
        // 边界：空树深度为 0，递归的出口
        if (root == nullptr) return 0;

        // 先递归求解两个子问题：左右子树各自的最大深度
        int leftDepth  = maxDepth(root->left);
        int rightDepth = maxDepth(root->right);

        // 后序位置合并：较大者 + 1（"1" 是根本身）
        return max(leftDepth, rightDepth) + 1;
    }
};
```

两个实现细节：

1. **先把递归调用结果存进局部变量**，而不是直接写进 `return max(maxDepth(...), maxDepth(...)) + 1`。两者结果相同，但前者结构清晰，方便日后扩展（比如 543 题要在这里插入对 `leftDepth + rightDepth` 的统计）。
2. **递归出口写在最前面**（guard clause），空指针直接返回 0，后面就可以放心解引用 `root`，不必再判空。

---

## 4. 复杂度分析

- **时间复杂度**：$O(n)$。每个节点恰好被访问一次，每个节点上只做常数次比较。
- **空间复杂度**：$O(h)$，其中 $h$ 为树高。递归调用栈的深度等于树高；最坏情况（链状树）$h = n$，为 $O(n)$；平衡树时 $h = O(\log n)$。注意**系统栈空间通常不计入"额外空间"的严格统计口径之争**，但面试中说清调用栈开销是加分项。

若采用 BFS 版本，空间是队列中最多一层的节点数，最坏（满二叉树最后一层）约 $\lceil n/2 \rceil$ 个，同样与树形有关。

---

## 5. 常见疑问

**疑问一：空树为什么返回 0 而不是 -1？**

因为本题定义深度为**路径上的节点数**：单节点树深度为 $1 = 0 + 1$，空树自然为 $0$。如果定义改成"边数"，单节点树是 $0$、空树就该返回 $-1$（543 题的直径版本就会遇到这个口径问题）。刷题时先看清题目数的是节点还是边，能避免一大类差一错误。

**疑问二：递归写法会不会栈溢出？**

$n \le 10^4$，最坏链状树递归深度 $10^4$，远低于一般的栈上限（约 $10^5 \sim 10^6$ 层），安全。但若 $n$ 放大到 $10^6$ 级别，链状树就会爆栈——那时必须改成 BFS 或用显式栈的迭代写法，这也是本题标签里带广度优先搜索的原因。

**疑问三：遍历版（带 depth 参数）和分解版到底该用哪个？**

能用分解就用分解：它没有外部状态，函数返回值即答案，天然可组合（树形 DP 的雏形）。只有当"答案必须依赖从根传下来的信息"（例如路径字符串、路径和上限）时，才不得不走遍历版。本题两者皆可，正好用来对照理解。

---

## 6. 相关题目与模板

**可复用模板**（"分解"式树形递归的通用骨架）：

```cpp
// 后序位置合并子问题：绝大多数"统计整棵树"的问题都套这个壳
int solve(TreeNode* root) {
    if (root == nullptr) return 0;          // 边界
    int l = solve(root->left);              // 左子问题
    int r = solve(root->right);             // 右子问题
    return merge(l, r, root->val);          // 后序位置合并
}
```

**同类题目：**

- [LeetCode 111. 二叉树的最小深度](https://leetcode.cn/problems/minimum-depth-of-binary-tree/)：把 `max` 换成 `min`？不行——单侧为空时会错误地返回 1，必须特判"有一侧为空"的情况，是本题最好的反思维练习
- [LeetCode 543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/)：在"分解"骨架的后序位置多统计一个 $\text{left} + \text{right}$ 的最大值
- [LeetCode 110. 平衡二叉树](https://leetcode.cn/problems/balanced-binary-tree/)：同样自底向上返回高度，顺便检测失衡
- [LeetCode 559. N 叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-n-ary-tree/)：把"两个孩子取 max"推广成"所有孩子取 max"

**一句话总结**：树的最大深度 = 左右子树最大深度的较大者加一——学会在**后序位置合并子问题**，一整族树形 DP 题就从这里出发。
