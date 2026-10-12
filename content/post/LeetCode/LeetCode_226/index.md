---
title: "LeetCode 226: 翻转二叉树"
date: 2026-11-06
math: true
tags:
  - LeetCode
  - 树
  - 深度优先搜索
  - 广度优先搜索
categories:
  - LeetCode
description: "LeetCode 226 翻转二叉树题解（递归交换）"
---

# LeetCode 226. 翻转二叉树

> 题目链接：[https://leetcode.cn/problems/invert-binary-tree/](https://leetcode.cn/problems/invert-binary-tree/)  
> 难度：简单  
> 标签：树、深度优先搜索、广度优先搜索

---

## 1. 题目描述

给你一棵二叉树的根节点 `root`，翻转这棵二叉树，并返回其根节点。

"翻转"指的是**镜像翻转**：对树中每一个节点，把它的左右子树互换位置。整棵树处理完后，得到的树与原树关于竖直中轴线对称。

**数据范围：**

- 树中节点数目 $0 \le n \le 100$
- $-100 \le Node.val \le 100$

**示例 1：**

```
输入：root = [4,2,7,1,3,6,9]

        4                  4
       / \                / \
      2   7      ==>     7   2
     / \ / \            / \ / \
    1  3 6  9          9  6 3  1

输出：[4,7,2,9,6,3,1]
```

**示例 2：**

```
输入：root = [2,1,3]
输出：[2,3,1]
```

**示例 3：**

```
输入：root = []
输出：[]
```

> 题外话：这道题因 Homebrew 作者 Max Howell 的著名吐槽而出名——他发推说谷歌面试拒绝了他："Google: 90% of our engineers use the software you wrote (Homebrew), but you can't invert a binary tree on a whiteboard so fuck off." 简单题不代表不值得会。

---

## 2. 思路分析

### 2.1 从单步操作入手：翻转 = 交换每个节点的左右孩子

先把"整棵树的镜像"拆小：观察示例 1 的箭头两侧，根 4 的孩子 2 和 7 互换了，节点 2 的孩子 1 和 3 互换了，节点 7 的孩子 6 和 9 互换了……规律很清楚：

> **镜像 = 对树中每一个节点执行一次 `swap(left, right)`。**

这是一个典型的"对每个节点做同样的事"的问题，不需要任何巧妙的算法——只要保证**每个节点恰好被访问一次，且访问时执行交换**。用什么顺序访问都行，于是有了 DFS 与 BFS 两个写法。

### 2.2 递归版：在"前序位置"交换

用**深度优先搜索 (DFS)** 遍历整棵树，对每个节点：先交换它的左右孩子，再递归处理（交换后的）新孩子。以示例 2 的 `[2,1,3]` 走一遍：

| 步骤 | 当前节点 | 操作 | 树的状态（层序） |
|-----|---------|------|-----------------|
| 1 | 2 | 交换孩子 1, 3 | [2,3,1] |
| 2 | 3（原右，现左） | 无孩子，无事发生 | [2,3,1] |
| 3 | 1（原左，现右） | 无孩子，无事发生 | [2,3,1] |

递归返回时整棵树已翻转完毕。注意第 2、3 步递归进入的是**交换后**的子树——但由于左右两侧各自递归，交换不影响子树内部的翻转，先交换还是先递归都正确（唯一要避开的是"中序"式的半吊子顺序，见第 5 节）。

### 2.3 迭代版：BFS 或显式栈

不习惯递归，可以显式用队列做**广度优先搜索 (BFS)**：根入队；每轮弹出队首，交换其左右孩子，再把非空的孩子压入队中。队列空即完成。

两种写法对比：

| 维度 | 递归 DFS | 迭代 BFS（队列）/ DFS（栈） |
|------|---------|---------------------------|
| 代码量 | 最短（3 行核心） | 稍长，需手写循环 |
| 空间 | 调用栈 $O(h)$ | 队列/栈 $O(w)$（$w$ 为最宽层） |
| 溢出风险 | 深树爆栈（本题 $n \le 100$ 无虞） | 无 |
| 可扩展性 | 改写成其他遍历顺序容易 | 天然"按层"，方便统计层信息 |

本题 $n \le 100$，怎么写都过；面试中先给递归版讲清思路，再主动提一句"深树可用迭代避免爆栈"，表现会更好。

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
    TreeNode* invertTree(TreeNode* root) {
        if (root == nullptr) return nullptr;   // 空树 / 递归出口

        // 前序位置：先交换当前节点的左右孩子
        TreeNode* tmp = root->left;
        root->left = root->right;
        root->right = tmp;

        // 再分别递归翻转两棵子树
        invertTree(root->left);
        invertTree(root->right);

        return root;    // 原地修改，返回根即可
    }
};
```

实现细节：

1. **原地修改**：交换的是指针本身，不需要新建任何节点，返回传入的 `root` 即可；
2. **交换必须在解引用 `root->left` 之前完成判空**——出口子句保证了 `root` 非空；
3. 若用 `#include <utility>`，前序位置三行可简写成 `swap(root->left, root->right);`。

迭代 BFS 版核心骨架（同样可提交）：

```cpp
TreeNode* invertTree(TreeNode* root) {
    if (root == nullptr) return nullptr;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        TreeNode* node = q.front(); q.pop();
        swap(node->left, node->right);          // 访问时交换
        if (node->left)  q.push(node->left);
        if (node->right) q.push(node->right);
    }
    return root;
}
```

---

## 4. 复杂度分析

- **时间复杂度**：$O(n)$。每个节点恰好入队/入栈一次、被交换一次，总工作量与节点数线性相关。
- **空间复杂度**：递归版为调用栈深度 $O(h)$，$h$ 是树高，最坏（链状）$O(n)$、最好（平衡）$O(\log n)$；BFS 版为队列容量 $O(w)$，$w$ 是最宽一层的节点数，满二叉树时约 $n/2$，即最坏 $O(n)$。

---

## 5. 常见疑问

**疑问一：为什么"中序"式的递归顺序会出错？**

设想"模仿中序"的写法：先递归翻转左子树，再交换左右孩子，最后**递归翻转右子树**。问题出在最后一步——交换之后，`root->right` 指向的已经是**刚翻完的原左子树**，再翻一次等于翻回去；而原右子树（此刻挂在左边）一次都没被翻。反例：对 `[4,2,7]` 执行该顺序，2 被翻两次回到原位，7 从未被翻。结论：**交换会改变"左右"指针的指向，交换之后不能再按旧语义递归**；用"前序先交换再递归"或"后序先递归再交换"最不容易错。

**疑问二：翻转后返回的是原树还是新树？**

是**原树**，且是原地修改——所有节点对象都没变，变的只是 `left`/`right` 指针的指向。所以调用方若在翻转前持有某个内部节点的指针，翻转后从该指针看到的子树也是镜像的。理解这一点对做 101（对称二叉树）有帮助：把一棵树复制后翻转，再和原树比较，就是对称性的一个（虽然不是最优的）判定方法。

**疑问三：单节点或空树要不要特判？**

不用。空树在递归出口直接返回 `nullptr`；单节点交换两个空指针，什么也没变，逻辑天然正确。"边界情况交给递归出口处理"正是树形递归优雅的地方——主逻辑里只写非空节点的事。

---

## 6. 相关题目与模板

**可复用模板**（"访问每个节点并做一件事"的两种骨架）：

```cpp
// DFS：visit() 放前序、后序皆可，本题做的事是 swap
void dfs(TreeNode* node) {
    if (node == nullptr) return;
    visit(node);          // 例如 swap(node->left, node->right)
    dfs(node->left);
    dfs(node->right);
}

// BFS：队列逐层扩散
queue<TreeNode*> q;
if (root) q.push(root);
while (!q.empty()) {
    TreeNode* node = q.front(); q.pop();
    visit(node);
    if (node->left)  q.push(node->left);
    if (node->right) q.push(node->right);
}
```

**同类题目：**

- [LeetCode 101. 对称二叉树](https://leetcode.cn/problems/symmetric-tree/)：翻转 + 比较的思路可以解，但更优雅的是双指针同步递归
- [LeetCode 100. 相同的树](https://leetcode.cn/problems/same-tree/)：另一道"对每个节点做同样判断"的入门题，和本题并称递归三件套
- [LeetCode 95. 不同的二叉搜索树 II](https://leetcode.cn/problems/unique-binary-search-trees-ii/)：从"翻转"到"构造"的进阶，同样大量使用递归分治
- [LeetCode 617. 合并二叉树](https://leetcode.cn/problems/merge-two-binary-trees/)：对两棵树同时递归，练习双树递归的指针操作

**一句话总结**：翻转二叉树 = 对每个节点交换左右孩子——把"整树问题"拆成"单节点操作 + 遍历"，是所有树形递归题的第一课。
