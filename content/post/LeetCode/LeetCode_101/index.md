---
title: "LeetCode 101: 对称二叉树"
date: 2026-11-07
math: true
tags:
  - LeetCode
  - 树
  - 深度优先搜索
  - 广度优先搜索
categories:
  - LeetCode
description: "LeetCode 101 对称二叉树题解（双指针递归）"
---

# LeetCode 101. 对称二叉树

> 题目链接：[https://leetcode.cn/problems/symmetric-tree/](https://leetcode.cn/problems/symmetric-tree/)  
> 难度：简单  
> 标签：树、深度优先搜索、广度优先搜索

---

## 1. 题目描述

给你一棵二叉树的根节点 `root`，检查它是否**轴对称**（symmetric）：是否存在一条竖直中轴线，使得树沿这条轴折叠后完全重合。

注意这不仅是"结构对称"，**节点的值也必须对应相等**。递归定义的说法是：二叉树对称，当且仅当根的左子树与右子树互为镜像；而两棵树互为镜像，当且仅当它们的根值相等、且每棵树的左子树与另一棵树的右子树互为镜像。

**数据范围：**

- 树中节点数目 $1 \le n \le 100$
- $-100 \le Node.val \le 100$

**示例 1：**

```
输入：root = [1,2,2,3,4,4,3]

        1
       / \
      2   2
     / \ / \
    3  4 4  3

输出：true
```

**示例 2：**

```
输入：root = [1,2,2,null,3,null,3]

      1
     / \
    2   2
     \   \
      3   3

输出：false
解释：节点 3 挂的位置分别是"左的右"与"右的右"，折叠不重合。
```

---

## 2. 思路分析

### 2.1 顺着直觉走：判断"左子树自己对称"和"右子树自己对称"？

第一反应往往是递归：`isSymmetric(root) = isSymmetric(root->left) && isSymmetric(root->right)`。**这是错的**。对称性比较的是左右子树**互相之间**的镜像关系，而不是各自内部的关系。反例：

```
      1
     / \
    2   2
   /     \
  3       3
```

左子树 `[2,3]` 自己不对称，右子树 `[2,null,3]` 也不对称，但整棵树是对称的。这个反例说明：**对称性是"左树与右树"的二元关系，必须把两棵树一起传进递归**。

### 2.2 正确姿势：双指针同步递归

定义辅助函数 `mirror(l, r)`，回答"以 `l` 和 `r` 为根的两棵树是否互为镜像"。按定义展开：

- 两个都为空 → 镜像，`true`；
- 一空一非空 → 不镜像，`false`；
- 都非空 → 要求 $l.val = r.val$，**且** `mirror(l->left, r->right)`（外侧对外侧）**且** `mirror(l->right, r->left)`（内侧对内侧）。

关键是那个"交叉"：镜像折叠时，左树的**左**孩子对上右树的**右**孩子。这正是直觉最容易翻车的地方——写反成 `mirror(l->left, r->left)` 的话，示例 2 这种"值全对但挂错边"的用例就会误判为 true。

用示例 2 `[1,2,2,null,3,null,3]` 走一遍调用过程（记 `N` 为空节点）：

| 调用 | 内容 | 结果 | 说明 |
|------|------|------|------|
| mirror(左 2, 右 2) | 值相等，继续分解 | 待定 | 根的左右子树 |
| mirror(N, N) | 左 2 的左 vs 右 2 的右 | true | 双空 |
| mirror(3, N) | 左 2 的右 vs 右 2 的左 | **false** | 一空一非空 |

第三行失败，`&&` 短路，整棵树返回 false——正是"值都对但挂错了边"的场景。

### 2.3 迭代版：队列成对出队

递归本质上是系统在压栈，任何递归都能改成显式队列。做法：初始把 `(root->left, root->right)` 入队；每次**成对**取出两个节点，检查"双空跳过、单空返回 false、值不等返回 false"，再把 `(l->left, r->right)` 和 `(l->right, r->left)` 两对依次入队。队列空了还没失败，就是对称。

两种实现对比：

| 维度 | 递归 DFS | 迭代（队列成对） |
|------|---------|----------------|
| 代码量 | 更短，逻辑即定义 | 稍长，但把"对"显式化 |
| 空间 | 调用栈 $O(h)$ | 队列 $O(n)$ 最坏 |
| 提早退出 | 短路 `&&` 天然剪枝 | 值不等立刻 return，同样高效 |
| 通用性 | 改成 BFS 层序判断也容易 | 便于扩展到"迭代比较两棵树" |

本题 $n \le 100$，两者皆可；递归版更推荐——它是"把数学定义直接翻译成代码"的典范。

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
    // 判断以 l、r 为根的两棵子树是否互为镜像
    bool mirror(TreeNode* l, TreeNode* r) {
        if (l == nullptr && r == nullptr) return true;   // 双空：镜像
        if (l == nullptr || r == nullptr) return false;  // 单空：必不镜像
        return l->val == r->val                          // 根值相等
            && mirror(l->left,  r->right)                // 外侧 vs 外侧（交叉！）
            && mirror(l->right, r->left);                // 内侧 vs 内侧（交叉！）
    }
public:
    bool isSymmetric(TreeNode* root) {
        // 题目保证 n >= 1，root 不会为空
        return mirror(root->left, root->right);
    }
};
```

实现细节：

1. **两个判空子句覆盖全部四种空/非空组合**：双空、左空、右空、双非空，一路 if 下来不会漏，也不需要在别处再判空——这是双树递归的标准起手式；
2. **`&&` 的短路**是隐含的性能保障：值不等时后面两次递归直接不发生，最坏情况下也只会把"注定失败"的路径走到失败为止；
3. `mirror` 里**先比根值再比子树**，顺序照抄定义即可，不需要任何额外变量。

---

## 4. 复杂度分析

- **时间复杂度**：$O(n)$。每个节点至多作为"某个对子中的一员"被访问一次，每次访问做常数工作。最坏（完全对称的树）遍历全部节点；最坏（根部两子树根值就不等）只访问 $O(1)$ 个节点。
- **空间复杂度**：递归版调用栈深度等于树高 $h$，最坏链状 $O(n)$、平衡 $O(\log n)$。迭代版队列最坏要存一层节点对，满二叉树底层约 $n/2$ 个节点，即 $O(n)$。

---

## 5. 常见疑问

**疑问一：为什么不能分别翻转左右子树再比较？**

可以，但绕：把右子树整个翻转（226 题），再和左子树做"相同树"比较（100 题），两遍 $O(n)$ 下来也是线性。问题是它**把"镜像"拆成了"翻转 + 相等"两个动作**，多写一个函数、多走一遍树。双指针递归直接在遍历中完成交叉比对，一次 $O(n)$ 结束。前者胜在思路直白，后者胜在简洁——面试先讲前者建立直觉，再优化到后者，是漂亮的答题节奏。

**疑问二：`mirror(l->left, r->right)` 和 `mirror(l->right, r->left)` 写反了会怎样？**

写反成 `mirror(l->left, r->left)` 后，程序判定的是"两棵树**形状相等**"而不是镜像。示例 2（`[1,2,2,null,3,null,3]`）正好是为此准备的用例：按错误版本，左侧 `2` 的左是空、右侧 `2` 的左也是空，一路"相等"到最后返回 true，但正确答案是 false。**值相同、挂的位置镜像不镜像，只有交叉比较才能区分**——这也是本题最容易的 WA 点。

**疑问三：要不要处理 `root == nullptr`？**

数据范围保证 $n \ge 1$，理论上不必。但写 `if (root == nullptr) return true;` 也无妨——空树视为对称是合理约定，且让代码在面试验证时不依赖题目保证。注意返回值应是 `true` 而不是 `false`："什么都没有"折叠后依然重合。

---

## 6. 相关题目与模板

**可复用模板**（"两棵树同时递归"骨架，判断类问题的万能起手式）：

```cpp
bool twoTree(TreeNode* a, TreeNode* b) {
    if (!a && !b) return true;    // 双空
    if (!a || !b) return false;   // 单空
    // 此处 a、b 均非空，按需比较与递归：
    return a->val == b->val
        && twoTree(a->left,  b->right)   // 对称：交叉
        && twoTree(a->right, b->left);
    // 若是"相同树"（LeetCode 100），把两个 right 改成同侧即可
}
```

**同类题目：**

- [LeetCode 100. 相同的树](https://leetcode.cn/problems/same-tree/)：同侧比较版，和本题构成一组对照练习
- [LeetCode 226. 翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/)：镜像问题的"构造"版——本题是"判定"，它是"操作"
- [LeetCode 572. 另一棵树的子树](https://leetcode.cn/problems/subtree-of-another-tree/)：在每个节点上调用"相同树"判断，双树递归的套娃应用
- [LeetCode 95. 不同的二叉搜索树 II](https://leetcode.cn/problems/unique-binary-search-trees-ii/)：从判定走向构造，同样吃透递归定义

**一句话总结**：对称是"左树与右树"的二元关系——双指针交叉递归，`mirror(l.left, r.right)` 一写对，这道题就结束了。
