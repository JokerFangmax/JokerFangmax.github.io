---
title: "LeetCode 98: 验证二叉搜索树"
date: 2026-11-11
math: true
tags:
  - LeetCode
  - 树
  - 深度优先搜索
  - 二叉搜索树
categories:
  - LeetCode
description: "LeetCode 98 验证二叉搜索树题解（中序遍历与上下界剪枝）"
---

# LeetCode 98. 验证二叉搜索树

> 题目链接：[https://leetcode.cn/problems/validate-binary-search-tree/](https://leetcode.cn/problems/validate-binary-search-tree/)  
> 难度：中等  
> 标签：树、深度优先搜索、二叉搜索树

---

## 1. 题目描述

给你一个二叉树的根节点 `root`，判断其是否是一棵**有效的二叉搜索树 (Binary Search Tree, BST)**。

有效 BST 的**定义**：

- 节点的左子树只包含**严格小于**该节点值的节点；
- 节点的右子树只包含**严格大于**该节点值的节点；
- 左右子树也必须分别是 BST。

**数据范围：**

- 树中节点数目范围 $[1, 10^4]$
- $-2^{31} \le Node.val \le 2^{31} - 1$

**示例：**

```
输入一：
        5
       / \
      1   4
         / \
        3   6
输出：false
解释：根的右子树里出现了 3，3 < 5，违反"右子树所有节点都大于根"。

输入二：
    2
   / \
  1   3
输出：true
```

---

## 2. 思路分析

### 2.1 暴力：逐节点检查"直接孩子"——错在哪

最直观的写法：对每个节点检查 `left->val < root->val < right->val`，再递归检查左右子树。

```
        5
       / \
      3   7
     / \
    1   6      <- 6 在 3 的右子树里，3 < 6 局部合法
```

节点 3 与其右孩子 6 满足 $3 < 6$，但 6 同时位于根 5 的**左子树**里，$6 > 5$ 违反了 BST 定义。也就是说：

> **BST 的约束是从根到叶逐层累积的，每个节点必须落在由祖先们围出的开区间内，而不仅仅是比父亲大/小。**

局部检查只看到父子关系，看不到祖先传下来的约束，所以必然出错。

### 2.2 优化一：中序遍历应严格递增

换个角度。BST 最经典的性质：**中序遍历 (inorder traversal) 得到的序列严格递增**。

为什么？中序遍历按"左 - 根 - 右"访问，而在 BST 中，左子树所有值 $<$ 根 $<$ 右子树所有值，所以每个节点被访问时，它的值必然大于**中序意义下的前驱**——前驱要么在左子树中（值更小），要么是某个"最近左祖先"（值也更小）。反之，若中序序列严格递增，任何节点的左右子树中所有值都被区间夹住，定义成立。**这是充要条件**。

于是算法变成：中序遍历一遍，实时比较"当前值是否 $>$ 前一个访问的值"：

```
对上面的反例中序遍历：1, 3, 6, 5, 7
                        ^
                  6 -> 5 递减，立即返回 false
```

不需要存下整个序列，只要维护一个"上一个访问的值"，$O(1)$ 附加空间即可。

### 2.3 优化二：自顶向下传递上下界

另一个同样优雅的角度——把 2.1 的"局部检查"修好：既然约束是祖先累积的，那就**把祖先围出的区间作为参数传下去**。

节点 `x` 应满足 $lower < x.val < upper$；进入左子树时上界收紧为 $x.val$，进入右子树时下界收紧为 $x.val$。用例子走一遍：

| 节点 | 到达时约束 $(lower, upper)$ | 判定 | 传给左孩子 | 传给右孩子 |
|------|----------------------------|------|-----------|-----------|
| 5（根） | $(-\infty, +\infty)$ | 通过 | $(-\infty, 5)$ | $(5, +\infty)$ |
| 3（5 的左） | $(-\infty, 5)$ | 通过 | $(-\infty, 3)$ | $(3, 5)$ |
| 6（3 的右） | $(3, 5)$ | **失败**：$6 \ge 5$ | — | — |

6 在 $(3, 5)$ 之外被当场抓住——这正是局部检查漏掉的情形。

两种解法的对比：

| 方案 | 思路 | 空间（不含栈） | 额外注意点 |
|------|------|---------------|-----------|
| 中序 + 前驱比较 | 利用中序严格递增 | $O(1)$ | 用 `long long` 存前驱，或加"是否首个元素"标记 |
| 上下界递归 | 祖先区间逐层收紧 | $O(1)$ | 初始界取 ±LONG_MAX 而非 INT_MAX |

数据范围 $-2^{31} \le val \le 2^{31}-1$ 正好覆盖 `int` 全域，边界值 $-2^{31}$ 与 $2^{31}-1$ 本身可能出现在树里，所以初始上下界必须用比 `int` 更宽的 `long long`。

### 2.4 找到第一个违规点就提前返回

无论哪种解法，都不必遍历完整棵树：中序版一旦发现"当前 $\le$ 前驱"立刻短路返回；上下界版一旦发现节点越界也立刻返回。最坏（合法 BST 或违规在最后）仍是 $O(n)$，但实际测试中常常远快于全量遍历。

---

## 3. 代码实现

```cpp
class Solution {
    // 中序遍历：prev 记录中序前驱的值，hasPrev 标记是否已访问过节点
    long long prev = LLONG_MIN;
    bool hasPrev = false;

    bool inorder(TreeNode* root) {
        if (root == nullptr) return true;
        if (!inorder(root->left)) return false;   // 左子树违规，短路
        if (hasPrev && root->val <= prev)         // 必须严格大于前驱
            return false;
        prev = root->val;
        hasPrev = true;
        return inorder(root->right);              // 右子树
    }
public:
    bool isValidBST(TreeNode* root) {
        prev = LLONG_MIN;
        hasPrev = false;
        return inorder(root);
    }
};
```

等价的上下界写法（两版任选其一提交）：

```cpp
class Solution {
    bool check(TreeNode* root, long long lo, long long hi) {
        if (root == nullptr) return true;
        if (root->val <= lo || root->val >= hi) return false;  // 越界即违规
        return check(root->left,  lo, root->val)               // 上界收紧
            && check(root->right, root->val, hi);              // 下界收紧
    }
public:
    bool isValidBST(TreeNode* root) {
        return check(root, LLONG_MIN, LLONG_MAX);
    }
};
```

几个实现细节：

1. **`prev` 用 `long long` 初始化为 `LLONG_MIN`**。若初始化成 `INT_MIN`，树里真实存在值 $-2^{31}$ 时会误判（以为它 $\le$ 前驱）。用 `hasPrev` 标记可以彻底绕开哨兵值问题。
2. **比较必须用 $\le$ 而非 $<$**。定义要求**严格**大于/小于，出现相等（如两个孩子值相同）即违规。
3. **短路返回**：`&&` 与提前 `return false` 保证找到第一个违规点后不再深入，最坏 $O(n)$ 不变。

---

## 4. 复杂度分析

- **时间复杂度**：$O(n)$。每个节点至多访问一次，每个节点上的判断是 $O(1)$；找到违规点后提前退出，最好情况 $O(h)$。
- **空间复杂度**：$O(h)$，其中 $h$ 为树高。递归栈深度等于树高，最好（平衡树）$O(\log n)$，最坏（退化为链）$O(n)$；迭代版用显式栈同理。

两种解法复杂度完全相同，中序版揭示了 BST 的本质性质，上下界版训练"把约束作为参数传递"的思维方式，面试中都值得讲。

---

## 5. 常见疑问

**疑问一：为什么 `root->val > prev` 不够，必须同时管住整个子树？**

这就是 2.1 反例的教训：BST 定义是**全局**的（"左子树**所有**节点小于根"），而父子比较是**局部**的。违规节点 6 与它的父亲 3 关系合法，违规的是它与祖先 5 的关系。中序递增和上下界传递之所以正确，正是因为它们都携带了完整的历史信息：前者通过"前驱"隐式编码了整棵左子树的最大值，后者把祖先区间显式地传了下去。

**疑问二：用 `INT_MIN`/`INT_MAX` 当初始上下界为什么会 WA？**

树里可能出现值为 $-2^{31}$ 或 $2^{31}-1$ 的节点（数据范围明确写了）。若初始区间是 $[INT\_MIN, INT\_MAX]$ 的开区间，值为 $-2^{31}$ 的根会被判"越下界"而误报 false。解决办法两选一：把界放宽到 `LLONG_MIN`/`LLONG_MAX`，或者把参数改成"传指针、`nullptr` 表示无界"。中序版的 `prev` 哨兵同理。

**疑问三：中序遍历为什么保证"前驱"恰好是需要比较的对象？**

中序前驱是按访问顺序紧挨着当前节点的前一个。若当前节点的左子树非空，前驱是左子树的最右节点（左子树最大值），当前值必须大于它；若左子树为空，前驱是最近的"左祖先"（即当前节点位于其右子树里的最低祖先），当前值必须大于它。两种情况合起来恰好覆盖了"所有必须小于当前节点的节点"中最大的那个——只需与前驱比较一次，就等价于和全部约束比较。

---

## 6. 相关题目与模板

**可复用模板**（中序遍历 + 前驱哨兵，是 BST 类问题的通用骨架）：

```cpp
// BST 中序遍历，按访问顺序拿到严格递增序列
long long prev = LLONG_MIN;
bool inorder(TreeNode* root) {
    if (!root) return true;
    if (!inorder(root->left)) return false;
    if (root->val <= prev) return false;   // 违规判断按题意修改
    prev = root->val;
    return inorder(root->right);
}
```

**方法论总结**：BST 三件套——(1) 中序序列有序；(2) 节点值受祖先区间约束；(3) 左子树最大值是中序前驱、右子树最小值是中序后继。绝大多数 BST 判定/搜索题都是这三条的排列组合。

**同类题目：**

- [LeetCode 230. 二叉搜索树中第 K 小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/)：中序遍历数到第 $k$ 个即答案（本系列下一篇）
- [LeetCode 501. 二叉搜索树中的众数](https://leetcode.cn/problems/find-mode-in-binary-search-tree/)：中序遍历中"连续相等段"计数
- [LeetCode 530. 二叉搜索树的最小绝对差](https://leetcode.cn/problems/minimum-absolute-difference-in-bst/)：最小差只可能出现在中序相邻两数之间，套用前驱模板
- [LeetCode 99. 恢复二叉搜索树](https://leetcode.cn/problems/recover-binary-search-tree/)：中序序列中找"逆序对"定位被交换的两个节点，本题的加强版
- [LeetCode 235. 二叉搜索树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/)：利用"祖先区间"思想的搜索版

**一句话总结**：BST 的约束是祖先逐层累积的**区间**，不是局部父子比较——中序严格递增与上下界传递是同一事实的两种表述。
