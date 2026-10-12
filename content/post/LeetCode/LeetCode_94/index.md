---
title: "LeetCode 94: 二叉树的中序遍历"
date: 2026-11-04
math: true
tags:
  - LeetCode
  - 栈
  - 树
  - 深度优先搜索
categories:
  - LeetCode
description: "LeetCode 94 二叉树的中序遍历题解（显式栈与 Morris 遍历）"
---

# LeetCode 94. 二叉树的中序遍历

> 题目链接：[https://leetcode.cn/problems/binary-tree-inorder-traversal/](https://leetcode.cn/problems/binary-tree-inorder-traversal/)  
> 难度：简单  
> 标签：栈、树、深度优先搜索

---

## 1. 题目描述

给定一个二叉树的根节点 `root`，返回它的**中序遍历 (inorder traversal)**——即按照"**左子树 → 根 → 右子树**"的顺序访问所有节点，得到的节点值序列。

**进阶**：递归法很简单，你可以用**迭代**法完成吗？空间能压到 $O(1)$ 吗？

**数据范围：**

- 树中节点数目在范围 $[0, 100]$ 内
- $-100 \le Node.val \le 100$

**示例：**

```
输入：
        1
       / \
      2   3
     / \
    4   5

输出：[4, 2, 5, 1, 3]
```

---

## 2. 思路分析

### 2.1 递归：三行写完，但它"作弊"了

```cpp
void inorder(TreeNode* root, vector<int>& res) {
    if (!root) return;
    inorder(root->left, res);   // 左
    res.push_back(root->val);   // 根
    inorder(root->right, res);  // 右
}
```

递归版之所以简单，是因为**系统调用栈**替你保管了"回来的路"：递归深入左子树时，当前节点被自动压入栈帧，左子树返回后从栈帧里恢复现场，接着访问根、再去右子树。

但它有两个不满足进阶要求的地方：空间是 $O(h)$ 的调用栈（$h$ 为树高），且树退化成链、$n$ 达到 $10^5$ 量级时栈深 $10^5$ 可能**爆栈**（本题 $n \le 100$ 安全，但面试官会追问）。

### 2.2 迭代：把系统栈换成显式栈

把递归的过程手工模拟出来。核心问题只有一个：**什么时候访问节点，什么时候下钻？**

中序的规则：一个节点要等它的**整棵左子树全部处理完**才能轮到自己。所以遇到节点不能立刻访问，先压栈、向左走；左边走到头（`cur == null`）时，栈顶就是"左子树已完工"的节点——弹出、访问、转向右子树。

状态机只有两拍：

| 当前状态 | 动作 | 说明 |
|----------|------|------|
| `cur != null` | `cur` 压栈，`cur = cur->left` | 左边还没钻完，先欠着 |
| `cur == null` 且栈非空 | 弹栈顶、访问，`cur = 栈顶->right` | 欠账还上了，换右边 |
| `cur == null` 且栈空 | 结束 | 所有欠账还清 |

**用示例树走一遍**（栈从底到顶，箭头是当前 `cur`）：

| 步 | 动作 | 栈（底→顶） | 输出 |
|----|------|-------------|------|
| 1 | 1 压栈，向左 | [1] | |
| 2 | 2 压栈，向左 | [1, 2] | |
| 3 | 4 压栈，向左 | [1, 2, 4] | |
| 4 | 左边到头，弹 4 访问，右转 | [1, 2] | 4 |
| 5 | 4 无右子，弹 2 访问，右转 | [1] | 4, 2 |
| 6 | 走向 5：5 压栈向左 | [1, 5] | 4, 2 |
| 7 | 弹 5 访问，右转 | [1] | 4, 2, 5 |
| 8 | 弹 1 访问，右转 | [] | 4, 2, 5, 1 |
| 9 | 3 压栈向左 | [3] | 4, 2, 5, 1 |
| 10 | 弹 3 访问；栈空且 `cur` 空，结束 | [] | **4, 2, 5, 1, 3** |

### 2.3 Morris：把空间压到 O(1)

栈存的是"回去的路"。Morris 遍历 (Morris traversal) 的洞察是：**树里其实自带回路的原料**——每个节点有 `left`/`right` 两个指针，而遍历过程中大量指针处于"用完即弃"状态。不如**临时借用空指针搭回头路，用完再拆掉还原**。

规则：在每个节点 `cur` 找它的**中序前驱**（左子树中最右的节点）：

- 若前驱的 `right` 为空：`前驱->right = cur`（搭线索），`cur = cur->left`；
- 若前驱的 `right == cur`：说明左子树已通过线索回来过——拆线索、访问 `cur`、`cur = cur->right`；
- 无左子树：直接访问 `cur`，`cur = cur->right`。

每条边至多被走两次（一次下探、一次沿线索回来），时间仍 $O(n)$，空间 $O(1)$，代价是遍历期间树被临时改写、遍历完才复原。

---

## 3. 代码实现

```cpp
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> st;
        TreeNode* cur = root;

        while (cur || !st.empty()) {
            if (cur) {
                st.push(cur);          // 欠账：cur 的左子树还没处理
                cur = cur->left;       // 一路向左钻到底
            } else {
                cur = st.top(); st.pop();
                res.push_back(cur->val);   // 左子树完工，轮到根
                cur = cur->right;          // 转向右子树
            }
        }
        return res;
    }
};
```

几个实现细节：

1. **循环条件是 `cur || !st.empty()` 两个或**：`cur` 非空表示"还有没钻完的左边"，栈非空表示"还有没访问的欠账"。两者同时为空才是真正结束——只写其中一个都会提前收工或漏掉右子树。
2. **`else` 分支里 `cur = cur->right` 之后不判空**：`cur` 为空也没关系，下一轮循环会自动落入"压栈"或"弹栈"的对应分支，这正是两拍状态机的自洽之处。
3. **弹栈后访问的是 `cur->val`**（先弹后用），不要先置 `cur = cur->right` 再访问——那访问的就是右儿子了。

---

## 4. 复杂度分析

- **时间复杂度**：$O(n)$。每个节点恰好入栈一次、出栈一次（Morris 版则是每条边至多走两次，同样 $O(n)$）。
- **空间复杂度**：显式栈为 $O(h)$，$h$ 为树高——满二叉树时 $h = \log n$，链状退化时 $h = n$。Morris 遍历做到 $O(1)$。

| 解法 | 时间 | 空间 | 备注 |
|------|------|------|------|
| 递归 | $O(n)$ | $O(h)$ 调用栈 | 三行写完，可能爆栈 |
| 显式栈（本解） | $O(n)$ | $O(h)$ 显式栈 | 深度可控，进阶第一档 |
| Morris | $O(n)$ | $O(1)$ | 临时改树、访问完还原 |

---

## 5. 常见疑问

**Q1：为什么中序是"左根右"，这个顺序有什么特别的？**

对**二叉搜索树 (BST)**，中序遍历恰好按**升序**输出所有键——因为"左 < 根 < 右"的定义逐层向下传递。这不是巧合而是结构性的：[98. 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/)、[230. 二叉搜索树中第 K 小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/) 的标准解法都是"中序遍历 + 顺序检查"。记住这条性质：**见到 BST + 顺序问题，先想中序**。

**Q2：迭代版弹栈访问后，为什么是 `cur = cur->right` 而不是继续弹栈？**

弹栈访问节点 $x$ 的前提是"$x$ 的左子树全部完工"。访问完 $x$ 后，按"左根右"的剧本该轮到**$x$ 的右子树**——右子树内部还有自己的"左根右"结构要展开，所以先把 `cur` 指向右儿子，让下一轮循环从头跑状态机；若右子树为空，`cur == null`，下一轮自然回到弹栈分支。直接继续弹栈会跳过右子树——这是把前序的直觉带进中序的典型错误。

**Q3：Morris 遍历改了树的指针，会破坏树吗？**

不会，前提是按规则"原路拆回"：线索只在**第二次**回到前驱时存在，检测到 `前驱->right == cur` 就立刻置回 `nullptr`。遍历结束时所有线索都被拆除，树恢复原样。风险点在于提前 `return`（比如"找到第 K 小就收工"的题）——树还挂着线索就走了，调用方后续遍历会死循环。所以**用 Morris 写带剪枝的遍历时，退出前要手动清理剩余线索**，或者干脆换回栈版本。

---

## 6. 相关题目与模板

**三种遍历的迭代统一模板**（"标记法"：栈里放 `(节点, 是否已处理)`，处理过才访问）：

```cpp
// 先序：入栈顺序 right、left、自己(标记)，弹出标记项时访问
stack<pair<TreeNode*, bool>> st;
st.push({root, false});
while (!st.empty()) {
    auto [node, seen] = st.top(); st.pop();
    if (!node) continue;
    if (seen) res.push_back(node->val);     // 第二次弹出：访问
    else {
        st.push({node->right, false});      // 中序：left 先于自己入栈
        st.push({node->left, false});       // 先序：root 最先入栈
        st.push({node, true});              // 后序：root 最后入栈
    }
}
```

只调整三个 `push` 的相对顺序，先序 / 中序 / 后序共用一套骨架——比背三份不同代码省心得多。

**Morris 中序模板**（$O(1)$ 空间，BST 类题目首选）：

```cpp
TreeNode* cur = root;
while (cur) {
    if (!cur->left) {                   // 无左子：直接访问，右行
        visit(cur);
        cur = cur->right;
    } else {
        TreeNode* pre = cur->left;      // 找中序前驱：左子树最右节点
        while (pre->right && pre->right != cur) pre = pre->right;
        if (!pre->right) {              // 第一访：搭线索，左行
            pre->right = cur;
            cur = cur->left;
        } else {                        // 第二访：拆线索，访问，右行
            pre->right = nullptr;
            visit(cur);
            cur = cur->right;
        }
    }
}
```

**同类题目：**

- [LeetCode 144. 二叉树的前序遍历](https://leetcode.cn/problems/binary-tree-preorder-traversal/) / [145. 二叉树的后序遍历](https://leetcode.cn/problems/binary-tree-postorder-traversal/)：三兄弟，用统一模板一把带走
- [LeetCode 173. 二叉搜索树迭代器](https://leetcode.cn/problems/binary-search-tree-iterator/)：把本题的栈版本封装成类，`next()` 每次只走一步
- [LeetCode 98. 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/)：中序 + "严格递增"检查
- [LeetCode 230. 二叉搜索树中第 K 小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/)：中序走到第 $k$ 个即停

**一句话总结**：递归靠系统栈记路，迭代用显式栈还账，Morris 借空指针搭回头路——中序遍历的三种写法，本质是同一句"左根右"在不同空间约束下的三次翻译。
