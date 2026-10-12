---
title: "LeetCode 105: 从前序与中序遍历序列构造二叉树"
date: 2026-11-15
math: true
tags:
  - LeetCode
  - 树
  - 数组
  - 哈希表
  - 分治
categories:
  - LeetCode
description: "LeetCode 105 从前序与中序遍历序列构造二叉树题解（分治 + 哈希表定位根节点）"
---

# LeetCode 105. 从前序与中序遍历序列构造二叉树

> 题目链接：[https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)  
> 难度：中等  
> 标签：树、数组、哈希表、分治

---

## 1. 题目描述

给出一棵二叉树的**前序遍历 (preorder traversal)** 与**中序遍历 (inorder traversal)** 结果（节点值互不相同），请构造出这棵二叉树并返回其根节点。

两个序列足以唯一确定一棵树的关键在于：前序遍历的第一个元素一定是根，而在中序遍历中找到这个根的位置，就能把序列切成「左子树 | 根 | 右子树」三段，进而递归处理。

**数据范围：**

- $1 \le \text{preorder.length} = \text{inorder.length} \le 3000$
- $-3000 \le \text{preorder}[i], \text{inorder}[i] \le 3000$
- `preorder` 和 `inorder` 均**无重复元素**
- 输入保证是一对合法的遍历序列

**示例：**

```
输入：preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]

输出：[3,9,20,null,null,15,7]

对应二叉树：
        3
       / \
      9  20
         / \
        15  7
```

---

## 2. 思路分析

### 2.1 暴力：每层递归都线性扫描找根

分治的骨架是显然的：

1. `preorder` 的第一个元素 `pre[pl]` 是当前子树的根；
2. 在 `inorder` 中找到 `pre[pl]` 的位置 `idx`，则 `inorder[il..idx-1]` 是左子树、`inorder[idx+1..ir]` 是右子树；
3. 由左子树大小 `idx - il` 可算出 `preorder` 中左、右子树各自的区间，递归构造。

问题出在第 2 步：每次都要在 `inorder` 里**线性扫描**找根。树退化成链时，每层扫描 $O(n)$、共 $n$ 层，总时间 $O(n^2)$。$n = 3000$ 时约 $4.5 \times 10^6$ 次比较，本题能过，但数据范围再放大一个量级就会超时——这个瓶颈必须消掉。

### 2.2 优化一：哈希表 $O(1)$ 定位根节点

值互不相同，这正是题目条件埋下的伏笔：**把「值 → 中序下标」的映射预处理进哈希表 (hash table)**，之后每次查根的位置都是 $O(1)$。

`pos[v]` 存值为 `v` 的节点在中序遍历中的下标，找根从 $O(n)$ 降到 $O(1)$，总时间降到 $O(n)$。

### 2.3 优化二：传下标区间，不切数组

初学者常见的写法是每层递归都把 `preorder`、`inorder` 切成两个新 `vector` 传下去。逻辑没错，但每层拷贝要 $O(\text{子树大小})$，$n$ 层下来又退化成 $O(n^2)$，还产生大量临时对象。

正确姿势是**始终传原数组 + 四个下标** `(pl, pr, il, ir)`，表示当前子树对应 `preorder[pl..pr]` 与 `inorder[il..ir]`，切分只靠下标运算，零拷贝。

### 2.4 用示例走一遍关键步骤

`preorder = [3,9,20,15,7]`，`inorder = [9,3,15,20,7]`，`pos = {9:0, 3:1, 15:2, 20:3, 7:4}`。

初始调用 `build(0, 4, 0, 4)`，每一步的推理：

| 步骤 | preorder 区间 | inorder 区间 | 根 | 根在中序位置 | 左子树大小 | 划分结果 |
|---|---|---|---|---|---|---|
| 1 | `[3,9,20,15,7]` | `[9,3,15,20,7]` | 3 | $pos[3]=1$ | $1-0=1$ | 左 `[9]` / 右 `[15,20,7]` |
| 2（左） | `[9]` | `[9]` | 9 | — | 0 | 叶子，左右为空 |
| 3（右） | `[20,15,7]` | `[15,20,7]` | 20 | $pos[20]=3$ | $3-2=1$ | 左 `[15]` / 右 `[7]` |
| 4 | `[15]` | `[15]` | 15 | — | 0 | 叶子 |
| 5 | `[7]` | `[7]` | 7 | — | 0 | 叶子 |

两个推导细节值得盯着看：

- **左子树在中序里占 `il..idx-1`**，长度 `idx - il`；
- 左子树在 preorder 里紧跟在根后面，占 `pl+1 .. pl+(idx-il)`；右子树占 `pl+(idx-il)+1 .. pr`。两条「跨序列换算」全靠**左子树大小**这个桥梁，区间端点的差一错误是本题最常见的 WA 来源。

---

## 3. 代码实现

```cpp
class Solution {
    unordered_map<int, int> pos;   // 值 -> 中序下标，O(1) 定位根

    // 用 preorder[pl..pr] 与 inorder[il..ir] 构造子树并返回根
    TreeNode* build(vector<int>& pre, vector<int>& in,
                    int pl, int pr, int il, int ir) {
        if (pl > pr) return nullptr;          // 空区间 -> 空树

        TreeNode* root = new TreeNode(pre[pl]); // 前序第一个必是根
        int idx = pos[pre[pl]];                 // 根在中序中的位置
        int leftSize = idx - il;                // 左子树节点数

        // 左子树：preorder 紧跟根的 leftSize 个；inorder 在根左侧
        root->left  = build(pre, in, pl + 1, pl + leftSize, il, idx - 1);
        // 右子树：preorder 剩下的一段；inorder 在根右侧
        root->right = build(pre, in, pl + leftSize + 1, pr, idx + 1, ir);
        return root;
    }

public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int n = preorder.size();
        for (int i = 0; i < n; i++)
            pos[inorder[i]] = i;               // 预处理哈希表
        return build(preorder, inorder, 0, n - 1, 0, n - 1);
    }
};
```

实现要点：

1. **递归边界 `pl > pr`** 用 preorder 的区间判断即可（两个序列区间长度永远相等，判哪个都行）；
2. **`pos` 作为成员变量**只建立一次；若写成每层递归重建，哈希表预处理本身就会变成 $O(n^2)$；
3. 所有切分都是下标运算，自始至终不复制数组。

---

## 4. 复杂度分析

- **时间复杂度**：$O(n)$。哈希表预处理 $O(n)$；递归共 $n$ 个节点，每个节点做 $O(1)$ 工作（一次哈希查询、几次下标运算）。
- **空间复杂度**：$O(n)$。哈希表 $O(n)$；递归深度最坏为树高 $h$，链状时 $h = n$，平衡时 $h = O(\log n)$（不计输出树本身）。

| 做法 | 时间 | 空间 | 瓶颈 |
|---|---|---|---|
| 暴力：每层线性找根 | $O(n^2)$ | $O(h)$ | 重复扫描 inorder |
| 切数组递归 + 哈希 | $O(n^2)$ | $O(n^2)$ | 每层拷贝子数组 |
| **下标递归 + 哈希** | $O(n)$ | $O(n)$ | — |

---

## 5. 常见疑问

**疑问一：只有前序遍历为什么不行？缺的信息到底是什么？**

前序能告诉你「根是谁」，但看不出左右子树的分界线。中序的价值正在于：根左边的全部属于左子树、右边的全部属于右子树——它提供了**切分点**。同理，只有中序也不行（知道切分点却不知道谁是根）。反例：$[1,2]$ 与 $[2,1]$ 两种形态的中序 + 前序组合需要交叉验证。有趣的是，**前序 + 后序**也不能唯一确定二叉树（一个节点的独生子在左在右无法区分），除非题目保证是满二叉树——见第 6 节的 889 题。

**疑问二：为什么必须无重复元素？**

哈希表 `pos` 要求「值 → 下标」是函数：值重复时不知道该定位到哪个出现位置，切分就可能出现歧义。更本质地，有重复值时**答案本身可能不唯一**（例如 `preorder = [1,1]`、`inorder = [1,1]` 既能是右链也能是左链），题目无从判定。若面试中被追问有重复值怎么办，可以退一步输出任意一棵合法的树，把哈希查询改成「取区间内最左出现」并允许随意归边。

**疑问三：`leftSize` 算错端点，典型症状是什么？**

最常见的是把左子树的 preorder 区间写成 `pl+1 .. pl+leftSize` 时手滑成 `pl .. pl+leftSize`（把根也算了进去），或 inorder 右子树起点写成 `idx` 而不是 `idx+1`（把根重复进右子树）。症状是输出树多出重复节点或触发越界。建议像 2.4 节那样，先用五元素小例子手算一遍四个端点再动手写代码。

---

## 6. 相关题目与模板

**跨序列换算模板**（「前序定根、中序切分」的通用骨架）：

```cpp
// root = pre[pl]; idx = pos[root]; leftSize = idx - il;
// 左: (pl+1, pl+leftSize, il, idx-1)
// 右: (pl+leftSize+1, pr, idx+1, ir)
```

这个「定位根 → 算出子树大小 → 同时换算两个序列的区间」流程，是所有双遍历序列建表题的公共内核，改的只是「哪一侧给切分点」。

**同类题目：**

- [LeetCode 106. 从中序与后序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)：镜像题，根在 `post[pr]`，建议**先递归右子树再左子树**（与 105 唯一的顺序差异）
- [LeetCode 889. 根据前序和后序遍历构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-postorder-traversal/)：答案不唯一，用「左子树根 = `pre[pl+1]` 在后序中的位置」来切分
- [LeetCode 1008. 前序遍历构造二叉搜索树](https://leetcode.cn/problems/construct-binary-search-tree-from-preorder-traversal/)：BST 的中序就是排序结果，等于隐式给出了中序，可插入法 $O(n\log n)$
- [LeetCode 654. 最大二叉树](https://leetcode.cn/problems/maximum-binary-tree/)：同样是「找根 + 切区间 + 递归」，只是定位根的条件从哈希查询换成了区间最大值

**一句话总结**：前序回答「根是谁」，中序回答「左右在哪分」——哈希表把「回答第二个问题」的价格从 $O(n)$ 打到 $O(1)$，剩下的只是小心端点换算。
