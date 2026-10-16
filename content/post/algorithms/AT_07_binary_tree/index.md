---
title: "AT 07: 二叉树——遍历与构造模板"
date: 2026-11-30
math: true
tags:
  - Notes
  - 算法模板
  - 二叉树
  - 遍历
  - 递归
categories:
  - 算法模板
description: "二叉树刷题模板全家桶：递归三段式统一先中后序、迭代版（藤缠树/左链下探/根右左反转）、层序按层分组、前中序构造（哈希定位根）、序列化（增强序列）、LCA 递归后序；样题 LC 94/102/105/236 + 洛谷 P1030（中序+后序求先序），附 10 题清单"
---

# AT 07：二叉树——遍历与构造模板

> 本篇是 [DSA 5：二叉树](../../DSA/DSA_5/) 的刷题武器化。理论（藤缠树为什么对、重构为什么需要中序、增强序列）在那边；这边只回答**怎么写、怎么套题**。核心思想一句话：**所有二叉树问题都是「遍历顺序 + 在遍历中做什么」的组合——先定位节点被访问的时机，问题就解决了一半。**

---

## 1. 模板解决什么问题

| 操作 | 递归版 | 迭代版 | 层序 |
|------|--------|--------|------|
| 先序 | 三段式（访问→左→右） | 藤缠树：沿左链访问，右孩子压栈 | ❌ |
| 中序 | 三段式（左→访问→右） | 左链下探压栈，弹出即访问 | ❌ |
| 后序 | 三段式（左→右→访问） | 变形先序（根右左）再整体反转 | ❌ |
| 层序 | —— | —— | 队列 + 按层计数 |

四类题直接套表：**遍历输出**（LC 94/144/145）、**按层处理**（LC 102/199）、**从遍历序列重构**（LC 105/106、洛谷 P1030）、**在树上递归求解**（LC 104/124/236——遍历框架一样，只是「访问」变成「合并子树答案」）。

---

## 2. 代码模板

约定 `struct TreeNode { int val; TreeNode *left, *right; };`（LeetCode 标准定义）。

### 2.1 递归三段式（先/中/后序共用一个骨架）

```cpp
void traverse(TreeNode* x) {
    if (!x) return;              // ① 终止：空树直接返回
    // —— 先序位置：visit(x) ——
    traverse(x->left);
    // —— 中序位置：visit(x) ——
    traverse(x->right);
    // —— 后序位置：visit(x) ——
}
```

三种遍历**只差 `visit` 放哪个空位**。这不是巧合：访问时机 = 节点在三段流程中的相对位置。刷题时先想清楚「我需要节点在什么时候被处理」（进入时？左子树回来后？两个孩子都回来后？），遍历顺序自然确定——例如 LC 104 求深度在**后序位置**合并 `max(左, 右) + 1`，LC 236 的 LCA 同样在后序位置合并两侧结果（§4.4）。

### 2.2 迭代先序：藤缠树

[DSA 5 §8.2](../../DSA/DSA_5/) 的观察：沿左链一路向下的过程是「自上而下访问 + 右子树暂存」：

```cpp
vector<int> preorder(TreeNode* root) {
    vector<int> out;
    stack<TreeNode*> s;
    TreeNode* x = root;
    while (true) {
        while (x) {                  // 访问藤上节点
            out.push_back(x->val);
            s.push(x->right);        // 右子树暂存（可能为空，压 nullptr 也无妨）
            x = x->left;
        }
        if (s.empty()) break;        // 没有暂存的右子树了 → 结束
        x = s.top(); s.pop();
    }
    return out;
}
```

每个节点入栈至多一次（作为右孩子被压），出栈一次，总 $O(n)$。

### 2.3 迭代中序：左链下探

与先序的差别只有一个：沿左链下行时**不能访问**，必须全部压栈、等左子树处理完（弹出时）再访问（[DSA 5 §9.2](../../DSA/DSA_5/)）：

```cpp
vector<int> inorder(TreeNode* root) {
    vector<int> out;
    stack<TreeNode*> s;
    TreeNode* x = root;
    while (true) {
        while (x) { s.push(x); x = x->left; }   // 左链整体压栈
        if (s.empty()) break;
        x = s.top(); s.pop();                    // 左子树处理完，访问栈顶
        out.push_back(x->val);
        x = x->right;                            // 转向右子树
    }
    return out;
}
```

### 2.4 迭代后序：变形先序 + 反转

后序 = 左右根 = 根右左的**镜像反转**。把先序迭代版的左右互换（先压左、走右链），最后反转输出：

```cpp
vector<int> postorder(TreeNode* root) {
    vector<int> out;
    if (!root) return out;
    stack<TreeNode*> s; s.push(root);
    while (!s.empty()) {
        TreeNode* x = s.top(); s.pop();
        out.push_back(x->val);          // 访问顺序：根、右、左
        if (x->left)  s.push(x->left);  // 左先压（后弹出）
        if (x->right) s.push(x->right);
    }
    reverse(out.begin(), out.end());    // 反转 → 左、右、根
    return out;
}
```

三个迭代版对照记：**先序在下行时访问，中序在弹出时访问，后序借道反转**。DSA 5 §11.3 另有一份「正统」的 `gotoLeftmostLeaf` 版后序，刷题用反转版足够。

### 2.5 层序：队列 + 按层计数

```cpp
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> out;
    if (!root) return out;
    queue<TreeNode*> q; q.push(root);
    while (!q.empty()) {
        int sz = q.size();              // 关键：进入循环前锁死本层节点数
        out.emplace_back();
        for (int i = 0; i < sz; i++) {  // 只处理本层，天然按层分组
            TreeNode* x = q.front(); q.pop();
            out.back().push_back(x->val);
            if (x->left)  q.push(x->left);
            if (x->right) q.push(x->right);
        }
    }
    return out;
}
```

**坑**：`sz` 必须在 `for` 之前取——循环里 `push` 会改 `q.size()`，写 `i < (int)q.size()` 是新手第一大 bug。只取每层最后一个（右视图 LC 199）就是 `out.back().back()`。

---

## 3. 常见变形与坑

| 变形 | 一句话做法 |
|------|-----------|
| 每层取一个（右视图 LC 199） | 层序模板取 `out.back().back()`，或每层循环结束取最后入队的 |
| 之字形层序 | 层序 + 偶数层 `reverse(out.back())` |
| 判断对称（LC 101） | 递归 `(l, r)` 双参数：`l->val == r->val` 且 `(l->l, r->r)`、`(l->r, r->l)` 对称递归 |
| 最大深度（LC 104） | 后序位置 `return 1 + max(d(l), d(r))` |
| 翻转（LC 226） | 先序位置 `swap(x->left, x->right)` 再递归 |
| 序列化 / 反序列化（LC 297） | 先序 + 把 `nullptr` 也写出（增强序列，DSA 5 §13.2）；反序列化用引用消费队列 |
| 由前序 + 空指针标记直接建树 | 与序列化同构：递归消费序列，遇到 `#` 返回空 |

**坑清单**：

1. **`q.size()` 是无符号**：`i < q.size() - 1` 在空队列时下溢成天文数字——取层大小用 `int sz = q.size()`；
2. **空树也是答案**：所有模板对 `root == nullptr` 必须直接返回，层序尤其常见 WA 点；
3. **先序 + 后序不能唯一重构**（DSA 5 §13.1）：单儿子时分不清在左在右——题目若这么给，必有附加信息；
4. **构造题必须用哈希定位根**：每层线性扫中序找根是 $O(n^2)$，`n` 到 $10^5$ 会 TLE。

---

## 4. 样题精讲

### 4.1 LC 94. 二叉树的中序遍历（模板题）

[题目](https://leetcode.cn/problems/binary-tree-inorder-traversal/) ｜ [仓库题解](../../LeetCode/LeetCode_94/)

直接套 §2.3；递归版把 `visit` 放中序空位即可。这题的价值是**练手迭代中序**——它是 BST 系（LC 98/230/173，见 [AT 08](../AT_08_bst_balanced_tree/)）的基础设施：BST 的中序遍历 = 有序序列。

### 4.2 LC 102. 二叉树的层序遍历（模板题）

[题目](https://leetcode.cn/problems/binary-tree-level-order-traversal/) ｜ [仓库题解](../../LeetCode/LeetCode_102/)

套 §2.5，无变形。看这题是否 5 分钟内写对，是检验层序模板是否肌肉记忆的标尺。

### 4.3 LC 105. 从前序与中序遍历序列构造二叉树（构造题代表）

[题目](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) ｜ [仓库题解](../../LeetCode/LeetCode_105/)

**思路**：先序第一个是根；用哈希表在中序里 $O(1)$ 定位根，把中序切成左/右两段——两段长度反过来切先序，递归下去。

```cpp
class Solution {
    unordered_map<int, int> pos;               // 值 → 中序下标
    TreeNode* build(vector<int>& pre, vector<int>& in,
                    int pl, int pr, int cl, int cr) {   // 双闭区间
        if (pl > pr) return nullptr;
        TreeNode* root = new TreeNode(pre[pl]);
        int k = pos[pre[pl]];                  // 根在中序中的位置
        int L = k - cl;                        // 左子树节点数
        root->left  = build(pre, in, pl + 1, pl + L, cl, k - 1);
        root->right = build(pre, in, pl + L + 1, pr, k + 1, cr);
        return root;
    }
public:
    TreeNode* buildTree(vector<int>& pre, vector<int>& in) {
        for (int i = 0; i < (int)in.size(); i++) pos[in[i]] = i;
        return build(pre, in, 0, pre.size() - 1, 0, in.size() - 1);
    }
};
```

**不变量**：`[pl, pr]` 与 `[cl, cr]` 始终描述同一棵子树；`L = k - cl` 是唯一需要的「翻译」——中序里的分界长度，原样搬去切先序。复杂度 $O(n)$。

<details>
<summary><b>坑：本题可以反向遍历先序 + 引用下标，为什么推荐双闭区间写法</b></summary>

另一种流行写法是先序用全局引用下标 `pl++` 消费，只传中序区间。代码更短，但「先序与中序的对应关系」藏在递归顺序里，不容易看出错在哪。双闭区间写法把对应关系全部写进参数，调试时区间打出来就能定位；两种都是 $O(n)$，推荐先用区间版建立心智模型，熟了再换短版。

</details>

### 4.4 LC 236. 二叉树的最近公共祖先（递归后序代表）

[题目](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/) ｜ [仓库题解](../../LeetCode/LeetCode_236/)

**思路**：后序位置拿到左右子树各自的汇报——「子树里有没有 p 或 q」。两侧都有 → 当前节点就是 LCA；只有一侧有 → 答案在那一侧。

```cpp
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* x, TreeNode* p, TreeNode* q) {
        if (!x || x == p || x == q) return x;   // 空树 / 命中目标
        TreeNode* L = lowestCommonAncestor(x->left,  p, q);
        TreeNode* R = lowestCommonAncestor(x->right, p, q);
        if (L && R) return x;                   // p、q 分居两侧 → 我是答案
        return L ? L : R;                       // 都在一侧 → 往那侧走
    }
};
```

**为什么对**：自底向上第一次遇到「p、q 分居两棵不同子树」的节点就是 LCA。隐藏前提：p、q 都存在（题目保证）。若 p 可能不存在，此模板会误报——需要把返回值改成「子树内找到几个」。

### 4.5 洛谷 P1030. [NOIP 2001 普及组] 求先序排列（中序 + 后序 → 先序）

[题目](https://www.luogu.com.cn/problem/P1030)

LC 105 的镜像：**后序最后一个是根**（先序是第一个），切分方向完全对称。这道题同时验证了 DSA 5 §13.1 的结论——中序提供分界、另一种序提供根，两者就能重构。

```cpp
#include <bits/stdc++.h>
using namespace std;

string in, post;

void solve(int cl, int cr, int pl, int pr) {   // 双闭区间，根在后序末位
    if (cl > cr) return;
    char root = post[pr];
    putchar(root);
    int k = cl; while (in[k] != root) k++;     // 根在中序中的位置
    int L = k - cl;                            // 左子树大小
    solve(cl, k - 1, pl, pl + L - 1);          // 左：中序 [cl,k-1]，后序 [pl, pl+L-1]
    solve(k + 1, cr, pl + L, pr - 1);          // 右：中序 [k+1,cr]，后序 [pl+L, pr-1]
}

int main() {
    cin >> in >> post;
    solve(0, in.size() - 1, 0, post.size() - 1);
    return 0;
}
```

题面给的是大写字母串，各字母互不相同，线性扫 `in` 找根也够快（无需哈希）。输出的先序 = 「根 + 左子树先序 + 右子树先序」，`putchar` 顺序天然正确。

---

## 5. 题目清单

| 题号 | 题名 | 难度 | 考的变形 | 链接 |
|------|------|------|---------|------|
| LC 94 | 二叉树的中序遍历 | 简单 | 迭代中序 | [仓库题解](../../LeetCode/LeetCode_94/) |
| LC 101 | 对称二叉树 | 简单 | 双参数递归 | [仓库题解](../../LeetCode/LeetCode_101/) |
| LC 102 | 二叉树的层序遍历 | 中等 | 层序按层分组 | [仓库题解](../../LeetCode/LeetCode_102/) |
| LC 104 | 二叉树的最大深度 | 简单 | 后序位置合并 | [仓库题解](../../LeetCode/LeetCode_104/) |
| LC 105 | 从前序与中序遍历序列构造二叉树 | 中等 | 重构 + 哈希定位根 | [仓库题解](../../LeetCode/LeetCode_105/) |
| LC 124 | 二叉树中的最大路径和 | 困难 | 后序返回值与全局答案分离 | [仓库题解](../../LeetCode/LeetCode_124/) |
| LC 199 | 二叉树的右视图 | 中等 | 层序取层尾 | [仓库题解](../../LeetCode/LeetCode_199/) |
| LC 226 | 翻转二叉树 | 简单 | 先序位置 swap | [仓库题解](../../LeetCode/LeetCode_226/) |
| LC 236 | 二叉树的最近公共祖先 | 中等 | 递归后序 | [仓库题解](../../LeetCode/LeetCode_236/) |
| 洛谷 P1030 | [NOIP 2001 普及组] 求先序排列 | 普及− | 中序+后序重构 | [题目](https://www.luogu.com.cn/problem/P1030) |

进阶：LC 106（中序+后序构造，P1030 的返回树版）、LC 297（序列化——增强序列思想）、LC 437（路径总和，前缀和上树，配合 [前缀和专题](../prefix_sum/)）。

---

## 6. 与其他篇的联系

- **理论后盾**：[DSA 5](../../DSA/DSA_5/)——藤缠树（§8.2）、迭代中序（§9.3）、重构定理（§13）、增强序列（§13.2）；
- **[AT 08：BST 与平衡树](../AT_08_bst_balanced_tree/)**：中序遍历是 BST 一切性质的地基（LC 98/230 都靠它）；
- **[AT 05：单调栈](../AT_05_monotonic_stack/)**：LC 94→LC 173（BST 迭代器）是把迭代中序拆成 `next()` 逐步吐出，与单调栈同为「状态可暂停的遍历」；
- 层序是图的 BFS 在树上的特例，一般图版见 [AT 11：图的存储与遍历](../AT_11_graph_bfs_dfs/)；
- 既有专题 [单调队列](../monotonic_queue/)、[前缀和](../prefix_sum/) 与本篇在 LC 437 汇合。
