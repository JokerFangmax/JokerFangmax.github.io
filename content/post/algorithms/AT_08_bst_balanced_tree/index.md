---
title: "AT 08: BST 与平衡树——何时手写，何时 STL"
date: 2026-12-01
math: true
tags:
  - Notes
  - 算法模板
  - 二叉搜索树
  - 平衡树
  - Treap
categories:
  - 算法模板
description: "BST 查找/插入/删除模板（双分支删除的后继化归）、验证 BST 与第 K 小；「何时手写、何时 STL」决策表（multiset / pb_ds / 手写 Treap）；FHQ-Treap 完整模板（insert/erase/rank/kth/pre/nxt，2 万次随机操作对照 multiset 全通过）与文艺平衡树区间翻转变体；样题 LC 98/230 + 洛谷 P3369/P2234/P3391"
---

# AT 08：BST 与平衡树——何时手写，何时 STL

> [DSA 6：BST](../../DSA/DSA_6/) 与 [DSA 8：高级搜索树](../../DSA/DSA_8/) 的刷题武器化。核心思想一句话：**BST 的一切刷题用法都来自「中序有序」，而平衡树的一切复杂度保证都来自「树高被压在 $O(\log n)$」——刷题时先用 STL 吃掉 80% 的题，手写 Treap 只为剩下那 20% 需要 rank / kth / 区间切割的题。**

---

## 1. 先给结论：选型决策表

| 需求 | 选择 | 备注 |
|------|------|------|
| 判存在、按值增删 | `unordered_set/map` | $O(1)$，见 [AT 17](../AT_17_hash_table/) |
| 有序遍历、前驱后继、lower_bound | `set / multiset / map` | $O(\log n)$，覆盖绝大多数题 |
| 名次（rank）、第 k 小（kth） | `pb_ds` 的 `tree` 或手写 Treap | pb_ds 带 `order_of_key / find_by_order` |
| 区间切割/翻转（文艺平衡树类） | 手写 FHQ-Treap（或 Splay） | split / merge 天生为区间而生 |
| 可持久化（历史版本） | 手写 FHQ-Treap | merge/split 不改原树结构路径即可复制 |
| AVL / 红黑树 / B 树 | **不手写** | 面试口头说清旋转思想即可（[DSA 8](../../DSA/DSA_8/)） |

pb_ds 的 `tree` 写法（GCC 专属，洛谷可用；$O(\log n)$ 且不用自己写）：

```cpp
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
tree<int, null_type, less<int>, rb_tree_tag,
     tree_order_statistics_node_update> T;   // 无重复；要重复用 pair<值,时间戳>
// T.insert(v);  T.erase(v);
// T.order_of_key(v)   → 严格小于 v 的个数（rank - 1）
// T.find_by_order(k)  → 第 k+1 小的迭代器（0-indexed）
```

---

## 2. BST 增删查模板

理论（为什么新节点必是叶子、双分支删除为何化归为后继）见 [DSA 6 §5–7](../../DSA/DSA_6/)，此处只给刷题形态。

### 2.1 查找 / 插入

```cpp
TreeNode* search(TreeNode* x, int v) {           // 查找：命中返回节点，否则 nullptr
    while (x && x->val != v) x = v < x->val ? x->left : x->right;
    return x;
}

void insert(TreeNode*& root, int v) {            // 插入：失败的查找位置即新叶子
    TreeNode** p = &root;                        // 指针的指针：免掉父节点特判
    while (*p) p = v < (*p)->val ? &(*p)->left : &(*p)->right;
    *p = new TreeNode(v);
}
```

`TreeNode**` 的技巧值得专门记：`p` 始终指向「待接新节点的那个槽位」（根指针本身，或某个节点的 `left/right` 字段），走到空即插入位置，`*p = new` 一步接上——不需要记录父节点、不需要区分插左还是插右。

### 2.2 删除（双分支用后继化归）

[DSA 6 §7.2](../../DSA/DSA_6/) 的化归：双分支节点**换值不换拓扑**——把中序后继（右子树最左节点，必无左孩子）的值拷上来，问题化归为删除一个至多单分支的节点：

```cpp
TreeNode* remove(TreeNode* x, int v) {           // 返回「顶替者」，父辈用它接上
    if (!x) return nullptr;
    if      (v < x->val) x->left  = remove(x->left,  v);
    else if (v > x->val) x->right = remove(x->right, v);
    else {
        if (!x->left || !x->right) {             // 零/单分支：孩子直接顶上
            TreeNode* ch = x->left ? x->left : x->right;
            return ch;
        }
        TreeNode* s = x->right;                  // 双分支：右子树一路向左 = 中序后继
        while (s->left) s = s->left;
        x->val = s->val;                         // 值替换，拓扑不动
        x->right = remove(x->right, s->val);     // 化归：删右子树里的那个后继值
    }
    return x;
}
```

单分支与双分支的公共出口都是「最终删的是至多单分支的节点」，所以代码量集中在化归一步。本模板经 2 万次随机增删查对照 `multiset` 中序验证。

---

## 3. FHQ-Treap 完整模板（手写平衡树）

需要 rank / kth / 区间操作时上场。选 **FHQ-Treap**（无旋 Treap）而不是旋转版 Treap/Splay：只有 `split / merge` 两个原语，代码短、可持久化友好。结构：**键值满足 BST 性质，优先级满足堆性质**——`split` 按键值切，随机优先级让期望树高 $O(\log n)$。

```cpp
#include <bits/stdc++.h>
using namespace std;
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

struct FHQ {
    struct Node { int l, r; unsigned pri; int val, siz; };
    vector<Node> t;                              // 0 号是空哨兵
    int root = 0, cnt = 0;
    FHQ() { t.push_back({0, 0, 0, 0, 0}); }
    int newNode(int v) { t.push_back({0, 0, rng(), v, 1}); return ++cnt; }
    void pull(int x) { t[x].siz = t[t[x].l].siz + t[t[x].r].siz + 1; }
    // 按值分裂：<= v 去 A，> v 去 B（破坏性！用完必须 merge 回去）
    pair<int, int> split(int x, int v) {
        if (!x) return {0, 0};
        if (t[x].val <= v) {
            auto [a, b] = split(t[x].r, v); t[x].r = a; pull(x); return {x, b};
        } else {
            auto [a, b] = split(t[x].l, v); t[x].l = b; pull(x); return {a, x};
        }
    }
    int merge(int a, int b) {                    // 约定 A 全部 <= B
        if (!a || !b) return a + b;
        if (t[a].pri > t[b].pri) { t[a].r = merge(t[a].r, b); pull(a); return a; }
        else                     { t[b].l = merge(a, t[b].l); pull(b); return b; }
    }
    void insert(int v) {
        auto [a, b] = split(root, v);
        root = merge(merge(a, newNode(v)), b);
    }
    void erase(int v) {                          // 只删一个 v
        auto [a, b] = split(root, v - 1);
        auto [c, d] = split(b, v);               // c 里全是 == v 的
        if (c) c = merge(t[c].l, t[c].r);        // 丢掉 c 的根 = 删掉一个 v
        root = merge(a, merge(c, d));
    }
    int rank(int v) {                            // v 的名次（<v 的个数 +1）
        auto [a, b] = split(root, v - 1);
        int r = t[a].siz + 1;
        root = merge(a, b);                      // ★ split 是破坏性的，必须重组！
        return r;
    }
    int kth(int k) {                             // 第 k 小（纯查询，不 split）
        int x = root;
        while (x) {
            int ls = t[t[x].l].siz;
            if (k <= ls) x = t[x].l;
            else if (k == ls + 1) return t[x].val;
            else { k -= ls + 1; x = t[x].r; }
        }
        return INT_MIN;                          // k 越界
    }
    int pre(int v) { return kth(rank(v) - 1); }  // 严格小于 v 的最大值
    int nxt(int v) {                             // 严格大于 v 的最小值
        int r = rank(v);
        while (kth(r) == v) r++;                 // 跳过重复的 v
        return kth(r);
    }
};
```

**实测记录**：该模板经 2 万次随机六操作对照 `std::multiset` 逐步核对（名次、第 k 小、前驱、后继、树的中序一致性）全部通过。测试中真实踩到一个坑——见 §4 坑清单第 1 条，正是模板里打 ★ 的那一行。

<details>
<summary><b>文艺平衡树变体：按大小分裂 + 翻转懒标记（洛谷 P3391）</b></summary>

区间 $[l, r]$ 翻转：按**大小**（而非键值）split 成 $A\,[1,l-1]$、$B\,[l,r]$、$C$ 三段，给 $B$ 的根打翻转懒标记再 merge 回去。懒标记的语义：**本节点已换向，孩子还没**——下放时 `swap(l, r)` 并给孩子翻转标记。

```cpp
// 延续上文的 mt19937 rng;（与 FHQ 模板共用）
struct Node { int l, r; unsigned pri; int val, siz; bool rev; };
struct Treap {
    vector<Node> t; int root = 0, cnt = 0;
    Treap() { t.push_back({0,0,0,0,0,false}); }
    int newNode(int v) { t.push_back({0,0,rng(),v,1,false}); return ++cnt; }
    void pull(int x) { t[x].siz = t[t[x].l].siz + t[t[x].r].siz + 1; }
    void push(int x) {                          // ★ 访问孩子前必须 push
        if (t[x].rev) {
            swap(t[x].l, t[x].r);
            if (t[x].l)  t[t[x].l].rev  ^= 1;
            if (t[x].r)  t[t[x].r].rev  ^= 1;
            t[x].rev = false;
        }
    }
    pair<int,int> split(int x, int k) {          // 前 k 个去 A，其余去 B
        if (!x) return {0, 0};
        push(x);                                 // 分裂要下探，先下放
        if (t[t[x].l].siz >= k) { auto [a,b] = split(t[x].l, k); t[x].l = b; pull(x); return {a,x}; }
        else { auto [a,b] = split(t[x].r, k - t[t[x].l].siz - 1); t[x].r = a; pull(x); return {x,b}; }
    }
    int merge(int a, int b) {
        if (!a || !b) return a + b;
        if (t[a].pri > t[b].pri) { push(a); t[a].r = merge(t[a].r, b); pull(a); return a; }
        else                     { push(b); t[b].l = merge(a, t[b].l); pull(b); return b; }
    }
    void revRange(int l, int r) {                // 1-indexed 闭区间
        auto [a, bc] = split(root, l - 1);
        auto [b, c]  = split(bc, r - l + 1);
        if (b) t[b].rev ^= 1;
        root = merge(a, merge(b, c));
    }
    void inorder(int x, vector<int>& out) {      // 输出前 push
        if (!x) return; push(x);
        inorder(t[x].l, out); out.push_back(t[x].val); inorder(t[x].r, out);
    }
};
```

初始建树：把 $1..n$ 依次 `merge` 到右边（每棵单节点树键值天然有序，满足 merge 的「A ≤ B」约定）。此变体经 2000 次随机区间翻转对照暴力 `reverse` 全部通过。P3391 全解就是「建树 + 循环 revRange + 中序输出」。

</details>

---

## 4. 常见变形与坑

| 变形 | 一句话做法 |
|------|-----------|
| 验证 BST（LC 98） | 中序遍历检查严格递增（或递归带 `(lo, hi)` 开区间上下界） |
| BST 第 k 小（LC 230） | 中序遍历数到第 k 个；若树节点带子树大小则沿树 $O(h)$ 定位 |
| 有序数组转平衡 BST（LC 108） | 取中点当根、两半递归——分治即平衡 |
| 迭代器（LC 173） | 把 [AT 07](../AT_07_binary_tree/) 的迭代中序拆成 `next()` 按需吐一个 |
| 前驱后继查询 | `set::lower_bound/upper_bound`，或 Treap 的 `pre/nxt` |
| 区间第 k 小（不带修改） | 排序后离线 / 主席树（可持久化线段树，超出本篇） |

**坑清单**：

1. **`split` 是破坏性操作**（本篇实测踩坑）：`split(root, v)` 返回两棵子树后，原 `root` 只剩其中一半可达——**只查询不重组，树就静默损坏**。症状极具迷惑性：增删路径全对、纯增删的树完全一致，一掺 `rank` 类查询就错。凡是「split 完只读一下就丢」的写法，必须 `root = merge(a, b)` 还原；
2. **`multiset::erase(v)` 删的是所有等于 v 的**：只删一个要 `erase(s.find(v))`；
3. **LC 98 不能只比较父子**：`[5,4,6,null,null,3,7]` 每对父子都合法但整体不是 BST（DSA 6 §3.2）——必须中序全局有序或传上下界；
4. **带懒标记的 Treap：凡下探孩子前必 `push`**（split / merge / 遍历三处），漏一处答案时对时错，最难查；
5. **merge 的前置约定**：A 中所有键 ≤ B 中所有键——违反则树直接坏，文艺平衡树按大小分裂正是为守住这条约定。

---

## 5. 样题精讲

### 5.1 LC 98. 验证二叉搜索树（BST 性质题代表）

[题目](https://leetcode.cn/problems/validate-binary-search-tree/) ｜ [仓库题解](../../LeetCode/LeetCode_98/)

**思路**：BST 的定义是**对整棵子树的约束**（左子树全部 `< root < 右子树全部），不是只看直接孩子。最稳的写法是递归传开区间 `(lo, hi)`：

```cpp
class Solution {
    bool valid(TreeNode* x, long lo, long hi) {   // 开区间 (lo, hi)
        if (!x) return true;
        if (x->val <= lo || x->val >= hi) return false;
        return valid(x->left, lo, x->val) && valid(x->right, x->val, hi);
    }
public:
    bool isValidBST(TreeNode* root) { return valid(root, LONG_MIN, LONG_MAX); }
};
```

用 `long` 是为了容纳 `INT_MIN/INT_MAX` 的节点值贴边界。等价写法：中序遍历一边走一边检查「严格大于前一个」。

### 5.2 LC 230. 二叉搜索树中第 K 小的元素（中序计数）

[题目](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/) ｜ [仓库题解](../../LeetCode/LeetCode_230/)

**思路**：BST 中序有序 → 第 k 小 = 中序第 k 个访问的节点。把迭代中序模板（[AT 07 §2.3](../AT_07_binary_tree/)）的 `out.push_back` 换成计数返回：

```cpp
class Solution {
public:
    int kthSmallest(TreeNode* root, int k) {
        stack<TreeNode*> s;
        TreeNode* x = root;
        while (true) {
            while (x) { s.push(x); x = x->left; }
            x = s.top(); s.pop();
            if (--k == 0) return x->val;   // 第 k 次弹出即答案
            x = x->right;
        }
    }
};
```

若树节点带子树大小（如手写 Treap 的 `kth`），可 $O(h)$ 沿树定位；本题普通 TreeNode 只能 $O(k)$ 中序——面试时点出这个区别是加分项。

### 5.3 洛谷 P3369. 【模板】普通平衡树（FHQ-Treap 模板题）

[题目](https://www.luogu.com.cn/problem/P3369)

六操作与 §3 模板一一对应（操作 1/2 → `insert/erase`，3/4 → `rank/kth`，5/6 → `pre/nxt`），主程序读入分发即可：

```cpp
int main() {
    int n; scanf("%d", &n);
    FHQ f;
    while (n--) {
        int op, x; scanf("%d%d", &op, &x);
        switch (op) {
            case 1: f.insert(x); break;
            case 2: f.erase(x);   break;
            case 3: printf("%d\n", f.rank(x)); break;
            case 4: printf("%d\n", f.kth(x));  break;
            case 5: printf("%d\n", f.pre(x));  break;
            case 6: printf("%d\n", f.nxt(x));  break;
        }
    }
    return 0;
}
```

**练习建议**：先写一版 `multiset` + 遍历数 rank 的暴力对拍（这就是本篇模板的验证方法），再上手 Treap——坑 1（split 不重组）几乎人人会踩一次，有对拍能立刻发现。[P6136 数据加强版](https://www.luogu.com.cn/problem/P6136)同题面，卡掉 $O(n \sqrt n)$ 级做法，专测平衡树真伪。

### 5.4 洛谷 P2234. [HNOI2002] 营业额统计（前驱后继的应用）

[题目](https://www.luogu.com.cn/problem/P2234)

**题意**：每天营业额 $a_i$，定义其「最小波动值」为 $\min_{j<i} |a_i - a_j|$（第一天为 $a_i$ 本身），求总和。**思路**：$\min |a_i - a_j|$ 只会取在 $a_i$ 的**前驱或后继**上——这正是有序集合的原生操作。每个新数插入前查 `pre/nxt`，取近者累计：

```cpp
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n; scanf("%d", &n);
    multiset<int> s;
    long long ans = 0;
    for (int i = 0; i < n; i++) {
        int x; scanf("%d", &x);
        if (s.empty()) { ans += x; }             // 第一天
        else {
            long long best = LLONG_MAX;
            auto it = s.lower_bound(x);          // 第一个 >= x
            if (it != s.end())   best = min(best, (long long)*it - x);        // 后继
            if (it != s.begin()) best = min(best, (long long)x - *prev(it)); // 前驱
            ans += best;
        }
        s.insert(x);
    }
    printf("%lld\n", ans);
    return 0;
}
```

这题是「何时用 STL」的完美注脚：核心洞察（波动只看前驱后继）想通之后，`multiset` 十行写完，**没有任何理由手写平衡树**。同一洞察 + rank/kth 需求，才轮到 Treap 上场（如 P3369）。

---

## 6. 题目清单

| 题号 | 题名 | 难度 | 考的变形 | 链接 |
|------|------|------|---------|------|
| LC 98 | 验证二叉搜索树 | 中等 | 开区间上下界 / 中序有序 | [仓库题解](../../LeetCode/LeetCode_98/) |
| LC 108 | 将有序数组转换为二叉搜索树 | 简单 | 中点分治建树 | [仓库题解](../../LeetCode/LeetCode_108/) |
| LC 230 | 二叉搜索树中第 K 小的元素 | 中等 | 中序计数 | [仓库题解](../../LeetCode/LeetCode_230/) |
| LC 450 | 删除二叉搜索树中的节点 | 中等 | 双分支删除化归 | [题目](https://leetcode.cn/problems/delete-node-in-a-bst/) |
| 洛谷 P3369 | 【模板】普通平衡树 | 模板 | Treap 六操作 | [题目](https://www.luogu.com.cn/problem/P3369) |
| 洛谷 P6136 | 【模板】普通平衡树（数据加强版） | 模板 | 同上，卡暴力 | [题目](https://www.luogu.com.cn/problem/P6136) |
| 洛谷 P3391 | 【模板】文艺平衡树 | 模板 | 按大小分裂 + 翻转懒标记 | [题目](https://www.luogu.com.cn/problem/P3391) |
| 洛谷 P2234 | [HNOI2002] 营业额统计 | 普及+/提高 | 前驱后继应用 | [题目](https://www.luogu.com.cn/problem/P2234) |

进阶：LC 173（BST 迭代器——迭代中序的分步版）、LC 96/95（BST 计数与构造，卡特兰数）、P3835（可持久化平衡树——FHQ 的 merge/split 不碰原路径即可复制）。

---

## 7. 与其他篇的联系

- **理论后盾**：[DSA 6](../../DSA/DSA_6/)（BST 增删查、旋转与等价类）、[DSA 8](../../DSA/DSA_8/)（AVL/Splay/B树/红黑树全景——为什么刷题不手写它们）、[DSA 9 §24–26](../../DSA/DSA_9/)（跳表：概率平衡的另一种答案，`std::set` 的思维近亲）；
- **[AT 07：二叉树遍历](../AT_07_binary_tree/)**：中序遍历是本篇一切性质的地基；
- **[AT 09：堆](../AT_09_heap/)**：同为「树形有序结构」，堆只管最值（$O(1)$ 取 max）不管全序——TopK 用堆、名次查询用平衡树；
- **[AT 15：线段树](../AT_15_segment_tree/)**：文艺平衡树的区间翻转与线段树的区间操作是同一套「懒标记」思想的两种载体；
- **[AT 17：哈希表](../AT_17_hash_table/)**：决策表的第一行——无序需求别用平衡树。
