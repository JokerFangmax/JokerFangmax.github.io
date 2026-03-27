---
title: "DSA 5: Binary Tree"
date: 2026-03-27
math: true
tags:
  - Notes
  - DSA
categories:
  - DSA
description: "DSA CHAP5"
---

# 数据结构与算法笔记（五）：二叉树

> 本文基于课程第五章“二叉树”整理。内容不只是按 PPT 顺序摘录，而是在保留课程主线的基础上，尽量补充树与二叉树的概念、树的多种表示、二叉树模板类设计、四种遍历的递归与迭代实现、后继节点、层次遍历、重构问题、Huffman 编码树，以及比较模型下的下界分析，整理成适合直接上传博客的 Markdown 笔记。

---

## 1. 为什么要研究树

### 1.1 树的应用背景

课件开头先讲“动机”，指出树结构非常适合表示**层次关系**，例如：

- 表达式
- 文件系统
- URL 结构
- 各种组织层级关系

这些对象的共同点是：

- 不是简单线性排列
- 存在明显的“父子 / 包含 / 从属”关系
- 同时又希望保留一定的顺序结构

因此树可以看作一种介于“线性结构”和“图结构”之间的中间对象。课件还特别用了“半线性（semi-linear）”这个词来描述它：  
它虽然不再是简单线性结构，但一旦指定某种遍历次序后，又会表现出明显的线性特征。 fileciteturn7file0

---

### 1.2 树兼具哪些优势

课件把树描述成一种很“综合”的数据结构：

- 兼具向量与列表的某些优点
- 同时兼顾较高效的查找、插入、删除

这句话背后真正想表达的是：

> 树能够在“随机访问的组织性”和“局部修改的灵活性”之间取得平衡。

例如：

- 纯线性结构虽然简单，但表达层次能力差
- 一般图虽然强大，但很多操作会过于复杂
- 树由于无环、连通、层次清晰，因此既容易描述，又容易分析

---

## 2. 树的基本概念

## 2.1 无根树与有根树

课件先从图论视角定义树：

> 树是**极小连通图**、也是**极大无环图**。 fileciteturn7file0

这两个说法其实是等价的：

- **极小连通图**：保持连通的前提下，再删任意一条边都会断开
- **极大无环图**：保持无环的前提下，再加任意一条边都会形成环

若图中有 `n` 个节点，则树一定有：

\[
e = n - 1
\]

条边。课件第 3～4 页反复强调了这个关系，并指出它在复杂度分析时常常可以用来把边数 `e` 直接等同于 `Θ(n)`。 fileciteturn7file0

---

### 2.2 有根树

在一棵树中指定任意一个节点 `r` 作为根（root）之后，这棵树就变成了**有根树（rooted tree）**。 fileciteturn7file0

一旦有了根节点，很多方向性的概念就随之出现：

- 父亲（parent）
- 孩子（child）
- 祖先（ancestor）
- 后代（descendant）
- 子树（subtree）

此时，根节点把整棵树自然地划分为若干棵以其孩子为根的子树。

---

### 2.3 有序树

若进一步规定：

- 根的第 1 棵子树、第 2 棵子树、……是有顺序的
- 或者说孩子之间存在“从左到右”的顺序

那么这棵树就叫**有序树（ordered tree）**。 fileciteturn7file0

这点非常重要，因为后面的**二叉树**其实天然就是有序树：

- 左孩子 ≠ 右孩子
- 左子树与右子树不能任意互换

---

## 2.4 路径、深度、高度

### 2.4.1 路径与长度

课件第 5 页定义了路径（path）：

> 一串节点依次经边相连构成的序列。 fileciteturn7file0

路径长度通常指边数，而不是节点数。  
例如从根到某节点若经过了 3 条边，则路径长度是 3。

---

### 2.4.2 深度（depth）

节点 `v` 的深度定义为：

\[
depth(v) = |path(v)|
\]

也就是从根到 `v` 的唯一路径长度。 fileciteturn7file0

特别地：

- 根节点深度为 0
- 深度相同的节点位于同一层

---

### 2.4.3 高度（height）

某节点 `v` 的高度 `height(v)` 定义为：

- 以 `v` 为根的子树中，所有叶节点深度的最大值（相对于 `v`）

对于整棵树，高度就是根节点的高度。  
课件特别规定：

> **空树高度取作 -1**。 fileciteturn7file0

这一约定在递归定义中非常方便，例如叶节点高度自然变成：

\[
1 + \max(-1,-1) = 0
\]

---

### 2.4.4 深度与高度的关系

课件给出了一个不等式：

\[
depth(v) + height(v) \le height(T)
\]

这里 `T` 是整棵树。 fileciteturn7file0

直观上很好理解：

- `depth(v)` 表示从根走到 `v`
- `height(v)` 表示从 `v` 再往下走到它最深的叶子
- 这两段拼起来，显然不会超过整棵树的最大根叶路径长度

当 `v` 恰好位于某条最长根叶路径上时取等号。

---

## 3. 树的几种表示方法

第五章第二部分先讲一般树的表示，然后才转到二叉树。这一步其实很重要，因为它说明：

> 二叉树不仅是一类特殊树，它还可以作为更一般有根有序树的统一表示方式。 fileciteturn7file0

---

### 3.1 仅记录父节点

课件第 11 页首先介绍了一种简单表示法：  
把所有节点放进一个序列，每个节点记录：

- `data`
- `parent`

即每个节点只知道自己的父亲是谁。 fileciteturn7file0

优点：

- `parent()` 可在 `O(1)` 时间得到
- 表示简单

缺点：

- 若要找孩子，就必须扫描整个节点集合
- `firstChild()` / `children()` 很低效

---

### 3.2 仅记录孩子列表

课件第 12 页又介绍另一种方式：  
每个节点不记父亲，而是直接维护一个“孩子序列”。 fileciteturn7file0

优点：

- 找孩子快
- 很适合从上往下遍历

缺点：

- 要找父亲时就变慢了，除非再做额外索引

---

### 3.3 父节点 + 孩子节点

课件第 13 页把前两种合并：

- 同时记录 `parent`
- 同时维护 `children`

这样几乎所有接口都能高效支持，但代价是：

- 结构更重
- 存储冗余更多 fileciteturn7file0

---

### 3.4 长子-兄弟表示法（firstChild + nextSibling）

课件第 14 页给出了一种非常经典的树表示方式：

- 每个节点保留两个引用：
  - `firstChild()`：长子
  - `nextSibling()`：右兄弟
- 若再加 `parent`，则可 `O(1)` 找父亲

这就是**长子-兄弟表示法**。 fileciteturn7file0

它的好处是：

1. 每个节点只需常数个指针
2. 能表示任意有根有序树
3. 后面还能自然地转成二叉树视角

这其实已经在暗示：

> **有根有序树 = 一棵二叉树的另一种解读。**

---

## 4. 二叉树：有根有序树的特例

## 4.1 二叉树的定义

课件第 16 页定义：

> 二叉树中每个节点的度数不超过 2，且孩子可明确区分为左、右。 fileciteturn7file0

因此二叉树天然是有序的：

- `lc()`：左孩子 / 左子树
- `rc()`：右孩子 / 右子树

注意这里“左右”非常关键。  
即使两个孩子都存在，也不能随便交换，否则会改变树的含义。

---

## 4.2 二叉树的规模性质

### 4.2.1 深度为 `k` 的节点数至多为 `2^k`

因为：

- 深度 0 最多 1 个
- 深度 1 最多 2 个
- 深度 2 最多 4 个
- …

于是高为 `h` 的二叉树，节点总数满足：

\[
h+1 \le n \le 2^{h+1} - 1
\]

课件第 17 页把这两个极端也画出来了： fileciteturn7file0

- `n = h + 1`：退化为一条单链
- `n = 2^{h+1} - 1`：满二叉树（full binary tree）

---

### 4.2.2 度为 0、1、2 的节点关系

设：

- `n0`：叶节点数（度为 0）
- `n1`：度为 1 的节点数
- `n2`：度为 2 的节点数

课件第 18 页给出：

\[
e = n - 1 = n_1 + 2n_2
\]

结合：

\[
n = n_0 + n_1 + n_2
\]

可推出：

\[
n_0 = n_2 + 1
\]

这是二叉树一个非常经典的性质，而且与 `n1` 无关。 fileciteturn7file0

---

### 4.2.3 真二叉树（proper binary tree）

若所有内部节点度数都恰为 2，也就是不存在单分支节点，那么叫**真二叉树（proper binary tree）**。课件第 19 页指出： fileciteturn7file0

- 通过引入适当外部节点（NULL 节点）
- 可把任意二叉树扩充为真二叉树

这在后面描述红黑树、Huffman 树等时都很有帮助，因为它能统一很多边界情况。

---

## 4.3 为什么“有根有序树 = 二叉树”

课件第 20 页明确指出：

> 任意有根且有序的多叉树，都可以转化并表示为二叉树。 fileciteturn7file0

转换规则正是长子-兄弟表示法：

- 长子 `firstChild()` → 左孩子 `lc()`
- 兄弟 `nextSibling()` → 右孩子 `rc()`

这意味着：

> 二叉树不仅是一类特殊树，它其实还是“有根有序树的统一底层表示形式”。

这也是为什么二叉树在数据结构课程里地位极高。

---

## 5. 二叉树的实现：BinNode 与 BinTree

## 5.1 `BinNode` 模板类

课件第 22 页给出的节点模板类大意如下： fileciteturn7file0

```cpp
template <typename T> using BinNodePosi = BinNode<T>*;

template <typename T> struct BinNode {
    BinNodePosi<T> parent, lc, rc;
    T data;
    int height;
    int size();

    BinNodePosi<T> insertAsLC(T const&);
    BinNodePosi<T> insertAsRC(T const&);
    BinNodePosi<T> succ();

    template <typename VST> void travLevel(VST&);
    template <typename VST> void travPre(VST&);
    template <typename VST> void travIn(VST&);
    template <typename VST> void travPost(VST&);
};
```

可以看出，一个节点里通常维护：

- `parent`：父节点
- `lc`：左孩子
- `rc`：右孩子
- `data`：数据
- `height`：高度缓存

---

## 5.2 `BinNode` 的几个基本接口

### 5.2.1 `insertAsLC()` / `insertAsRC()`

课件实现： fileciteturn7file0

```cpp
template <typename T>
BinNodePosi<T> BinNode<T>::insertAsLC(T const& e) {
    return lc = new BinNode(e, this);
}

template <typename T>
BinNodePosi<T> BinNode<T>::insertAsRC(T const& e) {
    return rc = new BinNode(e, this);
}
```

也就是把当前节点作为父亲，创建一个新的左 / 右孩子节点。

---

### 5.2.2 `size()`

课件把 `size()` 定义为：

> 以当前节点为根的子树规模。 fileciteturn7file0

代码为：

```cpp
template <typename T>
int BinNode<T>::size() {
    int s = 1;
    if (lc) s += lc->size();
    if (rc) s += rc->size();
    return s;
}
```

这是一个非常标准的后序递归：

- 先求左右子树规模
- 再加上自己

复杂度为：

\[
O(n)
\]

其中 `n` 是该子树节点数。

---

## 5.3 `BinTree` 模板类

课件第 24 页给出树类框架： fileciteturn7file0

```cpp
template <typename T>
class BinTree {
protected:
    int _size;
    BinNodePosi<T> _root;
    virtual int updateHeight(BinNodePosi<T> x);
    void updateHeightAbove(BinNodePosi<T> x);

public:
    int size() const { return _size; }
    bool empty() const { return !_root; }
    BinNodePosi<T> root() const { return _root; }
    /* 子树接入、删除和分离；遍历接口 ... */
};
```

树对象层面维护的是：

- `_root`
- `_size`

而局部节点细节则放在 `BinNode` 中。

---

## 6. 节点插入、子树接入、删除与分离

## 6.1 节点插入

课件第 25 页给出 `insert()`： fileciteturn7file0

```cpp
BinNodePosi<T> BinTree<T>::insert(T const& e, BinNodePosi<T> x) {
    _size++;
    x->insertAsLC(e);
    updateHeightAbove(x);
    return x->lc;
}
```

作为左孩子或右孩子插入后，需要做两件事：

1. 更新树规模 `_size`
2. 从 `x` 开始一路向上更新高度

---

### 6.2 为什么更新高度只需沿祖先链

因为插入一个新节点后：

- 被影响的只有该节点及其祖先
- 与之无关的其他子树高度不会变

所以课件第 27 页给出： fileciteturn7file0

```cpp
#define stature(p) ((p) ? (p)->height : -1)

int BinTree<T>::updateHeight(BinNodePosi<T> x) {
    return x->height = 1 + max(stature(x->lc), stature(x->rc));
}

void BinTree<T>::updateHeightAbove(BinNodePosi<T> x) {
    while (x) {
        updateHeight(x);
        x = x->parent;
    }
}
```

这里再次用到了“空树高度为 -1”的约定。

---

## 6.3 子树接入 `attach()`

课件第 26 页给出接口：  
把一整棵子树 `S` 接到节点 `x` 的左 / 右侧。 fileciteturn7file0

其核心逻辑包括：

1. 让 `x->lc` 或 `x->rc` 指向 `S->_root`
2. 更新被接入子树根节点的 `parent`
3. 把 `S->_size` 累加到当前树
4. 更新祖先高度
5. 释放原空壳树对象 `S`

这说明“接入子树”并不是逐节点复制，而是：

> **直接重连指针，把整棵树并进来。**

若设计得当，这种操作是非常高效的。

---

## 6.4 子树删除 `remove()`

课件第 28 页给出删除某节点为根的整棵子树： fileciteturn7file0

```cpp
int BinTree<T>::remove(BinNodePosi<T> x) {
    FromParentTo(*x) = NULL;
    updateHeightAbove(x->parent);
    int n = removeAt(x);
    _size -= n;
    return n;
}
```

内部递归：

```cpp
static int removeAt(BinNodePosi<T> x) {
    if (!x) return 0;
    int n = 1 + removeAt(x->lc) + removeAt(x->rc);
    release(x->data);
    release(x);
    return n;
}
```

可以看到：

- 真正释放节点的过程是一个后序递归
- 因为必须先删孩子，再删自己

---

## 6.5 子树分离 `secede()`

课件第 29 页给出：  
把某节点 `x` 为根的子树从原树中剪下来，并重新封装成一棵独立新树。 fileciteturn7file0

核心步骤：

1. 断开 `x` 与原父节点的连接
2. 更新原树祖先高度
3. 新建一棵空 `BinTree`
4. 让新树以 `x` 为根
5. 用 `x->size()` 计算新树规模
6. 从原树规模中扣除该部分

这和 `remove()` 的差别在于：

- `remove()`：释放整棵子树
- `secede()`：保留整棵子树，只是脱离原树

---

## 7. 二叉树遍历：总览

课件从第 31 页开始进入整章最核心的部分：遍历。 fileciteturn7file0

对任意二叉树：

\[
T = L \cup x \cup R
\]

遍历有三种最经典的 DFS 次序：

- **先序**：`x | L | R`
- **中序**：`L | x | R`
- **后序**：`L | R | x`

以及一种 BFS 次序：

- **层次遍历**：按深度从上到下、从左到右

课件特别强调：

> 遍历结果 ~ 遍历过程 ~ 遍历次序 ~ 遍历策略

也就是说，遍历不仅是“输出顺序”，也是组织递归 / 迭代算法的一种方式。 fileciteturn7file0

---

## 8. 先序遍历

## 8.1 递归实现

课件第 32 页给出标准递归： fileciteturn7file0

```cpp
template <typename T, typename VST>
void traverse(BinNodePosi<T> x, VST& visit) {
    if (!x) return;
    visit(x->data);
    traverse(x->lc, visit);
    traverse(x->rc, visit);
}
```

复杂度显然是：

\[
O(n)
\]

因为每个节点恰被访问一次。

---

## 8.2 观察：藤缠树

课件第 34 页给出一个非常漂亮的直观图景：  
把不断沿左孩子向下走的这条链看作一根“藤”。 fileciteturn7file0

先序遍历可以分解为：

1. 自上而下访问藤上的节点
2. 各节点的右子树被暂存起来
3. 再按适当顺序遍历这些右子树

这就自然引出迭代算法。

---

## 8.3 迭代实现：`visitAlongVine()`

课件第 36 页给出辅助函数： fileciteturn7file0

```cpp
static void visitAlongVine(
    BinNodePosi<T> x, VST& visit, Stack<BinNodePosi<T>>& S
) {
    while (x) {
        visit(x->data);
        S.push(x->rc);
        x = x->lc;
    }
}
```

其意义是：

- 访问沿左链所有节点
- 同时把每个节点的右孩子压栈，供以后处理

---

## 8.4 完整迭代版

课件第 37 页： fileciteturn7file0

```cpp
void travPre_I2(BinNodePosi<T> x, VST& visit) {
    Stack<BinNodePosi<T>> S;
    while (true) {
        visitAlongVine(x, visit, S);
        if (S.empty()) break;
        x = S.pop();
    }
}
```

### 为什么是 `O(n)`

- 每个节点被访问一次
- 每个右孩子最多入栈一次
- 每个入栈节点最多弹出一次

因此总复杂度仍然是：

\[
O(n)
\]

---

## 9. 中序遍历

## 9.1 递归实现

课件第 40 页： fileciteturn7file0

```cpp
void traverse(BinNodePosi<T> x, VST& visit) {
    if (!x) return;
    traverse(x->lc, visit);
    visit(x->data);
    traverse(x->rc, visit);
}
```

---

## 9.2 核心观察

中序遍历与先序的差别在于：

- 沿左链向下时，不能立刻访问
- 必须等左子树完全处理完后，才访问当前节点
- 然后再去右子树

课件第 42 页把它分解为：

- 沿左藤不断下探
- 自底向上依次访问这些节点
- 每访问一个节点后，再转入其右子树 fileciteturn7file0

---

## 9.3 迭代实现：`goAlongVine()`

课件第 44 页给出： fileciteturn7file0

```cpp
template <typename T>
static void goAlongVine(BinNodePosi<T> x, Stack<BinNodePosi<T>>& S) {
    while (x) {
        S.push(x);
        x = x->lc;
    }
}
```

它的作用是：

> 把当前节点及其沿左链所有祖先依次压栈。

---

## 9.4 完整迭代版

课件第 45 页： fileciteturn7file0

```cpp
void travIn_I1(BinNodePosi<T> x, V& visit) {
    Stack<BinNodePosi<T>> S;
    while (true) {
        goAlongVine(x, S);
        if (S.empty()) break;
        x = S.pop();
        visit(x->data);
        x = x->rc;
    }
}
```

### 正确性理解

每次节点出栈时：

- 它的左子树要么不存在，要么已处理完
- 它的右子树还没处理

因此此时访问它正好符合 `L | x | R`。  
课件第 48 页正是用这个不变式解释正确性的。 fileciteturn7file0

---

### 9.5 为什么总体仍是 `O(n)`

虽然单次 `goAlongVine()` 可能压很多节点，但课件第 49 页提醒：

- 每个节点只会入栈一次
- 也只会出栈一次

因此用**分摊分析**看，总复杂度仍然是：

\[
O(n)
\]

而不是 `O(n^2)`。 fileciteturn7file0

---

## 10. 中序后继 `succ()`

课件第 51～52 页讲了一个非常常用的局部操作：  
给定节点 `x`，求其中序遍历下的**直接后继**。 fileciteturn7file0

---

### 10.1 情况一：有右子树

若 `x` 有右孩子，则其中序后继必然是：

> 右子树中的最左节点

代码逻辑：

```cpp
s = rc;
while (HasLChild(*s)) s = s->lc;
```

因为中序次序是：

\[
L | x | R
\]

访问完 `x` 之后，下一步一定进入右子树，而右子树中最先访问的是最左节点。

---

### 10.2 情况二：没有右子树

若 `x` 没有右子树，则后继应是：

> 第一个把 `x` 放在其左子树中的祖先

因此要不断向上走，直到当前节点不再是其父亲的右孩子：

```cpp
while (IsRChild(*s)) s = s->parent;
s = s->parent;
```

最后这个祖先就是后继。  
若一直找不到，则说明 `x` 已是中序最后一个节点，后继为 `NULL`。 fileciteturn7file0

---

### 10.3 复杂度

两种情况下复杂度都不超过树高：

\[
O(h)
\]

其中 `h` 是树高度。 fileciteturn7file0

---

## 11. 后序遍历

## 11.1 递归实现

课件第 54 页： fileciteturn7file0

```cpp
void traverse(BinNodePosi<T> x, VST& visit) {
    if (!x) return;
    traverse(x->lc, visit);
    traverse(x->rc, visit);
    visit(x->data);
}
```

这类遍历特别适合做：

- 计算子树规模
- 更新高度
- 删除整棵子树

因为它总是先处理完孩子，再处理根。

---

## 11.2 迭代难点

后序遍历比先序 / 中序都更难迭代化，因为访问根节点时必须确保：

- 左子树已遍历
- 右子树也已遍历

课件第 56 页给出的观察是：

> 后序遍历第一次访问的一定是最左叶子。 fileciteturn7file0

于是可以从“如何走到当前子树的最左叶子”入手。

---

## 11.3 `gotoLeftmostLeaf()`

课件第 58 页： fileciteturn7file0

```cpp
static void gotoLeftmostLeaf(Stack<BinNodePosi<T>>& S) {
    while (BinNodePosi<T> x = S.top())
        if (HasLChild(*x)) {
            if (HasRChild(*x)) S.push(x->rc);
            S.push(x->lc);
        } else {
            S.push(x->rc);
        }
    S.pop();
}
```

它的核心思想是：

- 尽量沿左走
- 实在没左孩子才沿右
- 同时用栈记录“将来还要回来的路径”

---

## 11.4 完整后序迭代版

课件第 59 页： fileciteturn7file0

```cpp
void travPost_I(BinNodePosi<T> x, VST& visit) {
    Stack<BinNodePosi<T>> S;
    if (x) S.push(x);
    while (!S.empty()) {
        if (S.top() != x->parent)
            gotoLeftmostLeaf(S);
        x = S.pop();
        visit(x->data);
    }
}
```

---

### 11.5 为什么仍是 `O(n)`

与中序类似：

- 单次 `gotoLeftmostLeaf()` 可能花很久
- 但每个节点最多入栈 / 出栈常数次
- 用分摊分析看，总体仍是线性的

因此时间复杂度仍为：

\[
O(n)
\]

课件第 65 页明确要求用分摊分析理解它。 fileciteturn7file0

---

## 12. 层次遍历（BFS）

## 12.1 算法思想

层次遍历即广度优先遍历。  
课件第 70 页给出标准实现：借助辅助队列。 fileciteturn7file0

```cpp
template <typename T> template <typename VST>
void BinNode<T>::travLevel(VST& visit) {
    Queue<BinNodePosi<T>> Q;
    Q.enqueue(this);
    while (!Q.empty()) {
        BinNodePosi<T> x = Q.dequeue();
        visit(x->data);
        if (HasLChild(*x)) Q.enqueue(x->lc);
        if (HasRChild(*x)) Q.enqueue(x->rc);
    }
}
```

---

## 12.2 为什么队列天然适合层次遍历

因为 BFS 的本质就是：

- 先访问浅层
- 再访问深层
- 同层内部从左到右

当访问某节点时，把其孩子按左、右顺序入队。  
于是：

- 较浅层节点总更早出队
- 同层中较左节点更早出队

这正符合层次遍历定义。课件第 73 页对这一点做了详细分析。 fileciteturn7file0

---

## 12.3 复杂度

每个节点：

- 入队一次
- 出队一次
- 访问一次

因此总时间复杂度：

\[
O(n)
\]

---

## 12.4 完全二叉树与紧凑表示

课件第 75～78 页进一步讨论：

- 完全二叉树的层次遍历队列规模具有单峰对称结构
- 最大可能达到 `⌈n/2⌉`
- 完全二叉树尤其适合**向量式紧凑存储** fileciteturn7file0

这也是为什么完全二叉堆通常用数组存储，而普通二叉树更多用指针结构。

---

## 13. 二叉树重构

## 13.1 哪些遍历序列组合可以重构树

课件第 80～84 页讨论重构问题： fileciteturn7file0

- **先序 + 中序**：可以重构
- **后序 + 中序**：可以重构
- **先序 + 后序**：一般**不能唯一重构**

原因很简单：

- 中序提供了“左子树 / 根 / 右子树”的分界信息
- 先序或后序提供了根的位置
- 两者配合才足以递归分割

而仅有先序 + 后序时，左右子树边界可能不唯一。

---

## 13.2 增强序列：把 `NULL` 也输出

课件第 85～86 页提出一个非常漂亮的办法：  
在遍历时把空孩子也作为“真实节点”输出，统一记成元字符 `^` 或 `NULL`。 fileciteturn7file0

例如增强先序 / 中序 / 后序中：

- 每个空指针也占一个位置
- 一个子树对应一个连续子序列
- 且其中空节点总比非空节点多 1

于是：

> **单独一份增强后的先序、或中序、或后序，就足以唯一重构原树。**

这是因为原本丢失的边界信息，被这些 `NULL` 标记补回来了。

---

## 14. 表达式树与后序遍历

课件第 66～68 页指出：

- 表达式树
- 后序遍历
- 逆波兰表达式（RPN）

三者天然对应。 fileciteturn7file0

例如表达式：

\[
(a+b)	imes c
\]

其表达式树中：

- 内部节点是运算符
- 叶子节点是操作数

若做后序遍历，就会得到：

```text
a b + c *
```

这正是 RPN。

所以：

> **后序遍历 = 表达式树的自然求值 / 线性化顺序**

这也是为什么上一章的 RPN 与这一章的表达式树能自然衔接起来。

---

## 15. Huffman 编码树

二叉树在本章一个非常重要的应用是 **Huffman 编码树**。  
课件从第 87 页开始专门讲这一部分。 fileciteturn7file0

---

## 15.1 编码问题的目标

若字符集为 `Σ`，希望为每个字符分配一个二进制串，使得整个文件尽量短。  
文件总长度取决于：

- 各字符出现频率
- 对应编码长度

因此问题变成：

> 如何构造一棵二叉编码树，使平均带权编码长度最小？

---

## 15.2 前缀无歧义编码（Prefix-Free Code）

课件第 89 页先讲一种树形编码：

- 左边记 0
- 右边记 1
- 每个字符放在叶节点上
- 根到该叶的路径就是字符编码 fileciteturn7file0

这样的编码天然具有：

> **任一字符编码都不是另一个字符编码的前缀**

因此解码时不会有歧义。  
这就是前缀码（Prefix-Free Code, PFC）。

---

## 15.3 平均带权路径长度

若字符 `x` 的出现频率为 `w(x)`，其编码长度等于在树中的深度 `depth(v(x))`。  
则整棵树的平均带权编码长度可写为：

\[
WPL(T) = \sum_x w(x)\cdot depth(v(x))
\]

课件用 `wald(T)` 等记号表达这一思想。 fileciteturn7file0

最优 Huffman 树的目标就是最小化这个值。

---

## 15.4 最优编码树的性质

课件第 91～99 页给出了几个关键性质： fileciteturn7file0

### 15.4.1 双子性

最优编码树一定是真二叉树。  
否则若某内部节点只有一个孩子，可以直接把它压缩掉，使带权路径长度更小。

---

### 15.4.2 最低频字符应最深

频率低的字符应尽量放在更低层；频率高的字符应尽量靠近根。  
这很符合直觉：

- 高频字符希望编码更短
- 低频字符可以容忍更长编码

---

### 15.4.3 最低频的两个字符必互为兄弟

这是 Huffman 贪心算法正确性的核心基础之一。  
课件指出：

> 频率最低的两个字符，在某棵最优编码树中必位于最底层并互为兄弟。 fileciteturn7file0

这使得我们可以把它们“合并成一个超字符”，递归处理更小问题。

---

## 15.5 Huffman 算法

课件第 95 页给出贪心策略： fileciteturn7file0

1. 为每个字符建立一棵单节点树
2. 构成一个森林 `F`
3. 反复执行，直到只剩一棵树：
   - 取出权重最小的两棵树 `T1, T2`
   - 合并成新树 `T`
   - 新根权重为两者之和
   - 再把新树放回森林

这正是经典 Huffman 算法。

---

## 15.6 正确性：为什么贪心是对的

课件第 97～102 页从“双子性”“层次性”“数学归纳”三个角度证明： fileciteturn7file0

- 最低频两个字符可视作互为兄弟
- 合并它们形成新字符 `z`
- 对缩小后的字符集 `Σ'` 求得最优树 `T'`
- 再把 `z` 展开回 `x,y`
- 就得到原问题的一棵最优树

因此 Huffman 每一步“合并最小两棵树”的贪心选择是安全的。

---

## 15.7 代码实现中的数据结构

课件第 105～111 页给出实现思路： fileciteturn7file0

- `HuffChar`：字符 + 频率
- `HuffTree = BinTree<HuffChar>`
- `HuffForest`：一组 Huffman 树构成的森林

初始实现里，森林用列表维护；  
每次通过 `minHChar()` 在线性时间找出最小两棵树，于是总复杂度是：

\[
O(n^2)
\]

---

## 15.8 如何优化到 `O(n log n)`

课件第 111 页指出三种底层实现方式： fileciteturn7file0

1. 有序向量：总体 `O(n^2)`
2. 有序列表：总体 `O(n^2)`
3. **优先级队列（堆）**：总体 `O(n log n)`

因为 Huffman 算法的瓶颈就是：

- 反复取最小
- 再插回新权值

这恰好是优先级队列最擅长的场景。

---

## 16. 比较模型下的下界：判定树

第五章最后一部分其实已经从“树作为数据结构”转向“树作为分析工具”。  
课件第 114 页开始讲： fileciteturn7file0

> 同一问题的不同算法复杂度可能相差很大，那么问题本身有没有“难度下界”？

---

## 16.1 问题难度与最优算法

若问题 `P` 存在算法，则所有算法中最低可达的复杂度，就是问题 `P` 的难度。  
研究问题难度通常有两个方向：

1. 设计更快的算法
2. 证明更高的复杂度下界

当某算法达到下界时，就可说它在大 O 意义下已经最优。课件第 114 页以排序问题为例说明这一点。 fileciteturn7file0

---

## 16.2 比较树 / 判定树

课件第 117～119 页引入：

- Comparison Tree
- Algebraic Decision Tree

其中最简单的是**比较树**：  
每个内部节点表示一次关键码比较，例如：

\[
K_i - K_j
\]

比较结果可能是：

- `<`
- `=`
- `>`

因此对应一棵三叉树。 fileciteturn7file0

---

## 16.3 为什么排序下界是 `Ω(n log n)`

对于基于比较的排序算法：

- 每个输入排列都必须对应某个叶节点
- 因此至少需要区分 `n!` 种可能输出
- 一棵高度为 `h` 的常叉树能区分的叶子数至多指数级于 `h`

所以必须满足：

\[
3^h \ge n!
\]

或在二叉比较模型下：

\[
2^h \ge n!
\]

由 Stirling 近似可推出：

\[
h = \Omega(n \log n)
\]

因此：

> 任何基于比较的排序算法，在最坏情况下都至少需要 `Ω(n log n)` 时间。 fileciteturn7file0

这也是归并排序、堆排序等算法“最优”的理论基础。

---

## 16.4 归约（Reduction）

课件第 121～122 页还提到另一个证明下界的重要工具：**归约**。 fileciteturn7file0

思路是：

- 若已知问题 `U` 很难
- 且能在线性时间把 `U` 归约到问题 `W`
- 那么 `W` 不可能比 `U` 更容易

例如课件给出一些例子：

- Element Uniqueness 的下界可归约到 Closest Pair
- Sorting 可归约到 Huffman Tree / Optimal Encoding Tree 等

这说明“树”不仅是结构对象，也可以作为复杂度理论中的证明工具。

---

## 17. 本章主线总结

### 17.1 树是层次结构最自然的抽象

与向量、列表相比，树更适合表示：

- 从属关系
- 包含关系
- 递归结构
- 组合结构

### 17.2 二叉树不仅是特殊树，也是一般有根有序树的统一表示

通过长子-兄弟表示法：

- 有根有序多叉树
- 可以自然编码成二叉树

### 17.3 二叉树的递归定义决定了它的遍历方式

- 先序：根优先
- 中序：左根右
- 后序：孩子优先
- 层次：按深度扩展

并且所有递归遍历都能改写成高效的迭代算法。

### 17.4 树的很多性质都依赖“局部 + 递归”思维

- 子树规模 `size()`
- 高度更新
- 子树接入 / 删除 / 分离
- 后继节点
- 树重构
- 表达式树与 RPN

都体现了这一点。

### 17.5 Huffman 树展示了树在贪心算法与编码中的威力

- 最优前缀码
- 带权路径长度最小化
- 贪心合并最小权重
- 优先级队列加速到 `O(n log n)`

### 17.6 判定树说明“树”还可以作为算法下界分析工具

- 比较树刻画了基于比较算法的执行过程
- 排序下界 `Ω(n log n)` 正是由此得到

---

## 18. 一句话总结

> 二叉树这一章真正重要的，不只是学会节点指针和四种遍历，而是理解：树既是一种表示层次世界的数据结构，也是一种组织递归过程、设计贪心算法、分析复杂度下界的统一语言。
