---
title: "DSA 6: BST"
date: 2026-03-27
math: true
tags:
  - Notes
  - DSA
categories:
  - DSA
description: "DSA CHAP6"
---

# 数据结构与算法笔记（六）：二叉搜索树

> 本文基于课程第六章“二叉搜索树”整理。内容不只是逐页复述课件，而是在保留课程主线的基础上，系统补充 BST 的顺序性、查找 / 插入 / 删除的实现逻辑、树高与平均高度、平衡的必要性、AVL 树的平衡因子与高度界、插入 / 删除后的重平衡，以及统一的 `(3+4)` 重构与 `rotateAt()` 框架，整理成适合直接上传博客的 Markdown 笔记。主要依据你上传的课件内容：fileciteturn8file0

---

## 1. 为什么需要二叉搜索树

### 1.1 线性结构的两难

课件一开始先回顾了前面学过的几类线性结构，并指出一个很现实的问题：

- 无序向量：查找 `O(n)`，插入 / 删除 `O(n)`
- 有序向量：查找 `O(log n)`，插入 / 删除 `O(n)`
- 无序列表：查找 `O(n)`，插入 / 删除 `O(1)`
- 有序列表：查找 `O(n)`，插入 / 删除 `O(n)` fileciteturn8file0

这张对比表传达的核心是：

> 向量、列表都不能同时兼顾**高效查找**与**高效动态修改**。

因此课程提出一个自然问题：

> 能否综合二者的优点？

这正是二叉搜索树（BST, Binary Search Tree）出现的动机。

### 1.2 从“循秩访问”转向“循关键码访问”

前几章的数据结构，大多围绕：

- 下标（rank）
- 位置（position）

来访问元素。

而 BST 引入的是：

> **call-by-key（循关键码访问）**。 fileciteturn8file0

也就是说，数据项不再主要通过“排在第几个”来区分，而是通过其**关键码（key）**来区分。

这意味着：

- 关键码之间必须支持大小比较
- 同时还要支持相等判定

因此课件把数据项统一包装成了 **Entry（词条）** 形式。

---

## 2. 词条（Entry）与关键码

### 2.1 为什么要引入词条

课件第 3 页定义了如下模板：fileciteturn8file0

```cpp
template <typename K, typename V> struct Entry {
    K key;
    V value;
    Entry(K k = K(), V v = V()) : key(k), value(v) {}
    Entry(Entry<K, V> const& e) : key(e.key), value(e.value) {}

    bool operator<(Entry<K, V> const& e)  { return key < e.key; }
    bool operator>(Entry<K, V> const& e)  { return key > e.key; }
    bool operator==(Entry<K, V> const& e) { return key == e.key; }
    bool operator!=(Entry<K, V> const& e) { return key != e.key; }
};
```

它的作用是把：

- **关键码 key**
- **关联数值 value**

统一打包成一个对象。

### 2.2 为什么后面“节点 ~ 词条 ~ 关键码”经常不严格区分

课件第 5 页明确写出：

> 三位一体：节点 ~ 词条 ~ 关键码。 fileciteturn8file0

这并不是偷懒，而是因为在 BST 中，顺序性完全由关键码决定。  
所以很多时候：

- 说“查找节点”
- 实际是在查找“其关键码为 e 的词条”
- 进一步可近似成“查找关键码 e”

这种简化在理论分析里很常见，但实际代码层面仍要清楚：

- 节点是树结构单位
- 词条是节点中存的数据
- 关键码是词条中决定顺序的字段

---

## 3. BST 的定义与顺序性

### 3.1 BST 的核心性质

课件第 5 页给出 BST 的顺序性定义：

> 任一节点均不小于其左后代，且不大于其右后代。 fileciteturn8file0

更常见的表达是：

- 左子树中所有关键码 `<= v`
- 右子树中所有关键码 `>= v`

在本章初期为简化起见，课件暂时假设**关键码互异**，于是可以理解成严格小于 / 大于。  
第 6 页也特别说明，这只是教学简化，实际中重复关键码完全可以处理，例如通过 `searchAll()`、`searchFirst()` 等机制。 fileciteturn8file0

### 3.2 为什么“只比较左右孩子”还不够

课件在第 5 页还提出一个很值得注意的问题：

> “任一节点均不小于 / 不大于其左右孩子”，是否与上面的定义等价？

答案是：**不完全等价**。

因为 BST 真正要求的是：

- 左子树所有后代都不大于当前节点
- 右子树所有后代都不小于当前节点

如果只检查左右孩子，仍可能漏掉更深层后代的越界情况。

### 3.3 BST 的整体性质：中序遍历单调非降

课件第 7 页给出 BST 最重要的全局性质：

> BST 的中序遍历序列，必然单调非降。 fileciteturn8file0

这个结论可以通过对树高做数学归纳证明：

- 对空树或单节点树显然成立
- 若左、右子树本身中序有序
- 且左子树所有元素 `<= root <=` 右子树所有元素
- 则整棵树中序自然有序

这条性质非常重要，因为它说明：

> **BST = 把“有序序列”嵌入到一棵树形结构中。**

于是后面的查找过程就很像“树上的二分查找”。

---

## 4. BST 的接口设计

### 4.1 对外接口

课件第 9 页给出了 BST 类的公开接口：fileciteturn8file0

```cpp
template <typename T> class BST : public BinTree<T> {
public:
    virtual BinNodePosi<T>& search(const T&);
    virtual BinNodePosi<T> insert(const T&);
    virtual bool remove(const T&);
protected:
    /* ...... */
};
```

这三个接口正是 BST 最核心的动态操作：

- `search(e)`：查找关键码为 `e` 的节点
- `insert(e)`：插入关键码为 `e` 的新节点
- `remove(e)`：删除关键码为 `e` 的节点

### 4.2 为什么 `search()` 返回的是“节点位置的引用”

这里最特别的是：

```cpp
BinNodePosi<T>& search(const T&);
```

注意返回类型不是普通指针，而是**指针的引用**。  
课件第 13～14 页反复强调这一设计。fileciteturn8file0

它的好处在于：

- 若查找成功，返回的是某个真实节点指针的引用
- 若查找失败，返回的是最后一次试图转向的那个 `NULL` 指针位置的引用

于是后续插入、删除时，就可以直接修改这个引用所对应的父子链接，而不必再额外区分“我是父亲的左孩子还是右孩子”。

### 4.3 内部辅助成员 `_hot`

课件第 10 页给出 BST 的内部成员：fileciteturn8file0

```cpp
BinNodePosi<T> _hot; // 命中节点的父亲
```

更精确地说，`_hot` 表示：

> 查找过程中最后访问到的非空节点，也即返回位置 `v` 的父亲。

因此：

- 若查找成功，则 `_hot` 是命中节点的父亲（若命中根则为 `NULL`）
- 若查找失败，则 `_hot` 是那个失败空位的父亲

这个变量在 `insert()` 和 `remove()` 中都非常关键。

---

## 5. 查找：树上的“仿射二分”

### 5.1 查找思想

课件第 12 页把 BST 查找概括为：

> 从根节点出发，逐步缩小查找范围，直到发现目标或抵达空树。  
> 对照中序遍历序列可见，整个过程可视作是在仿效有序向量的二分查找。 fileciteturn8file0

为什么说它像二分查找？

因为当前节点 `x` 相当于把全集分成三部分：

- 左子树：全部比 `x` 小
- 节点自身：等于 `x`
- 右子树：全部比 `x` 大

于是每比较一次，就能把搜索范围缩到左或右子树。

### 5.2 代码实现

课件第 13 页给出实现：fileciteturn8file0

```cpp
template <typename T>
BinNodePosi<T>& BST<T>::search(const T& e) {
    if (!_root || e == _root->data) {
        _hot = NULL;
        return _root;
    }
    for (_hot = _root; ; ) {
        BinNodePosi<T>& v = (e < _hot->data) ? _hot->lc : _hot->rc;
        if (!v || e == v->data) return v;
        _hot = v;
    }
}
```

### 5.3 这段代码真正精妙的地方

最关键的一句是：

```cpp
BinNodePosi<T>& v = (e < _hot->data) ? _hot->lc : _hot->rc;
```

它不是得到“某个节点”，而是直接得到：

- `_hot->lc` 的引用，或
- `_hot->rc` 的引用

因此后续：

- 若 `v` 非空且命中，直接返回
- 若 `v` 为空，也直接返回该空位引用

于是 `search()` 返回值就天然兼具：

- “查找结果”
- “插入入口”

两重语义。

### 5.4 查找失败时返回的到底是什么

课件第 14 页专门解释：

- 成功时：指向一个真实存在且关键码为 `e` 的节点
- 失败时：指向最后一次试图转向的空节点 `NULL` fileciteturn8file0

甚至可以把这个失败位置理解为：

> “如果随后要插入一个关键码为 `e` 的新节点，它就应该填到这里。”

### 5.5 查找复杂度

每一步都会下移一层，因此复杂度正比于节点深度，不超过树高：

`O(h)`

其中 `h` 是 BST 的高度。 fileciteturn8file0

---

## 6. 插入：把失败的查找位置变成新叶子

### 6.1 插入思想

课件第 16 页把插入讲得非常直观：fileciteturn8file0

1. 先调用 `search(e)` 找到应插入的位置
2. 若关键码 `e` 不存在，则此时返回值 `x` 正好是那个 `NULL` 空位
3. 把该空位替换成一个新节点
4. 新节点的父亲就是 `_hot`

因此 BST 插入其实就是：

> **先查找，再把失败位置转正。**

### 6.2 插入代码

课件第 17 页：fileciteturn8file0

```cpp
template <typename T>
BinNodePosi<T> BST<T>::insert(const T& e) {
    BinNodePosi<T>& x = search(e);
    if (!x) {
        x = new BinNode<T>(e, _hot);
        _size++;
        updateHeightAbove(x);
    }
    return x;
}
```

### 6.3 为什么新节点一定插成叶子

因为 `search(e)` 失败时返回的是一个 `NULL` 空位。  
该位置本来就没有子树，所以新节点插进去后自然没有孩子，只能是一个叶子。

### 6.4 为什么这样能保持 BST 顺序性

因为这个空位是按查找路径找到的：

- 若走向左子树，说明 `e` 比祖先更小
- 若走向右子树，说明 `e` 比祖先更大

最终抵达的空位，恰好是“所有路径比较结果都满足”的位置。  
因此把 `e` 插进去后 BST 顺序性自动保持。

### 6.5 插入复杂度

插入的主要成本在：

1. `search(e)`
2. `updateHeightAbove(x)`

它们都与节点深度成正比，因此总复杂度仍为：

`O(h)`

---

## 7. 删除：两大类情况

BST 删除是本章第一个真正复杂的操作。

课件第 19 页给出主算法：fileciteturn8file0

```cpp
template <typename T>
bool BST<T>::remove(const T& e) {
    BinNodePosi<T>& x = search(e);
    if (!x) return false;
    removeAt(x, _hot);
    _size--;
    updateHeightAbove(_hot);
    return true;
}
```

这里的关键在于内部辅助函数：

```cpp
removeAt(x, _hot)
```

它真正处理“删哪个节点、如何接替”的逻辑。

### 7.1 单分支或零分支情况

#### 7.1.1 思想

课件第 20～21 页先讨论较简单情况：  
若待删节点 `x` 的某一子树为空，则直接用另一子树替换它。 fileciteturn8file0

也就是：

- 左子树空，则用右子树替换
- 右子树空，则用左子树替换
- 两边都空，则等价于删叶子

#### 7.1.2 代码

课件第 21 页：fileciteturn8file0

```cpp
template <typename T>
static BinNodePosi<T>
removeAt(BinNodePosi<T>& x, BinNodePosi<T>& hot) {
    BinNodePosi<T> w = x;
    BinNodePosi<T> succ = NULL;

    if (!HasLChild(*x)) succ = x = x->rc;
    else if (!HasRChild(*x)) succ = x = x->lc;
    else { /* ...双分支... */ }

    hot = w->parent;
    if (succ) succ->parent = hot;
    release(w->data); release(w);
    return succ;
}
```

#### 7.1.3 为什么顺序性仍然保持

如果左子树为空，用右子树替代：

- 右子树中所有元素原本就比 `x` 大
- 同时它们也符合 `x` 所在位置对祖先的所有约束

因此直接接上去后，BST 顺序性不变。右子树为空时同理。

#### 7.1.4 复杂度

这一类情况只需修改常数个链接，因此 `removeAt()` 本身是 `O(1)`。 fileciteturn8file0

### 7.2 双分支情况：用后继化归

#### 7.2.1 思想

若待删节点 `x` 左、右子树都存在，课件第 22 页指出：

> 调用 `succ()` 找到 `x` 的直接后继 `w`（它必无左孩子）；  
> 再交换 `x` 与 `w` 的数据；  
> 问题就转化为删除节点 `w`。 fileciteturn8file0

#### 7.2.2 为什么后继一定“至多一侧有孩子”

在 BST 中，`x` 的中序后继是右子树中最左的那个节点。  
因此它不可能有左孩子，否则还能继续往左走，就不是“最左”了。  
所以后继节点 `w` 至多只有右孩子。

#### 7.2.3 代码实现

课件第 23 页：fileciteturn8file0

```cpp
else {
    w = w->succ();
    swap(x->data, w->data);
    BinNodePosi<T> u = w->parent;
    (u == x ? u->rc : u->lc) = succ = w->rc;
}
```

#### 7.2.4 为什么只交换数据而不交换节点

因为交换数据最简单：

- 不破坏树的拓扑结构
- 不需要大规模改指针

交换后，“值为 e 的数据”被移到了后继节点 `w` 处；  
删除 `w` 就等于逻辑上删除原节点 `x` 的值。

#### 7.2.5 删除复杂度

双分支删除的额外成本主要在 `succ()`。  
因此 BST 删除整体复杂度仍为：

`O(h)`

---

## 8. BST 的性能瓶颈：树高

### 8.1 三个核心接口都受树高支配

课件第 25 页直接总结：

- `search()`
- `insert()`
- `remove()`

在最坏情况下，运行时间都线性正比于树高 `O(h)`。 fileciteturn8file0

这把问题一下子聚焦到了 BST 的本质：

> **BST 快不快，不在于“有没有顺序性”，而在于“树高能否控制住”。**

### 8.2 最坏情况：退化为列表

如果不断插入单调序列，例如：

```text
1, 2, 3, 4, 5, ...
```

那么 BST 会一路向右生长，最终退化成单链。  
此时树高：

`h = n - 1`

于是所有操作都退化为：

`O(n)`

课件第 25 页对此讲得非常明确。 fileciteturn8file0

---

## 9. 平均树高：两种随机模型

课件第 26～28 页给出两种统计口径，它们结论差异很大。fileciteturn8file0

### 9.1 随机生成模型：随机排列依次插入

设 `n` 个互异关键码，按一个随机排列依次插入空 BST。  
若假设各排列等概率出现（概率 `1/n!`），则平均树高是：

`Θ(log n)`

课件第 26 页给出这一结论，并指出多数实际应用中的 BST 总体上都“更接近这样逐步生成和演化”。 fileciteturn8file0

### 9.2 随机组成模型：所有 BST 形态等概率

另一种统计口径是：

- 不看插入过程
- 直接看由 `n` 个互异节点能组成多少棵不同 BST
- 假设这些 BST 形态都等概率出现

课件第 27 页指出，这时总数满足 Catalan 递推，且平均树高会变成：

`Θ(√n)`

这个结论明显比 `Θ(log n)` 差得多。 fileciteturn8file0

### 9.3 为什么两个模型结论差这么多

课件第 28 页对此做了点评：

- `log n` 结论偏乐观
- `sqrt(n)` 结论偏悲观
- 现实输入通常既不完全随机，也远非“所有 BST 形态等概率” fileciteturn8file0

更关键的是：

> 实际数据常常具有局部性、关联性、单调性、近似周期性……

因此较高、甚至很高的 BST 并不稀奇。  
这就说明：

> **靠“运气好就矮”是不可靠的，平衡化处理很有必要。**

---

## 10. 理想平衡与渐近平衡

### 10.1 理想平衡

课件第 30 页先定义一种很严格的目标：

> 节点数固定时，兄弟子树高度越接近，全树越低。  
> 若二叉树高度达到理论下界，则称作理想平衡。 fileciteturn8file0

这种树大致接近：

- 完全二叉树
- 甚至满二叉树

它要求叶子只出现在最底部两层，条件过于苛刻。

### 10.2 渐近平衡

课件第 31 页指出：

> 理想平衡出现概率太低、维护成本太高，因此应适当放松。  
> 只要高度渐近地不超过 `O(log n)`，即可接受。 fileciteturn8file0

这就是所谓：

> **平衡二叉搜索树（BBST）**

---

## 11. 等价 BST 与旋转

### 11.1 什么叫等价 BST

课件第 33 页指出：

- 上下可变：父子承袭关系可能改变
- 左右不乱：中序遍历序列必须完全一致 fileciteturn8file0

因此两棵 BST 若：

- 中序遍历序列相同
- 关键码集合相同

则可视作**等价 BST**。

### 11.2 局部性与旋转

课件第 34～35 页强调，所有 BBST 的重平衡都依赖两个思想：fileciteturn8file0

1. **限制条件 + 局部性**  
   单次动态修改后，至多少数局部位置失衡

2. **等价变换 + 旋转调整**  
   通过常数次旋转，在局部把失衡修复

所谓基本旋转包括：

- `zig`
- `zag`

它们只涉及常数个节点与若干子树，时间是 `O(1)`。

---

## 12. AVL 树：最早的平衡 BST

### 12.1 平衡因子

课件第 41 页给出了宏定义：fileciteturn8file0

```cpp
#define Balanced(x) ( stature((x).lc) == stature((x).rc) )
#define BalFac(x) ( stature((x).lc) - stature((x).rc) )
#define AvlBalanced(x) ( (-2 < BalFac(x)) && (BalFac(x) < 2) )
```

因此 AVL 的平衡条件是：

`-1 <= BalFac(x) <= 1`

### 12.2 AVL 树的定义

AVL 树是满足如下条件的 BST：

> 任一节点左右子树高度差绝对值不超过 1。 fileciteturn8file0

它未必理想平衡，但一定是**渐近平衡**。

### 12.3 AVL 高度为什么是 `O(log n)`

课件第 38～39 页给出经典分析：  
设高度为 `h` 的 AVL 树所含最少节点数为 `S(h)`，则有递推：

`S(h) = 1 + S(h-1) + S(h-2)`

这与 Fibonacci 递推同型，因此最瘦 AVL 树也叫：

> **Fibonaccian Tree** fileciteturn8file0

由此可得 AVL 的高度至多为 `O(log n)`。

---

## 13. AVL 失衡：插入与删除有何不同

### 13.1 插入后的失衡特点

课件第 42 页指出：

- 插入后，祖先链上多个节点都可能失衡
- 但只要修复**最低失衡节点**
- 其子树高度就会恢复原状
- 更高祖先也随之自动恢复平衡 fileciteturn8file0

这就是为什么 AVL 插入时：

> **只需做一次重平衡就够了。**

### 13.2 删除后的失衡特点

而删除后情况不同。  
课件第 49～51 页指出：

- 首个可能失衡的节点是 `_hot`
- 修复某个失衡点后，其子树高度**未必恢复原值**
- 因此更高祖先仍可能继续失衡
- 最多可能一路向上传播 `O(log n)` 次 fileciteturn8file0

所以：

- AVL 插入：最多一次调整
- AVL 删除：最坏可能 `Ω(log n)` 次调整

---

## 14. AVL 插入

### 14.1 算法框架

课件第 47 页给出 AVL 插入代码：fileciteturn8file0

```cpp
template <typename T>
BinNodePosi<T> AVL<T>::insert(const T& e) {
    BinNodePosi<T>& x = search(e);
    if (x) return x;
    BinNodePosi<T> xx = x = new BinNode<T>(e, _hot);
    _size++;
    for (BinNodePosi<T> g = _hot; g; g = g->parent)
        if (!AvlBalanced(*g)) {
            FromParentTo(*g) = rotateAt(tallerChild(tallerChild(g)));
            break;
        } else
            updateHeight(g);
    return xx;
}
```

### 14.2 为什么只处理“最低失衡点 g”

设新节点为 `x`。  
插入后，沿 `x` 到根的路径上，可能有多个祖先高度变化。  
但课件第 45～46 页强调：

- 最低失衡点 `g` 不低于 `x` 的祖父
- 以 `g` 为根做一次单旋或双旋后
- 局部子树高度恢复到插入前
- 因而更高祖先不再可能继续失衡 fileciteturn8file0

所以找到最低失衡点后即可 `break`。

### 14.3 四种失衡类型

令：

- `g`：最低失衡节点
- `p`：`g` 的较高孩子
- `v`：`p` 的较高孩子

则有四种相对位置：

1. LL（zig-zig）
2. LR（zig-zag）
3. RL（zag-zig）
4. RR（zag-zag）

课件第 45～46 页分别用单旋和双旋图展示了这些情况。 fileciteturn8file0

---

## 15. AVL 删除

### 15.1 算法框架

课件第 51 页代码如下：fileciteturn8file0

```cpp
template <typename T>
bool AVL<T>::remove(const T& e) {
    BinNodePosi<T>& x = search(e);
    if (!x) return false;
    removeAt(x, _hot);
    _size--;
    for (BinNodePosi<T> g = _hot; g; g = g->parent) {
        if (!AvlBalanced(*g))
            g = FromParentTo(*g) = rotateAt(tallerChild(tallerChild(g)));
        updateHeight(g);
    }
    return true;
}
```

### 15.2 与插入版的最大差异

差异只有一行，但意义巨大：

- 插入版：重平衡一次后 `break`
- 删除版：重平衡后**继续往上检查**

因为删除会让局部子树高度下降，而这种下降可能层层向上传播。  
所以 AVL 删除不能在第一次复衡后就停止。

### 15.3 复杂度

- 查找目标：`O(log n)`
- 删除节点：`O(log n)`
- 向上检查与重平衡：最坏 `O(log n)` 次

因此 AVL 删除最坏复杂度仍是 `O(log n)`。

---

## 16. `(3+4)` 重构：统一四种旋转情况

### 16.1 为什么要统一

如果分别手写：

- LL
- LR
- RL
- RR

四种旋转代码，容易又长又乱。  
课件第 52～55 页提出了更优雅的做法：

> 把祖孙三代 `g, p, v` 和四棵子树统一按中序重命名为  
> `a < b < c` 与 `T0 < T1 < T2 < T3`，  
> 然后通过一次 `(3+4)` 重构直接恢复平衡。 fileciteturn8file0

### 16.2 `connect34()`

课件第 54 页给出实现：fileciteturn8file0

```cpp
template <typename T>
BinNodePosi<T> BST<T>::connect34(
    BinNodePosi<T> a, BinNodePosi<T> b, BinNodePosi<T> c,
    BinNodePosi<T> T0, BinNodePosi<T> T1,
    BinNodePosi<T> T2, BinNodePosi<T> T3
) {
    a->lc = T0; if (T0) T0->parent = a;
    a->rc = T1; if (T1) T1->parent = a;
    c->lc = T2; if (T2) T2->parent = c;
    c->rc = T3; if (T3) T3->parent = c;
    b->lc = a; a->parent = b;
    b->rc = c; c->parent = b;
    updateHeight(a); updateHeight(c); updateHeight(b);
    return b;
}
```

其结果是：

- `b` 成为新子树根
- `a` 作为左子树根
- `c` 作为右子树根
- `T0..T3` 依中序顺序接回

### 16.3 `rotateAt()`

课件第 55 页给出统一旋转框架：fileciteturn8file0

```cpp
template<typename T>
BinNodePosi<T> BST<T>::rotateAt(BinNodePosi<T> v) {
    BinNodePosi<T> p = v->parent, g = p->parent;
    if (IsLChild(*p))
        if (IsLChild(*v)) { // zig-zig
            p->parent = g->parent;
            return connect34(v, p, g, v->lc, v->rc, p->rc, g->rc);
        } else { // zig-zag
            v->parent = g->parent;
            return connect34(p, v, g, p->lc, v->lc, v->rc, g->rc);
        }
    else { /* zag-zig & zag-zag */ }
}
```

这说明：

- 四种旋转情况最终都能归约到同一个 `connect34()`
- 差别只在于 `a,b,c,T0,T1,T2,T3` 的对应关系不同

---

## 17. AVL 的综合评价

课件第 56 页对 AVL 做了一个平衡评价：fileciteturn8file0

### 17.1 优点

- 查找 / 插入 / 删除的最坏复杂度都是 `O(log n)`
- 总存储空间仍为 `O(n)`

### 17.2 缺点

- 需要额外维护高度或平衡因子
- 插入 / 删除后的旋转有额外代价
- 删除操作后最坏可能需要 `Ω(log n)` 次旋转
- 若动态更新非常频繁，实际常数开销不一定理想

课件还引用 Knuth 的结果说：

- 删除后平均旋转次数其实很小（约 0.21 次）

这说明 AVL 在理论上很漂亮，但工程中是否划算，要看具体场景。

---

## 18. 本章主线总结

### 18.1 BST 通过顺序性把“有序序列”嵌入树中

它的本质不是简单的二叉树，而是：

- 左子树较小
- 右子树较大
- 中序遍历单调非降

因此查找像“树上的二分”。

### 18.2 BST 的所有核心操作都受树高支配

- `search`
- `insert`
- `remove`

最坏都为 `O(h)`，而树高失控时会退化为线性。

### 18.3 插入和删除都依赖 `search()` 返回“位置引用”

这是 BST 接口设计最精妙的地方之一。  
查找失败时返回空位引用，使后续插入、删除逻辑大为简化。

### 18.4 双分支删除的关键是“后继化归”

当左右子树都在时：

- 找中序后继
- 交换数据
- 转化成单分支删除

这把复杂情况规约成简单情况。

### 18.5 单靠普通 BST 不够，必须平衡化

现实数据往往并不随机，退化树并不罕见，因此平衡 BST 很有必要。

### 18.6 AVL 用高度差约束换来 `O(log n)` 最坏界

- 插入：只需处理最低失衡点，一次重平衡即可
- 删除：可能一路向上传播，最坏需 `O(log n)` 次修复

### 18.7 `(3+4)` 重构统一了所有旋转

它是 BST / AVL 实现中最值得记住的工程技巧之一。

---

## 19. 一句话总结

> 二叉搜索树这一章真正重要的，不只是学会查找、插入、删除三段代码，而是理解：BST 的效率本质上取决于树高，而所有平衡化技术——从旋转到 AVL——都是在不破坏中序顺序的前提下，尽可能把树重新压低。
