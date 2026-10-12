---
title: "DSA 8: Advanced Search Tree"
date: 2026-03-27
math: true
tags:
  - Notes
  - DSA
categories:
  - DSA
description: "DSA CHAP8"
---

# 数据结构与算法笔记（八）：高级搜索树

> 本文基于课程第八章“高级搜索树”整理。内容围绕三条主线展开：  
> 第一条是**伸展树（Splay Tree）**，强调局部性、自调整与分摊分析；  
> 第二条是**B-树**，强调外存 / 分级存储 / I-O 优化背景下的多路平衡搜索树；  
> 第三条是**红黑树（Red-Black Tree）**，强调与 `(2,4)`-树的等价关系、插入时的双红修正、删除时的双黑修正，以及其在并发与持久化场景下的优势。  
> 这份笔记不是逐页抄录 PPT，而是在课件基础上尽量把每一部分背后的设计动机、算法逻辑与复杂度分析串成一条更清晰的博客叙述线。主要依据你上传的课件内容：fileciteturn10file0

---

## 1. 这一章在前面几章中的位置

如果把前面的搜索树内容串起来，大致是这样一条线：

1. **普通 BST**  
   有顺序性，但高度不可控，最坏可退化成链表。
2. **AVL**  
   通过严格高度平衡，把最坏复杂度压到 `O(log n)`，但更新时需要维护高度、旋转可能沿祖先链传播。
3. **更高级的搜索树**  
   当我们继续往前走，会出现三个更现实的问题：

- 若访问具有**局部性**，能否利用“最近访问的节点很可能马上再被访问”来加速？
- 若数据规模远超内存，I/O 成本远高于 RAM 运算，能否把树设计得更“胖”，减少磁盘访问层数？
- 若我们关心**并发性 / 持久化 / 局部重构规模**，能否找到一种更新时只修改 `O(1)` 个局部结构的平衡树？

于是本章自然分成三块：

- **Splay Tree**：自调整，利用访问局部性
- **B-Tree**：多路平衡搜索树，优化外存 I/O
- **Red-Black Tree**：局部重构受控、与 `(2,4)`-树等价的 BBST

---

# 第一部分：伸展树（Splay Tree）

## 2. 局部性：为什么“刚访问过的节点”值得被提升

### 2.1 课件想强调的 Locality

课件第 2～3 页从系统和软件访问模式出发，强调了一个经验事实：

> 刚被访问过的数据，极有可能很快地再次被访问；  
> 下一次将被访问的数据，也极有可能就在刚访问过的数据附近。 fileciteturn10file0

这就是所谓的 **locality（局部性）**。

在 BST 里，这意味着：

- 某个节点一旦刚刚被查到
- 它以及它附近的一小片区域，接下来很可能还会被频繁访问

如果还是老老实实把它放在原来的深层位置，那么接下来每次访问都要再次走很长一条路径，这显然浪费。

### 2.2 自适应链表的类比

课件第 3 页提到一个非常直观的类比：fileciteturn10file0

- 在**自适应链表**里，一个节点一旦被访问，就立即被挪到最前面
- 那么在 BST 中，自然会想到：

> 一个节点一旦被访问，能否也立刻被“推到根”？

这就是伸展树的最初动机。

---

## 3. 逐层伸展：直观但不够好

### 3.1 最直接的想法：一层一层往上转

课件第 4 页先讲了最朴素的方案：fileciteturn10file0

- 若访问到节点 `v`
- 就不断对其父亲做单旋：
  - 若 `v` 是左孩子，就 `zig(v->parent)`
  - 若 `v` 是右孩子，就 `zag(v->parent)`

于是 `v` 会一点一点往上“爬”，直到爬到根。

课件用“与其说推，不如说爬”来形容这个过程，非常贴切。fileciteturn10file0

### 3.2 为什么逐层单旋会出问题

课件第 5～6 页给出一个最坏例子：fileciteturn10file0

若树本来是一条长链，而访问序列又刚好沿着某种次序反复出现，那么逐层单旋的总代价会形成周期性的算术级数。  
结果是：

- 某一个周期内总代价可达 `Ω(n^2)`
- 分摊下来仍有 `Ω(n)`

这显然不够好。

也就是说：

> “访问后把节点挪到根”这个思想没有错，  
> 但**怎么挪**非常关键。

---

## 4. 双层伸展：Sleator-Tarjan 的核心改进

### 4.1 关键变化：不是追一层，而是追两层

课件第 8 页引用了 Sleator 与 Tarjan 1985 年的经典论文《Self-Adjusting Binary Trees》，并指出其构思精髓：fileciteturn10file0

> 反复考察祖孙三代：  
> `g = parent(p), p = parent(v), v`  
> 然后一次做两次旋转，使 `v` 直接上升两层。

这就是**双层伸展（double splaying）**。

### 4.2 四种相对位置

根据 `v, p, g` 的相对位置，有四种基本情形：

1. `zig-zig`
2. `zig-zag`
3. `zag-zig`
4. `zag-zag`

课件第 9～10 页分别配图展示了这些情形。fileciteturn10file0

这四种情形其实与 AVL 的四种失衡形态在拓扑上非常像，但目的不同：

- AVL 的旋转是为恢复平衡
- Splay 的双旋是为把访问节点快速提到更高处

### 4.3 `zig-zag / zag-zig` 为什么“看起来和逐层调整一样”

课件第 9 页指出：

- 当 `v` 在中间位置（按中序居中）
- 若想让它最终成为子树根
- 最终拓扑似乎与逐层单旋结果一致 fileciteturn10file0

这会让人误以为双层伸展“只是换个写法”。

但真正本质区别在 `zig-zig / zag-zag`。

### 4.4 `zig-zig / zag-zag` 的折半效果

课件第 11 页强调了这一页最重要的话：

> 节点访问之后，对应路径的长度随即折半。  
> 最坏情况不致持续发生。 fileciteturn10file0

这正是双层伸展优于逐层单旋的关键。

尤其在一条长链中：

- 若采用逐层单旋，链条只是局部扭一扭，整体仍可能很糟
- 若采用 `zig-zig` 或 `zag-zag`，路径会像“含羞草折叠”一样大幅压短

因此：

> 伸展树的优势，本质上不是“每次把目标移到根”，  
> 而是“在这个过程中顺带显著压缩访问路径”。

### 4.5 最后一层：只有父亲没有祖父怎么办

课件第 12 页指出，若 `v` 只有父亲、没有祖父，则说明：

- `v.parent() == T.root()`

此时只需再做一次单旋：

- `zig(r)` 或
- `zag(r)` fileciteturn10file0

并且这种情况在一次完整伸展过程中**至多出现一次**，也就是最后收尾时。

---

## 5. 伸展树的接口与总体框架

### 5.1 Splay 类接口

课件第 14 页给出接口设计：fileciteturn10file0

```cpp
template <typename T> class Splay : public BST<T> {
protected:
    BinNodePosi<T> splay(BinNodePosi<T> v); // 将v伸展至根
public:
    BinNodePosi<T>& search(const T& e);     // 重写
    BinNodePosi<T> insert(const T& e);      // 重写
    bool remove(const T& e);                // 重写
};
```

这里 `search()` 之所以必须重写，是因为：

> 在伸展树里，查找不再是“纯静态操作”，  
> 因为只要访问了某个节点，就会引起整棵树拓扑调整。 fileciteturn10file0

### 5.2 `splay(v)` 的总体框架

课件第 15 页给出总体结构：fileciteturn10file0

```cpp
template <typename T> BinNodePosi<T> Splay<T>::splay(BinNodePosi<T> v) {
    if (!v) return NULL;
    BinNodePosi<T> p;
    BinNodePosi<T> g;
    while ((p = v->parent) && (g = p->parent)) {
        /* 自下而上，反复双层伸展 */
    }
    if (p = v->parent) {
        /* 若p果真是根，只需再额外单旋一次 */
    }
    v->parent = NULL;
    return v;
}
```

可以看到，这与前面讲的思想完全一致：

1. 只要还有祖父，就做双层伸展
2. 最后若只剩父亲，则再补一次单旋
3. 最终把 `v` 变成根

### 5.3 双层伸展中的“曾祖父 gg”

课件第 16 页进一步写出：fileciteturn10file0

```cpp
BinNodePosi<T> gg = g->parent;
```

因为双旋结束后：

- `v` 将成为这一局部子树的新根
- 原来整棵局部子树是挂在 `gg` 下面的
- 所以必须把新根 `v` 再接回 `gg` 的左或右孩子位置

这和 AVL 中 `(3+4)` 重构后“把新子树根接回曾祖父”是完全同一类局部修补操作。

---

## 6. 伸展树的查找、插入与删除

### 6.1 查找：成功和失败都要伸展

课件第 18 页给出 `search()`：fileciteturn10file0

```cpp
template <typename T> BinNodePosi<T>& Splay<T>::search(const T& e) {
    BinNodePosi<T> p = BST<T>::search(e);
    _root = splay(p ? p : _hot);
    return _root;
}
```

关键点在于：

- 若查找成功：把命中节点 `p` 伸展到根
- 若查找失败：把最后访问到的节点 `_hot` 伸展到根

于是伸展树查找的语义变成：

> **不管成功还是失败，最后访问的那个“最相关”节点都会被拉到根。**

这非常契合局部性：  
即使没找到目标，说明查询点最接近的是 `_hot`，它以后也很可能再次相关。

### 6.2 插入：`search + splay + split + join`

课件第 19 页先给出直观思路：fileciteturn10file0

1. 先调用 `search(e)`
2. 查找（失败）后，`_hot` 已经被伸展到根
3. 此时整棵树相当于被根分成左右两部分：
   - 左侧都 `< e`
   - 右侧都 `> e`
4. 于是直接在根附近接入新节点最自然

课件右侧配图正是在说明这个过程：`search -> splay -> split -> join`。fileciteturn10file0

### 6.3 插入实现

课件第 20 页：fileciteturn10file0

```cpp
template <typename T> BinNodePosi<T> Splay<T>::insert(const T& e) {
    if (!_root) { _size = 1; return _root = new BinNode<T>(e); }
    BinNodePosi<T> t = search(e);
    if (e == t->data) return t;
    if (t->data < e) {
        t->parent = _root = new BinNode<T>(e, NULL, t, t->rc);
        if (t->rc) { t->rc->parent = _root; t->rc = NULL; }
    } else {
        t->parent = _root = new BinNode<T>(e, NULL, t->lc, t);
        if (t->lc) { t->lc->parent = _root; t->lc = NULL; }
    }
    _size++;
    updateHeightAbove(t);
    return _root;
}
```

这段代码的本质是：

- 若 `e > t->data`，则新根左子树是 `t`，右子树是 `t` 原来的右子树
- 若 `e < t->data`，则新根右子树是 `t`，左子树是 `t` 原来的左子树

它非常巧妙地利用了：

- `t` 已被伸展到根
- 所以根两侧天然就是关于 `e` 的合法分割

### 6.4 删除：`search + splay + release + join`

课件第 21 页给出删除的直观流程：fileciteturn10file0

1. `search(e)` 成功后，目标节点已被伸展到根
2. 此时它的左子树 `L` 和右子树 `R` 分别独立
3. 把根释放掉
4. 再把 `L` 与 `R` 合并（join）

### 6.5 删除实现

课件第 22 页给出代码：fileciteturn10file0

```cpp
template <typename T> bool Splay<T>::remove(const T& e) {
    if (!_root || (e != search(e)->data)) return false;
    BinNodePosi<T> L = _root->lc, R = _root->rc;
    release(t);
    if (!R) {
        if (L) L->parent = NULL;
        _root = L;
    } else {
        _root = R; R->parent = NULL;
        search(e); // 在R中再找e：注定失败，但最小节点必伸展至根
        if (L) L->parent = _root;
        _root->lc = L;
    }
    _size--;
    if (_root) updateHeight(_root);
    return true;
}
```

这里最妙的一步是：

```cpp
_root = R;
search(e);
```

因为 `e` 不在 `R` 中，查找必失败；  
但失败后，`R` 中最小的节点会被伸展到根。  
此时根没有左孩子，于是可把整棵 `L` 直接接成它的左子树。

这就完成了 `L` 与 `R` 的 join。

---

## 7. 伸展树的复杂度与综合评价

### 7.1 单次最坏情况并不受控

课件第 23 页指出一个非常重要的事实：

> 伸展树不能杜绝单次最坏情况。  
> 因此不适用于对“单次响应时间”特别敏感的场合。 fileciteturn10file0

也就是说：

- 单次 `search / insert / remove`
- 仍可能退化到 `O(n)`

这与 AVL 的“每次都保证 `O(log n)`”不同。

### 7.2 分摊复杂度是 `O(log n)`

但课件也给出核心优点：

> 伸展树无需记录高度或平衡因子，编程实现简单；  
> 且分摊复杂度与 AVL 相当。 fileciteturn10file0

也就是说，对任意长操作序列：

- 平均每次操作只需 `O(log n)` 分摊时间

这使它在很多非实时系统中非常有吸引力。

### 7.3 伸展树的“自适应性”

课件第 23 页进一步指出：fileciteturn10file0

- 若局部性很强
- 则效率甚至还可以比 AVL 更高
- 反复顺序访问某一子集时，分摊成本可降为常数

这正体现了伸展树的最大特色：

> **它不是严格平衡树，而是“自适应搜索树”。**

树形会随着访问模式不断变化，自动把“热点”拉近根部。

---

## 8. 伸展树的分摊分析：势能法直觉

课件第 24～29 页开始进入伸展树的势能分析。fileciteturn10file0

虽然课件这里主要是证明框架和直觉，并没有把全部细节完全展开，但核心思想很值得记录。

### 8.1 每棵伸展树都带一个势能

课件第 25 页说：

> 任何时刻的任何一棵伸展树，都可以被假想地认为具有势能。  
> 越平衡的树势能越小，越倾斜的树势能越大。 fileciteturn10file0

这和摊还分析里最典型的“势能法”完全一致：

- 当前操作若把树变得更有序 / 更平衡
- 就是在释放一部分势能
- 这部分释放出来的势能可以“支付”本次旋转成本

### 8.2 伸展开销受节点势能变化控制

课件第 26 页指出：fileciteturn10file0

只要能证明每次伸展操作的实际开销，不会超过某种“势能变化量 + 常数”，  
那么对一长串操作求和后，很多中间项会望远镜式抵消，从而得到总分摊上界。

这就是为什么：

- 单次可能很慢
- 但连续操作总量仍受控

---

# 第二部分：B-树

## 9. 为什么要从 AVL / Splay 走向 B-Tree

### 9.1 RAM 模型在外存时代不够用了

课件第 31～33 页指出：fileciteturn10file0

传统 RAM / Turing 模型默认：

- 存储是统一、随机可访问的
- 每次访问成本差不多

但现实系统中：

- 内存容量增长远慢于数据规模
- 磁盘 / 外存访问比 RAM 慢几个数量级
- 真正瓶颈不再是“比较次数”，而是 **I/O 次数**

课件第 32 页给出非常形象的说法：

> 若一次内存访问需要 1 秒，则一次磁盘访问要一天。 fileciteturn10file0

所以在外存模型下，我们应尽量：

- 一次 I/O 读更多内容
- 用更少层数到达目标

### 9.2 批量访问的启示

课件第 34 页提到 page / block / buffer 的概念：fileciteturn10file0

- 磁盘访问通常按块进行
- 读 1B 和读 1KB 的代价几乎差不多
- 因此若每次只读一个 key，就极其浪费

这正是 B-Tree 的根本动机：

> **既然一次 I/O 读整块几乎不更贵，那就把一整组关键码塞进一个节点。**

---

## 10. B-Tree 的结构：多路平衡搜索树

### 10.1 从 BBST 到“超级节点”

课件第 42 页指出：fileciteturn10file0

- 若把一棵平衡二叉搜索树每 `d` 代合并成一个“超级节点”
- 就可以得到一棵 `m = 2^d` 路的多路搜索树

这就是 B-Tree 的直观来源。

因此从逻辑上说：

> B-树与 BBST 是等价的搜索结构；  
> 但在物理实现上，它更适合外存块访问。

### 10.2 m 阶 B-树的定义

课件第 44～45 页给出 m 阶 B-树的关键条件：fileciteturn10file0

1. 它是一棵 **m 路完全平衡搜索树**
2. 外部节点深度统一相等，记为树高 `h`
3. 内部节点含有至多 `m-1` 个关键码
4. 内部节点有：
   - 根节点：至少 2 个分支（若非空树）
   - 其他节点：至少 `⌈m/2⌉` 个分支
5. 所有叶节点深度相同

因此 B-Tree 也常被写成 `(a,b)`-tree，例如：

- `(2,3)`-tree = 2-3 树
- `(2,4)`-tree = 2-3-4 树

### 10.3 节点内部的紧凑表示

课件第 48 页给出 B-树节点结构：fileciteturn10file0

```cpp
template <typename T> struct BTNode {
    BTNodePosi<T> parent;
    Vector<T> key;
    Vector<BTNodePosi<T>> child;
    BTNode() { parent = NULL; child.insert(NULL); }
    BTNode(T e, BTNodePosi<T> lc = NULL, BTNodePosi<T> rc = NULL) {
        parent = NULL;
        key.insert(e);
        child.insert(lc); if (lc) lc->parent = this;
        child.insert(rc); if (rc) rc->parent = this;
    }
};
```

一个节点里：

- `key.size() = d`
- `child.size() = d + 1`

这正对应多路搜索树的经典结构：

```text
child[0] key[0] child[1] key[1] ... key[d-1] child[d]
```

---

## 11. B-Tree 查找

### 11.1 查找逻辑

课件第 51～53 页给出查找思路：fileciteturn10file0

1. 根节点常驻 RAM
2. 在当前节点内部顺序查找目标关键码
3. 若命中则成功返回
4. 若未命中，则沿相应孩子指针深入下一层
5. 若走到外部节点，则失败

课件代码：

```cpp
template <typename T> BTNodePosi<T> BTree<T>::search(const T& e) {
    BTNodePosi<T> v = _root; _hot = NULL;
    while (v) {
        Rank r = v->key.search(e);
        if (0 <= r && e == v->key[r]) return v;
        _hot = v;
        v = v->child[r + 1];
    }
    return NULL;
}
```

### 11.2 为什么外存下查找这么快

课件第 54～56 页强调：

- 真正代价主要取决于 I/O 次数
- 每深入一层，至多发生一次 I/O
- 所以总查询时间约等于树高 `h` fileciteturn10file0

而 B-树高度非常低。  
例如课件举例：

- 若 `m = 256`
- 相比 BBST，高度大约能降到原来的 `1/7 ~ 1/8`

因此查找时的磁盘 I/O 次数会大幅减少。fileciteturn10file0

---

## 12. B-Tree 插入：上溢与分裂

### 12.1 插入主流程

课件第 58 页给出主算法：fileciteturn10file0

```cpp
template <typename T> bool BTree<T>::insert(const T& e) {
    BTNodePosi<T> v = search(e);
    if (v) return false;
    Rank r = _hot->key.search(e);
    _hot->key.insert(r + 1, e);
    _hot->child.insert(r + 2, NULL);
    _size++;
    solveOverflow(_hot);
    return true;
}
```

逻辑很清晰：

1. 先查找确认不存在
2. 插入到 `_hot` 这个失败位置所在的叶节点中
3. 若节点关键码数超出上限，则修复上溢

### 12.2 上溢（overflow）是什么意思

对 m 阶 B-树来说，一个节点最多只能有：

`m - 1`

个关键码。  
若插入后达到 `m` 个关键码，就发生了**上溢**。

课件第 59 页指出：

- 取中位数为轴点
- 将其上升一层
- 原节点分裂成左右两个节点 fileciteturn10file0

这就是标准 split 操作。

### 12.3 `solveOverflow()` 的实现思想

课件第 64～65 页给出代码：fileciteturn10file0

```cpp
if (_m > v->key.size()) return;
Rank s = _m / 2;
BTNodePosi<T> u = new BTNode<T>();

for (Rank j = 0; j < _m - s - 1; j++) {
    u->child.insert(j, v->child.remove(s + 1));
    u->key.insert(j, v->key.remove(s + 1));
}
u->child[_m - s - 1] = v->child.remove(s + 1);
```

本质是：

- 把 `v` 右半边的 key 和 child 移到新节点 `u`
- `v` 中位数 key 上升到父节点
- 父节点多接入一个孩子 `u`

### 12.4 为什么上溢会向上传播

若父节点本来就接近满载，那么接纳这个“被提升的中位数”后，父节点也可能上溢。  
课件第 60 页指出：

> 上溢可能持续发生，并逐层向上传播；  
> 最坏情况下会一直传到根。 fileciteturn10file0

若根也上溢，则：

- 以中位数为新根
- 左右两半变成它的两个孩子

这也是：

> **B-树长高的唯一方式。**

### 12.5 插入复杂度

每一层至多分裂一次，因此分裂次数不超过树高：

`O(h)`

而 B-树高度本身是：

`O(log_m N)`

所以插入复杂度是：

`O(log_m N)`

---

## 13. B-Tree 删除：下溢、旋转与合并

### 13.1 删除前先把目标“挪到底层”

课件第 67～68 页指出：fileciteturn10file0

若目标关键码在内部节点中，则不能直接删，因为会破坏多路搜索树结构。  
标准做法是：

- 在其右子树中一直向左
- 找到后继（必在底层叶节点）
- 交换关键码
- 然后把问题化归为在叶节点中删除

这与普通 BST 删除“后继化归”的思想完全一致。

### 13.2 下溢（underflow）是什么意思

删除关键码后，某个非根节点若孩子数降到小于：

`⌈m/2⌉`

就发生了**下溢**。  
课件第 69～70 页给出三种修复策略：fileciteturn10file0

1. 向左兄弟借
2. 向右兄弟借
3. 若兄弟都不够胖，则与某个兄弟合并

### 13.3 旋转：向兄弟借关键码

若某个兄弟（左或右）足够“胖”，也就是多于下界，则可以：

- 把父节点的一个分界关键码下移给下溢节点
- 再把兄弟中的一个关键码上移给父节点
- 同时调整一棵对应子树的归属

这就是课件第 69 页所说的“旋转”。fileciteturn10file0

### 13.4 合并：兄弟都不够胖时

若左右兄弟都不够胖，则说明它们都恰好在下界附近。  
课件第 70 页指出：

- 从父节点拿下一个分界关键码
- 把下溢节点与某个兄弟粘接成一个更大的节点
- 同时把相应孩子引用一起合并 fileciteturn10file0

但这样会使父节点失去一个关键码和一个孩子，  
因此父节点自己也可能继续下溢。

所以：

> **下溢修复可能向上传播。**

### 13.5 `solveUnderflow()` 的实现轮廓

课件第 73～76 页给出代码框架：fileciteturn10file0

```cpp
if ((_m + 1) / 2 <= v->child.size()) return;   // 未下溢

BTNodePosi<T> p = v->parent;
if (!p) { /* 已到根节点 */ }

Rank r = 0; while (p->child[r] != v) r++;

if (0 < r) { /* 尝试向左兄弟借 */ }
if (p->child.size() - 1 > r) { /* 尝试向右兄弟借 */ }

if (0 < r) { /* 与左兄弟合并 */ }
else { /* 与右兄弟合并 */ }

solveUnderflow(p);
```

这正体现了标准的三步逻辑：

1. 先看能不能借
2. 借不到就合并
3. 合并后递归修父节点

---

## 14. B-Tree 的扩展话题

课件第 77 页最后提到两个思考方向：fileciteturn10file0

### 14.1 上溢是否也能“借位”而不一定分裂

从原理上说，上溢修复与下溢修复是对偶的：

- 下溢：可借位、可合并
- 上溢：理论上也可先尝试与兄弟重新分配，再决定是否分裂

但工程中通常还是直接用分裂，因为实现简单、局部性清晰。

### 14.2 B\*-Tree

课件提到 B\*-Tree 的思想：

- 上溢时不一定立即单节点分裂
- 可以联合兄弟共同均摊关键码
- 提高空间利用率（比 50% 更高）

这也是数据库索引中非常重要的家族分支。

---

# 第三部分：红黑树（Red-Black Tree）

## 15. 为什么在 AVL 之后还要学红黑树

课件第 79～82 页先讲了两个新动机：fileciteturn10file0

### 15.1 并发性（Concurrency）

在并发环境下，修改树结构时往往需要加锁。  
真正影响延迟的是：

- 一次更新需要改多少局部结构
- 也就是“锁住多少局部区域”

课件指出：

- Splay：结构变化很剧烈，最坏可很大
- AVL：`remove()` 时可能一路向上传播
- Red-Black：无论插入还是删除，局部重构数量都受更严格控制 fileciteturn10file0

因此红黑树在并发友好性上更有优势。

### 15.2 持久性（Persistence）

若希望保存历史版本（partial persistence），则理想情况是：

- 每个新版本只复制极少数新节点
- 其余大部分结构与旧版本共享

课件指出：

> 为此，相邻版本在树形拓扑上的差异最好只限于 `O(1)` 局部重构。  
> AVL、Splay 这类树并不具备这一性质，需另辟蹊径。 fileciteturn10file0

这再次说明红黑树的独特价值：

> 它不只是“另一种平衡树”，  
> 更是工程上更适合并发 / 持久化 / 标准库实现的平衡树。

---

## 16. 红黑树的定义与 `(2,4)`-树等价

### 16.1 红黑树的规则

课件第 85～86 页给出四条经典规则：fileciteturn10file0

1. 根节点为黑
2. 外部 `NULL` 节点均为黑
3. 红节点不能有红孩子（也即不会出现相邻红节点）
4. 从任意节点到其所有外部节点的路径上，黑节点数相同

其中第 4 条也可以用“黑高度（black height）一致”来表达。

### 16.2 红黑树 = `(2,4)`-树

课件第 87 页是理解红黑树最关键的一页：fileciteturn10file0

- 若把红节点“提升”到与其黑父亲等高
- 并把它们视为一个超级节点
- 那么红黑树恰好对应一棵 `(2,4)`-树

可能的组合有四种：

- 黑黑
- 黑红
- 红黑
- 红红

分别对应 `(2,4)`-树内部节点中 1、2、3 个关键码的不同情况。

这说明：

> 红黑树其实就是 `(2,4)`-树的二叉编码形式。

### 16.3 为什么红黑树高度是 `O(log n)`

课件第 88 页给出直觉证明：fileciteturn10file0

设：

- `H` 为红黑树的黑高度
- `h` 为红黑树普通高度

由于红节点不能相连，所以有：

`h <= 2H`

而黑节点折叠后得到一棵 `(2,4)`-树，其高度就是 `H`，必为 `O(log n)`。  
于是：

`h = O(log n)`

因此红黑树属于 BBST。

---

## 17. 红黑树插入：双红修正

### 17.1 基本流程

课件第 91～92 页指出插入流程非常简单：fileciteturn10file0

1. 按普通 BST 插入新节点 `x`，它必为叶子
2. 将 `x` 染成红色
3. 若未违反规则则结束
4. 若出现 `x` 与父亲 `p` 同为红色，则形成 **double-red**
5. 调用 `solveDoubleRed(x)`

之所以新插入节点先染红，是因为：

- 若染黑，会增加一条根到叶路径的黑高度
- 更容易破坏规则 4
- 染红则只可能破坏规则 3（相邻红）

这让修复更局部。

### 17.2 情况一：叔父 `u` 为黑（或 NULL）

课件第 94～96 页称之为 `RR-1`。fileciteturn10file0

此时：

- `x, p, g` 之间的关系可归为四种 LL/LR/RL/RR
- 执行一次 `(3+4)` 重构（即 `rotateAt(x)`）
- 同时配合重新染色：
  - 新根 `b` 染黑
  - 另两侧节点适当染红

课件第 95 页解释其本质：

> 在等价 `(2,4)`-树里，这对应于一个超级节点内部关键码重新居中，  
> 即把非法的 `RRB / BRR` 调整成合法的 `RBR`。 fileciteturn10file0

### 17.3 情况二：叔父 `u` 为红

课件第 97～100 页称之为 `RR-2`。fileciteturn10file0

此时：

- `p` 和 `u` 都染黑
- `g` 染红
- 等价于 `(2,4)`-树中一个超级节点发生了**分裂**
- 问题向上提升到 `g`

若 `g` 的父亲又是红色，就继续递归处理。  
若最后递归到根，则强行把根染黑，整棵树黑高度加一。

因此这一类情况的特征是：

> **不旋转，只重染色，但可能向上传播。**

### 17.4 插入复杂度

课件第 101 页总结：fileciteturn10file0

- 若 `u` 为黑：最多 1~2 次旋转，2 次重染色，随后结束
- 若 `u` 为红：0 次旋转，3 次重染色，但问题会上升两层

因此插入总复杂度是：

`O(log n)`

而且真正旋转次数很少，局部性很好。

---

## 18. 红黑树删除：双黑修正

红黑树删除比插入复杂得多。  
课件第 102～117 页都在讲这一部分。fileciteturn10file0

### 18.1 等效删除与“双黑”

课件第 103～105 页先做了一个非常重要的抽象：fileciteturn10file0

删除时，按普通 BST 的 `removeAt(x, _hot)` 进行，真正被摘除的节点可能是：

- 原目标节点
- 或其前驱 / 后继

把它统称为 `x`。

删除后，由某个孩子 `r` 替代 `x`。  
这时有三种可能：

1. `x` 红
2. `r` 红
3. `x` 和 `r` 都黑

前两种都比较简单：

- 若 `x` 红，删掉后不影响黑高度
- 若 `r` 红，让 `r` 转黑即可补齐黑高度

真正麻烦的是第 3 种：

> `x` 与替代者 `r` 都是黑色

这时相当于某条路径“少了一个黑节点”，于是引入了**double black（双黑）**概念。

### 18.2 `remove()` 框架

课件第 106～107 页给出主框架：fileciteturn10file0

```cpp
template <typename T> bool RedBlack<T>::remove(const T& e) {
    BinNodePosi<T>& x = search(e);
    if (!x) return false;
    BinNodePosi<T> r = removeAt(x, _hot);
    if (!(--_size)) return true;

    if (!_hot) {
        _root->color = RB_BLACK;
        updateHeight(_root);
        return true;
    }

    if (BlackHeightUpdated(*_hot)) return true;

    if (IsRed(r)) {
        r->color = RB_BLACK;
        r->height++;
        return true;
    }

    solveDoubleBlack(r);
    return true;
}
```

这正体现了前面的三类分流：

- 若树删空，直接结束
- 若父亲黑高度仍平衡，结束
- 若替代者 `r` 是红，只需染黑
- 否则进入 `solveDoubleBlack(r)`

### 18.3 双黑修正的三大类

课件第 108 页给出总框架：fileciteturn10file0

令：

- `p`：`r` 的父亲
- `s`：`r` 的兄弟
- `t`：`s` 的某个红孩子（若有）

则分成三类：

1. **BB-1**：兄弟 `s` 黑，且 `s` 至少有一个红孩子
2. **BB-2**：兄弟 `s` 黑，且 `s` 没有红孩子
   - 又分 `BB-2R`（父亲红）
   - 和 `BB-2B`（父亲黑）
3. **BB-3**：兄弟 `s` 红

### 18.4 BB-1：黑兄弟有红孩子

课件第 109～111 页给出：fileciteturn10file0

这时执行：

- 一次 `(3+4)` 重构（`rotateAt(t)`）
- 新根继承原父亲 `p` 的颜色
- 左右孩子转黑

代码核心：

```cpp
RBColor oldColor = p->color;
BinNodePosi<T> b = FromParentTo(*p) = rotateAt(t);
if (HasLChild(*b)) { b->lc->color = RB_BLACK; updateHeight(b->lc); }
if (HasRChild(*b)) { b->rc->color = RB_BLACK; updateHeight(b->rc); }
b->color = oldColor; updateHeight(b);
```

这一操作对应于 `(2,4)`-树中通过关键码旋转来消除下溢。  
修复后全局立即恢复平衡，删除完成。

### 18.5 BB-2R：黑兄弟无红孩子，父亲红

课件第 112 页：fileciteturn10file0

若：

- `s` 黑
- `s` 两个孩子都黑
- `p` 红

则做：

- `s` 染红
- `p` 染黑

这相当于在 `(2,4)`-树中：

- 用父节点关键码向下与兄弟、下溢节点完成一次合并

而且因为 `p` 原本是红色，所以不会继续向上传播。  
因此此情形处理后直接结束。

### 18.6 BB-2B：黑兄弟无红孩子，父亲也黑

课件第 113～114 页：fileciteturn10file0

若：

- `s` 黑
- `s` 无红孩子
- `p` 也黑

则只能：

- `s` 染红
- `p` 保持黑，但其黑高度下降 1

于是问题上升到 `p`，继续递归：

```cpp
s->color = RB_RED;
s->height--;
p->height--;
solveDoubleBlack(p);
```

这就是删除中最麻烦的一类，因为它会沿祖先链继续传播。

### 18.7 BB-3：兄弟红

课件第 115～116 页给出：fileciteturn10file0

若兄弟 `s` 为红，则其孩子必为黑。  
此时先做一次单旋：

- `s` 转黑
- `p` 转红
- 围绕 `p` 旋转

这样问题会转化成：

- BB-1
- 或 BB-2R

也就是说：

> **BB-3 只是一个过渡型情况，用来把“红兄弟”变成“黑兄弟”情况。**

代码里也明确写成：

```cpp
s->color = RB_BLACK;
p->color = RB_RED;
BinNodePosi<T> t = IsLChild(*s) ? s->lc : s->rc;
_hot = p;
FromParentTo(*p) = rotateAt(t);
solveDoubleBlack(r);
```

### 18.8 删除复杂度

课件第 117 页总结：fileciteturn10file0

- `BB-1`：1~2 次旋转，3 次染色，结束
- `BB-2R`：0 次旋转，2 次染色，结束
- `BB-2B`：0 次旋转，1 次染色，问题上升一层
- `BB-3`：1 次旋转，2 次染色，再转成 `BB-1` 或 `BB-2R`

因此总删除复杂度仍为：

`O(log n)`

而且每一步局部修改规模仍受到严格控制。

---

## 19. 红黑树的综合评价

### 19.1 与 AVL 的差别

可以把红黑树和 AVL 粗略对比成：

#### AVL

- 平衡更严格
- 查找性能略优
- 插入 / 删除后维护成本更高
- 删除可能一路旋转 / 更新

#### Red-Black

- 平衡稍弱，但仍是 `O(log n)`
- 更新规则更适合局部修补
- 更适合工程实现、标准库、并发与持久化场景

这也是为什么很多语言标准库（如 `java.util.TreeMap`）选红黑树而不是 AVL。  
课件第 83 页专门用 `TreeMap` 举了例子。fileciteturn10file0

---

## 20. 本章主线总结

### 20.1 高级搜索树不是“更多旋转技巧”，而是三类现实约束下的不同答案

- **Splay Tree**：面向局部性，自适应调整
- **B-Tree**：面向外存和 I/O，多路胖节点
- **Red-Black Tree**：面向并发 / 持久化 / 工程实现，局部重构受控

### 20.2 伸展树的核心不是平衡，而是访问驱动的自调整

- 访问后立即把节点推到根附近
- 双层伸展带来路径折半效果
- 单次最坏不受控，但分摊 `O(log n)`
- 在强局部性下甚至更优

### 20.3 B-树的核心不是“多叉”，而是“块访问”

- 一次 I/O 读入整页关键码
- 让树高极低
- 以更少 I/O 换取查找 / 插入 / 删除效率

### 20.4 红黑树的核心是与 `(2,4)`-树的等价

- 双红对应上溢
- 双黑对应下溢
- 插入和删除都能用“旋转 + 染色”局部修复
- 最坏复杂度稳定在 `O(log n)`

### 20.5 这一章真正值得记住的思想

不是具体的每一行代码，而是下面这些“设计动机”：

- 数据访问有局部性 → 伸展树
- 磁盘 I/O 太贵 → B-树
- 并发和持久化要求局部重构小 → 红黑树

---

## 21. 一句话总结

> 这一章真正重要的，不只是学会三种高级搜索树的插入删除代码，而是理解：不同搜索树的平衡策略，其实是在为不同的现实约束做取舍——伸展树为局部性服务，B-树为 I-O 服务，红黑树为局部重构与工程可用性服务。
