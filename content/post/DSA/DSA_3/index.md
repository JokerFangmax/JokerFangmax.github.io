---
title: "DSA 3: List"
date: 2026-03-26
math: true
tags:
  - Notes
  - DSA
categories:
  - DSA
description: "DSA CHAP3"
---

# 数据结构与算法笔记（三）：列表

> 本文基于课程第三章“列表”整理。内容不是逐页复述 PPT，而是在保留课程结构的基础上，系统补充“为什么要从向量过渡到列表”“双向链表的接口与实现”“无序 / 有序列表的查找、去重、排序”“逆序对与输入敏感性”“游标实现”等内容，整理成适合博客发布的 Markdown 笔记。

---

## 1. 从静态存储到动态存储

### 1.1 为什么会从向量转向列表

上一章的向量本质上是**顺序存储结构**：

- 一段地址连续的空间
- 元素逻辑顺序与物理顺序严格一致
- 支持高效的循秩访问（call-by-rank）

因此，向量特别擅长：

- `get`
- `search`
- 通过下标随机访问

但它的弱点也同样明显：

- 中间位置插入元素，需要整体后移后缀
- 中间位置删除元素，需要整体前移后缀
- 这些“动态操作”都可能是 `O(n)`

课件一开始就强调了一个整体视角：

1. **静态操作**：仅读取，不改变结构，如 `get`、`search`
2. **动态操作**：会写入、会改变结构，如 `put`、`insert`、`remove`

而不同操作方式，往往对应不同的存储组织方式。

### 1.2 静态结构与动态结构

#### 1.2.1 静态存储策略

- 数据空间整体创建、整体销毁
- 物理顺序与逻辑顺序严格一致
- 支持高效静态操作

向量就是典型代表。

#### 1.2.2 动态存储策略

- 每个元素对应的物理空间可独立申请和回收
- 相邻元素通过“引用 / 指针”维持逻辑次序
- 更适合频繁插入、删除等动态操作

而**列表（list）**就是采用动态存储策略的典型结构。

---

## 2. 列表的基本概念

### 2.1 什么是列表

列表由一系列节点（node）组成，这些节点在逻辑上构成一个线性序列：

L = { a0, a1, a2, ..., a(n-1) }

但与向量不同，列表中的相邻元素不要求在内存中物理相邻，而是通过指针彼此链接。

### 2.2 前驱、后继与首末节点

对于任意一个内部节点：

- 它前面的相邻节点叫**前驱（predecessor）**
- 它后面的相邻节点叫**后继（successor）**

没有前驱的节点叫首节点（first / front），没有后继的节点叫末节点（last / rear）。

### 2.3 从“循秩访问”到“循位置访问”

这是本章最重要的观念转换之一。

#### 2.3.1 向量中的循秩访问

在向量里，第 `r` 个元素的地址可以由下标直接算出，因此循秩访问复杂度是 `O(1)`。

#### 2.3.2 列表中为什么不适合循秩访问

如果也给列表重载 `operator[](r)`，我们只能从首节点或末节点出发，沿着指针一步步走到第 `r` 个节点。

于是其复杂度就变成：

- 最坏：`O(r)`
- 均匀分布时的期望：`O(n)`

这已经失去了“随机访问”的意义。课件也明确写出：列表的 `operator[]` 可以做，但效率低下，“可偶尔为之，却不宜常用”。

### 2.4 列表真正适合的访问方式：循位置访问

因此，对列表应该采用的是：

> **call-by-position（循位置访问）**

也就是说，算法不再试图“通过秩直接跳到节点”，而是：

- 手里先拿着某个节点位置 `p`
- 再通过 `p->pred`、`p->succ`
- 沿着链接一步步走到目标位置

这正是列表式访问的本质：不是靠下标算地址，而是靠节点关系逐步导航。

---

## 3. 列表节点：接口与模板类

### 3.1 为什么节点本身也要被抽象

列表不是“一个数组 + 若干偏移量”，而是由节点构成。  
因此，在设计列表类之前，首先需要把**节点**本身抽象出来。

节点 ADT 接口主要包括：

- `pred()`：前驱位置
- `succ()`：后继位置
- `data()`：节点数据
- `insertAsPred(e)`：在当前节点之前插入
- `insertAsSucc(e)`：在当前节点之后插入

### 3.2 节点模板类

典型定义如下：

```cpp
template <typename T>
using ListNodePosi = ListNode<T>*;

template <typename T>
struct ListNode {
    T data;
    ListNodePosi<T> pred;
    ListNodePosi<T> succ;

    ListNode() {}
    ListNode(T e, ListNodePosi<T> p = NULL, ListNodePosi<T> s = NULL)
        : data(e), pred(p), succ(s) {}

    ListNodePosi<T> insertAsPred(T const& e);
    ListNodePosi<T> insertAsSucc(T const& e);
};
```

其中：

- `data` 存数据
- `pred` 指向前驱
- `succ` 指向后继

这实际上就是**双向链表节点**。

### 3.3 为什么用双向链表而不是单向链表

本章默认使用的是**双向链表**，因为它同时维护：

- `pred`
- `succ`

这样做的好处是：

1. 插入前驱、插入后继都方便
2. 删除当前节点时可以直接连通两边
3. 从后往前查找也容易
4. 更适合实现有序列表的 `search()` 等接口

---

## 4. 列表类：哨兵节点与整体结构

### 4.1 列表模板类的基本框架

列表类骨架如下：

```cpp
template <typename T>
class List {
private:
    int _size;
    ListNodePosi<T> header;
    ListNodePosi<T> trailer; // 哨兵
protected:
    /* ... */
public:
    /* ... */
};
```

其中最关键的设计是：

- `_size`：当前节点数
- `header`：头哨兵
- `trailer`：尾哨兵

### 4.2 什么是哨兵节点

哨兵节点（sentinel）不是正常数据节点，而是为了简化边界处理而额外设置的“辅助节点”。

可以把它们理解成：

- `header`：排在首元素之前的一个假节点
- `trailer`：排在末元素之后的一个假节点

虽然哨兵不存真实数据，但它极大地统一了边界逻辑。

### 4.3 哨兵的好处

如果没有哨兵，很多操作都要分情况讨论：

- 当前节点是否为首节点？
- 当前节点是否为末节点？
- 插入位置是不是表头？
- 删除位置是不是表尾？

而有了 `header` 与 `trailer` 后，很多操作可以统一写成普通内部节点的形式。

---

## 5. 构造、初始化与析构

### 5.1 初始化 `init()`

初始化函数如下：

```cpp
template <typename T>
void List<T>::init() {
    header = new ListNode<T>;
    trailer = new ListNode<T>;
    header->succ = trailer;
    header->pred = NULL;
    trailer->pred = header;
    trailer->succ = NULL;
    _size = 0;
}
```

这说明空表不是“什么都没有”，而是至少有两个哨兵节点：

```text
header <-> trailer
```

### 5.2 为什么空表仍然有两个节点

因为这样可以保证：

- `header->succ` 一定存在
- `trailer->pred` 一定存在

于是：

- `first()` 可以统一为 `header->succ`
- `last()` 可以统一为 `trailer->pred`

### 5.3 区间复制与拷贝构造

`copyNodes()` 如下：

```cpp
template <typename T>
void List<T>::copyNodes(ListNodePosi<T> p, int n) {
    init();
    while (n--) {
        insertAsLast(p->data);
        p = p->succ;
    }
}
```

对应拷贝构造：

```cpp
List<T>::List(List<T> const& L) {
    copyNodes(L.first(), L._size);
}
```

时间复杂度为 `O(n)`。

### 5.4 清空与析构

析构函数与 `clear()` 为：

```cpp
template <typename T>
List<T>::~List() {
    clear();
    delete header;
    delete trailer;
}

template <typename T>
int List<T>::clear() {
    int oldSize = _size;
    while (0 < _size)
        remove(header->succ);
    return oldSize;
}
```

析构分两步：

1. 删除所有真实节点
2. 再释放两个哨兵节点

`clear()` 总复杂度是 `O(n)`。

---

## 6. 列表的循秩访问：可以做，但不应该常用

### 6.1 `operator[]` 的实现

```cpp
template <typename T>
T List<T>::operator[](Rank r) const {
    ListNodePosi<T> p = first();
    while (0 < r--) p = p->succ;
    return p->data;
}
```

这说明列表当然也能“模仿向量”做下标访问，但只是从首节点开始顺着 `succ` 走 `r` 步而已。

### 6.2 为什么复杂度是 `O(r)`

因为每次走一步只能跨过一个节点。  
若访问第 `r` 个元素，就必须做 `r` 次后继跳转。

因此：

- 最坏情况：`O(n)`
- 均匀分布下期望复杂度：`O(n)`

这再次说明：列表的优势不在随机访问，而在已知位置时的快速修改。

---

## 7. 无序列表：插入与删除

### 7.1 列表插入的思路

列表接口层的插入：

```cpp
template <typename T>
ListNodePosi<T> List<T>::insert(T const& e, ListNodePosi<T> p) {
    _size++;
    return p->insertAsPred(e);
}
```

即：把元素 `e` 作为节点 `p` 的前驱插入。  
真正的核心逻辑在节点方法 `insertAsPred()` 中。

### 7.2 `insertAsPred()` 的实现

```cpp
template <typename T>
ListNodePosi<T> ListNode<T>::insertAsPred(T const& e) {
    ListNodePosi<T> x = new ListNode(e, pred, this);
    pred->succ = x;
    pred = x;
    return x;
}
```

### 7.3 这一插入过程到底做了什么

假设当前节点是 `this`，它原来的前驱是 `pred`。  
现在要在它们中间插入一个新节点 `x`，其结果应变成：

```text
原来： pred <-> this
现在： pred <-> x <-> this
```

具体过程是：

1. 建立新节点 `x`
2. 让原前驱的 `succ` 改指向 `x`
3. 让当前节点的 `pred` 改指向 `x`

### 7.4 为什么赋值顺序不能颠倒

如果先做 `pred = x`，那么当前节点原来的前驱信息就丢了，后续再想修补原链就不对了。  
这是链表操作中非常经典的“先保存旧关系，再改指针”原则。

### 7.5 插入复杂度

只要插入位置 `p` 已知：

- 创建一个节点
- 改常数个指针

因此插入的时间复杂度是 `O(1)`。

### 7.6 列表删除的思路

删除位置 `p` 的节点时，就是把它从双向链中摘掉。

```cpp
template <typename T>
T List<T>::remove(ListNodePosi<T> p) {
    T e = p->data;
    p->pred->succ = p->succ;
    p->succ->pred = p->pred;
    delete p;
    _size--;
    return e;
}
```

### 7.7 删除的本质

设待删节点为 `p`，其前驱和后继分别是：

- `a = p->pred`
- `b = p->succ`

原结构：

```text
a <-> p <-> b
```

删除后变为：

```text
a <-> b
```

### 7.8 删除复杂度

若待删节点位置 `p` 已知，则删除只需修改常数个指针，因此时间复杂度也是 `O(1)`。

---

## 8. 无序列表：查找、去重与遍历

### 8.1 `find()`：在前驱中逆向查找

```cpp
template <typename T>
ListNodePosi<T> List<T>::find(T const& e, int n, ListNodePosi<T> p) const {
    while (0 < n--)
        if (e == (p = p->pred)->data)
            return p;
    return NULL;
}
```

它的语义是：在节点 `p` 的前面 `n` 个真前驱中，找等于 `e` 的**最靠后者**。

### 8.2 `find()` 复杂度

每次最多走 `n` 个前驱，因此复杂度为 `O(n)`。  
在整个无序列表中调用，本质上也是线性查找。

### 8.3 无序列表去重 `deduplicate()`

```cpp
template <typename T>
int List<T>::deduplicate() {
    int oldSize = _size;
    ListNodePosi<T> p = first();
    for (Rank r = 0; p != trailer; p = p->succ)
        if (ListNodePosi<T> q = find(p->data, r, p))
            remove(q);
        else
            r++;
    return oldSize - _size;
}
```

### 8.4 去重思路

- 维护一个“无重复前缀”
- 当前扫描到节点 `p` 时，在其前面长度为 `r` 的无重复前缀中查找是否已有相同值
- 若找到，说明出现重复，执行删除
- 若未找到，则无重复前缀长度加一

### 8.5 去重复杂度

外层扫描最多 `n` 次；每次调用 `find()` 最坏 `O(r)`。  
因此总复杂度是 `O(n^2)`。

### 8.6 `traverse()`：列表遍历

#### 函数指针版

```cpp
template <typename T>
void List<T>::traverse(void (*visit)(T&)) {
    for (NodePosi<T> p = header->succ; p != trailer; p = p->succ)
        visit(p->data);
}
```

#### 函数对象版

```cpp
template <typename T> template <typename VST>
void List<T>::traverse(VST& visit) {
    for (NodePosi<T> p = header->succ; p != trailer; p = p->succ)
        visit(p->data);
}
```

遍历本质上就是从首节点一路顺着 `succ` 扫到尾哨兵前，复杂度是 `O(n)`。

---

## 9. 有序列表：唯一化与查找

### 9.1 `uniquify()`：有序列表去重

```cpp
template <typename T>
int List<T>::uniquify() {
    if (_size < 2) return 0;
    int oldSize = _size;
    ListNodePosi<T> p = first(); ListNodePosi<T> q;
    while (trailer != (q = p->succ))
        if (p->data != q->data) p = q;
        else remove(q);
    return oldSize - _size;
}
```

### 9.2 为什么有序后去重能降到 `O(n)`

因为有序列表中，重复元素一定连续出现。  
所以只需检查相邻节点对 `(p, q)`：

- 若值不同，则 `p = q`
- 若值相同，则删去后者 `q`

总复杂度是 `O(n)`。

### 9.3 有序列表查找 `search()`

```cpp
template <typename T>
ListNodePosi<T> List<T>::search(T const& e, int n, ListNodePosi<T> p) const {
    do { p = p->pred; n--; }
    while ((-1 < n) && (e < p->data));
    return p;
}
```

它返回的是：

> 在节点 `p` 的前面 `n` 个真前驱中，**不大于 `e` 的最靠后者**。

失败时返回区间左边界前驱，可能是 `header`。

### 9.4 这个返回值语义为什么很重要

它不是简单返回“找到 / 没找到”，而是返回一个**边界位置**。  
于是后续就可以自然地用于插入：

```cpp
insert(search(e, r, p), e)
```

### 9.5 为什么有序列表查找仍然不能做到 `O(log n)`

原因不在“有序性”，而在“访问方式”：

- 有序向量能二分，是因为能 `O(1)` 随机访问中点
- 链表不能高效跳到中点，只能沿指针顺着走
- 因此无法真正享受二分查找的优势

所以有序列表查找复杂度仍然是：

- 最好 `O(1)`
- 最坏 `O(n)`
- 平均 `O(n)`

---

## 10. 列表上的选择排序

### 10.1 为什么要重新思考“交换”

每一趟扫描里，我们其实只是在找当前最大元素并把它送到正确位置。  
因此，比起做大量相邻交换，更直接的思路是：

1. 扫描找到最大者 `M`
2. 一次性把 `M` 挪到末端就位

这就是**选择排序**的核心思想。

### 10.2 列表版选择排序框架

```cpp
template <typename T>
void List<T>::selectionSort(ListNodePosi<T> p, int n) {
    ListNodePosi<T> head = p->pred, tail = p;
    for (int i = 0; i < n; i++) tail = tail->succ;
    while (1 < n) {
        insert(remove(selectMax(head->succ, n)), tail);
        tail = tail->pred; n--;
    }
}
```

### 10.3 这个算法在做什么

设待排序区间为 `(head, tail)`，长度为 `n`。

每轮迭代：

1. 在无序区里找到最大元素位置
2. 删除该节点
3. 把它插到 `tail` 之前
4. 尾部有序区增长 1
5. 无序区缩小 1

### 10.4 `selectMax()` 的实现

```cpp
template <typename T>
ListNodePosi<T> List<T>::selectMax(ListNodePosi<T> p, int n) {
    ListNodePosi<T> max = p;
    for (ListNodePosi<T> cur = p; 1 < n; n--)
        if (!lt((cur = cur->succ)->data, max->data))
            max = cur;
    return max;
}
```

### 10.5 为什么要用 `!lt()` 而不是 `>`

若有多个重复元素同时命中，通常约定“靠后者优先返回”。  
于是只要 `cur >= max` 就更新 `max = cur`，最终会得到“最靠后的最大元素”。

### 10.6 稳定性

若每轮把“靠后的最大值”拿出来放到后面，则相同元素最终相对顺序仍保持不变，这保证了选择排序的**稳定性**。

### 10.7 复杂度分析

每轮迭代中：

- `selectMax()` 需要 `Θ(n-k)` 次比较
- `remove()` 与 `insert()` 都是 `O(1)`

因此总复杂度为：

`Θ(n^2)`

---

## 11. 循环节（cycle）视角与无效交换

### 11.1 序列与循环节

任一排列都可分解为若干互不相交的循环节。  
这意味着，排序过程也可以从“如何逐步打散循环节、让元素归位”的角度理解。

### 11.2 交换法的单调性

采用交换法时，每迭代一步，最大元素 `M` 都会脱离原循环节，自成一个循环节。  
原循环节长度恰好减少 1，其余循环节不变。

### 11.3 无效交换与循环节个数

若 `M` 本来就已经在位，则无需交换。  
一个排列越接近正确位置分解，排序中的无效交换就越多。  
这正为后面的“输入敏感性”分析做铺垫。

---

## 12. 列表上的插入排序

### 12.1 插入排序的不变性

任意时刻都可把序列分为两部分：

S[0, r) + U[r, n)

其中：

- `S`：前缀有序
- `U`：后缀无序

初始化时 `r = 0`，然后不断把 `U` 的第一个元素取出，插入到 `S` 中适当位置，从而使 `S` 逐步扩大。

### 12.2 列表版插入排序实现

```cpp
template <typename T>
void List<T>::insertionSort(ListNodePosi<T> p, int n) {
    for (int r = 0; r < n; r++) {
        insert(search(p->data, r, p), p->data);
        p = p->succ;
        remove(p->pred);
    }
}
```

### 12.3 这段代码的逻辑

对当前节点 `p`：

1. 在其前面长度为 `r` 的有序前缀中执行 `search(p->data, r, p)`
2. 找到“不大于它的最后一个位置”
3. 在其后插入一个新的同值节点
4. 再删掉原节点

这等效于把当前元素从无序后缀中摘出来，插入到有序前缀的正确位置。

### 12.4 为什么这是就地算法（in-place）

它只使用常数个辅助指针，不额外申请与 `n` 成比例的空间，因此属于**就地算法**。

同时它还是**在线算法（online）**，因为新元素到来时，只需插入到已经有序的前缀中，不必重新看未来元素。

### 12.5 输入敏感性

插入排序具有输入敏感性：

#### 最好情况：完全有序

每次插入都只需 1 次比较、0 次交换，因此总时间为 `O(n)`。

#### 最坏情况：完全逆序

第 `k` 次插入需 `O(k)` 次比较，因此总时间为 `O(n^2)`。

### 12.6 为什么不在有序前缀里做二分查找

原因和前面一样：

- 列表不能 `O(1)` 随机访问中点
- 即便逻辑上想二分，也得线性走到中点
- 因此无法真正获得 `log n` 的好处

---

## 13. 逆序对与输入敏感性

### 13.1 什么是逆序对

若在序列 `A[0,n)` 中存在一对下标 `i < j`，但却有：

A[i] > A[j]

则称 `(A[i], A[j])` 构成一个逆序对。

### 13.2 起泡排序与逆序对

每交换一对紧邻逆序元素，逆序对总数恰好减一。  
因此对于起泡排序来说：

- 交换次数 = 输入序列逆序对总数

### 13.3 插入排序与逆序对

插入排序每次把一个元素插到有序前缀中，其比较次数也与它前方有多少更大的元素密切相关。  
因此，逆序对数实际上是衡量“距离有序有多远”的天然指标。

### 13.4 如何高效统计逆序对

蛮力方法需要 `O(n^2)`。  
借助归并排序，仅需 `O(n log n)`。

---

## 14. 列表上的归并排序

### 14.1 为什么归并排序特别适合链表

归并排序的核心需求是：

- 把序列分成两半
- 递归排序
- 在线性时间归并

对于链表来说，节点本身就是分散存在的，重新拼接链接往往比数组搬运更自然。

### 14.2 主算法

```cpp
template <typename T>
void List<T>::mergeSort(ListNodePosi<T>& p, int n) {
    if (n < 2) return;
    ListNodePosi<T> q = p;
    int m = n >> 1;
    for (int i = 0; i < m; i++) q = q->succ;
    mergeSort(p, m);
    mergeSort(q, n - m);
    p = merge(p, m, *this, q, n - m);
}
```

### 14.3 `merge()` 的实现思想

```cpp
template <typename T>
ListNodePosi<T>
List<T>::merge(ListNodePosi<T> p, int n, List<T>& L, ListNodePosi<T> q, int m) {
    ListNodePosi<T> pp = p->pred;
    while ((0 < m) && (q != p))
        if ((0 < n) && (p->data <= q->data)) {
            p = p->succ; n--;
        } else {
            insert(L.remove((q = q->succ)->pred), p);
            m--;
        }
    return pp->succ;
}
```

若：

- 左段当前首节点为 `p`
- 右段当前首节点为 `q`

则不断比较两边当前元素，较小者归入结果。  
由于链表支持已知位置的删除与插入都是 `O(1)`，因此整个归并过程是线性的。

### 14.4 复杂度分析

- 找中点要 `O(n)`
- 归并要 `O(n)`
- 递推式为：

T(n) = 2T(n/2) + O(n)

因此总复杂度是：

`O(n log n)`

---

## 15. 列表的游标实现（Cursor Implementation）

### 15.1 为什么还要讨论“无指针链表”

有些语言不支持指针；即便支持，频繁动态分配空间也可能影响效率。  
于是可以用数组模拟链表，也就是**游标实现（cursor implementation）**。

### 15.2 基本思路

用两组数组：

- `elem[]`：存储数据
- `link[]`：存储“下一个元素”的下标

数组下标就充当了“伪指针”。

同时维护两条逻辑链：

1. **data 链**：当前实际使用中的元素
2. **free 链**：尚未使用的空闲位置

### 15.3 插入与删除时在做什么

#### 插入

- 从 `free` 链中取出一个空闲单元
- 把数据写进 `elem[]`
- 修改 `link[]` 把它连入 `data` 链

#### 删除

- 从 `data` 链中摘掉某个位置
- 再把该位置挂回 `free` 链表头

### 15.4 游标实现的意义

这种结构的意义不只是“古老技巧”，而在于：

1. 它证明了“链表的本质不是指针语法，而是前后关系”
2. 它体现了内存池 / 自由链表（free list）的基本思想
3. 在某些系统编程、嵌入式场景中仍然有现实意义

---

## 16. Java 与 Python 列表：接口与语言机制

### 16.1 Java：接口与实现

Java 里的 `interface` 与 `implements` 很适合表达 ADT 思想：

- `interface Vector`
- `class Vector_Array implements Vector`
- `class Vector_ExtArray implements Vector`
- `interface List`
- `interface Sequence extends Vector, List`

这些例子说明：

- ADT 是接口层
- 数据结构是实现层
- 面向对象语言可以天然支持这种分离

### 16.2 Python：内置 list 的双重性

Python 的 `list`：

- 支持 `reverse()`
- 支持 `sort()`
- 支持切片
- 支持负下标
- 支持嵌套列表

但要特别注意：

> Python 的 `list` 在底层更接近动态数组，而不是本章讲的双向链表。

因此，虽然名字都叫 “list”，底层实现可能完全不同。

---

## 17. 本章主线总结

### 17.1 从向量到列表，是一次“访问方式”的改变

- 向量：循秩访问，擅长随机读
- 列表：循位置访问，擅长已知位置的局部修改

### 17.2 双向链表的核心优势

若节点位置已知，则：

- 插入是 `O(1)`
- 删除是 `O(1)`

### 17.3 哨兵节点统一了边界逻辑

`header` 与 `trailer` 让首尾节点的插入、删除、查找都可统一处理，显著减少特殊情况。

### 17.4 有序性能带来收益，但受底层结构限制

- 有序列表去重可降到 `O(n)`
- 有序列表插入位置也可更自然地确定
- 但查找无法像有序向量那样做到 `O(log n)`，因为列表缺少高效随机访问

### 17.5 三种排序在链表上的表现

- 选择排序：`Θ(n^2)`
- 插入排序：最好 `O(n)`，最坏 `O(n^2)`
- 归并排序：稳定且可达 `O(n log n)`，非常适合链表

### 17.6 游标实现说明：链表本质上是“关系结构”

即使没有真实指针，也能通过数组 + link[] 模拟逻辑上的链式连接。

---

## 18. 一句话总结

> 列表这一章真正重要的，并不只是学会写双向链表，而是理解：当底层存储不再支持高效随机访问时，算法设计必须改用“循位置”的思维；同时，插入、删除、排序、去重、查找等操作的优劣，也都将随这一结构变化而重新洗牌。
