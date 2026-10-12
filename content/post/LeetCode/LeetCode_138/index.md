---
title: "LeetCode 138: 随机链表的复制"
date: 2026-10-31
math: true
tags:
  - LeetCode
  - 哈希表
  - 链表
categories:
  - LeetCode
description: "LeetCode 138 随机链表的复制题解（哈希表与原地拼接）"
---

# LeetCode 138. 随机链表的复制

> 题目链接：[https://leetcode.cn/problems/copy-list-with-random-pointer/](https://leetcode.cn/problems/copy-list-with-random-pointer/)  
> 难度：中等  
> 标签：哈希表、链表

---

## 1. 题目描述

给你一个长度为 $n$ 的链表，每个节点除了 `val` 和 `next` 之外，还多出一个 `random` 指针——它可以指向链表中的**任意节点**（包括自己），也可以为空。

要求构造这个链表的**深拷贝 (deep copy)**：新链表由全新节点组成，`val`、`next`、`random` 的指向关系与原链表完全一致，但每个节点都是重新 `new` 出来的，与原链表不共享任何节点。返回新链表的头节点。

**数据范围：**

- $0 \le n \le 1000$
- $-10^4 \le Node.val \le 10^4$
- `Node.random` 为空，或指向链表中的任意节点
- 链表保证不成环（`next` 方向）

**示例：**

```
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]

原链表：
  节点0(7)  random = null
  节点1(13) random = 节点0
  节点2(11) random = 节点4
  节点3(10) random = 节点2
  节点4(1)  random = 节点0

输出：结构完全相同的全新链表（5 个新节点，random 关系一一对应）
```

`[val, random_index]` 表示一个节点：`val` 是节点值，`random_index` 是 `random` 指向的节点下标（从 0 开始），`null` 表示空指针。

---

## 2. 思路分析

### 2.1 朴素：一次遍历直接复制？

顺着链表扫，每遇到原节点 `cur` 就 `new` 一个 `cur'`，抄下 `val`，接上 `next`——到目前为止一切顺利。问题出在 `random` 上：

- 如果 `cur->random` 指向**后面**还没遍历到的节点，此刻它的拷贝根本不存在，`cur'->random` 无处可指；
- 如果 `cur->random` 指向**前面**的节点，虽然拷贝已经建好，但你没有记录"原节点 → 拷贝节点"的对应关系，照样找不到它。

也就是说，**一次遍历 + 直接建链行不通**，除非题目保证 `random` 只向后指。既然 `random` 是"任意指"，我们需要一份**任意原节点到其拷贝的映射**。怎么存这份映射，就是本题的全部戏份。

### 2.2 哈希表：把映射显式记下来

最直接的办法：开一个 `unordered_map<Node*, Node*>`，键是原节点，值是它的拷贝。

| 遍历 | 做什么 | 为什么 |
|------|--------|--------|
| 第一遍 | 逐个 `new` 出拷贝节点（只抄 `val`），存入 `map[cur] = cur'` | 先把所有节点造出来，指针后面再接 |
| 第二遍 | `map[cur]->next = map[cur->next]`；`map[cur]->random = map[cur->random]` | 此时任何节点的拷贝都已存在，查表即可 |

注意 `cur->next` 或 `cur->random` 为 `nullptr` 时查表会落空——预先插入 `map[nullptr] = nullptr` 一行即可统一处理。

时间 $O(n)$、空间 $O(n)$（哈希表）。已经达标，但它有一个不优雅之处：**明明原链表自己就能记住映射，却额外开了一张表**。能不能省掉？

### 2.3 原地拼接：把映射藏进链表结构里

观察 2.2 的痛点：需要一个"原节点 → 拷贝节点"的查询结构。其实链表本身就有现成的位置——**把拷贝节点直接插在原节点后面**，那么"原节点 $x$ 的下一个节点"天然就是 $x$ 的拷贝，映射关系被编织进了链表结构。

整个过程分三步：

1. **交错复制**：遍历原链表，每个节点 `cur` 后面插入拷贝 `cur'`，得到 $A \to A' \to B \to B' \to \cdots$；
2. **设置 random**：此时 `cur->random` 是某个**原**节点，而原节点的下一个就是它的拷贝，于是有公式
   $$cur\text{->}next\text{->}random = cur\text{->}random\text{->}next$$
3. **拆链**：把交错链表分离成原链表和新链表两条。

**用小例子走一遍。** 设原链表为 $A(7) \to B(13) \to C(11)$，其中 $B.random = A$，$C.random = C$，$A.random = null$：

| 阶段 | 链表状态 | 说明 |
|------|----------|------|
| 第 1 步后 | $A \to A' \to B \to B' \to C \to C'$ | 每个 `cur` 后插 `cur'` |
| 第 2 步·A | $A'.random = null$ | $A.random = null$，直接置空 |
| 第 2 步·B | $B'.random = A' = A$ 的下一个 | $B.random = A$，抄"下一个" |
| 第 2 步·C | $C'.random = C' = C$ 的下一个 | $C.random = C$，自指也成立 |
| 第 3 步 | 原：$A \to B \to C$；新：$A' \to B' \to C'$ | 隔一个取一个，拆成两条 |

空间降到 $O(1)$（不计输出链表），时间仍是两三遍线性扫。

---

## 3. 代码实现

```cpp
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (!head) return nullptr;

        // 第 1 步：交错复制，每个原节点后面插入它的拷贝
        // 交错后 cur->next 就是 cur 的拷贝，映射关系藏在链表里
        for (Node* cur = head; cur; ) {
            Node* copy = new Node(cur->val);
            copy->next = cur->next;
            cur->next = copy;
            cur = copy->next;          // 跳过新节点，走到下一个原节点
        }

        // 第 2 步：设置每个拷贝节点的 random
        for (Node* cur = head; cur; cur = cur->next->next) {
            if (cur->random)
                cur->next->random = cur->random->next;
            // cur->random 为空时，拷贝的 random 保持默认的空即可
        }

        // 第 3 步：拆链，分离原链表与新链表
        Node* dummy = new Node(0);
        Node* tail = dummy;
        for (Node* cur = head; cur; ) {
            tail->next = cur->next;    // 摘下拷贝节点接到新链表
            tail = tail->next;
            cur->next = cur->next->next;  // 原链表跳过拷贝，恢复原样
            cur = cur->next;
        }
        return dummy->next;
    }
};
```

几个实现细节：

1. **第 1 步的推进是 `cur = copy->next`**，一次跳两个节点（原 → 拷贝 → 下一个原），否则会把拷贝节点当原节点再复制一遍。
2. **第 2 步用 `cur->next->next` 推进**，同样只落在原节点上。
3. **拆链时同步恢复原链表**（`cur->next = cur->next->next`）。虽然 LeetCode 不检查原链表，把它完好地还回去是深拷贝题的礼仪。
4. `Node` 的构造函数中 `next`、`random` 默认初始化为空，第 2 步的"判空跳过"因此成立。

---

## 4. 复杂度分析

- **时间复杂度**：$O(n)$。三遍线性扫描，每遍对每个节点做 $O(1)$ 工作。
- **空间复杂度**：$O(1)$。只用了常数个指针变量；`new` 出来的 $n$ 个拷贝节点是**输出本身**，按惯例不计入。

对比两种解法：

| 解法 | 时间 | 额外空间 | 特点 |
|------|------|----------|------|
| 哈希表 | $O(n)$ | $O(n)$ | 思路直白，两遍扫，写错概率低 |
| 原地拼接 | $O(n)$ | $O(1)$ | 三步曲，公式巧妙，面试加分 |

---

## 5. 常见疑问

**Q1：哈希表版本里，`map[cur->random]` 会不会查不到？**

会——当 `cur->random == nullptr` 时，`nullptr` 没有作为键插入过，`operator[]` 会现场插入一个"空指针 → 默认构造值"的条目，返回的值是垃圾。两种修法：提前写一句 `map[nullptr] = nullptr;`，或者拆成

```cpp
Node* copy = map[cur];
copy->next  = cur->next  ? map[cur->next]  : nullptr;
copy->random = cur->random ? map[cur->random] : nullptr;
```

原地拼接版对应的坑是忘记判空：`cur->next->random = cur->random->next` 在 `cur->random` 为空时直接解引用空指针，所以判空必须放在前面。

**Q2：第 3 步拆链，为什么原链表和新链表不会互相"扯断"？**

拆链用的是"摘一个、还一个"的交替节奏：先把拷贝节点 `cur->next` 摘下来接到新链表尾部，**再**让原节点跨过拷贝接回 `cur->next->next`。两步都基于同一个局部快照，任何一个时刻每条链都至多有一处"断口"，且下一句立刻缝合。如果先恢复原链再取拷贝，拷贝节点就丢了——顺序不能反。

**Q3：原地拼接是严格更优的吗？什么时候还是该选哈希表？**

原地拼接胜在空间，但代价是：遍历过程中链表处于"交错"的中间状态，如果中途抛异常或提前返回，链表是被污染的；三步的推进方式（跳一个 vs 跳两个）也容易写错。当空间不是硬约束（$n \le 1000$ 时哈希表毫无压力），或图的结构更复杂（比如 [133. 克隆图](https://leetcode.cn/problems/clone-graph/) 有环，节点无法"插在自己后面"），哈希表映射是更稳的通用解法。

---

## 6. 相关题目与模板

> 📚 哈希表"用映射解决对应关系"的思想见模板笔记：[哈希表：从计数到映射](../../algorithms/hash_table/)。

**哈希表版本**（通用、稳，图 / 复杂结构克隆首选）：

```cpp
class Solution {
public:
    Node* copyRandomList(Node* head) {
        unordered_map<Node*, Node*> map;
        map[nullptr] = nullptr;                 // 统一处理空指针
        for (Node* cur = head; cur; cur = cur->next)
            map[cur] = new Node(cur->val);      // 第一遍：只建节点
        for (Node* cur = head; cur; cur = cur->next) {
            map[cur]->next   = map[cur->next];
            map[cur]->random = map[cur->random]; // 第二遍：接指针
        }
        return map[head];
    }
};
```

**方法论总结**：深拷贝的难点从来不是"复制"，而是**任意引用的对应关系**——`random`、图的边、回边都一样。两条路线：

1. **显式记账**：哈希表维护"旧 → 新"，先建点后接边，两遍扫描；
2. **结构内嵌**：把新节点挂在旧节点身边（本题的交错链），让指针本身充当映射，最后拆开。

**同类题目：**

- [LeetCode 133. 克隆图](https://leetcode.cn/problems/clone-graph/)：同款"旧 → 新"映射，但图有环，必须用哈希表防止重复复制 + 死循环
- [LeetCode 143. 重排链表](https://leetcode.cn/problems/reorder-list/)：同为"原地改指针 + 交错重排"的链表 surgery，练手感
- [LeetCode 146. LRU 缓存](https://leetcode.cn/problems/lru-cache/)：哈希表 + 双向链表的组合拳，映射思想的另一种形态

**一句话总结**：复制一条带任意指针的链表，本质是维护一张"旧 → 新"的字典——要么真的开一张哈希表，要么把字典写成链表本身。
