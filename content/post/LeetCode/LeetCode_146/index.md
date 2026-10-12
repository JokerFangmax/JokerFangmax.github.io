---
title: "LeetCode 146: LRU 缓存"
date: 2026-11-03
math: true
tags:
  - LeetCode
  - 设计
  - 哈希表
  - 双向链表
categories:
  - LeetCode
description: "LeetCode 146 LRU 缓存题解（哈希表加双向链表）"
---

# LeetCode 146. LRU 缓存

> 题目链接：[https://leetcode.cn/problems/lru-cache/](https://leetcode.cn/problems/lru-cache/)  
> 难度：中等  
> 标签：设计、哈希表、双向链表

---

## 1. 题目描述

请你设计并实现一个满足 **LRU (最近最少使用，Least Recently Used)** 约束的缓存结构，支持：

- `LRUCache(int capacity)`：以**正整数**容量初始化；
- `get(int key)`：如果 `key` 存在于缓存中，返回其值，否则返回 $-1$；
- `put(int key, int value)`：`key` 已存在则**变更**其值；不存在则**插入**。若插入导致键数超过容量，**逐出最久未使用**的键。

`get` 和 `put` 的每次调用都要求 $O(1)$ 平均时间复杂度。

**数据范围：**

- $1 \le capacity \le 3000$
- $0 \le key \le 10^4$
- $0 \le value \le 10^4$
- 最多调用 $3 \times 10^4$ 次 `get` 和 `put`
- 保证逐出操作发生在 `put` 且缓存已满时

**示例：**

```
输入：
  LRUCache(2)
  put(1, 1)            缓存 {1=1}
  put(2, 2)            缓存 {1=1, 2=2}
  get(1)      返回 1   使用了 1，1 变为最新
  put(3, 3)            淘汰最久未用的 2，缓存 {1=1, 3=3}
  get(2)      返回 -1  2 已被逐出
  put(4, 4)            淘汰 1，缓存 {3=3, 4=4}
  get(1)      返回 -1
  get(3)      返回 3
  get(4)      返回 4
```

"最近使用"包括 `get` 命中和 `put` 写入（无论新增还是更新），**任何被碰到的 key 都算"刚使用过"**。

---

## 2. 思路分析

### 2.1 暴力：数组或单向链表，各缺一半

先看这个数据结构需要同时支持哪些操作：

| 操作 | 谁需要它 |
|------|----------|
| 按 `key` 查找 | `get`、`put` |
| 把某节点移到"最新"端 | 每次 `get` 命中、`put` 命中 |
| 删除"最旧"节点 | 容量满时的 `put` |

**用数组存 `(key, value)`**：查找 $O(n)$（或先排序再二分，但插入要挪元素），淘汰尾部 $O(1)$，移动到头部 $O(n)$。
**用单链表按新旧排序**：移动到头部需要先摘下节点，而摘节点需要**前驱**，单链表找不到前驱，只能从头 $O(n)$ 扫。

两条暴力路都是 $O(n)$ 每次操作，$3 \times 10^4$ 次调用、容量 3000 时最坏 $9 \times 10^7$ 次指针操作——能蹭过去但达不到题目"每次 $O(1)$"的要求。

### 2.2 正解：两个数据结构各管一件事

暴力失败的根源是**让一个结构干两件它不擅长的事**。拆开看：

- 查找要快 → **哈希表 (hash table)**：$O(1)$ 定位 key；
- 移动节点要快 → **双向链表 (doubly linked list)**：节点持有前驱指针，摘除自身、插入任意位置都是 $O(1)$。

组合方式：哈希表存 `key → 链表节点指针`，链表按**从新到旧**排列（头部最新、尾部最旧）。

```
哈希表                     双向链表（头 = 最新）
key=1 ──> [1,one] <===> [3,three] <===> [4,four]
key=3 ──>      ^            ^             ^
key=4 ──>      └────────────┴─────────────┘
```

三种操作全部 $O(1)$ 化：

| 操作 | 步骤 | 代价 |
|------|------|------|
| `get(k)` 命中 | 查哈希表 → 摘下节点 → 接到链表头 → 返回值 | $O(1)$ |
| `put(k,v)` 已存在 | 查哈希表 → 摘下 → 改值 → 接到头 | $O(1)$ |
| `put(k,v)` 新增且满 | 淘汰链表尾节点 → 从哈希表 `erase` 它的 key → 新节点接头部、哈希表登记 | $O(1)$ |

### 2.3 用示例走一遍

容量 2，链表用 `[头..尾]` 表示新旧顺序（左新右旧）：

| 调用 | 链表变化 | 说明 |
|------|----------|------|
| `put(1,1)` | `[1]` | 新增 |
| `put(2,2)` | `[2,1]` | 新增，2 在头部 |
| `get(1)` | `[1,2]` | 命中，1 移到头部，返回 1 |
| `put(3,3)` | 淘汰尾部的 2 → `[3,1]` | 满了，逐出最久未用的 2 |
| `get(2)` | `[3,1]` | 未命中，返回 $-1$ |
| `put(4,4)` | 淘汰尾部的 1 → `[4,3]` | 逐出 1 |
| `get(1)` / `get(3)` / `get(4)` | — | $-1$、3、4 |

注意第 3 行：**`get` 命中也改变新旧顺序**——这是 LRU 与普通缓存最大的差别，也是"最近使用"定义的一部分。

---

## 3. 代码实现

```cpp
class LRUCache {
    struct Node {
        int key, val;
        Node *prev, *next;
        Node(int k, int v) : key(k), val(v), prev(nullptr), next(nullptr) {}
    };

    int cap;                                  // 容量
    unordered_map<int, Node*> map;            // key -> 链表节点
    Node *head, *tail;                        // 哨兵：head->next 最新，tail->prev 最旧

    // 把节点 x 从链表中摘下（O(1)，双向链表的核心红利）
    void remove(Node* x) {
        x->prev->next = x->next;
        x->next->prev = x->prev;
    }

    // 把节点 x 插到头部（最新位置）
    void pushFront(Node* x) {
        x->next = head->next;
        x->prev = head;
        head->next->prev = x;
        head->next = x;
    }

public:
    LRUCache(int capacity) : cap(capacity) {
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head->next = tail;
        tail->prev = head;                    // 两个哨兵互相牵手，空链表成立
    }

    int get(int key) {
        auto it = map.find(key);
        if (it == map.end()) return -1;
        Node* node = it->second;
        remove(node);                         // 命中也要"变新"
        pushFront(node);
        return node->val;
    }

    void put(int key, int value) {
        auto it = map.find(key);
        if (it != map.end()) {                // 已存在：改值 + 变新
            Node* node = it->second;
            node->val = value;
            remove(node);
            pushFront(node);
            return;
        }
        if ((int)map.size() == cap) {         // 满：淘汰最旧（尾部）
            Node* victim = tail->prev;
            remove(victim);
            map.erase(victim->key);           // 摘链和 erase 用同一个 key
            delete victim;
        }
        Node* node = new Node(key, value);
        pushFront(node);
        map[key] = node;
    }
};
```

几个实现细节：

1. **节点里必须存 `key`**：淘汰尾部节点时要从哈希表里删掉对应条目，只知道 value 不够用——这是最容易漏的字段。
2. **伪头 + 伪尾 (dummy head/tail)**：真正的首尾节点永远不空，`remove` / `pushFront` 不用做任何判空，四个指针赋值走天下。没有哨兵的版本每处都要写 `if (prev)`、`if (next)`，出 bug 概率翻倍。
3. **淘汰顺序**：先摘链、再 `erase` 哈希、再 `delete`，一气呵成；忘记 `erase` 会导致哈希表里挂着悬垂指针，后续 `get` 命中已死节点直接崩溃。
4. `put` 的两条分支（已存在 / 新增）**先查后动**，已存在分支绝不触发淘汰。

---

## 4. 复杂度分析

- **时间复杂度**：`get` 和 `put` 均为均摊 $O(1)$。哈希表查找 $O(1)$（平均），链表摘除、头插、尾删都是常数次指针操作。
- **空间复杂度**：$O(capacity)$。哈希表与链表各存一份引用，节点总数被容量钳制在 $\le capacity$。

对比暴力解：

| 结构 | get | put(淘汰时) | 是否达标 |
|------|-----|------------|----------|
| 数组扫 | $O(n)$ | $O(n)$ | 否 |
| 单链表 | $O(n)$ | $O(n)$ | 否 |
| 哈希表 + 双向链表 | $O(1)$ | $O(1)$ | 是 |

哈希表解决"**找到**"，双向链表解决"**移动**"——任何"查找快 + 顺序维护快"的复合需求，都可以想想这对组合。

---

## 5. 常见疑问

**Q1：为什么 `get` 命中也要把节点移到头部？少做这一步会怎样？**

LRU 的淘汰依据是"**最久未被使用**"，而"使用"= `get` 命中 + `put` 写入。如果 `get` 命中不更新位置，链表记录的就只是"插入顺序"（FIFO 缓存），行为完全不同：容量 2 时依次 `put(1,1)`、`put(2,2)`、`get(1)`、`put(3,3)`——正确 LRU 淘汰 2，FIFO 却淘汰 1。面试官最爱在示例里埋这种坑。

**Q2：淘汰时为什么不能只删链表、不删哈希表（或反之）？**

两个结构必须**严格同步**。只删链表：哈希表残留"key → 已释放内存"，下次 `get` 这个 key 会命中死节点（可能返回陈旧值，也可能直接段错误）；只删哈希：链表越长越长，内存泄漏且容量判断 `map.size() == cap` 失真。节点里存 `key` 就是为了在摘链之后还能反查哈希条目。

**Q3：可以直接用 `std::list` + `unordered_map<int, list<pair<int,int>>::iterator>` 吗？**

可以，C++ 标准库的 `std::list` 就是双向链表，`list::erase` / `splice` 均 $O(1)$，`list` 的迭代器在元素被删除前始终有效——这是库版能工作的关键保证。生产代码里该用库；但**面试建议手写**：本题考的正是哨兵节点、双向指针操作这些被标准库封装掉的细节，而且很多公司（包括以这道题出名的某电商公司）明确要求手写。

---

## 6. 相关题目与模板

> 📚 哈希表"用空间换时间"的核心思想见模板笔记：[哈希表：从计数到映射](../../algorithms/hash_table/)。

**手写双向链表模板**（哨兵头尾 + 摘下 + 头插，LRU / LFU / O(1) 数据结构题通用骨架）：

```cpp
struct Node { int k, v; Node *prev, *next; };

void remove(Node* x) {            // 摘下任意节点：4 次指针赋值，无需判空（有哨兵）
    x->prev->next = x->next;
    x->next->prev = x->prev;
}
void pushFront(Node* x) {         // 接到哨兵头之后（最新位）
    x->next = head->next; x->prev = head;
    head->next->prev = x; head->next = x;
}
```

**方法论总结**："哈希表 + 双向链表"是**索引与顺序分离**的范式：哈希表负责随机定位，链表负责维护动态顺序，节点本体作为两者的交汇点（所以节点要把 key 存下来，供反向删除索引）。操作系统页面置换、Redis `LRU` 近似策略、数据库缓冲池，用的都是这套思想。

**同类题目：**

- [LeetCode 460. LFU 缓存](https://leetcode.cn/problems/lfu-cache/)：困难版，再叠加一层"按频率分桶"，每桶是一个双向链表
- [LeetCode 432. 全 O(1) 的数据结构](https://leetcode.cn/problems/all-oone-data-structure/)：同款"哈希 + 双向链表 + 桶"的极致运用
- [LeetCode 380. O(1) 时间插入、删除和获取随机元素](https://leetcode.cn/problems/insert-delete-getrandom-o1/)：另一种组合——哈希表 + 数组，体会"两个 $O(1)$ 结构互补"的通用套路
- [LeetCode 138. 复制带随机指针的链表](https://leetcode.cn/problems/copy-list-with-random-pointer/)：同为"哈希表维护节点对应关系"的链表设计题

**一句话总结**：LRU 缓存 = 哈希表管"找得到" + 双向链表管"排得动"，容量溢出时从链表尾巴上摘掉最旧的那一个——两个平凡的结构拼出一个 $O(1)$ 的不平凡整体。
