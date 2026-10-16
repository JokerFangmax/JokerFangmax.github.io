---
title: "AT 03: 链表——虚拟头节点、反转全家桶与快慢指针"
date: 2026-11-26
math: true
tags:
  - Notes
  - 算法模板
  - 链表
  - 双指针
categories:
  - 算法模板
description: "链表刷题三板斧：虚拟头节点统一边界、反转全家桶（迭代/递归/区间/K 组）、快慢指针（中点/环/倒数第 k），合并有序链表与 LRU，样题：LC 206/92/25/141/142/19/21/23/146/234"
---

# AT 03：链表——虚拟头节点、反转全家桶与快慢指针

> 核心思想一句话：**链表题的一切恐惧来自"头节点和普通节点不一样"和"断链后找不回后继"——虚拟头节点消灭前者，`nxt` 暂存消灭后者，快慢指针把"数长度"变成一次扫描。**
> 理论后盾：[DSA 3（列表）](../../DSA/DSA_3/)——哨兵节点（§4）、指针操作（§7）、循环节视角（§11）。

---

## 1. 这个模板解决什么问题

链表题在面试中密度极高，因为每个操作都考察指针细节；在竞赛中主要作为其他结构的底座（邻接表、前向星）。刷题套路收敛成三件武器：

| 武器 | 消灭的痛点 | 代表题 |
|------|-----------|--------|
| 虚拟头节点 dummy | 头节点被删/被换时不用特判 | LC 19/21/23/146 |
| 反转骨架 `prev/cur/nxt` | 断链丢后继 | LC 206/92/25 |
| 快慢指针 | 两次扫描合一：中点/环/倒数第 k | LC 141/142/876 |

为什么 `vector` 明明能做一切，还要学链表——不是性能（缓存局部性上链表完败），是**指针操纵能力本身在考**：树、图、Trie 全是链表操作的推广（[AT 07](../AT_07_binary_tree/)、[AT 11](../AT_11_graph_bfs_dfs/)、[AT 20](../AT_20_trie/)）。

---

## 2. 代码模板

### 2.1 虚拟头节点：消灭"第一个"特判

```cpp
ListNode* dummy = new ListNode(0, head);   // 指向头的前一个假节点
// ... 一切操作都从 dummy 出发，不存在"操作对象是头节点"的特例
ListNode* p = dummy;
while (p->next && 条件) p = p->next;       // 停在"目标的前驱"上
delete dummy;                              // LeetCode 上可省
```

要删任何节点，都先走到它的**前驱**再 `p->next = p->next->next`。头节点的前驱本来不存在——dummy 补上了它。什么时候必用：**答案链表从空开始构造**（LC 21 合并）、**可能删头**（LC 19/237）、**K 组反转**（LC 25）。刷题记一条：链表题第一行先写 dummy，九成不亏。

### 2.2 反转骨架：prev / cur / nxt 三指针

```cpp
// 迭代反转整条链，返回新头（原尾）
ListNode* reverse(ListNode* head) {
    ListNode *prev = nullptr, *cur = head;
    while (cur) {
        ListNode* nxt = cur->next;   // ① 暂存后继（马上要断）
        cur->next = prev;            // ② 反指
        prev = cur;                  // ③ prev 前进
        cur  = nxt;                  // ④ cur 前进
    }
    return prev;                     // cur 为空时 prev 恰是原尾
}
```

四行的顺序不可换：`nxt` 必须在 ② 断链**之前**存下。**不变量**：循环任何时刻，`prev` 领着已反转的前缀、`cur` 领着未反转的后缀，两段由 `cur->next`（原方向）尚未处理的那条边分界。

区间反转（LC 92）= 定位前驱 + 局部跑这个骨架 + 接回两端：

```cpp
// 反转 [left, right]（1 起下标），返回头
ListNode* reverseBetween(ListNode* head, int left, int right) {
    ListNode* dummy = new ListNode(0, head);
    ListNode* p = dummy;
    for (int i = 0; i < left - 1; i++) p = p->next;   // p = 前驱
    ListNode *a = p->next, *b = a->next;
    for (int i = 0; i < right - left; i++) {          // 逐个头插到 p 后
        a->next = b->next;                            // 摘下 b
        b->next = p->next;                            // b 接到反转段头
        p->next = b;                                  // p 认 b 作段头
        b = a->next;                                  // 下一个待摘的
    }
    return dummy->next;
}
```

这个"摘下-头插"写法比"整段反转再接回"少一个循环，也顺便是 K 组反转（LC 25）的内核：数够 $k$ 个就做一次区间反转，不够就保持原样。

<details><summary>递归反转（理解递归的练习题，实战用迭代）</summary>

```cpp
ListNode* reverse(ListNode* head) {
    if (!head || !head->next) return head;    // 空或单节点：自己就是新头
    ListNode* newHead = reverse(head->next);  // 反转后面的，newHead 传到底
    head->next->next = head;                  // 后继反指我
    head->next = nullptr;                     // 我暂时收尾（若是新头就永久收尾）
    return newHead;
}
```

心智模型：递归到尾拿到新头，回溯时每个节点只做一件事——让我的后继指向我。$O(n)$ 时间、$O(n)$ 栈空间。
</details>

### 2.3 快慢指针：一次扫描定中点 / 判环 / 倒数第 k

```cpp
// 中点（偶数长度停在右中点；slow 走一步 fast 走两步）
ListNode *slow = head, *fast = head;
while (fast && fast->next) { slow = slow->next; fast = fast->next->next; }

// 判环：fast 追上 slow 即有环（Floyd 判圈）
while (fast && fast->next) {
    slow = slow->next; fast = fast->next->next;
    if (slow == fast) { /* 有环 */ }
}

// 环入口：相遇后一指针回头，同速前进再相遇即入口
ListNode* p1 = head, p2 = slow;             // slow 为相遇点
while (p1 != p2) { p1 = p1->next; p2 = p2->next; }
// p1 == p2 即入口

// 倒数第 k 个的前驱（间隔 k 的平行指针）
ListNode *fast = head;
for (int i = 0; i < k; i++) fast = fast->next;   // fast 先走 k 步
ListNode *slow = head;
while (fast) { slow = slow->next; fast = fast->next; }
// slow 即倒数第 k 个（从 dummy 出发则得其前驱）
```

三个应用共享同一个机制：**两指针速率差或相位差编码了"距离"**。判环为什么一定会相遇：fast 每轮追近 1 步，环周长有限，最多一圈内追上。环入口公式的推导见折叠块。

<details><summary>环入口公式的三行推导（LC 142 常被追问）</summary>

设头到入口距离 $a$，入口到相遇点 $b$，相遇点绕回入口 $c$。相遇时 slow 走 $a + b$，fast 走 $a + b + k(b + c)$（$k \ge 1$ 圈），且 fast 路程是 slow 的两倍：

$$
2(a+b) = a + b + k(b+c) \;\Rightarrow\; a = (k-1)(b+c) + c
$$

即"从头走 $a$ 步" = "从相遇点绕 $(k-1)$ 圈再走 $c$ 步"——同速前进必然在入口相遇。这就是模板里 `p1` 回头再走的依据。
</details>

### 2.4 合并有序链表：穿针引线

```cpp
// 合并两条有序链（LC 21），返回合并头
ListNode* merge(ListNode* a, ListNode* b) {
    ListNode* dummy = new ListNode(0);
    ListNode* tail = dummy;                     // 尾插：tail 永远指向已合并段尾
    while (a && b) {
        if (a->val <= b->val) { tail->next = a; a = a->next; }
        else                   { tail->next = b; b = b->next; }
        tail = tail->next;
    }
    tail->next = a ? a : b;                     // 剩下的一整条直接挂上
    return dummy->next;
}
```

k 条合并（LC 23）：每次取 $k$ 条中最小头——小根堆（[AT 09 堆](../AT_09_heap/)）$O(n \log k)$，或分治两两合并同复杂度。**`tail` 尾插 + 剩余段直挂** 是构造链表的通用姿势。

---

## 3. 常见变形与坑

| 变形 | 一句话做法 |
|------|-----------|
| 删除倒数第 N 个 | dummy + 快慢指针（fast 先走 n+1），slow 停在前驱 |
| 回文链表 | 快慢指针找中点 + 后半反转 + 双向比对（$O(1)$ 空间，LC 234） |
| 重排链表 | 找中点 + 后半反转 + 交错合并 |
| 相交链表 | 双指针走完自己换到对方头，路程相等必同步到达交点 |
| 排序链表 | 归并：快慢分半 + 递归 + 2.4 合并（$O(n\log n)$） |
| 两数相加 | dummy + 逐位 + 进位，尾插 |

坑：

1. **断链前不存 `nxt`**——后继永久丢失，链断成两截；
2. **删节点时手里没有前驱**——从 dummy 出发，循环条件写 `p->next` 而不是 `p`；
3. **快指针初始化错误**：中点要 `slow = fast = head`；倒数第 k 要 fast 先走 k 步、slow 从 dummy 出发——两类题的相位差不同，混用差一；
4. `while (fast && fast->next)` 两个条件**顺序不能反**（先解引用 `fast->next` 会空指针解引用）；
5. K 组反转（LC 25）数不足 $k$ 组时**不要反转**，且要接回"当前已处理段"的尾部——忘记接尾产生游离链。

---

## 4. 样题精讲

### 4.1 LC 206 反转链表：三指针骨架的原型

[题解](../../LeetCode/LeetCode_206/)（Easy）。§2.2 模板原样套用。面试官常追问的三个延伸：递归版（折叠块）、反转前 N 个（区间反转取 `right = N`）、每 k 个一组（LC 25）——全在同一骨架上加定位。

### 4.2 LC 141/142 环形链表：Floyd 判圈双题

[LC 141 题解](../../LeetCode/LeetCode_141/)（Easy）判有无环：快慢指针，相遇即有环；`fast` 到 `nullptr` 则无环。[LC 142 题解](../../LeetCode/LeetCode_142/)（Medium）找环入口：相遇后一指针回头，同速再走，相遇点即入口（折叠块有三行推导）。$O(n)$ 时间、$O(1)$ 空间——哈希表存访问过的节点也能做但 $O(n)$ 空间，面试必须给指针解。

### 4.3 LC 25 K 个一组翻转链表（Hard）：全部套路合体

[题解](../../LeetCode/LeetCode_25/)。每 $k$ 个一组反转，不足 $k$ 保持原样。dummy + 分组定位 + 头插反转三合一：

```cpp
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode* dummy = new ListNode(0, head);
        ListNode* prev = dummy;                     // 上一组的尾
        while (true) {
            // 1. 探路：从 prev 起数 k 个，看本组是否凑齐
            ListNode* tail = prev;
            for (int i = 0; i < k && tail; i++) tail = tail->next;
            if (!tail) break;                       // 不足 k 个，保持原样收工
            // 2. 头插反转 [prev->next, tail]
            ListNode *a = prev->next, *b = a->next;
            while (a != tail) {                     // 摘 b 头插，直到 a 到达 tail
                a->next = b->next;
                b->next = prev->next;
                prev->next = b;
                b = a->next;
            }
            a->next = b;                            // 接上下一组
            prev = a;                               // prev 挪到本组新尾
        }
        return dummy->next;
    }
};
```

结构与 §2.2 区间反转完全一致，只是外面包了一层"分组探路"。这道题为什么是 Hard：每一步都是 Easy 骨架，**组合时指针状态多**——dummy 消掉头组特判、`prev` 跨组衔接、`tail` 判组完整性，三个不变量缺一即断链。

### 4.4 LC 146 LRU 缓存：链表 + 哈希的经典设计题

[题解](../../LeetCode/LeetCode_146/)（Medium）。`get`/`put` 均 $O(1)$：**双向链表按访问序排布 + 哈希表定位节点**。

- 哈希表 `key → 链表节点`：$O(1)$ 找到任意 key 的节点；
- 双向链表（带 dummy 头尾哨兵，正是 [DSA 3 §4](../../DSA/DSA_3/) 的结构）：头侧是最新，尾侧是最旧；访问即摘下接回头部；满员插入逐出尾部。

为什么必须**双向**：摘除节点需要其前驱，单链表拿前驱要 $O(n)$；哨兵头尾让"摘头/接尾"永无空判。这道题把本篇所有武器（哨兵、摘接、边界）与 [AT 17 哈希表](../AT_17_hash_table/)缝在一起，是链表专题的收官题。

---

## 5. 题目清单

| 题号 | 题名 | 难度 | 考的变形 | 链接 |
|------|------|------|---------|------|
| LC 206 | 反转链表 | Easy | 反转骨架 | [题解](../../LeetCode/LeetCode_206/) |
| LC 92 | 反转链表 II | Medium | 区间反转 | [题](https://leetcode.cn/problems/reverse-linked-list-ii/) |
| LC 25 | K 个一组翻转链表 | Hard | 分组 + 反转 | [题解](../../LeetCode/LeetCode_25/) |
| LC 141 | 环形链表 | Easy | Floyd 判圈 | [题解](../../LeetCode/LeetCode_141/) |
| LC 142 | 环形链表 II | Medium | 环入口 | [题解](../../LeetCode/LeetCode_142/) |
| LC 19 | 删除链表的倒数第 N 个结点 | Medium | 倒数第 k + dummy | [题解](../../LeetCode/LeetCode_19/) |
| LC 21 | 合并两个有序链表 | Easy | 尾插合并 | [题解](../../LeetCode/LeetCode_21/) |
| LC 23 | 合并 K 个升序链表 | Hard | 堆 / 分治合并 | [题解](../../LeetCode/LeetCode_23/) |
| LC 146 | LRU 缓存 | Medium | 双向链表 + 哈希 | [题解](../../LeetCode/LeetCode_146/) |
| LC 234 | 回文链表 | Easy | 中点 + 后半反转 | [题解](../../LeetCode/LeetCode_234/) |
| LC 160 | 相交链表 | Easy | 双指针换轨 | [题解](../../LeetCode/LeetCode_160/) |
| LC 24 | 两两交换链表中的节点 | Medium | K 组反转取 k=2 | [题解](../../LeetCode/LeetCode_24/) |

---

## 6. 与其他篇的联系

- 理论：哨兵节点、循环节视角、游标实现见 [DSA 3](../../DSA/DSA_3/)；
- 本系列：反转与快慢指针是树的遍历/判环（[AT 07](../AT_07_binary_tree/)）的思维预演；合并链表升级为堆合并（[AT 09](../AT_09_heap/) 与 LC 23）；邻接表就是"数组的链表"（[AT 11](../AT_11_graph_bfs_dfs/)）；LRU 的哈希侧见 [AT 17](../AT_17_hash_table/)；
- 既有专题文：[哈希表](../hash_table/)有 LRU 的另一实现视角。

---

*下一篇：[AT 04：栈](../AT_04_stack/)——从括号匹配到表达式求值。*
