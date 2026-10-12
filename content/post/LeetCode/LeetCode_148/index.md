---
title: "LeetCode 148: 排序链表"
date: 2026-11-01
math: true
tags:
  - LeetCode
  - 链表
  - 分治
  - 归并排序
categories:
  - LeetCode
description: "LeetCode 148 排序链表题解（链表归并排序）"
---

# LeetCode 148. 排序链表

> 题目链接：[https://leetcode.cn/problems/sort-list/](https://leetcode.cn/problems/sort-list/)  
> 难度：中等  
> 标签：链表、分治、归并排序

---

## 1. 题目描述

给你链表的头节点 `head`，请将其**按升序排列**并返回排序后的链表。

**进阶**：你可以在 $O(n \log n)$ 时间复杂度和常数级空间复杂度下，对这个链表进行排序吗？

**数据范围：**

- 链表中节点的数目在范围 $[0, 5 \times 10^4]$ 内
- $-10^5 \le Node.val \le 10^5$

**示例：**

```
输入：head = [4,2,1,3]

输出：[1,2,3,4]

过程：
  4 -> 2 -> 1 -> 3
  ├── 切成  4 -> 2     与  1 -> 3
  ├── 各自排成 2 -> 4  与  1 -> 3
  └── 归并成  1 -> 2 -> 3 -> 4
```

---

## 2. 思路分析

### 2.1 暴力：为什么链表排序这么别扭

在数组上你会直接 `sort`，但链表没有随机访问——快速排序的 `partition` 需要双端扫描、堆排序需要建堆下滤，搬到链表上都别扭。两个朴素的候选：

| 做法 | 时间 | 空间 | 结局 |
|------|------|------|------|
| 抽出所有值 → 数组 `sort` → 塞回节点 | $O(n \log n)$ | $O(n)$ | 能过，但完全没碰链表结构，面试不算数 |
| 链表插入排序（[147 题](https://leetcode.cn/problems/insertion-sort-list/)） | $O(n^2)$ | $O(1)$ | $n = 5 \times 10^4$ 时最坏约 $1.25 \times 10^9$ 次比较，超时 |

突破口在于想清楚一件事：**数组上哪个"好排序"天生不需要随机访问？**——归并排序。它的整个流程只有两种动作：**找中点切一半**、**合并两个有序段**。前者链表可以用快慢指针做，后者链表比数组做得还干净（合并时直接改指针，不用辅助数组）。

### 2.2 自顶向下：递归归并

于是得到链表版归并排序，分三步：

1. **找中点**：快慢指针，`slow` 每次走一步、`fast` 走两步，`fast` 到尾时 `slow` 在中点；
2. **从中间切断**，对左右两半递归排序；
3. **归并**两个有序子链表（即 [21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)，双指针摘最小节点接尾）。

**用 `4 -> 2 -> 1 -> 3` 走一遍**：

| 层 | 切分 | 递归排序后 | 待归并 |
|----|------|-----------|--------|
| 第 1 层 | `[4,2]` 与 `[1,3]` | `[2,4]` 与 `[1,3]` | `2->4` 和 `1->3` |
| 第 2 层 | `[4,2]` 切成 `[4]` 与 `[2]`；`[1,3]` 切成 `[1]` 与 `[3]` | 单节点天然有序 | `4` 和 `2`；`1` 和 `3` |

归并 `2->4` 与 `1->3` 的双指针过程：

```
左: 2 -> 4        右: 1 -> 3       已产出
------------------------------------------------
2 vs 1  摘 1      2 -> 4                3        1
2 vs 3  摘 2      4                     3        1 -> 2
4 vs 3  摘 3      4              (空)             1 -> 2 -> 3
右空，剩左整条接上                            1 -> 2 -> 3 -> 4
```

递归深度 $O(\log n)$（$5 \times 10^4$ 时约 16 层），完全安全。

### 2.3 进阶：把递归栈也省掉——自底向上

递归版空间是 $O(\log n)$ 的调用栈。要达到题目进阶要求的 $O(1)$，改为**自底向上 (bottom-up)**：没有长度信息也不需要——从"每段长度 1"开始，把相邻两段两两归并成有序段，段长每轮翻倍（$1 \to 2 \to 4 \to \cdots$），直到段长 $\ge n$。

```
4 -> 2 -> 1 -> 3
段长1: [4] [2] [1] [3]  →  [2,4] [1,3]
段长2: [2,4] [1,3]      →  [1,2,3,4]
段长4 ≥ n，结束
```

它和递归版做的是**同一棵归并树**，只是自顶向下靠递归"先切到底再合并"，自底向上靠循环"从叶子上来"，后者不需要栈。

---

## 3. 代码实现

```cpp
class Solution {
    // 合并两个有序链表：双指针，摘较小者接在 tail 后
    ListNode* merge(ListNode* a, ListNode* b) {
        ListNode dummy(0);
        ListNode* tail = &dummy;
        while (a && b) {
            if (a->val <= b->val) { tail->next = a; a = a->next; }
            else                  { tail->next = b; b = b->next; }
            tail = tail->next;
        }
        tail->next = a ? a : b;   // 剩下的整条直接挂上
        return dummy.next;
    }

public:
    ListNode* sortList(ListNode* head) {
        // 递归出口：空或单节点，天然有序
        if (!head || !head->next) return head;

        // 快慢指针找中点：slow 停在前一半的最后一个节点
        ListNode* slow = head;
        ListNode* fast = head->next;      // fast 从第二个出发 → 偶数长度时 slow 偏左
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }

        // 从 slow 之后切断，两半各排各的
        ListNode* right = slow->next;
        slow->next = nullptr;             // 关键：断开，否则递归无法收敛

        ListNode* leftSorted  = sortList(head);
        ListNode* rightSorted = sortList(right);
        return merge(leftSorted, rightSorted);
    }
};
```

几个实现细节：

1. **`fast` 从 `head->next` 出发**：长度为偶数时 `slow` 停在**左半最后一个**（如 `4->2->1->3` 中停在 2），切分是 `[4,2]` 与 `[1,3]`，两半都不超过 $\lceil n/2 \rceil$。若 `fast` 从 `head` 出发，`slow` 会停在右半第一个，单元素链表的切分将出现"左空右全"——配合第 2 条会死循环。
2. **`slow->next = nullptr` 必须写**：链表不像数组有天然的边界，不切断的话，"左半"仍顺着 `next` 连到右半，递归规模不减小，栈溢出。
3. **`merge` 里用 `<=`**：相等时取左边的，保证排序**稳定 (stable)**——这是归并排序的招牌性质。
4. 用栈上构造的哨兵 `ListNode dummy(0)` 而不是 `new`，避免泄漏。

---

## 4. 复杂度分析

- **时间复杂度**：$O(n \log n)$。递归树共 $O(\log n)$ 层，每层所有节点恰好参与一次切分或一次归并，单层总代价 $O(n)$；$T(n) = 2T(n/2) + O(n)$，主定理解得 $O(n \log n)$。
- **空间复杂度**：$O(\log n)$，即递归栈深度（不计输出）。若改用 2.3 节的自底向上迭代版，可降到 $O(1)$。

| 做法 | 时间 | 空间 |
|------|------|------|
| 插入排序 | $O(n^2)$ | $O(1)$ |
| 数组中转 | $O(n \log n)$ | $O(n)$ |
| 递归归并（本解） | $O(n \log n)$ | $O(\log n)$ |
| 自底向上归并 | $O(n \log n)$ | $O(1)$ |

---

## 5. 常见疑问

**Q1：为什么快慢指针的起点差一步这么要命？**

规则要记住：`fast` 从 `head->next` 出发，`slow` 停在**前半最后一个**；`fast` 从 `head` 出发，`slow` 停在**中间偏后**。本题必须切断 `slow` 之后，若 `slow` 落在链表**最后一个节点**（发生在某些起点配置下），切出的左半是整条链、右半为空，递归原样重演自己——直接爆栈。凡是"找中点 + 切两半"的题（本题、[143. 重排链表](https://leetcode.cn/problems/reorder-list/)、[234. 回文链表](https://leetcode.cn/problems/palindrome-linked-list/)）都要先想清楚 `slow` 停在哪、断在哪。

**Q2：归并时把 `tail->next = a ? a : b` 换成循环逐个接，行不行？**

功能上等价，但归并的经典收尾就是把**未耗尽的那半条链整体挂上**——它本身就是有序的，剩余部分无需再逐节点处理。逐个接不会错，只是白写循环；真正的错误方向是漏掉这一句，让尾巴整个丢失。

**Q3：既然要 $O(1)$ 空间，为什么不直接写自底向上？**

可以写，但自底向上的"数段长、切边界、处理 $\bmod$ 长度的零头段"边界条件多、易错，而递归版 15 行且思路直白。工程上的排序（如 C++ 标准库的 `std::list::sort`）确实用的是自底向上，因为它还顺便省了栈。**面试策略**：先写对递归版讲清分治，再口头描述自底向上的改法——两个版本合起来才是完整答案。

---

## 6. 相关题目与模板

**链表归并排序模板**（找中点 + 断链 + 递归 + 归并，可直接套用）：

```cpp
ListNode* sortList(ListNode* head) {
    if (!head || !head->next) return head;
    ListNode *slow = head, *fast = head->next;
    while (fast && fast->next) slow = slow->next, fast = fast->next->next;
    ListNode* right = slow->next;
    slow->next = nullptr;
    return merge(sortList(head), sortList(right));
}
```

**方法论总结**：链表题的三板斧——**快慢指针**（定位）、**断链接链**（切分与合并）、**哨兵节点 (dummy node)**（消除头部特判）。本题三板全用上，是链表综合操的最佳练习题。

**同类题目：**

- [LeetCode 21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)：本题的子过程，单独考过，必须秒写
- [LeetCode 23. 合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)：归并的"多路"版本，堆或分治两两合并
- [LeetCode 147. 对链表进行插入排序](https://leetcode.cn/problems/insertion-sort-list/)：$O(n^2)$ 的对照组，体会链表上"移动元素"有多贵
- [LeetCode 876. 链表的中间结点](https://leetcode.cn/problems/middle-of-the-linked-list/)：快慢指针找中点的单项训练

**一句话总结**：链表没有随机访问，快排和堆排施展不开，而归并排序只需要"切一半、合两段"——这两件事链表反而做得比数组更优雅。
