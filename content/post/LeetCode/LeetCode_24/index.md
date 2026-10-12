---
title: "LeetCode 24: 两两交换链表中的节点"
date: 2026-10-29
math: true
tags:
  - LeetCode
  - 递归
  - 链表
categories:
  - LeetCode
description: "LeetCode 24 两两交换链表中的节点题解（迭代指针三连改 + 递归）"
---

# LeetCode 24. 两两交换链表中的节点

> 题目链接：[https://leetcode.cn/problems/swap-nodes-in-pairs/](https://leetcode.cn/problems/swap-nodes-in-pairs/)  
> 难度：中等  
> 标签：递归、链表

---

## 1. 题目描述

给你一个链表，**两两交换**其中相邻的节点，并返回交换后链表的头节点。你必须在不修改**节点内部值**的情况下完成本题（即只能进行节点交换）。

**数据范围：**

- 链表中节点的数目范围是 $[0, 100]$
- $0 \le Node.val \le 100$

**示例：**

```
输入：head = [1,2,3,4]
输出：[2,1,4,3]

输入：head = []
输出：[]

输入：head = [1]
输出：[1]
```

注意两点：一是题目**明确禁止交换值**，必须真的改 `next` 指针；二是节点数可以是 0 或 1，此时什么都不做。

---

## 2. 思路分析

### 2.1 伪解法：交换值

如果只看输出，`swap(head->val, head->next->val)` 两行就能出一组答案。但题目 follow-up 明确禁止——而且这个口子一开，题目的训练价值就没了：本题练的正是**指针操作的顺序感**，即"改一条 `next` 之前，先想清楚谁还指着谁"。面试中直接交换值基本等于宣告放弃。

### 2.2 迭代：哑节点 + 三条赋值

要交换的一对节点记作 `a`（前）和 `b`（后），它们前方的节点记作 `prev`（第一对时不存在，用哑节点顶上）。交换前后的接线关系：

| 指针 | 交换前 | 交换后 |
|------|--------|--------|
| `prev->next` | a | b |
| `a->next` | b 后面的节点（记作 `rest`） | b |
| `b->next` | rest | a |

三条 `next` 都要改，写成代码就是三行赋值：

```cpp
a->next = b->next;   // a 先抓住后段，防止链断
b->next = a;         // b 反指 a
prev->next = b;      // 前段接到新头 b
```

**顺序是第一位的**。`b->next` 原本指着 `rest`，这是找到后段的唯一线索；如果先执行 `b->next = a`，`rest` 就永远失联了。所以"先让 a 把后段接住，再让 b 反指"是唯一安全的顺序。

### 2.3 用示例走一遍

`head = 1 -> 2 -> 3 -> 4`，哑节点 `d`：

| 轮次 | prev | a, b | 三连改之后 | 链表状态 |
|------|------|------|-----------|----------|
| 1 | d | 1, 2 | `1->3`, `2->1`, `d->2` | d -> 2 -> 1 -> 3 -> 4 |
| 2 | 1 | 3, 4 | `3->null`, `4->3`, `1->4` | d -> 2 -> 1 -> 4 -> 3 -> null |

每轮结束后，**a 就是下一对的前驱**（它在链表中的新位置恰好垫在下一对前面），`prev = a` 后进入下一轮。循环条件 `prev->next && prev->next->next` 自动处理尾部：若剩 0 个节点，第一项为假；若剩 1 个落单节点，第二项为假——**落单的最后那个节点原样保留**，正是题目要求。

### 2.4 递归视角

两两交换是天然的递归 (recursion) 结构：前两个节点内部怎么交换是"本层的事"，后面的链表怎么交换是"下一层的事"。

```cpp
ListNode* newHead = head->next;             // 交换后的新头 = 原第二
head->next = swapPairs(newHead->next);      // 原头变组尾，接住子问题答案
newHead->next = head;                       // 新头反接原头
return newHead;
```

出口：`head` 为空或 `head->next` 为空（不足一对），直接返回 `head`。递归深度 $\lfloor n/2 \rfloor$，本题 $n \le 100$ 安全；节点上万时用迭代版。

---

## 3. 代码实现

```cpp
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode dummy(0, head);            // 哑节点：第一对交换也无特判
        ListNode* prev = &dummy;

        while (prev->next && prev->next->next) {  // 凑得够一对才动手
            ListNode* a = prev->next;       // 待交换对的前节点
            ListNode* b = a->next;          // 待交换对的后节点

            a->next = b->next;              // 1) a 先抓住后段，防断链
            b->next = a;                    // 2) b 反指 a
            prev->next = b;                 // 3) 前段接到新头 b

            prev = a;                       // a 是下一对的前驱
        }
        return dummy.next;
    }
};
```

递归版逻辑见 2.4 节的伪代码，完整可提交版本放在第 6 节与模板一起给出。

---

## 4. 复杂度分析

- **时间复杂度**：$O(n)$。迭代版每轮处理 2 个节点，共 $\lfloor n/2 \rfloor$ 轮；递归版每个节点被访问一次。
- **空间复杂度**：迭代版 $O(1)$——只有哑节点和几个指针；递归版 $O(n)$ 调用栈（准确说是 $O(\lfloor n/2 \rfloor)$ 层）。

---

## 5. 常见疑问

**Q1：三条赋值语句的顺序到底哪些能换、哪些不能？**

唯一**不能换**的是 1) 和 2)：`a->next = b->next` 必须在 `b->next = a` 之前执行，否则 `b->next` 里保存的"后段入口"信息被覆盖，后面的节点全部失联（这是最典型的 WA：输出只剩交换好的前几个节点，后面全丢）。1) 与 3) 之间、2) 与 3) 之间可以互换——`prev->next = b` 只是把前段接过来，不依赖另外两条的先后。总结成一句：**覆盖谁的 `next` 之前，先把 `next` 里的信息用掉**。

**Q2：奇数个节点时最后一个怎么办？循环条件为什么是两个判断？**

题目要求落单的尾节点**保持原位**。`prev->next` 为空说明剩余 0 个节点；`prev->next->next` 为空说明只剩 1 个——两种情况都不够一对，循环直接不进，落单者自然保留。很多 WA 来自只写了一个判断（如只判 `prev->next`），在偶数位置的空指针上崩溃。

**Q3：递归版为什么 `head->next = swapPairs(newHead->next)` 要先于 `newHead->next = head`？**

递归消耗掉 `newHead->next`（即第三开始的子链表）这个信息，必须先把它的返回值挂到 `head->next` 上，之后再让 `newHead->next = head`。若反过来，`newHead->next` 先被改成 `head`，通往子链表的入口就没了——和迭代版里"先抓住后段"是同一条铁律：**用信息在先，覆盖信息在后**。

---

## 6. 相关题目与模板

**可复用模板**（哑节点 + prev 推进的"局部改链"骨架，链表原地操作通用）：

```cpp
// prev 永远指向"已处理完成的尾"，对 [prev->next, ...] 做任意局部重排
ListNode dummy(0, head), *prev = &dummy;
while (凑得够本组 && 还有得处理) {
    // ... 对 prev->next 开始的一段做指针手术 ...
    prev = 本组处理完后的新尾;
}
return dummy.next;
```

这个骨架的要点：**哑节点保证从头开始的第一组不需要特判；prev 推进保证组与组之间的接线永不丢失**。

**递归版完整实现**（与迭代版等价，节点数小时可读性更佳）：

```cpp
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if (!head || !head->next) return head;      // 不足一对，原样返回
        ListNode* newHead = head->next;             // 交换后的组头 = 原第二
        head->next = swapPairs(newHead->next);      // 原头变组尾，接住子问题答案
        newHead->next = head;                       // 新头反接原头
        return newHead;
    }
};
```

**方法论总结**：链表指针手术只有一条核心纪律——改写任何 `next` 之前，确认它指向的信息已经被保存或使用。画图时把每个指针画成箭头、每条赋值在图上擦一个画一个，三行以内的问题就永远不会错。

**同类题目：**

- [LeetCode 25. K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/)：本题的 $k$ 推广，本题是 $k = 2$ 的特例（外加"不足 k 保持原序"的边界）
- [LeetCode 206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/)：最基础的三指针改链训练
- [LeetCode 92. 反转链表 II](https://leetcode.cn/problems/reverse-linked-list-ii/)：区间翻转，同样是哑节点 + prev 骨架
- [LeetCode 61. 旋转链表](https://leetcode.cn/problems/rotate-list/)：成段重排的另一种形态
- [LeetCode 21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)：哑节点思想的入门应用

**一句话总结**：两两交换 = 哑节点兜住第一对 + 三条赋值搞定每一对，而三条赋值的灵魂只有一句——先抓住后段，再反指，最后接线。
