---
title: "LeetCode 19: 删除链表的倒数第 N 个结点"
date: 2026-10-28
math: true
tags:
  - LeetCode
  - 链表
  - 双指针
categories:
  - LeetCode
description: "LeetCode 19 删除链表的倒数第 N 个结点题解（快慢双指针一趟扫描）"
---

# LeetCode 19. 删除链表的倒数第 N 个结点

> 题目链接：[https://leetcode.cn/problems/remove-nth-node-from-end-of-list/](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)  
> 难度：中等  
> 标签：链表、双指针

---

## 1. 题目描述

给你一个链表，删除链表的**倒数第 $n$ 个结点**，并且返回链表的头结点。

题目进阶要求：**能否尝试使用一趟扫描实现？**

**数据范围：**

- 链表中结点的数目为 $sz$，$1 \le sz \le 30$
- $0 \le Node.val \le 100$
- $1 \le n \le sz$

**示例：**

```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]

输入：head = [1], n = 1
输出：[]

输入：head = [1,2], n = 1
输出：[1]
```

第二个示例是关键边界：被删的正是**头结点**，删除后链表为空。

---

## 2. 思路分析

### 2.1 暴力：先数长度，再定位

"倒数第 $n$ 个"不直观，先翻译成"正数第几个"。设链表长度为 $L$，则：

| 目标 | 正数位置 |
|------|---------|
| 倒数第 $n$ 个结点（被删者） | 第 $L - n + 1$ 个 |
| 它的前驱（真正要动手的地方） | 第 $L - n$ 个 |

于是暴力流程：第一趟扫描数出 $L$；第二趟从头走 $L - n$ 步停在前驱处，执行 `prev->next = prev->next->next`。时间 $O(L)$，完全正确，但**扫描了两趟**，且不满足进阶要求。

### 2.2 优化动机：单链表不能回头，但可以让"到终点的距离"替我们倒数

单链表 (singly linked list) 的麻烦在于没有 `prev`、不能倒着走，所以必须先知道 $L$ 才能换算正数位置。但换个角度想：**如果两个指针之间保持固定的间隔 $n$，那么当前面的指针到达终点时，后面的指针离终点恰好还有 $n$ 步**——它停的位置正是"倒数第 $n$ 个的前一个"。

这就是**快慢双指针 (fast & slow pointers)**：

1. `fast` 从哑节点出发先走 $n$ 步，与 `slow` 拉开 $n$ 的间隔；
2. 两指针**同速**后移，直到 `fast` 到达最后一个结点；
3. 此时 `slow` 恰好停在倒数第 $n$ 个结点的**前驱**，跨过 `slow->next` 即完成删除。

一趟扫描，长度 $L$ 无需显式求出——间隔 $n$ 这个不变量替我们完成了"倒数→正数"的换算。

### 2.3 用示例走一遍

`head = 1 -> 2 -> 3 -> 4 -> 5`，$n = 2$。哑节点记作 `d`，指针位置用括号标出：

| 阶段 | fast | slow | 说明 |
|------|------|------|------|
| 初始 | d | d | 都从哑节点出发 |
| fast 先走 2 步 | 2 | d | 间隔拉开为 2 |
| 同移 1 步 | 3 | 1 | |
| 同移 1 步 | 4 | 2 | |
| 同移 1 步 | 5 | **3** | fast 到尾结点，停！ |

`fast = 5` 是最后一个结点，`slow = 3` 正好是 `4`（倒数第 2 个）的前驱。执行 `slow->next = 4 -> 5`，结果 `1 -> 2 -> 3 -> 5`。

再验证删除头结点的情形 `head = [1], n = 1`：`fast` 从哑节点走 1 步到 `1`，`while (fast->next)` 立即不成立，`slow` 仍停在哑节点，删除的是 `dummy->next` 即头结点——**哑节点在这里不是锦上添花，而是必需品**（见 5.1）。

### 2.4 另一种等价写法（差一陷阱高发区）

常见写法还有"`fast` 先走 $n + 1$ 步、循环条件 `while (fast)`"。它与本文的"走 $n$ 步 + `while (fast->next)`"**完全等价**，只是把"fast 停在尾结点"换成了"fast 走过空指针"。两种写法都行，但千万别混搭——"走 $n + 1$ 步"配"`while (fast->next)`"会少走一步，删错人。

---

## 3. 代码实现

```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode dummy(0, head);          // 哑节点：删除头结点也无特判
        ListNode* fast = &dummy;
        ListNode* slow = &dummy;

        for (int i = 0; i < n; i++)       // 1. 拉开 n 的间隔
            fast = fast->next;

        while (fast->next) {              // 2. 同速前进，fast 停在尾结点
            fast = fast->next;
            slow = slow->next;
        }
        // 3. slow 恰在倒数第 n+1 个：跨过被删者
        ListNode* victim = slow->next;
        slow->next = slow->next->next;
        delete victim;                    // 归还内存，工程习惯
        return dummy.next;
    }
};
```

注意 `slow` 从**哑节点**而非 `head` 出发。如果从 `head` 出发，当 $n = L$（删头）时 `slow` 会停在被删结点自身上，永远凑不出"前驱"——哑节点相当于给整条链表垫了一层永远删不到的底座。

---

## 4. 复杂度分析

- **时间复杂度**：$O(L)$。`fast` 先走 $n$ 步、再与 `slow` 一起走 $L - n$ 步，总计 $L$ 步，恰好一趟扫描。
- **空间复杂度**：$O(1)$。两个指针加一个哑节点。

与暴力版对比：渐进复杂度同为 $O(L)$，但快慢指针把两趟压成一趟。本题 $L \le 30$ 体现不出差距，意义在于**模式**：链表无法随机访问、无法回退，"固定间隔的双指针"是用时间上的并行换空间上的一次通过，同族方法还有找中点（间隔不定的 1:2 快慢针）、判环（Floyd 判圈）。

---

## 5. 常见疑问

**Q1：不用哑节点行不行？**

逻辑上可以：先特判"倒数第 $n$ 个是不是头"，是就 `return head->next`。但这个特判恰好是最容易忘的分支——示例 2（`[1]`, `n = 1`）就是专门为此设的测试点。哑节点把"删头"归约成"删哑节点的下一个"，一个分支都不用写。链表题的通用军规：**凡是可能动到头结点，先挂哑节点**。

**Q2：为什么 `fast` 要先走 $n$ 步而不是 $n + 1$ 步？**

取决于你希望循环结束时谁当参照系。走 $n$ 步 + `while (fast->next)`：`fast` 停在**尾结点**，`slow` 停在倒数第 $n + 1$ 个（前驱）。走 $n + 1$ 步 + `while (fast)`：`fast` 走到**空指针**，`slow` 同样停在前驱。两者差一互补，选一种记牢即可；临场现推"到底走几步"是 WA 的最大来源，建议固定用本文写法并记住结论：**fast 多走的步数 = slow 落后终点的步数**。

**Q3：`delete victim` 必须写吗？**

LeetCode 不判内存泄漏，删掉这行也能 AC。但注意写法顺序：必须先把 `victim` 存下来再改 `slow->next`，否则改完指针就再也找不到被删结点了。`delete` 本身是工程素养问题，面试时写上并说一句"生产环境里被删结点还可能被外部引用，需要按所有权约定处理"，是明显的加分项。

---

## 6. 相关题目与模板

**可复用模板**（间隔为 $n$ 的快慢双指针，"倒数定位于一趟"骨架）：

```cpp
// 快指针先行 n 步，同速前进到终点，慢指针停在倒数第 n+1 个
ListNode dummy(0, head), *fast = &dummy, *slow = &dummy;
for (int i = 0; i < n; i++) fast = fast->next;
while (fast->next) { fast = fast->next; slow = slow->next; }
// slow->next 即倒数第 n 个，slow 即其前驱
```

**方法论总结**：链表上"位置依赖总数"的问题（倒数第 k 个、中点、交点、环），通用武器都是**双指针的不变量设计**——先问"两个指针保持什么关系，才能在到达终点时自动停在目标位置"，间隔、速度比、出发点是三个可调的旋钮。

**同类题目：**

- [LeetCode 876. 链表的中间结点](https://leetcode.cn/problems/middle-of-the-linked-list/)：间隔改速度比 1:2 的快慢指针
- [LeetCode 141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/) / [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)：Floyd 判圈，快慢指针最经典的应用
- [LeetCode 61. 旋转链表](https://leetcode.cn/problems/rotate-list/)：同样需要"倒数第 k 个"的定位，先接成环再断
- [LeetCode 83. 删除排序链表中的重复元素](https://leetcode.cn/problems/remove-duplicates-from-sorted-list/)：哑节点 + 删除操作的组合练习
- [LeetCode 160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/)：双指针走两遍的浪漫解法

**一句话总结**：单链表不能回头，但让快指针先替你走掉 $n$ 步，"倒数"就在一趟里自动换算成了"正数"——哑节点负责兜住删头的边界。
