---
title: "AT 06: 队列与单调队列——循环队列、双栈当队与 getMax 扩展"
date: 2026-11-29
math: true
tags:
  - Notes
  - 算法模板
  - 队列
  - 单调队列
categories:
  - 算法模板
description: "队列的刷题模板全集：手写循环队列（空一格判满的约定）、双栈当队（摊还 O(1) 的记账法直觉）、Steap/Queap 把 getMax 做进栈和队列（平行最大值栈模式）、约瑟夫问题与滑动窗口最值；样题 LeetCode 232/622/933 与洛谷 P1540 机器翻译、P1996 约瑟夫问题，附单调队列优化 DP 的指路图"
---

# AT 06：队列与单调队列——循环队列、双栈当队与 getMax 扩展

> **核心思想一句话：队列是"先来先服务"的公平契约——凡是问题里出现时间上的等待、轮流、过期淘汰（BFS 的逐层扩散、滑动窗口的出界、缓存的 FIFO 逐出），就该想到队列。**
> 理论后盾：[DSA 4：栈与队列](../../DSA/DSA_4/) §12–15（队列接口、双栈当队的摊还分析、Steap 与 Queap）。本篇不重复推导，只给武器。

---

## 1. 这个模板解决什么问题

刷题里队列扮演三个角色，认出角色就知道该套哪个模板：

| 角色 | 特征 | 典型题 | 模板 |
|------|------|--------|------|
| **轮次推进器** | 元素处理一次后重新排队，循环往复 | 约瑟夫问题、Round-Robin 调度 | `queue` + 取出再放回 |
| **层次/时间扩散器** | 按到达顺序逐层处理 | BFS、最近的请求次数 | `queue`，先进先出天然保序 |
| **滑动淘汰器** | 窗口左端不断出界，需要淘汰+取最值 | 滑动窗口最值 | 单调队列（见 [§2.4](#24-单调队列模板骨架)） |

而**设计题**（implement X using Y）考察的是队列的实现层：

| 实现需求 | 关键机制 | 摊还复杂度 |
|---------|---------|-----------|
| 数组实现循环队列 | 取模回绕 + 空一格判满 | 每次 $O(1)$ |
| 两个栈实现队列 | `in` 栈倒进 `out` 栈 | 摊还 $O(1)$ |
| 队列 + `getMax()` | 平行最大值栈（Queap） | 摊还 $O(1)$ |

---

## 2. 代码模板

### 2.1 手写循环队列：空一格判满

数组做队列，队头只出不进、队尾只进不出，下标只增不减——数组前部很快浪费。解法是**把数组首尾相接当成环**：下标到头就取模回绕。

取模回绕带来一个新问题：`head == tail` 既可能是空也可能是满。工业上有两种约定：

| 约定 | 判满条件 | 代价 |
|------|---------|------|
| **空一格**（本模板） | `(tail + 1) % cap == head` | 有效容量 $k-1$；开 $k+1$ 的数组存 $k$ 个元素 |
| 计数器 | `cnt == cap` | 多维护一个变量，判空判满都要看它 |

竞赛和面试都推荐第一种——少一个状态量，出 bug 的面就小一分：

```cpp
struct CirQueue {
    vector<int> q;
    int head = 0, tail = 0, cap;          // [head, tail) 为有效区间（环上）
    CirQueue(int k) : q(k + 1), cap(k + 1) {}
    bool empty() const { return head == tail; }
    bool full()  const { return (tail + 1) % cap == head; }   // 牺牲一格
    void push(int e) { q[tail] = e; tail = (tail + 1) % cap; }
    int  pop()      { int e = q[head]; head = (head + 1) % cap; return e; }
    int  front() const { return q[head]; }
    int  rear()  const { return q[(tail - 1 + cap) % cap]; }  // 队尾在 tail 的前一格
};
```

两个高频坑：

1. **`rear()` 是 `tail - 1` 不是 `tail`**——`tail` 指向"下一个写入位置"，队尾元素在它前一格；取模前先 `+cap` 防负数（`(tail - 1 + cap) % cap`）。
2. **入队前判满、出队前判空**的调用责任在使用方；LC 622 要求返回 `false`/`-1`，洛谷模拟题则通常自己保证不越界。

<details><summary>为什么不用 <code>deque</code> 一了百了？</summary>

`deque` 双端 $O(1)$，确实覆盖循环队列的全部功能，刷题首选。但两点让手写循环队列仍是必修：(1) LC 622/641 这类设计题**明着考你**判满约定；(2) 定长数组版缓存友好、无指针开销，在卡常的模拟题里有实际优势。原理层面，它也是理解"环形缓冲区"（操作系统、网络包收发都靠它）的最小标本。

</details>

### 2.2 双栈当队：摊还 O(1) 的经典

[DSA 4 §14](../../DSA/DSA_4/) 的招牌例子：两个栈 `in` 和 `out`，入队压 `in`；出队时若 `out` 空，把 `in` 整体倒过去再弹。

```cpp
struct MyQueue {
    stack<int> in_, out_;
    void transfer() {                       // 倒搬：每元素一生只经历一次
        while (!in_.empty()) { out_.push(in_.top()); in_.pop(); }
    }
    void push(int e) { in_.push(e); }
    int  pop()  { if (out_.empty()) transfer(); int e = out_.top(); out_.pop(); return e; }
    int  front(){ if (out_.empty()) transfer(); return out_.top(); }
    bool empty() const { return in_.empty() && out_.empty(); }
};
```

**为什么 FIFO 成立**：元素进 `in` 是一次逆序，栈的倒出又是一次逆序——负负得正，顺序复原。

**为什么摊还 $O(1)$**：单次 `pop` 最坏 $O(n)$（大搬运），但记账法给每个新元素 4 枚硬币——1 枚付入队、2 枚付未来的搬运、1 枚付出队，全部生命周期费用预付完毕。$n$ 个元素总代价 $O(n)$，摊还每次 $O(1)$。

一个实战细节：**别在 `push` 时主动倒搬**（"顺手把 `in` 清空"）。倒搬必须在 `out` 空且被要求出队时才发生，提前倒会把"一次大搬运"碎片化成多次，还容易破坏顺序假设。

### 2.3 Steap 与 Queap：把 getMax 做进去

[DSA 4 §15](../../DSA/DSA_4/) 把支持 `getMax()` 的栈叫 **Steap**（Stack + Heap），队列版叫 **Queap**。思想完全一致——**平行最大值结构**：

```cpp
struct Steap {
    stack<int> s, p;                     // p 与 s 平行：p.top() = s 全栈最大值
    void push(int e) { s.push(e); p.push(p.empty() ? e : max(e, p.top())); }
    void pop()  { s.pop(); p.pop(); }
    int  top()  { return s.top(); }
    int  getMax() const { return p.top(); }
};
```

`p` 的每个槽位存"从栈底到这一格为止的最大值"——**前缀最大值的栈版**。出栈时两栈同步弹，最大值信息自动退化到正确状态，三个操作全是 $O(1)$。

Queap = 双栈当队 + 平行最大值栈，`getMax()` 取两个平行栈顶的较大者：

```cpp
struct Queap {
    stack<int> in_, out_;                // 数据双栈（同 2.2）
    stack<int> gin_, gout_;              // 平行最大值栈
    void transfer() {
        while (!in_.empty()) {
            int e = in_.top(); in_.pop(); gin_.pop();
            out_.push(e);
            gout_.push(gout_.empty() ? e : max(e, gout_.top()));
        }
    }
    void push(int e) { in_.push(e); gin_.push(gin_.empty() ? e : max(e, gin_.top())); }
    void pop()  { if (out_.empty()) transfer(); out_.pop(); gout_.pop(); }
    int  front(){ if (out_.empty()) transfer(); return out_.top(); }
    int  getMax() const {                // 两栈的 max 之较大者 = 全队 max
        int m = INT_MIN;
        if (!gin_.empty())  m = gin_.top();
        if (!gout_.empty()) m = max(m, gout_.top());
        return m;
    }
};
```

全队元素分成 `in_`、`out_` 两段，全队最大值 = 两段各自最大值的较大者——这就是 `getMax()` 取 `max(gin_.top(), gout_.top())` 的理由。**LC 155 最小栈**就是这个模式（min 版本，单栈即可，因为栈只有一端）。

### 2.4 单调队列模板骨架

滑动窗口最值与单调队列优化 DP 的完整推导在[单调队列专题](../monotonic_queue/)（含逐步可视化与"早来的弱者判死刑"的直觉），此处只留骨架供速查：

```cpp
deque<int> dq;                          // 存下标；求 max：值从队头到队尾递减
for (int i = 0; i < n; i++) {
    while (!dq.empty() && a[dq.back()] <= a[i]) dq.pop_back();  // 弹尾
    dq.push_back(i);
    if (dq.front() <= i - k) dq.pop_front();                    // 队头过期
    if (i >= k - 1) ans.push_back(a[dq.front()]);               // 队头即答案
}
```

四步顺序固定：**弹尾 → 入队 → 过期检查 → 取答案**。DP 优化版（转移取滑动区间最值，$O(nk) \to O(n)$）与单调栈的对照表见[该篇 §3–4](../monotonic_queue/)。

---

## 3. 常见变形与坑

| 变形 | 一句话做法 |
|------|-----------|
| 循环双端队列（LC 641） | 循环队列加 `pushFront / popBack`，取模方向对称扩展 |
| 两个队列实现栈（LC 225） | 入栈倒腾：把新元素放队首——把前面的全部出队再入队，`O(n)` 每次 |
| 数据流移动平均（LC 面试题） | 定长队列 + 维护窗口和：入队加、出队减，`O(1)` 每次 |
| 滑动窗口中位数 | 双 `multiset` 或对顶堆（[AT 09](../AT_09_heap/)），单调队列做不了（要的是第 k 大不是最值） |
| 带过期时间的优先队列（缓存逐出） | 队列记时间戳 + 惰性清理，与堆的懒删除同族（[AT 09](../AT_09_heap/)） |

高频坑清单：

1. **循环队列判满写反**：`(tail + 1) % cap == head` 是满、`head == tail` 是空——两个条件都用模运算，别漏 `% cap`；
2. **`rear()` 忘了 `-1`**：`tail` 指向下一个写入位，队尾元素在 `(tail - 1 + cap) % cap`；
3. **双栈当队提前倒搬**：倒搬时机只能是"`out` 空且要出队"，`push` 里倒搬会破坏摊还分析且引入顺序 bug；
4. **Queap 忘了弹平行栈**：`pop()` 只弹数据栈不弹 `g*` 栈，`getMax()` 立刻错——平行结构必须同步；
5. **单调队列取答案前不过期检查**：上一轮的队头可能已出界（[专题](../monotonic_queue/) §1.2 的 `i = 4` 教训）；
6. **约瑟夫问题模拟到空队列还在出队**：循环条件是 `!q.empty()`，出队计数 `k == m` 才输出，否则放回——`k` 归零别忘。

---

## 4. 样题精讲

### 4.1 洛谷 P1540 [NOIP 2010 提高组] 机器翻译（队列 = FIFO 缓存）

> 题目链接：[P1540 机器翻译](https://www.luogu.com.cn/problem/P1540)。内存容量 $M$ 个单元，逐个查询 $N$ 个单词；不在内存则查词典（计数 +1）并入内存，满了逐出**最早进入**的。

「逐出最早进入」就是 FIFO 逐出——队列的本体定义。用 `unordered_set` 补上 $O(1)$ 查重：

```cpp
#include <cstdio>
#include <deque>
#include <unordered_set>
using namespace std;

int main() {
    int m, n; scanf("%d %d", &m, &n);
    deque<int> mem;                      // 内存：按进入顺序排队，队头最老
    unordered_set<int> in;               // O(1) 查重
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        int w; scanf("%d", &w);
        if (in.count(w)) continue;       // 命中，不查词典
        if ((int)mem.size() == m) {      // 满了：逐出最早进入的（队头）
            in.erase(mem.front());
            mem.pop_front();
        }
        mem.push_back(w); in.insert(w); cnt++;
    }
    printf("%d\n", cnt);
    return 0;
}
```

复杂度 $O(n)$。样例验证：$M=3$，查询 `1 2 1 5 4 4 1` → 查词典 5 次（`1,2,5,4` 各一次？不——`1,2` 进内存后命中，`5` `4` 触发两次逐出，共 5 次），输出 5 ✓。

**与 LRU 的分界**：本题逐出"最早**进入**"的（FIFO），若改成逐出"最久**未使用**"的（访问也刷新年龄），就是 LRU——需要哈希表 + 双向链表，见 [AT 03 链表篇](../AT_03_linked_list/)的 LC 146 精讲与[题解](../../LeetCode/LeetCode_146/)。

### 4.2 洛谷 P1996 约瑟夫问题（队列 = 轮次推进器）

> 题目链接：[P1996 约瑟夫问题](https://www.luogu.com.cn/problem/P1996)。$n$ 人围圈报数，数到 $m$ 出列，从下一人继续，求出列顺序。

队列模拟环形结构的套路：**取出来不数死就放回队尾**——"放回"等价于"绕圈"：

```cpp
#include <cstdio>
#include <queue>
using namespace std;

int main() {
    int n, m; scanf("%d %d", &n, &m);
    queue<int> q;
    for (int i = 1; i <= n; i++) q.push(i);
    int k = 0;
    while (!q.empty()) {
        int x = q.front(); q.pop();      // 队首获得报数机会
        if (++k == m) { printf("%d ", x); k = 0; }   // 数到 m：出列不回头
        else q.push(x);                  // 没数到：排到队尾（绕圈）
    }
    puts("");
    return 0;
}
```

复杂度 $O(nm)$。样例 $n=10, m=3$ 输出 `3 6 9 2 7 1 8 5 10 4` ✓。数据范围大时（$n \le 10^6$ 级）要用递推 $f(i) = (f(i-1)+m) \bmod i$，那是数学题不是队列题了。

### 4.3 LeetCode 622 设计循环队列（手写模板的直接考察）

> 题目链接：[LC 622. 设计循环队列](https://leetcode.cn/problems/design-circular-queue/)。实现 `MyCircularQueue(k)`：`enQueue / deQueue / Front / Rear / isEmpty / isFull`。

就是 [§2.1](#21-手写循环队列空一格判满) 的模板加 LC 的返回值约定（失败返回 `false` / `-1`）：

```cpp
class MyCircularQueue {
    vector<int> q; int head, tail, cap;
public:
    MyCircularQueue(int k) : q(k + 1), head(0), tail(0), cap(k + 1) {}
    bool enQueue(int v) { if (isFull()) return false; q[tail] = v; tail = (tail + 1) % cap; return true; }
    bool deQueue() { if (isEmpty()) return false; head = (head + 1) % cap; return true; }
    int Front() { return isEmpty() ? -1 : q[head]; }
    int Rear()  { return isEmpty() ? -1 : q[(tail - 1 + cap) % cap]; }
    bool isEmpty() const { return head == tail; }
    bool isFull()  const { return (tail + 1) % cap == head; }
};
```

自测样例：$k=3$ 依次 `enQueue(1,2,3)` → `enQueue(4)` 返回 `false`（满）、`Rear()=3`、`Front()=1`、`deQueue()` 后 `enQueue(4)` 成功且 `Front()=2` ✓。坑集中在 `Rear()` 的 `tail - 1` 与判满的 `(tail + 1) % cap`——见 [§3](#3-常见变形与坑) 坑 1、2。

### 4.4 LeetCode 933 最近的请求次数（队列 = 时间窗口过期）

> 题目链接：[LC 933. 最近的请求次数](https://leetcode.cn/problems/number-of-recent-calls/)。每次 `ping(t)`（严格递增）返回 $[t-3000, t]$ 内的请求数。

时间戳单调递增 → 过期元素从队头按序淘汰 → 队头弹出即可，标准的时间窗骨架：

```cpp
class RecentCounter {
    queue<int> q;
public:
    int ping(int t) {
        q.push(t);
        while (q.front() < t - 3000) q.pop();   // 队头过期
        return q.size();
    }
};
```

复杂度：每个时间戳最多进出队各一次，摊还 $O(1)$。自测：`ping(1)=1, ping(100)=2, ping(3001)=3, ping(3002)=3`（`t=1` 过期）✓。这个骨架（push → while 弹过期头 → 统计）是所有"定长时间窗计数"题的通用形态。

### 4.5 快速对照：LeetCode 232 用栈实现队列

[LC 232. 用栈实现队列](https://leetcode.cn/problems/implement-queue-using-stacks/) 就是 [§2.2](#22-双栈当队摊还-o1-的经典) 的 `MyQueue` 原样提交（`pop` / `peek` / `empty` 接口）。自测：`push(1), push(2)` → `peek()=1`、`pop()=1`、`empty()=false`、再 `push(3)` → `pop()=2, pop()=3` ✓。反过来用队列实现栈（LC 225）见 [§3](#3-常见变形与坑) 变形表——方向不同复杂度天差地别（$O(1)$ vs $O(n)$），面试常拿来追问"为什么不对称"。

---

## 5. 题目清单

| 题号 | 题名 | 难度 | 考的变形 | 链接 |
|------|------|------|---------|------|
| LC 232 | 用栈实现队列 | 简单 | 双栈当队 | [力扣](https://leetcode.cn/problems/implement-queue-using-stacks/) |
| LC 225 | 用队列实现栈 | 简单 | 倒腾法入栈 | [力扣](https://leetcode.cn/problems/implement-stack-using-queues/) |
| LC 622 | 设计循环队列 | 中等 | 空一格判满 | [力扣](https://leetcode.cn/problems/design-circular-queue/) |
| LC 641 | 设计循环双端队列 | 中等 | 循环队列 + 双端 | [力扣](https://leetcode.cn/problems/design-circular-deque/) |
| LC 933 | 最近的请求次数 | 简单 | 时间窗过期 | [力扣](https://leetcode.cn/problems/number-of-recent-calls/) |
| 面试题 346 | 数据流中的移动平均值 | 简单 | 定长窗 + 增量和 | [力扣](https://leetcode.cn/problems/moving-average-from-data-stream/) |
| LC 950 | 按递增顺序显示卡牌 | 中等 | 反向倒腾模拟 | [力扣](https://leetcode.cn/problems/reveal-cards-in-increasing-order/) |
| 洛谷 P1540 | [NOIP 2010 提高组] 机器翻译 | 普及/提高− | FIFO 缓存逐出 | [洛谷](https://www.luogu.com.cn/problem/P1540) |
| 洛谷 P1996 | 约瑟夫问题 | 普及− | 轮次推进 | [洛谷](https://www.luogu.com.cn/problem/P1996) |
| 洛谷 P2058 | [NOIP 2016 普及组] 海港 | 普及+ | 时间窗去重计数 | [洛谷](https://www.luogu.com.cn/problem/P2058) |
| 洛谷 P1886 | 【模板】单调队列 / 滑动窗口 | 普及+ | 窗口最值双练 | [洛谷](https://www.luogu.com.cn/problem/P1886) |
| LC 239 | 滑动窗口最大值 | 困难 | 单调队列原型 | [题解](../../LeetCode/LeetCode_239/) |

进阶（用到本文结构但不以队列为主）：LC 1823 约瑟夫递推版、LC 649 Dota2 参议院（双队列对打）、P1160 队列安排（双端链表，见 [AT 03](../AT_03_linked_list/)）。

---

## 6. 与其他篇的联系

| 方向 | 连接点 |
|------|--------|
| [DSA 4](../../DSA/DSA_4/) §12–15 | 理论后盾：队列 ADT、双栈当队的三种摊还证明（记账/聚合/势能）、Steap 与 Queap 的原始定义 |
| [单调队列专题](../monotonic_queue/) | 滑动窗口最值与 DP 优化的完整推导、逐步可视化、"早来的弱者判死刑"直觉——本篇 §2.4 只是骨架速查 |
| [AT 04 栈](../AT_04_stack/) | 对称篇：栈的两种角色与队列的三个角色对照着记；双栈当队/两队列当栈互为镜像题 |
| [AT 05 单调栈](../AT_05_monotonic_stack/) | 单调队列的"没有过期概念"版本，两者淘汰机制对照表见专题 §4 |
| [AT 03 链表](../AT_03_linked_list/) | FIFO 逐出（P1540）升级成 LRU 逐出（LC 146）就要双向链表 + 哈希表 |
| [AT 09 堆](../AT_09_heap/) | 窗口要"第 k 大/中位数"时单调队列失效，改对顶堆/懒删除堆 |
| [AT 11 图的遍历](../AT_11_graph_bfs_dfs/) | BFS 就是队列做"层次扩散器"的最大舞台 |

---

*下一篇：[AT 07：二叉树——遍历与构造](../AT_07_binary_tree/)，从递归三段式到前中序重建。*
