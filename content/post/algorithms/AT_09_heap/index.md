---
title: "AT 09: 堆与优先队列——只维护最值，不维护全序"
date: 2026-12-02
math: true
tags:
  - Notes
  - 算法模板
  - 堆
  - 优先队列
categories:
  - 算法模板
description: "堆模板全家桶：priority_queue 比较器陷阱、手写二叉堆（上滤/下滤/Floyd O(n) 建堆）、TopK 两种解法、对顶堆求中位数、懒删除堆。样题：洛谷 P3378 堆模板、P1090 合并果子、LC 295 数据流的中位数"
---

# AT 09：堆与优先队列——只维护最值，不维护全序

> 本篇是 [DSA 刷题模板系列](../AT_00_reading_guide/)第 09 篇。核心思想一句话：**堆只维护"当前最值在哪"，不维护全局有序——用放弃全序换取插入/取最值都 $O(\log n)$。**
> 理论后盾：[DSA 12：优先级队列](../../DSA/DSA_12/)——为什么偏序就够（§3）、上滤/下滤的完整推导（§5–6）、Floyd 建堆为什么是 $O(n)$（§7）。本篇只留"怎么写"，推导去那边看。

---

## 1. 这个模板解决什么问题

凡是"**动态集合里反复取最值**"，就轮到堆出场：

| 场景 | 形式 | 本篇模板 |
|------|------|---------|
| 合并果子式贪心 | 每次取最小两个，处理后放回 | `priority_queue` |
| TopK（第 K 大/前 K 个） | 只要前 K 个，不要全排序 | K 小顶堆 / 快速选择 |
| 数据流中位数 | 前后两半各自维护最值 | 对顶堆 |
| 优先级调度 / K 路归并 | 按权值逐个出队 | `priority_queue`（[LC 23](../../LeetCode/LeetCode_23/)） |

关键认知（DSA 12 §3 的结论）：排序是全序 $O(n\log n)$，堆是偏序——如果你**只消费最值、从不中序遍历**，堆就比有序结构便宜。反过来，如果你需要"第 2 小、第 3 小一起拿"，堆要弹 $k$ 次 $O(k\log n)$；要全序就直接排序，别用堆模拟。

---

## 2. 代码模板

### 2.1 `priority_queue` 速查（刷题默认起手）

```cpp
priority_queue<int> pq;                                    // 大根堆（默认）
priority_queue<int, vector<int>, greater<int>> pq;          // 小根堆：greater 别忘 vector
priority_queue<pair<int,int>> pq;                           // pair 按字典序，先比 first

// 自定义结构体：重载 operator<（注意 pq 取反：< 定义的是"大"的方向）
struct Node { int w, id; };
struct Cmp { bool operator()(const Node& a, const Node& b) const { return a.w > b.w; } };
priority_queue<Node, vector<Node>, Cmp> pq;                 // w 最小者在顶
```

三个高频坑：

1. **小根堆要写全三个模板参数**，漏掉 `vector<int>` 直接编译错误；
2. **比较器语义是"返回 true 的排后面"**——与 `sort` 相反的直觉来源是堆的方向，写反了就是求反问题，WA 得毫无症状。自查口诀：**比较器返回 true = 你不如别人 = 你沉底**；
3. **lambda 比较器要 `decltype`**：`auto cmp = [](auto& a, auto& b){ return a.w > b.w; }; priority_queue<Node, vector<Node>, decltype(cmp)> pq(cmp);`——构造时必须把 `cmp` 传进去。

### 2.2 手写二叉堆（需要 $O(n)$ 建堆或可检视内部时）

竞赛里 `priority_queue` 够用 95% 的题；手写版的价值在 **Floyd $O(n)$ 批量建堆**（`priority_queue` 只能逐个 push，$O(n\log n)$）和数组下标可定位（某些题要修改堆内元素）。1-indexed 写法：

```cpp
struct MinHeap {                    // 1-indexed 小根堆：下标 1 是堆顶
    vector<long long> h;
    MinHeap() { h.push_back(0); }   // 占住下标 0，父 = i/2，孩子 = 2i / 2i+1
    void up(int i) {                // 上滤：push 后新元素就位
        while (i > 1 && h[i] < h[i / 2]) swap(h[i], h[i / 2]), i /= 2;
    }
    void down(int i) {              // 下滤：pop 后堆顶空洞就位
        int n = (int)h.size() - 1;
        while (2 * i <= n) {
            int c = 2 * i;                          // 左孩子
            if (c + 1 <= n && h[c + 1] < h[c]) c++; // 挑更小的孩子换
            if (h[i] <= h[c]) break;                // 堆序性已满足，停
            swap(h[i], h[c]); i = c;
        }
    }
    void push(long long x) { h.push_back(x); up((int)h.size() - 1); }
    void pop() { h[1] = h.back(); h.pop_back(); if ((int)h.size() > 1) down(1); }
    bool empty() { return h.size() == 1; }
    long long top() { return h[1]; }
    void build(vector<long long> a) {               // Floyd O(n) 建堆：自底上下滤
        h = {0}; h.insert(h.end(), a.begin(), a.end());
        for (int i = (int)h.size() / 2; i >= 1; i--) down(i);
    }
};
```

不变量：任意 $i$ 有 `h[i] <= h[2i]` 且 `h[i] <= h[2i+1]`（堆序性）；`pop` 把**最后一个**元素补到堆顶再下滤——不是"顶的儿子上移"，这是手写最常写错的点（DSA 12 §6）。

Floyd 建堆为什么 $O(n)$：一半叶子不下滤，第 $k$ 层节点各下滤 $O(k)$，求和 $\sum_k \frac{n}{2^k} \cdot k = O(n)$（DSA 12 §7）。

### 2.3 对顶堆：动态中位数/第 K 大

把序列分成两半：较小一半放进**大根堆** `lo`（顶 = 左半最大），较大一半放进**小根堆** `hi`（顶 = 右半最小）。任意时刻满足：

$$
\text{lo 的所有元素} \le \text{hi 的所有元素},\qquad 0 \le |lo| - |hi| \le 1
$$

则中位数就在两个堆顶。插入用"**先进左半，左半最大移交右半，右半超员再还一个**"三步走，不变量自动保持（完整代码见 §4.3）。

### 2.4 懒删除堆（折叠）

`priority_queue` 不支持删除任意元素、也没有 decrease-key。替代方案是**删除只打标记，取顶前才清理**（均摊后每个元素仍只进出一次）：

<details><summary>懒删除堆模板</summary>

```cpp
// 场景：堆里存 (值, 编号)，外部可能随时作废某个编号
priority_queue<pair<long long,int>> pq;   // 大根堆
bool removed[MAXN];                        // removed[id] = true 表示作废

void erase(int id) { removed[id] = true; }             // O(1)，不碰堆
void clean() {                                          // 每次读 top 前调用
    while (!pq.empty() && removed[pq.top().second]) pq.pop();
}
// 使用：clean(); auto best = pq.top();  —— 永远先 clean 再读
```

要点：删除是 $O(1)$ 打标记而非真删；**每次读 `top()` 前必须 `clean()`**，忘了就读到尸体。变体：Dijkstra 里"一个节点入堆多次、出堆时发现已处理就跳过"（`vis` 数组），本质就是懒删除。
</details>

---

## 3. 常见变形与坑

| 变形 | 一句话做法 |
|------|-----------|
| 第 K 大（静态数组） | 维护 K 大小的小顶堆 $O(n\log k)$，或快速选择 $O(n)$（见 [AT 02](../AT_02_binary_search/) 二分思想） |
| 前 K 个高频元素 | 哈希计数 + K 小顶堆（[LC 347](../../LeetCode/LeetCode_347/)） |
| 合并 K 个有序序列 | K 元素小顶堆，弹出哪个就补哪个的下一个（[LC 23](../../LeetCode/LeetCode_23/)、洛谷 P1631） |
| 动态中位数 | 对顶堆（§2.3） |
| 双端都要取（又要最大又要最小） | 大小根各一个 + 懒删除同步作废 |
| 反悔贪心（后悔药） | 堆存"决策 + 反悔收益"，取堆顶判断是否反悔 |
| 每次取最小**两个**合并后放回 | 循环 `pop` 两次、`push` 一次（P1090） |

坑清单：

- **比较器方向反了**：小根堆比较器写 `<`，得到的是大根堆——上面 §2.1 的口诀背下来；
- **`pop` 前不判 `empty()`**：UB，最阴险的一类错；
- **`long long` 溢出**：合并果子类"累加答案"题，$n$ 个 $10^9$ 相加超 `int`，堆元素和答案都用 `long long`；
- **对顶堆插入顺序错**：必须"先进 lo、lo 顶移交 hi、hi 超员返还"，先判断再插会破坏 `|lo|-|hi|≤1`；
- **把堆当有序表遍历**：堆数组的中序不是有序序列（只有堆顶保证最值）。

---

## 4. 样题精讲

### 4.1 洛谷 P3378 【模板】堆——手写堆的验收

> [P3378 【模板】堆](https://www.luogu.com.cn/problem/P3378)（普及/提高-）：操作 1 x 插入，2 查询最小，3 删除最小。

模板题，直接套 §2.2 的 `MinHeap`。值得说的只有一点：**为什么讲模板题**——它是检验手写堆的正确姿势：`up`/`down` 的边界（`2*i <= n`）、`pop` 的"尾补顶"。完整可提交程序：

```cpp
#include <cstdio>
#include <vector>
using namespace std;

struct MinHeap {
    vector<long long> h;
    MinHeap() { h.push_back(0); }
    void up(int i) { while (i > 1 && h[i] < h[i/2]) swap(h[i], h[i/2]), i /= 2; }
    void down(int i) {
        int n = (int)h.size() - 1;
        while (2*i <= n) {
            int c = 2*i;
            if (c+1 <= n && h[c+1] < h[c]) c++;
            if (h[i] <= h[c]) break;
            swap(h[i], h[c]); i = c;
        }
    }
    void push(long long x) { h.push_back(x); up((int)h.size()-1); }
    void pop() { h[1] = h.back(); h.pop_back(); if ((int)h.size() > 1) down(1); }
    bool empty() { return h.size() == 1; }
    long long top() { return h[1]; }
};

int main() {
    int n; scanf("%d", &n);
    MinHeap pq;
    while (n--) {
        int op; scanf("%d", &op);
        if (op == 1) { long long x; scanf("%lld", &x); pq.push(x); }
        else if (op == 2) printf("%lld\n", pq.top());
        else pq.pop();
    }
    return 0;
}
```

每个操作均摊 $O(\log n)$。

### 4.2 洛谷 P1090 合并果子——堆贪心的原型

> [P1090 [NOIP 2004 提高组] 合并果子](https://www.luogu.com.cn/problem/P1090)（普及/提高-）：$n$ 堆果子每次合并两堆，代价为两堆之和，求把所有果子并成一堆的最小总代价。

**为什么贪心对**：每次合并的代价会"传染"——果子并入大堆 A 后，之后每次搬动 A 都要再付一次这份代价。所以**越早合并的堆，被重复计费的次数越多**，理应让小的先合并（交换论证：任意交换相邻两次合并中"更大者先合"的次序，总代价不降）。

**怎么套模板**：小根堆循环"取两个最小 → 合并 → 放回"，恰好是 §3 变形表的最后一行：

```cpp
#include <cstdio>
#include <queue>
#include <vector>
using namespace std;

int main() {
    int n; scanf("%d", &n);
    priority_queue<long long, vector<long long>, greater<long long>> pq;
    for (int i = 0; i < n; i++) { long long x; scanf("%lld", &x); pq.push(x); }
    long long ans = 0;
    while ((int)pq.size() > 1) {           // 循环 n-1 次，每次消灭一个堆
        long long a = pq.top(); pq.pop();
        long long b = pq.top(); pq.pop();
        ans += a + b;                       // 本次代价
        pq.push(a + b);                     // 新堆放回候选
    }
    printf("%lld\n", ans);
    return 0;
}
```

$O(n \log n)$。这个"每次取最小两个处理完放回"的骨架就是哈夫曼树构造过程本身，石子合并（区间 DP 版，不能套堆）是它的对照组——**两堆合并任意选 ↔ 只能相邻合并**，前者贪心后者 DP，刷题时要先分清。

### 4.3 LC 295 数据流的中位数——对顶堆

> [LC 295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/)（困难）：流式加入整数，随时查询当前中位数。仓库已有[逐行题解](../../LeetCode/LeetCode_295/)。

**怎么套模板**：§2.3 的对顶堆，三步插入法：

```cpp
class MedianFinder {
    priority_queue<int> lo;                            // 大根堆：较小一半（顶 = 左半最大）
    priority_queue<int, vector<int>, greater<int>> hi; // 小根堆：较大一半（顶 = 右半最小）
public:
    void addNum(int num) {
        lo.push(num);                     // 步骤 1：先无脑进左半
        hi.push(lo.top()); lo.pop();      // 步骤 2：左半最大移交右半 → 保证 lo ≤ hi
        if (hi.size() > lo.size()) {      // 步骤 3：右半超员一个，还一个给左半
            lo.push(hi.top()); hi.pop();
        }
    }
    double findMedian() {
        return lo.size() > hi.size() ? lo.top() : (lo.top() + hi.top()) * 0.5;
    }
};
```

不变量（§2.3 的两条）在每次 `addNum` 后自动恢复——三个步骤分别处理"新元素可能属于右半"和"两半规模差不超过 1"，无需任何分类讨论，这是对顶堆模板的优雅之处。`addNum` $O(\log n)$，`findMedian` $O(1)$。同款：洛谷 P1168（离线版中位数）、P1801（带 lazy 的动态第 K 大，黑匣子）。

---

## 5. 题目清单

| 题号 | 题名 | 难度 | 考的变形 |
|------|------|------|---------|
| [洛谷 P3378](https://www.luogu.com.cn/problem/P3378) | 【模板】堆 | 普及/提高- | 手写堆三操作 |
| [洛谷 P1090](https://www.luogu.com.cn/problem/P1090) | [NOIP 2004 提高组] 合并果子 | 普及/提高- | 堆贪心（哈夫曼） |
| [洛谷 P1168](https://www.luogu.com.cn/problem/P1168) | 中位数 | 普及/提高- | 对顶堆 |
| [洛谷 P1631](https://www.luogu.com.cn/problem/P1631) | 序列合并 | 普及+/提高 | K 路归并 + 堆 |
| [LC 23](../../LeetCode/LeetCode_23/) | 合并 K 个升序链表 | 困难 | K 路归并小根堆 |
| [LC 215](../../LeetCode/LeetCode_215/) | 数组中的第K个最大元素 | 中等 | TopK：K 小顶堆 / 快速选择 |
| [LC 295](../../LeetCode/LeetCode_295/) | 数据流的中位数 | 困难 | 对顶堆 |
| [LC 347](../../LeetCode/LeetCode_347/) | 前 K 个高频元素 | 中等 | 计数 + K 小顶堆 |

---

## 6. 与其他篇的联系

- 理论：[DSA 12](../../DSA/DSA_12/)——偏序就够的论证、上滤/下滤推导、Floyd 建堆的求和、左式堆（可合并堆，刷题几乎不手写，知道存在即可）；
- [AT 02 二分查找](../AT_02_binary_search/)：TopK 的另一条路——快速选择，平均 $O(n)$ 不需要堆；
- [AT 21 排序](../AT_21_sorting/)：堆排序 = 建堆 + 逐个"顶尾交换 + 下滤"，原地 $O(n\log n)$；
- [AT 12 最短路](../AT_12_shortest_path/)：堆优化 Dijkstra 是堆最重要的算法级应用——"节点重复入堆 + 出堆跳过"就是 §2.4 的懒删除；
- [AT 06 单调队列](../AT_06_queue_monotonic_queue/)（及[单调队列专题](../monotonic_queue/)）对照：窗口最值要"过期淘汰"用单调队列，动态插入取最值没有过期概念，用堆。

---

*下一篇 [AT 10：并查集](../AT_10_union_find/)——把"是否同属一个集合"维护到近乎 $O(1)$。*
