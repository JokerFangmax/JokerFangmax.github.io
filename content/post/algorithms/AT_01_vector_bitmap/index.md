---
title: "AT 01: 向量与位图——倍增扩容、原地去重与压位集合"
date: 2026-11-24
math: true
tags:
  - Notes
  - 算法模板
  - 向量
  - 位图
categories:
  - 算法模板
description: "动态数组倍增扩容与分摊 O(1)、有序数组双指针原地去重（uniquify）、位图三档模板（bool 数组 / 位运算 / bitset），样题：LC 26/27/189/448 与洛谷 P1059"
---

# AT 01：向量与位图——倍增扩容、原地去重与压位集合

> 本系列是 [DSA 14 章](../../DSA/DSA_1/)的刷题武器库。本篇核心思想一句话：**动态数组的"倍增扩容"让插入的分摊成本恒为 $O(1)$；有序数组的双指针让所有"原地压缩"操作降到线性；位图用一个 bit 顶一个布尔，把空间压到 1/8。**
> 理论后盾：[DSA 2（向量）](../../DSA/DSA_2/)§6–12（扩容与分摊分析）、§23–25（位图与校验环）。

---

## 1. 这个模板解决什么问题

三类高频场景：

| 场景 | 朴素做法的痛 | 模板做法 |
|------|------------|---------|
| 未知规模的输入流式读入 | 开固定数组怕炸，开太大浪费 | `vector` 倍增扩容，分摊 $O(1)$ |
| 有序数组原地删除/去重 | 每删一个元素搬移 $O(n)$ | 双指针原地压缩 $O(n)$ |
| 大范围整数的存在性判断（去重、枚举、标记） | `map`/哈希表有常数和内存开销 | 位图，每元素 1 bit |

为什么刷题时几乎总写 `vector` 而不是原生数组——不是语法洁癖，是**分摊复杂度**保证（见折叠块）。

<details><summary>为什么倍增扩容的分摊成本是 O(1)（DSA 2 §8 的速记版）</summary>

设从空开始连续插入 $n$ 个元素。倍增策略下，扩容发生在容量 $1, 2, 4, \dots, 2^k$（$2^k \ge n$）处，每次扩容复制当前全部元素，总复制代价：

$$
1 + 2 + 4 + \cdots + 2^k = 2^{k+1} - 1 < 2n
$$

即 $n$ 次插入总代价 $O(n)$，**分摊每次 $O(1)$**。对比固定增量扩容（每次加 `INCREMENT`）：复制代价是算术级数 $0 + I + 2I + \cdots = O(n^2)$，分摊 $O(n)$——差一个数量级。这就是 `vector` 内部用倍增、而旧式顺序表教程用 `INCREMENT` 的全部原因。
</details>

---

## 2. 代码模板

### 2.1 倍增扩容（手写版，理解 vector 内部）

```cpp
template <typename T>
struct Vec {
    T* a = nullptr;
    int sz = 0, cap = 0;              // 元素数 / 容量

    void push(const T& x) {
        if (sz == cap) {              // 满了才扩
            cap = cap ? cap * 2 : 4;  // 容量倍增（首个容量随意，如 4）
            T* b = new T[cap];
            for (int i = 0; i < sz; i++) b[i] = a[i];  // 搬家
            delete[] a;
            a = b;
        }
        a[sz++] = x;
    }
};
```

关键点只有一条：**扩容倍率 $\ge 2$ 才有分摊 $O(1)$**；倍率太大（如 $\times 10$）浪费空间。工程上直接用 `std::vector`，刷题唯一需要记的是：`push_back` 分摊 $O(1)$，且扩容会使**所有迭代器/指针失效**——存了元素指针的代码要小心。

### 2.2 有序数组原地去重（uniquify 双指针）

DSA 2 §14 给出的低效版（逐个删除，$O(n^2)$）在刷题里几乎不用，直接写双指针版：

```cpp
// 有序数组原地去重，返回去重后长度 k，前 k 个元素为去重结果
int uniquify(vector<int>& a) {
    if (a.empty()) return 0;
    int k = 1;                        // [0, k) 是已固化的去重结果
    for (int i = 1; i < (int)a.size(); i++)
        if (a[i] != a[k - 1])         // 与结果段最后一个比较
            a[k++] = a[i];
    return k;
}
```

**不变量**：`a[0..k-1]` 始终是 `a[0..i]` 的去重结果；`k-1` 是结果段的"守门员"。有序是前提——相等元素必然相邻，只跟守门员比较就够。保留至多 $m$ 次重复的变体见折叠块。

<details><summary>变体：允许重复 ≤ m 次（LC 80 型）</summary>

```cpp
int uniquify_m(vector<int>& a, int m) {   // 每个元素至多保留 m 份
    int k = 0;
    for (int x : a)
        if (k < m || a[k - m] != x)       // 与往前数第 m 个比较
            a[k++] = x;
    return k;
}
```

守门员从"最后一个"变成"往前数第 $m$ 个"：`a[k-m] != x` 说明 $x$ 在结果段里出现不足 $m$ 次（有序性保证相等元素连续）。
</details>

### 2.3 移除/移动类原地操作（LC 27 / 283 型）

同样的双指针骨架换个语义：

```cpp
// 原地移除所有值为 val 的元素（LC 27），返回新长度
int removeElement(vector<int>& a, int val) {
    int k = 0;                          // [0, k) 是保留的元素
    for (int x : a)
        if (x != val) a[k++] = x;       // 慢指针 k 收留所有幸存者
    return k;
}
```

与 uniquify 的唯一区别是判定条件（`x != val` vs `a[k-1] != x`）——**快指针筛选，慢指针收留**，这个骨架覆盖一切"原地压缩"。

### 2.4 位图三档

DSA 2 §23 的 Bitmap 用 `char` 数组压位。刷题按场景选档：

| 档位 | 写法 | 每元素空间 | 适用 |
|------|------|-----------|------|
| 简单档 | `vector<bool>` / `bool vis[N]` | 1 字节（`vector<bool>` 特化后也是位） | 不卡空间，图遍历 `vis` 数组 |
| 手写档 | `char M[]` + 位运算 | 1 bit | 范围已知且要省空间，学习位运算 |
| STL 档 | `bitset<N>` | 1 bit | $N$ 编译期已知；自带整体位运算 |

```cpp
// 手写档：范围 [0, n) 的集合
struct Bitmap {
    vector<unsigned char> M;
    Bitmap(int n) : M((n + 7) / 8, 0) {}
    void set(int k)   { M[k >> 3] |=  (1u << (k & 7)); }
    void clear(int k) { M[k >> 3] &= ~(1u << (k & 7)); }
    bool test(int k)  { return M[k >> 3] >>  (k & 7) & 1; }
};
```

三行位运算的固定拆法：`k >> 3` 定位字节（除 8），`k & 7` 定位位（模 8），置位用 `|`、清位用 `& ~`、测试用 `>> & 1`。`bitset<200000000>` 不能开在栈上（约 24 MB），要么 `static` 要么全局。

---

## 3. 常见变形与坑

| 变形 | 一句话做法 |
|------|-----------|
| 去重但保留最后一份 | 倒着扫，或用 `unordered_map` 记录最后位置 |
| 去重同时保持频次 | `k < m \|\| a[k-m] != x`（§2.2 折叠块） |
| 数组循环右移 k 位 | 三次反转（见样题 LC 189） |
| 值域大但元素少（$10^9$ 范围） | 离散化 + 位图，或直接哈希表 |
| 快照/恢复标记 | 位图清零 $O(n/8)$ 优于哈希表 `.clear()` |

坑：

1. **uniquify 忘了先排序**——相等元素不相邻，守门员比较失效；
2. **`vector<bool>` 不是容器**（位压缩特化），取 `&a[i]` 拿不到指针，需要真字节数组时用 `vector<char>`；
3. 扩容后旧指针/迭代器失效——先 `push_back` 再取 `&a.back()` 的顺序问题；
4. 位图 `n` 为 0 时 `(n+7)/8 = 0`，`M` 为空但 `set(0)` 会越界——构造前检查范围。

---

## 4. 样题精讲

### 4.1 LC 26 / 27：原地压缩双子星

[LC 26. 删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)（Easy）、[LC 27. 移除元素](https://leetcode.cn/problems/remove-element/)（Easy）——§2.2 / §2.3 模板的原样套用，各一行判定条件。返回值是新长度，**不用真的删掉后面的元素**（这是"原地"的语义：前 $k$ 个对就行）。各 $O(n)$ 时间、$O(1)$ 空间。

### 4.2 LC 189 轮转数组（Medium）：三次反转

[题解](../../LeetCode/LeetCode_189/)。把数组右移 $k$ 位，朴素做法是 $k$ 次整体搬移 $O(nk)$。三次反转 $O(n)$：

```cpp
class Solution {
    void rev(vector<int>& a, int lo, int hi) {   // 反转 [lo, hi]
        while (lo < hi) swap(a[lo++], a[hi--]);
    }
public:
    void rotate(vector<int>& nums, int k) {
        int n = nums.size();
        k %= n;                                  // k 可能 >= n
        rev(nums, 0, n - 1);                     // 整体反转
        rev(nums, 0, k - 1);                     // 前 k 个翻回来
        rev(nums, k, n - 1);                     // 其余翻回来
    }
};
```

直觉：右移 $k$ 位 = 尾部 $k$ 个元素搬到头部。整体反转后这 $k$ 个确实到了头部，但两段各自内部顺序反了——再各反转一次恢复。**用"反转"实现"循环移位"** 是数组题的常用小技巧。

### 4.3 洛谷 P1059 明明的随机数（去重模板题）

[P1059](https://www.luogu.com.cn/problem/P1059)。读入 $n$ 个随机数，去重后升序输出。教科书式的 sort + uniquify：

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    scanf("%d", &n);
    vector<int> a(n);
    for (auto& x : a) scanf("%d", &x);
    sort(a.begin(), a.end());
    int k = 1;                        // uniquify 双指针
    for (int i = 1; i < n; i++)
        if (a[i] != a[k - 1]) a[k++] = a[i];
    printf("%d\n", k);
    for (int i = 0; i < k; i++)
        printf("%d%c", a[i], i + 1 == k ? '\n' : ' ');
    return 0;
}
```

### 4.4 LC 448 找到所有数组中消失的数字（位图思想）

[LC 448](https://leetcode.cn/problems/find-all-numbers-disappeared-in-an-array/)（Easy）。数组长度 $n$，元素取值 $[1, n]$，找出没出现过的数。位图一行：开 `vis[n+1]` 标记，再扫一遍没标记的下标。进阶要求 $O(1)$ 空间时，**把原数组自身当位图用**——出现过的值 $v$ 对应下标 `a[v-1]` 加 $n$ 或取负，第二遍扫值域仍在 $[1,n]$ 的下标即为消失的数。这是"值域 = 下标域"时的经典空间复用。

---

## 5. 题目清单

| 题号 | 题名 | 难度 | 考的变形 | 链接 |
|------|------|------|---------|------|
| LC 26 | 删除有序数组中的重复项 | Easy | uniquify 双指针 | [题](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/) |
| LC 27 | 移除元素 | Easy | 快慢指针原地移除 | [题](https://leetcode.cn/problems/remove-element/) |
| LC 189 | 轮转数组 | Medium | 三次反转 | [题解](../../LeetCode/LeetCode_189/) |
| LC 448 | 找到所有数组中消失的数字 | Easy | 原数组当位图 | [题](https://leetcode.cn/problems/find-all-numbers-disappeared-in-an-array/) |
| LC 136 | 只出现一次的数字 | Easy | 位图视角：异或 | [题解](../../LeetCode/LeetCode_136/) |
| LC 283 | 移动零 | Easy | 移除 + 尾部填零 | [题解](../../LeetCode/LeetCode_283/) |
| 洛谷 P1059 | 明明的随机数 | 普及- | sort + uniquify | [题](https://www.luogu.com.cn/problem/P1059) |
| 洛谷 P1177 | 【模板】排序 | 普及- | 排序后去重的底座（排序篇 [AT 21](../AT_21_sorting/) 详解） | [题](https://www.luogu.com.cn/problem/P1177) |

---

## 6. 与其他篇的联系

- 理论：扩容与分摊分析见 [DSA 2 §6–8](../../DSA/DSA_2/)；位图与校验环（$O(1)$ 初始化位图的技巧）见 [DSA 2 §23–25](../../DSA/DSA_2/)；
- 本系列：去重后的有序数组正是二分（[AT 02](../AT_02_binary_search/)）的输入；值域大的"去重"走哈希表（[AT 17](../AT_17_hash_table/)）；
- 既有专题文：[前缀和](../prefix_sum/)与[滑动窗口](../sliding_window/)都在 `vector` 骨架上加一层维护。

---

*下一篇：[AT 02：二分查找与二分答案](../AT_02_binary_search/)——同一个 `while` 循环，查找和答案两种身份。*
