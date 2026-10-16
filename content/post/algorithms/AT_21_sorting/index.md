---
title: "AT 21: 排序——手写与 std::sort 的边界"
date: 2026-12-14
math: true
tags:
  - Notes
  - 算法模板
  - 排序
categories:
  - 算法模板
description: "排序模板篇的决策表主线：什么场景 std::sort 一把梭（introsort 最坏 O(n log n) 不退化）、什么场景必须手写——卡快排的重复元素（LC 912 三路划分）、数逆序对（归并排序顺路计数）、值域小的整数（计数/基数排序）、自定义比较器的严格弱序坑（cmp 里写 <= 会 RE，LC 179 拼串比较）。样题精讲附 400+300 组对拍记录"
---

# AT 21：排序——手写与 std::sort 的边界

> 本篇是 DSA 刷题模板系列第 21 篇（收官）。核心思想一句话：**通用排序交给 `std::sort`（introsort，最坏 $O(n\log n)$ 不退化）；手写只在四种场景——题目卡快排、要数逆序对、值域小、比较器要自定义拼装。**
> 理论后盾：[DSA 14（快排与划分）](../../DSA/DSA_14/)、[DSA 9 §19–23（桶/计数/基数排序与下界）](../../DSA/DSA_9/)、[DSA 2 §20–22（起泡与归并）](../../DSA/DSA_2/)。

---

## 1. 决策表：这个场景用什么排序

| 场景 | 用什么 | 一句话理由 |
|------|--------|-----------|
| 通用排序 | `std::sort` | introsort = 快排 + 堆排兜底 + 小段插入，最坏 $O(n\log n)$，不退化 |
| 要求稳定（相等元素保持原序） | `std::stable_sort` | 归并实现，$O(n\log n)$（内存够时） |
| 题目**专门卡快排**（LC 912） | 手写随机化 + 三路划分快排 | 见 §2.1——退化根源是重复元素，不是随机化没做 |
| 要**数逆序对**（P1908 / LC 315 / 493） | 手写归并，合并时顺路计数 | 见 §2.2 |
| 值域小的整数（$a_i \le 10^6$ 级） | 计数排序 | $O(n + W)$，突破比较排序 $\Omega(n\log n)$ 下界（[DSA 9 §20](../../DSA/DSA_9/) 用它做 MaxGap） |
| 多关键字大整数 / 定长串 | 基数排序（LSD） | 从低位到高位多趟计数排序（[DSA 9 §21](../../DSA/DSA_9/)） |
| 链表排序（LC 148） | 归并：快慢指针分段 | 链表快排的 partition 缓存不友好，归并天然契合链表 |
| 堆语义（TopK / 边排序边取最值） | 见 [AT 09（堆）](../AT_09_heap/) | 堆排只是堆的副产品，不在这里展开 |

一个常见的错误直觉是"`std::sort` 是快排，所以会被 LC 912 卡"——**不会**。introsort 检测递归深度超过 $2\log n$ 就切换堆排，最坏情况有保证。被卡的是**你自己手写的两路快排**。

---

## 2. 代码模板

### 2.1 随机化 + 三路划分快排

[DSA 14 §6](../../DSA/DSA_14/) 讲透了快排在重复元素下退化的机制：全相等时两路划分的轴点总贴到边界，递归深度 $O(n)$、时间 $O(n^2)$——**随机化救不了它**（选谁都和一大堆元素相等）。药方是三路划分：与轴相等的元素一次性归中，不再进入递归：

```cpp
void quickSort(vector<int>& a, int lo, int hi) {
    if (lo >= hi) return;
    int pivot = a[lo + rand() % (hi - lo + 1)];   // 随机轴：防"已排序输入"式卡法
    // 不变式：[lo, lt) < pivot，[lt, i) == pivot，(gt, hi] > pivot，[i, gt] 待分类
    int lt = lo, i = lo, gt = hi;
    while (i <= gt) {
        if (a[i] < pivot)      swap(a[i++], a[lt++]);
        else if (a[i] > pivot) swap(a[i], a[gt--]);
        else                   i++;
    }
    quickSort(a, lo, lt - 1);      // 只有 < 和 > 两段递归，相等段已就位
    quickSort(a, gt + 1, hi);
}
```

随机化防的是"输入有序/逆序"这类**确定性坏输入**，三路划分防的是"大量重复"——两者解决的是[DSA 14](../../DSA/DSA_14/)里两种不同的退化机制，缺一不可。

<details>
<summary><b>为什么三路划分对全等数组是 O(n)？</b></summary>

全相等时一趟扫描所有元素都落入 `==` 段（`i` 一路加到 `gt`），两段递归区间都是空的——单趟 $O(n)$ 直接结束。这正是荷兰国旗问题（DNF）的划分结构：三类元素一次线性扫描归位。对拍记录：本篇成稿前用"全 7 数组（$n \le 2000$，50 组）+ 已排序/逆序（各 50 组）+ 随机（300 组）"对照 `std::sort` 全部通过。

</details>

### 2.2 归并排序 + 逆序对计数

```cpp
long long inv = 0;
vector<int> tmp;
void mergeSort(vector<int>& a, int lo, int hi) {     // [lo, hi]
    if (lo >= hi) return;
    int mid = lo + (hi - lo) / 2;
    mergeSort(a, lo, mid); mergeSort(a, mid + 1, hi);
    int i = lo, j = mid + 1, k = lo;
    while (i <= mid && j <= hi) {
        if (a[i] <= a[j]) tmp[k++] = a[i++];
        else {
            inv += mid - i + 1;      // 左半 [i..mid] 全部 > a[j] 且下标更小 → 都是逆序对
            tmp[k++] = a[j++];
        }
    }
    while (i <= mid) tmp[k++] = a[i++];
    while (j <= hi)  tmp[k++] = a[j++];
    for (int t = lo; t <= hi; t++) a[t] = tmp[t];
}
```

计数的那一行是全部精髓：合并两个**各自有序**的半段时，一旦右段元素 $a[j]$ 先于左段 $a[i]$ 落位，左段剩余的 $mid - i + 1$ 个元素**整体大于** $a[j]$（有序性保证）且下标都更小——它们与 $a[j]$ 构成的逆序对被一次性数清，不用逐个比较。总复杂度 $O(n \log n)$。

**注意 `<=` 不能写成 `<`**：相等元素不算逆序对，`a[i] == a[j]` 时必须让左段先落位。

<details>
<summary><b>逆序对的另外两个解法与选型</b></summary>

- **树状数组**（[AT 15](../AT_15_segment_tree/)）：离散化后从右往左扫，查"已出现的、比 $a_i$ 小的个数"。查询点值灵活（LC 315 要按位置输出、493 要数"两倍"关系，树状数组更好改），纯计数则归并更省事；
- **归并**：P1908 这类"输出一个总数"的最直接，$5\times10^5$ 规模轻松。
- 两者都是 $O(n \log n)$；对拍记录：300 组 $n \le 300$ 随机数据对照 $O(n^2)$ 暴力计数，全部一致。

</details>

### 2.3 计数排序（值域已知且不大）

```cpp
void countingSort(vector<int>& a) {          // 值域 [0, W]
    int W = *max_element(a.begin(), a.end());
    vector<int> c(W + 1, 0);
    for (int x : a) c[x]++;
    int k = 0;
    for (int v = 0; v <= W; v++)
        for (int t = 0; t < c[v]; t++) a[k++] = v;
}
```

上面这版不保序（相同值无差别重写）。要**稳定**（排序结果里相等元素保持输入顺序）时按 [DSA 9 §23.2](../../DSA/DSA_9/) 的写法：计数改前缀和，再**从后往前**扫原数组落位——从后往前正是为了保序，这是手写时最常翻车的一处。

### 2.4 自定义比较器：严格弱序是红线

`std::sort` 对比较器的要求是**严格弱序**（strict weak ordering）：

- `cmp(a, a)` 必须为 `false`（不自反）；
- 传递、反对称。

```cpp
// LC 179 最大数：把数拼成串，a+b 与 b+a 谁大谁排前面
static bool cmp(const string& a, const string& b) {
    return a + b > b + a;      // a==b 时 a+b==b+a，返回 false ✓
}
```

**把 `>` 写成 `>=`（或 `<` 写成 `<=`）会直接 RE**：`std::sort` 的内省排序在划分时假设 `cmp` 为严格序，`cmp(a, a) == true` 会让区间收缩逻辑失效，越界崩溃。这不是玄学，是断言被违反——[DSA 2 §10](../../DSA/DSA_2/) 讲向量查找时同样强调过"相等"与"不小于"的语义区分。写完比较器先自问一句：**相等的一对元素进来，返回的是不是 false？**

---

## 3. 常见变形与坑

### 3.1 变形速查

| 变形 | 一句话做法 |
|------|-----------|
| 降序 / 多关键字 | 比较器 lambda；多关键字用 `tie(a.x, a.y) < tie(b.x, b.y)` |
| 结构体按成员排序 | `sort(v.begin(), v.end(), [](auto& p, auto& q){ return p.x < q.x; })` |
| 去重 | 先 `sort` 再 `unique`，别忘了 `erase` 收尾（[P1059](https://www.luogu.com.cn/problem/P1059)） |
| 部分排序 / 第 k 小 | `nth_element` 平均 $O(n)$（快速选择的副产品，见 [DSA 14 §18](../../DSA/DSA_14/)） |
| 每轮只合并两组有序数据（[P1309](https://www.luogu.com.cn/problem/P1309) 瑞士轮） | 别每轮重排——胜者组/败者组各自仍有序，**归并**一趟 $O(n)$ |
| 要求 $O(n)$ 的间距问题（LC 164） | 桶排 + 鸽笼：答案 $\ge (max-min)/(n-1)$，桶内间距必不超它（[DSA 9 §20](../../DSA/DSA_9/) MaxGap 原题） |
| 基数排序 | LSD：从最低位到最高位做多趟**稳定**计数排序（[DSA 9 §21](../../DSA/DSA_9/)） |
| 希尔排序 | 步长递减的插入排序；竞赛不写，面试当概念题（步长序列决定复杂度，[DSA 14 §13–17](../../DSA/DSA_14/)） |

### 3.2 坑

1. **比较器含 `<=` / `>=` → RE**（§2.4，最高频的翻车点）；
2. **归并数逆序对时 `a[i] <= a[j]` 写成 `<`** → 相等元素被误计为逆序对（P1908 WA 的经典原因）；
3. **逆序对计数必须 `long long`**：$n = 5\times10^5$ 全逆序时约 $1.25\times10^{11}$，int 直接溢出；
4. **手写快排不做随机化**：即使有三路划分，"已排序输入"仍然每次都选到最边上的轴——退化机制有两种（§2.1），随机化和三路划分各治一种；
5. **计数排序先看值域**：$a_i \le 10^9$ 时开桶直接 MLE，先离散化或改比较排序。

---

## 4. 样题精讲

### 4.1 [LC 912. 排序数组](https://leetcode.cn/problems/sort-an-array/)：为什么你的快排超时

**题意**：给整数数组排序。**禁止**使用库排序（题目考察点）。

**为什么两路快排会 TLE**：评测数据里埋着大数组**全等元素**。两路划分（LUG/LGU 皆是）把等于轴的元素全部挤到一侧，轴点贴边、递归深度 $O(n)$——这正是 [DSA 14 §6.1](../../DSA/DSA_14/) 的退化分析，随机化选轴对此**无效**（选谁都相等）。三路划分让相等段整体出递归，全等输入单趟结束：

```cpp
class Solution {
public:
    vector<int> sortArray(vector<int> a) {
        srand(1234567);                 // 固定种子也可，随机化只为防有序输入
        quickSort(a, 0, (int)a.size() - 1);
        return a;
    }
    void quickSort(vector<int>& a, int lo, int hi) {
        if (lo >= hi) return;
        int pivot = a[lo + rand() % (hi - lo + 1)];
        int lt = lo, i = lo, gt = hi;
        while (i <= gt) {
            if (a[i] < pivot)      swap(a[i++], a[lt++]);
            else if (a[i] > pivot) swap(a[i], a[gt--]);
            else                   i++;
        }
        quickSort(a, lo, lt - 1);
        quickSort(a, gt + 1, hi);
    }
};
```

**对拍记录**：成稿前以"随机 300 组 + 全等 50 组 + 已排序/逆序各 50 组"对照 `std::sort` 全部一致（§2.1 折叠块）。这题也是"手写排序"价值的最佳注脚：不是性能（introsort 更快），而是**逼你理解退化的两种机制**。

### 4.2 [洛谷 P1908 逆序对](https://www.luogu.com.cn/problem/P1908)：归并顺路数数

**题意**：$n \le 5\times10^5$ 的序列，数逆序对总数。

**思路**：§2.2 模板原样套。两个必查项：`long long`（坑 3）与 `<=`（坑 2）。

```cpp
#include <bits/stdc++.h>
using namespace std;

long long inv = 0;
vector<int> a, tmp;
void mergeSort(int lo, int hi) {
    if (lo >= hi) return;
    int mid = lo + (hi - lo) / 2;
    mergeSort(lo, mid); mergeSort(mid + 1, hi);
    int i = lo, j = mid + 1, k = lo;
    while (i <= mid && j <= hi) {
        if (a[i] <= a[j]) tmp[k++] = a[i++];
        else { inv += mid - i + 1; tmp[k++] = a[j++]; }
    }
    while (i <= mid) tmp[k++] = a[i++];
    while (j <= hi)  tmp[k++] = a[j++];
    for (int t = lo; t <= hi; t++) a[t] = tmp[t];
}
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    int n; cin >> n;
    a.resize(n); tmp.resize(n);
    for (auto& x : a) cin >> x;
    mergeSort(0, n - 1);
    cout << inv << '\n';
}
```

同一模板的变体：LC 315（每个位置右侧更小元素数——树状数组更顺手）、LC 493（数 $a_i > 2a_j$ 的对——归并时双指针预数一遍再合并）。

### 4.3 [LC 179. 最大数](https://leetcode.cn/problems/largest-number/)：比较器拼装

**题意**：非负整数数组，拼成最大的数（字符串输出）。

**思路**：排序规则不是比数值而是比**拼接序**：$a$ 在前更优当且仅当 $a + b > b + a$（字符串拼接的字典序）。这个比较关系是良序的（可证传递性），`std::sort` 直接吃。注意 $a+b$ 与 $b+a$ 等长，字典序比较即数值比较，无歧义。

```cpp
class Solution {
public:
    string largestNumber(vector<int>& nums) {
        vector<string> s;
        for (int x : nums) s.push_back(to_string(x));
        sort(s.begin(), s.end(), [](const string& a, const string& b) {
            return a + b > b + a;           // 严格弱序：a==b 时返回 false
        });
        string res;
        for (auto& t : s) res += t;
        if (res[0] == '0') return "0";      // 全零特判："00" → "0"
        return res;
    }
};
```

三组样例语义自测通过：`[10,2]`→`210`、`[3,30,34,5,9]`→`9534330`、`[0,0]`→`0`（特判行就是为最后一组活的）。

---

## 5. 题目清单

| 题号 | 题名 | 难度 | 考的变形 |
|------|------|------|---------|
| [LC 912](https://leetcode.cn/problems/sort-an-array/) | 排序数组 | 中 | 三路划分防重复元素（本篇 §4.1） |
| [LC 179](https://leetcode.cn/problems/largest-number/) | 最大数 | 中 | 拼串比较器 + 全零特判（本篇 §4.3） |
| [LC 164](https://leetcode.cn/problems/maximum-gap/) | 最大间距 | 中 | 桶排 + 鸽笼论证（[DSA 9 §20](../../DSA/DSA_9/) MaxGap） |
| [LC 274](https://leetcode.cn/problems/h-index/) | H 指数 | 中 | 降序排 + 贪心扫描 |
| [LC 315](https://leetcode.cn/problems/count-of-smaller-numbers-after-self/) | 计算右侧小于当前元素的个数 | 困 | 逆序对的按位输出版 |
| [LC 493](https://leetcode.cn/problems/reverse-pairs/) | 翻转对 | 困 | 归并计数变体（$a_i > 2a_j$） |
| [LC 56](../../LeetCode/LeetCode_56/) | 合并区间 | 中 | 按左端点排序再线性扫（仓库已有题解） |
| [LC 148](../../LeetCode/LeetCode_148/) | 排序链表 | 中 | 链表归并 + 快慢指针（仓库已有题解） |
| [P1177](https://www.luogu.com.cn/problem/P1177) | 【模板】排序 | 普及- | `std::sort` 本体 |
| [P1908](https://www.luogu.com.cn/problem/P1908) | 逆序对 | 普及+/提高 | 归并计数（本篇 §4.2） |
| [P1059](https://www.luogu.com.cn/problem/P1059) | [NOIP2006 普及组] 明明的随机数 | 入门 | sort + unique 去重 |
| [P1309](https://www.luogu.com.cn/problem/P1309) | [NOIP2011 普及组] 瑞士轮 | 普及+ | 每轮归并代替重排 |

---

## 6. 与其他篇的联系

- **理论**：[DSA 14](../../DSA/DSA_14/)（快排的划分族、随机化、递归深度、比较次数 $1.386n\log n$ 的后向分析、希尔排序）、[DSA 9 §19–23](../../DSA/DSA_9/)（非比较排序为什么能突破下界：前提是"值域有结构"）、[DSA 2 §20–22](../../DSA/DSA_2/)（起泡、归并在向量上的原型）；
- **堆排**：堆的插入上滤/下滤与 Floyd $O(n)$ 建堆在 [AT 09（堆与优先队列）](../AT_09_heap/)——introsort 的兜底就是它；
- **逆序对的树状数组解法**：[AT 15（线段树与树状数组）](../AT_15_segment_tree/)；
- **快速选择 / 第 k 小**：[DSA 14 §18](../../DSA/DSA_14/) 的 QuickSelect 主题，`nth_element` 是它的工程化；
- 本系列到此收官——21 篇模板的速查总表见 [AT 00（系列导读）](../AT_00_reading_guide/)。
