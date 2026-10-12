---
title: "DSA 14: Sorting"
date: 2026-03-27
math: true
tags:
  - Notes
  - DSA
categories:
  - DSA
description: "DSA CHAP14"
---

# 数据结构与算法笔记（十四）：排序

> 本文基于课程第十四章“排序”整理。内容主线包括：  
> 快速排序（QuickSort）的分而治之思想、LUG / DUP / LGU 三类划分策略、随机化与递归深度分析、比较次数的递推与后向分析；  
> 选取问题（Selection）中的众数、中位数、QuickSelect 与 LinearSelect（Median of Medians）；  
> 以及希尔排序（ShellSort）的框架、Shell 序列、Papernov–Stasevic 序列、Pratt 序列与 Sedgewick 序列。  
> 这一章真正的核心可以概括为一句话：  
> **排序与选取的关键，不只是“比较大小”，而是如何构造更聪明的划分、缩减和局部有序性，让问题规模以尽可能高效的方式递减。**  
> 主要依据你上传的课件内容：fileciteturn16file0

---

## 1. 这一章在排序体系中的位置

到这一章时，前面课程里其实已经零散接触过不少排序思想：

- 向量中的插入排序、选择排序、归并排序
- 堆与堆排序
- 词典章节里的桶排序、计数排序、基数排序

而这一章更像是三块重要内容的集中补完：

1. **快速排序（QuickSort）**  
   重点不在“递归代码怎么写”，而在：  
   - 轴点（pivot）究竟是什么  
   - 划分（partition）如何做到线性时间、原地、并尽量随机均衡  
   - 为什么平均复杂度优秀、实际表现极佳

2. **选取（Selection）**  
   重点在：  
   - 不必完整排序，也能找出第 `k` 小  
   - 众数与中位数之间的关系  
   - QuickSelect 与最坏线性的 LinearSelect

3. **希尔排序（ShellSort）**  
   重点在：  
   - 通过“步长序列 + 逐步求精”构造局部有序性  
   - 步长选择如何决定性能  
   - 为什么某些序列在理论与实践中更优

可以说，这一章真正训练的是一种更高级的算法感觉：

> **不要急着把全局问题一次性做完，而要先想办法制造“有利的局部结构”，再借此递减规模。**

---

# 第一部分：快速排序

## 2. QuickSort 的分而治之思想

## 2.1 轴点（pivot）是什么

课件第 2 页先从轴点的概念讲起：fileciteturn16file0

若在区间 `[lo, hi)` 中有某个位置 `mi`，其元素满足：

- 左侧所有元素都不大于它
- 右侧所有元素都不小于它

那么这个元素就是一个**轴点（pivot）**。

课件把这一点写成了非常直观的形式：

\[
\max([lo, mi)) \le \min((mi, hi))
\]

这意味着：

- 只要轴点一旦就位
- 整个序列就会自然分成左右两个子问题
- 左边只需递归排好
- 右边也只需递归排好
- 然后整体自然有序

也就是：

\[
sorted(S) = sorted(S_L) + sorted(S_R)
\]

课件第 2 页还特别对比了归并排序与快速排序：fileciteturn16file0

- **MergeSort 的难点在“合”**
- **QuickSort 的难点在“分”**

这句话非常值得记住。  
因为快速排序真正的全部技术含量，几乎都集中在：

> **如何高效地“培养轴点”**

---

## 2.2 QuickSort 的递归框架

课件第 3 页给出标准递归框架：fileciteturn16file0

```cpp
template <typename T> void Vector<T>::quickSort( Rank lo, Rank hi ) {
    if ( hi - lo < 2 ) return;
    Rank mi = partition( lo, hi );
    quickSort( lo, mi );
    quickSort( mi + 1, hi );
}
```

这段代码非常短，但已经完整表达了 QuickSort 的结构：

1. 区间太短则直接返回
2. 调用 `partition(lo, hi)`  
   把某个候选元素培养成真正轴点，返回其秩 `mi`
3. 递归处理左右两侧

所以：

- QuickSort 的整体框架极其简单
- 真正困难、也真正决定性能的，是 `partition()`

---

## 2.3 轴点“未必原本就存在”

课件第 4 页特别提醒了一个非常容易误解的点：fileciteturn16file0

> 在原始序列中，轴点未必存在。

也就是说：

- 并不是说我们扫描一遍就总能找到某个天然已经两侧分好的元素
- 相反，QuickSort 的工作，本质上就是：

> **通过交换，把某个候选者“培养成”轴点。**

课件还指出：

- 在有序序列中，所有元素都是轴点
- 但一般序列中可能完全没有轴点
- 特别地，某些 derangement（没有元素在原位的排列）中，初始时一个轴点都可能不存在 fileciteturn16file0

因此快速排序可以理解为：

> **把所有元素逐个转化为轴点的过程。**

---

## 3. 快速划分：LUG 版

从课件 14-A2 开始，进入最重要的 `partition()` 设计。fileciteturn16file0

## 3.1 LUG 的三段视角

课件第 6 页给出一个非常清晰的分区视角：fileciteturn16file0

把当前区间分成三段：

- `L`：小于等于 pivot 的部分
- `U`：尚未确定归属的未处理部分
- `G`：大于等于 pivot 的部分

也就是：

> `L + U + G`

然后让 `lo` 和 `hi` 从两端交替向中间靠拢：

- 右侧扫描，遇到小于 pivot 的元素，就送进 `L`
- 左侧扫描，遇到大于 pivot 的元素，就送进 `G`

最终当 `lo == hi` 时，把候选 pivot 填进中间空位，它就自然成为轴点。

课件对这一过程的总结是：fileciteturn16file0

- 各元素最多移动一次（候选者两次）
- 因而：
  - 时间 `O(n)`
  - 辅助空间 `O(1)`

这已经说明：

> 一个好的 partition，必须至少做到线性时间、原地完成。

---

## 3.2 LUG 代码

课件第 7 页给出 LUG 版实现：fileciteturn16file0

```cpp
template <typename T> Rank Vector<T>::partition( Rank lo, Rank hi ) { //[lo, hi)
    swap( _elem[ lo ], _elem[ lo + rand() % ( hi - lo ) ] ); //随机交换
    hi--; T pivot = _elem[ lo ];
    while ( lo < hi ) {
        while ( lo < hi && pivot <= _elem[ hi ] ) hi--;
        _elem[ lo ] = _elem[ hi ];
        while ( lo < hi && _elem[ lo ] <= pivot ) lo++;
        _elem[ hi ] = _elem[ lo ];
    }
    _elem[ lo ] = pivot; return lo;
}
```

---

## 3.3 这段代码的逻辑怎么理解

可以把整个过程想成“挖坑填坑”：

1. 一开始把候选轴点暂存在变量 `pivot` 里  
   原位置 `lo` 就空出来，成为一个“坑”

2. 从右往左找第一个 `< pivot` 的元素  
   把它填到左边的坑里  
   此时右侧被取走的位置变成新的坑

3. 再从左往右找第一个 `> pivot` 的元素  
   把它填到右边的坑里  
   此时左侧又形成新的坑

4. 如此反复，直到 `lo == hi`

5. 最后把 `pivot` 填回最后一个坑

所以这个版本其实不是“每发现两个错位元素就直接 swap”，而是：

> **交替扫描 + 交替搬运 + 最后回填轴点**

这也正对应课件第 8 页所说的不变性：fileciteturn16file0

- 始终保持：

\[
L \le pivot \le G
\]

- 且 `U = [lo, hi]` 中始终有一个空位在两端交替出现

---

## 3.4 LUG 版的复杂度与缺点

课件第 9 页总结了 LUG 版的三个性质：fileciteturn16file0

### 线性时间

尽管 `lo` 和 `hi` 交替移动，但累计移动距离不超过 `O(n)`。

### 原地（in-place）

只需 `O(1)` 附加空间。

### 不稳定（unstable）

因为：

- `lo` 和 `hi` 方向相反
- 左右两侧的重复元素在搬运后相对顺序可能颠倒

这也是快速排序通常不稳定的根源之一。

---

## 4. QuickSort 的随机化、空间复杂度与递归深度

课件 14-A3 到 14-A5 专门分析了快速排序的性能。fileciteturn16file0

## 4.1 最好、最坏与平均情况

课件第 13 页给出经典结论：fileciteturn16file0

### 最好情况

每次划分都接近平均，轴点总在中间附近：

\[
T(n)=2T(n/2)+O(n)=O(n\log n)
\]

### 最坏情况

每次划分都极度偏侧，例如轴点总是最小或最大元素：

\[
T(n)=T(n-1)+T(0)+O(n)=O(n^2)
\]

这说明：

- QuickSort 的性能完全取决于划分是否均衡
- 而划分是否均衡，又取决于 pivot 是否“够中间”

---

## 4.2 为什么要随机化选轴点

LUG 代码一开始做了：

```cpp
swap( _elem[ lo ], _elem[ lo + rand() % ( hi - lo ) ] );
```

课件第 13 页解释：fileciteturn16file0

- 随机选取（randomization）
- 或三者取中（sampling）

都只能**降低**最坏情况发生的概率，而不能从理论上彻底杜绝最坏情况。  
但这已经很有价值，因为现实输入往往不是 adversarial（专门为卡算法而设计）。

因此快速排序之所以“快速”，不是因为它绝不退化，而是因为：

> **它在高概率下会持续得到足够均衡的划分。**

---

## 4.3 递归深度与空间复杂度

课件第 11、15、16 页把空间复杂度与递归深度直接联系起来：fileciteturn16file0

- 最好情况：递归深度 `O(log n)`
- 最坏情况：递归深度 `O(n)`
- 平均情况：递归深度高概率接近 `O(log n)`

课件进一步引入“准居中（near-centered）pivot”的概念：fileciteturn16file0

- 若某次 pivot 落入宽度为 `λ` 的居中区间，则视为“准居中”
- 每一递归层都有 `λ` 的概率出现准居中
- 因而随着层数增加，出现足够多次准居中的概率会很高
- 从而以高概率保证递归深度仍是对数级

这段分析的核心思想是：

> 不需要每一步都非常均衡；  
> 只要“足够均衡的划分”出现得不太少，整体深度就压得住。

---

## 4.4 非递归 + 贪心：如何把辅助栈空间压到 `O(log n)`

课件第 12 页给出一个非常实用的技巧：fileciteturn16file0

```cpp
if ( mi-lo < hi-mi ) { 
    Put( Task, mi+1, hi ); 
    Put( Task, lo, mi ); 
}
else { 
    Put( Task, lo, mi ); 
    Put( Task, mi+1, hi ); 
}
```

也就是说：

- 每次先处理较小子问题
- 把较大子问题后压栈 / 先留着

这样可保证辅助栈中同时积压的任务数不超过 `O(log n)`。

这个技巧本质上是一种“递归消除 + 贪心调度”：

> **总把较短任务优先做掉，让长任务少积压。**

这在很多分治算法中都非常值得借鉴。

---

## 5. QuickSort 的比较次数：为什么平均是 `≈ 1.386 n log n`

课件 14-A5 做了两个分析：递推分析与后向分析。fileciteturn16file0

## 5.1 递推分析

课件第 18～19 页令 `T(n)` 为期望比较次数，推出：fileciteturn16file0

\[
T(n)=(n-1)+\frac{1}{n}\sum_{k=0}^{n-1}\left[T(k)+T(n-k-1)\right]
\]

进一步整理后可得：

\[
T(n)\approx 2n\ln n = (2\ln 2)\, n \log_2 n \approx 1.386 \, n \log_2 n
\]

这给出了快速排序非常经典的平均比较次数常数。

---

## 5.2 后向分析（Backward Analysis）

课件第 20～21 页给出了另一种更优雅的解释：fileciteturn16file0

设排序后输出序列为：

\[
\{a_0, a_1, \dots, a_{n-1}\}
\]

对于任意一对元素 `(a_i, a_j)`，它们在排序过程中会不会被比较？

课件指出：

> 当且仅当在区间 \(\{a_i, a_{i+1}, \dots, a_j\}\) 中，  
> `a_i` 或 `a_j` 率先被确认成 pivot 时，这一对才会被比较。 fileciteturn16file0

于是该对被比较的概率是：

\[
\Pr(i,j)=\frac{2}{j-i+1}
\]

再对所有元素对求和，就得到与上面同样的结论：

\[
T(n)\approx 2n\ln n
\]

---

## 5.3 为什么 QuickSort 实际上常比 MergeSort 更快

课件第 22 页给出了一个非常有意思的对比表：fileciteturn16file0

- MergeSort：
  - 比较次数严格 `O(1.00 * n log n)`
  - 移动次数也严格 `O(1.00 * n log n)`
  - 但实际数据移动往往更多，且非原地

- QuickSort：
  - 平均比较次数约 `O(1.386 * n log n)`
  - 但移动次数平均更少
  - 原地（in-place）
  - cache locality 往往更好

所以即使从“比较次数常数”看，QuickSort 并不占优，  
但真实机器上它常常仍跑得更快。

这非常说明问题：

> 算法的实际性能，不只由比较次数决定，还高度依赖：
> - 数据移动
> - 原地性
> - cache 友好性
> - 分支行为
> - 常数因子

---

## 6. 重复元素：为什么普通 partition 会退化

课件 14-A6 专门讨论重复元素问题。fileciteturn16file0

## 6.1 重复元素为什么会害惨 QuickSort

课件第 24 页指出：fileciteturn16file0

若大量甚至全部元素重复，则：

- 轴点位置总接近 `lo`
- 子序列划分极度失衡
- 二分递归退化为线性递归
- 递归深度接近 `O(n)`
- 运行时间接近 `O(n^2)`

这说明：

> 重复元素会让“看似平衡的随机选轴点”失去意义，  
> 因为无论选谁，都可能和一大堆元素相等。

---

## 6.2 LUG'、DUP' 与 DUP

课件第 25～27 页依次给出三个版本：fileciteturn16file0

### LUG' 版

与原始 LUG 非常接近，只是把赋值动作写得更严谨，避免越界和多余移动。

### DUP' 版

将比较条件从：

- `pivot <= _elem[hi]`
- `_elem[lo] <= pivot`

改成更严格 / 更对称的形式，使得与 pivot 相等的元素不会总被挤到同一边。

### DUP 版

进一步调整循环写法，使得当遇到与 pivot 相等的元素时：

- `lo` 与 `hi` 会交替移动
- 两边对相等元素的分流更均匀
- 因而 pivot 最终更接近中间位置

课件第 28 页总结得很清楚：fileciteturn16file0

- 处理重复元素时，`lo` 和 `hi` 会交替移动
- 轴点最终大致落在 `(lo+hi)/2`
- 一般情况复杂度并未实质增高
- 代价是交换操作增加，且“更”不稳定

这其实是一种非常重要的工程化修正：

> **在真实数据中，重复值往往很多；一个只在“互异元素”模型下表现好的 QuickSort 是不够的。**

---

## 7. LGU 版划分：更简单的单向扫描分区

课件 14-A7 给出另一类 partition：LGU 版。fileciteturn16file0

## 7.1 核心不变性

课件第 30～31 页用区间表示其不变性，大意是：fileciteturn16file0

- `[lo+1, mi]`：都 `< pivot`
- `[mi+1, k)`：都 `>= pivot`
- `[k, hi)`：未处理

于是只需让 `k` 从左到右扫描：

- 若 `_elem[k] < pivot`
  - 就把它交换到 `mi+1`
  - 同时扩张左区
- 否则
  - 直接扩张右侧“非小于”区

---

## 7.2 代码实现

课件第 32 页给出：fileciteturn16file0

```cpp
template <typename T> Rank Vector<T>::partition( Rank lo, Rank hi ) { //[lo, hi)
    swap( _elem[ lo ], _elem[ lo + rand() % ( hi – lo ) ] );
    T pivot = _elem[ lo ]; int mi = lo;
    for ( Rank k = lo + 1; k < hi; k++ )
        if ( _elem[ k ] < pivot )
            swap( _elem[ ++mi ], _elem[ k ]);
    swap( _elem[ lo ], _elem[ mi ] );
    return mi;
}
```

这段代码本质上非常简洁：

- 单指针 `k` 扫描
- 一个边界 `mi` 标记“小于 pivot 区”的右边界
- 最后把 pivot 放回 `mi`

---

## 7.3 LGU 的优缺点

### 优点

- 代码短
- 单向扫描，逻辑清楚
- 对 cache 更友好
- 很像许多工程实现中的 Lomuto partition 变体

### 缺点

- 对重复元素的分流不如双向划分精细
- 交换次数可能更多
- 常数性能视输入而定

所以从思想上看：

> LUG / DUP 更强调双向相向靠拢  
> LGU 更强调单向扫描和边界维护

它们本质上都在做一件事：  
把候选元素培养成 pivot，只是维护不变性的方式不同。

---

# 第二部分：选取与中位数

## 8. 选取问题：不排序也能找第 `k` 小

课件 14-B1 开始进入 Selection。fileciteturn16file0

## 8.1 k-selection 的定义

课件第 35 页定义：fileciteturn16file0

> 在任意一组可比较元素中，找出次序为 `k` 的元素。  

也就是：

- 若把全部元素按非降序排好
- 找出排在第 `k` 个位置的元素

这就是：

> **第 `k` 小 / 第 `k` 大问题**

课件还指出：

- 中位数（median）是 `k-selection` 的特例
- 而且通常也是其中最典型、最重要的一种 fileciteturn16file0

---

## 8.2 为什么不能总是“先排序再取”

最直接方法当然是：

1. 先排序，`O(n log n)`
2. 再走到第 `k` 个位置，`O(1)` 或 `O(k)`

但问题在于：

> **我们只是想知道第 `k` 小是谁，并不关心其余元素之间的完整次序。**

既然不需要完整排序，就应该有希望做得更快。  
课件第 51 页甚至明确给出下界判断：fileciteturn16file0

- 最快也不可能快过 `Ω(n)`，因为每个元素至少要看一眼
- 于是自然追问：能否做到 `O(n)`？

这就导向 QuickSelect 和 LinearSelect。

---

## 9. 众数（majority）与中位数的关系

课件第 36～39 页先讲了一个看似旁支、其实很有启发的问题：**众数**。fileciteturn16file0

## 9.1 众数的定义

若在向量 `A` 中，某元素出现次数超过一半，则称其为：

> **众数（majority element）** fileciteturn16file0

课件第 36 页指出一个关键必要条件：

> 若众数存在，则它必然也是中位数。

原因很简单：

- 若某元素超过一半
- 那它必定跨越有序序列中间位置

因此，若我们会找中位数，就很容易验证它是不是众数。

---

## 9.2 Boyer–Moore Majority Vote 思想

在“高效中位数算法尚未知”之前，课件第 37～39 页给出更巧妙的思路：fileciteturn16file0

通过更弱的必要条件，先筛出唯一候选者，再最后验证。

代码如下：fileciteturn16file0

```cpp
template <typename T> T majCandidate( Vector<T> A ) {
    T maj;
    for ( int c = 0, i = 0; i < A.size(); i++ )
        if ( 0 == c ) {
            maj = A[i]; c = 1;
        } else
            maj == A[i] ? c++ : c--;
    return maj;
}
```

这其实就是著名的 **Boyer–Moore Majority Vote** 算法。

---

## 9.3 为什么“配对抵消”是对的

课件第 38 页给出它的减而治之解释：fileciteturn16file0

若某个偶数长度前缀中，元素 `x` 恰好占一半，则：

- 删去这个前缀后
- 原数组有众数，当且仅当剩余后缀也有同一众数

直观上理解：

- 每次拿一个“候选者”与一个“不同者”配对抵消
- 若真正众数存在，它的净优势不会被完全消灭
- 最终留下来的候选者若真是众数，再做一次线性验证即可

这段内容非常值得学，因为它展示了：

> **选取问题里经常可以先用更弱条件做线性筛选，再用一次验证收尾。**

---

## 10. 两个有序向量的中位数

课件 14-B2 接着讨论另一类中位数问题：**归并向量的中位数**。fileciteturn16file0

## 10.1 为什么不直接归并

给定两个已排序向量 `S1` 和 `S2`。  
最直观做法当然是：

- 先 merge 成一个有序向量
- 再取中位数

但这需要：

\[
O(n)
\]

甚至更多空间。  
课件第 41 页指出，这样“未能充分利用有序性”。fileciteturn16file0

---

## 10.2 等长情形的减而治之

课件第 42～43 页先处理等长情形。fileciteturn16file0

若比较两个子向量的中位点：

- 若 `S1[mi1] < S2[mi2]`
  - 则全局中位数不可能在 `S1` 的左半边，也不可能在 `S2` 的右半边
  - 可各删去一半
- 反之同理
- 若相等，则直接得到答案

于是每一步规模减半，  
总复杂度为：

\[
O(\log n)
\]

---

## 10.3 任意长度情形

课件第 44～45 页进一步推广到 `n1 != n2` 的一般情形。fileciteturn16file0

代码更复杂，但思想仍一样：

- 利用两边的有序性
- 每一步排除不可能包含中位数的大片区间
- 递归规模取决于较短那个向量

于是最终复杂度是：

\[
O(\log(\min(n_1,n_2)))
\]

课件还特别指出：

> 实际上，等长版本才是难度最大的。 fileciteturn16file0

这句话很有意思，因为一开始大家常觉得“等长最简单”，  
但从推广与逻辑边界来看，等长情形反而是最核心的骨架。

---

## 11. QuickSelect：基于 partition 的线性期望选取

课件 14-B3 进入最经典的第 `k` 小算法：**QuickSelect**。fileciteturn16file0

## 11.1 思路与 QuickSort 的关系

QuickSelect 和 QuickSort 几乎共享同一个 partition 操作。  
区别在于：

- QuickSort 会递归处理 pivot 左右两侧
- QuickSelect 只递归进入**包含第 `k` 小元素的那一侧**

因为我们只要一个元素，不需要两边都完全有序。

所以它相当于：

> **QuickSort 的“单分支版本”**

---

## 11.2 代码实现

课件第 52 页给出：fileciteturn16file0

```cpp
template <typename T> void quickSelect( Vector<T> & A, Rank k ) {
    for ( Rank lo = 0, hi = A.size() - 1; lo < hi; ) {
        Rank i = lo, j = hi; T pivot = A[lo];
        while ( i < j ) {
            while ( i < j && pivot <= A[j] ) j--; A[i] = A[j];
            while ( i < j && A[i] <= pivot ) i++; A[j] = A[i];
        }
        A[i] = pivot;
        if ( k <= i ) hi = i - 1;
        if ( i <= k ) lo = i + 1;
    }
}
```

它其实是在不断缩小“第 `k` 小可能落入的区间”：

- 若 `k < i`，只需去左边
- 若 `k > i`，只需去右边
- 若 `k = i`，就找到了

---

## 11.3 为什么 QuickSelect 的期望是线性的

课件第 53 页指出：fileciteturn16file0

- 与 QuickSort 类似，若 pivot 平均分布足够随机
- 每轮只需线性扫描做一次划分
- 且之后只进入一侧

因此可证明其期望运行时间为：

\[
O(n)
\]

这已经比“先排序再取”的 `O(n log n)` 更优。

不过要注意：

- 最坏情况仍可能是 `O(n^2)`
- 所以若想要严格最坏线性，还需要 LinearSelect

---

## 12. LinearSelect：Median of Medians

课件 14-B4 给出最坏 `O(n)` 的选取算法。fileciteturn16file0

## 12.1 算法步骤

课件第 55 页给出标准框架（以常数 `Q` 分组）：fileciteturn16file0

1. 若 `n < Q`，直接平凡求解
2. 否则把 `A` 均匀分成若干个大小为 `Q` 的小组
3. 各组排序并求出组中位数
4. 递归地求“中位数的中位数” `M`
5. 按 `M` 把原数组划成：
   - `L`：小于 `M`
   - `E`：等于 `M`
   - `G`：大于 `M`
6. 判断第 `k` 小落在哪一部分，递归进入对应子集

这就是著名的 **Median of Medians**。

---

## 12.2 为什么它能保证最坏线性

课件第 56 页给出了复杂度分析框架：fileciteturn16file0

设总时间为 `T(n)`，则：

- 分组：`O(n)`
- 组内排序：`O(n)`
- 递归找中位数的中位数：`T(n/Q)`
- 划分 `L/E/G`：`O(n)`
- 递归进入较大一侧：`T(3n/4)`（课件这里给出的是一个足够好的上界）

于是递推可写成：

\[
T(n) \le T(n/Q) + T(3n/4) + O(n)
\]

其解为：

\[
T(n)=O(n)
\]

这说明：

> 只要每次选到的 pivot “保证不太坏”，就能把 QuickSelect 从“期望线性”提升到“最坏线性”。

---

## 12.3 为什么组大小常取 5

虽然课件这里用符号 `Q` 表示一般常数分组大小，但经典 Median of Medians 通常取 5。  
原因就在于：

- 5 足以保证 pivot 不会太偏
- 又不会让组内排序常数过大
- 因而能得到最经典的最坏 `O(n)` 证明

---

# 第三部分：希尔排序

课件 14-C1 到 14-C5 进入 ShellSort。fileciteturn16file0

## 13. ShellSort 的总体思想：递减增量、逐步求精

## 13.1 为什么插入排序会慢

普通插入排序之所以慢，本质原因是：

- 元素一旦离最终位置很远
- 只能一步一步向前挪
- 长距离逆序很难快速消除

ShellSort 的想法正是：

> **先用大步长把“远距离逆序”粗略打散，  
> 再逐步缩小步长做精细修正。**

课件第 59 页把它总结为：fileciteturn16file0

- 将序列看作一个矩阵，逐列排序
- 采用递减增量（diminishing increment）
- 由粗到细，逐步求精
- 最后当步长变成 1 时，就等价于全排序

也就是：

\[
1\text{-sorted} = ordered
\]

---

## 13.2 h-sorting 与 h-ordered

课件第 71 页给出定义：fileciteturn16file0

若序列 `S[0,n)` 满足：

- 对所有 `i`，都有：

\[
S[i] \le S[i+h]
\]

（在下标合法范围内）

则称其为：

> **h-ordered**

而对序列做一趟“按步长 `h` 分成若干列，并分别排序”的操作，就叫：

> **h-sorting**

于是 ShellSort 的过程就是：

- 先做某个大 `h` 的 h-sorting
- 再换一个更小步长
- 最终做 `1-sorting`

---

## 13.3 一维数组如何模拟矩阵按列排序

课件第 65 页特别解释：fileciteturn16file0

其实不需要真的开二维数组。

若当前步长为 `h`，则原一维数组中下标：

\[
j,\ h+j,\ 2h+j,\ \dots,\ ih+j,\dots
\]

就对应“矩阵中的同一列”。

所以只要把“插入排序”的相邻比较从：

- `i` 与 `i-1`

改成：

- `i` 与 `i-h`

就能在一维数组上完成 h-sorting。

---

## 13.4 代码实现

课件第 66 页给出 ShellSort 实现：fileciteturn16file0

```cpp
template <typename T> void Vector<T>::shellSort( Rank lo, Rank hi ) {
    for ( Rank d = 0x3FFFFFFF; 0 < d; d >>= 1 )
        for ( Rank j = lo + d; j < hi; j++ ) {
            T x = _elem[j]; Rank i = j - d;
            while ( lo <= i && _elem[i] > x ) {
                _elem[i + d] = _elem[i];
                i -= d;
            }
            _elem[i + d] = x;
        }
}
```

这本质上就是：

> **步长为 `d` 的插入排序**

随着 `d` 逐渐减小，局部有序性不断增强，最终 `d=1` 时完成全排序。

---

## 14. 为什么步长序列如此关键

课件 14-C2、14-C3、14-C4、14-C5 基本都在研究“步长序列”。fileciteturn16file0

## 14.1 Shell 原始序列为什么不好

课件第 68 页指出 Shell 1959 的原始序列虽然自然，但最坏情况并不好。fileciteturn16file0

根源在于：

> **步长之间往往不互素，甚至相邻项也可能不互素。**

这会导致：

- 某些列结构彼此独立
- 一次 h-sorting 建立的局部有序性，不能充分传递给下一次步长
- 最终仍可能留下大量逆序，导致最后一次 `1-sorting` 很重

所以 ShellSort 的性能高度依赖：

> **步长序列能否让不同层次的“局部有序”有效叠加。**

---

## 14.2 Theorem K 与线性组合思想

课件第 72～76 页给出一条非常有名的性质（Knuth, ACP）：fileciteturn16file0

> 一个 `g`-ordered 序列，在经过 `h`-sorting 后，仍保持 `g`-ordered。

并进一步推得：

若一个序列同时是 `g`-ordered 与 `h`-ordered，  
则它还会是：

\[
(g+h)\text{-ordered},\quad (mg+nh)\text{-ordered}
\]

对任意自然数 `m,n` 都成立。 fileciteturn16file0

这非常关键，因为它说明：

> 不同步长带来的局部有序性，可以通过线性组合不断积累和传播。

这就是课件后面引入“互素”“线性组合”“邮票问题”的原因。  
它其实是在分析：

- 什么样的步长集合，能让最终只剩很少逆序对
- 从而让最后几步特别快

---

## 15. Papernov–Stasevic（PS）序列

课件第 79～81 页介绍了 Papernov & Stasevic 序列，也常叫 Hibbard 序列一类。fileciteturn16file0

其核心特点是：

- 序列项之间并不要求全部互素
- 但相邻项必须互素
- 这样能保证局部有序性逐步传递

课件给出其复杂度结论：fileciteturn16file0

- 外层迭代次数是对数级
- 总复杂度优于原始 Shell 序列

并说明这一上界还是紧的。

---

## 16. Pratt 序列

课件 14-C4 进一步介绍 Pratt 序列（1971）。fileciteturn16file0

## 16.1 基本思想

Pratt 序列由所有形如：

\[
2^p 3^q
\]

的数构成，并按大小递增排列。  
这样做的好处是：

- 2 与 3 互素
- 大量步长之间的线性组合能力很强
- 可把序列逐步打造成 `(2,3)`-ordered，再进一步逼近 `1`-ordered

---

## 16.2 课件中的复杂度结论

课件第 83～85 页给出分析：fileciteturn16file0

- Pratt 序列项数较多
- 迭代轮数因此偏大
- 但可以证明其最坏复杂度有很好的理论上界

其代价是：

> **理论较强，但实际迭代次数偏多。**

---

## 17. Sedgewick 序列

课件最后第 87 页指出：fileciteturn16file0

- Pratt 序列迭代太多，对“预排序输入”不友好
- Sedgewick 序列结合了 PS 序列与 Pratt 序列的优点
- 在实践中通常表现最好

也就是说：

- ShellSort 的理论研究非常依赖步长设计
- 而实践中，Sedgewick 往往是最常用的高质量步长序列之一

这再次说明：

> ShellSort 不是一个单独算法，而是一整个“算法家族”。  
> 它的具体性能，很大程度上取决于你选了哪一套 gap sequence。

---

# 第四部分：本章整体串联

## 18. QuickSort、Selection 与 ShellSort 之间的共同主题

这三块内容表面上看差异很大：

- QuickSort：分治
- Selection：找第 `k` 小
- ShellSort：递减增量

但它们实际上共享几个非常深的共同主题。

---

## 18.1 主题一：不要一次性做完所有事

### QuickSort

不去直接全局归并，而是先通过 pivot 把问题切成左右两块。

### QuickSelect

不去完整排序，只处理包含第 `k` 小的那一边。

### ShellSort

不去直接做一步到位的插入排序，而是先用大步长构造粗糙局部有序。

所以三者都体现了一种思想：

> **先制造一个对后续特别有利的局部结构，再递减规模。**

---

## 18.2 主题二：划分（partition）是核心

### QuickSort

pivot 划分决定性能。

### QuickSelect

仍是依靠 partition 缩小搜索区间。

### LinearSelect

也是先找到足够好的 pivot，再三路划分成 `L/E/G`。

这说明在排序 / 选取问题中，  
真正厉害的往往不是“比大小”本身，而是：

> **如何让一次划分尽量多地排除掉不必要的候选。**

---

## 18.3 主题三：局部有序性能被积累和传递

### K-selection / median

利用“比它小”和“比它大”的分组结构。

### ShellSort

利用 `g`-ordered、`h`-ordered 的叠加，形成更强有序性。

### QuickSort

一旦某个元素变成轴点，它就永远不必再动。

这些都说明：

> **好的算法会尽量把已经建立起来的秩序保存下来，而不是每轮都推倒重来。**

---

## 18.4 主题四：最坏情况、平均情况与工程表现要区分看待

课件在 QuickSort 与 ShellSort 两部分都反复体现了这一点：

- QuickSort 最坏 `O(n^2)`，但平均极佳、实践极强
- ShellSort 理论复杂度受步长序列影响很大，但某些序列在工程上非常好
- LinearSelect 理论最坏线性，但常数大，实际有时不如 QuickSelect

所以：

> **理论最优、平均最优、工程最优，三者常常并不完全一致。**

这也是算法课程中非常成熟的一种视角。

---

## 19. 本章主线总结

### 19.1 QuickSort 的核心不在递归，而在 partition

- 轴点不是“天然存在”的，而是被培养出来的
- LUG / DUP / LGU 都是在设计不同的不变性来完成线性划分
- 随机化保证高概率均衡
- 平均比较次数约为 `1.386 n log n`
- 实际往往比 MergeSort 更快

### 19.2 Selection 说明“找第 k 小”不必完整排序

- 众数若存在，则必是中位数
- QuickSelect：期望 `O(n)`
- LinearSelect：最坏 `O(n)`

### 19.3 中位数问题展示了减而治之的力量

- 两个有序向量的中位数可做到 `O(log(min(n1,n2)))`
- Median of Medians 则展示了“保证 pivot 不太坏”就能得到最坏线性

### 19.4 ShellSort 的本质是逐步构造局部有序性

- h-sorting / h-ordered 是其数学核心
- 步长序列决定性能
- 原始 Shell 序列并不理想
- PS / Pratt / Sedgewick 等序列代表了不同的理论与工程折中

### 19.5 这一章最值得学的是“划分与缩减”能力

无论是 QuickSort、QuickSelect 还是 ShellSort，  
本质都在训练一种能力：

> **如何通过一次局部操作，最大限度地缩小剩余问题规模，或最大限度地增强局部有序性。**

---

## 20. 一句话总结

> 这一章真正重要的，不只是学会快速排序、QuickSelect 和 ShellSort 的代码，而是理解：排序与选取问题的高效解法，往往来自更聪明的划分、更合理的递减策略，以及对“已经建立起来的局部秩序”的充分利用。
