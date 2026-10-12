---
title: "哈希表：从查找到计数的问题模式"
date: 2026-09-25
math: true
tags:
  - Notes
  - 算法模板
  - 哈希表
categories:
  - 算法模板
description: "哈希表在算法题中的四大问题模式（查找/计数/分组/配对前缀和）、unordered_map 工程细节与题目清单"
---

# 哈希表：从查找到计数的问题模式

> 本篇是刷题沉淀的算法模板之一。核心思想一句话：**把"在一堆东西里找满足条件的某个"从 $O(n)$ 扫描变成 $O(1)$ 查表——用空间换"倒排索引"。**

---

## 1. 哈希表在算法题中到底扮演什么角色

哈希表本身不产生算法，它是**其他思路的加速器**。刷题中它几乎总以下面四种模式出现，认出模式比记住 API 重要：

| 模式 | 问题形态 | 哈希表存什么 |
|------|---------|-------------|
| ① 值查找 | "是否存在 / 找到那个数" | 值 → 下标（或值 → `true`） |
| ② 频次计数 | "有多少个 / 出现几次 / 是否重复" | 值 → 次数 |
| ③ 分组归类 | "把……归为一类 / 按某键聚合" | 键 → 元素列表 |
| ④ 前缀/状态配对 | "前面有没有和我配对的状态" | 状态（如前缀和）→ 次数或最早下标 |

---

## 2. 模式①：值查找——Two Sum 范式

[LC 1. 两数之和](https://leetcode.cn/problems/two-sum/)是原型：找 $a_i + a_j = target$，即对每个 $i$ 查询"前面是否出现过 $target - a_i$"。

```cpp
unordered_map<int, int> pos;               // 值 → 下标
for (int i = 0; i < n; i++) {
    auto it = pos.find(target - nums[i]);
    if (it != pos.end()) return {it->second, i};
    pos[nums[i]] = i;                      // 先查后存：不重复使用自己
}
```

关键点：**先查询、后插入**，保证只和"前面的元素"配对，避免元素和自己匹配（与 LC 560 中防止 `k=0` 自配对是同一个细节）。

同类题：[LC 217. 存在重复元素](https://leetcode.cn/problems/contains-duplicate/)、[LC 128. 最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/)（只从"没有前驱的数"起跳，$O(n)$）、[LC 41. 缺失的第一个正数](https://leetcode.cn/problems/first-missing-positive/)（原地哈希，键即下标）。

---

## 3. 模式②：频次计数

### 3.1 静态计数

"统计每个元素出现次数"是 `unordered_map` 最朴素的用法，通常只是前菜，真正的菜在后面（比较、筛除、组合）：

```cpp
unordered_map<int, int> cnt;
for (int x : nums) cnt[x]++;
```

代表题：[LC 347. 前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/)（计数 + 桶排序/堆）、[LC 383. 赎金信](https://leetcode.cn/problems/ransom-note/)（两串计数相减）、[LC 76. 最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/)（窗口计数与目标计数比对）。

### 3.2 定长计数（配合滑动窗口）

窗口滑动时"进一个、出一个"，计数增量维护，免得每步重数：

```cpp
unordered_map<int, int> cnt;
for (int i = 0; i < n; i++) {
    cnt[nums[i]]++;                        // 进
    if (i >= k) {
        if (--cnt[nums[i - k]] == 0) cnt.erase(nums[i - k]);
                                            // 出；归零后 erase，cnt.size() 才准确
    }
    if (i >= k - 1) { /* 用 cnt 判断窗口性质 */ }
}
```

**`erase` 归零键**这个细节常被忽略：不删的话值为 0 的键仍占着 `size()`，"窗口内不同元素个数"这类判断就会出错。

代表题：[LC 438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/)、[LC 567. 字符串的排列](https://leetcode.cn/problems/permutation-in-string/)。

### 3.3 键是数组/字符串时

`unordered_map` 没有内置针对 `vector<int>` / `string` 的哈希，C++17 前要么手写哈希函数，要么把键转成 `string`；C++17 起可用**透明哈希的自定义 struct**。竞赛里常见偷懒法：把小写字母串编码成整数键。代表题：[LC 49. 字母异位词分组](https://leetcode.cn/problems/group-anagrams/)（模式③，键用排序后的串最省心）。

---

## 4. 模式③：分组归类

按某个"键"把元素聚成桶：异位词按排序串分组、同余数按余数分组、坐标按 `(x/3, y/3)` 分组……

```cpp
unordered_map<string, vector<string>> groups;
for (auto& s : strs) {
    string key = s;
    sort(key.begin(), key.end());
    groups[key].push_back(s);              // operator[] 不存在时自动建桶
}
```

这里 `operator[]` 的"不存在则默认构造插入"从坑变成了便利——注意与 6.2 节的坑对照。

代表题：LC 49、[LC 1282. 用户分组](https://leetcode.cn/problems/group-the-people-given-the-group-size-they-belong-to/)。

---

## 5. 模式④：状态配对——哈希表的最高频进阶用法

当问题可以表达成"**我需要知道前面是否出现过某个状态**"时，哈希表存的是**累积状态**而不是元素本身。最典型的就是前缀和配对（详见[前缀和](../prefix_sum/)一文）：

```cpp
// LC 560：和为 k 的子数组计数
unordered_map<long long, int> cnt;
cnt[0] = 1;                                 // 初始状态：空前缀
long long pre = 0;
int ans = 0;
for (int x : nums) {
    pre += x;                               // 当前累积状态
    ans += cnt.count(pre - k) ? cnt[pre - k] : 0;
    cnt[pre]++;
}
```

同族变形：

- 存**次数** → 数个数（LC 560、974）；
- 存**最早下标** → 求最长（LC 525、325）；
- 存**上次出现下标** → 滑动窗口左边界收缩（LC 3. 无重复字符的最长子串）；
- 存**前缀异或** → 区间异或配对（[LC 1442. 形成两个异或相等数组的三元组数目](https://leetcode.cn/problems/count-triplets-that-can-form-two-arrays-with-equal-xor/)）。

**识别信号**：题目问"子数组/子串满足……的个数或长度"，且条件能写成两个前缀状态的差/相等——先想这个模式。

---

## 6. C++ `unordered_map` 工程细节

这些细节在刷题里都是真实得分点：

### 6.1 平均 $O(1)$ 但最坏 $O(n)$

哈希冲突时单次操作退化到 $O(n)$。LeetCode 常规题不卡这个；但 Codeforces 等竞技平台**专门构造反哈希数据**卡 `std::hash`，此时需要自定义随机化哈希：

```cpp
struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }
    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED);
    }
};
unordered_map<long long, int, custom_hash> safe_mp;
```

### 6.2 `operator[]` 的副作用

`mp[key]` 在键不存在时会**插入默认值**。只读场景这是坑——会把垃圾键塞进表、污染 `size()`，语义上想表达"查"却改了结构。查询一律用：

```cpp
if (mp.count(key)) ...            // C++20: mp.contains(key)
auto it = mp.find(key);
if (it != mp.end()) ...
```

例外：分组场景（第 4 节）恰恰要利用自动建桶。

### 6.3 其他速记

- `mp.reserve(n)`：规模已知时预分配，避免 rehash，$10^5$ 级别有可感知加速；
- 遍历中 `erase` 当前迭代器会失效，用 `it = mp.erase(it)` 接住返回值；
- 键是自定义 struct 需同时提供 `operator==` 和哈希函数；
- 元素小范围非负整数时，`vector<int>` 当计数桶比哈希表快数倍——**能用数组别用哈希**（LC 560 若 $k$、元素范围小可用数组优化）。

---

## 7. 什么时候不该用哈希表

- 键是**小范围整数** → 直接数组，更快；
- 需要**有序遍历 / 前驱后继查询**（"比 x 小的最大数"）→ `map` / `set`（红黑树，$O(\log n)$）或排序 + 二分；
- 需要**最值** → 堆 / 单调队列（见[单调队列](../monotonic_queue/)）；
- 只需要**判存在、且宇宙很小** → 位图 `bitset`。

哈希表是"无序、等值查找"场景的最优解，出了这个舒适区就要换结构。

---

## 8. 题目索引

- [LeetCode 560. 和为 K 的子数组](../../LeetCode/LeetCode_560/) — 模式④（前缀和配对）的代表
- [LeetCode 76. 最小覆盖子串](../../LeetCode/LeetCode_76/) — 模式②（频次计数）的进阶：窗口计数增量维护 + 欠账计数 `less`
- [LeetCode 239. 滑动窗口最大值](../../LeetCode/LeetCode_239/) — 对照：问最值用单调队列，不用哈希
- 待补充：LC 1（模式①）、LC 76（模式②）、LC 49（模式③）、LC 3（模式④）刷到后持续更新
