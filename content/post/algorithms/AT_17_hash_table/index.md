---
title: "AT 17: 哈希表——从原理到防卡"
date: 2026-12-10
math: true
tags:
  - Notes
  - 算法模板
  - 哈希表
categories:
  - 算法模板
description: "DSA 刷题模板系列第 17 篇：手写拉链法/开放定址哈希表模板、unordered_map 防卡哈希（splitmix64 随机种子）、自定义键（pair/vector/lambda 注入），样题精讲洛谷 P1102 A-B 数对与 LC 1 两数之和"
---

# AT 17：哈希表——从原理到防卡

> 本篇是 [DSA 刷题模板系列](../AT_00_reading_guide/)第 17 篇，对应理论章 [DSA 9：词典](../../DSA/DSA_9/)。核心思想一句话：**把任意关键码压进 $O(1)$ 可查的桶数组——散列函数决定"压得好不好"，冲突策略决定"撞了怎么办"。**
> 分工说明：哈希表在题目里的**四大使用模式**（值查找/计数/分组/状态配对）在既有文章[《哈希表：从查找到计数的问题模式》](../hash_table/)里已经写透，本篇不重复；本篇写**武器本身**——什么时候手写、怎么防被卡、自定义键怎么注入。

---

## 1. 这个模板解决什么问题

三种场景需要你关心哈希表的**实现**而不是直接 `unordered_map` 一把梭：

| 场景 | 为什么 `unordered_map` 不够 | 用什么 |
|------|---------------------------|--------|
| 出题人构造数据卡 `std::hash` | 整数哈希是恒等映射，桶策略公开，同余数据把所有元素挤进一个桶，单次操作退化 $O(n)$ | 自定义哈希（splitmix64 + 随机种子） |
| 卡常（$10^6$ 级插入） | `unordered_map` 每次寻址带指针跳转、rehash 开销 | 手写拉链法（静态数组实现） |
| 键是 `pair`/`vector`/自定义结构 | 没有内置哈希 | hash 仿函数注入 |

理论依据在 [DSA 9](../../DSA/DSA_9/) §4–13：除余法的两个缺陷（不动点、相邻关键码映射相邻桶，§7.3）、装填因子 $\lambda = N/M$ 与冲突的关系（§5.2）、拉链法 vs 开放定址的取舍（§11–14）——本篇模板里每一行都能在那边找到出处。

---

## 2. 代码模板

### 2.1 手写拉链法（DSA 9 §11.2 的竞赛版）

DSA 课件用指针链表讲拉链法；竞赛里指针换**静态数组 + 数组下标当 next 指针**，避免动态分配：

```cpp
#include <cstring>
using namespace std;

const int M = 1000003;              // 桶数：大于数据规模的双素数（DSA 9 §7.2：素数不易与数据规律共振）
int head[M], nxt[N], key[N], val[N], tot = 0;
// head[h]：槽 h 上第一条边的编号（0 表示空，下标从 1 用起）
// nxt[e]：同一条冲突链上的下一条边；tot 为已用槽位数

int locate(int x) {                 // 散列函数：除余法（DSA 9 §7.1）
    return ((x % M) + M) % M;       // C++ 负数取模得负，先 +M 再 %M 归一到 [0, M)
}
int* findp(int x) {                 // 返回 key 对应 val 的指针；不存在返回 nullptr
    for (int e = head[locate(x)]; e; e = nxt[e])
        if (key[e] == x) return &val[e];
    return nullptr;
}
void insert(int x, int v) {         // 不存在则插入，存在则覆盖
    int* p = findp(x);
    if (p) { *p = v; return; }
    ++tot; key[tot] = x; val[tot] = v;
    nxt[tot] = head[locate(x)];     // 头插：新结点接在链头
    head[locate(x)] = tot;
}
```

三个要点：

- **下标从 1 用起**，让 `0` 可以充当空指针标记（`head[]` 初始化为 0 即可，不用 memset）；
- **`tot` 上限 $N$**（元素个数），数组只开元素规模，不开桶规模——这是拉链法对开放定址的空间优势；
- 除余法不动点、相关性两个缺陷（[DSA 9 §7.3](../../DSA/DSA_9/)）在竞赛数据下一般无感，真被卡再换 MAD。

<details>
<summary><b>变体：开放定址（线性试探）+ 懒惰删除</b></summary>

对应 [DSA 9](../../DSA/DSA_9/) §12–14：冲突时不挂链，沿试探链 $h, h+1, h+2, \dots \pmod M$ 找空桶。**删除不能真清空**——会把试探链拦腰截断，只能打懒惰标记（DSA 9 §14 的 `Bitmap removed` 思想）：

```cpp
const int M = 1000003;
int key[M]; bool used[M], rem_[M];

int* findp(int x) {
    for (int k = 0; k < M; k++) {
        int s = (locate(x) + k) % M;
        if (!used[s]) return nullptr;                      // 真空桶：试探链终止
        if (!rem_[s] && key[s] == x) return &key[s];       // 命中
    }                                                      // 懒删除桶或异 key：沿链继续
    return nullptr;
}
bool insert(int x) {
    if (findp(x)) return false;
    for (int k = 0; k < M; k++) {
        int s = (locate(x) + k) % M;
        if (!used[s] || rem_[s]) {                         // 空桶或已删桶可复用
            key[s] = x; used[s] = true; rem_[s] = false;
            return true;
        }
    }
    return false;                                          // 表满
}
void erase(int x) {
    for (int k = 0; k < M; k++) {
        int s = (locate(x) + k) % M;
        if (!used[s]) return;
        if (!rem_[s] && key[s] == x) { rem_[s] = true; return; }
    }
}
```

代价：数组要开到桶数 `M`（不是元素数），且装填因子必须压低（DSA 9 §5.2），否则 clustering 让试探链越走越长（§13.3）。竞赛里除非卡常卡到极致，一般用拉链版。

</details>

### 2.2 防卡哈希：`unordered_map` + splitmix64

被卡的原理：`std::hash<int>` 对整数是**恒等映射**，`unordered_map` 的桶数与取模策略是公开标准——出题人对每个桶构造一个同余 key，全部元素挤进同一条冲突链，总复杂度 $O(n^2)$。防法是**换一个出题人猜不到的散列函数**：

```cpp
#include <chrono>
#include <unordered_map>
struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {       // 常数时间的强混淆函数
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }
    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED);              // 随机种子：每次运行不同，赛前无法构造
    }
};
unordered_map<long long, int, custom_hash> mp;
```

两重防线：splitmix64 把恒等映射变成雪崩混淆（输出位对所有输入位敏感）；`FIXED` 随机种子让攻击者无法离线构造针对你这次运行的碰撞数据。工程细节（`reserve` 预分配、`operator[]` 副作用等）见[既有哈希表文 §6](../hash_table/)。

### 2.3 自定义键：pair / vector / lambda 注入

```cpp
struct PairHash {                                   // pair 键：拼成一个宽整数再散列
    size_t operator()(const pair<int,int>& p) const {
        return hash<long long>()(((long long)p.first << 32) ^ (unsigned)p.second);
    }
};
unordered_map<pair<int,int>, int, PairHash> mp1;

struct VecHash {                                    // vector 键：逐元素混合（boost::hash_combine 公式）
    size_t operator()(const vector<int>& v) const {
        size_t h = 0;
        for (int x : v) h ^= hash<int>()(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
unordered_map<vector<int>, int, VecHash> mp2;

auto lh = [](const string& s) { return hash<string>{}(s + "#salt"); };   // C++17 lambda 注入
unordered_map<string, int, decltype(lh)> mp3(10, lh);                    // 构造时传入 lambda
```

注意键类型还需要 `operator==`（`pair`/`vector`/`string` 自带；自定义 struct 要自己写）。

---

## 3. 常见变形与坑

| 变形 | 一句话做法 |
|------|-----------|
| 键是小范围非负整数 | 直接 `vector<int>` 计数桶，比任何哈希表快数倍（[hash_table 文 §6.3](../hash_table/)） |
| 字符串去重 | `unordered_set<string>`（内置 `hash<string>`）或手写多项式哈希（[AT 18](../AT_18_string_hash/)） |
| 需要有序遍历/前驱后继 | `map`/`set`（红黑树，$O(\log n)$），别用哈希 |
| 双射计数（A-B 数对） | 哈希计数后对每个键查 `key + C` |
| 卡常 | 手写拉链版（§2.1）+ `reserve` + 避免逐个 `operator[]` |

坑清单：

- **`operator[]` 会插入默认值**——只读查询用 `count`/`find`，否则垃圾键污染 `size()`；
- **负数取模**：`-7 % 3 == -1`（C++ 向零取整），手写 `locate` 必须 `((x % M) + M) % M`；
- **同 key 迭代中删除**：`it = mp.erase(it)` 接住返回值，别 `erase(it++)` 后继续用；
- **被卡不是玄学是工程**：`unordered_map` 平均 $O(1)$、最坏 $O(n)$ 是标准白纸黑字写的（DSA 9 §4.3 的 expected-$O(1) \ne O(1)$），竞技平台真的会出这种数据。

---

## 4. 样题精讲

### 4.1 洛谷 P1102 A-B 数对（计数模式的洛谷版）

> [P1102 A-B 数对](https://www.luogu.com.cn/problem/P1102)：给正整数数列与正整数 $C$，数出满足 $A - B = C$ 的数对个数（不同位置的相同数字算不同数对）。

**思路**：固定 $B$ 的值 $x$，需要的就是 $x + C$ 的出现次数——把"数对计数"改写成"单值计数 × 单值查询"。先一遍扫描建 `cnt`（值 → 次数），再对每个**不同的值** $k$ 累加 $cnt[k] \times cnt[k+C]$。按不同值聚合而不是按元素聚合，乘法一次算完，省掉对相同值的重复查询。

$N \le 2 \times 10^5$ 且值域可达 $10^9$（超出数组桶范围），必须哈希表；洛谷不卡 `std::hash`，但模板习惯性带上 `custom_hash` 有备无患：

```cpp
#include <cstdio>
#include <chrono>
#include <unordered_map>
using namespace std;

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

int main() {
    int n; long long c;
    scanf("%d %lld", &n, &c);
    unordered_map<long long, int, custom_hash> cnt;
    cnt.reserve(n * 2);                          // 预分配，避免 rehash
    for (int i = 0; i < n; i++) {
        long long x; scanf("%lld", &x);
        cnt[x]++;
    }
    long long ans = 0;
    for (auto& [k, v] : cnt) {                   // 题目保证 C 为正整数，k 与 k+C 必为不同键
        auto it = cnt.find(k + c);               // 只读查询用 find，不用 operator[]
        if (it != cnt.end()) ans += (long long)v * it->second;
    }
    printf("%lld\n", ans);
    return 0;
}
```

细节两处：`find` 而非 `[]`（§3 坑一）；若题目允许 $C = 0$，按元素聚合 `ans += cnt[a_i + C]` 会把元素和自己配对，需减 $n$ 或改为 $v \times (v-1)$——本题题面保证 $C$ 为正整数，无此边界。

复杂度：期望 $O(n)$，空间 $O(n)$。

### 4.2 LC 1 两数之和（值查找模式的原型）

> [LC 1. 两数之和](https://leetcode.cn/problems/two-sum/)（简单）：找 $a_i + a_j = target$ 的两个下标。

**思路**：对每个 $i$ 查询"前面是否出现过 $target - a_i$"——[hash_table 文 §2](../hash_table/) 的模式①原型，此处只点一个模板细节：**先查后存**，保证只和前面的元素配对，天然排除"自己配自己"（当 $target = 2a_i$ 时 `find` 不会命中未插入的 $a_i$ 自己）。完整题解见仓库 [LeetCode 1](../../LeetCode/LeetCode_1/)。

---

## 5. 题目清单

| 题号 | 题名 | 难度 | 考的变形 | 链接 |
|------|------|------|---------|------|
| 洛谷 P1102 | A-B 数对 | — | 计数模式 + 值域大必须哈希 | [题目](https://www.luogu.com.cn/problem/P1102) |
| 洛谷 P2580 | 于是他错误的点名开始了 | — | `unordered_set<string>` 查重 | [题目](https://www.luogu.com.cn/problem/P2580) |
| LC 1 | 两数之和 | 简单 | 值查找（先查后存） | [题解](../../LeetCode/LeetCode_1/) |
| LC 49 | 字母异位词分组 | 中等 | 分组模式（键 = 排序串） | [题解](../../LeetCode/LeetCode_49/) |
| LC 128 | 最长连续序列 | 中等 | 哈希集 + 只从无前驱处起跳 | [题解](../../LeetCode/LeetCode_128/) |
| LC 217 | 存在重复元素 | 简单 | 判重（`unordered_set`） | [题目](https://leetcode.cn/problems/contains-duplicate/) |
| LC 560 | 和为 K 的子数组 | 中等 | 状态配对（前缀和 × 哈希） | [题解](../../LeetCode/LeetCode_560/) |
| LC 41 | 缺失的第一个正数 | 困难 | 原地哈希（键即下标） | [题目](https://leetcode.cn/problems/first-missing-positive/) |

（LC 217/41 的链接沿用[既有哈希表文](../hash_table/)中已核对的 URL。）

---

## 6. 与其他篇的联系

- 理论：[DSA 9：词典](../../DSA/DSA_9/)——散列函数设计（§6–9）、冲突排解（§11–14）、懒惰删除（§14）全推导；
- 使用模式：[哈希表：从查找到计数的问题模式](../hash_table/)——四大模式 + `unordered_map` 工程细节，与本篇互补；
- 字符串键的进阶：[AT 18：字符串哈希](../AT_18_string_hash/)——把子串压成指纹，`unordered_set<ull>` 判重；
- 前缀和配对：[前缀和](../prefix_sum/)——LC 560 一族的理论根；
- 系列导航：[AT 00 导读](../AT_00_reading_guide/)。
