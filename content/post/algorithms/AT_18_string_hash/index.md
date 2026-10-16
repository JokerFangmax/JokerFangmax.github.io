---
title: "AT 18: 字符串哈希——O(1) 子串判等与二分求 LCP"
date: 2026-12-11
math: true
tags:
  - Notes
  - 算法模板
  - 字符串哈希
categories:
  - 算法模板
description: "DSA 刷题模板系列第 18 篇：前缀 hash 预处理 + O(1) 子串指纹（h[r]-h[l-1]·p[r-l+1]），滚动哈希定长窗口，自然溢出 vs 单模数 vs 双哈希的取舍，二分 + hash 求 LCP / 最长重复子串，样题精讲洛谷 P3370 模板题、LC 1044 最长重复子串、LC 187 重复的 DNA 序列、LC 718 最长重复子数组"
---

# AT 18：字符串哈希——O(1) 子串判等与二分求 LCP

> 本篇是 [DSA 刷题模板系列](../AT_00_reading_guide/)第 18 篇，对应理论章 [DSA 13：串](../../DSA/DSA_13/)（§14–16 的 Karp–Rabin）。核心思想一句话：**把字符串看成一个 $B$ 进制大数、对大素数取模当指纹——前缀指纹预处理一次，任意子串的指纹 $O(1)$ 取出，"两子串相等"从 $O(len)$ 的逐字符比较变成一次整数比较。**
> 理论后盾：[DSA 13 §14–16](../../DSA/DSA_13/)——为什么不能直接维护精确大整数（指纹超过 64 位后比较不再是 $O(1)$，§14.2）、散列压缩只求快速筛选（§14.3）、rolling hash 的 $O(1)$ 窗口递推（§15）。本篇把这些理论落成三行核心公式和一份能背的模板。

---

## 1. 这个模板解决什么问题

字符串题里最贵的操作是**比较**：判 `s[l1..r1] == s[l2..r2]` 要逐字符扫，$O(len)$。而一大族题目（判重、找重复子串、求 LCP、回文判定）本质上只是在反复问"这两段一样吗"。字符串哈希把这族问题统一打下来：

| 场景 | 没有哈希时 | 有哈希后 | 代表题 |
|------|-----------|---------|--------|
| 串判重（$n$ 个串去重） | `set<string>`，比较 $O(len)$ | 每串算一个指纹，`set<ull>` 判重 | 洛谷 P3370 |
| 存在长度为 $L$ 的重复子串吗 | 滑窗 + 后缀结构 | 所有长度 $L$ 窗口的指纹进哈希表 | LC 187、LC 1044 |
| 两子串的 LCP | 逐字符比，$O(LCP)$ | **二分长度 + $O(1)$ 判等**，$O(\log n)$ | 本篇 §2.4 模板 |
| 回文判定（正反两遍哈希） | 中心扩展 $O(n)$/询问 | $O(1)$/询问 | 见 §3 变形表 |

与同章工具的分工：**单模式匹配问"P 在哪"用 [KMP](../AT_19_kmp/)（确定性、还送 border 性质）；"子串判等/多次比较"用哈希（更短、更通用、但有极小概率碰撞）**。两者经常同题可选——选哪个取决于你要不要匹配失败时 prefix-function 的副产物。

---

## 2. 代码模板

### 2.1 核心公式（先背这三行）

把 $s$ 看成 $B$ 进制数（$B$ 是大于字符集的基数，常用 131 或 137），所有运算模 $M$（$M$ 是大素数，或直接用 `unsigned long long` 自然溢出）。定义**前缀指纹**与**幂表**：

$$
h[i] = \big(h[i-1] \cdot B + s_i\big) \bmod M, \qquad p[i] = p[i-1] \cdot B \bmod M
$$

于是子串 $s[l..r]$（1 下标）的指纹是**前缀指纹的差**：

$$
\mathrm{hash}(s[l..r]) = h[r] - h[l-1] \cdot p[\,r-l+1\,]
$$

**为什么这成立**：$h[r]$ 是"前 $r$ 位的 $B$ 进制数"，$h[l-1] \cdot p[r-l+1]$ 恰好是"前 $l-1$ 位整体左移 $r-l+1$ 位"——相减就是把高位段清零，剩下的正好是第 $l$ 到 $r$ 位。它就是十进制里"1234567 取中间 345 = 1234567 − 1230000"的多进制版。

### 2.2 通用模板（1 下标，单模数）

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long ull;

const ull B  = 131;                // 基数：大于字符集
const ull M  = 1e18 + 9;           // 大素数（用自然溢出则去掉所有 % M，见 §3）
const int N  = 1e6 + 5;

ull h[N], p[N];                    // h[i]: 前 i 个字符的指纹; p[i]: B^i mod M

void build(const string& s) {      // 预处理 O(n)，之后任意子串 O(1)
    p[0] = 1;
    for (int i = 1; i <= (int)s.size(); i++) {
        h[i] = (h[i-1] * B + (ull)(unsigned char)s[i-1]) % M;
        p[i] = p[i-1] * B % M;
    }
}

ull sub(int l, int r) {            // s[l..r] 的指纹（1 下标，闭区间）
    return (h[r] + M * M - h[l-1] * p[r-l+1] % M) % M;
}
```

三个易错点都藏在 `sub` 的一行里：

1. **减法取模出负数**：`h[r] - x % M` 在 C++ 里可能是天文数字的负数，先加 $M^2$ 级的量再取模（`h[r] + M*M - ...`）。自然溢出版没有这个问题；
2. **`(unsigned char)` 强转**：字符直接进乘法前是带符号 `char`，遇到非 ASCII（负值）会污染模运算；
3. **下标体系全局统一**：模板用 1 下标（`h[0]=0` 是空串哨兵）。题目给的若是 0 下标，先在脑子里 +1，别混用。

### 2.3 定长窗口滚动（rolling hash）

若只要**固定长度 $L$** 的所有窗口指纹（DSA 13 §15 的原始形态），不必建前缀表，逐窗 $O(1)$ 递推：

$$
x_{\text{next}} = \big(x - s_{\text{out}} \cdot B^{L-1}\big) \cdot B + s_{\text{in}}
$$

去掉最高位贡献、整体乘 $B$、加新字符——与进制转换是同一件事（[DSA 4 §7](../../DSA/DSA_4/) 的进制转换栈）。LC 187 的解法（§4.3）就是这个形态。

### 2.4 二分 + hash 求 LCP

**子串判等只是 LCP 的特例**（判等 = 问 LCP 是否 $\ge len$），而 LCP 有单调性：若前 $k$ 个字符相同则前 $k-1$ 个也相同。于是 LCP 长度可二分，每次判定是一次指纹比较：

```cpp
// 前置：build(S) 已完成。bound = min(剩余长度, 关心的上界)
int lcp(int i, int j, int bound) {   // S[i..] 与 S[j..] 的最长公共前缀（0 下标）
    int lo = 0, hi = bound, ans = 0;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (sub_hash(i, i + mid - 1) == sub_hash(j, j + mid - 1)) {
            ans = mid; lo = mid + 1;          // 前 mid 个相同 → 试更长
        } else hi = mid - 1;                  // 不同 → 只可能更短
        }
    return ans;
}
```

这就是**哈希把"比较"变成"判定"**的完整闭环：LCP 从 $O(LCP)$ 降到 $O(\log n)$。后缀数组能干的很多事（最长重复子串、比大小排序）都能用"二分 + LCP"凑出来——常数大但代码只有十几行。

---

## 3. 常见变形与坑

### 3.1 模数选择：三档风险水平

| 方案 | 写法 | 碰撞概率 | 被卡风险 | 适用 |
|------|------|---------|---------|------|
| **自然溢出** | 全程 `ull`，不取模（模 $2^{64}$） | 形式上同双模数 | **能被专门构造数据卡**（Thue–Morse 串卡 $B$ 为 2 的幂的溢出哈希是经典攻击） | 面试/练习、题目不针对哈希 |
| **单大素数** | 模 $10^{18}+9$ 等 | 单次比较 $\approx len/M$，可忽略 | 出题人专门卡特定 $B,M$ 时会翻车 | 大多数竞赛题 |
| **双哈希** | 两个 $(B,M)$ 各算一遍，指纹为 pair | 两个独立指纹同时碰撞，概率平方级 | 几乎免疫（卡它等于同时猜中两组参数） | 确定性要求高/怀疑被针对 |

经验法则：**练习用自然溢出（最快写），正式赛用单素数，被人卡了就升双哈希**。双哈希的代价只是两倍的预处理和一个 `pair`。

<details>
<summary><b>为什么自然溢出能被卡：Thue–Morse 攻击一句话版</b></summary>

模 $2^{64}$ 时，若基数 $B$ 满足某些数论条件（如 $B$ 为 2 的幂，或 $B^{2^k} \equiv \pm 1$），存在长度 $O(2^k)$ 的字符串对 $(u, v)$ 使两者在模 $2^{64}$ 下**指纹恒等而内容不同**。Thue–Morse 串（递归地 $u \to uv,\ v \to u\bar v$）就是这样一对。出题人把它嵌进数据，你的哈希直接失明。防御：换奇素数 $B$、或升双哈希、或用随机化 $B$（`mt19937` 在 `[233, 1e9]` 里取）。

</details>

### 3.2 变形速查表

| 变形 | 一句话做法 |
|------|-----------|
| 二维/数组哈希（LC 718） | 同一公式，元素是 `a[i]+1`（**+1 防 0**：若允许 0，前导零不同长度指纹可能相撞，也把恒等元素削弱） |
| 回文判定 | 正向哈希 + 反向哈希各一遍；`s[l..r]` 回文 ⇔ 正指纹 = 反指纹 |
| 任意两子串比大小（字典序） | 二分 LCP，LCP 之外的第一字符定序 |
| 可持久化/多串 | 每串各 `build`，幂表全局共享一份 |
| 子串出现次数 | 长度固定的窗口指纹进 `unordered_map<ull,int>` 计数 |
| 禁止 64 位乘法溢出（Java 无 ull） | 用 $M < 2^{31}$ 的双模数代替 |

### 3.3 高频坑清单

1. **取模减法出负**（§2.2 已解，最常见的 WA 来源）；
2. **忘开 `(n+1)` 的表**：`h[0]` 是哨兵，循环到 `i <= n`，数组少开一位直接越界；
3. **基数小于字符集**：$B$ 必须**严格大于**最大字符值，否则不同串可能数值相同（如 $B=10$ 时 `"1"+"2"` 与 `"12"` 数值相同——进位撞车）；
4. **0 值元素不做偏移**：数组哈希里元素值可以是 0，`{0,1}` 与 `{1}` 的指纹会撞（前导零被吸收）——LC 718 的 `+1` 就是防这个；
5. **哈希相等直接断言串相等**：理论上有碰撞概率（DSA 13 §14.3：相等只是必要条件）。比赛里通常信单大素数/双哈希的指纹；要绝对确定，在指纹命中后再逐字符核对一次（只核对命中对，均摊代价极小）；
6. **多测不清空**：`h/p` 数组在多组数据间残留，`memset` 或按长度覆盖。

---

## 4. 样题精讲

### 4.1 洛谷 P3370【模板】字符串哈希

> [P3370【模板】字符串哈希](https://www.luogu.com.cn/problem/P3370)：给定 $n$ 个字符串，求其中**互不相同**的字符串个数。

**思路**：整串指纹 = 前缀表取到末位，`sort + unique` 去重。这是模板的最小完整应用（测试数据以题面为准；本题实测存在刻意构造的相同哈希值不同串数据，单自然溢出哈希可能 WA——官方数据就是用来卡哈希的，用双哈希或大素数才稳）。

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long ull;

const ull B = 131, M = 1e18 + 9;     // 大素数版；被卡就升双哈希

int main() {
    int n; cin >> n;
    vector<ull> fp(n);
    for (int i = 0; i < n; i++) {
        string s; cin >> s;
        ull x = 0;
        for (char c : s) x = (x * B + (ull)(unsigned char)c) % M;   // 整串滚动
        fp[i] = x;
    }
    sort(fp.begin(), fp.end());
    cout << unique(fp.begin(), fp.end()) - fp.begin() << "\n";
    return 0;
}
```

**复杂度**：$O(\sum|s_i|)$——每个字符只进一次乘加。对照：`set<string>` 的做法比较开销 $O(len \log n)$，长串下慢一个量级。

**本地验证**：`{abcaaaa, abc, abcc, 12345, abcdef}` → 5；`{abc×2, abd×2, 123×3, x}` → 4，均与 `python set` 手工核对一致。

### 4.2 LC 1044. 最长重复子串（Hard）——二分 + hash 的完全体

> [LC 1044. 最长重复子串](https://leetcode.cn/problems/longest-duplicate-substring/)：返回 `s` 中出现至少两次的最长子串（可重叠）。

**思路**：答案长度有单调性——若长度 $L$ 的重复子串存在，则长度 $L-1$ 的也存在（取它的前缀）。所以二分长度，`check(L)` 枚举所有长度 $L$ 的窗口指纹进哈希表查重。总复杂度 $O(n \log n)$。

```cpp
typedef unsigned long long ull;
const ull B1 = 131;                    // 第一指纹：自然溢出
const ull B2 = 137, M2 = 1000000007;   // 第二指纹：单模数 → 组合成双哈希

class Solution {
    string s; int n;
    vector<ull> h1, pw1, h2, pw2;

    pair<ull,ull> fp(int l, int len) { // s[l..l+len-1] 双指纹（0 下标）
        ull a = h1[l+len] - h1[l] * pw1[len];
        ull b = (h2[l+len] + M2 * M2 - h2[l] * pw2[len] % M2) % M2;
        return {a, b};
    }
    int tryLen(int len) {              // 长度 len 的重复子串存在则返回起点
        unordered_map<ull, vector<int>> bucket;   // 第一指纹分桶，桶内比第二指纹
        for (int l = 0; l + len <= n; l++) {
            auto pr = fp(l, len);
            for (int o : bucket[pr.first])
                if (fp(o, len) == pr) return o;
            bucket[pr.first].push_back(l);
        }
        return -1;
    }
public:
    string longestDupSubstring(string s_) {
        s = s_; n = s.size();
        h1.assign(n+1,0); pw1.assign(n+1,1);
        h2.assign(n+1,0); pw2.assign(n+1,1);
        for (int i = 0; i < n; i++) {
            h1[i+1]  = h1[i] * B1 + (ull)(unsigned char)s[i];
            pw1[i+1] = pw1[i] * B1;
            h2[i+1]  = (h2[i] * B2 + (ull)(unsigned char)s[i]) % M2;
            pw2[i+1] = pw2[i] * B2 % M2;
        }
        int lo = 1, hi = n - 1, start = -1, best = 0;
        while (lo <= hi) {             // 二分重复子串长度
            int mid = (lo + hi) / 2, pos = tryLen(mid);
            if (pos >= 0) { best = mid; start = pos; lo = mid + 1; }
            else hi = mid - 1;
        }
        return start >= 0 ? s.substr(start, best) : "";
    }
};
```

两个值得学的细节：**桶内比 pair**（分桶用第一指纹，命中后比完整 pair，省得给 `unordered_map` 写 pair 哈希）；**答案记起点**（二分 `check` 成功时顺手把 `pos` 存下，省最后一次重扫）。

**本地验证**：`banana`→`ana`、`abcd`→空、`aa`→`a`、`aaaaa`→`aaaa`，全部通过。

### 4.3 LC 187. 重复的 DNA 序列（Medium）——rolling hash 最小应用

> [LC 187. 重复的DNA序列](https://leetcode.cn/problems/repeated-dna-sequences/)：找 `s` 中所有出现超过一次的长度 10 的子串（字符集只有 ACGT）。

**思路**：字符集 4，窗口定长 10——2 进制两位编码一个字符，窗口指纹恰好塞进 20 bit，连取模都不需要。这是 [DSA 13 §15](../../DSA/DSA_13/) 滚动递推的最纯粹落地。

```cpp
class Solution {
    int val(char c) {
        switch (c) { case 'A': return 0; case 'C': return 1;
                     case 'G': return 2; default: return 3; }
    }
public:
    vector<string> findRepeatedDnaSequences(string s) {
        vector<string> ans;
        if (s.size() < 10) return ans;
        unordered_map<ull, int> cnt;
        ull h = 0;
        for (int i = 0; i < (int)s.size(); i++) {
            if (i >= 10) h -= (ull)val(s[i-10]) << (2*9);   // 摘最高位
            h = (h << 2) | val(s[i]);                        // 左移 + 进新字符
            if (i >= 9 && ++cnt[h] == 2) ans.push_back(s.substr(i-9, 10));
        }
        return ans;
    }
};
```

**注意 `++cnt[h] == 2`**：出现第二次才收，第三次不重复收（收的是"出现 ≥ 2 次"的**子串**，不是出现次数）。

**本地验证**：`AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT` → `{AAAAACCCCC, CCCCCAAAAA}`；`AAAAAAAAAAAAA` → `{AAAAAAAAAA}`，通过。

### 4.4 LC 718. 最长重复子数组（Medium）——数组哈希 + 判定二分

> [LC 718. 最长重复子数组](https://leetcode.cn/problems/maximum-length-of-repeated-subarray/)：两个整数数组，求最长公共**子数组**（连续）。

**思路**：DP 是 $O(nm)$；哈希 + 二分是 $O((n+m)\log \min(n,m))$——对长度二分，`check(L)` 把 `a` 的所有长度 $L$ 子数组指纹入桶、`b` 的子数组去查。**关键变形：元素 +1**（§3.3 坑 4，值可以为 0）。

```cpp
typedef unsigned long long ull;
const ull B = 131;

class Solution {
public:
    int findLength(vector<int>& a, vector<int>& b) {
        int n = a.size(), m = b.size();
        vector<ull> h(n+1,0), p(n+1,1), g(m+1,0), q(m+1,1);
        for (int i = 0; i < n; i++) { h[i+1]=h[i]*B+(ull)(a[i]+1); p[i+1]=p[i]*B; } // +1 防 0
        for (int i = 0; i < m; i++) { g[i+1]=g[i]*B+(ull)(b[i]+1); q[i+1]=q[i]*B; }
        auto fa = [&](int l, int len){ return h[l+len]-h[l]*p[len]; };
        auto fb = [&](int l, int len){ return g[l+len]-g[l]*q[len]; };
        int lo = 1, hi = min(n, m), best = 0;
        while (lo <= hi) {
            int len = (lo+hi)/2, found = 0;
            unordered_map<ull, vector<int>> bucket;
            for (int l = 0; l+len <= n; l++) bucket[fa(l,len)].push_back(l);
            for (int l = 0; l+len <= m && !found; l++)
                if (bucket.count(fb(l,len))) found = 1;      // 自然溢出下直接信指纹
            if (found) { best = len; lo = len+1; } else hi = len-1;
        }
        return best;
    }
};
```

**本地验证**：`{1,2,3,2,1} × {3,2,1,4,7}` → 3（`[3,2,1]`）；`{0,0,0,0,0} × {0,0,0,0,0}` → 5（全 0 数组正是 +1 偏移防住的场景），通过。

---

## 5. 题目清单

| 题号 | 题名 | 难度 | 考的变形 | 链接 |
|------|------|------|---------|------|
| 洛谷 P3370 | 【模板】字符串哈希 | 普及- | 整串指纹判重（注意官方数据卡单哈希） | [题目](https://www.luogu.com.cn/problem/P3370) |
| 洛谷 P3375 | 【模板】KMP | 普及- | 对照组：同一判等问题的确定性解 | [题目](https://www.luogu.com.cn/problem/P3375) |
| 洛谷 P4391 | [BalticOI 2009] 无线传输 | 普及+ | 最小循环节（哈希或 KMP 双解） | [题目](https://www.luogu.com.cn/problem/P4391) |
| 洛谷 P4503 | [CTSC2014] 企鹅 QQ | 提高+ | 单字符删除后的串判等（枚举删位 + 前后缀哈希拼接） | [题目](https://www.luogu.com.cn/problem/P4503) |
| 洛谷 P4824 | [USACO15FEB] Censoring S | 提高 | 栈 + 实时指纹（删串后指纹回退） | [题目](https://www.luogu.com.cn/problem/P4824) |
| LC 187 | 重复的 DNA 序列 | 中等 | 定长窗口滚动 | [题目](https://leetcode.cn/problems/repeated-dna-sequences/) |
| LC 718 | 最长重复子数组 | 中等 | 数组哈希 + 元素偏移 + 二分 | [题目](https://leetcode.cn/problems/maximum-length-of-repeated-subarray/) |
| LC 1044 | 最长重复子串 | 困难 | 双哈希 + 二分 + 桶内比 pair | [题目](https://leetcode.cn/problems/longest-duplicate-substring/) |
| LC 459 | 重复的子字符串 | 简单 | 判周期（哈希枚举周期长，KMP 更优） | [题目](https://leetcode.cn/problems/repeated-substring-pattern/) |
| LC 1143 | 最长公共子序列 | 中等 | 对照组：**子序列**用 DP，子串才轮到哈希 | [题解](../../LeetCode/LeetCode_1143/) |

（LC 459 的哈希解法在 $O(n \sqrt n)$ 量级、劣于 KMP 的 border 判定，列在这里是做工具对比；LC 1143 仓库题解里正好有一段"子序列 vs 子串"的辨析。）

---

## 6. 与其他篇的联系

- 理论：[DSA 13：串](../../DSA/DSA_13/) §14–16——指纹压缩的必要条件性、rolling hash 递推、字宽提醒；
- 同章工具：[AT 19（KMP 与串匹配）](../AT_19_kmp/)——确定性匹配 + border/循环节副产物；单模式匹配优先 KMP，多次子串判等/定长窗口计数优先哈希；[AT 20（Trie）](../AT_20_trie/)——前缀问题（"有哪些串以 p 开头"）用 Trie，判等问题用哈希；
- 哈希容器本身：[AT 17（哈希表）](../AT_17_hash_table/)——本篇的指纹往 `unordered_map`/`unordered_set` 里放，防卡哈希的 splitmix64 技巧在那篇；既有文[《哈希表：从查找到计数的问题模式》](../hash_table/)是四大使用模式总论；
- 二分骨架：[AT 02（二分查找与二分答案）](../AT_02_binary_search/)——本篇"二分长度 + check"正是二分答案形态在字符串上的应用；
- 系列导航：[AT 00 导读](../AT_00_reading_guide/)。
