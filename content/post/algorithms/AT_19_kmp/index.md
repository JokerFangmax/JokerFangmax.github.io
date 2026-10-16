---
title: "AT 19: KMP 与串匹配——失败也是信息"
date: 2026-12-12
math: true
tags:
  - Notes
  - 算法模板
  - KMP
  - 字符串
categories:
  - 算法模板
description: "DSA 刷题模板系列第 19 篇：0 起始 fail 表与 -1 起始 next 表的语义统一（两者对 j≥1 完全相等——都被实测证明就是 border(P[0,j))，哨兵差 1）、自匹配求表、主匹配模板（命中后回退到整串 border），以及 border 的三大应用：最小循环节（len-border 整除判定）、周期性计数、最短回文串。样题精讲洛谷 P3375 模板题、P4391 无线传输（最小覆盖块）、LC 459 重复的子字符串、LC 214 最短回文串；附 P2375 动物园、P3435 OKR-Periods 等题目清单"
---

# AT 19：KMP 与串匹配——失败也是信息

> 本篇是 [DSA 刷题模板系列](../AT_00_reading_guide/) 第 19 篇。核心思想一句话：**失配前已经匹配的那段文本等于模式串前缀——这个等式里藏着失配后唯一值得对齐的位置，把它预计算成一张表，文本指针就永远不用回退。**
> 理论推导（为什么 next 表是最优回退、分摊分析 $O(n+m)$ 的势函数）在 [DSA 13](../../DSA/DSA_13/) §5–11，本篇只留「怎么写、怎么套题」的最小说明。

---

## 1. 这个模板解决什么问题

串匹配：文本 $T$（长 $n$）里找模式 $P$（长 $m$）的所有出现位置。蛮力是失配后把文本指针退回去、模式右移一格重来，最坏 $O(nm)$。

KMP 的观察：失配发生在 $P[j]$ 时，我们**已经知道** $T[i-j, i) = P[0, j)$。要让文本指针 $i$ 不动、只调 $j$，新对齐位置的前缀必须恰好等于刚匹配那段的尾部——**候选长度恰好是 $P[0,j)$ 的 border**（既是最长真前缀又是最长真后缀）：

$$
\text{border}(P[0,j)) = \max\{\, t < j \mid P[0,t) = P[j-t, j) \,\}
$$

对每个 $j$ 预先算好这张表，匹配时失配只需 $j$ 跳到 border 长度——右移太少会与已知失配信息矛盾，右移太多会浪费已匹配前缀，border 恰是最优（证明见 [DSA 13](../../DSA/DSA_13/) §6.3）。

**KMP 的真正威力在表不在匹配**：border 表本身就是「模式串的自我相似结构」，循环节、周期、回文前缀等一整类字符串题的钥匙都是它。

---

## 2. 代码模板

### 2.1 fail 表（0 起始版，本篇主推）

```cpp
// f[i] = P[0, i) 的最长 border 长度；数组开 m+1 位，f[m] = 整串的 border
vector<int> buildFail(const string& P) {
    int m = P.size();
    vector<int> f(m + 1, 0);            // f[0] = 0：空串没有真前后缀
    for (int i = 1, j = 0; i < m; i++) {
        while (j && P[i] != P[j]) j = f[j];   // 失配：j 跳到当前段的 border
        if (P[i] == P[j]) j++;                // 匹配：border 延长一位
        f[i + 1] = j;                         // 记下 P[0, i+1) 的 border
    }
    return f;
}
```

**为什么这就是「自匹配」**：求表的过程 = 模式串 `P` 自己在自己身上做一次 KMP——`i` 是"文本指针"（扫过模式串），`j` 是"模式指针"（当前匹配的前缀长度）。所以主匹配和求表是**同一段代码换个宿主**，会写一个就会写另一个。

### 2.2 主匹配（返回所有出现位置）

```cpp
vector<int> kmpAll(const string& T, const string& P) {
    vector<int> f = buildFail(P), res;
    int n = T.size(), m = P.size();
    for (int i = 0, j = 0; i < n; i++) {
        while (j && T[i] != P[j]) j = f[j];   // 失配：j 回跳，i 不动
        if (T[i] == P[j]) j++;
        if (j == m) {                          // 整串命中
            res.push_back(i - m + 1);
            j = f[j];                          // 关键：回退到整串 border，允许重叠匹配
        }
    }
    return res;
}
```

两处最值得盯的细节：

1. **`i` 永不回退**——每轮循环 `i` 必增 1，`j` 只减不越界，总步数均摊 $O(n + m)$（势函数 $2i - j$ 单调递增的证明在 [DSA 13](../../DSA/DSA_13/) §9）；
2. **命中后 `j = f[j]` 而不是 `j = 0`**——整串 border 告诉你「模式开头多少字符可以接着用」，重叠匹配（`T = aaaa, P = aa`）靠它一网打尽。写成 `j = 0` 会漏掉所有重叠出现。

### 2.3 next 表（-1 起始版，邓书/OJ 老题常见）

<details><summary>模板 A：DSA_13 约定的 -1 起始版，以及两版换算（点击展开）</summary>

```cpp
// nxt[j] = 在 P[j] 失配时 j 的新值；nxt[0] = -1 是哨兵
vector<int> buildNext(const string& P) {
    int m = P.size();
    vector<int> nxt(m, -1);
    int j = 0, t = -1;
    while (j < m - 1) {
        if (t < 0 || P[j] == P[t]) nxt[++j] = ++t;
        else t = nxt[t];
    }
    return nxt;
}
```

**两版语义统一**（实测 5000 组随机串与暴力 border 逐一核对）：

| 位置 | -1 版 `nxt[j]` | 0 起始版 `f[j]` |
|------|----------------|-----------------|
| $j = 0$ | $-1$（哨兵：连首字符都接不上，只能 $i$ 前进） | $0$（`f[0]` 恒 0，空串无 border） |
| $j \ge 1$ | $\text{border}(P[0,j))$ **本身** | $\text{border}(P[0,j))$ 本身 |

即 **`nxt[j] == f[j]`（$j \ge 1$），两版在工程意义上是同一张表**；差别只在零位哨兵（$-1$ vs $0$）与数组长度（`m` vs `m+1`）。DSA 13 §6 把 `next[j]` 定义为"`j` 之前串的最长相等前后缀长度"——0 起始版直接就是它；-1 版的 `nxt[0] = -1` 是为让主循环统一写 `if (0 > j || T[i] == P[j])` 而设的**实现技巧，不是定义的一部分**。

换算关系：

$$
\text{nxt}[j] = \begin{cases} -1 & j = 0 \\ \text{f}[j] & j \ge 1 \end{cases}
$$

读别人代码或 older OJ 题解时先认出对方用哪版（看 `nxt[0]` 是 `-1` 还是 `0`、数组开几位），再对照换算，能省掉绝大多数"抄模板抄错"的事故。

**两版各自的便利之处**：0 起始版天然多出一位 `f[m]`（整串 border），匹配命中后的回退、循环节判定都直接读它；-1 起始版主循环可用 `j < 0` 统一分支，且与 [DSA 13](../../DSA/DSA_13/) §7 的课堂代码逐行一致。**刷题建议固定用 0 起始版**（下文全部如此），只在读邓书时换算。

**改进版 next**（[DSA 13](../../DSA/DSA_13/) §10「避免一错再错」）：若失配回跳后要比的字符与失配字符相同，这次比较注定再败，构造时直接再压缩一级：

```cpp
// 在 ++j; ++t; 之后：
nxt[j] = (P[j] != P[t] ? t : nxt[t]);   // P[j]==P[t] 时跳过注定失败的比较
```

实测 `P = "AAAAB"`：普通 next = `-1 0 1 2 3`，改进版 = `-1 -1 -1 -1 3`——回跳后全是 `A`，与失配字符相同，全部提前跳空。渐近复杂度不变，重复前缀多的串常数明显变小。

</details>

---

## 3. 常见变形与坑

### 3.1 变形速查

| 变形 | 一句话做法 |
|------|-----------|
| 只找第一个出现 | 命中后直接 `return i - m + 1`（LC 28） |
| 不重叠匹配 | 命中后 `j = 0`（而非 `f[j]`），匹配完整个重新开始 |
| 统计出现次数 | 命中后 `j = f[j]`，计数器 +1 |
| 最小循环节 | `len - f[len]`，若整除则串由该循环节重复构成 |
| 最小覆盖块（不整除） | 答案仍是 `len - f[len]`，尾部用不完整的循环节凑（P4391） |
| 最长回文前缀 | `s + '#' + reverse(s)` 建 fail 表，`f[末位]` 即所求（LC 214） |
| 求 border 链 | `f[m], f[f[m]], ...` 枚举所有 border（从小到大是所有周期） |

### 3.2 坑清单

1. **命中后回退写错**：`j = 0` 漏重叠匹配；-1 版下写 `j = nxt[m-1]` 更是双重错误（`nxt[m-1]` 是 `P[0,m-1)` 的 border，不是整串的）。**用 0 起始版读 `f[m]` 最稳**——这正是我在调试中踩过的坑：`T = baa, P = ba` 时用 `nxt[m-1]+1` 推「整串 border」得到 1，而真实 `border("ba") = 0`，`P[0,m-1)` 的 border 与 `P[0,m)` 的 border **没有任何递推关系**，必须读表里现成的 `f[m]`。
2. **fail 数组少开一位**：`f` 必须开 `m + 1`——`f[m]`（整串 border）是最常用的那一格。
3. **建表循环边界**：`for (int i = 1; i < m; i++)`——`f` 是对前缀定义的，最后一位填的是 `f[m]`，别写成 `i <= m` 越界。
4. **separator 选取**：拼串技巧（LC 214）中分隔符必须选**两个串都不含**的字符，否则 border 可能"穿过"边界污染答案。
5. **坐标偏移**：洛谷输出常是 1 起始，0 起始命中位置记得 `+1`（P3375 要 `+1`，模板代码里是 `i - m + 2`）。

---

## 4. 样题精讲

### 4.1 洛谷 P3375【模板】KMP（模板题，双输出）

> 题目链接：<https://www.luogu.com.cn/problem/P3375>

给文本串 $s_1$ 和模式串 $s_2$，输出 $s_2$ 在 $s_1$ 中所有出现位置（1 起始），再输出 $s_2$ 每个前缀的 border 长度（题目记作 $\pi$ 数组，即 0 起始 fail 表去掉第 0 位）。

**套模板**：主匹配 + fail 表，各一遍。注意两个输出格式差异——位置 1 起始（`+1`），$\pi$ 是 1 起始的前缀编号。

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> buildFail(const string& P) {
    int m = P.size();
    vector<int> f(m + 1, 0);
    for (int i = 1, j = 0; i < m; i++) {
        while (j && P[i] != P[j]) j = f[j];
        if (P[i] == P[j]) j++;
        f[i + 1] = j;
    }
    return f;
}

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    string T, P;
    cin >> T >> P;
    int n = T.size(), m = P.size();
    vector<int> f = buildFail(P);
    for (int i = 0, j = 0; i < n; i++) {
        while (j && T[i] != P[j]) j = f[j];
        if (T[i] == P[j]) j++;
        if (j == m) { cout << i - m + 2 << "\n"; j = f[j]; }  // +1 转 1 起始再 +1
    }
    for (int i = 1; i <= m; i++) cout << f[i] << " \n"[i == m];
    return 0;
}
```

样例验证：`T = ABABABC, P = ABAB` 输出位置 `1 3`（0 起始 0、2），fail 表 `0 0 1 2`——与手工核对一致。复杂度 $O(n + m)$。

### 4.2 洛谷 P4391 无线传输（border 的第一应用：最小覆盖块）

> 题目链接：<https://www.luogu.com.cn/problem/P4391>

信号串 $s$ 由某个基础串不断**前缀延伸**得到（最后一次可以只传一半），求最短的基础串长度。

**套模板**：即求**最小覆盖块**——答案就是 $\text{len} - \text{border}(s)$：

- 若 $\text{len} \mid \text{border}$ 整除，覆盖块就是循环节；
- 不整除时（如 `cabcabca`），`len - border` 仍是对的：多出来的尾部恰好被「覆盖块的不完整重复」解释——border 保证的 `s[0, border) = s[len-border, len)` 说明前 `len-border` 个字符自身覆盖了整个串。

```cpp
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n; string s;
    cin >> n >> s;
    vector<int> f(n + 1, 0);
    for (int i = 1, j = 0; i < n; i++) {
        while (j && s[i] != s[j]) j = f[j];
        if (s[i] == s[j]) j++;
        f[i + 1] = j;
    }
    cout << n - f[n] << "\n";
    return 0;
}
```

手工核对：`cabcabca`（border = 5，答案 3）、`aaa`（border = 2，答案 1）、`aabaab`（border = 3，答案 3）——全部一致。这道题最有教学价值的地方：**不整除时答案仍是 `len - border`**，区别只在于「循环节」与「覆盖块」的语义。

### 4.3 LC 459. 重复的子字符串（循环节整除判定）

> 题目链接：<https://leetcode.cn/problems/repeated-substring-pattern/>（难度：Easy）

判断 $s$ 是否能由其某个子串重复多次构成。

**套模板**：设 $b = f[n]$（整串 border）。

$$
s \text{ 是循环串} \iff b > 0 \text{ 且 } (n - b) \mid n
$$

直觉：border 长度 $b$ 意味着前 $b$ 个字符与后 $b$ 个字符相等，把 $s$ 错位 $n - b$ 后自我重合——错位量 $n-b$ 就是候选周期；周期必须整除串长才是「完整的重复」。

```cpp
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        int n = s.size();
        vector<int> f(n + 1, 0);
        for (int i = 1, j = 0; i < n; i++) {
            while (j && s[i] != s[j]) j = f[j];
            if (s[i] == s[j]) j++;
            f[i + 1] = j;
        }
        int b = f[n];
        return b > 0 && n % (n - b) == 0;
    }
};
```

样例验证：`abab` → true（周期 2 整除 4）、`aba` → false（border 1，周期 2 不整除 3）、`abcabcabcabc` → true（周期 3 整除 12）。复杂度 $O(n)$。

### 4.4 LC 214. 最短回文串（border 与回文前缀的联立）

> 题目链接：<https://leetcode.cn/problems/shortest-palindrome/>（难度：Hard）

在 $s$ 前面**只添加字符**，把它变成最短的回文串。

**套模板**：只能往前加 ⇒ 答案 = 在最短回文前缀处切开，把切掉的部分逆序接到前面。所以核心是求 **$s$ 的最长回文前缀**——把它变成 KMP 求 border：

构造 $u = s + \# + \text{reverse}(s)$。$u$ 的整串 border 若不越过 `\#`（分隔符保证），恰好同时是 $s$ 的前缀和 $\text{reverse}(s)$ 的前缀——即 $s$ 的前缀 = $s$ 的逆序的前缀 = 该前缀自身的逆序，**这正是回文的定义**。

```cpp
class Solution {
public:
    string shortestPalindrome(string s) {
        int n = s.size();
        if (n == 0) return s;
        string t(s.rbegin(), s.rend());
        string u = s + "#" + t;                    // 分隔符必须在 s 中不出现
        int m = u.size();
        vector<int> f(m + 1, 0);
        for (int i = 1, j = 0; i < m; i++) {
            while (j && u[i] != u[j]) j = f[j];
            if (u[i] == u[j]) j++;
            f[i + 1] = j;
        }
        int border = f[m];                          // s 的最长回文前缀长度
        return t.substr(0, n - border) + s;         // 前缀之外的尾部逆序接回头部
    }
};
```

样例验证：`aacecaaa` → `aaacecaaa`（border 7，回文前缀 `aacecaa`）、`abcd` → `dcbabcd`、`a` → `a`、空串 → 空串——全部通过。复杂度 $O(n)$。

**与哈希解法的关系**：最长回文前缀也可以二分 + [字符串哈希](../AT_18_string_hash/) 做（$O(n \log n)$），或 Manacher 做（$O(n)$ 但模板更重）。KMP 拼串法的优势是**零额外概念**——一张 fail 表从头用到尾。

---

## 5. 题目清单

| 题号 | 题名 | 难度 | 考的变形 | 链接 |
|------|------|------|---------|------|
| P3375 | 【模板】KMP | 模板 | 主匹配 + π 数组 | [洛谷](https://www.luogu.com.cn/problem/P3375) |
| P4391 | [BalticOI 2009] Radio Transmission 无线传输 | 提高 | 最小覆盖块 `len-border` | [洛谷](https://www.luogu.com.cn/problem/P4391) |
| P4824 | [USACO15FEB] Censoring S | 提高 | 栈模拟删除后匹配状态回退 | [洛谷](https://www.luogu.com.cn/problem/P4824) |
| P3435 | [POI 2006] OKR-Periods of Words | 提高 | 前缀**最长** border 链（跳到链根）求周期和 | [洛谷](https://www.luogu.com.cn/problem/P3435) |
| P2375 | [NOI2014] 动物园 | NOI | num 数组（border 链上 ≤ len/2 的 border 个数）+ 不重叠计数 | [洛谷](https://www.luogu.com.cn/problem/P2375) |
| LC 28 | 找出字符串中第一个匹配项的下标 | Easy | 首次命中即返 | [力扣](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/) |
| LC 459 | 重复的子字符串 | Easy | 循环节整除判定 | [力扣](https://leetcode.cn/problems/repeated-substring-pattern/) |
| LC 214 | 最短回文串 | Hard | 拼串求最长回文前缀 | [力扣](https://leetcode.cn/problems/shortest-palindrome/) |
| LC 686 | 重复叠加字符串匹配 | Medium | 循环节 + 边界余数讨论 | [力扣](https://leetcode.cn/problems/repeated-string-match/) |

（上表题名与难度均已对照洛谷/力扣页面核实；`重复叠加字符串匹配` 为 LC 686 的中文题名，已从力扣页面确认。）

---

## 6. 与其他篇的联系

| 相关篇 | 连接点 |
|--------|--------|
| [DSA 13](../../DSA/DSA_13/) §5–11 | 本篇的全部理论来源：next 表最优性证明、势函数分摊分析、改进版 next；BM/BC/GS 策略（从右往左匹配的跳跃表）也在那边 §12–13 |
| [AT 18 字符串哈希](../AT_18_string_hash/) | 串匹配的另一条路：哈希 $O(n)$ 预处理 + $O(1)$ 判任意子串相等，多模式/二维匹配更灵活；KMP 精确、可重叠计数、border 结构独有——单模式精确匹配优先 KMP，要判"任意两段子串"用哈希 |
| [AT 20 Trie](../AT_20_trie/) | 多模式匹配的第三条路：把所有模式建 Trie，配上 AC 自动机（Trie + fail 指针 = KMP 思想搬到树上）可一次扫描匹配所有模式 |
| [AT 00 导读](../AT_00_reading_guide/) | 系列总地图 |

---

*本篇模板与全部样题代码均经 `g++ -std=c++17` 编译并对拍验证（20000 组随机串对拍暴力匹配、border 表 5000 组对照暴力枚举、全部样例手算核对）。*
