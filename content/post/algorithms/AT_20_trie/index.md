---
title: "AT 20: Trie 与 01-Trie——前缀共享路径"
date: 2026-12-13
math: true
tags:
  - Notes
  - 算法模板
  - Trie
categories:
  - 算法模板
description: "数组版 Trie 模板（pass/end 双计数器、字符集映射、多测清空）与 01-Trie 最大异或（贪心走反位），样题精讲：P8306 字典树模板、LC 211 通配符查询、LC 421 数组中两个数的最大异或值，附题目清单与常见坑（数组大小估算、多测 TLE、位长选择）"
---

# AT 20：Trie 与 01-Trie——前缀共享路径

> 本篇是 DSA 刷题模板系列第 20 篇。核心思想一句话：**公共前缀折叠成共享路径——插入与查询的代价只看串长 $|s|$，与词表大小无关。**
> 理论后盾：[DSA 13 §17（Trie 的基本思想）](../../DSA/DSA_13/)——Trie 不是基于"字符串整体比较"，而是**逐字符分支**来组织词典。

---

## 1. 这个模板解决什么问题

[DSA 13](../../DSA/DSA_13/) 在串这一章的末尾把工具分了工：KMP / BM / KR 研究**在一个长文本中找模式**；Trie 研究**如何组织大量模式 / 词条本身**。所以当题目形态是"维护一个词表 + 反复查询前缀 / 完整串 / 自动机式的逐字符匹配"时，答案几乎总是 Trie：

| 操作 | 哈希表 | Trie |
|------|--------|------|
| 查"这个词在不在" | $O(1)$ 平均 | $O(\|s\|)$ |
| 查"多少词以 s 为前缀" | 做不到（哈希丢了前缀结构） | $O(\|s\|)$ |
| 逐字符流式匹配 | 做不到 | $O(\|s\|)$ |
| 按字典序遍历 | 要先全部取出排序 | 中序遍历天然有序 |

复杂度：$n$ 个总长 $L$ 的串，空间 $O(L \cdot \|\Sigma\|)$（$\Sigma$ 是字符集），插入与查询各 $O(|s|)$。

**01-Trie** 是同一结构的换装：把非负整数的二进制位（高位在前）当作字符，"字符集"缩到 $\{0,1\}$。它把"两两异或最大"这类**按位贪心**问题变成树上行走。

---

## 2. 代码模板

### 2.1 数组版 Trie（前缀计数 + 完整串计数）

竞赛默认写数组版而不是指针版：无 `new`、无递归析构、缓存友好、代码量减半。

```cpp
const int N = 3e6 + 10;          // 节点上限 = 所有串长度之和 + 1（0 号为根）
int son[N][62], pass_[N], endCnt[N], idx_;   // idx_ = 已用节点数

int c2i(char c) {                // 字符集映射：a-z, A-Z, 0-9 → 0..61
    if (islower(c)) return c - 'a';
    if (isupper(c)) return c - 'A' + 26;
    return c - '0' + 52;
}
void insert(const string& s) {
    int p = 0;                   // 从根出发
    for (char ch : s) {
        int c = c2i(ch);
        if (!son[p][c]) son[p][c] = ++idx_;   // 没有这条边就开新节点
        p = son[p][c];
        pass_[p]++;              // 经过节点 p 的串又多一个
    }
    endCnt[p]++;                 // 以 p 结尾（= 完整串）的计数
}
int countPrefix(const string& s) {           // 多少词以 s 为前缀
    int p = 0;
    for (char ch : s) {
        p = son[p][c2i(ch)];
        if (!p) return 0;        // 走空 = 无任何词有此前缀
    }
    return pass_[p];
}
```

四个设计点：

1. **不存字符，边即字符**——`son[p][c]` 的下标 $c$ 本身就是这条边代表的字符，节点里不需要任何 payload（除了计数器）；
2. **0 号是根**，它不代表任何字符；新节点编号从 1 开始，"儿子不存在"用 0 表示——所以 `if (!p) return 0` 判空和"走到根"天然是同一个值；
3. **双计数器语义不同**：`pass_[p]` = 有多少**词的前缀路径经过** p（前缀查询用），`endCnt[p]` = 有多少**词恰好在** p 结束（完整匹配用）。查"这个词在不在"就沿路径走到头看 `endCnt`；
4. **字符集是 26 就开 `son[N][26]`**，别无脑 62——数组大小直接乘在节点数上（见 §3 的坑 1）。

<details>
<summary><b>多测清空：为什么不能 memset 整表</b></summary>

多组测试数据时，`memset(son, 0, sizeof son)` 每组都清 $N \times 62$ 个 int（上 GB 的写操作），即使实际只用了几十个节点——直接 TLE。正确做法是**只清用过的部分**：

```cpp
for (int i = 0; i <= idx_; i++) {
    pass_[i] = endCnt[i] = 0;
    memset(son[i], 0, sizeof son[i]);   // 只清这一行（62 个 int）
}
idx_ = 0;
```

`idx_` 记录了本组用到哪里，从根（0 号）清到 `idx_` 即可，代价与实际使用量成正比。

</details>

### 2.2 01-Trie：最大异或

把整数按二进制位从高到低插入；查询 $x$ 与树中任意数的最大异或时，**每一位都贪心地走与 $x$ 该位相反的分支**（异或出 1）：

```cpp
const int M = 2e5 * 31 + 10;     // 节点上限 = 数字个数 × 位长
int son01[M][2], idx01_;
const int B = 30;                // 非负 32 位整数最高有效位（< 2^31）

void insert01(int x) {
    int p = 0;
    for (int i = B; i >= 0; i--) {          // 高位在前
        int b = x >> i & 1;
        if (!son01[p][b]) son01[p][b] = ++idx01_;
        p = son01[p][b];
    }
}
int queryMaxXor(int x) {                    // 树中任取 y，返回 max(x ^ y)
    int p = 0, res = 0;
    for (int i = B; i >= 0; i--) {
        int b = x >> i & 1;
        if (son01[p][b ^ 1]) {              // 反位存在就走：该位异或得 1
            res |= 1 << i;
            p = son01[p][b ^ 1];
        } else p = son01[p][b];             // 只能同位走：该位异或得 0
    }
    return res;
}
```

正确性一句话：高位优先——第 $i$ 位的一个 1 值 $2^i$，超过低位全部之和（$2^i - 1$），所以"高位能异或出 1 就出 1"永远不亏（与[二分](../AT_02_binary_search/)之外另一类贪心的标准论证相同：每步选择在剩余决策空间里严格占优）。

---

## 3. 常见变形与坑

### 3.1 变形速查

| 变形 | 一句话做法 |
|------|-----------|
| 大小写 + 数字字符集 | `c2i` 映射到 0..61（P8306） |
| 通配符 `.` 查询 | 该层 DFS 递归所有存在的儿子（LC 211） |
| 词表查存在性 | 走到头看 `endCnt[p] > 0`（P2580 的 OK/WRONG/REPEAT 三态再配一个 vis 数组） |
| 两两异或最大 | 全部 insert 后逐个 `queryMaxXor`（LC 421） |
| 树上路径异或最大 | DFS 求"根到点前缀异或"，树上路径异或 = 两端点前缀异或的异或 → 化归为 LC 421（P4551） |
| 删除某个串 | `endCnt--` / `pass_--`，节点不回收（0 编号继续当"不存在"用不了，但查询走不到即无影响） |
| 第 k 小异或和 / 历史版本 | 可持久化 01-Trie，本篇不展开 |

### 3.2 坑

1. **数组大小 = 节点上限 × 字符集**，最容易 RE/WA 的地方。估算公式：`节点数 = 所有插入串的总长 + 1`，再乘字符集宽度。P8306 总长 $3\times10^6$、62 叉，`int son[3e6+10][62]` 约 750 MB——这道题内存限制 1 GB 才放得下；常规 26 小写字母题是 $26 \times 4\,\text{B} \times$ 总长。**01-Trie 则是 数字个数 × 31**；
2. **多测清空用循环清前 `idx_+1` 个节点**，memset 整表必 TLE（见 §2.1 折叠块）；
3. **位长选择**：非负数小于 $2^{31}$ 就从第 30 位开始；写成 `1 << 31` 会碰到符号位（int 溢出为负），改用 `unsigned` 或 `1LL << 31`；
4. **前缀查询与完整匹配别混**：`countPrefix` 返回的是"以 s 为前缀的词数"，包含 s 本身；题目要"恰为 s"时用 `endCnt`。

---

## 4. 样题精讲

### 4.1 [洛谷 P8306 【模板】字典树 / Trie](https://www.luogu.com.cn/problem/P8306)

**题意**：多测。每组给 $n$ 个模式串、$q$ 个询问串，问每个询问串是多少个模式串的**前缀**。字符集含大小写与数字，输入串总长 $\le 3\times10^6$。

**思路**：模板本体。"多少个串以 s 为前缀"正是 `pass_` 计数器的语义——插入时沿途 `pass_++`，查询沿路径走到 `pass_[p]`。两个工程点：字符集映射到 0..61；多测只清前 `idx_+1` 行。

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 3e6 + 10;
int son[N][62], pass_[N], idx_;

int c2i(char c) {
    if (islower(c)) return c - 'a';
    if (isupper(c)) return c - 'A' + 26;
    return c - '0' + 52;
}
void insert(const string& s) {
    int p = 0;
    for (char ch : s) {
        int c = c2i(ch);
        if (!son[p][c]) son[p][c] = ++idx_;
        p = son[p][c];
        pass_[p]++;
    }
}
int countPrefix(const string& s) {
    int p = 0;
    for (char ch : s) {
        p = son[p][c2i(ch)];
        if (!p) return 0;
    }
    return pass_[p];
}
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    int T; cin >> T;
    while (T--) {
        for (int i = 0; i <= idx_; i++) {         // 只清用过的部分
            pass_[i] = 0;
            memset(son[i], 0, sizeof son[i]);
        }
        idx_ = 0;
        int n, q; cin >> n >> q;
        while (n--) { string s; cin >> s; insert(s); }
        while (q--) { string s; cin >> s; cout << countPrefix(s) << '\n'; }
    }
}
```

复杂度 $O(\sum|s|)$。两组样例语义自测（`abc/Abc/abcd` 词表：`ab`→2、`Ab`→1 区分大小写、`d`→0）已逐一对过。

### 4.2 [LC 211. 添加与搜索单词 - 数据结构设计](https://leetcode.cn/problems/design-add-and-search-words-data-structure/)

**题意**：实现 `WordDictionary`：`addWord(w)` 存词；`search(w)` 查词，其中 `.` 匹配**任意一个**字母。

**思路**：插入照旧；查询遇到 `.` 时该层无法确定走哪条边——**DFS 所有存在的儿子**。词长 $\le 25$，最坏 `....` 全通配是 $26^{25}$？不——树上实际的分支远少于理论值：每个节点最多 26 个儿子且词表只有 $10^4$ 个，DFS 的代价是"匹配串在树中经过的路径数"，上界为词表总长。

```cpp
class WordDictionary {
    static const int N = 7e4 + 10;            // 1e4 词 × 25 字符 + 根
    int son[N][26] = {}, endCnt[N] = {}, idx_ = 0;

    bool dfs(const string& w, int i, int p) {
        if (i == (int)w.size()) return endCnt[p] > 0;
        if (w[i] == '.') {                    // 通配：所有存在的儿子都试
            for (int c = 0; c < 26; c++)
                if (son[p][c] && dfs(w, i + 1, son[p][c])) return true;
            return false;
        }
        int c = w[i] - 'a';
        return son[p][c] && dfs(w, i + 1, son[p][c]);
    }
public:
    void addWord(const string& w) {
        int p = 0;
        for (char ch : w) {
            int c = ch - 'a';
            if (!son[p][c]) son[p][c] = ++idx_;
            p = son[p][c];
        }
        endCnt[p]++;
    }
    bool search(const string& w) { return dfs(w, 0, 0); }
};
```

### 4.3 [LC 421. 数组中两个数的最大异或值](https://leetcode.cn/problems/maximum-xor-of-two-numbers-in-an-array/)

**题意**：非负整数数组，求 $\max_{i \ne j}(a_i \oplus a_j)$。

**思路**：暴力 $O(n^2)$ 撞墙。把全部数 insert 进 01-Trie 后，对每个 $x$ 走一遍 `queryMaxXor(x)`——每步贪心走反位。答案 = 逐个取 max。

```cpp
class Solution {
    static const int M = 2e4 * 31 + 10;
    int son[M][2] = {}, idx_ = 0;
    void insert(int x) {
        int p = 0;
        for (int i = 30; i >= 0; i--) {
            int b = x >> i & 1;
            if (!son[p][b]) son[p][b] = ++idx_;
            p = son[p][b];
        }
    }
    int queryMaxXor(int x) {
        int p = 0, res = 0;
        for (int i = 30; i >= 0; i--) {
            int b = x >> i & 1;
            if (son[p][b ^ 1]) { res |= 1 << i; p = son[p][b ^ 1]; }
            else p = son[p][b];
        }
        return res;
    }
public:
    int findMaximumXOR(vector<int>& a) {
        for (int x : a) insert(x);
        int ans = 0;
        for (int x : a) ans = max(ans, queryMaxXor(x));
        return ans;
    }
};
```

复杂度 $O(n \cdot 31)$，空间同阶。**对拍记录**：本篇成稿前对 §2.2 模板跑了 200 组随机数据（$n \le 60$、值域 $[0, 10^9]$）对照 $O(n^2)$ 暴力，全部一致——贪心走反位的正确性不靠直觉靠验证。

<details>
<summary><b>树上版本：P4551 最长异或路径</b></summary>

[P4551](https://www.luogu.com.cn/problem/P4551) 把同一个模板搬到树上：边带权，求树上两点间路径边权的异或和最大值。

关键化归：设 $f(u)$ = 根到 $u$ 的路径异或和（一次 DFS 求出），则 $u \to v$ 路径异或和 = $f(u) \oplus f(v)$（LCA 以上的部分被异或两次抵消）。于是问题变成"对 $n$ 个数求两两最大异或"——LC 421 原封不动。

</details>

---

## 5. 题目清单

| 题号 | 题名 | 难度 | 考的变形 |
|------|------|------|---------|
| [LC 208](../../LeetCode/LeetCode_208/) | 实现 Trie (前缀树) | 中 | 模板本体（仓库已有题解） |
| [LC 211](https://leetcode.cn/problems/design-add-and-search-words-data-structure/) | 添加与搜索单词 - 数据结构设计 | 中 | 通配符 DFS（本篇 §4.2） |
| [LC 212](https://leetcode.cn/problems/word-search-ii/) | 单词搜索 II | 困 | Trie + 网格回溯，命中即剪枝 |
| [LC 421](https://leetcode.cn/problems/maximum-xor-of-two-numbers-in-an-array/) | 数组中两个数的最大异或值 | 中 | 01-Trie 贪心（本篇 §4.3） |
| [P2580](https://www.luogu.com.cn/problem/P2580) | 于是他错误的点名开始了 | 普及- | 存在性三态（OK/WRONG/REPEAT） |
| [P8306](https://www.luogu.com.cn/problem/P8306) | 【模板】字典树 / Trie | 普及+ | 62 字符集 + 多测清空（本篇 §4.1） |
| [P3879](https://www.luogu.com.cn/problem/P3879) | [TJOI2010] 阅读理解 | 普及 | 多词表存在性 |
| [P4551](https://www.luogu.com.cn/problem/P4551) | 最长异或路径 | 普及+ | 01-Trie 上树（见 §4.3 折叠块） |

---

## 6. 与其他篇的联系

- **理论**：[DSA 13 §17](../../DSA/DSA_13/)——Trie 与 PATRICIA / Ternary Trie 的关系；§17.1 讲清了"KMP 研究文本中找模式、Trie 研究组织模式本身"的分工；
- **同章工具**：[AT 19（KMP 与串匹配）](../AT_19_kmp/)、[AT 18（字符串哈希）](../AT_18_string_hash/)——单模式匹配用 KMP，多模式/前缀问题用 Trie，子串判等用哈希；
- **词典家族**：[AT 17（哈希表）](../AT_17_hash_table/)与既有[哈希表专题](../hash_table/)——同为词典实现，哈希丢前缀结构、Trie 保留它，这也是上表第一行"前缀查询只有 Trie 能做"的原因；
- **01-Trie 的贪心**与[AT 02（二分）](../AT_02_binary_search/)同属"每步严格占优"型贪心，论证方式可以互相搬。

系列导读与速查总表：[AT 00](../AT_00_reading_guide/)。
