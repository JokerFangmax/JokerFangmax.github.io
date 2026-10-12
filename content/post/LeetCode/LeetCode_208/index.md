---
title: "LeetCode 208: 实现 Trie (前缀树)"
date: 2026-11-22
math: true
tags:
  - LeetCode
  - 设计
  - 字典树
  - 哈希表
  - 字符串
categories:
  - LeetCode
description: "LeetCode 208 实现 Trie (前缀树)题解（字典树）"
---

# LeetCode 208. 实现 Trie (前缀树)

> 题目链接：[https://leetcode.cn/problems/implement-trie-prefix-tree/](https://leetcode.cn/problems/implement-trie-prefix-tree/)  
> 难度：中等  
> 标签：设计、字典树、哈希表、字符串

---

## 1. 题目描述

**Trie**（发音类似 try，又称**前缀树**或**字典树**）是一种树形数据结构，用于高效地存储和检索字符串集合。这棵树的每条边上有一个字符，从根走到某节点的路径拼出一个字符串，树中仅利用字符串的**公共前缀**来节约空间。

请你实现 Trie 类：

- `Trie()` 初始化前缀树对象；
- `void insert(String word)` 将字符串 `word` 插入前缀树；
- `boolean search(String word)` 若 `word` **完整地**在树中（不只是前缀），返回 `true`；
- `boolean startsWith(String prefix)` 若树中存在以 `prefix` 为**前缀**的字符串，返回 `true`。

**数据范围：**

- $1 \le word.length,\ prefix.length \le 2000$
- `word` 和 `prefix` 仅由小写英文字母组成
- `insert`、`search`、`startsWith` 总调用次数 $\le 3 \times 10^4$
- 保证调用 `search` 和 `startsWith` 时，至少已经插入过一个字符串

**示例：**

```
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");        // 返回 true：apple 完整存在
trie.search("app");          // 返回 false：只有 app 这个前缀，没有 "app" 这个词
trie.startsWith("app");      // 返回 true：apple 就是以 app 开头的
trie.insert("app");
trie.search("app");          // 返回 true：现在 "app" 是一个完整的词了
```

---

## 2. 思路分析

### 2.1 暴力：把字符串全塞进哈希表

最朴素的做法：`unordered_set<string>` 存所有插入的词。`search` 是 $O(1)$ 哈希查找，完美；但 `startsWith("ap")` 呢？你只能**遍历集合里每个字符串**，逐个检查前 $2$ 个字符是否为 `ap`——单次 $O(总字符数 \times L)$。调用 $3\times 10^4$ 次前缀查询时代价不可忽视，而且哈希表把所有词**整词打散**存储，公共前缀 `app...` 被重复存了无数遍，前缀信息完全丢失。

问题的本质：**查询词与前缀共享的是「前缀」这一结构信息，哈希表却只能整词比较**。需要一种让「公共前缀只存一份、且能沿前缀逐级导航」的结构。

### 2.2 前缀树：让公共前缀物理上共享

把词按字符逐级分叉地挂到树上——**从根到某节点的路径恰好拼出一个前缀**，相同的字符序列自动汇合到同一条路径。插入 `apple`、`app`、`april` 之后：

```
          (root)
            │ a
            ●
            │ p
            ●
           / \ p      ← 路径 "ap"：三个词共享的物理节点
          ●   │ r
    isEnd ✓ │ p       ← "app" 到此结束，打上结束标记
           ●│
            │ l
            ●
            │ e
            ● isEnd ✓  ← "apple"
```

两个关键设计：

1. **节点不存字符串，只存「下一字符的指针」**。走到节点 $u$ 时，走过的路径就是当前前缀，无需任何字符串比较。
2. **`isEnd` 结束标记**。`app` 是 `apple` 的真前缀，树中 `apple` 的路径必然**经过** `app` 对应的节点，所以「路径存在」和「这是一个词」是两回事，必须在每个完整词的终点打标记。

### 2.3 逐步走一遍

依次执行 `insert("apple")`、`insert("app")`、`search("app")`、`startsWith("apx")`：

| 操作 | 逐步过程 | 结果 |
|------|---------|------|
| insert "apple" | 根 $\to$ a $\to$ p $\to$ p $\to$ l $\to$ e，沿途不存在则**新建节点**，末节点 `isEnd = true` | 树深 5 |
| insert "app" | a、p、p 三个节点**全部已存在，直接复用**，只在末节点打 `isEnd = true` | 零新建 |
| search "app" | 沿 a、p、p 走到节点 $u$，检查 `u->isEnd` | `true` |
| startsWith "apx" | 沿 a、p 走通，查 `x` 时子指针为空 | `false` |

注意 `search("app")` 若没有 `isEnd` 机制就会误判——这是本题最大的陷阱（见第 5 节）。

### 2.4 子节点用数组还是哈希表？

- **26 长度数组 `Node* children[26]`**：查孩子 $O(1)$、缓存友好；代价是每个节点固定占 26 个指针（约 208 字节），字符集小且已知时是标准选择。
- **`unordered_map<char, Node*>`**：按需分配，稀疏时省内存，且天然支持任意字符集（汉字、URL 等）；代价是每次查询一次哈希 + 一次指针追逐，常数更大。

本题字母表固定为 26 个小写字母，取数组版。

---

## 3. 代码实现

```cpp
class Trie {
private:
    struct Node {
        Node* children[26] = {nullptr};   // 26 个小写字母的子指针
        bool isEnd = false;               // 根到本节点的路径是否为一个完整单词
    };
    Node* root;

    // 从根出发沿 s 走，返回终点节点；中途断链返回 nullptr
    Node* walk(const string& s) {
        Node* cur = root;
        for (char ch : s) {
            int idx = ch - 'a';
            if (!cur->children[idx]) return nullptr;   // 前缀不存在
            cur = cur->children[idx];
        }
        return cur;
    }

public:
    Trie() : root(new Node()) {}

    void insert(string word) {
        Node* cur = root;
        for (char ch : word) {
            int idx = ch - 'a';
            if (!cur->children[idx])           // 已有则复用（前缀共享）
                cur->children[idx] = new Node();
            cur = cur->children[idx];
        }
        cur->isEnd = true;                     // 单词终点打标记
    }

    bool search(string word) {
        Node* node = walk(word);
        return node != nullptr && node->isEnd; // 走到 + 是完整词，缺一不可
    }

    bool startsWith(string prefix) {
        return walk(prefix) != nullptr;        // 只要路径存在即可
    }
};
```

实现要点：

1. **`walk` 抽象出「沿字符串逐级下行」**：`search` 与 `startsWith` 的唯一区别是到达终点后是否检查 `isEnd`，抽出公共逻辑后两个函数各剩一行判断，重复代码归零。
2. **`insert` 是「不存在才新建」**：遇到已有前缀直接复用节点，这正是前缀树省空间的来源——`app` 与 `apple` 只让树多长一个节点（`l-e` 两级中的新增部分）。
3. **数组初始化 `= {nullptr}`** 值聚合初始化会自动补齐其余 25 个指针为 `nullptr`，避免读到野指针。
4. 本题不要求析构；工程代码中应递归释放或在节点池中分配（LeetCode 判题环境不检查内存泄漏）。

---

## 4. 复杂度分析

设 $L$ 为查询/插入串的长度，$\Sigma = 26$ 为字符集大小。

- **单次操作时间复杂度**：$O(L)$。无论 `insert`、`search` 还是 `startsWith`，都只是从根出发走 $L$ 层，每层做一次 $O(1)$ 的数组寻址。**与树里存了多少个词无关**——这是前缀树相对哈希表的本质优势（哈希表的 `search` 也是 $O(L)$，但 `startsWith` 退化）。
- **空间复杂度**：$O(\text{总字符数} \times \Sigma)$ 量级。最坏（无任何公共前缀）每字符独占一个节点，每节点恒占 26 个指针；最好情况（全部互为前缀）近似 $O(最长词长 \times \Sigma)$。哈希表版的空间则精确等于「去重后的边数」，稀疏时更省。

对比表：

| 结构 | insert | search | startsWith | 前缀共享 |
|------|--------|--------|------------|---------|
| 哈希集合 | $O(L)$ | $O(L)$ | $O(N \cdot L)$ | 无 |
| 前缀树（数组版） | $O(L)$ | $O(L)$ | $O(L)$ | 有 |
| 排序数组 + 二分 | $O(N \log N)$ | $O(\log N + L)$ | $O(\log N)$（定位后看相邻） | 无 |

---

## 5. 常见疑问

**疑问一：`search` 和 `startsWith` 到底差在哪？**

两者沿树的走法完全一样，差别只在**到达终点后的验收标准**：`search` 要求终点 `isEnd == true`（这个词被完整插入过），`startsWith` 只要求终点存在（树里**有词**以它开头，哪怕该词比前缀长）。反例：只插入 `"apple"` 后，`search("app")` 为 `false` 而 `startsWith("app")` 为 `true`——`app` 对应的节点存在（它是 `apple` 路径的中途站），但没有被打过结束标记。忘记 `isEnd`、或者把它误设在中途节点，是本类题最普遍的 bug。

**疑问二：为什么需要 `isEnd`，节点上有「父亲/孩子数量」信息不能替代吗？**

「路径经过我」和「我在某个词的终点」确实可以用「我有没有孩子」区分一部分：无孩子的节点一定是某些词的终点。但**既有孩子又是词终点**的节点（`app` 与 `apple` 并存时的 `app` 节点）只有 `isEnd` 才能表达。反过来也不能只用 `isEnd` 的存在与否替代路径判断：`search("ap")` 会走到 `app` 之前的那个节点，它没标记，正确返回 `false`——两种信息维度缺一不可。

**疑问三：先 `insert("apple")` 再 `insert("app")`，会新建节点吗？**

一个都不会。第二个 `insert` 沿 `a-p-p` 走时发现节点全部存在，全部复用，只把 `app` 终点节点的 `isEnd` 从 `false` 翻成 `true`。这正体现了前缀树「前缀物理共享」的设计：插入一个已有词的前缀，代价仅仅是翻转一个布尔位。

---

## 6. 相关题目与模板

**可复用模板**（前缀树节点与三种操作的骨架，改动 `children` 的类型即可适配任意字符集）：

```cpp
struct TrieNode {
    TrieNode* children[26] = {nullptr};
    bool isEnd = false;
};
// 通用口诀：insert 走到头打标记；search 走到头验标记；startsWith 走到头就返回。
```

**同类题目：**

- [LeetCode 211. 添加与搜索单词 - 数据结构设计](https://leetcode.cn/problems/design-add-and-search-words-data-structure/)：`search` 支持 `.` 通配符，DFS 枚举 26 个分支，前缀树 + 回溯
- [LeetCode 212. 单词搜索 II](https://leetcode.cn/problems/word-search-ii/)：回溯 board 时**边走边在 Trie 上同步下行**，走到 `isEnd` 才收集——前缀树做剪枝的经典综合题
- [LeetCode 648. 单词替换](https://leetcode.cn/problems/replace-words/)：词根表建 Trie，逐词沿树下行，**最先碰到的 `isEnd` 即最短词根**
- [LeetCode 677. 键值映射](https://leetcode.cn/problems/map-sum-pairs/)：节点上增加 `sum` 字段，插入时沿途累加，展示 Trie 节点「携带统计信息」的扩展方式
- [LeetCode 421. 数组中两个数的最大异或值](https://leetcode.cn/problems/maximum-xor-of-two-numbers-in-an-array/)：把整数按位建成 01-Trie，贪心走「相反分支」

**方法论总结**：判断是否该用 Trie 的三个信号——(1) 查询以「前缀」为单位；(2) 大量字符串共享前缀；(3) 需要沿字符串逐字符地做决策（通配、替换、贪心选位）。命中任意两条，前缀树大概率优于哈希与排序。

**一句话总结**：前缀树把「字符串集合」改写成「字符转移图」——公共前缀只存一份，任何以词为单位的懒惰都在 $O(L)$ 的逐字符下行中现形。
