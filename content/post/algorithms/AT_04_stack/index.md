---
title: "AT 04: 栈——从括号匹配到表达式求值"
date: 2026-11-27
math: true
tags:
  - Notes
  - 算法模板
  - 栈
  - 表达式求值
categories:
  - 算法模板
description: "栈的刷题模板全集：括号匹配、双栈中缀表达式求值（优先级表 + 弹栈结算规则）、逆波兰求值、最小栈的辅助栈模式、字符串解码的双栈套路；样题 LeetCode 20/227/394 与洛谷 P1981（边读边取模的环同态技巧）"
---

# AT 04：栈——从括号匹配到表达式求值

> **核心思想一句话：栈是"最近发生的事最先被处理"——凡是问题里出现严格的嵌套结构（括号、递归、区间套区间）或"暂存待结算"的流水线（运算符等待高优先级到齐），就该想到栈。**
> 理论后盾：[DSA 4：栈与队列](../../DSA/DSA_4/) §7–11（进制转换、括号匹配、栈混洗、中缀求值、逆波兰）。本篇不重复推导，只给武器。

---

## 1. 这个模板解决什么问题

刷题里栈扮演两个角色，认出角色就知道该套哪个模板：

| 角色 | 特征 | 典型题 | 模板 |
|------|------|--------|------|
| **回溯点记录器** | 输入有严格嵌套（`[...]` 套 `[...]`），处理完内层要回到外层 | 括号匹配、字符串解码、二叉树非递归遍历 | 遇"开"压栈、遇"闭"弹栈 |
| **待结算暂存区** | 元素到达时还不能定案，要等后续信息到齐才结算 | 表达式求值、[单调栈](../AT_05_monotonic_stack/) | 压栈前把"已能结算的"弹掉 |

表达式求值是两个角色的合体：数字等待运算符（暂存），括号界定子表达式（嵌套）。

---

## 2. 代码模板

### 2.1 括号匹配（回溯点记录器的最小标本）

```cpp
bool isValid(string s) {
    stack<char> st;
    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') st.push(c);   // 遇"开"压栈
        else {                                               // 遇"闭"必须对上栈顶
            char need = c == ')' ? '(' : c == ']' ? '[' : '{';
            if (st.empty() || st.top() != need) return false;
            st.pop();
        }
    }
    return st.empty();       // 坑：循环结束栈非空 = 左括号多了，也失配
}
```

不变量：**栈里存的是"已出现但还没闭合"的左括号**，按嵌套深度排列——栈顶永远是最内层。最后必须回到空栈，`"(("` 这类输入就死在 `return st.empty()` 上。

### 2.2 双栈中缀表达式求值（本篇主模板）

一个数字栈、一个运算符栈，规则只有两条：

> **规则一（结算）**：运算符 $c$ 入栈前，把栈顶所有**优先级 $\ge c$** 的运算符先算掉（同级的从左到右，所以 `>=` 而非 `>`）。
> **规则二（括号）**：`(` 直接入栈当挡板；`)` 到来时把栈里运算符结算到 `(` 为止，弹掉挡板。

```cpp
long long calc(stack<long long>& nums, stack<char>& ops) { // 弹两个数一个符，算完压回
    long long b = nums.top(); nums.pop();
    long long a = nums.top(); nums.pop();
    char op = ops.top(); ops.pop();
    return op == '+' ? a + b : op == '-' ? a - b : op == '*' ? a * b : a / b;
}

int pri(char c) { return (c == '+' || c == '-') ? 1 : 2; }   // 优先级表

long long evaluate(const string& s) {
    stack<long long> nums;
    stack<char> ops;
    for (size_t i = 0; i < s.size(); ) {
        if (isdigit(s[i])) {                                 // 多位数整体解析
            long long x = 0;
            while (i < s.size() && isdigit(s[i])) x = x * 10 + (s[i++] - '0');
            nums.push(x);
        } else if (s[i] == '(') {
            ops.push(s[i++]);                                // 挡板直接进
        } else if (s[i] == ')') {
            while (ops.top() != '(') nums.push(calc(nums, ops));
            ops.pop(); i++;                                  // 弹掉挡板
        } else if (s[i] != ' ') {
            while (!ops.empty() && ops.top() != '(' && pri(ops.top()) >= pri(s[i]))
                nums.push(calc(nums, ops));                  // 规则一：先结算再入栈
            ops.push(s[i++]);
        } else { i++; }
    }
    while (!ops.empty()) nums.push(calc(nums, ops));         // 清仓
    return nums.top();
}
```

为什么这是对的：**任何时刻，运算符栈从栈底到栈顶优先级严格递增**（规则一保证），所以栈顶永远是"最急迫等待结算"的运算符；数字栈顶两个数恰好是它的操作数。中缀转后缀（RPN）用的是同一套框架——只是"结算"改成"输出"。

### 2.3 逆波兰（后缀表达式）求值

RPN 天生无括号、无优先级，遇到运算符直接取数结算：

```cpp
int evalRPN(vector<string>& tokens) {
    stack<long long> st;
    for (auto& t : tokens) {
        // 坑：负数 token（如 "-11"）长度 > 1，不能见 '-' 就当运算符
        if (t.size() == 1 && string("+-*/").find(t[0]) != string::npos) {
            long long b = st.top(); st.pop();
            long long a = st.top(); st.pop();
            if (t == "+") st.push(a + b);
            else if (t == "-") st.push(a - b);
            else if (t == "*") st.push(a * b);
            else st.push(a / b);
        } else st.push(stoll(t));
    }
    return st.top();
}
```

### 2.4 辅助栈模式：最小栈

维护"当前状态下的最值"，再加一个**同步的辅助栈**：

```cpp
class MinStack {
    stack<int> st, mn;       // mn 与 st 等高：mn.top() = 当前栈内最小值
public:
    void push(int val) { st.push(val); if (mn.empty() || val <= mn.top()) mn.push(val); }
    void pop() { if (st.top() == mn.top()) mn.pop(); st.pop(); }
    int top()    { return st.top(); }
    int getMin() { return mn.top(); }
};
```

注意 `<=` 不是 `<`：push 两个相等的最小值要记两次，否则弹掉一个后 `getMin` 就丢了。这一模式在后文的 [Steap/Queap](../AT_06_queue_monotonic_queue/) 和[单调栈](../AT_05_monotonic_stack/)里反复出现——**"用第二个栈维护前缀最值"是通用手法**。

<details>
<summary><b>双栈当队：为什么放在栈篇提一句</b></summary>

用两个栈拼出队列（[DSA 4 §14](../../DSA/DSA_4/) 的经典摊还分析）：入队压 `in`，出队从 `out` 弹，`out` 空了把 `in` 整体倒进去。每个元素至多被压四次弹两次，摊还 $O(1)$。它是"栈的巧妙组合"，但接口是队列——完整代码与坑见 [AT 06：队列与单调队列](../AT_06_queue_monotonic_queue/)，此处不重复。
</details>

---

## 3. 常见变形与坑

| 变形 | 一句话做法 |
|------|-----------|
| 括号匹配 + 其他字符 | 只处理括号字符，其余跳过 |
| 带一元运算符（`-3`、`(-1+2)`） | LC 224 的难点：见 `-` 前无操作数时补 0（`0-3`）或压一个 0 进数字栈 |
| 表达式含取模（洛谷常见） | 每次 calc 后立刻取模；解析多位数时**边读边取模**（见样题 2） |
| 字符串解码 `3[a2[c]]` | 倍数栈 + 前缀串栈：`[` 处保存现场，`]` 处结算展开 |
| 栈混洗判定 | 模拟：按目标序列逐个弹，弹不出就压入下一个；压完还弹不出则 No（[DSA 4 §9](../../DSA/DSA_4/)） |

**坑清单**（按出现频率排序）：

1. **空栈 `top()`/`pop()`**：括号匹配遇右括号先查 `st.empty()`；表达式求值结算前保证数字栈有两个元素；
2. **括号匹配忘记查空栈收尾**：`"((("` 循环内全过，最后 `return st.empty()` 兜底；
3. **优先级用 `>` 而非 `>=`**：同级运算从左到右，`1-1+1` 会错算成 `1-(1+1)`；
4. **多位数逐字符处理**：解析数字必须 while 循环读完整个数，不是读一个字符；
5. **LC 150 的负数 token**：`"-11"` 是数字不是减号，用 `t.size() == 1` 区分；
6. **除法语义**：C++ 整除向零截断，`-3 / 2 = -1`（LC 150 专门卡这个）。

---

## 4. 样题精讲

### 4.1 LeetCode 20. 有效的括号（简单）

模板 §2.1 原样套用，三行说清：左括号压栈、右括号对栈顶、收尾查空栈。不赘述，**值得做的是想清楚为什么它必须用栈**——递归下降的匹配本质是 DFS 回溯，栈就是显式维护的 DFS 路径。题解：[LeetCode 20](../../LeetCode/LeetCode_20/)。

### 4.2 洛谷 P1981 [NOIP 2013 普及组] 表达式求值

> [题目链接](https://www.luogu.com.cn/problem/P1981)。只有 `+` 和 `*` 的中缀表达式（无括号），求值结果对 $10^4$ 取模。

双栈模板的洛谷版，两个考点：

**考点一：取模的正确姿势。** $(a \circ b) \bmod m = ((a \bmod m) \circ (b \bmod m)) \bmod m$ 对 $+,\times$ 成立（环同态），所以**每步结算都立刻取模**是安全的，不必担心溢出——$a, b < 10^4$ 时 $a \times b < 10^8$，`int` 装得下。

**考点二：单个数字本身可能很长。** 解析多位数若先读完整再取模，`long long` 也可能被超长数字撑爆。利用同一个环同态：**边读边取模**——$(x \times 10 + d) \bmod m$ 逐步算，任意长度都安全。

```cpp
#include <iostream>
#include <stack>
#include <string>
#include <cctype>
using namespace std;

int main() {
    string s;
    getline(cin, s);
    const int MOD = 10000;
    stack<int> nums;
    stack<char> ops;
    auto pri = [](char c) { return c == '*' ? 2 : 1; };
    auto calc = [&]() {
        int b = nums.top(); nums.pop();
        int a = nums.top(); nums.pop();
        char op = ops.top(); ops.pop();
        nums.push(op == '*' ? a * b % MOD : (a + b) % MOD);
    };
    for (size_t i = 0; i < s.size(); ) {
        if (isdigit(s[i])) {
            int x = 0;
            while (i < s.size() && isdigit(s[i]))
                x = (x * 10 + s[i++] - '0') % MOD;   // 边读边取模
            nums.push(x);
        } else if (s[i] != ' ') {
            while (!ops.empty() && pri(ops.top()) >= pri(s[i])) calc();
            ops.push(s[i++]);
        } else { i++; }
    }
    while (!ops.empty()) calc();
    cout << nums.top() % MOD << '\n';
    return 0;
}
```

自查样例：`1+1*3+4` → `8`；`4+2*3+10*2+2` → `32`；输入 `10000` → `0`（单个数字也要取模）。复杂度 $O(n)$。

### 4.3 LeetCode 394. 字符串解码（中等）

`3[a2[c]]` → `accaccacc`。嵌套结构 + 展开，是"回溯点记录器"的进阶版：**两个同步的栈保存进入每层前的现场**。

```cpp
string decodeString(string s) {
    stack<int> numSt;          // 进入本层前的重复次数
    stack<string> strSt;       // 进入本层前已拼好的前缀
    string cur; int num = 0;
    for (char c : s) {
        if (isdigit(c)) num = num * 10 + (c - '0');
        else if (c == '[') { numSt.push(num); strSt.push(cur); num = 0; cur.clear(); }
        else if (c == ']') {
            int k = numSt.top(); numSt.pop();
            string prev = strSt.top(); strSt.pop();
            for (int i = 0; i < k; i++) prev += cur;   // 展开 k 份内层结果
            cur = prev;
        } else cur += c;
    }
    return cur;
}
```

读法：`cur` 是当前层的半成品；`[` 是"保存现场、开新层"，`]` 是"结算本层、恢复现场"。与括号匹配的区别只是"闭括号"处要多做一次展开。题解：[LeetCode 394](../../LeetCode/LeetCode_394/)。

---

## 5. 题目清单

| 题号 | 题名 | 难度 | 考的变形 | 链接 |
|------|------|------|---------|------|
| LC 20 | 有效的括号 | 简单 | 括号匹配本体 | [题解](../../LeetCode/LeetCode_20/) |
| LC 155 | 最小栈 | 中等 | 辅助栈模式 | [题解](../../LeetCode/LeetCode_155/) |
| LC 150 | 逆波兰表达式求值 | 中等 | RPN 求值 + 负数 token 坑 | [题目](https://leetcode.cn/problems/evaluate-reverse-polish-notation/) |
| LC 227 | 基本计算器 II | 中等 | 双栈中缀求值（无括号版） | [题目](https://leetcode.cn/problems/basic-calculator-ii/) |
| LC 224 | 基本计算器 | 困难 | + 括号 + 一元运算符 | [题目](https://leetcode.cn/problems/basic-calculator/) |
| LC 394 | 字符串解码 | 中等 | 双栈现场保存 | [题解](../../LeetCode/LeetCode_394/) |
| 洛谷 P1739 | 表达式括号匹配 | 入门 | 括号匹配（注意结束符） | [题目](https://www.luogu.com.cn/problem/P1739) |
| 洛谷 P4387 | 验证栈序列 | 普及- | 栈混洗模拟判定 | [题目](https://www.luogu.com.cn/problem/P4387) |
| 洛谷 P1449 | 后缀表达式 | 普及- | RPN 求值完整程序 | [题目](https://www.luogu.com.cn/problem/P1449) |
| 洛谷 P1981 | [NOIP 2013 普及组] 表达式求值 | 普及- | 双栈 + 边读边取模 | [题目](https://www.luogu.com.cn/problem/P1981) |

---

## 6. 与其他篇的联系

- **理论**：[DSA 4](../../DSA/DSA_4/) §7–11——栈混洗的能力边界（哪些输出序列根本造不出来）是 §9 的精彩内容，P4387 就是它的编程版；
- **下一篇**：[AT 05：单调栈](../AT_05_monotonic_stack/)——把"压栈前弹掉能结算的"这一步从表达式求值里抽出来，专攻"下一个更大元素"类问题；
- **姊妹篇**：[AT 06：队列与单调队列](../AT_06_queue_monotonic_queue/)——双栈当队的完整实现与摊还分析在那里；
- **应用外延**：二叉树的非递归遍历（[AT 07](../AT_07_binary_tree/)）是"显式栈模拟递归"；DFS 本身就是系统调用栈上的递归（[AT 11](../AT_11_graph_bfs_dfs/)）；
- 全系列目录见 [AT 00：导读](../AT_00_reading_guide/)。
