---
title: "LeetCode 207: 课程表"
date: 2026-11-21
math: true
tags:
  - LeetCode
  - 深度优先搜索
  - 广度优先搜索
  - 图
  - 拓扑排序
categories:
  - LeetCode
description: "LeetCode 207 课程表题解（拓扑排序）"
---

# LeetCode 207. 课程表

> 题目链接：[https://leetcode.cn/problems/course-schedule/](https://leetcode.cn/problems/course-schedule/)  
> 难度：中等  
> 标签：深度优先搜索、广度优先搜索、图、拓扑排序

---

## 1. 题目描述

你这个学期必须选修 `numCourses` 门课程，编号为 $0$ 到 $numCourses - 1$。数组 `prerequisites` 中每个元素 $[a, b]$ 表示**想修课程 $a$，必须先修完课程 $b$**。

请你判断是否可能修完所有课程——即这组「先修约束」是否存在矛盾（例如 $a$ 依赖 $b$、$b$ 又依赖 $a$，谁也开不了头）。可能返回 `true`，不可能返回 `false`。

**数据范围：**

- $1 \le numCourses \le 2000$
- $0 \le \text{prerequisites.length} \le 5000$
- $prerequisites[i].length = 2$
- $0 \le a, b < numCourses$，且 $a \neq b$（无自环）

**示例：**

```
输入：numCourses = 2, prerequisites = [[1,0],[0,1]]

依赖关系：1 -> 0（修 1 先修 0），0 -> 1（修 0 先修 1）
形成环 0 -> 1 -> 0，任何课程都无法开始。

输出：false
```

```
输入：numCourses = 3, prerequisites = [[1,0],[1,2],[0,1]]

1 依赖 0 和 2，0 又依赖 1：0 -> 1 之间成环。

输出：false
```

---

## 2. 思路分析

### 2.1 建模：这是一个有向图判环问题

把每门课看成节点，约束 $[a, b]$ 画一条有向边 $b \to a$（修完 $b$ 才能修 $a$）。一个合法的修课顺序，就是沿着边的方向依次输出节点的顺序——**拓扑排序 (topological sort)**。它存在当且仅当**有向图无环**，这样的图叫有向无环图 (DAG)。

为什么「无环」是充要条件？直观论证：

- **无环 $\Rightarrow$ 可排**：无环则必有入度为 0 的节点（没有前置课程的课），先修它、删掉它，剩下的图仍无环，归纳即可全部修完；
- **有环 $\Rightarrow$ 不可排**：环上每门课都在等另一门先修完，谁也不可能是第一个，死锁。

所以本题 = **判断有向图是否有环**。有两条标准路线，下面都推一遍。

### 2.2 路线一：BFS + 入度表（Kahn 算法）

刚才的归纳论证直接给出了算法——**反复摘「没有前置依赖」的课**：

1. 统计每门课的**入度** (in-degree)，即指向它的边数，可理解为「还剩几门前置课没修」；
2. 所有入度为 0 的课入队（现在就能修）；
3. 出队一门课，计数 $+1$，把它指向的每门课的入度减 1——若某门课入度降到 0，说明前置全部修完，入队；
4. 队列空后，若计数 $= numCourses$ 则无环。

用 `numCourses = 6`，边为 $0\to1,\ 0\to2,\ 1\to3,\ 2\to3,\ 3\to5,\ 4\to5$ 走一遍：

| 步骤 | 出队 | 剩余入度（0..5） | 新入队 | 已修 |
|------|------|------------------|--------|------|
| 初始 | — | [0, 1, 1, 2, 0, 2] | 0, 4 | {} |
| 1 | 0 | [-, 0, 1, 2, 0, 2] | 1 | {0} |
| 2 | 4 | [-, 0, 1, 2, -, 1] | — | {0,4} |
| 3 | 1 | [-, -, 0, 1, -, 1] | 2 | {0,4,1} |
| 4 | 2 | [-, -, -, 0, -, 1] | 3 | {0,4,1,2} |
| 5 | 3 | [-, -, -, -, -, 0] | 5 | {0,4,1,2,3} |
| 6 | 5 | 全部完成 | — | 6 门 |

计数到 6，返回 `true`。若图有环，环上节点的入度永远降不到 0，计数必然卡住——**环里的节点一个也出不了队**。

### 2.3 路线二：DFS 三色标记法

另一条路是对图做**深度优先搜索 (DFS)**，途中给节点染三种颜色：

- **白色**：未访问；
- **灰色**：在当前 DFS 递归栈上（访问了但还没回溯完）；
- **黑色**：已完成访问（它依赖的所有子树都走完了）。

**碰到灰色节点即发现环**：灰色意味着「从我出发正走着一条路径，现在又回到了我身上」，这条路径加上最后这条回边恰好构成环。黑色节点再次被碰到则直接跳过——它已被彻底检查过、确认安全，重复检查既浪费时间也可能误判（它不在当前栈上，不构成环）。

DFS 更适合递归直觉强的人，且是 210 题（课程表 II）里输出拓扑序的「逆后序」技巧的基础；Kahn 则天然就是拓扑序的生成过程，且是后续 210、802（ eventual-safe-states）、1136（平行课程）等的通用骨架。

---

## 3. 代码实现

以 Kahn 算法为主体（迭代、无递归深度顾虑，$numCourses \le 2000$ 时 DFS 栈深理论上也可能到 2000，两者都安全，但 Kahn 更稳）：

```cpp
class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> g(numCourses);       // 邻接表：b -> a 表示修完 b 才能修 a
        vector<int> indeg(numCourses, 0);        // 入度：还有几门前置没修
        for (auto& p : prerequisites) {
            g[p[1]].push_back(p[0]);             // [a, b] => 边 b -> a
            indeg[p[0]]++;
        }

        queue<int> q;
        for (int i = 0; i < numCourses; i++)
            if (indeg[i] == 0) q.push(i);        // 没有前置的课先入队

        int taken = 0;                           // 已成功修掉的课程数
        while (!q.empty()) {
            int u = q.front(); q.pop();
            taken++;
            for (int v : g[u]) {
                if (--indeg[v] == 0)             // v 的前置全部修完
                    q.push(v);
            }
        }
        return taken == numCourses;              // 卡住的差额就是环上的课程
    }
};
```

实现要点：

1. **边的方向别接反**：$[a, b]$ 是「修 $a$ 先修 $b$」，依赖箭头从 $b$ 指向 $a$。接反了 Kahn 会把「被依赖最多的课」先出队，逻辑完全颠倒。
2. **入队判据是 `--indeg[v] == 0`**——必须在减完之后判断，而不是减完再单独扫描一遍入度数组（后者也能对，但是 $O(V)$ 重复劳动）。
3. 不需要 `visited` 数组：入度降到 0 只可能发生一次，天然保证每个节点只入队一次。

判环逻辑与 DFS 三色版完全等价，三色版片段见第 6 节。

---

## 4. 复杂度分析

设节点数 $V = numCourses$，边数 $E = \text{prerequisites.length}$。

- **时间复杂度**：$O(V + E)$。建图扫一遍边；Kahn 中每个节点恰好入队出队一次，每条边恰好被「减入度」一次。DFS 同理：每个节点染一次色，每条边走一次。
- **空间复杂度**：$O(V + E)$。邻接表存所有边，入度数组与队列 $O(V)$（DFS 版为递归栈 $O(V)$ + 邻接表 $O(E)$）。

对比一个朴素想法「对每个节点做一次 DFS 看能否回到自己」，那是 $O(V(V+E))$——三色标记的价值在于**黑色节点免检**，每个节点只被完整检查一次。

---

## 5. 常见疑问

**疑问一：Kahn 里计数 `taken` 为什么能等价于判环？环到底卡在哪？**

环上每条边都指向环内节点，环外节点无论怎么减入度都减不到环上节点的入度归零——更准确地说，环上每个节点至少有 1 条入边来自环内，而那条边的起点永远出不了队。所以 `taken < numCourses` 的差额恰好等于**环及其下游**的课程数。这个性质在 802 题里被反过来用：求「不在任何环上、也不指向环」的安全节点。

**疑问二：有向图能用「无向图判环」的方法吗？**

不行。经典反例：$A \to B$ 与 $A \to C$、$B \to C$（菱形依赖）——转成无向图后有环（$A\text{-}B\text{-}C\text{-}A$），但有向图中毫无问题：先 $A$，再 $B$，再 $C$。无向图判环看的是「两点间是否存在第二条路径」，有向图判环要求**方向首尾相接**，必须回到 BFS 入度 / DFS 三色这类有向专用的方法。

**疑问三：数据范围里为什么强调 $a \neq b$？自环要特判吗？**

$[a, a]$ 意味着「修 $a$ 之前先修完 $a$」，一步死锁，是最小的环。本题数据范围替你排除了它；但在 802、210 等题或自己写图算法时，**自环会让 DFS 三色法在启动时就把起点染灰、随后立即撞灰**，正确返回有环；Kahn 则表现为该节点入度永不为 0，同样正确——两种方法对自环其实都是健壮的，但值得在面试时主动提一句。

---

## 6. 相关题目与模板

**DFS 三色判环模板**（`color`：0=白 1=灰 2=黑，遇灰即有环）：

```cpp
// 0=白 1=灰 2=黑；遇灰即有环
bool dfs(int u, vector<int>& color, vector<vector<int>>& g) {
    color[u] = 1;                                // 进栈：染灰
    for (int v : g[u]) {
        if (color[v] == 1) return true;          // 撞上栈上的自己 => 环
        if (color[v] == 0 && dfs(v, color, g)) return true;
    }
    color[u] = 2;                                // 出栈：染黑
    return false;
}
```

**Kahn 拓扑排序模板**（判环 / 输出顺序通用）：

```cpp
// Kahn 拓扑排序：order 即拓扑序；taken < n 说明有环
vector<vector<int>> g(n);
vector<int> indeg(n, 0);
for (auto& e : edges) { g[e[0]].push_back(e[1]); indeg[e[1]]++; }
queue<int> q;
for (int i = 0; i < n; i++) if (!indeg[i]) q.push(i);
vector<int> order;
while (!q.empty()) {
    int u = q.front(); q.pop();
    order.push_back(u);
    for (int v : g[u]) if (--indeg[v] == 0) q.push(v);
}
```

**同类题目：**

- [LeetCode 210. 课程表 II](https://leetcode.cn/problems/course-schedule-ii/)：本题加强版，输出任意一个拓扑序——Kahn 里把 `taken++` 换成 `order.push_back(u)` 即可
- [LeetCode 802. 找到最终的安全状态](https://leetcode.cn/problems/find-eventual-safe-states/)：三色标记法的直接应用，「黑色节点」就是安全节点
- [LeetCode 2115. 从给定原材料中找到所有可制作的菜肴](https://leetcode.cn/problems/find-all-possible-recipes-from-given-supplies/)：建图跑 Kahn，能出队的菜都能做
- [LeetCode 269. 火星词典](https://leetcode.cn/problems/alien-dictionary/)：拓扑排序的经典难题，需要先从字符串序列里「挖出」边

**方法论总结**：依赖关系类问题（课程、构建、任务调度）第一步永远是「画成有向图」；然后问自己两个问题——要判定可行性还是输出顺序？前者 DFS 三色与 Kahn 皆可，后者直接用 Kahn（它产出顺序是天然的）。环上节点入度永不归零、灰色节点重入即环，这两个不变量是所有变体题的钥匙。

**一句话总结**：拓扑排序就是「不断摘掉没有依赖的节点」——摘得完是 DAG，摘不完的部分就是环。
