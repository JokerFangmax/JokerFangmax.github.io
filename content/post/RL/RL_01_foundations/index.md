---
title: "RL 01: RL 到底在解决什么问题？"
date: 2026-09-25
math: true
tags:
  - RL
  - Notes
categories:
  - RL
description: "从 Bandit 到 MDP、Return、Value、Bellman：强化学习的公理系统"
---

# RL 01：RL 到底在解决什么问题？

> 📖 **直觉先行阅读**：本篇的直觉层在旧文 [强化学习到底在研究什么](../../RL_basic/) 中已有铺垫，建议先读。本篇在其之上补齐**数学定义、代码对应与手算例子**，是整个 RL 系列的公理系统——后面 21 篇的每一个公式都建立在本篇的记号上。
>
> 系列规划见 [RL_SERIES_PLAN](https://github.com/JokerFangmax/JokerFangmax.github.io)（Algorithm Family Tree 与 Master Table 持续更新）。

---

## 1. 这篇解决什么问题？

从一个具体问题开场：

> 你在教一个机器人开门。它面前有一扇门，可以选择"推、拉、转把手、什么都不做"。它没有任何示范、没有任何标注答案——它只能**试**。试了之后世界给它一个信号："成功了，+1"或者什么都没有。**它怎么从这种"只有结果反馈、没有过程指导"的信号里，学会一整套动作序列？**

这个问题有三个特征，是所有 RL 问题的公共结构：

1. **没有标准答案**：没人告诉它"正确动作是转把手"，只有事后 reward；
2. **反馈是延迟的**：可能要推 3 次、拉 1 次、再转把手才开门——最后那个 +1 该记在谁的头上？（credit assignment 问题）
3. **你的行为决定你看到什么数据**：它一旦只学会推门，就再也看不到"拉"的世界长什么样了。（exploration 问题）

监督学习完全处理不了这个结构：监督学习假设存在 $(x, y)$ 配对数据、标签正确、数据分布与行为无关。本篇的任务就是把这三个特征**数学化**——从最简单的 Bandit 开始，逐步加约束，最终得到 MDP + Return + Value + Bellman 这套公理。后面所有算法（DQN、PPO、SAC、GRPO……）都是"在这套公理下，怎么把 value 或 policy 学出来"的不同回答。

---

## 2. Previous Algorithm：监督学习

本篇没有前代 RL 算法，用你已熟悉的**监督学习**作为对照基线，按系列的六要素格式整理（这个格式之后每篇都会出现）：

| 要素 | 监督学习 | RL（本篇将建立的） |
|------|---------|-------------------|
| **Input** | 独立同分布的样本 $x$ | 智能体自己的经历 $(s, a, r, s')$ |
| **Model** | $f_\theta: x \mapsto y$ | policy $\pi: s \mapsto a$（和/或 value 函数） |
| **Action** | 不存在——只预测不动手 | **核心对象**：改变环境状态 |
| **Environment** | 不存在——数据是静态的 | 有状态、会转移、会反馈 |
| **Reward** | 不存在——损失直接定义在预测上 | 稀疏、延迟、只评价结果 |
| **Update** | 梯度下降最小化损失 | 还不知道（这正是后面 20 篇的内容） |

核心瓶颈：监督学习的损失是**每一步都有正确答案**的；开门问题里我们没有"每一步的正确答案"，只有"整段经历好不好"。**整个 RL 理论就是在回答：没有逐步标签时，学习信号从哪来。**

---

## 3. Core Idea

一句话总结本篇建立的形式体系：

> **RL 把"试错中学决策"形式化为：在马尔可夫决策过程（MDP）中，找一个策略 $\pi$，最大化期望折扣回报 $E[\sum \gamma^t r_t]$；而 Bellman 方程把这个"整段未来"的优化，拆成了"当前一步 + 剩余价值"的自洽方程——它是后续所有算法的引擎。**

只读这句话你应该带走三个词：**MDP（问题的形式）、Return（目标的形式）、Bellman（求解的引擎）**。

---

## 4. Mathematics

### 4.1 热身：Multi-Armed Bandit——砍掉"状态"的最小 RL

$k$ 台老虎机，第 $i$ 台以未知分布吐 reward（均值 $\mu_i$）。每一轮选一台，只能看到那一台的 reward。目标：最大化累计收益。

数学上在最小化**遗憾（regret）**：

$$
\text{Regret}(T) = T \mu^* - \mathbb{E}\left[\sum_{t=1}^{T} \mu_{a_t}\right], \qquad \mu^* = \max_i \mu_i
$$

- $a_t$：第 $t$ 轮选的机器（**这就是"动作"概念的雏形**）；
- $\mu^* - \mu_{a_t}$：这一轮"没选最优"付出的代价。

Bandit 的意义：**它是唯一能干净地研究 exploration vs exploitation 的模型**——选已知最好的机器（exploit）还是试试没试过的（explore）？遗憾恰好度量了"探索不足 + 探索过度"的总代价。它的致命简化是**没有状态**：今天选哪台机器不影响明天机器的输出。真实世界不是这样——所以我们要加回状态。

### 4.2 MDP：把"世界会变"数学化

马尔可夫决策过程（Markov Decision Process）是一个五元组：

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

逐项解释，**每一项都对应开门问题里的一个实体**：

| 符号 | 含义 | 开门问题中 |
|------|------|-----------|
| $\mathcal{S}$ | 状态空间 | （门的角度，手的位置，机器人姿态） |
| $\mathcal{A}$ | 动作空间 | {推, 拉, 转把手, 不动} |
| $P(s'\mid s, a)$ | **转移函数**：在 $s$ 做 $a$ 后到 $s'$ 的概率 | 门被推后开合多少（可能打滑→概率） |
| $R(s, a)$（或 $r(s,a,s')$） | 奖励函数 | 门开了 +1，能耗 −0.01 |
| $\gamma \in [0, 1)$ | 折扣因子 | 眼前收益 vs 长远收益的权衡 |

**马尔可夫性（Markov property）是整个体系的最强假设**：

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots, s_0, a_0) = P(s_{t+1} \mid s_t, a_t)
$$

即"未来只依赖现在，不依赖历史"。这不是环境的恩赐，而是**你对状态的设计要求**：如果门有"惯性"（上一秒在合上），而你的 $s$ 里没有速度信息，马尔可夫性就被破坏了。**记住这句话，它在 Blog 12（DeepMimic 的 state 设计）和 Blog 18（LLM 的 prompt=state）里都会回来。**

### 4.3 Return：目标的形式

轨迹（trajectory）是智能体与环境交互产生的历史：

$$
\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots)
$$

**回报（Return）** 是从时刻 $t$ 起的累计折扣奖励：

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k \, r_{t+k+1}
$$

为什么要折扣 $\gamma$？三个理由，重要性递增：

1. **数学上**：无限求和收敛。若每步 reward 恒为 +1 且 $\gamma = 1$，$G_t = \infty$，比较两条策略变成了比较两个无穷大。取 $\gamma < 1$ 后 $G_t \le \frac{r_{\max}}{1-\gamma}$，有限；
2. **建模上**：现实偏好"早拿到"的收益（钱、进展）；
3. **算法上（最重要）**：$\gamma$ 控制**有效视野**（effective horizon）$\frac{1}{1-\gamma}$。$\gamma=0.9$ 时视野约 10 步，$\gamma=0.99$ 时约 100 步——它直接决定了"多远的奖励值得为它规划"，是后面所有算法里最敏感的超参之一（PPO/SAC 调参时 $\gamma$ 永远在名单上）。

**为什么必须用"整段的和"而不是"当步 reward"作为目标？** 因为延迟反馈：开门的 +1 出现在第 4 步，但"走到门前"的第 1 步同样有功劳。Return 是把功劳**装进一个可优化的标量**的容器——后面所有的 credit assignment（Advantage、GAE、GRPO 的组内归一）都是在拆这个容器。

### 4.4 Policy、Value、Q：给"决策"和"局势"打分

**策略（policy）** $\pi(a \mid s)$：状态到动作的映射。可以是随机的（概率分布）或确定的（$a = \mu(s)$）。RL 的**最终产品**永远是一个 policy。

**状态价值函数（state-value function）**——策略 $\pi$ 下从 $s$ 出发的期望回报：

$$
V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid s_t = s\right]
$$

**动作价值函数（action-value function）**——多指定第一步动作：

$$
Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid s_t = s, a_t = a\right]
$$

两者的关系（**本系列出现频率最高的恒等式之一**）：

$$
V^\pi(s) = \sum_{a} \pi(a \mid s)\, Q^\pi(s, a)
$$

即 $V$ 是 $Q$ 对 policy 的期望。直觉：$V(s)$ 是"这个局势整体值多少分"，$Q(s,a)$ 是"这个局势下**先走这一步**值多少分"。

为什么不只用 $V$？因为在 $s$ 处想选出好动作，你需要比较**不同动作**的后果——$V(s)$ 对所有动作取了平均，把差别抹掉了；$Q(s,a)$ 保留了动作维度，$\arg\max_a Q(s,a)$ 就能直接选动作。**这是 Value-based 线（Blog 03–05）全部建立在 $Q$ 而不是 $V$ 上的根本原因。**

**优势函数（Advantage）** 在此先挂个号（Blog 07 的主角）：

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

即"这一步比该局势的平均水平好多少"。

### 4.5 Bellman 方程：把"整段未来"拆成"一步 + 剩余"

对 Return 做一次"提出第一项"的代数操作：

$$
G_t = r_{t+1} + \gamma\left(r_{t+2} + \gamma r_{t+3} + \cdots\right) = r_{t+1} + \gamma G_{t+1}
$$

**这就是 Bellman 方程的全部来源**——它不是新假设，只是 Return 的递归定义。把它两边取期望，得到**Bellman 期望方程**：

$$
V^\pi(s) = \sum_{a} \pi(a\mid s)\sum_{s'} P(s'\mid s,a)\left[R(s,a,s') + \gamma V^\pi(s')\right]
$$

$$
Q^\pi(s,a) = \sum_{s'} P(s'\mid s,a)\left[R(s,a,s') + \gamma \sum_{a'} \pi(a'\mid s') Q^\pi(s',a')\right]
$$

读法（以 $V$ 为例）："局势 $s$ 的价值 = 我按 $\pi$ 选动作 → 环境按 $P$ 转移 → 拿一步奖励 + $\gamma \times$ 新局势的价值"。**"当前一步 + 打折的未来"** ——价值信息可以从后往前一步步传播。

**最优版本**：把"按 $\pi$ 的期望"换成"取最好"，得 **Bellman 最优方程**：

$$
V^*(s) = \max_a \sum_{s'} P(s'\mid s,a)\left[R(s,a,s') + \gamma V^*(s')\right]
$$

$$
Q^*(s,a) = \sum_{s'} P(s'\mid s,a)\left[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')\right]
$$

两个方程的区别就是后面两条技术路线的分水岭：

- **期望方程**描述"评估一个给定策略"（$\pi$ 已知）→ Policy Iteration 的评估步、Actor-Critic 的 Critic；
- **最优方程**描述"直接逼近最优"（max 操作）→ Value Iteration、Q-Learning 的 `max` 、DQN 的 target。**Blog 03 会看到，正是这个 `max` 埋下了过估计（overestimation）的种子。**

**为什么说 Bellman 是"引擎"？** 因为它把一个"无限步的优化问题"变成了"相邻两个时刻价值之间的自洽条件"。之后所有算法的更新目标——TD target、Q-Learning target、soft Bellman——全是这两个方程在不同条件下的变体（去掉 $P$、去掉 max、加 entropy……）。**看懂一个新 RL 算法最快的方式，就是找到它的 Bellman 方程改了哪一项。**

---

## 5. Algorithm Pipeline

本篇还没有算法，先建立全系列共用的**交互回路**（interaction loop）：

```
        ┌──────────────── agent（要学的东西）────────────────┐
        │                                                    │
        │   policy π(a|s)        [Blog 06+ 是神经网络]        │
        │        ▲                                            │
        │        │ 选出动作 a                                  │
        └────────┼────────────────────────────────────────────┘
                 │
                 ▼
        ┌─── environment（MDP 的 P 和 R）───┐
        │      执行 a，转移 s → s'          │
        │      产生 reward r                │
        └───────────────────────────────────┘
                 │
                 ▼
        (s', r) 回到 agent，进入下一轮
```

标注本篇引入的变量属性（之后每篇都会更新这张表）：

| 变量/模块 | 来源 | 本篇状态 |
|-----------|------|---------|
| $s, s'$ | **environment 产生** | 外生变量 |
| $r$ | **environment 产生**（由 $R$ 决定） | 外生变量 |
| $a$ | policy 产生 | 本篇是抽象映射，还没有参数 |
| $\pi, V, Q$ | agent 内部 | 本篇是数学对象，还没有"学习"机制 |
| gradient / stop-gradient | — | 本篇尚无（Blog 04 起出现） |
| replay buffer / rollout 时机 | — | 本篇尚无（Blog 04 起出现） |

**rollout 与 update 的分离**是后面所有篇的关键概念：本篇的回路是纯 rollout（只交互不学习）；从 Blog 03 起，"什么时候用交互数据更新参数"开始出现不同答案——这些不同答案正是 on-policy / off-policy 分野的来源。

---

## 6. Pseudocode

本篇唯一能写的"算法"是 Bandit 的 ε-greedy——它是后面一切探索机制（ε-greedy、entropy、温度）的曾祖父：

```python
# ε-greedy for k-armed bandit
Q = [0.0] * k          # 每台机器的估计价值（action-value 的最简形态）
N = [0] * k            # 每台机器被拉的次数
epsilon = 0.1

for t in range(T):
    if random() < epsilon:
        a = randint(k)                  # 探索：随便选一台
    else:
        a = argmax(Q)                   # 利用：选当前认为最好的

    r = bandit(a)                       # 环境返回 reward（只有这一台的真实值）

    # 增量式更新均值 —— 对应公式 Q(a) ← Q(a) + α [r − Q(a)]
    N[a] += 1
    Q[a] = Q[a] + (r - Q[a]) / N[a]     # 步长 1/N(a) 即样本均值；换成常数 α 即指数滑动平均
```

**公式 ↔ 代码对应**（这个对应习惯贯穿全系列）：

| 数学 | 代码 |
|------|------|
| $Q(a) = \mathbb{E}[r \mid a]$ | `Q[a]` |
| $Q_{n+1}(a) = Q_n(a) + \frac{1}{N}\left[r_n - Q_n(a)\right]$（增量均值） | `Q[a] += (r - Q[a]) / N[a]` |
| $\epsilon$-greedy：以 $\epsilon$ 概率均匀探索 | `if random() < epsilon` |

注意更新式 `Q[a] + (r - Q[a]) * α` 的形状——**"估计 + 步长 × (目标 − 估计)"**。这个形状在 Blog 02 会原样出现在 TD 更新里，在 Blog 04 成为 DQN 的损失，认出它是 RL 的肌肉记忆。

---

## 7. Toy Example

### 7.1 Bandit：手算一次 ε-greedy 更新

两台老虎机，真实均值 $\mu_1 = 0.4$、$\mu_2 = 0.6$（agent 不知道）。初始估计 $Q_1 = Q_2 = 0$，$N_1 = N_2 = 0$。

第 1 轮（ε 触发，随机选中 1 号），拉出 $r = 1$：

$$
N_1 = 1,\qquad Q_1 \leftarrow 0 + \frac{1}{1}(1 - 0) = 1
$$

第 2 轮（贪心：$Q_1 = 1 > Q_2 = 0$，选 1 号），拉出 $r = 0$：

$$
N_1 = 2,\qquad Q_1 \leftarrow 1 + \frac{1}{2}(0 - 1) = 0.5
$$

第 3 轮（ε 触发，选中 2 号），拉出 $r = 1$：

$$
N_2 = 1,\qquad Q_2 \leftarrow 0 + 1 \times (1-0) = 1
$$

此刻 $Q = [0.5, 1]$，贪心会转向 2 号——**正确的方向**（$\mu_2 > \mu_1$）。注意风险：如果第 1 轮 1 号拉出的是 0，$Q = [0, 1]$ 前它可能长期被冷落——**ε-greedy 的遗憾正来自这种早期的运气**。

### 7.2 MDP：手算一个两状态世界的 V 和 Bellman 自洽

状态 $s_1$（门前）、$s_2$（门开，终止，$V(s_2)=0$）。动作与转移：

- $a = \text{推门}$：$r = 0$，回到 $s_1$（门没开，白费力气）；
- $a = \text{拉门}$：$r = +1$，转移到 $s_2$（开门！）。

取 $\gamma = 0.9$。

**第一步：固定策略求 $V^\pi$。** 设 $\pi(\text{拉} \mid s_1) = p$。由 Bellman 期望方程：

$$
V^\pi(s_1) = p\left[1 + 0.9 \cdot V(s_2)\right] + (1-p)\left[0 + 0.9 \cdot V(s_1)\right] = p + 0.9(1-p)\,V^\pi(s_1)
$$

这是关于 $V^\pi(s_1)$ 的一元一次方程（**Bellman 方程在固定 $\pi$ 下是线性的**——Blog 02 的 Policy Evaluation 就是解这个线性系统）：

$$
V^\pi(s_1) = \frac{p}{1 - 0.9(1-p)}
$$

- $p = 1$（永远拉）：$V^\pi(s_1) = 1$，一次开门，游戏结束；
- $p = 0$（永远推）：$V^\pi(s_1) = 0$，永远白费；
- $p = 0.5$：$V^\pi(s_1) = \frac{0.5}{1-0.45} = \frac{10}{11} \approx 0.909$。

**第二步：求 $Q^*$ 与最优策略。** Bellman 最优方程：

$$
Q^*(s_1, \text{推}) = 0 + 0.9 \max\left[Q^*(s_1,\text{推}),\ Q^*(s_1,\text{拉})\right],\qquad
Q^*(s_1, \text{拉}) = 1 + 0.9 \times 0 = 1
$$

设 $M = \max[\cdot]$，则 $Q^*(\text{推}) = 0.9M$，$Q^*(\text{拉}) = 1$。若 $M = Q^*(\text{拉}) = 1$：$Q^*(\text{推}) = 0.9 < 1$ ✅ 自洽。所以：

$$
V^*(s_1) = 1,\qquad \pi^*(\text{拉}\mid s_1) = 1
$$

最优策略就是永远拉门。**注意我们没用任何"学习"——是直接解方程。这正是 Blog 02 动态规划（DP）的做法：知道 $P$ 和 $R$ 时，最优性可以算出来；不知道时才需要 MC/TD，进而需要整个现代 RL。**

### 7.3 γ 的威力：把 reward 改成"每步活着 +1"

若把上面的环境改成：拉门后不终止，而是进入一个"每步 +1"的房间（即 $V(s_2)$ 不再是 0）。此时若 $\gamma \to 1$，$V(s_2) = \frac{1}{1-\gamma} \to \infty$；$\gamma = 0.9$ 时 $V(s_2) = 10$，有限且明确。**同一个环境，$\gamma$ 决定了"无限期苟着"值多少钱**——这是 γ 作为超参的直观感受。

---

## 8. Why It Works

从统计与优化角度，这套公理系统为什么成立：

1. **Markov 性 = 状态的充分统计量**。$P(s'\mid s,a)$ 与历史无关，意味着 $s$ 是预测未来的**充分统计量**——所有历史信息都应被压缩进 $s$。这不是环境自带的性质，而是状态设计的任务：**状态设计得好，Markov 近似成立，所有理论适用；设计得差（漏掉关键变量），问题变成 POMDP（部分可观测），一切难一个量级。** 研究实践中"state 加上速度/历史/参考帧后性能暴涨"的例子俯拾皆是——DeepMimic 的 state 设计（Blog 12）就是教科书案例。

2. **Return 解决 credit assignment 的"容器"问题**。单步 reward 无法归因，但 Return 是一条轨迹的标量总分。它不直接告诉你"哪一步的功劳"——那是 Advantage/GAE 的工作——但它给了优化一个**唯一明确的目标**。没有这个容器，连"哪个 policy 更好"都无法定义。

3. **γ 同时扮演三个角色**（数学收敛、时间偏好、视野控制），所以它才如此敏感。值得记住的换算：$\gamma = 0.99$ 的有效视野 $\approx 100$ 步—— humanoid 控制里一秒 30 帧、三秒的任务就需要 $\gamma \ge 0.99$，这不是玄学而是换算。

4. **Bellman 方程 = 把无穷维优化变成不动点问题**。原始目标 $\max_\pi \mathbb{E}[G]$ 是在轨迹空间上的优化；Bellman 方程说 $V^*$ 是一个算子的**不动点**（contraction mapping，Blog 02 会证），于是可以**迭代逼近**——DP、TD、DQN 全部是不同形式的不动点迭代。这是"为什么 RL 能学"的数学根基。

5. **Bandit 的遗憾理论给出 exploration 的下界直觉**：任何算法的总遗憾至少 $\Omega(\sqrt{T})$（Lai–Ro Robbins 下界）——探索的代价是**不可避免的**，好的算法（UCB、Thompson Sampling）只是把常数做小。这解释了为什么后面每个算法都要重新回答探索问题（ε-greedy → entropy → 最大熵 RL）。

---

## 9. Failure Modes

这套形式系统在什么情况下失效/不适用：

| 失效场景 | 具体表现 | 现实中的样子 | 系列中的呼应 |
|---------|---------|-------------|-------------|
| **非马尔可夫** | 状态遗漏关键变量，$P(s'\mid s,a)$ 实际依赖历史 | 相机看不到门的速度 → POMDP | Blog 12：state 设计是 humanoid 课题的一半 |
| **Reward 设计错误** | 优化的不是你想要的 | agent 学会堵住摄像头"看不见失败"；LLM 谄媚（sycophancy） | Blog 18 KL penalty、Blog 21 reward hacking |
| **γ 选错** | 太小：短视（放弃长线收益）；太大：学习信号方差爆炸、逼近发散 | 走迷宫的 agent 只顾眼前格子 | Blog 02：MC/TD 的方差随视野增长 |
| **状态/动作空间连续且高维** | 表格式的 $V, Q$ 存不下 | 机器人 30 个自由度 | Blog 04：函数逼近（DQN）的动机 |
| **环境非平稳** | $P, R$ 随时间变 | 和人对手下棋 | 本系列假设平稳 MDP（多智能体是另一门课） |
| **reward 稀疏到不可能探索到** | 正样本从未被采样到 | 开门 100 步精确序列 | Blog 10 熵探索、Blog 12 模仿奖励、Blog 16 想象 |

**核心认识：本篇的公理不是"世界的样子"，而是"问题的接口"。** 每个后续算法的 failure mode，本质上都是"它对这个世界接口的假设在某处不成立"——这个视角贯穿 Blog 22 的调试清单。

---

## 10. 与前后算法对比

本篇是起点，对照表以"问题形式"为主题：

| | 监督学习 | Bandit | 完整 RL（MDP） |
|---|---------|--------|---------------|
| 数据来源 | 静态标注集 | 自己交互（无状态） | 自己交互（有状态） |
| 核心对象 | $f_\theta$ | $Q(a)$ | $\pi(a\mid s)$, $V^\pi$, $Q^\pi$ |
| 反馈形式 | 每步正确标签 | 即时 reward | 延迟 reward 序列 |
| credit assignment | 不需要（标签直接对应输入） | 平凡（reward 即时归因） | **核心难题**（Return/Bellman 承担） |
| 行为影响数据 | 否 | 影响（探索） | 影响（探索 + 转移） |
| 可用优化工具 | 梯度下降直接可用 | 增量均值 | **尚无**——正是本系列主线 |

**链条位置**：

```
监督学习（对照基线）
   ↓  引入"动作改变世界"
Bandit（无状态的试错）
   ↓  引入"状态与转移" → MDP
Return / V / Q / Bellman（本篇：定义了目标与自洽条件）
   ↓  下一个问题：不知道 P 和 R 时，怎么解 Bellman 方程？
Blog 02：DP（知道模型）→ MC（完全不想知道模型）→ TD（折中）
```

**留给下一篇的问题**：本篇 7.2 里我们是**解方程**求出 $V^\pi$ 的——前提是 $P$ 和 $R$ 完全已知。真实世界（仿真器、现实机器人、LLM 对话）里 $P$ 和 $R$ 是黑盒，只能通过交互采样。**不知道模型时，Bellman 方程里的期望 $\sum_{s'} P(s'\mid s,a)[\cdot]$ 怎么算？** 这就是 Blog 02 的全部内容：三种答案分别叫 DP、MC、TD，而第三种将统治后面的所有算法。

---

## 11. Research Connection

本篇的概念在研究前沿的真实落点（只列真正相关的）：

**Robotics / Humanoid（路线 A）**
- **MDP 形式化就是 robotics 论文的第一张表**：DeepMimic (Peng et al., 2018) 的 state 是 ~200 维本体感知（关节角、速度、参考帧位姿差），action 是 PD target，reward 是 task + imitation 混合——每一项都是本篇五元组的具体化。**读懂那篇论文的前提是本篇的记号。**
- γ 的选择直接写在 humanoid 论文的实验设置里（典型 0.99，因为步态周期长）；reward 设计错误导致的"诡异步态"是 motion 领域最常见的 failure mode（Blog 12/13 用 AMP 解决它的动机）。

**LLM Post-training（路线 B）**
- **LLM 生成完全映射为本篇的形式系统**：prompt = $s_0$，每个 token = 一个 action $a_t$，已生成的前缀 = 当前状态 $s_t$（policy 就是 LLM 本身 $\pi_\theta(a_t \mid s_t)$），整条 response = 轨迹 $\tau$，reward model 打分 = return（**outcome reward，整条轨迹一个分数**——所以 LLM RL 天然是 Monte Carlo 语境，Blog 18 展开）。
- RLHF 的 KL penalty（Blog 18）本质是**对 policy 空间加约束**——它的存在恰恰因为 reward model 不可靠，呼应本篇 Failure Modes 的"reward 设计错误"。

**Human Motion / HOI**
- 物理仿真器（MuJoCo、Isaac Gym）就是 MDP 的 $P$ 的实现；"sim-to-real gap" 就是"仿真的 $P$ 与现实的 $P$ 不一致"——分布偏移问题在模型维度的表述。

---

## 12. Interview Questions

**Q1：为什么需要折扣因子 γ？γ = 1 会发生什么？**
A：三重作用——保证无限 horizon 回报收敛（γ=1 且持续正 reward 时 $G=\infty$，策略不可比）；表达时间偏好；控制有效视野 $1/(1-\gamma)$。γ=1 时即便有界（如含终止状态）也会显著放大远期 reward 的方差，使价值估计极难收敛。实践中 γ 是"任务时长"的换算：需要为百步之后的收益做规划就取 0.99。

**Q2：马尔可夫性是环境的性质吗？**
A：不是——它是"状态表示"的性质。同一环境，状态设计得好则近似 Markov，遗漏关键变量（速度、隐藏模式）则退化为 POMDP。所以"加一个状态特征让性能暴涨"在 RL 工程里极其常见。这题区分背定义的人和做过实验的人。

**Q3：V(s) 和 Q(s,a) 的关系？为什么控制（选动作）更偏爱 Q？**
A：$V^\pi(s) = \sum_a \pi(a\mid s) Q^\pi(s,a)$，V 是 Q 对策略的期望。选动作需要比较不同动作的后果，V 把动作维度平均掉了，$\arg\max_a Q$ 才能直接选。代价是 Q 的输入空间大一个维度（连续动作空间下 $\max_a Q$ 变成优化问题——Blog 09 DPG 的动机）。

**Q4：Bellman 期望方程和最优方程差一个 max，这个差别为什么重要？**
A：期望方程评估给定策略（线性，可解线性系统），服务于 policy evaluation；最优方程直接逼近最优（非线性，因为有 max），服务于 value iteration 和 Q-Learning。max 操作还引入系统性向上偏差（过估计），是 Double DQN / TD3 / Clipped Double Q 的存在理由——**一个 max 派生出 Blog 05 和 Blog 09 两条修补线**。

**Q5：Bandit 和 MDP 的本质区别是什么？**
A：有没有状态转移。Bandit 的动作不影响"下一时刻的世界"（没有 $s'$），因此没有 credit assignment、没有长期规划，唯一问题是探索。MDP 中动作改变状态分布，你的行为决定了你会见到什么数据——多了规划和 credit assignment 两个难题。

**Q6：如果去掉 Bellman 方程里的 γ（对 $V(s')$ 不打折），会怎样？**
A：价值不再有限（持续正 reward 时发散），且"当前一步"与"整个剩余未来"不可比——折扣本质是给不同时刻的 reward 统一换算到当前尺度。此外 $\gamma < 1$ 使 Bellman 算子成为压缩映射（contraction），保证不动点迭代收敛；γ=1 时压缩性丧失，收敛无保证（这是 Blog 02 要证的核心定理）。

**Q7：为什么说"LLM 的 RLHF 是 Monte Carlo 而非 TD"？**
A：reward 只在整条 response 结束时给出（outcome reward），没有中间信号可供 bootstrap；每条轨迹的 return 就是一个 MC 样本。这决定了 LLM RL 的方差特性和 GRPO 的组内归一化设计（Blog 18/19 展开）。答出此题说明能把本篇概念迁移到 LLM 场景。

**Q8：你的 agent 学会了"站在门前不动"。从本篇框架看，该从哪些维度排查？**
A：① reward 设计（不动是否恰好没有负 reward？γ 是否太小看不到开门的远期收益？）；② 探索（开门序列从未被采样到，正 reward 不可见——bandit 遗憾问题的 MDP 版）；③ 状态（Markov 性是否破坏，比如门的速度没进 state）；④ 任务本身是否不可达。这道题是 Blog 22 调试清单的预演——**先用公理排查，再怀疑算法**。

---

*下一篇：[RL 02：从动态规划到无需环境模型——DP、MC、TD](../RL_02_dp_mc_td/)*
