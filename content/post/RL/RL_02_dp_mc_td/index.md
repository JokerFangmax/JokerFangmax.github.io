---
title: "RL 02: 从动态规划到无需环境模型"
date: 2026-09-25
math: true
tags:
  - RL
  - Notes
categories:
  - RL
description: "DP、Monte Carlo、Temporal Difference：三种解 Bellman 方程的方式与 bias-variance 交易"
---

# RL 02：从动态规划到无需环境模型——DP、MC、TD

> 📖 **直觉先行阅读**：旧文 [Bellman Equation 到底在说什么](../../RL_basic_2/) 已铺过 DP/MC/TD 的直觉层，建议先读。本篇补齐**数学（压缩映射定理、bias-variance 分解）、伪代码与手算例子**。
> 前篇：[RL 01：RL 到底在解决什么问题](../RL_01_foundations/) —— 本篇直接使用其中的记号（MDP 五元组、$V^\pi$、$Q^\pi$、Bellman 方程）。

---

## 1. 这篇解决什么问题？

接着 Blog 01 结尾的问题：

> Blog 01 的 7.2 节里，我们是**解方程**求出 $V^\pi$ 的——但那需要 $P(s'\mid s,a)$ 和 $R(s,a)$ 完全已知。现在把你扔到真实世界：一个物理仿真器、一个真机器人、或者一个只能对话的 LLM——**你只有一个黑盒：扔进去 $(s,a)$，吐出来 $(r, s')$。转移概率和奖励函数是多少？不知道，也不让你知道。** 此时 Bellman 方程里的期望 $\sum_{s'} P(s'\mid s,a)[\cdots]$ 怎么算？

这是整个现代 RL 的起点。本篇给出三个答案，它们对"模型知道多少、样本怎么用"做了不同的取舍：

- **DP**：模型已知时的理论最优解（现实中几乎不可用，但是所有算法的标尺）；
- **MC**：完全不要模型，用整条轨迹的**实际回报**做估计；
- **TD**：不要模型，但用 Bellman 结构**自举（bootstrap）**——拿当前的估计拼出目标。

这三个答案的取舍关系——**偏差（bias）与方差（variance）、有模型与无模型、回合制与流式更新**——是后面 20 篇里几乎每一次算法演化的坐标轴。

---

## 2. Previous Algorithm：Blog 01 的形式系统

按六要素格式回顾（本篇的"前代"是形式系统而非算法）：

| 要素 | Blog 01 的状态 |
|------|---------------|
| **Input** | MDP 五元组 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$——但 $P, R$ 未知的场景没有覆盖 |
| **Model** | $V^\pi(s)$、$Q^\pi(s,a)$ 的定义；policy $\pi(a\mid s)$ |
| **Action** | 抽象映射，尚无学习机制 |
| **Environment** | MDP；Bandit 是无状态的退化情形 |
| **Reward** | $R(s,a)$；Return $G_t$ 承担 credit assignment 容器 |
| **Update** | **没有**——增量均值 $Q \leftarrow Q + \alpha(r - Q)$ 只在 Bandit 出现过 |

核心瓶颈：**Bellman 方程给了"价值的自洽条件"，但没有给"不知道 $P, R$ 时怎么满足它"的算法。** ε-greedy Bandit 的更新式是最原始的样本学习，但它没有状态、没有 bootstrap——本篇要把它推广到有状态的完整 MDP。

---

## 3. Core Idea

三句话，对应三个方法：

> **DP**：知道 $P, R$，直接对 Bellman 方程做不动点迭代——每轮扫全部状态，几何速度收敛到 $V^*$。
>
> **MC**：不知道模型，就**用真实回报的平均**逼近期望回报——"期望"变成"样本均值"，无偏但方差大，且必须等回合结束。
>
> **TD**：不知道模型，但**用当前的估计拼出目标**——$V(s')$ 代替未知的未来，更新目标为 $r + \gamma V(s')$；有偏但方差小、可以在线更新。

一句话浓缩本篇主旨：**MC 和 TD 是"拿什么当学习目标"的两个极端——真实但抖的 $G_t$ vs 估计但稳的 $r + \gamma V(s')$——后面所有的 n-step、TD(λ)、GAE 都在两者之间插值。**

---

## 4. Mathematics

### 4.1 DP：把 Bellman 方程变成迭代

**压缩映射（contraction mapping）**——DP 收敛的全部依据。定义 Bellman 最优算子 $\mathcal{T}$：

$$
(\mathcal{T}V)(s) = \max_a \sum_{s'} P(s'\mid s,a)\left[R(s,a,s') + \gamma V(s')\right]
$$

**定理**：$\mathcal{T}$ 在 sup 范数下是 $\gamma$-压缩的：$\|\mathcal{T}V_1 - \mathcal{T}V_2\|_\infty \le \gamma \|V_1 - V_2\|_\infty$。

证明骨架（值得手推一遍，面试常考）：

$$
\begin{aligned}
\big|(\mathcal{T}V_1)(s) - (\mathcal{T}V_2)(s)\big| 
&= \Big|\max_a \mathbb{E}_{s'}[\,\cdot + \gamma V_1(s')] - \max_a \mathbb{E}_{s'}[\,\cdot + \gamma V_2(s')]\Big| \\
&\le \max_a \mathbb{E}_{s'}\Big[\gamma \big|V_1(s') - V_2(s')\big|\Big] 
\le \gamma \|V_1 - V_2\|_\infty
\end{aligned}
$$

（第一步用了 $\big|\max_a f(a) - \max_a g(a)\big| \le \max_a |f(a)-g(a)|$；第二步期望的次线性。）

由 Banach 不动点定理：$\mathcal{T}$ 有唯一不动点 $V^*$，且从**任意**初始 $V_0$ 出发，$V_{k+1} = \mathcal{T}V_k$ 以几何速度 $\gamma^k \|V_0 - V^*\|$ 收敛。这就是**值迭代（Value Iteration）**。误差每迭代一轮至少缩小到 $\gamma$ 倍——$\gamma=0.9$ 时 40 轮后误差剩 $0.9^{40}\approx 1.5\%$。

**策略迭代（Policy Iteration）**则是另一个结构：交替执行

1. **策略评估**（Policy Evaluation）：解 $\pi$ 的 Bellman 期望方程 $V^\pi = \mathcal{T}_\pi V^\pi$（固定 $\pi$ 时这是**线性**方程组，Blog 01 的 $V^\pi(s_1) = \frac{p}{1-0.9(1-p)}$ 就是一个手工解）；
2. **策略改进**（Policy Improvement）：$\pi'(s) = \arg\max_a Q^\pi(s,a)$。

**策略改进定理**保证 $\pi' \ge \pi$（新策略不更差），于是有限 MDP 上有限步收敛到最优。评估和改进**都可以只做不完整的迭代**（不完全评估）——这正是后面广义策略迭代（GPI）的思想：Actor-Critic（Blog 07）就是"actor 和 critic 各自半成品地互相逼对方收敛"。

### 4.2 MC：期望 → 样本均值

$V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$ 里我们不知道期望里的分布，但**我们能采样**：跑完整条回合（episode），记录经过 $s$ 之后的实际回报 $G_t$，然后取平均：

$$
V(s) \leftarrow V(s) + \frac{1}{N(s)}\big(G_t - V(s)\big)
$$

（大数定律保证收敛到 $V^\pi$。）两种变体：**首次访问 MC**（first-visit，每个回合只取第一次经过 $s$ 的回报——样本独立性好，理论干净）与**每次访问 MC**（every-visit，全部计入——数据利用率高，现代实现几乎都用它）。

关键性质，为 4.4 的对比做准备：**$G_t$ 是 $V^\pi(s)$ 的无偏估计**——它的期望恰好就是 $V^\pi(s)$，一点不差。但它是"从 $t$ 到回合结束**所有**随机性的函数"：转移随机 × 策略随机 × 每一步都在叠加，方差随回合长度增长。

MC **没有 bootstrap**：目标 $G_t$ 完全由真实经验构成，不依赖任何当前估计。这个"干净"是要付出代价的——必须等回合结束才能更新（continuing task 直接不可用），且单样本信息量低。

### 4.3 TD：用估计拼目标

TD(0) 的更新：

$$
V(s_t) \leftarrow V(s_t) + \alpha\left[\underbrace{r_{t+1} + \gamma V(s_{t+1})}_{\text{TD target}} - V(s_t)\right], \qquad \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
$$

**逐项解剖**（本系列反复要用的形状）：

- **TD target** $r_{t+1} + \gamma V(s_{t+1})$：就是 Bellman 方程右边的一项，但 $\sum_{s'} P(\cdot)$ 的期望被**单个真实转移样本** $(s_t, a_t, r, s_{t+1})$ 替代——这是"不知道 $P$"的解法：**采样一次代替求期望**；
- $V(s_{t+1})$ 是**当前估计值**，不是真实值——这就是**自举（bootstrapping）**：用自己的猜测更新自己的猜测。名字的来源是"拉着靴带把自己提起来"（pull yourself up by your bootstraps）：没有任何真值，全靠自洽结构迭代逼近；
- $\delta_t$（**TD 误差**）：当前状态价值与新拼出的目标之间的落差——它度量"$V$ 还不自洽到什么程度"。收敛时 $\delta \to 0$（在期望意义下）。

对比 Blog 01 Bandit 的更新 $Q \leftarrow Q + \alpha(r - Q)$：那就是"TD(0) + $\gamma = 0$ + 无状态"——目标从"即时 reward"升级成了"即时 reward + 折扣的未来估计"。**同一个形状，装进了 Bellman 结构。**

### 4.4 MC vs TD：一次干净的 bias-variance 分解

这是本篇最重要的表，值得逐行理解：

| | MC 目标 $G_t$ | TD 目标 $r + \gamma V(s')$ |
|---|---|---|
| **是 $V^\pi(s)$ 的无偏估计？** | ✅ 是（期望恰为 $V^\pi(s)$） | ❌ 否（$V(s')$ 是当前估计，若不准则目标整体偏） |
| **方差** | ❌ 高：整条轨迹所有随机性叠加 | ✅ 低：只有一步转移的随机性 |
| **依赖初始值/中间估计？** | ✅ 不依赖（收敛性质干净） | ❌ 依赖（bootstrap 传播初值误差） |
| **更新时机** | 回合结束 | 每一步（在线、流式） |
| **Continuing task** | 不可用（没有回合边界） | 可用 |
| **收敛到** | $V^\pi$（大数定律） | $V^\pi$（表格情形，Robbins-Monro 条件下） |
| **函数逼近下** | 稳定（真梯度） | 可能发散（半梯度 + bootstrap，Blog 04 的 deadly triad） |

**为什么 TD 有偏？** 目标里的 $V(s_{t+1})$ 若当前等于 $\hat{V} \ne V^\pi$，则 $\mathbb{E}[r + \gamma \hat{V}(s')] = V^\pi(s) + \gamma(\hat{V} - V^\pi)(s')$ 的尺度——偏差恰好是**下一步估计的误差乘 $\gamma$**。所以 TD 的偏差会随 $V$ 的改善而消失（渐进无偏），但在训练全程存在。

**为什么 TD 方差低？** $G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots$ 抖动来自后面每一步；TD 目标只包含 $r_{t+1}$ 一步的随机性，剩余未来被"结算成了固定价格 $\gamma V(s_{t+1})$"。

### 4.5 n-step 与 TD(λ)：在两个极端之间插值

定义 n-step 目标：

$$
G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n})
$$

$n=1$ 是 TD(0)，$n \to \infty$ 是 MC。**n 越大越无偏、方差越大**——一个旋钮在两端之间滑动。TD(λ) 更聪明：把所有 n 加权平均（权重 $\propto \lambda^{n-1}$），等价于 **λ-回报（λ-return）** $G_t^\lambda$。**记住 λ 这个旋钮**：Blog 08 的 GAE(λ) 就是把同一思想用在 Advantage 上，它是 PPO 效果的最后一块拼图；Blog 16 的 Dreamer 在想象中用的也是 λ-回报。

---

## 5. Algorithm Pipeline

三种方法的**数据流与更新时机**对比——这是 on-policy / off-policy 之前最重要的结构性差异：

```
DP（无交互，纯计算）：
  对所有 (s,a) 循环 ──► 查 P, R（模型）──► 计算 TV ──► 更新表格
  ※ rollout 不存在；update = 全状态空间扫描（这就是它不可扩展的原因）

MC（回合制批处理）：
  rollout 整条 episode ──► 存 (s₀,a₀,r₁,…,s_T)
  回合结束 ──► 对每个访问过的 s：G_t ← 实际回报 ──► 更新 V(s)
  ※ rollout 与 update 严格分离：先跑完，再学习
  ※ 目标 G_t：来自 environment 的真实数据，不含任何网络输出

TD（在线流式）：
  每步 (s,a,r,s') 到手 ──► 立即更新 V(s) ──► 继续 rollout
  ※ rollout 与 update 交织进行
  ※ 目标 r + γV(s')：一半来自 environment（r, s'），一半来自 agent 的估计（V(s')）
```

变量属性表（新增 gradient 一列的雏形——本篇还是表格法，无梯度）：

| 变量/模块 | 来源 | DP | MC | TD |
|-----------|------|----|----|----|
| $(s, a)$ 遍历 | — | 全空间枚举 | 采样 | 采样 |
| $r, s'$ | environment / 模型查表 | 查表 | 实际 | 实际 |
| 学习目标 $G_t$ / $r+\gamma V(s')$ | — | 无（直接算 TV） | **纯环境数据** | **环境 + 自身估计混合** |
| $V$ 表格 | agent | ✅ 唯一学习对象 | ✅ | ✅ |
| 更新时机 | — | 每轮扫描后 | 回合结束 | 每步 |

**"目标含不含自身估计"**这一行请特别注意：它是 bootstrap 概念的精确表述，也是 Blog 04（target network 为什么要冻结一个"旧自己"）的直接伏笔。

---

## 6. Pseudocode

三个方法的可运行骨架（表格版；公式↔代码逐行对应）：

```python
# ---------- 1. Value Iteration（DP）----------
V = {s: 0.0 for s in S}
repeat:
    delta = 0
    for s in S:                              # 全状态扫描
        v_old = V[s]
        # TV(s) = max_a Σ_s' P(s'|s,a)[R + γV(s')]   ← Bellman 最优算子
        V[s] = max(
            sum(P[s][a][s_] * (Rw[s][a][s_] + gamma * V[s_]) for s_ in S)
            for a in A
        )
        delta = max(delta, abs(v_old - V[s]))
until delta < theta                           # 压缩映射保证收敛
```

```python
# ---------- 2. First-visit MC ----------
V = {s: 0.0 for s in S}; N = {s: 0 for s in S}
for episode in range(M):
    traj = rollout(policy_pi)                 # 跑完整条回合
    G = 0
    visited = set()
    for (s, a, r, s_) in reversed(traj):      # 从后往前累 G_t
        G = r + gamma * G                     # G_t = r_{t+1} + γG_{t+1}
        if s not in visited:                  # first-visit
            visited.add(s)
            N[s] += 1
            V[s] = V[s] + (G - V[s]) / N[s]   # V ← V + (G − V)/N   ←增量均值

# ---------- 3. TD(0) ----------
V = {s: 0.0 for s in S}; alpha = 0.1
for episode in range(M):
    s = env.reset()
    while not done:
        a = policy_pi(s)
        s_, r, done = env.step(a)
        # δ = r + γV(s') − V(s)          ← TD 误差
        # V(s) ← V(s) + α·δ              ← TD 更新
        td_target = r + gamma * (0 if done else V[s_])
        V[s] = V[s] + alpha * (td_target - V[s])
        s = s_
```

三个对应关系值得盯住：

| 数学 | 代码 |
|------|------|
| $(\mathcal{T}V)(s) = \max_a \sum_{s'} P[\,R+\gamma V\,]$ | `V[s] = max(sum(P*(Rw + gamma*V)))` |
| $G_t = r_{t+1} + \gamma G_{t+1}$ | `G = r + gamma * G`（逆序遍历） |
| $V(s) \leftarrow V(s) + \alpha[\,r + \gamma V(s') - V(s)\,]$ | `V[s] += alpha * (td_target - V[s])` |

注意 TD(0) 里 `done` 时目标里 $V(s')$ 置 0——终止状态的 $V = 0$ 是约定，漏掉它是最常见的实现 bug 之一。

---

## 7. Toy Example

### 7.1 DP：三格走廊上手工跑两轮值迭代

环境：$s_1 \to s_2 \to s_3 \to G$（终止），动作"右移"确定成功，奖励 $0, 0, +1$（在 $s_3$ 右移得 +1），$\gamma = 0.5$。

真值可以直接解出：$V^*(s_3) = 1$，$V^*(s_2) = 0 + 0.5 \times 1 = 0.5$，$V^*(s_1) = 0.5 \times 0.5 = 0.25$。

**从 $V_0 \equiv 0$ 出发，正向顺序（$s_1, s_2, s_3$）扫一轮**（用旧值，即 Jacobi 式）：

- $V_1(s_1) = 0 + 0.5 \, V_0(s_2) = 0$
- $V_1(s_2) = 0 + 0.5 \, V_0(s_3) = 0$
- $V_1(s_3) = 1 + 0.5 \times 0 = 1$

第二轮：$V_2(s_1) = 0.5 \times 0 = 0$，$V_2(s_2) = 0.5 \times 1 = 0.5$，$V_2(s_3) = 1$。第三轮：$V_3(s_1) = 0.25$。**奖励信息每次只向前传播一格**——这就是"扫描方向影响收敛速度"的直观来源。

**若用就地更新（Gauss-Seidel 式）且从 $s_3$ 往回扫**：$V(s_3)=1$，接着 $V(s_2) = 0.5 \times 1 = 0.5$，$V(s_1) = 0.25$——**一轮收敛**。工程上人人都用就地更新，原因就在这。

### 7.2 MC vs TD：双状态随机游走（本篇的核心例子）

环境：状态 A、B；每次等概率左移或右移；走出左边界 reward 0、终止，走出右边界 reward +1、终止；中间所有转移 reward 0；$\gamma = 1$。策略：均匀随机。

**先解真值**（Bellman 期望方程）：

$$
V(A) = 0.5 \cdot 0 + 0.5\,V(B), \qquad V(B) = 0.5\,V(A) + 0.5 \cdot 1
$$

代入：$V(B) = 0.25 V(B) + 0.5 \Rightarrow V(B) = \frac{2}{3},\ V(A) = \frac{1}{3}$。

**MC 视角**。跑两条回合：(i) `A→B→右出`；(ii) `A→左出`。

- 回合 (i)：$G(A) = 0 + 1 = 1$（A 的实际回报），$G(B) = 1$；
- 回合 (ii)：$G(A) = 0$。

对 A 的估计：首次访问 MC 下 $\hat{V}(A) = \frac{1+0}{2} = 0.5$——**真值是 1/3，两条样本就抖出 0.5**。而且这个抖动不是实现问题：$G(A)$ 的可能取值是 0 和 1 两个极端，回合越长可能取值越连续，方差越大。**MC 每条样本都"言之凿凿"（无偏），但每一票都很吵。**

**TD 视角**。初始化 $V(A) = V(B) = 0$，$\alpha = 0.1$。

- 跑回合 `B→右出`：在 B 处 $r=1$，$s'$ 为终止（$V=0$）。$\delta = 1 + 0 - 0 = 1$，$V(B) \leftarrow 0 + 0.1 \times 1 = 0.1$；
- 跑回合 `A→B→右出`：A→B 一步：$\delta = 0 + 1 \cdot V(B) - V(A) = 0.1 - 0 = 0.1$，$V(A) \leftarrow 0.01$。B→右出：$\delta = 1 - 0.1 = 0.9$，$V(B) \leftarrow 0.19$。

看见了吗：**MC 在第一条回合就把 $V(A)$ 直接推到 1（若回合是 (i)），而 TD 只让 $V(A)$ 挪了 0.01**——因为 TD 的目标里 B 的价值自己还没学好，A 只能"信一半"。**TD 慢，但稳；它逐回合把价值从终止边界往回渗**，这个"渗"的速度由 $\alpha$ 和访问频率控制。同样的数据量下 TD 通常更快收敛（Sutton & Barto 第 6 章的随机游走实验正是这个结论），代价是引入 bootstrap 偏差。

### 7.3 一个"MC 无偏但被坑"的例子

设 A 之后策略必走长路：`A→s₂→s₃→…→s₁₀→右出(+1)`。$V^\pi(A) = 1$（$\gamma=1$）。MC 的每条样本 $G(A)$ 都恰好等于 1（无偏且这情形下无随机性）；但如果最后一步有 10% 概率掉回 $s_2$ 绕远路，回合长度变成随机，$G(A)$ 在 $\{1, 0, \cdots\}$（长路可能中途 0 奖励结束）之间跳动——**回合越长，MC 单样本越不可靠，但每条样本照样"全额记账"**。TD 则永远只结算一步 + 固定价格，不背整条轨迹的锅。

---

## 8. Why It Works

1. **DP 的收敛性是定理不是经验**：压缩映射（4.1 的证明）保证任意初值几何收敛。这给了所有后续算法一把标尺：**任何"学 $V$ 或 $Q$"的算法，其理想极限都是 DP 的不动点迭代，区别只在"期望怎么算"（模型 vs 采样）和"何时更新"（扫描 vs 在线）**。
2. **MC 的正确性来自大数定律**：样本均值 → 期望，无偏、与 Markov 性无关（MC 甚至不要求 Bellman 方程成立——它直接估计定义 $V^\pi = \mathbb{E}[G]$）。代价是 $\text{Var}(G_t)$ 随有效视野 $\frac{1}{1-\gamma}$ 增长，样本效率差。
3. **TD 的有效性 = Bellman 结构作为先验知识**：TD 把"未来满足 Bellman 自洽"这一结构性强假设注入了估计器——**它不只用数据，还用了数据背后的方程**。这就是它样本效率高的原因：一次转移 $(s,a,r,s')$ 能传播信息到 $V(s)$（MC 里这条信息只影响轨迹上被访问过的状态）。统计学视角：牺牲无偏性（引入 $\gamma$ 尺度的可控偏差）换取方差数量级下降，通常是划算的交易。
4. **n-step/λ 的插值是同一个 trade-off 的参数化**：不存在普适最优的 n/λ——环境随机性大（转移噪声高）时短 horizon 的 TD 目标更稳；奖励延迟且方差低时长 horizon 甚至 MC 更好。**这也是为什么 PPO 的 GAE-λ 和 Dreamer 的 λ-回报是这些算法最敏感的超参之一。**
5. **半梯度视角（通往 Blog 04 的桥）**：TD(0) 更新不是任何目标函数的真正梯度下降（目标里的 $V(s')$ 没有被求导，相当于 stop-gradient）——这叫**半梯度（semi-gradient）**。表格情形无所谓，函数逼近 + bootstrap + off-policy 三者叠加时半梯度会发散（deadly triad）。**先记住结论，Blog 04 展开。**

---

## 9. Failure Modes

| 方法 | 失效场景 | 具体机理 | 后续修补 |
|------|---------|---------|---------|
| **DP** | 状态空间大/连续 | 扫描所有状态、每步求 max over A 都不可行 | 函数逼近（Blog 04 DQN）；采样代替期望（TD 系） |
| **DP** | 模型未知/有模型误差 | 需要 $P, R$ 精确已知 | 全部 model-free 方法；或学模型（Blog 16） |
| **MC** | 回合极长 / continuing task | 必须等到回合结束才能更新 | TD / n-step |
| **MC** | 方差爆炸 | $\text{Var}(G_t)$ 随视野增长，稀疏奖励更糟 | baseline（Blog 07）、GAE（Blog 08） |
| **MC** | off-policy 数据 | 目标分布 ≠ 行为分布，需重要性采样修正（方差进一步爆炸） | Blog 03/08 的 importance ratio |
| **TD** | bootstrap 偏差 | 初值烂 + $\alpha$ 大 → 目标持续偏，表格下仍收敛但慢；函数逼近下可能发散 | target network（Blog 04）、trust region（Blog 08） |
| **TD** | 稀疏奖励 + 价值传播慢 | 奖励信息每步只传一格（7.1 的"渗"） | n-step/λ、reward shaping（Blog 12）、探索（Blog 10） |
| **共性** | 探索不足 | 评估的只是**被访问到的**状态的价值——数据没到的地方一切免谈 | ε-greedy → entropy（Blog 07/10） |

**读表的方法**：每个后续算法都对应表里至少一格。例如 target network 治"bootstrap 偏差 + 数据相关"的乘积项；PPO 治"半梯度更新步长过大"；这就是"演化树"的机械构造法。

---

## 10. 与前后算法对比

```
Blog 01：MDP / Return / V / Q / Bellman（定义了"什么是解"）
    ↓
DP：知道模型 → 不动点迭代（理论最优、现实不可扩展）
MC：不要模型 → 真实回报平均（无偏、高方差、回合制）
TD：不要模型 → 自举目标（有偏、低方差、在线）
    ↓                                    ↓
Blog 03：TD 用于控制                n-step / TD(λ)（插值）
（SARSA / Q-Learning：             → GAE（Blog 08）
  把"评估 V"变成"学 Q 并选动作"）   → λ-return（Blog 16 Dreamer）
```

三方法对照表：

| | DP | MC | TD(0) |
|---|---|---|---|
| 需要模型 $P, R$ | ✅ | ❌ | ❌ |
| Bootstrapping | ✅（隐式，算子里全是 $V$） | ❌ | ✅ |
| 需要回合边界 | 否 | ✅ | 否 |
| Bias（对 $V^\pi$） | 0（收敛后精确） | 0（无偏） | $\gamma \cdot (\hat{V}-V^\pi)$（渐进 0） |
| Variance（单样本目标） | — | 高（全轨迹） | 低（单步） |
| 更新时机 | 全空间扫描 | 回合结束 | 每步在线 |
| 样本效率 | —（不算样本） | 低 | 高 |
| 延伸 | Value/Policy Iteration | REINFORCE（Blog 06）、RLHF 的 outcome return（Blog 18） | SARSA/Q-Learning/DQN/Actor-Critic（Blog 03+） |

**留给下一篇的问题**：本篇全在**评估**（学 $V^\pi$），还没**控制**（用学到的价值选动作、并且动作选择本身影响你采到什么数据）。把 TD 和"边学边选动作"组合起来会发生什么？两种组合方式——"按我实际走的路学"（SARSA）和"按理论最优的路学"（Q-Learning）——将分裂出 **on-policy / off-policy** 这个 RL 最重要的分野。

---

## 11. Research Connection

**Robotics / Human Motion（路线 A）**
- Humanoid 训练几乎从不用纯 MC（回合长达数千步、方差不可接受）也不用 DP（连续状态没法扫描）——**PPO + GAE(λ) 统治这个领域，正是"TD 与 MC 的插值"在连续控制中的落地**（Blog 08/12）。
- 物理仿真器给 humanoid 的 return 抖动极大（接触动力学的混沌性）——实践里用短 horizon n-step / 较小 λ 压方差，是 DeepMimic 类工作的隐性工程知识。

**LLM Post-training（路线 B）**
- **RLHF 的 return 是整条 response 一个分数（outcome reward）——这天然是 MC 目标**。没有中间 reward 就没有 TD 所需的逐步信号，所以 LLM RL 的很多设计（GRPO 的组内基线，Blog 19）本质是在做"MC 信号的方差缩减"——与 Blog 07 的 baseline 异曲同工。
- 过程奖励模型（PRM，Blog 21）试图给每个推理步骤打分——**那正是在把 LLM RL 从"MC 语境"改造成"TD 语境"**，这是当前 reasoning 研究的一条主线。

**Model-Based（通往 Blog 16）**
- Dreamer 在学到的世界模型里做 λ-回报的想象更新：模型给了"假的 $P$"，TD 目标在假模型里自举——model-based 的核心风险（模型误差 × bootstrap 误差叠加）在本篇已经能看懂一半。

---

## 12. Interview Questions

**Q1：为什么 TD 有偏、MC 无偏？偏差具体是多大？**
A：MC 目标 $G_t$ 的期望恰为 $V^\pi(s)$。TD 目标 $r+\gamma V(s_{t+1})$ 里 $V(s_{t+1})$ 是当前估计 $\hat{V}$，偏差为 $\gamma\,(\hat{V}-V^\pi)$ 的期望尺度——**偏差恰好是下一步估计误差打 $\gamma$ 折**，随估计改善渐进消失。所以 TD 是"训练全程有偏、收敛时无偏"。

**Q2：证明 Bellman 算子是 γ-压缩。**（口头版）
A：对任意两个 $V_1, V_2$：$|(\mathcal{T}V_1)(s)-(\mathcal{T}V_2)(s)| \le \max_a \mathbb{E}[\gamma|V_1(s')-V_2(s')|] \le \gamma\|V_1-V_2\|_\infty$。关键一步是 $\big|\max_a f(a) - \max_a g(a)\big| \le \max_a|f(a)-g(a)|$。由 Banach 不动点定理得唯一不动点 $V^*$ 和几何收敛。

**Q3：为什么实践中用就地更新（in-place）的值迭代？**
A：Gauss-Seidel 式就地更新让"已更新的新值"立即被本轮后面的状态使用——信息在**一次扫描内部**就向后传播（7.1 里反向扫描一轮收敛 vs Jacobi 三轮）。理论上两种都收敛（压缩性不变），就地版常数因子更小。

**Q4：first-visit 和 every-visit MC 有什么区别？现代实现用哪个？**
A：first-visit 只用每回合首次访问 $s$ 的回报，样本独立性可证收敛（经典理论结果）；every-visit 用全部访问，数据利用率高，偏差二阶小。现代实现几乎都用 every-visit（含函数逼近时区别更不重要）。答"为什么 first-visit 理论干净"（同一回合内两次访问的 $G_t$ 强相关）是加分项。

**Q5：TD(0) 为什么不是真正的梯度下降？"半梯度"是什么意思？**
A：若把 TD 视为最小化 $\mathbb{E}[\delta^2]$，对参数求梯度时目标里的 $V(s')$ 项也应被求导，但 TD 更新把它当常数（相当于对目标 stop-gradient）——所以更新方向不是任何损失函数的真梯度。表格参数化下无碍；函数逼近 + bootstrap + off-policy 组合时半梯度可致发散（deadly triad，Blog 04）。

**Q6：n-step TD 的 n 怎么选？n→∞ 和 n=1 各退化成什么？**
A：n=1 是 TD(0)，n→∞ 是 MC；n 是 bias-variance 旋钮。环境转移噪声大 → 短 n（早点"结算成固定价格"）；奖励稀疏但确定性高 → 长 n。实践中常用 TD(λ)/GAE 直接在所有 n 上加权，λ 是比单一 n 更平滑的旋钮。

**Q7：为什么说"RLHF 是 MC、PRM 是在把 LLM RL 改造成 TD"？**
A：outcome reward 只在 response 结束时给，return 就是单点 MC 样本，无中间信号可 bootstrap；PRM 给逐步打分后，每一步有了即时 reward，TD 式的自举与信用分配才成为可能。这题检验能否把本篇框架迁移到 LLM。

**Q8：MC 在什么条件下根本无法使用？**
A：continuing task（无回合边界，如持续的机器人控制任务）——MC 必须等到回合结束拿到完整 $G_t$。此外 off-policy 场景下 MC 需要重要性采样修正，方差随回合长度爆炸，实践中近乎不可用（这正是 Q-Learning 免 IS 的优势，Blog 03 的主题）。

**Q9：同样的数据量，为什么 TD 通常比 MC 收敛快？**
A：信息传播方式不同：MC 的每条样本只更新"被访问状态"的均值；TD 的每次转移 $(s,a,r,s')$ 把 $s'$ 的（可能是别处学来的）价值信息**即时回传给 $s$**——一条轨迹上价值信息沿 Bellman 图逆向扩散。Sutton & Barto 的随机游走实验定量展示这一点（本篇 7.2 是它的三状态简化版）。

---

*下一篇预告：RL 03——Q-Learning 与 SARSA：On-policy vs Off-policy（写作中，发布后此处更新链接）*
