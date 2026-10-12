# RL 系列博客规划（活文档）

> 本文件是 RL 系列博客的总体规划，**每发布一篇更新一次**。
> 系列组织原则：前一个方法有什么问题 → 为什么需要新方法 → 新方法修改了什么 → 数学如何实现 → 代码如何实现 → 新方法又留下什么问题。
> 每篇文章统一遵循 12 节模板（见文末附录）。

---

## 一、系列目录（22 篇）

优先级：★★★ 必须精读（数学+代码+手算 toy example）｜★★ 理解即可（抓住演化逻辑）｜★ 可跳过
路线：A = Robotics / Human Motion｜B = LLM Post-training｜AB = 两条路线共用

| # | 标题 | 核心问题 | 覆盖算法 | 优先级 | 时间 | 前置 | 路线 | 旧文互链 |
|---|------|---------|---------|--------|------|------|------|---------|
| 01 | RL 到底在解决什么问题？ | 没有标准答案、反馈延迟的学习问题怎么形式化？ | Bandit → MDP → Return → V/Q → Bellman | ★★★ | 3h | ML 基础 | AB | RL_basic |
| 02 | 从动态规划到无需环境模型 | 不知道转移概率和奖励函数时怎么学价值？ | DP → MC → TD（bias/variance、bootstrapping） | ★★★ | 4h | 01 | AB | RL_basic_2 |
| 03 | Q-Learning 与 SARSA：On-policy vs Off-policy | 学"我实际走的路"还是"理论上最优的路"？ | SARSA → Q-Learning（ε-greedy、importance sampling 伏笔） | ★★★ | 4h | 02 | AB | RL_basic_3 |
| 04 | 为什么 Q-table 不够？ | 状态无穷大时 Q 函数怎么表示、为什么深度化后不稳定？ | Q-Learning → DQN（replay buffer、target network、deadly triad） | ★★★ | 5h | 03 | AB | RL_basic_3 |
| 05 | DQN 为什么仍然会出问题？ | 每个已知的 DQN 病灶对应哪个修补？ | Double DQN → Dueling → PER（+distributional 一瞥） | ★★ | 3h | 04 | AB | — |
| 06 | 为什么不直接学 Policy？ | 不想绕道价值函数，能不能直接优化决策本身？ | REINFORCE（policy gradient theorem 完整推导） | ★★★ | 4h | 02 | AB | RL_basic_4 |
| 07 | 为什么需要 Critic？ | REINFORCE 方差大到训不动，怎么办？ | Baseline → Advantage → Actor-Critic → A2C/A3C | ★★★ | 5h | 04, 06 | AB | RL_basic_4 |
| 08 | 如何防止 Policy 一次更新太远？ | 为什么 policy 会突然崩掉？怎么限制更新步长？ | TRPO → PPO → GAE（KL、trust region、importance ratio） | ★★★ | 8h | 07 | AB | RL_basic_4 |
| 09 | 连续控制 I：确定性策略 | 动作是连续向量时 max Q 怎么办？ | DPG → DDPG → TD3 | ★★ | 4h | 04, 07 | A | RL_basic_4 |
| 10 | 连续控制 II：最大熵 RL | DDPG 系列为什么还是脆？探索从哪来？ | SAC（soft Q、entropy、twin Q、temperature） | ★★★ | 6h | 08, 09 | A | RL_basic_4 |
| 11 | 从 RL 到模仿学习 | 没有奖励函数、只有专家示范时怎么学？ | BC → DAgger → GAIL | ★★ | 4h | 06, 07 | A | — |
| 12 | RL 如何控制物理人体？ | PPO 怎么和一个物理仿真器、一个 PD 控制器、一段参考动作组装成一个学习系统？ | DeepMimic（state/action/reward 设计、PD target、Early Termination、RSSM of reference） | ★★★ | 6h | 08 | A | — |
| 13 | 如何学习自然的人体运动先验？ | 逐帧模仿参考动作太贵、太不自然，能否只学"风格"？ | DeepMimic → AMP（adversarial motion prior、GAIL discriminator 复用、ASE 一瞥） | ★★★ | 5h | 11, 12 | A | — |
| 14 | 有数据为什么还需要 Offline RL？ | 不能交互、只有固定数据集时，为什么直接跑 Q-Learning 会爆炸？ | BCQ → CQL → IQL（distribution shift、OOD action） | ★★★ | 6h | 04, 08 | A | — |
| 15 | RL 能不能直接变成序列建模？ | 把 return 当条件，轨迹当序列，Transformer 直接生成行为？ | Decision Transformer（+ offline 对比） | ★★ | 3h | 14 | A | — |
| 16 | Agent 能不能学习自己的世界？ | 交互太贵，能不能学一个想象中的环境来训练？ | Dyna → MBPO → World Models → Dreamer(V1–V3) | ★★★ | 6h | 04, 10 | A | — |
| 17 | 从 AlphaGo 到 MuZero | 规划（MCTS）和学习怎么结合？不知道规则也能规划吗？ | AlphaGo → AlphaZero → MuZero | ★★ | 3h | 16 | — | — |
| 18 | RL 如何进入 LLM？ | 一个语言模型的生成过程怎么映射成 MDP？PPO 哪里水土不服？ | SFT → Reward Model → RLHF-PPO（token=action、KL penalty） | ★★★ | 7h | 08 | B | — |
| 19 | 为什么 LLM RL 又出现 GRPO？ | LLM 的 critic 和 actor 一样大，太贵了，能不能去掉？ | GRPO（group advantage、去 critic）+ DAPO/GSPO/Dr.GRPO 变体 | ★★★ | 6h | 18 | B | — |
| 20 | DPO 到底是不是 RL？ | 能不能从偏好数据一步到位，跳过 reward model 和 rollout？ | DPO（+ IPO/KTO/SimPO 一览）、RLHF/PPO/DPO/GRPO 统一视角 | ★★★ | 5h | 18 | B | — |
| 21 | Reasoning RL / RLVR | 数学/代码这种能自动判分的任务，RL 还需要 reward model 吗？ | RLVR、outcome vs process reward、test-time scaling、长 CoT、agentic RL | ★★★ | 6h | 19 | B | — |
| 22 | Capstone：统一的 RL 世界观 | Humanoid control 和 LLM reasoning 底层共享什么数学？怎么选型、怎么调试？ | 全系列综合 + 选型决策树 + failure 排查清单 | ★★★ | 3h | 01–21 | AB | — |

**总计：约 100 小时**（含读博客、手算 toy example、浏览论文指定 Section）。

---

## 二、Algorithm Family Tree

```
Reinforcement Learning
│
├── Foundations（01–02）
│   ├── Bandit ────────────────────────────── 无状态，探索/利用的最小模型
│   ├── MDP / Bellman Equation ────────────── 一切后续算法的公理基础
│   ├── Dynamic Programming ───────────────── 需要模型的上界（value iteration / policy iteration）
│   └── Monte Carlo ──┐
│                      ├──→ Temporal Difference ──→ bootstrapping 主线
│                      │    （无模型学习的两大源头：MC=无偏高方差，TD=有偏低方差）
│
├── Value-Based（03–05）                     [学 Q，策略由 argmax Q 隐式给出]
│   ├── SARSA ─────────────────────────────── on-policy TD 控制
│   ├── Q-Learning ────────────────────────── off-policy TD 控制
│   ├── DQN ───────────────────────────────── Q-table → 神经网络 + replay + target net
│   └── 修补系：Double DQN / Dueling / PER ── 过估计 / 状态价值分解 / 采样效率
│
├── Policy-Based（06）                       [直接学 π_θ]
│   └── REINFORCE ─────────────────────────── policy gradient theorem，MC 回报
│
├── Actor-Critic（07–08）                    [value 为 policy 服务]
│   ├── Actor-Critic / A2C / A3C ──────────── baseline → advantage
│   ├── TRPO ──────────────────────────────── KL 信任区域（二阶优化）
│   └── PPO ───────────────────────────────── 一阶 clip 近似（工业默认 on-policy 算法）
│        └── GAE(λ) ───────────────────────── advantage 估计的 bias-variance 旋钮
│
├── Continuous Control（09–10）
│   ├── DPG → DDPG ────────────────────────── 确定性策略 + off-policy + replay
│   ├── TD3 ───────────────────────────────── twin Q + delayed update（过估计修补）
│   └── SAC ───────────────────────────────── 最大熵 RL + 随机策略（工业默认 off-policy 算法）
│
├── Imitation / Motion（11–13）              [奖励缺失 → 从示范构造信号]
│   ├── BC → DAgger ───────────────────────── 监督学习式模仿（distribution shift 病灶）
│   ├── GAIL ──────────────────────────────── 对抗模仿：判别器即学习到的奖励
│   ├── DeepMimic ─────────────────────────── PPO + 物理 simulation + 逐帧模仿奖励
│   └── AMP ───────────────────────────────── 风格先验（GAIL 判别器 + RL 任务奖励）
│
├── Offline RL（14）                          [不许交互，只有静态数据集]
│   └── BCQ → CQL → IQL ───────────────────── OOD action 三种解法：约束 / 悲观 / 期望回归
│
├── RL as Sequence Modeling（15）
│   └── Decision Transformer ──────────────── return-to-go 条件生成（与 18 的 LLM 视角呼应）
│
├── Model-Based（16–17）
│   ├── Dyna(-Q) ──────────────────────────── 真实经验 + 想象经验混合
│   ├── MBPO ──────────────────────────────── 短 horizon rollout 控制 model bias
│   ├── World Models / Dreamer ────────────── latent 世界模型 + 想象中训练 actor-critic
│   └── AlphaZero / MuZero ────────────────── 学习的模型 + MCTS 规划
│
└── Modern LLM RL（18–21）
    ├── RLHF-PPO ──────────────────────────── SFT → reward model → PPO + KL anchor
    ├── GRPO ──────────────────────────────── group relative advantage，去 critic
    │    └── 变体：Dr.GRPO / DAPO / GSPO（写作时搜索确认最新进展）
    ├── DPO ───────────────────────────────── 偏好数据直接优化，跳过 RM 与 rollout
    └── RLVR / Reasoning RL ────────────────── 可验证奖励、outcome vs process、test-time scaling
```

---

## 三、RL Master Table（完整版，每篇发布后更新）

缩写：V=Value-based, P=Policy-based, AC=Actor-Critic, MB=Model-based；
On/Off=数据来源策略；MF=Model-free；Disc=离散动作, Cont=连续动作, DC=皆可；
Stoch=随机策略, Det=确定性策略；RB=Replay Buffer；BS=Bootstrapping

### 3.1 基础与 Value-Based

| Algorithm | Year | 类型 | On/Off | MF/MB | Disc/Cont | Policy | RB | Critic | BS | Main Innovation | Main Weakness | Successor |
|-----------|------|------|--------|-------|-----------|--------|----|--------|----|-----------------|---------------|-----------|
| Bandit (ε-greedy/UCB) | — | V | — | MF | Disc | Stoch | ✗ | ✗ | ✗ | 探索/利用最小模型 | 无状态 | MDP |
| VI / PI (DP) | 1957 | V | — | MB | DC | Det | ✗ | ✗ | ✓ | Bellman 最优性的直接求解 | 需要完整模型 | MC/TD |
| MC Control | — | V | On | MF | Disc | Stoch | ✗ | ✗ | ✗ | 无模型、无偏 | 高方差、必须等 episode 结束 | TD |
| SARSA | 1994 | V | On | MF | Disc | Stoch | ✗ | ✗ | ✓ | on-policy TD 控制 | 学到的是当前（含探索）策略的值 | Q-Learning |
| Q-Learning | 1989 | V | Off | MF | Disc | Stoch | ✗ | ✗ | ✓ | max 操作直接学最优 Q | 过估计、表格规模爆炸 | DQN |
| DQN | 2015 | V | Off | MF | Disc | Stoch | ✓ | ✓(target) | ✓ | Q 神经网络化 + replay + target net | 过估计、相关样本、仅离散 | DDQN/TD3 |
| Double DQN | 2016 | V | Off | MF | Disc | Stoch | ✓ | ✓ | ✓ | 动作选择与评估分离治过估计 | 不解决其他病灶 | TD3(连续版) |
| Dueling DQN | 2016 | V | Off | MF | Disc | Stoch | ✓ | ✓ | ✓ | V 与 A 分解网络结构 | 结构性改进，非算法性 | — |
| PER | 2016 | V | Off | MF | Disc | Stoch | ✓(优先) | ✓ | ✓ | TD 误差驱动的非均匀采样 | 引入 bias（需 IS 修正） | — |

### 3.2 Policy-Based 与 Actor-Critic

| Algorithm | Year | 类型 | On/Off | MF/MB | Disc/Cont | Policy | RB | Critic | BS | Main Innovation | Main Weakness | Successor |
|-----------|------|------|--------|-------|-----------|--------|----|--------|----|-----------------|---------------|-----------|
| REINFORCE | 1992 | P | On | MF | DC | Stoch | ✗ | ✗ | ✗ | policy gradient theorem | 高方差、样本用完即弃 | Actor-Critic |
| Actor-Critic | 1983 | AC | On | MF | DC | Stoch | ✗ | ✓ | ✓ | critic 降方差 | 两条网络互相追逐不稳定 | A2C |
| A2C / A3C | 2016 | AC | On | MF | DC | Stoch | ✗ | ✓ | ✓ | 并行 rollout 去相关 | 样本效率低 | PPO |
| TRPO | 2015 | AC | On | MF | DC | Stoch | ✗ | ✓ | ✓ | KL 信任区域约束更新 | 二阶优化实现复杂 | PPO |
| PPO | 2017 | AC | On | MF | DC | Stoch | ✗(小batch) | ✓ | ✓ | clip 一阶近似信任区域 | 样本效率、超参敏感 | GRPO(LLM)/固定主力 |
| GAE | 2016 | 技术 | — | MF | — | — | — | — | — | advantage 的 λ 偏差-方差插值 | — | — |

### 3.3 连续控制

| Algorithm | Year | 类型 | On/Off | MF/MB | Disc/Cont | Policy | RB | Critic | BS | Main Innovation | Main Weakness | Successor |
|-----------|------|------|--------|-------|-----------|--------|----|--------|----|-----------------|---------------|-----------|
| DPG | 2014 | AC | Off | MF | Cont | Det | ✓ | ✓ | ✓ | 确定性策略定理，绕过 max | 探索靠外部噪声 | DDPG |
| DDPG | 2016 | AC | Off | MF | Cont | Det | ✓ | ✓(target) | ✓ | DPG + DQN 工程技巧 | 脆、超参敏感、过估计 | TD3 |
| TD3 | 2018 | AC | Off | MF | Cont | Det | ✓ | ✓(twin) | ✓ | twin Q + delayed update + 平滑 | 仍是确定性策略，探索弱 | SAC |
| SAC | 2018 | AC | Off | MF | DC | Stoch | ✓ | ✓(twin) | ✓ | 最大熵目标 + 温度自适应 | 计算/样本 trade-off 固定主力 | — |

### 3.4 模仿、运动控制与 Offline RL

| Algorithm | Year | 类型 | On/Off | MF/MB | Disc/Cont | Policy | RB | Critic | BS | Main Innovation | Main Weakness | Successor |
|-----------|------|------|--------|-------|-----------|--------|----|--------|----|-----------------|---------------|-----------|
| BC | — | 监督 | — | MF | DC | 任意 | — | ✗ | ✗ | 示范直接监督学习 | compounding error / 分布漂移 | DAgger |
| DAgger | 2011 | 模仿 | On | MF | DC | 任意 | — | ✗ | ✗ | 专家在线纠正分布漂移 | 需要可查询专家 | GAIL |
| GAIL | 2016 | 模仿 | On | MF | DC | Stoch | ✓ | ✓(D) | ✗ | 判别器=学习到的奖励（对抗） | 训练不稳定、需环境交互 | AMP |
| DeepMimic | 2018 | AC(PPO) | On | MF(物理) | Cont | Stoch | ✗(小batch) | ✓ | ✓ | 任务+逐帧模仿混合奖励、物理仿真控制 | 需要逐帧对齐的参考数据 | AMP |
| AMP | 2021 | AC(PPO) | On | MF(物理) | Cont | Stoch | ✓ | ✓(D) | ✗ | 对抗式风格先验替代逐帧奖励 | 奖励 hacking/伪影、需调平衡 | ASE/PULSE 等 |
| BCQ | 2019 | V | Off(静态) | MF | DC | — | ✓(静态) | ✓ | ✓ | 限制候选动作在数据支撑集内 | 需生成模型，工程复杂 | CQL |
| CQL | 2020 | V | Off(静态) | MF | DC | — | ✓(静态) | ✓ | ✓ | 对 OOD 动作的 Q 做悲观惩罚 | 保守过头、调参难 | IQL |
| IQL | 2021 | V | Off(静态) | MF | DC | — | ✓(静态) | ✓ | ✓ | expectile 回归，绝不查询 OOD 的 Q | 依赖数据质量 | Decision Transformer 对比 |

### 3.5 Model-Based 与规划

| Algorithm | Year | 类型 | On/Off | MF/MB | Disc/Cont | Policy | RB | Critic | BS | Main Innovation | Main Weakness | Successor |
|-----------|------|------|--------|-------|-----------|--------|----|--------|----|-----------------|---------------|-----------|
| Dyna-Q | 1990 | MB | Off | MB | Disc | Stoch | ✓ | ✗ | ✓ | 真实+想象经验混合更新 | 模型误差累积 | MBPO |
| MBPO | 2019 | MB | Off | MB | Cont | Stoch | ✓ | ✓ | ✓ | 短 horizon 想象 rollout 控模型偏差 | 模型训练昂贵 | Dreamer |
| World Models | 2018 | MB | — | MB | Cont | Stoch | ✓ | ✗ | ✗ | VAE+RNN+evolution 的 latent 梦境 | 无 RL 微调 | Dreamer |
| Dreamer (V1–V3) | 2020–23 | MB | On(想象) | MB | DC | Stoch | ✓ | ✓ | ✓ | latent 世界模型内做 actor-critic | 世界模型偏差、长 horizon | … |
| AlphaZero | 2017 | MB+规划 | On(自对弈) | MB | Disc | Stoch | ✓ | ✓ | ✓ | MCTS 作策略改进算子 | 需完美模拟器 | MuZero |
| MuZero | 2019 | MB+规划 | On(自对弈) | MB | Disc | Stoch | ✓ | ✓ | ✓ | 学习的隐式模型支撑 MCTS | 规划昂贵、不连续动作 | … |

### 3.6 Modern LLM RL

| Algorithm | Year | 类型 | On/Off | MF/MB | Disc/Cont | Policy | RB | Critic | BS | Main Innovation | Main Weakness | Successor |
|-----------|------|------|--------|-------|-----------|--------|----|--------|----|-----------------|---------------|-----------|
| RLHF-PPO | 2022 | AC(PPO) | On | MF | token | Stoch | ✓(prompt) | ✓(RM+V) | ✗(MC) | RM + PPO + KL anchor 到 SFT | critic=actor 同规模，昂贵 | GRPO/DPO |
| DPO | 2023 | 偏好优化 | Off(静态) | MF | token | Stoch | — | ✗ | ✗ | 偏好→策略闭式重参数，跳过 RM/rollout | 偏离 RL 假设、离线偏弱 | 变体繁多 |
| GRPO | 2024 | P | On | MF | token | Stoch | ✓(prompt) | ✗ | ✗(MC) | 组内相对 advantage 替代 critic | 组内全对/全错时梯度为 0 | DAPO 等 |
| RLVR | 2024– | P | On | MF | token | Stoch | ✓ | 视实现 | ✗ | 可验证奖励取代 RM | 稀疏、可被 hack、任务面窄 | 前沿 |

> ⚠️ 表中 2024 之后的条目（DAPO、GSPO、Dr.GRPO 等）在 Blog 19/21 写作前必须重新搜索确认——本次规划时的三次网络搜索均未返回结果，信息以写作时为准。

---

## 四、易混淆概念出现地图

| 概念 | 首次正式定义 | 后续深化 |
|------|-------------|---------|
| On-policy vs Off-policy | 03（SARSA/Q-Learning） | 08（PPO 为何需要 old policy）、09（off-policy+replay）、14（offline=极端 off-policy）、19（GRPO 的 IS 修正） |
| Model-free vs Model-based | 02 | 16（学的模型拿来想象）、17（模型服务于规划） |
| Value-based vs Policy-based | 04 vs 06 | 07（AC 合流：value 为 policy 服务） |
| MC vs TD | 02 | 08（GAE-λ 是两者的插值）、18（LLM RL 天然 MC：无中间 bootstrapping） |
| Bootstrapping | 02（TD target） | 04（target network 为何必需）、05（deadly triad）、10（soft V 的 bootstrap） |
| Bias vs Variance | 02（MC 无偏/TD 有偏） | 07（baseline 只降方差不引入偏差）、08（GAE 旋钮） |
| Exploration vs Exploitation | 01（bandit） | 03（ε-greedy）、10（entropy 即内在探索）、21（LLM 的熵坍缩与温度） |
| Q(s,a) vs V(s) | 01 | 03（控制为何偏爱 Q）、07（A = Q − V）、09（连续空间 max Q 之难） |
| Advantage | 07 | 08（GAE）、19（group-relative advantage） |
| Replay Buffer | 04 | 09、14（静态数据集=不许再采样的 buffer）、18（LLM 只有 prompt buffer，无 transition buffer） |
| Target Network | 04 | 09（DDPG 的用法）、10（twin Q + target min） |
| Importance Sampling | 03（伏笔：Q-Learning 为何不需要） | 08（PPO ratio）、19（GRPO 裁掉 IS 项的争议） |
| Entropy | 07（正则项） | 10（最大熵目标=内在奖励）、18/21（LLM 采样与熵监控） |
| KL Divergence | 08（TRPO 约束） | 18（RLHF 的 KL penalty 防Reward hacking+语言漂移）、20（DPO 的隐式 KL） |
| Trust Region | 08 | 18（LLM 中的 KL/clip-anchor） |
| Distribution Shift | 05（相关样本） | 11（BC 的 compounding error）、14（offline 核心）、12（sim-to-real） |
| OOD Action | 14 | BCQ/CQL/IQL 三种解法对照 |
| Reward Model | 18 | 20（DPO 隐式跳过）、21（verifiable reward 取代） |
| Preference Optimization | 20 | RLHF/DPO/GRPO 统一视角 |
| Outcome vs Process Reward | 21 | test-time scaling 的关联 |

---

## 五、学习路径

**完整路径（默认，≈100h）**：01 → 02 → … → 22 线性推进。每个 ★★★ 算法完成：读文章 → 手算 toy example → 读论文指定 Section → （可选）跑 CleanRL 单文件实现。

**路线 A：Robotics / Human Motion（≈75h）**：
01–14 + 16 + 22（17 选读）。核心链：PPO → BC/GAIL → DeepMimic → AMP → Offline/Model-based。

**路线 B：LLM Post-training（≈60h）**：
01–08 + 18–22（09/10 有助于理解 replay/entropy 概念，可选）。核心链：policy gradient → PPO → RLHF → GRPO/DPO → RLVR。

**面试速成（≈50h）**：
01–08 + 10（SAC）+ 14（offline）+ 18–20。

**节奏建议**：每周 2 篇 ★★ 或 1 篇 ★★★，约 3 个月完成主线。

---

## 六、知识断层检查（规划阶段自查结果）

已发现并处理的断层：

1. **SAC 被挤压**：原方案的 Blog 09 需同时装下 DPG/DDPG/TD3/SAC 四个算法，而 SAC 是"最大熵"范式转变（目标函数都变了），且为必精读算法 → 拆为 09（确定性策略线）+ 10（SAC）。
2. **缺终章收束**：目标里"看到新方法能回答 10 个问题"（选型、调试、两条路线的统一结构）没有落点 → 新增 Blog 22 Capstone。
3. **GAIL → AMP 依赖**：AMP 的判别器直接复用 GAIL 结构，原结构已保证 11 在 13 之前 ✅。
4. **KL 与重要性采样提前引入**：TRPO（08）需要 KL、PPO 需要 importance ratio，但两者在 03 只埋伏笔——在 08 开篇的"Previous Algorithm"中补严格定义，避免断层。
5. **LLM 线的前置**：18 需要 PPO 全套（08）+ reward 概念（01）+ KL（08），不依赖 09–17 ✅。
6. **与已有旧文的关系**：`content/post/RL/` 已有 4 篇直觉向文章（RL_basic 1–4），覆盖 01–04、06–07 的直觉层，但无推导、无代码、无 toy example。**处理方式（默认）**：新系列独立完整成文，旧文作为"直觉先行阅读"在对应篇开头互链，不重写、不删除。
7. **Exploration 专门方法**（ICM/RND/count-based）：不设独立篇（防膨胀），在 10 与 21 的 Research Connection 中介绍并给出指针。
8. **2024–2026 前沿**：DAPO、GSPO、Dr.GRPO、agentic RL 等在 19/21 写作前必须搜索最新论文与官方 repo，当前规划不固定结论。

---

## 七、写作与资料规范（每篇执行）

- 原始论文 + 官方代码 + 高质量实现（优先级：CleanRL 单文件（教学最优）> Stable-Baselines3 > 官方 repo > Tianshou/TorchRL）。
- 每篇给论文"值得读的 Section / Figure / Equation"清单，不要求全文精读。
- Blog 18–21：写作前搜索当月最新资料，引用 DeepSeek-R1、Tulu 3、DAPO、GSPO 等原论文与官方 repo。
- 发布位置：`content/post/RL/RL_<NN>_<slug>/index.md`，与旧文同系列目录。
- 每篇发布后更新本文件的 Master Table 与进度表。

## 八、进度追踪

| # | 状态 | 发布路径 |
|---|------|---------|
| 01 | ✅ 已发布 | `content/post/RL/RL_01_foundations/index.md` |
| 02 | ✅ 已发布 | `content/post/RL/RL_02_dp_mc_td/index.md` |
| 03–22 | 未开始 | — |

---

## 附录：单篇统一模板（12 节）

1. 这篇解决什么问题？（具体问题开场，不先堆公式）
2. Previous Algorithm（Input→Model→Action→Environment→Reward→Update 六要素 + 核心瓶颈）
3. Core Idea（一句话本质）
4. Mathematics（每个变量、每项存在的理由、相对前代改了什么、删掉会怎样）
5. Algorithm Pipeline（数据流图：NN 模块 / 环境变量 / gradient vs stop-gradient / buffer / rollout vs update 时机）
6. Pseudocode（PyTorch 风格，公式 ↔ 代码逐行对应）
7. Toy Example（可手算的数值例子）
8. Why It Works（bias/variance、bootstrapping、exploration、distribution shift、overestimation、stability、sample efficiency）
9. Failure Modes（这个算法什么时候会失败）
10. 与前后算法对比（Previous → Current → Next 表格）
11. Research Connection（Games/Robotics/Humanoid/Human Motion/HOI/VLA/LLM Post-training，只连真正相关的）
12. Interview Questions（5–10 个有区分度的问题 + 简洁严谨答案）
