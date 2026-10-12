---
title: "视频生成 09: DreamReward——3D 生成的人类偏好对齐"
date: 2026-10-16
math: true
tags:
  - Notes
  - Video Generation
categories:
  - Video Generation
description: "「生成什么」四部曲收官，进入「如何变好」的第一个杠杆：对齐。DreamReward（ECCV 2024，TPAMI 2025 扩展为 DreamReward-X）把 RLHF 整套装备搬进 text-to-3D：25k 专家成对比较（Cap3D 图聚类选 5000 prompt → MVDream 生成 4-10 资产 → 过滤 2530 组 → 三维 1-6 分 rating + 排名 + 冲突实时校验）训练 Bradley-Terry 排名损失的 Reward3D（ImageReward 底座、冻结 80% 防过拟合），再由 DreamFL 把奖励直接写进 SDS：L_Reward ≈ L_SDS − λ_r·r（Eq.8）——reward 提供方向，SDS 项充当 KL 锚。GPTEval3D Elo 1480 vs MVDream 1224，user study 4.88/5.03/5.07。附 toy：奖励代理的过优化机制——分布内 corr 0.97 的代理，纯 reward 优化走 38σ 外推爆掉（真实偏好 −4.5 → −1288，代理却升到 +53）——hacking 不是学错，是把输入推到没标注过的地方；配平 λ 的锚定版消掉 90% 偏好差距（−4.5 → −0.48）；λ 扫描给出完整倒 U 曲线（峰值 λ=1，λ=5 时 −24.2）——论文 Fig.8 的三种 λ_r 形态就是这条曲线的左端、峰值与右端"
---

# 视频生成 09：DreamReward——3D 生成的人类偏好对齐

> 📺 视频生成系列第 9 篇。前篇：[视频生成 08：Physics3D——视频扩散教会 3D 高斯懂物理](../VG_08_physics3d/)。
> 论文：*DreamReward: Text-to-3D Generation with Human Preference*（arXiv:2403.14613，ECCV 2024，Junliang Ye 与 Fangfu Liu 共同一作，清华 & 生数科技）；TPAMI 2025 扩展版 *DreamReward-X*（Fangfu Liu、Junliang Ye、Yikai Wang、Hanyang Wang、Zhengyi Wang、Jun Zhu、Yueqi Duan）。
> 系列从「生成什么」（03–08：对象→场景→语义→物理）转入「如何变好」：对齐（09）→ 搜索（10）→ 引导（11）三个正交杠杆。第一杠杆：让人类偏好直接进训练循环。

---

## 1. 这篇论文解决什么问题？

**text-to-3D 输出与人类偏好不对齐**：模型能生成「一只猫」，但提示词说「跳舞的象」，生成的是站着的象（subject 对、properties 丢）；多脸伪影、质感廉价、配色难看——CLIP 分数不低，人不满意。

### 1.1 已有路线与其天花板

| 路线 | 代表 | 机制 | 天花板 |
|---|---|---|---|
| 更好的先验 | MVDream / ProlificDreamer | 多视角扩散、VSD | 先验分布本身偏离偏好：2D 社区已证明（ImageReward）预训练扩散模型的输出分布与人类偏好有系统差 |
| 评估靠人 | GPTEval3D / T3Bench | GPT-4V 当裁判 | 只评不改；贵且慢 |
| RLHF（NLP/2D） | InstructGPT / ImageReward / DPOK / DiffusionDPO | 偏好数据 → reward model → 微调 | 3D 领域没人做——没有偏好数据集、没有 3D 感知的 reward model、多步 SDS 管线难微调 |

### 1.2 DreamReward 的答案：三件套 + 一个推导

1. **3DRewardDB**：第一个 text-to-3D 偏好数据集——25k 专家成对比较；
2. **Reward3D**：第一个通用 text-to-3D 偏好奖励模型（Bradley-Terry 排名损失）；
3. **DreamFL**：把奖励以 $\lambda_r r$ 项写进 SDS 损失的直接微调算法（$\mathcal{L}_{\text{Reward}}\approx\mathcal{L}_{\text{SDS}}-\lambda_r r$），并有数学推导支撑。

TPAMI 扩展版 **DreamReward-X** 的增量是 reward-aware noise sampling——在保偏好对齐的同时释放文本驱动的多样性（本篇 §6.4 详述；TPAMI 全文在付费墙后，该节只覆盖公开信息并明确标注）。

---

## 2. 所需背景最小集

1. **RLHF 最小集**（RL 系列已铺垫）：Bradley-Terry 模型——把「A 比 B 好」的排名转化为胜率 $\sigma(r_A-r_B)$ 的极大似然；reward model 在偏好对上训练后，作为人类偏好的可微代理；微调时需 KL 锚防止策略跑飞（PPO 的 KL 惩罚 / DPO 的隐式锚）。
2. **SDS**（VG 02/03/08）：$\nabla_\theta\mathcal{L}_{\text{SDS}}\approx\mathbb{E}[\omega(t)(\hat\epsilon_\phi(\mathbf{x}_t,t,y)-\epsilon)\,\partial\mathbf{g}/\partial\theta]$，等价于最小化渲染分布与扩散先验的加权 KL。本篇要在它上面动手术。
3. **MVDream**：多视角扩散——一次生成 4 视角一致图像，DreamFL 的底座（也是偏好数据的生成器）。
4. **ImageReward**：2D 版前驱（本文 Reward3D 的底座架构）——18k 偏好对训练的文生图 reward model，证明了「偏好数据 → BT 损失 → reward 反馈微调」在生成模型上可行。

---

## 3. 方法总览

**编号约定**：正文中的 Eq.N 一律指论文原公式编号；被引用的公式块右侧以 \tag{N} 标出同一编号，可与论文对照。

**一句话主线**：输入一段文字提示，先用人工两两比较的数据训出一个「哪个 3D 资产更好」的打分器，再把这个打分当作奖励项挂进原有的 SDS 训练目标里，边保住先验边顺着人类偏好微调，输出更贴合提示词、更好看的 3D 资产。

| 步骤 | 做什么（白话） | 为什么需要这一步 |
|---|---|---|
| ① 造候选资产 | 从现成的大型 3D 提示词池里聚类挑 5000 个有代表性的提示词，用 MVDream 每条生成 4–10 个 3D 资产当「候选选手」（§4.1） | 3D 领域没有任何现成偏好数据，必须从零自建；候选由 MVDream 生成，被评的和被微调的是同一个模型，打分器恰好定义在要优化的输出附近 |
| ② 人工排名标注 | 标注员对每个候选按「文-3D 对齐 / 整体质量 / 多视角一致性」三维打 1–6 分并排名，管线实时检测排名矛盾、冲突对二次校验（§4.1） | 让人打绝对分既难又不稳定，「谁更好」的序判断认知负担低、一致性好；把「人会犯错」当一等公民设计进管线，数据才可靠 |
| ③ 训练打分器 Reward3D | 用 Bradley-Terry 排名损失在 25k 个比较对上训练：输入提示词和资产的多视角图，输出一个偏好分（§4.2） | 微调需要的是能对任意 3D 资产连续打分、梯度能反传的「人类偏好代理」——人本身没法进训练循环 |
| ④ 奖励写进 SDS（DreamFL） | 在原有 SDS 损失上减去一个奖励项；奖励权重与两边损失量级配比、随训练课程式上升，且只在去噪图较清晰的时段计入（§4.3） | 光有打分器改不了生成结果；而纯奖励优化会被「钻空子」——SDS 项同时充当锚，把解钉在模型自己会生成的资产范围内（§5 toy 验证） |
| ⑤ 纯奖励收官 | 训练末尾关掉 SDS 项，用大权重的奖励再微调约 200 步（§4.3） | 锚定优化到后期改善变慢，短期放开锚能再提升美感与对齐；安全余量来自步数短，不是结构保证 |

```
┌─ 上游: 偏好基础设施 ────────────────────────────────┐
│ Cap3D 提示词 → 图聚类(语言模型相似度) → 5000 代表 prompt │
│   → MVDream 每提示生成 4~10 个 3D 资产                  │
│   → 过滤模式坍塌 → 2530 组, 25,304 候选比较对            │
│   → 标注员三维 1-6 分 rating(对齐/质量/一致性) + 排名    │
│     + 实时数据结构检测排名冲突 → 二次校验                │
│   → Reward3D (Bradley-Terry 排名损失, ImageReward 底座) │
└──────────────────────────────────────────────────────┘
                      │ r(y, x̂_t, c)
                      ▼
┌─ 下游: DreamFL 反馈学习 ─────────────────────────────┐
│ MVDream SDS 优化 NeRF:                                │
│   L_Reward = L_SDS − λ_r · r(y, x̂_t, c)  (Eq.8)      │
│   λ_r = t(L_SDS, r)·μ  (量级配平 + 课程)              │
│   t < t_threshold 才计 reward 项;                     │
│   收官阶段关掉 L_SDS, 纯 reward 微调 200 步            │
└──────────────────────────────────────────────────────┘
```

上游管「把人类偏好变成一个可微的标量场」，下游管「在不毁掉先验的前提下顺着这个场走」。整个框架是 RLHF 的 3D 移植，但锚的形式变了——没有显式的 KL 惩罚项，**先验锚藏在 $\mathcal{L}_{\text{SDS}}$ 里**（§5 的 toy 会证明这一项就是信任域）。

---

## 4. 核心机制完整推导

### 4.1 标注管线：排名比评分可靠（§4.1）

数据工程的三个决策都值得学：

- **prompt 选择**：不随机采，用基于语言模型相似度的图聚类从 Cap3D（Objaverse 的重标注）里选 5000 个**代表性** prompt——覆盖分布而不是堆量；
- **资产来源**：每个 prompt 用 MVDream 生成 4–10 个资产——**被评的和被微调的是同一个模型族**（好处：reward 场直接定义在模型的输出流形附近；代价：见 §8 的循环性问题）；
- **标注设计**：每资产按三维（文-3D 对齐/整体质量/多视角一致性）打 1–6 分，按均分排名；**实时数据结构检测排名冲突**（A>B、B>C、C>A），冲突对标记后二次校验——把「人会犯错」当一等公民设计进管线。过滤掉模式坍塌后剩 2530 组、25,304 候选比较对。

### 4.2 Reward3D：Bradley-Terry 排名损失（Eq.1）

同一 prompt 下排名出 $n$ 个资产，两两组合最多 $\binom{n}{2}$ 对。设 $x_i$ 优于 $x_j$，奖励模型 $r_\theta$（输入提示 $T$、资产 $x$、相机 $c$）的损失：

$$
\operatorname{loss}(\theta)=-\mathbb{E}_{(T,x_i,x_j,c_i,c_j)\sim\mathcal{D}}\Big[\log\big(\sigma\big(r_\theta(T,x_i,c_i)-r_\theta(T,x_j,c_j)\big)\big)\Big]. \tag{1}
$$

推导只需两步：BT 模型假设 $x_i$ 胜 $x_j$ 的概率是 $\sigma(r_i-r_j)$（sigmoid 链接的 Luce 选择公理），排名数据转成成对胜负后做极大似然——**排名信息的价值在于它对人只要求序判断（认知负担低、一致性好），BT 把序还原成隐分数**。这与 InstructGPT/ImageReward 的 reward model 同型。

实现细节里的两条经验：① 底座用 ImageReward（图像 + 文本特征 cross-attention + MLP 出标量），**冻结 80% 的 transformer 层**——观察到快速收敛后过拟合，冻结适量层数反而更好（小数据微调大模型的常规智慧）；② 单卡 4090（24 GB）、batch 8、AdamW、lr 1e-5。

### 4.3 DreamFL：把奖励写进 SDS（Eq.2–8）

**出发点**：SDS 最小化的是到预训练分布 $p_t(\mathbf{x}_t|y)$ 的 KL（Eq.2–3），但这个分布本身偏离人类偏好（问题一），且 SDS 的多视角一致性有限（问题二）。理想目标应是到**偏好对齐 + 3D 感知**的分布 $p_t^r$ 的 KL（Eq.4）。

**关键构造**：直接得到 $p_t^r$ 不可能，但（受 ProlificDreamer 用 LoRA 近似分布的启发）**近似一个分布的预测噪声就足以近似该分布**。设理想噪声预测 $\epsilon_\phi^r=\epsilon_\phi-\delta\epsilon$，把 $\mathcal{L}_{\text{Reward}}$ 的梯度（Eq.5）里的 $\epsilon_\phi^r$ 展开（Eq.6–7）：

$$
\delta\epsilon=-\frac{\partial r(y,\hat{\mathbf{x}}_t,c)}{\partial\epsilon_\phi(x_t,t,y)}=\lambda_r\frac{\partial r(y,\hat{\mathbf{x}}_t,c)}{\partial g(\theta,c)}, \tag{6}
$$

$$
\nabla_\theta\mathcal{L}_{\text{Reward}}\approx\mathbb{E}\Big[\omega(t)\big(\underbrace{\epsilon_\phi(\mathbf{x}_t,t,y)-\lambda_r\tfrac{\partial r}{\partial g(\theta,c)}}_{\epsilon_\phi^r}-\epsilon\big)\frac{\partial\mathbf{g}(\theta,c)}{\partial\theta}\Big], \tag{7}
$$

其中 $\hat{\mathbf{x}}_t=(\mathbf{x}_t-\sigma_t\epsilon_\phi)/\alpha_t$ 是一步去噪预测（附录 Eq.11 的详细版：$\delta\epsilon=\lambda\,\partial r/\partial\hat{\mathbf{x}}_t\cdot\partial\hat{\mathbf{x}}_t/\partial\mathbf{x}$，链上 $\partial\hat{\mathbf{x}}_t/\partial\epsilon_\phi=-\sigma_t/\alpha_t$ 常数并入 $\lambda$）。整理回损失形式（Eq.8）：

$$
\mathcal{L}_{\text{Reward}}(\theta)\approx\mathcal{L}_{\text{SDS}}(\theta)-\lambda_r\,r(y,\hat{\mathbf{x}}_t,c). \tag{8}
$$

**批判性读法（必须诚实指出）**：Eq.6 的第一步 $\delta\epsilon=-\partial r/\partial\epsilon_\phi$ 是**构造而非定理**——「reward 对噪声预测的梯度等于两分布噪声预测之差」没有任何从 BT 模型或 KL 目标出发的推导支撑，论文称之为 theoretical proof 名不副实。它的合法性完全来自经验：消融（Fig.8）显示这个形式 work。把它读作「**用 reward 的梯度场定义一个修正后的分数场**」比读作「推导出理想分布」更准确。

**实现细节（§5.3 + 附录 0.B）**：$\lambda_r=\mathrm{t}(\mathcal{L}_{\text{SDS}}, r)\times\mu$，加权函数 $\mathrm{t}$ 配平两个损失的量级，$\mu$ 走课程（论文原文「increases from 0 to 0.25 throughout the training process until reaching 0.6」——0.25 与 0.6 自相矛盾，疑为笔误，取其意：课程式上升）；Algorithm 1 里只有 $t<t_{\text{threshold}}$ 的低噪声段才计 reward 项（高噪声段去噪预测 $\hat{\mathbf{x}}_t$ 太糊，reward 信号无意义）；**收官阶段关掉 $\mathcal{L}_{\text{SDS}}$，纯 reward 大 $\mu$ 微调 200 步**提升美感与稳定性。训练 40 分钟（与 MVDream 相当）；DreamFL 也可当独立微调器用——已有 3D 资产 2 分钟改善纹理细节与对齐。

---

## 5. toy 实验：奖励代理的过优化与先验锚

DreamFL 的 Eq.8 只有两项——reward 提供方向、$\mathcal{L}_{\text{SDS}}$ 提供锚。这个结构可以用一个极小系统机制化：$x\in\mathbb{R}^4$，「模型生成的资产云」$\sim\mathcal{N}(0,I)$（800 项）；真实偏好 $h(x)=-\|x-x^*\|^2+1.5e^{-3\|x-x_b\|^2}$（主峰 $x^*$ 距云心 2.12σ——**先验错位**：SDS 收敛点 $h=-4.5$，偏好上限 $h=0$）；标注者以 $P(i\succ j)=\sigma(1.5(h_i-h_j))$ 给出 25k 带噪比较对；Reward3D 抽象为 8 维随机 ReLU 特征上的 BT 拟合 $r̂$（**容量有限的真实奖励模型**）；DreamFL = 对 $p+\lambda r̂$ 梯度上升（$p=-\|x\|^2/2$ 是 SDS 锚）：

| 方案 | 真实偏好 $h$ ↑ | 轨迹峰值 $h$ | 代理 $r̂$ | 距云心 (σ) |
|---|---|---|---|---|
| **A $\lambda=0$**（标准 SDS） | −4.500 | −4.500 | +0.17 | 0.00 |
| **B 纯 reward 无锚** | −1287.5 | −0.06 | **+53.4** | 37.89 |
| **C 先验 + 1.0·reward**（DreamFL 配平） | **−0.480** | −0.468 | +2.81 | 1.46 |
| **C′ 固定 $\lambda=3$**（未配平） | −4.683 | −0.081 | +6.63 | 4.19 |
| **D = C 后 400 步纯 reward 收官**（放大版） | −1403.6 | −0.064 | +55.6 | 39.48 |
| **E = C，$r̂$ 只用 10% 比较对训练** | −0.588 | −0.582 | +2.69 | 1.39 |
| **F oracle 直接优化 $h$** | 0.000 | 0.000 | +3.62 | 2.12 |

**λ 扫描**（先验 + λ·reward，400 步）：$\lambda=0/0.1/0.3/0.5/1/2/3/5$ → 终点 $h=-4.50/-3.51/-2.83/-1.94/\mathbf{-0.48}/-0.68/-4.68/-24.21$。

七条读法：

- **A vs C（对齐收益）**：先验错位下 SDS 停在 $h=-4.5$；配平的锚定 reward 把 89% 的偏好差距消掉（−4.5 → −0.48）。这就是 Table 1 里 478 vs 375 的机制版。
- **B（hacking 的纯形态）**：代理一路上扬到 +53，真实偏好崩到 −1288，走出云 38σ。关键在诊断行：**分布内 corr($h,\hat r$)=0.969——代理没有学错**。失效全部发生在分布外：优化器把输入推到了没有标注过的地方，随机 ReLU 特征在那里无界外推。Goodhart 定律的机制版：**不是模型烂，是优化太用力**。
- **C（L_SDS 就是 KL 锚）**：Eq.8 的两项各司其职——reward 给方向，SDS 项把解钉在资产云内（1.46σ）。DreamFL 等价于一个**没有显式 KL 惩罚项的 PPO**：信任域不在损失里，而在「先验项的梯度恰好指向模型自己的分布中心」这个事实里。RLHF 的标准装备（KL 锚）在这里换了件衣服。
- **C′ + λ 扫描（倒 U 曲线）**：$\lambda$ 从 0 扫到 5，终点 $h$ 先升后崩（峰值 λ=1）——教科书级 **Goodhart 曲线**。论文 Fig.8 的三种形态精确对应：$\lambda_r=0$ 是左端（只有 subject 没有 properties）、常数 10000 是右端（崩溃）、加权函数 $\mathrm{t}()$ 是找峰值。**量级配平不是工程细节，是这条曲线的横轴定位问题**。
- **D（收官阶段是悬崖边跳舞）**：论文的 200 步纯 reward 收官，在 toy 里放大成 400 步——轨迹峰值 −0.064 就贴着最优，但继续走就是 B 的下场。收官阶段之所以在论文里 work，是因为步数短、$\hat{\mathbf{x}}_t$ 在低噪声段——**安全余量是步数给的，不是结构给的**。
- **E（数据量的位置）**：10% 标注量只让终点从 −0.48 退到 −0.59——这个 toy 里瓶颈是**外推**而非**标注方差**。但真实世界里 25k 对要覆盖 5000 个 prompt 的偏好多样性，瓶颈恰恰是覆盖缺口（论文自认多样性受限）——toy 的外推问题和现实的覆盖问题是同一个 Goodhart 曲线在两个轴上的投影。
- **F（锚的代价）**：oracle 能到 0，C 停在 −0.48——最后 0.66σ 走不完，因为锚不允许离开先验太远。**对齐幅度与保持先验质量是一对显式交易**，λ 是这条交易线的滑块（也解释了为什么对齐方法普遍报告「提升对齐但略降多样性/一致性」——Table 2 里 DreamReward 的 consistency 5.30 输给 MVDream 的 5.38）。

---

## 6. 训练与推理细节 + 论文实验

### 6.1 超参速查

| 项 | 值 |
|---|---|
| 偏好数据 | 2530 组 × 4–10 资产，25,304 比较对；三维 1–6 分 + 排名 + 冲突校验 |
| Reward3D | ImageReward 底座 + cross-attn + MLP；冻结 80% 层；4090 24GB、batch 8、AdamW lr 1e-5 |
| DreamFL | MVDream 底座；$\lambda_r=\mathrm{t}(\mathcal{L}_{\text{SDS}},r)\cdot\mu$；$t<t_{\text{threshold}}$ 才计 reward；收官 200 步纯 reward |
| 代价 | 训练 40 分钟（≈MVDream）；已有资产微调 2 分钟 |

### 6.2 主对比（Table 1，110 个 GPTEval3D prompt）

| 方法 | Human Win ↑ | GPTEval3D ↑ | Reward3D ↑ | ImageReward ↑ | CLIP ↑ |
|---|---|---|---|---|---|
| DreamFusion | 97 | 1000 | −1.597 | −1.489 | 0.224 |
| Fantasia3D | 167 | 1006 | −1.582 | −1.521 | 0.222 |
| ProlificDreamer | 246 | 1152 | −0.195 | −0.639 | 0.252 |
| Latent-NeRF | 287 | 1173 | −0.012 | −0.350 | 0.257 |
| MVDream | 375 | 1224 | 0.246 | −0.541 | 0.243 |
| **DreamReward** | **478** | **1480** | **2.594** | **1.833** | **0.274** |

（表下附与人工评估的 Spearman ρ：GPTEval3D 1.00、Reward3D 1.00、ImageReward 0.80、CLIP 0.60。）读表三点：① CLIP 几乎无区分度（0.222–0.274）——**像素级图文相似度测不出偏好**，这是整个对齐方向的价值论证；② Reward3D 与 GPTEval3D 都达到 ρ=1.00，但 Reward3D 是轻量前馈模型（推理快、免费），GPT-4V 贵——reward model 当自动评估器的卖点；③ Reward3D 给自家方法打分存在循环性（§8 再谈），ρ=1.00 的对照缓解但不消除。

**user study**（Table 2，30 人、20 prompt、1–6 分）：对齐/质量/一致性/平均 = **4.88/5.03/5.30/5.07**，vs MVDream 3.88/4.40/**5.38**/4.55、ProlificDreamer 3.90/3.78/3.13/3.60。注意**一致性一栏 DreamReward 第二**（5.30 < 5.38）——reward 模型的三维标注里一致性只占其一，优化主体是对齐与质量；偏好率 65%。

**Elo 评估**（附录 Table 3/4，GPTEval3D 锦标赛，六维 = Text-Asset / 3D-Plausibility / Text-Geometry / 几何细节 / 纹理细节 / Overall）：混战中 DreamReward 六维全第一（**1567 / 1305 / 1465 / 1479 / 1622 / 1442**；对照 MVDream 1098 / 1148 / 1251 / 1256 / 1325 / 1271，Text-Asset 一列 ProlificDreamer 1262 反而高于 MVDream）；单独对 MVDream 的重评（换 $\lambda_r$ 后统一重生成 110 资产）**1221 vs 779**（六维 Ours 1185–1221 vs MVDream 779–815）。

### 6.3 消融

- **$\lambda_r$ 形态**（Fig.8，提示词「喷火的大理石老鼠半身像」）：$\lambda_r=0$ 退化为标准 SDS——**只有 subject（老鼠）没有 properties（喷火）**；常数 10000——形崩；加权函数 $\mathrm{t}()$——质量与对齐兼得。对应 toy λ 扫描的左端/右端/峰值。
- **底座无关性**（Fig.7）：DreamFL 换到 DreamFusion 架构（SD2.1-base）照样提升——算法不绑定 MVDream。
- **Reward3D 的 3D 感知**（Fig.3）：把资产的一个视角图换成错误内容（不一致/多脸负例），Reward3D 打低分——2D 打分器（ImageReward/CLIP）做不到。

### 6.4 DreamReward-X（TPAMI 2025）增量（摘要级）

TPAMI 扩展版（Fangfu Liu、Junliang Ye、Yikai Wang、Hanyang Wang 等）的核心增量是 **reward-aware noise sampling**：在生成过程中用 reward 信号引导噪声采样，**释放文本驱动的多样性**同时保住偏好对齐——直指基础版的软肋（论文自己承认多样性受标注数据规模限制；toy 的 F 行也显示锚定优化天然压缩多样性）。**诚实标注**：TPAMI 全文在 IEEE 付费墙后且未挂 arXiv，本节仅覆盖作者主页公开的摘要级信息，公式与数字未核对——增量部分以官方开放信息为准。

---

## 7. 与其他论文的联系

**系列内**：

- **RL 系列正式汇合**：Bradley-Terry 排名损失 = RLHF 的 reward model 训练目标（InstructGPT 同型）；DreamFL vs PPO/DPO 的差异是本篇最有讲头的对比——PPO 用显式 KL 惩罚、DPO 把锚写进目标函数的闭式解、DreamFL 的锚是 SDS 项本身（toy 的 C 行证明了这个锚真的能当信任域用）。
- **VG 03 / VG 08（SDS 家族三态）**：同一台蒸馏机器，蒸馏三种东西——VG 03 蒸馏几何先验（并暴露均值回归的坑）、VG 08 蒸馏物理参数（低维参数反而安全）、本篇蒸馏偏好分数（一个标量场）。蒸馏目标的维度越低，SDS 的病态越可控。
- **VG 07（LangScene-X）**：$\mathcal{L}_{mask}$ 监督查询行为、Reward3D 监督偏好排序——**「监督你要的东西」家族**；区别是 07 监督的是可观测的激活图，本篇监督的是不可观测的偏好（要经 BT 转译）。
- **VG 06（VideoScene）**：DDPNet 的奖励是任务 MSE（客观可算），Reward3D 的奖励是人类偏好（要学的代理）——reward 信号的「来源谱系」从客观到主观，VG 10 会继续这条谱系（verifier 引导搜索）。
- **VG 12 预告**：对齐杠杆的完整版图——本篇是「reward 进训练损失」，VG 10 是「reward 进推理搜索」，两篇合成「何时用对齐信号」的完整答案。

**系列外**：

- **ImageReward / DPOK / DiffusionDPO**：2D 前驱三兄弟（reward 反传 / 策略优化 + KL / 偏好对直接优化）。DreamReward 的独特处境是 3D：没有现成偏好数据（要自建管线）、微调对象是多步 SDS 管线而非扩散采样器（所以锚天然存在）。
- **InstructGPT（RLHF 谱系源头）**：标注管线的冲突检测、三维 rating、排名设计都是它的精神后裔——**把标注工程当方法贡献**，这条传统从 InstructGPT 一路传到 3D。
- **ProlificDreamer**：「近似预测噪声即近似分布」的经验来自它的 LoRA 技巧——DreamFL 把这个技巧从「近似真分布」挪用为「构造理想分布的梯度场」。

---

## 8. 局限与思考

**论文自认的局限**：偏好数据集规模限制多样性；未来在更大数据集上优化 Reward3D、加入更多相机与朝向信息。

**我的补充思考**：

1. **Eq.6 是构造不是证明**：$\delta\epsilon=-\partial r/\partial\epsilon_\phi$ 没有从任何第一性原理推出，论文的「theoretical analysis」实为「合理构造 + 经验证实」。这不是否定——构造得好照样 work——但读文献时把「有推导」和「有推导支撑的构造」分开，是避免被 abstract 骗的基本功。（顺带：μ 的「0.25 直到 0.6」自相矛盾，复现者需自行试参。）
2. **评估的循环性**：偏好数据由 MVDream 生成、Reward3D 在其上训练、DreamFL 再微调 MVDream 管线、Table 1 里 Reward3D 指标给 DreamReward 打最高分——**裁判由被告的家属担任**。Spearman ρ=1.00（对人工评估）是有效缓解，但人工评估本身只覆盖 110 个 prompt；对 MVDream 之外底座（Shap-E 等前馈模型）的偏好覆盖，数据集里没有。
3. **toy 的核心教训——hacking 是分布问题不是精度问题**：分布内 corr 0.97 的代理照样被 hack 到 −1288。这意味着 reward model 的迭代方向不是「更大更准」，而是**标注覆盖要跟上优化器的足迹**（对抗式数据收集、或者给 reward 外推加不确定性惩罚）。RLHF 社区的 over-optimization 文献早已给出同样结论——3D 不豁免。
4. **收官纯 reward 阶段的结构性风险**：200 步无锚优化在 toy 里是悬崖边的舞蹈（峰值 −0.06 之后就是 −1400）。步数短是安全余量的全部来源——这在论文里 work，但任何把收官阶段加长的「改进」都可能翻车。
5. **一致性输给 MVDream 的深层原因**：偏好是三维标注的加权（人评的是「整体好不好」），锚定优化在提升对齐与美感的同时微降一致性（5.30 vs 5.38）——**多目标偏好压成标量 reward 的必然代价**。DreamReward-X 的 reward-aware noise sampling 把多样性拉回一等公民，正是在补这个洞。
6. **「第一个」的真正含义**：论文反复强调 first（first 3D preference dataset / first 3D reward model）。first 的价值不在算法深度（BT + SDS 都是新瓶装旧酒），而在**把 3D 对齐的问题定义、数据基础设施、评估协议一次配齐**——这类工作定义了一个方向的可做性，后续工作（包括 DeepMesh 用 DPO 做 3D 网格）都站在它上面。

---

## 附录 A：toy 实验设定

偏好对齐 toy（numpy，seed 固定）：$x\in\mathbb{R}^4$；资产云 $X\sim\mathcal{N}(0,I_4)$（800 项）；真实偏好 $h(x)=-\|x-x^*\|^2+1.5e^{-3\|x-x_b\|^2}$，$x^*=(1.5,1.5,0,0)$（距云心 2.12σ）、$x_b=(-1.5,1.5,0,0)$（局部凸起，代理容量装不下）；先验 $p(x)=-\|x\|^2/2$（SDS 锚）。标注：25k 随机对，$P(i\succ j)=\sigma(1.5(h_i-h_j))$。Reward3D 抽象：8 维随机 ReLU 特征（含线性跳连）上的 BT logistic 损失，衰减学习率 GD 3000 步（全量 / 10% 两个规模）；分布内 corr($h,\hat r$)=0.969（>2σ 的 340 项上 0.941）。优化：带步长范数上限（0.1）的梯度上升，lr 0.1，400 步；reward 梯度用数值差分。各行目标：A $\nabla p$；B $\nabla\hat r$；C $\nabla(p+\hat r)$；C′ $\nabla(p+3\hat r)$；D 自 C 终点起 $\nabla\hat r$ 400 步；E 同 C 但 $\hat r$ 用 10% 对训练；F $\nabla h$。相对论文的简化：偏好是静态已知函数（现实里只能通过标注间接观测）；优化对象是 4 维向量而非 NeRF；SDS 锚抽象为二次势（真实 SDS 的锚是「扩散模型自己会生成什么」，随优化漂移）。

---

*下一篇：[视频生成 10：Video-T1——视频生成的测试时扩展](../VG_10_video_t1/)（对齐杠杆把 reward 用在训练时，搜索杠杆把它搬到推理时：一次生成花更多算力，用 verifier 在噪声候选里挑最好的。线性搜索、树搜索、reward 引导——LLM 的测试时扩展范式移植到视频生成，与 RL 系列的 RLVR/测试时搜索正面汇合）*
