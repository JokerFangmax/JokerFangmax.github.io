---
title: "GG 04: 自监督 draping 与隐式缝纫版型"
date: 2026-09-29
math: true
tags:
  - Notes
  - Garment Generation
  - Self-Supervised Learning
  - Implicit Representation
categories:
  - Garment Generation
description: "物理损失自监督的谱系 PBNS→SNUG→DrapeNet（把仿真器从数据生产者降级为损失函数）与 ISP 的隐式缝纫版型（panel SDF + 缝合标签场 + UV 映射）——含 backward-Euler-即-损失的推导，以及'准静态优化的三种失败模式'与'SDF panel 拟合'两个数值实验"
---

# GG 04：自监督 draping 与隐式缝纫版型

> 📖 服装生成系列第四篇（前篇：[GG 03：人体上的布料——从子网格到隐式场](../GG_03_body_conditioned_garments/)）。
> 本篇把两条线索拧在一起：**物理损失自监督**（PBNS → SNUG → DrapeNet：不再跑仿真造数据，直接最小化布料势能）与 **ISP**（把 GG 02 的版型表示与 GG 03 的隐式场焊接成"隐式缝纫版型"）。
> 论文细节均对照 arXiv 原文核实（DrapeNet/PBNS 的作者归属与坊间流传差异较大，见 §8 勘误表）。

---

## 1. 这组论文解决什么问题？

GG 03 的两篇（TailorNet、SMPLicit）都是**仿真监督**：先跑 Marvelous Designer / Blender 造出 (身体, 服装) 对，再让网络回归。这条路有两个结构性代价：

1. **数据贵**：TailorNet 的 55,800 帧仿真数据约 29 小时生成 + 6.5 小时训练；SMPLicit 靠 Blender 裁剪增广到百万级样本；
2. **物理只在数据里**：网络学到的是"仿真器的输出长什么样"，不是物理本身——穿透、悬浮等物理违例只能靠数据覆盖度压制。

本篇第一条线（物理损失自监督）问：**能不能跳过数据，直接把物理仿真的目标函数当损失？** 布料在重力下的平衡态本来就是势能的局部极小——那把网络输出直接丢进势能函数 $\mathcal{E}(\mathbf{X})$（GG 01 §3 的膜应变 + 弯曲 + 重力 + 碰撞罚），最小化它就行：

$$
\mathcal{L}_{\text{phys}}(\mathbf{X}) = \mathcal{E}(\mathbf{X}) = \mathcal{E}_{\text{strain}} + \mathcal{E}_{\text{bend}} + \mathcal{E}_{\text{grav}} + \mathcal{E}_{\text{col}},
$$

**仿真器从"数据生产者"降级为"损失函数"**——不再需要跑它，只需要能对 $\mathbf{X}$ 求导。三代方法沿着这条线推进：PBNS（静态、逐件服装）→ SNUG（动态、逐件）→ DrapeNet（准静态、潜码条件、拓扑自由）。

第二条线（ISP）问的是 GG 03 结尾留下的悬案：**隐式场的拓扑自由（SMPLicit）与版型的可制造性（GG 02）能不能兼得？** ISP 的答案是隐式缝纫版型：panel 边界写成 2D SDF（版型空间里 panel 是**闭合**曲线——绕开了 GG 03 实验 3 证明的开曲面 SDF 歧义），缝合写成连续标签场，3D 形状由 UV→3D 映射给出——而且这套表示**可微到底**，能直接接物理损失。

### 方法流程鸟瞰

本篇两条线，先给骨架（细节在 §3–4）：

**物理损失自监督**（PBNS → SNUG → DrapeNet，共同流程三步）：

| 步骤 | 做什么 | 目的 |
|---|---|---|
| 1 | 网络直接输出服装网格（或位移） | 不跑仿真造数据——pose/shape 免费采样，要多少有多少 |
| 2 | 把输出代入布料势能：应变 + 弯曲 + 重力 + 碰撞罚 | 仿真器从"数据生产者"降级为"损失函数" |
| 3 | 势能当损失对网络权重反传 | 物理写进权重，不靠数据覆盖度压制穿透/悬浮 |

三代的差异全在步骤 1 的表示与输入输出：PBNS 逐件服装、静态（pose → 网格）；SNUG 加惯性项抓动态；DrapeNet 换 UDF 潜码 + 潜码条件变形场（一个网络管多件服装）。

**ISP**（隐式缝纫版型）四阶段——版型与物理的分工就在这个拆分里：

| 阶段 | 训什么 | 监督 | 目的 |
|---|---|---|---|
| 1 | panel 边界 SDF + 缝合标签网络 | 几何（L1 + 交叉熵） | 版型的形状与缝合语义——"是什么" |
| 2 | UV→3D 映射 | Chamfer + 法向 + 缝合一致性 | 版型怎么折到 3D——几何监督定型 |
| 3 | 单层 drape 网络（输出 UV 位移图） | 物理损失（零 GT 数据） | 把定型的版型穿到身体上——物理只在这里 |
| 4 | 多层 layering 网络 | 物理 + 服装间碰撞罚 | 多件叠穿不互穿 |

---

## 2. 背景最小集

| 需要 | 出处 |
|---|---|
| 布料势能的组成（膜应变/弯曲/重力/碰撞罚）与 XPBD | [GG 01](../GG_01_smpl_cloth_basics/) §3–4 |
| XPBD 从隐式 Euler 的"约束最小化"推导 | GG 01 §4.2（本篇 §4.1 会把它再推一步） |
| panel/edge/stitch 与四种版型编码、ISP 的表示条目 | [GG 02](../GG_02_pattern_representations/) §2, §5.4 |
| UDF vs SDF、开曲面零水平集污染 | [GG 03](../GG_03_body_conditioned_garments/) §3.1, 实验 3 |
| 服装 = pose×shape×style 函数、模板方法的拓扑死穴 | GG 03 §7 |

---

## 3. 表示法详解

### 3.1 PBNS / SNUG：模板网格上的位移（老表示）

PBNS 与 SNUG 的服装表示和 TailorNet 一脉相承（GG 03 §2.1）：**rest-pose 模板网格 + 位移**。PBNS 的位移来自 SMPL pose $\theta$（72 维）经 4 层 MLP 得到的嵌入乘一个 PSD 矩阵 $D$ 的 blendshape 形式，加到模板上再 LBS 蒙皮——**输出就是 pose 空间里的服装网格，映射唯一、纯静态**。SNUG 把输入扩到 $(\beta, \phi)$（体型 + 运动：当前 pose 与根关节速度），用 4 个 GRU 产生逐顶点位移，按 3 帧子序列训练。

关键约束：两者都**逐件服装训练一个网络**（SNUG 原文 "we trained one regressor for each garment type"），且依赖正确摆位的模板。DrapeNet 原文对此的批评是这两者"require training a separate network for each garment"——模板顶点级输出"severely limits their generalization abilities"。

### 3.2 DrapeNet：UDF 潜码 + 潜码条件的变形场

DrapeNet 的服装**不是模板网格**，而是 GG 03 的 UDF 路线：

- **每件服装 ↔ 一个 32 维潜码 $z$**：编码器 DGCNN 采样 10,000 个表面点，4 层 edge-conv（特征 64/64/128/256，K=16 邻居）max-pool 后线性层出 32 维；解码器是带位置编码（$\mathbb{R}^{63}$）的 MLP：线性到 $\mathbb{R}^{512}$、5 个残差块、对 $z$ 做 Conditional Batch Normalization、sigmoid 输出乘截断距离 $\delta=0.1$；
- 选 UDF 而非膨胀 SDF 的理由：精度更高、穿插更少、能表示服装的开曲面（GG 03 §3.1 的同一论证）；网格经 **MeshUDF**（可微 Marching Cubes）提取，因此**拓扑无关**——上装/下装各训一个独立的生成模型（同架构不同权重），归一化到 $[-1,1]^3$ 的上半/下半；
- **变形网络**（把 canonical 服装穿到任意 $(\beta,\theta)$ 身上）：

$$
\mathcal{D}(x, \beta, \theta, z) = W\big(x',\ \beta,\ \theta,\ W(x)\big),\qquad x' = x + \Delta x(x,\beta) + \Delta x_{\text{ref}}(x,\beta,\theta,z),
$$

  其中融合权重 $W(\cdot)$ 与粗位移 $\Delta x(\cdot)$ 沿用 Santesteban et al. 2022 / DIG 的 9 层 MLP（带 skip connection），**这一阶段只需要 SMPL 融合权重与形状位移，无任何仿真数据**；真正的新东西是**依赖服装的精化场**

$$
\Delta x_{\text{ref}}(\cdot) = B(\beta,\theta)\cdot M(x,z):\quad \underbrace{B(\beta,\theta)\in\mathbb{R}^{128}}_{\text{身体嵌入}}\times\underbrace{M(x,z)\in\mathbb{R}^{128\times3}}_{\text{潜码条件的位移矩阵}},
$$

  **一个网络给多件服装悬垂**的关键就在这个 $z$ 条件项——消融去掉它，结果明显失真（论文 Fig. 6，定性）；
- 另有一个小的 **Intersection Solver**：MLP 预测 $\Delta x_{\text{IS}}(x, z_{\text{top}}, z_{\text{bot}})$ 修正下装位置防止上下装互穿。

**读法**：DrapeNet 不是"预测合身服装的网络 + pose 变形网络"两个网络（常见误读，见 §8），而是**生成网络（UDF 潜码，只需静态 T-pose 网格监督）+ 潜码条件的变形场（物理损失自监督）**——款式自由度在 $z$ 里，物理在变形场的损失里，两者解耦。

### 3.3 ISP：隐式缝纫版型

ISP（NeurIPS 2023，EPFL：Ren Li, Benoît Guillard, Pascal Fua）把 GG 02 §5.4 的条目展开成完整方案：

$$
I_\Theta:\ \Omega\times\mathbb{R}^{|z|} \to \mathbb{R}\times\mathcal{N},\quad (x, z)\mapsto(s, c),
$$

- $\Omega=[-1,1]^2$ 是 2D 版型空间，$s$ 是到 panel 边界的符号距离，$c$ 是**缝合标签**（GG 02 §4.3：每点输出"最近边界边"的缝合类别，自由边/缝合边）；
- 每件服装一个潜码 $z$（auto-decoding，DeepSDF 式无 encoder；$|z|=32$ 见于官方代码），前/后片是共享 $z$ 的独立网络；
- 配套 **UV→3D 映射** $A_\Phi(x,z)\mapsto X$（AtlasNet 式）：版型空间的每一点映射到 3D 悬垂位置——这就是"rest shape = 版型"（GG 02 §1）的可微版本；
- 缝合一致性损失（GG 02 已给出）：

$$
L_{\text{consist}} = \sum \|A_{\Phi_f}(x,z) - A_{\Phi_b}(x,z)\|^2,
$$

  同一标签的前后片边界边折到 3D 后必须重合——**缝合第一次成为损失，而不是数据结构里的边对**。

**为什么是"最优结合"**：panel 在 2D 版型空间里是闭合曲线（SDF 适定，绕开 GG 03 实验 3 的开曲面歧义）；拓扑变化 = $z$ 的连续扰动（SMPLicit 的隐式优势保留）；可制造性保留（marching squares 提取边环 + 标签配对 → 版型）；UV 网格天然提供布料的材料坐标（GG 01 §3.1 的参考构型）与跨帧对应。**注意分工**（常见误读，见 §8.2）：版型网络本身不带物理——物理损失在第四阶段训练的是独立的 drape 网络（§4.3）。

---

## 4. 方法核心与数学推导

### 4.1 物理损失的正确姿势：backward Euler 即损失

先看 SNUG 的核心推导——它解释了物理损失谱系里"静态 vs 动态"的精确含义。GG 01 §4.2 推过：XPBD 的单步投影是隐式 Euler 积分写成约束最小化的一次迭代。完整的隐式 Euler 步是

$$
\mathbf{x}_{t+1} = \underset{\mathbf{x}}{\arg\min}\ \underbrace{\sum_i \frac{m_i}{2\Delta t^2}\big\|\mathbf{x}_i - \mathbf{x}_{t,i} - \Delta t\, \mathbf{v}_{t,i}\big\|^2}_{\text{惯性项}}\ +\ \underbrace{\mathcal{E}(\mathbf{x})}_{\text{势能}},
$$

（对 $\mathbf{x}$ 求梯度置零即得 $m(\mathbf{x}_{t+1}-\mathbf{x}_t-\Delta t\mathbf{v}_t)/\Delta t^2 = -\nabla\mathcal{E}(\mathbf{x}_{t+1})$，正是 backward Euler；承 Martin et al. 2011。）

**这就是 SNUG 的全部损失**：$\mathcal{L} = \mathcal{L}_{\text{inertia}} + \mathcal{L}_{\text{static}}$——训练目标不是别的，就是"我输出的序列每一步都应该是隐式 Euler 的解"。$\mathcal{L}_{\text{static}}$ 里是 StVK 膜应变 + Green 应变、二面角弯曲、重力、对身体的三次碰撞惩罚（2mm 安全边距）。SNUG 相对 PBNS 的两点推进因此可以精确表述为：

1. **PBNS 是上式丢掉惯性项的准静态极限**（输入都不含时间，pose→网格映射唯一；论文自述势能公式 "take into account" 不了时间维度）；SNUG 加回 $\mathcal{L}_{\text{inertia}}$，能抓动态（摆裙甩动）；
2. **材料升级**：PBNS 用 mass-spring（边长应变 $\|\mathbf{E}-\mathbf{E}_T\|^2$ + 法向 Laplace-Beltrami 弯曲），SNUG 换 StVK（SNUG 原文评 PBNS 的材料 "lead[s] to less expressive deformations"，且 "does not generalize to different bodies"）。

谱系里一个容易忽略的事实：**DrapeNet 倒退回了准静态**——原文明确 "Unlike in [Santesteban et al. 2022], we only consider the quasi-static state after draping, that is, without acceleration"。代价是抓不住动态摆动，收益是损失更简单、训练更稳。

### 4.2 各家的物理损失清单

| | PBNS (TOG'21) | SNUG (CVPR'22) | DrapeNet (CVPR'23) | ISP drape 网络 (NeurIPS'23) |
|---|---|---|---|---|
| 应变 | 边长应变（spring） | StVK 膜 + Green | 膜应变能 | StVK 膜 + Green（SNUG 同款） |
| 弯曲 | 法向 Laplace-Beltrami | 二面角 | 相邻面折叠能 | 二面角 |
| 碰撞 | 身体二次罚（4mm 阈值，$\lambda_c{=}25$，含层间布-布） | 三次罚 + 2mm 边距 | 身体-服装惩罚 | 身体碰撞罚 |
| 特有 | $\mathcal{L}_{\text{pin}}$（腰部固定顶点 L2，$\lambda_{\text{pin}}{=}10$） | $\mathcal{L}_{\text{inertia}}$ | $\mathcal{L}_{\text{pin}}$（下装防掉，$\lambda{=}0.1$）+ $\mathcal{L}_{\text{layer}}$（hinge：$\max(d_{\text{bot}} - \gamma d_{\text{top}}, 0)$，$\gamma{=}0.5$，保证下装更贴身）+ Intersection Solver | 多层 $L_{\text{GGcol}}$（外层顶点到内层的有向 hinge-三次罚）；版型阶段另有 $L_{\text{consist}}$（几何损失，非物理） |

三个设计细节值得记：

- **$\mathcal{L}_{\text{pin}}$ 是"缝进现实"的最小手段**：物理损失里没有任何东西阻止裤子整体滑落（重力只会往下拉），PBNS 钉住腰/臀附近顶点、DrapeNet 对下装加 pin——**真实服装靠腰带与身体摩擦对抗重力，简化模型里只能显式钉住**；
- **$\mathcal{L}_{\text{layer}}$ 用 hinge 不用 L2**：只惩罚"下装比上装离身体更远"的违例方向，不惩罚合理的层间距离——多层穿衣的软约束范式；
- **训练分布**：DrapeNet 的 pose 从 AMASS 采样、身体形状 $\mathcal{U}[-3,3]^{10}$——物理损失免费造数据的代价只是前向 + 反传，采样密度随便堆（对比 TailorNet 55,800 帧）。

### 4.3 ISP：四阶段训练与版型域的物理

ISP（arXiv:2305.14100，全称 *Multi-Layered Garment Draping with Implicit Sewing Patterns*）把管线拆成**四个独立训练的阶段**，版型与物理的分工就在这个拆分里：

1. **$I_\Theta$（SDF + 缝合标签）+ 潜码 auto-decode 联合训练**：损失 $L_I = \sum_{x\in\Omega} |s - s^{gt}| + \lambda_{CE}\cdot\text{CE}(c, c^{gt}) + \lambda_{reg}\|z\|^2$（L1 + 交叉熵 + 潜码 L2；λ 数值论文未给，推给 supplementary）；
2. **$A_\Phi$（UV→3D 映射）**，用已学到的潜码：$L_A = L_{\text{CHD}} + \lambda_n L_{\text{normal}} + \lambda_c L_{\text{consist}}$——Chamfer + 法向 + 缝合一致性（$L_{\text{consist}}$ 见 §3.3）。与 AtlasNet 的区别：**只在 panel 内部（$s\le0$）求值**，无需把方形 UV patch 拉伸到整片；网格化在 Ω 的规则三角网格上逐顶点提升到 3D、丢弃 $s>0$ 顶点、边界顶点沿 $-s\nabla s$ 投影到零水平集，用 MeshSDF/DeepMesh 式梯度（论文 Eq. 6）恢复网格抽取的可微性；
3. **$\mathcal{D}_s$（单层 drape 网络，SNUG 式物理自监督）**：MLP 输入 82+32 维（SMPL 72 pose + 10 shape + 潜码 $z$），输出 6N² 维——前/后两张 $N\times N\times3$ 的**UV 位移图**（代码 $N{=}200$）。损失即 §4.1 的 $\mathcal{L}_{\text{phys}}$（StVK 膜应变 + 二面角弯曲 + 重力 + 身体碰撞罚），**零 GT 数据**；
4. **$\mathcal{D}_m$（多层 layering，CNN）**：$L_{D_m} = \mathcal{L}_{\text{phys}} + \lambda_g L_{\text{GGcol}} + \lambda_r\|\mathcal{D}_m\|^2$，服装间碰撞 $L_{\text{GGcol}} = \sum \max(0, \epsilon - (\tilde v^*_o - \tilde v_i)\cdot n_i)^3$（外层顶点到最近内层顶点的有向距离 hinge-三次罚）。

**读法**：物理不碰版型网络——版型的形状先由前两阶段的几何监督定型，物理只训练"怎么把定型的版型穿到身体上"（$\mathcal{D}_s$ 的位移图）。这比"物理直接雕版型"（我们原来猜想的方案）更稳：版型几何有干净的学习信号（SDF/标签/Chamfer），物理这个非凸的麻烦被隔离在它唯一必需的地方。代价是版型空间里**不保证** drape 后的物理有效性——两层各学各的。

**数据**：Korosteleva & Lee 的版型数据生成器（PBS），400 shirts / 300 skirts / 200 trousers（测试各 20），只需 rest-pose（T-pose）draped 网格 + 版型对；drape 网络的 pose 从 AMASS 采样、shape 从 $\mathcal{U}[-2,2]^{10}$——与 DrapeNet 同款的自监督采样。

**关键数字**（测试集，与 DrapeNet 的 UDF 表示直接对比——同一团队的内战）：

| 分辨率 | CHD ×10⁻⁴ ↓ | 法向一致 % ↑ | 网格提取耗时 (V100) |
|---|---|---|---|
| UDF-128 / ISP-128 | 0.803 / **0.579** | 97.71 / **98.70** | 601 / **25** ms |
| UDF-256 / ISP-256 | 0.493 / **0.392** | 98.44 / **98.83** | 2379 / **77** ms |
| UDF-512 / ISP-512 | 0.424 / **0.349** | 98.09 / **98.89** | 13258 / **261** ms |

ISP 全面占优，且 UDF 在 512 分辨率反而变差（零等值面难学），ISP 无此问题。**30 倍的网格提取提速**是版型域表示的直接红利：3D UDF 要在体素网格上 Marching Cubes，ISP 只需在 2D UV 网格上查表 + 提升。drape 质量：交叠率 0.276% vs DrapeNet 0.536%；用户研究（64 人 884 份）偏好 62.69% vs 25.00%。

代码实现里与论文不一致的细节（对读代码的人有用）：SDF 头与标签头是**共享 trunk 的两个独立分支头**；$|z|=32$、shirt 前/后片标签数 11/10、pants 7、skirt 4（论文只说"标签数随类型"）；仓库只发布了推理/拟合代码，无训练代码。

---

## 5. 数值实验

实验代码沿 GG 01 的 12×12 布料 + 球体设置（结构弹簧 $k{=}10^4$ / 剪切 $10^2$ / 弯曲 $10$，碰撞罚 $k_c{=}500$，地板 $-0.8$ m）。所有数字为实测输出。

### 5.1 实验 1：物理损失准静态 draping 的三种求解路径

| 路径 | E (J) | 质心 (m) | 接触 | 穿透 | 结构应变 max |
|---|---|---|---|---|---|
| (a) XPBD 动力学（GG 01 同款，从 $z{=}1$ 静止下落） | **4.838** | +0.294 | 98/144 | 0.00 mm | 2.25% |
| (b) L-BFGS 最小化势能（同一初值） | **−11.286** | −0.800 | 0 | 0.00 mm | 0.00% |
| (c) L-BFGS，模板在身初值（(a) + 1mm 噪声） | 7.08 | +0.293 | 98 | 0.64 mm | 5.13% |

三个读数，一个比一个意外：

1. **(b) 找到了比 draped 更低 16.12 J 的构型——平铺在地板上**。这不是 bug：平铺地板的势能本来就比重挂球面低（质心高度差 × 总重）。**draped 状态根本不是全局最小，只是个亚稳态**——XPBD 里它稳定存在，是因为下落动能被接触摩擦耗散掉了。L-BFGS 的线搜索没有"动力学路径"的概念，一步跨过球体屏障（Wolfe 条件只看落点，不看路径），直接落到全局最小。**物理损失的"物理"只有能量，没有动量与耗散**——这正是它与众仿真的本质区别；
2. **(a) 是货真价实的局部极小**：残差力 $\|\nabla E\|_\infty \approx 1.6$ N（碰撞罚的力平衡残差），从 (a) 出发再跑物理损失精炼，E 不再下降。**仿真到准静态后停住的状态，物理损失动不了它**——两者在盆地内一致；
3. **(c) 揭示精炼的上限**：1mm 噪声的"模板在身"初值（$E_{\text{init}}=7.47$）只能精炼到 7.08，回不到 4.84。

顺带记录两个真实的踩坑（都写进了实验脚本注释）：第一版用 **Adam 直接优化顶点位置**——Adam 的步长与梯度尺度无关，每步固定挪 ~lr，6000 步把布料推到 $-12$ m、应变 950%、$E=4.9\times10^6$ J。物理损失需要带线搜索的优化器（L-BFGS）或多轮重启；第二版发现**单次 L-BFGS 会在非光滑罚函数上线搜索失速**（19 次迭代就停），多轮重启可继续下降。

### 5.2 实验 2：准静态优化的初值敏感性

**(i) 卷绕模板（等距卷到球上，不拉伸）**：浴帽式（上半球）与围裙带式（下半球）两个初值，精炼后都**整体滑离球面**（接触 0，E = 67.0 / 65.5 J）——无摩擦的势能里，没有任何东西阻止"已穿上"的布滑下来。真实训练里 PBNS/SNUG 靠 $\mathcal{L}_{\text{pin}}$ 与摩擦项对付这个；我们这个玩具里它裸露无遗。

**(ii) 在身模板 + 噪声扫描**（量化"模板要多好"）：

| 初始噪声 σ | $E_{\text{init}}$ (J) | 精炼终点 (J) | XPBD 平衡 (J) |
|---|---|---|---|
| 1 mm | 7.34 | 6.89 | 4.838 |
| 2 mm | 15.01 | 12.71 | 4.838 |
| 5 mm | 82.47 | 48.27 | 4.838 |
| 10 mm | 273.07 | 169.50 | 4.838 |

**精炼终点随噪声单调恶化，且回不到无噪声平衡**。1mm——远小于 5mm 缝纫公差——就永久抬高 2 J。原因：噪声制造的褶皱是弹簧系统自己的局部极小（拉伸锁定），准静态下降解不开；**动力学能解开它**，因为惯性能把顶点带过势垒、耗散再把动能吃掉。

**这就是物理损失方法的完整生存条件**：(1) 模板必须已贴合身体（PBNS/SNUG 的 rest-pose 模板、DrapeNet 的 canonical 网格）；(2) 网络在**大量 pose/shape 上联合训练**——单样本的局部极小被批次平均与网络平滑性洗掉，这是"学一个函数"比"解一个优化问题"的真正优势；(3) 需要显式处理整体滑落（pin/摩擦）。

### 5.3 实验 3：SDF panel + 缝合标签场（ISP 的核心组件）

手写 MLP（numpy 反传）：输入 2D 点的 Fourier 特征（频率 ×[1,2,4,8]，共 10 维），2×32 tanh 隐藏层，输出 1 维 SDF + 3 类缝合标签 logits。目标：GG 02 的 T 恤前片轮廓（9 段身体边 + 40 段领口 Bezier，48 段闭合环），标签 = 每段"缝不缝、缝到哪"（领口/肩侧缝合、下摆自由）。监督：4000 均匀点 + 2000 边界法向偏移点，SDF 用 L1、标签只在近边界（|s|<5cm）用交叉熵，Adam 训 4000 步。

| 指标 | 实测 |
|---|---|
| SDF 精度（3000 随机测试点） | mean 16.0 mm / median 11.2 mm / 95% 48.2 mm |
| 零水平集覆盖（轮廓加密点 → 最近零点） | mean 4.4 mm |
| 零水平集精度（零点 → 真轮廓线段） | median 9.2 mm / 90% 34.9 mm |
| 虚假零点（离真轮廓 >5cm） | 3.7% |
| 边界缝合标签精度 | 领口 95.2% / 肩侧 100% / 总体 99.1% |

三个读数：

1. **单一网络同时输出 (SDF, 标签)**：稀疏的"每段一个标签"被稠密化成标签场——任意查询点都能回答"最近边界缝不缝、缝到哪类"。这是 GG 02 §4.4 表里"隐式缝合"的实证：**缝合从数据结构（边对）变成了可微函数**；
2. **覆盖好（4.4mm）、精度略差（中位 9.2mm）**：真轮廓附近零点密集，但远场（低采样密度区域）出现 3.7% 的虚假零穿越——marching squares 提取边环前需要按连通域清理。隐式表示的"解码税"（GG 02 §5.4）在这里具体化；
3. **标签比 SDF 好学**（99.1% vs 毫米级 SDF）：标签是分段常数，对场的局部误差不敏感——**语义信息在隐式场里比几何信息更稳**。

---

## 6. 论文间关系

```
物理损失自监督线                          版型线
PBNS (TOG'21)                            2021 版型数据集 (JSON)
  静态 · 逐件 · mass-spring                 ↓ GG 06
    ↓                                    NeuralTailor (边序列+缝合分类)
SNUG (CVPR'22)                             ↓ GG 07–08
  动态 · 逐件 · StVK                     DressCode / SewingLDM
    ↓                                        ↑ 表示供给
DrapeNet (CVPR'23) ←──── UDF 潜码 ──── GG 03 SMPLicit
  准静态 · 潜码条件 · 拓扑自由              ↑
    │                                    ISP (NeurIPS'23) ← 本篇
    └── Intersection Solver / L_layer    panel SDF + 标签场 + UV→3D
         (多层穿衣)                       + SNUG 式 drape 网络（UV 位移图）
```

定量数字（DrapeNet 论文，CLOTH3D 测试集 30 件未见上装 + 30 件下装；**注意 baseline 是 DeePSD 与 DIG**——PBNS/SNUG 因逐件训练的设定无法在这个多服装测试集上对比，论文从未给出对它们的定量表格）：

| 指标 | DrapeNet | DeePSD | DIG |
|---|---|---|---|
| ED-top (mm) ↓ | 47.9 | **28.1** | 29.6 |
| ED-bottom (mm) ↓ | 27.3 | **18.3** | 20.0 |
| B2G 穿透率 top (%) ↓ | **0.9** | 7.2 | 1.8 |
| B2G 穿透率 bottom (%) ↓ | **0.3** | 3.4 | 0.8 |
| G2G 上下装互穿 (%) ↓ | **0.5** | 2.0 | 4.0 |
| 补充材料: 上装总能量 ↓ | **1.81** | 8.21 | 7.38 |
| 补充材料: 上装应变能 ↓ | **0.43** | 7.22 | 6.32 |

这个表本身就是一堂课：**DrapeNet 的顶点对应误差（ED）最差，物理量全部最好**。论文的辩解是悬垂问题存在多个物理有效解（同一件衣服可以合理地挂出不同褶皱），ED 对此不公平；补充材料里的物理能量差 4–16 倍才是物理损失路线的直接证据。187 人用户研究（3738 条意见）里 >50% 选 DrapeNet 为最真实（DIG <30%）。

谱系内部的速度/成本对比：

| | 数据生成 | 训练 | 推理 |
|---|---|---|---|
| TailorNet (仿真监督) | ~29 h | 6.5 h | 1–2 ms/帧（模型 GB 级） |
| PBNS | 0 | **15 min** | 455 FPS GPU 单帧 / 14286 FPS batch（模型 MB 级） |
| SNUG | 0 | **2 h** | 2.2 ms/帧（19 MB vs TailorNet 2114 MB） |
| DrapeNet | 0（生成网络用 CLOTH3D 静态网格） | — | 悬垂网络 ~5 ms/件（V100, 5K 顶点）；编码 25 ms；**MeshUDF 提网格 ~2 s**（256³ Marching Cubes） |

PBNS 的另一个值得记的对比：与同架构**有监督**（L2 对仿真数据）训练比，ED 15.52 vs 7.59 mm（差一倍），但穿透率 0.45% vs 3.15%（好 7 倍）——**物理损失用误差换物理一致性**，这是整条线的通用 trade-off。

---

## 7. 与 Work 3 的连接

Work 3（视频先验蒸馏 → 缝纫版型生成，SewingLDM baseline）与本篇的三处接口：

1. **准静态假设的边界**：我们从视频蒸馏的先验本质上是静态观测（单帧姿态下的悬垂形态），DrapeNet 的准静态设定与之一致——但如果目标服装是摆裙/飘带类，视频里的动态信息在准静态框架里无处安放。SNUG 的 $\mathcal{L}_{\text{inertia}}$ 是把动态加回来的现成模板（且不增加任何数据成本）；
2. **物理量当评测**：DrapeNet 表格的教训——生成版型 → 仿真 → 报物理能量/穿透率，比纯几何指标（ED/IoU）更能区分"看起来对"与"物理对"。Work 3 的评测协议里应该加这一组（版型 → Warp XPBD 仿真 → B2G/G2G 穿透率 + 应变能）；
3. **ISP 的缝合一致性损失**：SewingLDM 的 29 维边向量里缝合信息是显式 flag（GG 02 §3.3），生成时靠扩散模型自己维持缝合边长度匹配。$L_{\text{consist}}$ 式的显式损失（生成的缝合边对折到 3D 必须重合）是可以在版型域直接搬的**训练期正则**——不需要改表示，只需要一个可微的"缝合代价"头。

---

## 8. 局限与勘误

### 8.1 各家自列局限

- **PBNS**：静态（抓不住宽松服装的动态）；非法自穿插身体 pose 会污染结果；裙子仍有伪影；
- **SNUG**：难以预测 fine-level dynamics；未见动作上仍有身体-服装碰撞（需后处理）；**布料自碰撞未处理**；逐件训练；
- **DrapeNet**：ED 上不敌 DeePSD/DIG（物理有效多解的辩解）；Intersection Solver 消融后 G2G 从 0.5% 升到 1.1%——**多层互穿只被这个小 MLP 压着**；MeshUDF 提网格 2 s/件，比悬垂网络慢 400 倍——隐式表示的解码税在推理端再现；
- **ISP**：论文自列——准静态假设（无动态）；多层 draping **只变形外层**来解决碰撞（内层不动）；未来工作是引入动态与更准确的层间物理交互。加上我们实验看到的：marching squares 前的连通域清理是版型提取的脆弱环节（§5.3 的 3.7% 虚假零点）。

### 8.2 勘误表（写进正文的依据）

本组论文的坊间流传错误格外多，以下均已按 arXiv 原文核实：

1. **DrapeNet 的作者与题目**：真实论文是 *DrapeNet: Garment Generation and Self-Supervised Draping*（arXiv:2211.11277，CVPR 2023），作者 Luca De Luigi*、Ren Li*（同等贡献）、Benoît Guillard、Mathieu Salzmann、Pascal Fua（CVLab EPFL；De Luigi 兼 University of Bologna）。**不是** Fangzhou Hong，也不存在题为 "A Fast and Generalizable Model for Garment Draping" 的 DrapeNet 论文（arXiv 全文库只有这一篇 DrapeNet）；
2. **PBNS 的作者与 arXiv ID**：*PBNS: Physically Based Neural Simulator for Unsupervised Garment Pose Space Deformation*，Hugo Bertiche、Meysam Madadi、Sergio Escalera（Universitat de Barcelona + Computer Vision Center），arXiv:2012.11310，ACM TOG 2021。流传的 "arXiv:2109.14638"（实为一篇隐私政策问答论文）与 "Santesteban 的 PBNS" 均误——SNUG 论文里也明确把 PBNS 归于 Bertiche 等人；
3. **DrapeNet 不是"canonical 服装预测网络 + pose 变形网络"**：它是 UDF 潜码生成网络（只用 CLOTH3D 静态 T-pose 网格监督，$\mathcal{L}_{\text{garm}} = \mathcal{L}_{\text{dist}} + 0.1\mathcal{L}_{\text{grad}}$）+ 潜码条件的变形场，两部分职责如 §3.2；
4. **DrapeNet 的定量对比对象是 DeePSD 与 DIG**，不是 PBNS/SNUG/TailorNet（后者因逐件训练无法进同一测试集）；
5. **ISP 的 latent 是 auto-decoding**（DeepSDF 式，无 encoder），不是 CVAE；$|z|=32$ 只见于官方代码（GG 02 §9 已勘）。arXiv ID 为 **2305.14100**，全称 *ISP: Multi-Layered Garment Draping with Implicit Sewing Patterns*；
6. **ISP 没有"物理精炼版型"步骤**：不存在 L-BFGS 优化、边长权重分阶段退火、或作用在版型表示上的物理损失——物理只训练独立的 drape 网络 $\mathcal{D}_s$（SNUG 式，§4.3）。我们最初按谱系外推写错过一版，逐文件核对 ISP 与 SNUG/PBNS 原文后确认此步骤不存在（推理期唯一的优化是拟合逆问题：对 $z$ 做 Adam 1000 步、仅 Chamfer 损失）；
7. **SNUG 的 $\mathcal{L}_{\text{static}}$ 无边长项**：是 StVK 膜应变 + 二面角弯曲 + 重力 + 三次碰撞罚；"边长应变"是 PBNS 的 mass-spring 形式（GG 01 §3.2 的弹簧能正是这一档）。

---

## 9. 习题

**题 1**：从 $\mathbf{x}_{t+1} = \arg\min_\mathbf{x} \sum_i \frac{m_i}{2\Delta t^2}\|\mathbf{x}_i - \mathbf{x}_{t,i} - \Delta t\,\mathbf{v}_{t,i}\|^2 + \mathcal{E}(\mathbf{x})$ 出发，证明其最优性条件就是 backward Euler，并说明当 $\Delta t \to \infty$ 时该目标退化为纯势能最小化（PBNS 的设定）。
*提示*：对 $\mathbf{x}_i$ 求梯度：$\frac{m_i}{\Delta t^2}(\mathbf{x}_i - \mathbf{x}_{t,i} - \Delta t\mathbf{v}_{t,i}) + \nabla_i\mathcal{E} = 0$。$\Delta t\to\infty$ 时惯性项系数 $\to 0$。

**题 2**：实验 1(b) 里 L-BFGS 的线搜索"跳过"了球体屏障。解释为什么 strong Wolfe 条件无法阻止这种跳跃（提示：条件只约束**落点**的函数值与方向导数，不约束两点之间路径），并给出两个工程对策。
*提示*：对策一：信赖域（trust region）类方法限制步长；对策二：屏障惩罚改为对数壁垒（IPC 式，GG 01 §4.3）使势能在穿透方向真正无界。

**题 3**：实验 2(ii) 中精炼终点随噪声单调恶化。设一条边被噪声拉伸 $\varepsilon$，证明在只有这条边自由的局部系统里松弛到自然长度是能量下降的，但给出一个两边共享顶点的具体构型，使任何单顶点移动都不能同时降低两条边的能量——即噪声构型可以是局部极小。
*提示*：构造 $Z$ 形三顶点链：中间顶点被两条反向拉伸的边拉扯，移动它必使其中一条更长。

**题 4**：DrapeNet 的 $\Delta x_{\text{ref}} = B(\beta,\theta)\cdot M(x,z)$ 把"身体状态"与"服装潜码"做成**乘性交互**。对照 GG 03 实验 1（分解表示的交互赌注）：这为什么比加性组合 $\Delta x_a(\beta,\theta) + \Delta x_b(z)$ 更符合"悬垂 = 身体与服装的交互"这一物理事实？代价是什么？
*提示*：悬垂本质上就是交互（身体形状决定服装怎么挂），加性模型有 GG 03 证明的不可约下限；代价是交互项的采样覆盖问题——(β,θ,z) 三元组的联合分布要足够密。

**题 5**（思考）：ISP 的缝合一致性损失 $L_{\text{consist}}$ 只约束"同标签边折到 3D 后重合"，不约束"不同标签的边不重合"。构造一个退化解（网络输出使 $L_{\text{consist}}=0$ 但版型完全不可用），并讨论需要哪些额外损失/正则排除它。
*提示*：令 $A_\Phi$ 是常数映射——所有点映到同一点，缝合边"完美重合"，panel 塌缩成点。排除手段：$A_\Phi$ 的双射正则（面积/雅可比行列式下界）、版型 SDF 的重建损失、以及物理项（塌缩构型的应变能无穷大）。

---

*下一篇：[GG 05：版型 DSL 与数据基座](../GG_05_garmentcode_data/)（GarmentCode 的 OOP 版型程序 + GarmentCodeData 的 11.5 万件 made-to-measure 数据量产）*
