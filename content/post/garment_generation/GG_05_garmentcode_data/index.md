---
title: "GG 05: 版型 DSL 与数据基座——GarmentCode 与 GarmentCodeData"
date: 2026-09-29
math: true
tags:
  - Notes
  - Garment Generation
  - Sewing Pattern
  - Procedural Generation
categories:
  - Garment Generation
description: "版型怎么被'编程'：GarmentCode 的 Edge/Panel/Component/Interface OOP 设计、made-to-measure 的公式依赖与参数重映射（不是钳位）、缝合的弧长展平与 ruffle 褶量均摊；GarmentCodeData 怎么用采样规则量产 11.5 万件跨 5000 体型数据——配 ∂版型/∂测量 Jacobian、Interface 展平、组合空间占据问题三个数值实验"
---

# GG 05：版型 DSL 与数据基座

> 📖 服装生成系列第五篇（前篇：[GG 04：自监督 draping 与隐式缝纫版型](../GG_04_self_supervised_draping_isp/)）。
> GG 02 §5.2 说过 GarmentCode 把版型从"几何数据"变成"程序"——本篇把那个条目展开成完整方案，并讲它的衍生品：**GarmentCodeData 怎么用这套程序化表示量产出整个领域最大版型数据集**。
> 论文与 pygarment 代码细节均对照原文核实（见 §8 勘误表）。

---

## 1. 这组论文解决什么问题？

GG 02 讲过版型的三种编码。其中**程序 DSL** 路线回答的问题与其他三种都不同：

- 向量序列（NeuralTailor 系）问"**怎么学**一个版型"；
- 程序 DSL 问"**怎么写**一个版型"——版型是一段代码，设计 = 编程。

这个问题有两个完全不同的受众：**设计师**（要一个比 CAD 软件里的自由曲线更高层的抽象——"连一条裙子"而不是"画 40 条边"），以及**机器学习**（要一个能批量生成带 GT 的数据的过程——程序空间采样即数据集）。GarmentCode（SIGGRAPH Asia 2023）先解决前者，GarmentCodeData（ECCV 2024）立刻把后者兑现：**11.5 万件 made-to-measure 版型 + 仿真 + 渲染**——此后 GarmentDiffusion、SewingLDM 的训练数据（GG 07–08）几乎全部来自它（勘误：DressCode 是例外——它训练在 2021 数据集上，见 GG 07 §8.2）。

本篇的核心论点：**程序化表示同时是设计语言与数据引擎，而它的两个关键机制——made-to-measure 的公式依赖与 Interface 的弧长展平——都是可以用数值实验拆开验证的确定性算法**（不涉及任何学习）。这在全系列里是异类：唯一一篇"零神经网络"的论文篇，却供应了后面所有生成模型的数据。

### 方法流程鸟瞰

两条管线，先给骨架（细节在 §3–4）：

**GarmentCode——怎么"写"一个版型**（设计师视角）：

| 步骤 | 做什么 | 目的 |
|---|---|---|
| 1. 组装 | 从组件库挑 bodice/sleeve/skirt… 递归组合成整件 | 高层抽象："连一条裙子"而不是"画 40 条边" |
| 2. 缝合 | Interface 配对（袖窿↔袖口），弧长展平 + ruffle 系数 | 边数/边长不等的两边也能缝，褶量沿缝线均摊 |
| 3. 调款 | style 参数（领宽/裙长/袖型…） | 设计自由度；参数空间即合法空间（重映射保证，不是钳位） |
| 4. 量体 | 26 项身体测量经制版公式折算成几何 | 换身体 → 同一程序自动产出合身版型（made-to-measure） |

**GarmentCodeData——怎么量产数据**（机器学习视角）：**采样 → 量体 → 仿真 → 渲染**四步流水线（§4.1 逐条展开）——程序空间非均匀采样出 ~11.2 万个设计（默认值概率压制怪款式），每设计拟合规范 + 随机两个身体（合身由 made-to-measure 构造保证），XPBD 悬垂仿真（成功率 ~72%，失败即弃不修复），渲染出图（下游 SewingLDM 的草图条件直接读这里的渲染图）。

---

## 2. 背景最小集

| 需要 | 出处 |
|---|---|
| panel/edge/stitch 原语、Korosteleva JSON 格式、1-1 边对缝合 | [GG 02](../GG_02_pattern_representations/) §2 |
| 四种版型编码对照（DSL 条目） | GG 02 §5.2, §5.5 |
| GG 02 习题 3（Interface 展平的提示） | GG 02 §10 |
| XPBD（GarmentCodeData 的仿真引擎） | [GG 01](../GG_01_smpl_cloth_basics/) §4.2 |

---

## 3. GarmentCode：版型的面向对象语法

### 3.1 类层次：Edge → Panel → Component → Interface

对照论文与 pygarment 代码（v2.0.0 起并入 GarmentCode 主仓库），版型程序的七个核心类：

| 类 | 职责 | 关键成员 |
|---|---|---|
| `Edge` | 一条边 | 基类**就是直线边**（start/end 顶点）——没有 StraightEdge 类 |
| `CircleEdge(Edge)` | 圆弧 | 第三点只存相对 $y$ 坐标 $c_y$，$x$ 固定 0.5——圆弧单参数 |
| `CurveEdge(Edge)` | Bezier | 1 个控制点 = 二次、2 个 = 三次 |
| `EdgeSequence` | 有序边列 | Interface 与 panel 轮廓的载体 |
| `Panel(BaseComponent)` | 闭环边序列 + 3D 放置 | 6D placement（旋转 + 平移） |
| `Component(BaseComponent)` | 子组件容器 | `subs` 列表——**递归组合**，任意层级 |
| `Interface` | 组件的可缝合边集合 | `(panel, edges, ruffle=1., right_wrong=False)` |

两个设计决定值得展开：

1. **控制量的局部性**：所有曲线控制量定义在边的局部坐标系里（起终点连线为横轴）——平移/旋转不变、均匀缩放保曲率。这与 NeuralTailor 4 维编码的 $(c_x, c_y)$（GG 02 §3.1）是同一思想：**边几何的参数化必须与摆放无关**，程序才能在 made-to-measure 换身体时保持语义不变；
2. **递归组合**：裙 + 上衣 → 连衣裙只是把两个组件挂到同一个父节点下（compound 裙组件约 50 行代码）。组件库 14 个文件（bodice/sleeve/skirt/pants/godet/hood...），`MetaGarment` 用 `globals()[name]` 按名字动态查类组装——**"程序"与"组装程序的程序"同构**，这是 GG 09 里 LLM agent 写 GarmentCode（GarmentoPIA）能成立的结构基础。

### 3.2 made-to-measure：26 项身体测量 + 派生公式 + 参数重映射

版型参数分两组：

- **body 组（26 项）**：围度 6（bust/waist/hip 等）、宽度 5、纵向 8、点间距 2、距离 2、角度 3；
- **style 组**：每个参数 `{v, range, type, default_prob}`，type ∈ {float, int, select, select_null, bool}——`default_prob` 是"取默认值的概率"，直接成为 GarmentCodeData 采样规则的表达（§4.1）。

派生量由公式定义：`waist_level = height − head_l − waist_line`、`_base_sleeve_balance = shoulder_w − 2`、`_bust_line = (1−1/3)·vert_bust_line + 1/3·bust_line`……这些公式是制版教科书（Armstrong 体系）的行业知识直接编进代码。

**约束的真实机制是参数重映射，不是钳位**。以领口宽度为例（`bodice.py`）：style 参数是归一化量 $t\in[-0.5, 1]$，真实领宽由分段线性插值得到：

$$
\text{collar\_w} = \begin{cases} \mathrm{lin\_interp}(0,\ \text{neck\_w},\ 1+t) & t < 0 \quad(\text{收窄支}) \\[2pt] \mathrm{lin\_interp}(\text{neck\_w},\ \text{shoulder\_w}-4,\ t) & t \geq 0 \quad(\text{常规支}) \end{cases}
$$

这不是 clamp：$t$ 的定义域本身就是合法域——**参数空间即合法空间**。换身体时插值端点跟着身体测量走，同一 $t$ 自动映射到新区间（§5.1 实验 1 数值验证：两支斜率连续、拼接处无跳变、换身体后仍落在新合法区间内）。这就是 GarmentCodeData 敢跨 5,000 体型量产的机制保证——**每件都合身不是被验证的，是被构造保证的**。

### 3.x Interface：缝合从"边对"升级为"组件接口"

GG 02 §4.3 给过 Interface 的定位：bodice 的"袖窿 Interface"与 sleeve 的"Armhole Interface"配对，**一条抽象 stitch = 一对 Interface**，而 Interface 可含多个 panel 的多条边——突破 2021 JSON 格式的 1-1 限制。语义上最关键的两个机制：

1. **弧长展平（边数不等）**：Interface 两边的边数 $m \ne n$ 时，两边都按累计弧长比例参数化到 $[0,1]$，在两边分割点（映射到公共参数域）的**并集**处插入新顶点——每个子段 $[\frac{k}{K}, \frac{k+1}{K}]$ 上两边一一对应。这是 GG 02 习题 3 的算法，本篇 §5.2 给出实现与守恒检验；
2. **ruffle 系数**：缝合允许两边长度不等（褶皱/荷叶边）。ruffle = B/A 总长比，弧长展平把这个比值**均摊**到每个配对段——褶量沿缝线均匀分布，不在接缝处堆积（§5.2 数值验证：比值 std $\sim 10^{-15}$）。

### 3.4 使用方式：DSL + GUI 双轨

设计者在浏览器 GUI 中组装组件、调 style 参数，程序在后台生成可执行的 DSL——GUI 是 DSL 的视图，不是独立编辑器，**版型的唯一真相是程序**。论文的评测方式是复现一件 Birch 连衣裙（多组件 + made-to-measure 全链路走通）加一张工具能力表，**没有用户研究、没有定量指标**——DSL 论文的评测困境：没法比"编程语言哪家强"。代码侧的几个怪癖（`Edge.__eq__` 的比较逻辑疑似写反、Panel 与 Component 是兄弟类而非继承关系、SimpleLapel 的 Turtle 命名 bug）不影响 DSL 语义，但读源码时要小心。

---

## 4. GarmentCodeData：采样规则即数据集

### 4.1 生成管线：采样 → 量体 → 仿真 → 渲染

四步流水线（前三步是确定性程序，第四步 GPU 仿真）：

1. **设计采样**：从 GarmentCode 组件库**非均匀**采样 style 参数。三条设计原则：(a) 参数对齐身体测量（裙/袖长不超过腿长）；(b) 每个参数以概率 $p_i$ 取默认值（日常惯例），否则在范围内均匀采样——$p=0.8$ 用于抑制左右不对称；(c) 与专业设计师合作改进默认模板（bodice/pencil skirt/pants）并**新增 hood 组件**。质量控制剔除自交 panel、空 pattern、纯腰带、拖地全长、易滑落裙等；
2. **身体生成与量体**：欧洲 CAESAR 子集 1,700 个 3D 扫描注册 + PCA 得统计身体模型，从中采 5,000 个体型（每 PCA 参数 $\sim\mathcal{N}(0, 0.7)$）。每设计拟合**两个**身体——规范中性身体 + 一个随机体型（每设计 2 个 pattern，made-to-measure 的直接兑现）。身体测量 25 项全自动：landmark 定位 + 测量平面 ±2cm 内以 5mm 步长取极值、凸包周长模拟皮尺；
3. **仿真**：NVIDIA Warp 的 XPBD 实现（官方 fork 补了 point-triangle / edge-edge 自碰撞——Warp 原生没有）。JSON 版型 → 合并 stitch 顶点 → CGAL 约束 Delaunay 三角化（目标 1cm 平均边长，保留原边长做 rest length）→ XPBD 悬垂至静态平衡（判据：<1.5% 顶点帧间移动 >0.04cm；每样本限时 5 分钟，RTX 3090 平均 ~30 秒）。**成功率 ~72%**——失败样本直接丢弃，不人工修复；
4. **渲染**：每样本附基础渲染（至少含 `render_front.png`——下游 SewingLDM 的 loader 直接读它做 sketch 条件）。渲染细节（纹理/光照/相机配置）论文未写，是数据集文档的空白。

### 4.2 规模与构成（含三个必须知道的"没有"）

| 项 | 数字 |
|---|---|
| 总样本 | **115,195**（论文口径 115,000；SewingLDM 论文写 120,000） |
| 唯一设计 | ~112,016（每设计 2 个身体拟合：default + random） |
| 体型 | 5,000（CAESAR-PCA） |
| 仿真成功率 | ~72%（每批 5,000 设计 ≈3,200 通过；~7h CPU + ~44h 仿真/批） |
| 官方划分 | train 102,660 / val 6,270 / test 6,265 |
| 材质 | 3 种纺织材料（弯曲软度不同，各 ~1/3） |
| 类别构成 | 无精确计数，仅采样概率：约 1/3 上装、1/3 下装、1/3 全身（dresses/jumpsuits） |

**三个"没有"**（引用这个数据集前必须知道）：

1. **论文没有 panel/边/stitch 数的分布统计**（均值、最大值均无）。常被引用的 "37 panel × 39 edge" 出自 SewingLDM 的预处理配置而非数据集论文，且方向常被记反——正确是 **39 个 panel 槽位（恰好 = GarmentCode 语义 panel 类别数）× 每panel 37 条边**，这是下游 padding 上限，不是数据集官方统计；
2. **论文没有定义任何基准任务、评测指标或 baseline**——它只是 dataset + pipeline 论文。仅有的定量表是仿真消融（加"pull"约束后 neutral body self-collision 中位数 10→0）和自动量体 vs CAESAR 人工测量的偏差（bust +2.81cm、hip +1.99cm、肩宽 −4.57cm）；
3. **论文正文没有 train/val/test 划分**——官方 split 只存在于 v2 发布物，且按**设计 id** 划分：同一设计的两个身体拟合进同一 split，体型在 split 间共享——**unseen 的是设计，不是身体**。

数据 v2 总大小 **368 GB**（DOI: 10.3929/ethz-b-000690432，CC BY 4.0）；代码已并入 GarmentCode 主仓库（`pattern_sampler.py --name garmentcodedata` 可按种子复现官方批次）。

---

## 5. 数值实验

实验均为 numpy 实现、确定性算法（无学习成分）——这本身就是 DSL 路线的特点：**可拆开验证的是机制，不是训练出来的统计性质**。

### 5.1 实验 1：made-to-measure 的公式依赖——∂版型/∂测量

玩具 bodice 前片：8 项身体测量（height/chest/waist/hip/shoulder_w/neck_w/arm_len/back_len）经真实制版法风格的公式（前片宽 = 胸围/4 + 1cm、袖窿深 = 胸围/8 + 10.5cm、腰线高 = 身高 − 身高/8 − 20cm……）派生出 12 顶点。数值差分求 Jacobian $\partial(\text{24 个顶点坐标})/\partial(\text{8 项测量})$：

| 测量 | 影响坐标数 | 解析检验（数值 = 理论） |
|---|---|---|
| height | 2/24 | ∂腰线高/∂height = **0.8750** ✓（= 1 − 1/8） |
| chest | 11/24 | ∂前片宽/∂chest = **0.2500** ✓；∂袖窿深/∂chest = **0.1250** ✓ |
| waist | 1/24 | ∂右腰 x/∂waist = **0.2500** ✓ |
| hip | 1/24 | — |
| shoulder_w | 2/24 | — |
| neck_w | 3/24 | — |
| arm_len / back_len | 0/24 | 只影响袖片/后片（本玩具只画前片） |

三个读数：

1. **Jacobain 稀疏且可解释**：每项测量只影响它公式上触及的顶点。对比神经网络的稠密 Jacobian——**可解释性不是玄学，是依赖结构的直读**。"改胸围会动哪里"这个问题在 DSL 里查公式表就能回答；
2. **系数就是制版知识**：0.875 = 1 − 1/8（腰线 = 身高 − 头长 − 20，头长 = 身高/8）——**公式里的每个数字都是百年制版经验**。made-to-measure 不是学出来的，是把行业知识直接编进代码；
3. **约束重映射的连续性**：领宽参数用 GarmentCode 的真实机制——归一化参数 collar_t ∈ [−0.5, 1] 经分段线性插值映射到真实领宽（t<0 收窄支映到 [0, 颈宽)，t≥0 常规支映到 [颈宽, 肩宽−4]，对照 `bodice.py`）。扫描 t 全程：两支斜率 15.0 / 21.0（端点跟着身体测量走），拼接处（t=0，领宽=颈宽）**连续无跳变**。换身体（颈宽 13/肩宽 36 → 颈宽 18/肩宽 46）：同一 t=0.6 映到 24.4 / 32.4 cm，都落在新合法区间内 ✓。**"换身体仍有效"的机制 = 公式（连续传播）+ 重映射（参数空间即合法空间）**——不是把越界参数推回来，而是参数的定义域本身构造在合法区间上。这就是 GarmentCodeData 敢跨 5,000 体型量产的底气——每件都合身不是被验证的，是被构造保证的。

### 5.2 实验 2：Interface 弧长展平与 ruffle 的均摊

两组真实缝纫场景（算法实现见实验脚本，GG 02 习题 3 的答案）：

**袖窿(3 边, 23cm) ↔ 袖山(5 边, 30cm)**——袖山比袖窿长 30% 是真实缝纫的"袖山吃势"（cap ease，让袖子有立体鼓形）：

```text
展平成 7 个配对段（两边分割点并集）; B/A 比值 mean 1.3043, std 5.6e-15（均匀）
守恒: A 侧 23.0000 = 23.0000 ✓, B 侧 30.0000 = 30.0000 ✓
对照·朴素逐边配对: 比值 [0.6, 1.4, 0.75]（参差）, 段长误差和 19.00 cm
```

**腰(2 边, 20cm) ↔ 裙上缘(6 边, 70cm)**——3.5 倍褶量：

```text
展平成 7 个配对段; B/A 比值 mean 3.5000, std 1.6e-15（均匀）
守恒: A 侧 20.0000 ✓, B 侧 70.0000 ✓
对照·朴素逐边配对: 段长误差和 54.00 cm
```

两个读数：

1. **弧长展平的三重保证**：总长守恒（精确）、每段比值恒等于 ruffle（std 在机器精度）、顶点只在两边原有分割点的并集处插入（不发明新几何）。朴素逐边配对在边数不等的场景里误差以 10cm 计——**多边对多边缝合必须参数化**；
2. **ruffle 的语义被算法精确化**："1.8 倍褶"不是模糊的设计语言，是"每个配对段的 B/A 长度比恒为 1.8"——褶量按弧长均分。荷叶边的物理形态（缝出来褶皱均匀）是算法的直接推论。

### 5.3 实验 3：采样规则 → 数据覆盖度（占据问题）

玩具组件库模仿 GarmentCode 的离散选择结构：bodice（领口 3 × 门襟 2 × 袖型 3 × 衣长 2 = 36）+ skirt（型 3 × 腰带 2 = 6）+ pants（型 2 = 2）+ 组合语法（dress = bodice × skirt），离散设计空间 $|D| = 260$。独立均匀采样 $N$ 件，对照占据问题解析解 $\mathbb{E}[\text{不同设计数}] = |D|(1-(1-1/|D|)^N)$：

| $N$ | 采到设计数 | 解析期望 | 偏差 |
|---|---|---|---|
| 100 | 84/260 | 83.1 | 1.0% |
| 1,000 | 254/260 | 254.5 | 0.2% |
| 10,000 | 260/260 | 260.0 | 0.0% |
| 115,000 | 260/260 | 260.0 | 0.0% |

读数：**115k 采样在玩具空间里把每个拓扑覆盖 ~442 次**——离散组合数根本不是多样性的瓶颈；瓶颈在**连续 style 参数与 made-to-measure 身体变化**（每个"设计"是一个参数族，不是单点）。组件库再大 50 倍（$|D|=13000$），115k 采样每设计也仅 ~8.8 件——"seen/unseen 划分"必须在连续参数或组合层级上才有意义。这为 GG 07–08 埋下伏笔：SewingLDM 在 GarmentCodeData 上实测的"同 prompt 结构多样性低"，正是组合空间覆盖不均的症状。

### 5.4 数据集不设考题——下游各比各的

§4.2 的"没有"之二直接决定了这个领域的评测生态：GarmentCodeData 不定义基准任务，考题全部由下游工作自拟。以 SewingLDM（GG 08，我们的 Work 3 baseline）为例，它在 v2 官方 split 上的主指标是 clothes-to-body distance（渲染视图 3D 重建质量）：SewingLDM 2.20cm vs DressCode 3.69 / SewFormer 5.45 / RichDreamer 6.19 / Wonder3D 6.54cm，外加用户研究与 runtime——**没有版型几何精度的直接指标**（如顶点级 panel IoU、stitch 拓扑正确率）。数据集论文不设考题的后果：下游指标不可比，"在 GarmentCodeData 上 SOTA"这句话本身没有单一定义。

---

## 6. 论文间关系

```
2021 Korosteleva & Lee 数据集（JSON 格式 + PBS 仿真, 20 类版型）
        ↓ 格式成为事实标准
GarmentCode (SIGA'23)  ←—— 本篇
  版型 = 程序（OOP DSL, made-to-measure, Interface）
        ↓ 采样规则
GarmentCodeData (ECCV'24)  ←—— 本篇
  11.5 万件 × 5000 体型 + Warp XPBD 仿真 + 渲染
        ↓ 训练数据
GarmentDiffusion / SewingLDM (GG 07–08)
（DressCode 例外: 训练在 2021 数据集, ~19k 件——GG 07 勘误）
        ↓ 表示革新对照
GarmentoPIA (GG 09): LLM agent 写 GarmentCode 程序——DSL 回归
```

**组内分工**：GarmentCode 解决"单个版型怎么写"（设计师视角），GarmentCodeData 解决"版型空间怎么采样"（机器学习视角）——同一 DSL 的两次兑现。与 GG 06 的 NeuralTailor 恰好是镜像关系：NeuralTailor 学"观测 → 版型"的逆映射，GarmentCodeData 给"参数 → 版型"的正映射 + 全套 GT（版型、悬垂 mesh、UV、渲染）——**逆问题的训练数据，只能从正问题来**（GG 10 的 Dress-1-to-3 依赖的也是这一点）。

**谱系定位**：GarmentCode 是全系列唯一的"表示先于学习"的路线——它不解决任何学习问题，但 GG 07 以后所有生成模型的**数据分布**由它的组件库决定。这个事实的深远影响常被低估：生成模型学到的是"组件库能表达的世界"，组件库外的设计（论文自列的 panel 开洞、口袋、荷叶边组件）对下游模型是不可想象的——**数据基座的天花板就是生成模型的天花板**。

---

## 7. 与 Work 3 的连接

Work 3（视频先验蒸馏 → 缝纫版型服装生成）的 P0 baseline 是 SewingLDM，其全部训练数据就是 GarmentCodeData v2。这带来三个直接约束：

1. **数据即天花板**：模型能生成的设计分布 = 组件库 × 采样规则的乘积分布（比组件库本身更窄——`default_prob` 偏向默认值）。我们复现实测 SewingLDM 同 prompt 结构多样性低，不是模型问题，是数据分布的性质。视频先验蒸馏的价值恰在这里：真实视频里的设计分布超出组件库，蒸馏能把先验注入数据分布之外的区域；
2. **复现即用其管线**：P0 直接用 v2 官方 split（102,660 训练样本）、`render_front.png` 做 sketch 条件、random_body 拟合 + `<id>_meta.json` 做身体条件（与 SewingLDM 官方一致）。三个规模口径（115,195 / 120,000 / 125,865）的差异必须在配置里选定一个并写进 README——这是复现时实际会踩的坑；论文正文没有 split 说明，split 只在 v2 发布物里；
3. **仿真就绪免费获得**：每样本自带 XPBD 悬垂 mesh + UV map + 分割标签——将来做视频蒸馏的几何监督（版型 → 悬垂 → 渲染链路）不需要自己仿真。另一个值得注意的数字是 **72% 仿真成功率**：约 28% 的合法设计因 Warp 仿真超时/不收敛被丢弃——这批样本版型合法、只是物理上没站住。如果我们的生成方法能覆盖这 28%（生成后不要求 Warp 收敛），等于免费扩展了组件库的设计空间。

---

## 8. 局限与勘误

### 8.1 局限

**GarmentCode 论文自列 8 条表达局限**（诚实得罕见），代表性的五条：panel 内部不能开洞（口袋、腰带袢做不到）；没有折叠线（fold line——面料上预压的结构褶）；不支持多层缝合（面料叠缝面料）；不支持缝在面料上的元素（贴袋、流苏、刺绣）；不支持三条以上边交汇于一点的缝法。**GarmentCodeData 侧再叠一层**（论文自列）：无面料分层、无锐利褶裥、无五金件（拉链/纽扣/挂钩）、无法表达整套 outfit；材料建模极简（只有 3 种）；CAESAR 身体型偏欧美成年人群。

**表达力 ≠ 采样分布**：即使组件库能表达某个设计，采样规则也未必采到（default_prob 偏向默认值、p=0.8 抑制不对称）。下游模型实际学到的是更窄的乘积分布——"数据基座的天花板"（§6）要打双重折扣。

### 8.2 勘误表

| 流传/想当然的说法 | 事实 | 依据 |
|---|---|---|
| 标题是 "GarmentCode: Programming Garment Designs" | **"GarmentCode: Programming Parametric Sewing Patterns"** | arXiv:2306.03642，SIGGRAPH Asia 2023（ACM TOG 42(6)） |
| 代码在 `mary-nya/pygarment` | 仓库已迁移到 **`maria-korosteleva/GarmentCode`**（pygarment v2.0.0 起并入主仓库），旧地址 404 | GitHub |
| 26 项身体测量 = "6 围度 + 3 段宽 + 14 距离 + 3 角度"（GG 02 初版也这么写） | 正确分类：**围度 6 / 宽度 5 / 纵向 8 / 点间距 2 / 距离 2 / 角度 3** | 代码 body 参数组（GG 02 已同步勘正） |
| 约束靠 clamp（越界参数推回边界） | 真实机制是**归一化参数的线性插值重映射**（§3.2）——参数定义域即合法域 | `bodice.py` `lin_interpolation` |
| GarmentCodeData 有 120,000 件 | 官方 split 合计 **115,195**；120,000 是 SewingLDM 论文的约数口径 | v2 split JSON |
| "37 panel × 39 edge 是数据集统计" | 是 SewingLDM 预处理 padding 上限，且方向常被记反：**39 panel 槽位 × 37 边/panel** | SewingLDM `configs/ae/ae.py` |
| GarmentCodeData 定义了 benchmark | **没有**——论文无任何基准任务/指标/baseline | arXiv:2405.17609 全文核查 |
| 仿真全部成功 | 成功率 **~72%**，失败样本直接丢弃 | 论文 Sec 4 |

---

## 9. 习题

**题 1**：实验 1 的 Jacobian 里，arm_len 与 back_len 两列全零（本玩具只画前片）。给 GarmentCode 的 bodice 组件补出袖片与后片的公式依赖，使 arm_len/back_len 的列非零，并写出你会预期的非零模式（哪些顶点、什么系数）。
*提示*：袖长 = arm_len × (1 − 袖口上提) + 袖山高；后片袖窿深常用 back_len/5 + 3 之类公式。系数直接从你写的公式读出。

**题 2**：弧长展平的配对段比值精确等于 ruffle（std $\sim10^{-15}$）。证明这一点的充要条件是两边分割点在映射后取并集（而不是只用 A 的分割点或只用 B 的），并构造一个反例：只用 A 的分割点时，某个配对段的比值可以偏离 ruffle 多少？
*提示*：只用 A 的分割点时，B 的段内可能包含 B 自己的分割点——但映射是线性的，段比值仍是 ruffle？想清楚什么时候会破：当 B 的边在 3D 缝合中需要不同处理（如一边是直线一边是圆弧）时，弧长参数化不是等距的。

**题 3**：占据问题的解析解 $\mathbb{E}[\text{distinct}] = C(1-(1-1/C)^N)$。设 GarmentCodeData 的真实离散空间 $|D| = 10^4$、采样 $N = 1.15\times10^5$ 件，求每个设计的期望覆盖数与"某设计完全未被采到"的概率。再问：如果要留 5% 的组合做 unseen 测试集，应该主动排除（先验划分）还是采样后检查（后验划分）？
*提示*：$\Pr(\text{未采到}) = (1-1/C)^N \approx e^{-N/C}$。先验划分对连续参数无效（参数空间不可枚举）——unseen 划分必须定义在离散组合层级。

**题 4**（思考）：GarmentCodeData 的 11.5 万件数据全部来自组件库的采样规则——分布是**设计者规定的**，不是从真实服装统计来的。讨论：用这个数据集训练的生成模型（SewingLDM 等）学到的是"服装的分布"还是"组件库的分布"？给出一个能区分两者的实验设计（提示：真实服装数据集里什么结构在 GarmentCodeData 里频率被系统性高估/低估？）。
*提示*：高估：规则剪裁的对称结构、组件库内的组合；低估：不对称设计、破洞/层叠/非常规缝法。实验：在真实服装照片（如 DeepFashion）上统计袖型/领型频率，对照 GarmentCodeData 的采样分布。

**题 5**（思考）：实验 1 显示 made-to-measure 的"合身"是构造保证的。但同一件 T 恤穿在胖瘦两人身上，舒适度差异很大——宽松度（ease）是 style 参数。讨论：ease 的取值应该跟着身体测量走（公式化）还是保持独立（设计自由）？GarmentCode 的参数重映射（collar_t 归一化后映到 [0, 肩宽−4]，§3.2）给了一个中间答案——找出另外两个你会加约束的 style 参数，并写出你会用的重映射区间（端点应依赖哪些身体测量）。
*提示*：袖窿深不能小于最小臂围通过周长；下摆宽不能小于臀围（否则穿不进）。

---

*下一篇：[GG 06：从观测恢复版型](../GG_06_pattern_recovery/)（NeuralTailor + SewFormer + Garment Recovery：点云/单图/多视角 → 版型的逆问题，病态性定量）*
