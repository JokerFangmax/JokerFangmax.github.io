---
title: "GG 06: 从观测恢复版型——NeuralTailor、SewFormer 与 Garment Recovery"
date: 2026-09-29
math: true
tags:
  - Notes
  - Garment Generation
  - Sewing Pattern
  - Inverse Problem
categories:
  - Garment Generation
description: "版型恢复逆问题的病态性定量：版型 1% 变化在悬垂形状上只留 0.05–0.36mm 位移、边环序列 2n 等价类的排序歧义（类内距离 14–33 > 类间 2–5）、单视角轮廓的信息漏斗（深度 -28% + 内域全盲）——然后看 NeuralTailor 的编码器→面板序列解码、SewFormer 的单图 Transformer、Garment Recovery 的双先验拟合怎么用学习先验补这些洞"
---

# GG 06：从观测恢复版型

> 📖 服装生成系列第六篇（前篇：[GG 05：版型 DSL 与数据基座](../GG_05_garmentcode_data/)）。
> GG 05 讲的是正问题：参数 → 程序 → 版型 → 仿真 → 数据。本篇反过来：**只给观测（点云/单张图像），把版型恢复出来**。论文细节对照原文与官方代码核实（含两个"标题级"勘误：NeuralTailor 与 Garment Recovery 的流行标题都是错的，见 §8.2）。

---

## 1. 这组论文解决什么问题？

正问题的链路是：**版型 → 缝合成 3D 网格 → 悬垂仿真 → 渲染/扫描**。每一步都有物理与几何的确定性规则（GG 01/05）。恢复（recovery）是逆着走：

| 论文 | 观测 | 输出 | 发表 |
|---|---|---|---|
| NeuralTailor | 3D 点云（2000 点，仿真悬垂采样） | 结构化版型（panels + 边环 + 缝合 + 放置） | SIGGRAPH 2022 (TOG 41(4)) |
| SewFormer | **单张** RGB 着装图像（任意姿态） | 同上（端到端回归） | SIGGRAPH Asia 2023 (TOG 42(6)) |
| Garment Recovery | 单张真实图像（ECON 法线 + SMPL 拟合） | 3D 网格 + 版型（双先验 + 可微物理精修） | CVPR 2024 |

三篇论文是同一个逆问题在三种"先验补洞策略"下的解：NeuralTailor 补**序列结构先验**（LSTM/注意力解码 + 类别槽位），SewFormer 补**图像条件先验**（百万级合成渲染对 + Transformer），Garment Recovery 补**形状与变形双先验**（ISP 的 SDF 版型先验 + 变形网络先验，再物理精修）。本篇的核心论点：**这个逆问题的病态性本身可以定量**——我们用三个数值实验（§5）把"观测里到底丢了多少版型信息"量出来，然后回头看每篇论文的架构选择是在补哪个洞：

1. **实验 1（病态敏感度）**：版型 1% 的变化（边长 ±1mm 级）在悬垂 3D 形状上只留下 **0.05–0.36 mm** 位移——逆问题要把信号放大 ~10× 才压得过扫描噪声。恢复版型必须有先验，纯优化不可行；
2. **实验 2（排序歧义）**：同一个多边形有 $2n$ 个等价顶点序列，序列空间里"同一件衣服换起点"比"另一件衣服"更远（类内距离 14–33 vs 类间 2–5）——序列表示必须规范化，而规范化在 unseen 拓扑上可能歧义；
3. **实验 3（信息漏斗）**：3D → 单视角轮廓平均丢 28% 能量（深度方向）+ 内域顶点位移对轮廓**零贡献**（数学上凸包不变）——单图恢复是两层衰减后的极端逆问题。

### 方法流程鸟瞰

同一个逆问题（观测 → 版型）的三种解法，各自用不同的先验补病态性的洞（细节在 §4）：

| 论文 | 流程 | 先验补在哪 |
|---|---|---|
| NeuralTailor | 3D 点云 → 逐点注意力按 31 类面板槽位加权池化 → 每槽位 LSTM 逐边解码边环 → 独立缝合分类器配对 | 序列结构先验：面板角色分类 + 类别槽位对齐 |
| SewFormer | 单张 RGB → CNN+Transformer 编码 → 23 个 panel query 回归放置 → 322 个 edge query 回归 Bézier 边 → 相似度矩阵贪心配缝合 → 仿真器补遮挡出 3D | 图像条件先验：~100 万合成渲染对烧进权重 |
| Garment Recovery | 单张真实图 → ECON 法线 + SMPL 拟合 → ISP 版型先验给初始原型 → 变形网络 deep prior → 测试时两阶段精修（Chamfer + 法向 + 物理） | 双先验 + 显式优化：形状先验 + 变形先验，物理精修兜底 |

一句话分工：前两篇是"学出来的先验"（推理即出结果、快），Garment Recovery 保留测试时优化——换来 simulation-ready 的输出（3D 网格与版型一致、物理上站得住），代价是慢且依赖初始化。

---

## 2. 背景最小集

| 需要 | 出处 |
|---|---|
| 4 维边编码 $(e_x, e_y, c_x, c_y)$、panel 类别槽位、stitch tags | [GG 02](../GG_02_pattern_representations/) §3.1, §5.1, §4.2 |
| 2021 Korosteleva & Lee 数据集（NeuralTailor 的训练数据） | GG 02 §2.2 |
| 悬垂仿真、准静态平衡（实验 1 的物理设置） | [GG 01](../GG_01_smpl_cloth_basics/) §4, [GG 04](../GG_04_self_supervised_draping_isp/) §4 |
| Korosteleva 系论文的演化线（数据集 → NeuralTailor → GarmentCode） | [GG 05](../GG_05_garmentcode_data/) §6 |

---

## 3. 表示法：边环序列 + 规范化 + 类别槽位

NeuralTailor 与 SewFormer 共用一套版型序列表示（边编码同构于 NeuralTailor 的 4 维向量、缝合匹配模块也类似；但训练数据各用各的——SewFormer 自建 SewFactory 数据集。GG 02 §9 勘误一：它不是 Korosteleva 组的论文）。三个组成部分：

### 3.1 边环与规范化

每个 panel 是闭合边环。顶点序列有两个自由度与形状无关：**起点选择**（$n$ 种）与**遍历方向**（2 种）——同一形状有 $2n$ 个等价序列。数据管线的处理：规范化为**最下最左顶点起、逆时针**（GG 02 §2.1 的"NeuralTailor 数据规范"）。数据集"通过设计保证"环路首顶点与方向跨样本一致，模型直接沿用数据顺序、损失无需对齐。

这个规范化的必要性与脆弱性都是本篇实验 2 的内容：它把 $2n$ 个等价序列坍缩为 1 个（必要），但"最下最左"在裙摆打裥等情形下不唯一（脆弱）。

### 3.2 面板级：类别槽位

版型的 panel 数量可变（T 恤 4 片、连衣裙 8+ 片）。NeuralTailor 的方案：把所有面板按角色/身体位置启发式聚为 **31 类**（SewFormer 用 23 类），输出固定长度序列、按类别 ID 排序，**缺失类别填零张量占位**（占位符分散在序列各处）——以此鼓励跨服装类型的同类面板相似。也测过 orderless 变体（预测与 GT 做指派问题匹配）。GG 02 习题 2 对照过它与 SewFormer query 解码、GarmentImage 栅格的优劣（unseen 组合 IoU：槽位 0.31 / query 0.43 / 栅格 0.81）。

### 3.3 放置（placement）

每个 panel 配 7 维放置：4 元数旋转 + 3 维平移。NeuralTailor 的关键细节：平移的**参考点**取"面板 2D 包围盒在 3D 视图中的顶部中点"——对应颈/腰等稳定身体特征，比面板局部原点好学。

---

## 4. 方法核心

### 4.1 NeuralTailor：点云 → 面板序列的集合回归

**Baseline（论文 Sec. 5）**——三级展开：

1. **点云编码器**：2 层 EdgeConv（DGCNN 式动态图，k=5 近邻），逐点特征 153 维（150 维 MLP 输出 + 3 维输入坐标 skip connection），全局平均池化得服装级隐码；
2. **面板级 LSTM**（2 层，hidden 250）：把全局隐码展开为面板编码序列；
3. **面板解码 LSTM**（3 层，hidden 250）：逐边输出 4 维边特征（含 stitch tag 时 8 维），每面板最多 14 条边（零填充）；放置由单独线性层从面板编码映射 7 维。

**最终架构（Sec. 6）**：用**逐点注意力模块**替换面板级 LSTM——注意力 MLP（3 层 ×153 神经元）逐点输出 31 维类别分数，接 SparseMax 促进稀疏（每点只参与一个或少数类别）；每个类别槽位的面板隐码 = 该类注意力分数对逐点特征加权池化。注意力天然对称、可扩散到对称部位——论文发现**显式鼓励分割式注意力反而有害**（分割损失只留 0.05 权重做正则）。

**缝合预测的两个方案**（GG 02 §4.2 的展开，消融数字见 §5.4）：

- **方案 A：stitch tags**（baseline）——每边 1 位 free/stitched 标志 + 3 维可学习 tag，triplet 式损失（同 stitch 的边 tag 靠近、异 stitch 至少差 margin δ；论文正文未给 δ，代码 `stitch_tags_margin: 0.3`），第 40 epoch 才引入；
- **方案 B：独立成对分类器**（最终采用）——MLP（3 层 ×200）输入**边对**：每边 8 维（起终点 3D 坐标 ×2 + 曲率坐标 ×2，坐标相对人体），一对共 16 维，输出"这两条边被缝合"。每件服装每 epoch 采 200 正对 + 200 负对。**关键设计：分类器在形状模型的预测输出（非 GT）上训练**——理由是形状与缝合纠缠在同一隐码会引入数据偏置，且实验证明仅凭版型几何与放置（不回看点云）就足以预测缝合。

### 4.2 SewFormer：单图 → 版型的两级 Transformer

正式标题 **"Towards Garment Sewing Pattern Reconstruction from a Single Image"**（Sewformer 是网络名，不是论文标题）。任务设定：输入单张 RGB 着装人像（**任意姿态，非 T-pose**；训练时随机旋转/仿射/透视增广后 resize 384×384）。

**架构**（论文 + 官方代码双重核实）——DETR 式两级 coarse-to-fine：

1. **编码器**：ResNet-50（dilated）提特征图 → 展平为 1D 序列 → 标准 Transformer encoder（6+6 层、hidden 256、8 heads、sine 位置编码）。**不是 ViT patch，是 CNN 特征 + Transformer**；
2. **Panel decoder**：23 个 panel query（高斯随机初始化、端到端学习——论文正文写 24 类，开源代码 `panel_classes_condenced.json` 实际 23 类，见 §8.2）对视觉 token cross-attention 得 panel token，MLP 直接回归每 panel 的 $R_i, T_i$；
3. **Edge decoder**：每 panel 的 edge query（23×14 = 322 个）与对应 panel 特征**逐元素相加**（同 panel 的边共享 panel 特征）再 cross-attention，MLP 回归 4 维 Bezier 边参数。另有 22 个 SMPL 关节 query 接在 panel query 后做人体姿态正则；
4. **缝合预测**：对所有 edge-token 对做对称相似度矩阵，去下三角后迭代取最大值并删除对应行列（与 NeuralTailor 的模块类似）。

**损失**：$L_{\text{total}} = 10 L_{\text{panel}} + 0.5 L_{\text{stitch}} + 1 L_{\text{SMPL}}$。两个新损失值得注意：$L_{\text{shape}}$ 在 panel 上采样 support vectors（连边顶点/中点）惩罚与 GT 的 L2——**2D mask loss 的可微近似**（直接算预测/GT 多边形的 IoU 不可微）；$L_{\text{SMPL}}$ 让 pose query 回归 3D 人体姿态做正则（β 不监督）。

**3D 从哪来**：不直接回归 3D mesh——panel decoder 预测各 panel 相对人体的 3D 放置，RSC-Net 从输入图估人体 shape/pose，再把版型交给布料模拟器（Qualoth）缝合悬垂得 3D 服装。**遮挡区域的完整性由仿真补**，这与 Garment Recovery 的物理精修是同一思路的轻量版。

**训练数据 SewFactory**（自建，非 2021 版型数据集）：**约 100 万张** RGB 图 + 版型 GT——19.1k 件服装 × 24 视角，Arnold 渲染、Qualoth 仿真；13.7k 个 AMASS 姿态经 Maya 插值扩充；SMPL β~U[−1.5, 1.5]；棉/丝/天鹅绒三种布料 + 属性随机扰动；85k 张测试图，train/test 无服装与姿态重叠。另配一个人体纹理合成网络（DensePose 引导 warping + StyleGAN adaptive-norm U-Net + SDEdit 精修）弥合 sim-to-real。训练 40 epochs、8×A100、batch 512。

### 4.3 Garment Recovery：单图 + 双先验 + 可微物理

正式标题 **"Garment Recovery with Shape and Deformation Priors"**（Ren Li, Dumery, Guillard, Fua，EPFL——ISP 的直接后续；GG 04 §4 已见过前三个名字）。标题里的"双先验"就是它的架构答案：**版型形状先验 + 变形网络先验**，观测本身比 SewFormer 还弱（单张真实图），但先验补得最重：

1. **观测提取**：单张真实图像 → ECON 估计服装**法线图**（稠密深度信息的替代源——绕开实验 3 的轮廓漏斗）+ SMPL(β,θ) 拟合 + SAM/human parsing 分割；
2. **形状先验 = ISP**：版型表示完全沿用 GG 04 的 ISP——2D SDF panel + UV 参数化网络；把最大覆盖的原型服装 skin 到人体上做初始化；
3. **变形先验 = 可微变形网络**：MLP + 2×ConvNeXt（pixel-aligned 特征）预测 occupancy 与位移——deep-prior 式的正则，比直接优化 mesh 顶点平滑；
4. **两阶段精修**：先精调变形网络参数（deep prior 约束下），再优化 mesh 顶点。损失 $L = \lambda_C L_{\text{CD}} + \lambda_n L_{\text{normal}} + \lambda_p L_{\text{physics}}$——2D Chamfer（投影可见面心 vs 前景分割像素）、法线余弦距离、**物理项 = 应变能 + 弯曲能 + 重力势能 + 人-衣碰撞**（Narain 2012 / Santesteban 2022 同款，GG 01/04 的物理损失家族）。

与 SewFormer 的本质区别：SewFormer 把先验全部**烧进网络权重**（百万合成对 + 端到端回归，推理即出结果）；Garment Recovery 保留**显式优化**（测试时仍在解一个带物理项的目标函数，优化变量是 ISP 潜码 z、变形网络参数、最后是 mesh 顶点）——换来的是输出 simulation-ready（3D 网格 + 版型一致、物理上站得住），代价是慢且依赖初始化原型覆盖。这是"学出来的先验"与"写下来的先验"两条路线在版型恢复上的第一次正面分工。

---

## 5. 数值实验：逆问题的病态性定量

三个实验共用 GG 01/04 的 12×12 布料 + 球体物理设置（结构边 $k=10^4$、弯曲边 $k=10$、碰撞罚函数、XPBD 动态 + 多重启 L-BFGS 准静态平衡）。实验脚本：`gg06_experiments.py`（numpy/scipy，无学习成分——病态性是问题的性质，与解法无关）。

### 5.1 实验 1：∂悬垂形状/∂版型 的敏感度谱

版型的几何自由度取各约束的 rest length（缝合前的 2D 边长）。基准：XPBD 悬垂 2 秒 + 准静态精化。对 rest lengths 施加 $\pm1\%$ 扰动（$\|u\|_\infty = 1$），重新平衡，测 3D 位移 RMS：

| 扰动方向 | 3D 位移 RMS |
|---|---|
| 全部结构边 +1%（整件大一号） | **1.55 mm** |
| 弯曲边 ±1%（高频自由度） | **0.00 mm**（低于求解器噪声地板） |
| 20 个随机方向 | min **0.05** / median **0.13** / max **0.36** mm（动态范围 7.7×） |

三个读数：

1. **逆问题要放大 ~10× 信号**：版型 1% 变化（边长 ±1mm 级）只留 0.05–0.36mm 悬垂位移，而扫描噪声就有 ~1mm（NeuralTailor 自报的噪声承受上限）——直接反演 rest lengths 是在噪声里捞信号，必须有先验（正则 / 学习先验）；
2. **敏感度谱有 7.7× 动态范围**：整体尺寸类方向可观测，弯曲/省道类方向不可观测——恢复出的版型在"整体尺寸"上可信、在"精细调型"上本质瞎猜。这与 NeuralTailor 的误差结构一致：unseen 类型上平移 L2 (3.22) 比旋转 L2 (0.07) 差一个数量级——平移是身体特征锚定的低频量，旋转同理；
3. **弯曲边方向低于噪声地板**意味着：**悬垂几何里根本没有省道信息**。恢复出的版型"能穿"（结构尺寸对）但"不合身"（局部调型丢）——版型恢复的上限不是模型容量，是观测的信息量。

### 5.2 实验 2：边环序列的排序歧义

T 恤前片（49 边，NeuralTailor 式 4 维边向量 $(v_x, v_y, c_x, c_y)$，直边 $c=(0.5,0)$），用**连续编辑距离**（替换代价 = $\min(1, \|e_1 - e_2\|/0.05)$）度量：

- **类内**（同一件 T 恤的 98 个等价序列两两距离，91 对采样）：min **14.0** / median **26.7** / max **33.2**；
- **类间**（三种版型——T 恤/挖深袖窿的背心/加长的长款——规范化后互距）：T恤↔背心 **4.0**，T恤↔长款 **2.0**，背心↔长款 **5.0**。

**类内距离（14–33）全部大于类间距离（2–5）**：未规范化的序列距离不是形状度量——"同一件衣服换起点/方向"在序列空间里比"另一件衣服"更远。规范排序（最下最左 + 逆时针）把 98 个等价序列坍缩为 1 个，这是 NeuralTailor/SewFormer 数据管线的第一步。脆弱性也在数字里：**起点选择本身是形状的函数**——最下最左点在裙摆打裥时不唯一，unseen 拓扑上的规范化歧义是序列表示的隐藏失败模式（数据集"通过设计保证"一致性，真实数据没有这个保证）。

### 5.3 实验 3：悬垂形状 → 单视角轮廓的信息漏斗

**(a) 深度盲区**：实验 1 的 20 个随机扰动方向，悬垂位移能量在深度（y）方向的占比 mean **0.28**——单视角投影平均丢 28% 的版型信号（前视角相机在 +y 方向）。

**(b) 内域盲区**：按**投影空间**（不是网格拓扑！）选顶点——距凸包轮廓边界 >1.5cm 的"轮廓内域"顶点 110 个，各施加 ≤1cm 随机位移（限幅保证投影后仍在凸包内）；对照组为距轮廓 <1cm 的 9 个近轮廓顶点：

| 扰动组 | 顶点数 | 3D 位移 RMS | 轮廓 Chamfer 变化 |
|---|---|---|---|
| 轮廓内域（距边 >1.5cm） | 110 | 4.97 mm | **0.0000 mm** |
| 近轮廓（距边 <1cm） | 9 | 4.27 mm | **0.9464 mm** |

内域顶点的位移对轮廓**精确零贡献**（凸包数学上不变）——1cm 的内部褶皱/省道变化在轮廓上完全不可见。且真实成衣轮廓不是凸包（袖窿内侧等凹陷处也看不见），实际盲区比这个演示更大。

**信息漏斗汇总**：

```text
版型自由度 ──(实验1: 病态, 弯曲类自由度丢失)──► 3D 悬垂
                                              ├──(实验3a: 深度 -28%)──► 双视角
                                              └──(实验3a+b: 深度 -28% + 内域全盲)──► 单视角轮廓
```

单图轮廓版型恢复（SewFormer）是两层衰减后的极端逆问题——它必须几乎完全依赖合成数据先验（百万渲染对烧进权重）；Garment Recovery 用**稠密法线（ECON）替代轮廓**作观测，逐像素深度信息绕开了实验 3(b) 的内域盲区——但实验 1 的第一层病态性（版型 → 悬垂丢高频信息）对任何观测强度都存在，所以它同样必须靠版型先验（ISP 形状先验）补洞。

### 5.4 NeuralTailor 的关键数字

训练数据：作者自己的 2021 数据集（22,000 件、19 个基础类型）；7 个类型整体留作 unseen 测试组；过滤"重叠设计"（不同版型产生相似 3D 形状）后训练集 9,678 件。精度（注意力模型）：

| 指标 | seen | unseen |
|---|---|---|
| Panel L2 ($\times 10^{-2}$) | 1.5 | 5.2 |
| 面板数正确率 | 99.7% | 83.6% |
| 边数正确率 | 99.7% | 87.3% |
| 旋转 L2 / 平移 L2 | 0.04 / 1.46 | 0.07 / 3.22 |

baseline LSTM 在 unseen 类型上面板数正确率仅 **1.6–6.5%**（基本失败）——注意力机制是跨类型泛化的关键。缝合消融（P/R）：

| 方案 | seen | unseen |
|---|---|---|
| stitch tags | 99.9% / 99.9% | 70.1% / 72.9% |
| 成对分类器（预测边上训练） | 96.3% / 99.4% | **74.7% / 83.9%** |
| 成对分类器（GT 边上训练） | 96.6% / 88.6% | 75.3% / 60.6% |

三个读数：GT 训练的分类器整线接入后**更弱**（对噪声输入不鲁棒——训练/推理分布一致性的教科书案例）；tag 方案代价大（训练慢 15 小时、面板形状明显变差：unseen Panel L2 8.31 vs 5.2）；unseen 分数只在面板数预测正确的样例上统计——缝合指标的水分要看清分母。

**SewFormer 的关键数字**（SewFactory 测试集，指标族与 NeuralTailor 相同——Panel L2 / Rot L2 / Trans L2 / #Panel / #Edges / stitch F1；**没有 IoU**）：

| 方法 | Panel L2 | Rot L2 | Trans L2 | #Panel | #Edges | stitch F1 |
|---|---|---|---|---|---|---|
| NeuralTailor（原版） | 4.41 | 0.0300 | 1.050 | 83.6% | 97.8% | 84.5% |
| NeuralTailor*（用本文损失重训） | 4.15 | 0.0347 | 0.995 | 83.8% | 97.5% | 86.7% |
| Sewformer†（单级 decoder） | 3.91 | 0.0322 | 0.979 | 87.5% | 95.6% | 90.1% |
| **Sewformer（两级）** | **3.57** | **0.0205** | **0.693** | **88.7%** | 97.5% | **95.7%** |

相对 NeuralTailor：Panel L2 −19%、Rot L2 −32%、Trans L2 −34%、stitch F1 +11.2%。注意换数据集的意义：NeuralTailor 在这里是被拿到 SewFactory（24 视角渲染、任意姿态）上重训/评测的——**观测从点云换成图像、姿态从 T-pose 换成任意，序列解码的 NeuralTailor 架构还能打**，两级 query + 图像注意力再拿走 19%。消融：去 $L_{\text{shape}}$ → Panel L2 3.71；去 $L_{\text{SMPL}}$ → #Panel 86.1%。真实图用户研究（51 张 DeepFashion、45 人、0–3 分）：无纹理 0.822 / SURREAL 纹理 1.12 / 完整纹理合成 2.35——**sim-to-real 的纹理差距比几何差距更致命**。

**Garment Recovery 的关键数字**（CLOTH3D，30 件 unseen/类；CD×10³↓ / IoU↑——IoU 是这篇的指标，不是 SewFormer 的）：

| 类别 | 最强基线 | ISP | Ours |
|---|---|---|---|
| Skirt | SMPLicit 8.05 / 0.657 | 7.26 / 0.709 | **2.87 / 0.940** |
| Shirt | DrapeNet 1.54 / 0.715 | 2.02 / 0.665 | **0.44 / 0.939** |
| Trousers | — | — | **0.28 / 0.946** |

对比里**完全没有 SewFormer/NeuralTailor**——基线是 BCNet、SMPLicit、ClothWild、DrapeNet、ISP：真实图像上的恢复与合成图像上的恢复在文献里还是两个几乎不对话的子领域（评测集、指标、基线都不同），这个割裂本身是领域现状的一部分。

---

## 6. 论文间关系

```
2021 Korosteleva & Lee 数据集（22k 件, JSON 格式）
        ↓ 训练数据
NeuralTailor (SIG'22)：点云 → 版型（序列解码 + 注意力 + 成对缝合分类器）
        ↓ 边表示/缝合模块同构, 训练数据自建（SewFactory 100 万渲染对）
SewFormer (SIGA'23, Sea AI Lab)：单图 → 版型（两级 Transformer, 先验烧进权重）
        ↓ 同为单图, 换先验策略：显式优化 + 可微物理（ISP 后续）
Garment Recovery (CVPR'24, EPFL)：单图 → 版型 + 3D 网格（形状/变形双先验 + 物理精修）
        ↓ 逆问题能力反过来喂生成（"GT 版型哪来"）
2021 数据集 → GarmentCodeData（GG 05）：程序化正问题全面接管数据生产
```

**谱系定位**：恢复三连是**逆问题方法**，但它们的训练数据全部来自**正问题程序**（2021 数据集的 PBS 仿真、SewFactory 的 Qualoth 仿真、GarmentCodeData 的 XPBD）——逆问题的上限被正问题的数据分布锁死（实验 1 的病态性 + 训练分布的双重约束）。NeuralTailor 发现的"重叠设计歧义"（连衣裙被误读成长 T 恤）本质是**正问题非单射**：两个版型仿真出相似的 3D 形状——逆问题在这对版型上没有唯一解，任何模型都只能猜训练分布里更常见的那个。

**三篇的先验策略光谱**：NeuralTailor（结构先验：类别槽位 + 序列解码）→ SewFormer（数据先验：百万渲染对端到端）→ Garment Recovery（模型先验：预训练 ISP + 变形网络 + 测试时物理优化）。先验越靠后越"显式"，可解释性与 simulation-ready 性越强，推理成本与对初始化的敏感性也越高——这个光谱在 GG 10 的 Dress-1-to-3（版型初估 + 扩散伪 GT + 可微物理联合优化）里被走完一个来回：初估用 SewFormer 式学出来的先验，精修用 Garment Recovery 式写下来的先验。

---

## 7. 与 Work 3 的连接

Work 3（视频先验蒸馏 → 缝纫版型服装生成）与恢复三连有三个交点：

1. **评测对照**：SewFormer 是 SewingLDM（我们的 P0 baseline）在 GarmentCodeData 上的对照方法之一——clothes-to-body distance 5.45cm vs SewingLDM 2.20cm（GG 05 §5.4）。P0 复现跑分时这行数字是 sanity check：如果我们的复现里 SewFormer 的相对位置不对，说明评测管线有 bug；
2. **病态性谱 → 蒸馏损失的加权方向**：实验 1 的谱结构（整体尺寸方向敏感 1.55mm、弯曲方向不可观测）直接告诉我们视频观测能约束版型的哪些自由度——**视频蒸馏的几何监督应该加权低频自由度（整体尺寸/衣长/袖长），高频自由度（省道/弯曲调型）交给 SewingLDM 的训练先验**。这是"视频先验蒸馏"与"纯重建"的本质区别：蒸馏是把分布先验注入，不是从观测反演；
3. **恢复与生成的分工**：恢复方法（本篇）受双重新增约束——观测病态性（实验 1/3）+ 训练分布（重叠设计歧义）。Work 3 选生成式路线（SewingLDM 扩散底座）的原因正在于此：条件生成不要求"从观测唯一反演版型"，允许输出训练分布内多个合理解释。GG 10 的 Dress-1-to-3 会给出混合方案（版型初估 + 扩散伪 GT + 可微物理联合优化）——那个"版型初估"正是本篇 SewFormer 式方法的角色。

---

## 8. 局限与勘误

### 8.1 局限

**NeuralTailor 自列**（Secs. 7.5–7.8）：重叠设计歧义（无法对同一输入给出多种合理解释）；噪声承受力有限（高斯 σ 最高约 1cm，Kinect/单目 RGB 级数据超出范围）；sim-to-real 差距（Deep Fashion3D 上结构猜测合理但形状/放置明显下降）；面板类别划分是启发式的且很关键（"分类问题可能比我们最初想的更复杂"）；未来工作：省道/褶/复杂曲线、多层穿着与遮挡、姿态体型变化、面料属性。

**方法层面**（本篇实验的推论）：恢复版型的"整体尺寸可信、精细调型瞎猜"（实验 1 谱结构）意味着下游用恢复版型做制造时会遇到"能穿但不合身"的系统性问题——评测里的 Panel L2 均值掩盖了这个谱结构。

**SewFormer 自列**：训练数据外的缝纫拓扑无法处理——经典失败案例是**旗袍**（"stitching relations are beyond the settings of our training data"，拼接关系预测显著出错）；无法预测训练中未见的配件（帽子等）；未来工作：更强的 attention、视频时序线索、更大更多样的数据与渲染变化、显式的 body-part-to-panel 建模。

**Garment Recovery 自列**：训练数据的材料/外力/动作范围有限，wild 图像直接推理不准；变形网络倾向学低频函数导致结果过滑，需要顶点级精修补；**因缺公开真实评测集，定量评估只能在合成数据上做**；DrapeNet 无裙装（用裤装替代）。未来工作：视频时序一致性。

### 8.2 勘误表

**Garment Recovery 的标题勘误（本篇写作时最大的坑）**：调研笔记里记的是 "Garment Recovery with NeRF-guided Sewing Pattern Estimation"——**这篇论文不存在**（arXiv 全字段、CVPR 2024/2025 全部标题清单、OpenReview、GitHub 穷尽检索均无）。真实论文是 **"Garment Recovery with Shape and Deformation Priors"**（arXiv:2311.10356），**单图方法，全文不涉及 NeRF/多视角**。曾记的 arXiv:2312.03390 是一篇二偶极超固体物理论文。"NeRF-guided" 的说法是记忆拼接产物——若在任何转述里看到，回到 Ren Li 的 publication list 核对。

NeuralTailor 一节的流行误传（写作时全部踩过，逐条核对原文与代码后修正）：

| 误传 | 事实 | 依据 |
|---|---|---|
| 标题 "Editing Sewing Patterns through Fluid Interface" | **"NeuralTailor: Reconstructing Sewing Pattern Structures from 3D Point Clouds of Garments"**——不存在"流体界面编辑"这篇论文 | arXiv:2201.13063 |
| 作者是 Michael Korosteleva（MPI） | **Maria Korosteleva & Sung-Hee Lee（KAIST）**；Kesdogan 是她后续 GarmentCodeData 的合作者，与本论文无关 | arXiv 作者页、GitHub |
| 代码在 `michael-korosteleva/NeuralTailor` | 该仓库 **404**；官方实现是 **`maria-korosteleva/Garment-Pattern-Estimation`** | GitHub 全站搜索 |
| 有交互式编辑 / seamguidance 机制 | **没有任何编辑功能**——单向前向推理的全自动重建 | ar5iv 全文 |
| 输入是扰动边序列 | 输入是**点云**：蒙在 SMPL 平均女性体型上的悬垂网格表面随机采样 **2000 点** | 论文 Sec. 3 |
| 成对分类器输入 8 维 | 单边 8 维（起终点 3D 坐标 ×2 + 曲率 ×2），**一对 16 维** | 代码 `element_size: 16` |
| triplet margin 论文给出 | 论文未给数值，代码为 **0.3** | `stitch_tags_margin` |

SewFormer 的对应勘误：

| 误传 | 事实 | 依据 |
|---|---|---|
| 论文标题是 "SewFormer" | 正式标题 **"Towards Garment Sewing Pattern Reconstruction from a Single Image"**（Sewformer 是网络名） | arXiv:2311.04218, TOG 42(6) Art.198 |
| 代码在 `microsoft/...` | 官方仓库 **`sail-sg/sewformer`** | GitHub |
| 有旋转不变性 / alignment 归一化贡献 | **没有**——全文无 rotation invariance/canonical 机制；只有边参数局部坐标系、训练时旋转增广、$L_{RT}$/$L_{\text{SMPL}}$ 损失。代码里两个 origin/order invariant loss 开关**默认关闭**且非论文贡献 | ar5iv 全文 + train.yaml |
| 指标是 IoU | **没有 IoU**——Panel L2 / Rot L2 / Trans L2 / #Panel / #Edges / stitch P-R-F1 | 论文 Table 2 |
| panel 类数 24（论文正文） | 开源代码 `panel_classes_condenced.json` 实际 **23 类**，`max_pattern_len` 覆盖为 23 | 官方代码 |
| 训练数据是 Korosteleva 2021 数据集 | 自建 **SewFactory**（19.1k 服装 × 24 视角 ≈ 100 万渲染对） | 论文 Sec. 5.1 |

---

## 9. 习题

**题 1**：实验 1 里 20 个随机方向的敏感度动态范围 7.7×（0.05–0.36mm/1% 版型变化）。设观测噪声 σ = 1mm（扫描级），对最敏感方向与最不敏感方向分别计算：要把版型恢复到 0.5mm 边长精度，逆问题需要多大的正则/先验强度（用 Tikhonov 的语言：观测项与先验项的权重比）？
*提示*：病态线性系统 $(J^TJ + \lambda I)\delta p = J^T\delta x$，$\lambda$ 与噪声/信号比的关系。弯曲边方向的 $J$ 列接近零——任何有限 $\lambda$ 下解都被先验主导。

**题 2**：证明规范化（最下最左顶点起 + 逆时针）在**一般位置**多边形上唯一确定顶点序列，然后构造反例：(a) 水平底边（最下不唯一）；(b) 对称裙摆打裥（最左不唯一）。讨论 NeuralTailor 数据集"通过设计保证一致性"在真实扫描数据上为什么不成立。
*提示*：一般位置 = 无两点共享最小 y（再无两点共享 x）。真实数据的退化是测度非零的事件：打裥制造的重复结构让"最下最左"系统性并列。

**题 3**：NeuralTailor 的 31 类槽位制在 unseen 类型上面板数正确率 83.6%（注意力版），baseline LSTM 只有 1.6–6.5%。结合实验 2（类内距离 14–33 > 类间 2–5）：为什么"按类别 ID 固定排序 + 零张量占位"对 unseen 拓扑如此脆弱？注意力机制缓解的是哪一环？
*提示*：unseen 类型 = 训练时没见过的类别组合出现模式。槽位制把"哪些类别出现"编在序列位置里；注意力按内容（点云局部特征）分配面板，绕过了位置编码。

**题 4**：缝合消融里，成对分类器在 GT 边上训练反而比在预测边上训练差（unseen recall 60.6% vs 83.9%）。解释这个"用更干净的数据训练出更差的模型"现象，并给出一个类比的其他领域案例。
*提示*：训练/推理分布偏移——推理时分类器吃的是形状网络的**预测边**（带噪声、带系统性误差），GT 训练的决策边界落在干净数据上，遇到预测边的分布外特征就翻车。类比：目标检测在合成干净数据上过拟合。

**题 5**（思考）：实验 3 说明单视角轮廓对内域变化零敏感。SewFormer 却能用单图恢复版型——它恢复的信息实际来自哪里？给出一个具体的失败预测：在什么输入上，SewFormer 会"自信地输出训练分布里的常见版型"而与真实观测矛盾？Garment Recovery 的稠密法线观测 + 显式物理精修把这个失败模式减轻了多少？
*提示*：图像先验 = 从百万合成渲染对里学到的"图像 → 版型"条件分布；轮廓外的信息（纹理、缝线、褶皱位置）都进了先验。失败预测：真实照片上的非常规设计——旗袍是论文自认的失败案例（缝合拓扑出训练集）。Garment Recovery 的法线图绕开轮廓漏斗（实验 3b）、物理项惩罚不合理形变，但实验 1 的病态性（版型→悬垂丢高频信息）对任何观测强度都存在——高频自由度永远来自先验而非观测。

---

*下一篇：[GG 07：文本到版型 I——自回归与并行扩散](../GG_07_text_to_pattern_ar_diffusion/)（DressCode + GarmentDiffusion：版型怎么 token 化，自回归为何在 18k/30k token 上撞墙，DiT 并行去噪怎么快 100×——三个数值实验拆开串行墙/量化结构损失/50 步步数消融）*
