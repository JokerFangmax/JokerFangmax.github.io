---
title: "GG 02: 缝纫版型表示法专题——全系列的地基"
date: 2026-09-28
math: true
tags:
  - Notes
  - Garment Generation
  - Sewing Pattern
  - Representation Learning
categories:
  - Garment Generation
description: "panel/edge/stitch 解剖、Korosteleva JSON 数据格式、边的 Bezier 编码（4 维向量到 29 维向量）、缝合匹配的对称歧义、四种版型编码（向量序列/程序 DSL/栅格/隐式场）对照——配曲线表达力/缝合匹配/量化误差三个数值实验"
---

# GG 02：缝纫版型表示法专题

> 📖 服装生成系列背景篇之二（前篇：[GG 01：SMPL 与布料仿真基础](../GG_01_smpl_cloth_basics/)）。本篇是**全系列的地基**：后面 10 篇里出现的每一种方法，首先都是一个"版型表示法 + 学习/生成算法"的组合——表示法决定了能表达什么、怎么学习、怎么生成。
> 本文所有论文细节均已对照原文与官方代码核实（含三处与坊间流传不符的勘误，见 §9）。

---

## 1. 这一篇解决什么问题？

服装是**柔性物体**：它的 3D 形状 = 布料物理（GG 01 §3）在某个 2D 裁剪方案下的平衡态。这个 2D 裁剪方案就是**缝纫版型**（sewing pattern）——裁缝手里的纸样：若干 2D 裁片（panel），边缘画着直线与曲线，标着"这条边与那条边缝合"。

为什么生成版型，而不是直接生成 3D 服装网格？三个不可替代的性质：

1. **可制造**（manufacturable）：版型可以直接打印、裁布、缝制——生成的衣服是**真实可穿的**。3D 网格不行：没有裁剪线的 3D 网格无法变成布料衣服；
2. **可仿真**（simulation-ready）：版型 + GG 01 的布料物理 = 任意身体姿态下的 3D 形状。**"rest shape = 版型"**（GG 01 §3.1）——版型是物理引擎的输入，不是输出；
3. **解耦了设计与时序**：同一件版型在不同身体、不同动作下仿真，得到的是**同一个设计**。直接学 3D 网格的话，设计自由度与物理形变纠缠在一起。

本篇回答四个问题：

1. **版型的数据结构是什么**：panel/edge/stitch 的解剖，领域的 JSON 格式；
2. **一条边怎么编码**：从 NeuralTailor 的 4 维到 SewingLDM 的 29 维，每多出来的维度在买什么能力？（§3 数值实验量化"每参数表达力"）
3. **缝合怎么表示**：为什么纯几何匹配存在原理性歧义（§4 实验），领域里三种缝合表示怎么解决；
4. **四种版型编码**（向量序列 / 程序 DSL / 栅格 / 隐式场）各怎么工作、各适合什么任务？

---

## 2. 版型解剖：panel、edge、stitch

### 2.1 三个原语

```
版型 (sewing pattern)
 ├── panels: 有序裁片列表
 │    ├── 边环 (edge loop): 有向闭合边序列，定义 panel 轮廓
 │    ├── placement: 该 panel 在 3D 中相对人体的位姿（旋转 R + 平移 t）
 │    └── 内部细节（省道 darts 等）
 └── stitches: 边对列表 [(panel_i, edge_k) ↔ (panel_j, edge_l), ...]
```

**panel**：一块 2D 面料裁片。轮廓是一个**有向闭合边环**（逆时针，从左下角顶点起——NeuralTailor 数据规范），panel 在 3D 里有一个相对人体的位姿（6 维：旋转 + 平移），用于把它"包裹"到身体周围做初始摆放。

**edge**：panel 轮廓的一条边——直线、圆弧、或 Bezier 样条。端点由环的连接关系隐式给出（闭合多边形存起点即可），曲线形状由**相对控制点**编码（§3）。所有控制量定义在边的**局部坐标系**里（起终点连线为横轴）——平移/旋转不变、均匀缩放保曲率（GarmentCode 论文的设计原则）。

**stitch**：把两条边（可在不同 panel，也可同 panel——省道）缝到一起的指令。缝合后两条边在 3D 中必须重合（长度近似相等是版型**有效性**的必要条件）。

### 2.2 领域的 JSON 格式（事实标准）

Korosteleva & Lee 的 NeurIPS 2021 数据集论文《Generating Datasets of 3D Garments with Sewing Patterns》定义的 JSON 版型格式，是整个领域的公共语言——NeuralTailor、SewingLDM、GarmentCodeData 全部直接沿用（GarmentCodeData 论文 Sec 3.4 明确写引用该格式）。以 GarmentCode 仓库的 `shirt_mean_specification.json` 为例，实际结构（已对照真实文件核实）：

```json
{
  "pattern": {
    "panel_order": ["front", "back", ...],
    "panels": {
      "front": {
        "translation": [x, y, z], "rotation": [x, y, z],
        "vertices": [[x, y], ...],
        "edges": [
          {"endpoints": [i, j], "curvature": {"type": "quadratic",
            "params": [[cx, cy]]}},
          {"endpoints": [i, j]}
        ]
      }, ...
    },
    "stitches": [
      {"panel": "front", "edge": 3}, {"panel": "back", "edge": 2}
    ]
  },
  "properties": {"curvature_coords": "relative", "units_in_meter": 100, ...}
}
```

要点：`endpoints` 是**顶点索引**（坐标存在 `vertices` 里，边只引用）；`curvature.type` ∈ {quadratic, cubic}，`params` 存相对控制点（1 个 = 二次、2 个 = 三次）；无 `curvature` 字段即直线。**每条 stitch 是 1 对 1 的边对**——这个"边对边"的粒度是后续所有缝合表示的起点。

这个"表示成为基础设施"的剧本与 SMPL（GG 01 §2.4）如出一辙：一个数据格式被数据集（2021 版型数据集 → GarmentCodeData 11.5 万件）反复固化后，即使方法演进（序列 → 栅格 → 隐式 → 潜空间扩散），底层语言不变——**学新方法前先读懂这个 JSON，能省一半的读论文时间**。

---

## 3. 边的几何编码：从 4 维到 29 维

### 3.1 谱系总览（每维都已核对原文/代码）

| 方法（年份） | 每边几何编码 | 维度 | 曲线类型 |
|---|---|---|---|
| NeuralTailor ('22) | $(v_x, v_y, c_x, c_y)$：位移向量 + 二次 Bezier 控制点 | 4 | 直线/二次 |
| DressCode ('24) | 同 NeuralTailor（量化成离散 token） | 4 | 直线/二次 |
| SewFormer ('23) | 同 NeuralTailor（边编码同构，缝合匹配模块类似） | 4 | 直线/二次 |
| GarmentDiffusion ('25) | $e_3 \oplus c_3 \oplus a_3$：3D 起点 + Bezier 控制 + 圆弧参数 | 9（几何折算；完整边 token 13 维含 stitch） | 直线/二次/三次/圆弧 |
| SewingLDM ('25) | $V_2 \oplus C_2 \oplus C^b_4 \oplus C^r_3$ + 位姿与 flag（§3.3） | 29（19 连续 + 10 离散） | 直线/二次/三次/圆弧 |

三个事实值得强调（都是核对原文后才确认的）：

1. **4 维向量 $(v_x, v_y, c_x, c_y)$ 是整个领域的公共底座**。NeuralTailor 原文：$(e_x, e_y)$ 是起点到终点的位移，$(c_x, c_y)$ 是二次 Bezier 控制点在边局部坐标系下的相对坐标——"$c_x$ 大致对应曲率峰值的位置，$c_y$ 控制曲率深度"。**没有"旋转角+长度"的分解，曲率是 2 个参数而非 1 个**（支持非对称曲线——单参数只能表达对称鼓包）。SewFormer 代码直接复用了 NeuralTailor 的 `_edge_as_vector`；DressCode 的 SewingGPT 每边 token 也是这 4 个几何参数；
2. **曲线类型的丰富是逐步的**：二次 Bezier（NeuralTailor 一代）→ 加三次 + 圆弧（GarmentDiffusion、SewingLDM 一代）。GarmentCode DSL 早就支持全部四类（直线/圆弧/二次/三次），生成模型花了三年才追平表示能力；
3. **维度膨胀的驱动是"完整性"**：DressCode/SewingLDM 一代把 stitch、panel 位姿、边存在性、附着约束全部拼进边向量——见 §3.3。

### 3.2 数值实验 1：每参数表达力——Bezier vs 折线

设置：目标是一条**不对称领口曲线**（前中下垂 + 轻微偏移，弧长 0.478 m，采样 200 点）。用不同参数化拟合（端点由邻边给定，只数**自由参数**），误差用双向 Chamfer 距离：

| 参数化 | 自由参数 | mean 误差 | max 误差 |
|---|---|---|---|
| (a) 直线段 | 0 | 73.69 mm | 120.28 mm |
| (b) 对称二次 Bezier（单鼓包量 $h$） | 1 | 3.33 mm | 7.60 mm |
| (c) 一般二次 Bezier（控制点 2 坐标，即 $c_x, c_y$） | 2 | 2.78 mm | 5.27 mm |
| (d) 三次 Bezier（两控制点） | 4 | **1.21 mm** | 3.56 mm |
| (e) 折线，2 段 | 2 | 28.93 mm | 66.24 mm |
| (e) 折线，4 段 | 4 | 14.24 mm | 34.44 mm |
| (e) 折线，8 段 | 8 | 6.89 mm | 17.31 mm |

三个读数：

1. **Bezier 的每参数表达力约为折线中点偏移的 4 倍**：(d) 用 4 参数做到 1.21 mm，折线要达到这个精度需要 ~30 段（30 参数）。这就是为什么全领域不约而同选 Bezier 而不是密化折线——**版型坐标的参数效率是学习难度的第一决定因素**；
2. **对称假设（b）的真实代价**：max 误差 7.60 mm vs (c) 的 5.27 mm——单参数鼓包拟合不对称曲线时，误差集中在"曲率峰被强行居中"的区域。 mm 级对服装是可感知的（缝纫精度约 5 mm），但不是灾难——这解释了为什么 NeuralTailor 的 4 维编码"够用但刚好够用"；
3. **三次 Bezier 的收益是 2.3 倍**（2.78→1.21 mm）而非数量级——真实版型的边大多是"单弯"形状，三次的两个控制点主要用于不对称单弯，不是 S 形多弯。**GarmentDiffusion/SewingLDM 加三次与圆弧，买的是"圆弧精确"（曲线缝合 CAD 生态）与长尾形状，不是平均精度**。

### 3.3 SewingLDM 的 29 维边向量（逐维分解）

SewingLDM（ICCV 2025，上海交大 MoE AI Lab；我们的 Work 3 baseline）把边向量推到 29 维。按代码 config（`continuous = (element_size 9 + stitch_tags 3) + pose 7`，`discrete = edge_type_size 7 + stitch mask/reverse 2 + existence 1`，即 $(9+3+7)+(7+2+1) = 19$ 连续 $+ 10$ 离散 $= 29$）逐维拆解。注意论文 Eq. 1 自身各分量加起来只有 28（论文把 $R_i$ 写成 $\mathbb{R}^3$，代码实际是四元数 4 维——29 的缺口正在这里，以代码为准）：

| 分量 | 维度 | 含义 |
|---|---|---|
| $V_{i,j}\in\mathbb{R}^2$ | 2 | 边位移向量（NeuralTailor 底座） |
| $C_{i,j}\in\mathbb{R}^2$ | 2 | 二次 Bezier 控制 $(c_x, c_y)$ |
| $C^b_{i,j}\in\mathbb{R}^4$ | 4 | 三次曲线两控制点 |
| $C^r_{i,j}\in\mathbb{R}^3$ | 3 | 圆弧：半径 + 方向 flags（逆/顺 × 锐/优角） |
| $S_{i,j}\in\mathbb{R}^3$ | 3 | stitch tag（对应边的 3D 位置） |
| $R_i$ | 4 | panel 旋转（四元数） |
| $T_i\in\mathbb{R}^3$ | 3 | panel 平移 |
| $E^t_{i,j}\in\{0,1\}^2$ | 2 | 线型 2 位编码（代码：直 $[0,0]$ / 二次 $[0,1]$ / 圆弧 $[1,0]$ / 三次 $[1,1]$——论文 Sec 6 的编码表此处有笔误，与 straight 撞码，以代码为准） |
| $E^f_{i,j}\in\{0,1\}$ | 1 | 边存在 flag |
| $A_{i,j}\in\{0,1\}^3$ | 3 | 附着约束（下缘/左领/右领/抹胸/左袖窿/右袖窿 6 类 + 无） |
| $M'_{i,j}\in\{0,1\}^2$ | 2 | stitch mask + 反向缝线 flag |

19 维连续 + 10 维离散。**注意 $R_i, T_i$ 是 panel 级量被广播到每条边**——这样"版型 = 边向量矩阵 (panel × edge, 29)"一个张量装下，代价是每 panel 的位姿重复 ~37 次。输入维度 **39 panel 槽 × 37 边/panel = 1443 token**（代码 config `max_pattern_len: 39` = panel 槽位、`max_panel_len: 37` = 每 panel 边数上限——键名易误导，乘积与 GarmentDiffusion 论文口径的 1443 一致但两文方向相反，见 GG 08 勘误），压缩到 256 token 的潜空间（GG 08 详述）。

**维度演进的本质**：4 维（NeuralTailor）只编码"这一条边是什么"；29 维编码"这条边 + 它缝到哪 + 它的 panel 摆哪 + 它受什么约束"——**从边几何到版型语义**。表示越完整，下游模型（扩散）要"脑补"的越少，但生成空间越高维——这个张力是 GG 07–09 的主线。

---

## 4. 缝合的表示：三种方案

### 4.1 问题：缝合不是几何，是语义

版型 JSON 里 stitch 是**显式边对**——但学习/生成版型时，缝合信息怎么进网络？先看纯几何方案的原理性缺陷（数值实验 2）。

**实验 2a（正常情形）**：把 5 对真缝合边弯到同一空间圆弧上（模拟缝合后的 3D 形态）+ 1 条离群假边，用"边中点 + 切向"特征做贪心互最近匹配（阈值 5 cm）：

```text
匹配结果: 5 对，正确 5/5，离群假边无误配 ✓
```

**实验 2b（对称歧义）**：镜像对称的服装里，身体袖窿边到左袖边、右袖边的距离**精确相等**：

```text
袖窿边到左袖边距离 = 0.011180
袖窿边到右袖边距离 = 0.011180   （|Δd| = 0）
```

纯几何特征**原理上**无法区分该缝哪只袖子。这不是算法弱点，是问题的对称性——必须引入语义级信息。领域给出了三种方案：

### 4.2 方案一：stitch tags——可学习 3D 标签（NeuralTailor/DressCode）

每条边配一个 3 维可学习 tag $S$，训练目标是**同一 stitch 的边 tag 靠近、不同 stitch 的 tag 远于 margin**（triplet 式）。推理时：tag 距离近的边 → 判为缝合。DressCode 与 SewingLDM 沿用（SewingLDM 的 $S_{i,j}$ 定义为"对应边的 3D 位置"——把可学习标签换成了确定性几何特征，训练更稳）。

NeuralTailor 自己的最终方案更强：**成对分类器**——MLP 输入两条边的 8 维编码（两端点 3D 坐标 + 曲率坐标），输出"这两条边被缝合"的概率，每样本采 200 正对 + 200 负对训练。stitch tags 在他们的消融里只是基线。

### 4.3 方案二：Interface——DSL 级组件配对（GarmentCode）

程序化路线里缝合不逐边表达，而是**组件级接口**（§5.2 详述）：bodice 的"袖窿 Interface"与 sleeve 的"Armhole Interface"配对，**一条抽象 stitch = 一对 Interface**。边数不等时自动展平（边长按 interface 总长的分数互相投影、细分顶点）。语义在组件层，几何细节在 panel 层——换一个袖子设计，缝合指令不变。

### 4.3 方案三：隐式缝合标签场（ISP）

ISP（EPFL，GG 04 详读）的 2D panel SDF 网络对 $\Omega$ 内**每一点**输出"最近边界边"的缝合标签 $c$：标签 0 = 自由边（领口/下摆/袖口），非零标签 = 要与另一 panel 同标签边缝合。稀疏边界监督被稠密化成标签场——CNN 可以直接处理（§5.4）。

### 4.4 方案对照

| 方案 | 出处 | 粒度 | 歧义处理 | 适用 |
|---|---|---|---|---|
| 显式边对（JSON） | 2021 数据集格式 | 1-1 边 | 数据里给定 | 存储/评测 |
| stitch tags / 成对分类器 | NeuralTailor | 边 | 可学习语义 | 重建/序列生成 |
| Interface | GarmentCode | 组件接口 | 结构内置 | 程序合成 |
| 缝合标签场 | ISP | 连续场 | 隐式函数 | 隐式表示/可微 |

---

## 5. 四种版型编码

### 5.1 向量序列（NeuralTailor → DressCode → GarmentDiffusion → SewingLDM）

版型 = **边向量序列**：panel 间按类别固定排序（NeuralTailor 31 个类别槽位，缺席填零张量；SewFormer config 23 类），panel 内边环从左下角起逆时针，padding 到固定长度（NeuralTailor 每 panel 14 边；GarmentCodeData 时代涨到 39 panel 槽 × 37 边/panel）。

**序列的优越性**：天然适配 LSTM（NeuralTailor）、Transformer（SewFormer 两级 query、DressCode 的 SewingGPT 自回归、GarmentDiffusion 的 DiT 并行去噪）——整个生成链路的模型动物园都是序列模型的动物园。

**序列的代价**（GG 07 的全部动机）：token 数随 panel×edge 上限膨胀。DressCode 的量化 token 序列在其训练数据（2021 数据集）上 ~1,500 封顶，被移植到 GarmentCodeData 上时序列暴涨——**GarmentDiffusion 论文测得 18,135+2 个 token，SewingLDM v2 论文则说 "over 30k"（两文口径不一致，引用须注明出处）**——自回归逐 token 生成彻底失去可扩展性（SewingLDM 还报告 DressCode 在 GarmentCodeData 上 failure rate 11.4%，部分样本直接超 GPU 限制无法训练）。

### 5.2 参数化程序 DSL（GarmentCode）

GarmentCode（SIGGRAPH Asia 2023，Korosteleva & Sorkine-Hornung，ETH）把版型从"几何数据"变成"**程序**"。类结构（对照论文与 pygarment 代码）：

- **Edge**：基类即直线边（start/end 顶点）——**没有 StraightEdge 类**；子类 `CircleEdge`（第三点相对 $y$ 坐标，$x$ 固定 0.5）与 `CurveEdge`（1 个控制点 = 二次、2 个 = 三次）；`EdgeSequence` 组合有序边列；
- **Panel / Component(BaseComponent)**：Panel 是闭环边序列 + 6D 放置；Component 持有 `subs` 子组件列表——**递归组合**，任意层级（裙 + 上衣 → 连衣裙；compound 裙组件仅 50 行代码）；
- **Interface**：组件的可连接边的集合，可含**多个 panel 的多条边**（突破 2021 格式的 1-1 限制）；`Interface(panel, edges, ruffle=1., right_wrong=False)`——ruffle 是褶皱系数（缝合边长比），right_wrong 控制正反面。

**made-to-measure**：版型参数分 body 组（26 项测量：围度 6 + 宽度 5 + 纵向 8 + 点间距 2 + 距离 2 + 角度 3）与 style 组（含取值范围，供采样）；派生量用公式定义（如 `waist_level = height − head_l − waist_line`）。**参数重映射**保证换身体后版型仍有效（style 参数归一化后经线性插值映到由身体测量决定的合法区间——参数空间即合法空间，[GG 05](../GG_05_garmentcode_data/) §3.2 详述）——这就是 GarmentCodeData 能量产 11.5 万件**跨 5,000 体型**数据的机制（每件都合身）。

**DSL 的本质**：表示的不是某一个版型，而是**版型的生成规则族**——泛化性来自参数化（新身体测量 → 新版型），而不是来自学习。代价：表达空间被组件库圈定（论文自列局限：panel 内部开洞、口袋/荷叶边等缝在面料上的部件不支持）。

### 5.3 栅格编码（GarmentImage）

GarmentImage（SIGGRAPH 2025，东京大学 Igarashi 组）把版型画到**图像网格**上：张量形状 **(34, 16, 16)**——16×16 网格、每格 34 通道 = front/back 两层各 17 通道（1 个占用 flag + 8 个边类型 one-hot + 8 个变形矩阵通道）。变形矩阵 $F\in\mathbb{R}^{2\times4}$ 的四列是四条边在有向四边形中的形变向量——**cell 内的局部仿射映射**，与 GG 01 §3.1 的变形梯度 $F$ 同一思想（版型平铺空间的离散版）。

边类型 8 通道 = cell 的 **bottom 与 left 两条边各一个 4 类 one-hot**（GG 09 按全文核实）：NON-BOUNDARY / NON-STITCH / **FRONT-TO-BACK**（缝到另一层同位置）/ **SIDE-BY-SIDE**（缝到同层相邻 cell）——**缝合信息被空间化了**：两条要缝合的边在网格上就是相邻或同位置，不需要额外匹配机制。

**决定性优势数字**（D2 数据集，训练只见 3 种 panel 组合、测试第 4 种 unseen 组合的 IoU）：

| 表示 | seen 组合 | unseen 组合 |
|---|---|---|
| NeuralTailor 向量 | 0.8390 | 0.3131 |
| SewFormer 向量 | 0.9311 | 0.4260 |
| **GarmentImage 栅格** | 0.9304 | **0.8127** |

**unseen 拓扑组合上 IoU 接近翻倍**。原因：序列表示里"panel 组合方式"分布在 token 顺序上（unseen 组合 = unseen 顺序模式）；栅格表示里组合是**空间叠加**——新组合的 panel 各自占自己的网格区域，卷积核照常工作。**拓扑泛化 ⟹ 空间化表示**，这是谱系的第二个断层（§6）。

### 5.4 隐式场（ISP）

ISP（NeurIPS 2023，EPFL：Ren Li, Benoît Guillard, Pascal Fua）把版型写成**连续函数**：

$$
I_\Theta:\ \Omega\times\mathbb{R}^{|z|} \to \mathbb{R}\times\mathcal{N},\quad (x, z)\mapsto(s, c),
$$

$\Omega = [-1,1]^2$ 是 2D 版型空间，$s$ 是到 panel 边界的符号距离（SDF），$c$ 是缝合标签（§4.3）；配套 UV→3D 映射 $A_\Phi(x,z)\mapsto X$（AtlasNet 式）。每件服装一个 latent $z$（**auto-decoding**，DeepSDF 式，无 encoder；$|z|=32$），前/后片是共享 $z$ 的独立网络。缝合一致性靠损失 $L_{\text{consist}} = \sum\|A_{\Phi_f}(x,z) - A_{\Phi_b}(x,z)\|^2$（同一标签边折到 3D 后必须重合）。

**隐式场的优势**：拓扑变化平滑（panel 数量/边数变化 = $z$ 的小扰动，不需要 padding/槽位）；分辨率无关；可微到底（缝合一致性等几何损失可以直接作用在 $A_\Phi$ 上；物理自监督则由 ISP 第四阶段的独立 drape 网络承担——GG 04 §4.3）。**代价**：从 $z$ 恢复可制造版型要过 marching squares + 边环提取 + 缝合配对的后处理管线——隐式到显式的"解码税"。

### 5.5 四种编码对照表

| | 向量序列 | 程序 DSL | 栅格 | 隐式场 |
|---|---|---|---|---|
| 表示单元 | 边向量 | 类/组件 | 网格 cell | 连续函数值 |
| 拓扑变化 | 难（padding/槽位） | 参数化内置 | **空间叠加，天然** | **连续，天然** |
| 学习模型 | LSTM/Transformer | LLM/agent | CNN/U-Net | MLP/CNN |
| 可直接制造 | ✓（就是 JSON） | ✓（求值即版型） | 需解码 | 需解码（ marching squares） |
| 代表 | NeuralTailor→SewingLDM | GarmentCode→GarmentoPIA | GarmentImage | ISP→Latent FM |
| 篇目 | GG 06–08 | GG 05, 09 | GG 09 | GG 04, 09 |

---

## 6. 数值实验 3：直接量化版型坐标不可行（为什么需要潜空间）

生成模型爱离散潜空间（VQ-VAE、FSQ）。版型坐标能直接量化吗？把领口曲线坐标归一化到 $[-1,1]$（1.6 m 版型范围）做 $L$ 级均匀量化，对照理论 RMS 误差 $\text{step}/\sqrt{12}$：

| $L$ 级 | 量化步长 | 实测 RMS | 理论 RMS |
|---|---|---|---|
| 5 | 400.0 mm | 189.9 mm | 115.5 mm |
| 8 | 228.6 mm | 102.5 mm | 66.0 mm |
| 16 | 106.7 mm | 47.0 mm | 30.8 mm |
| 32 | 51.6 mm | 19.6 mm | 14.9 mm |

（实测高于理论：坐标不是均匀分布——领口这类带状结构整体落在少数格点间，误差系统性偏大。）

**5 级直接量化的 RMS 是 19 cm**——比整条领口的宽度还大，版型直接摧毁；32 级也还有 2 cm（缝纫不可接受）。**但 SewingLDM 的潜空间恰恰就是 5 级 FSQ**（每维 $\{-1,-0.5,0,0.5,1\}$，隐空间 $5^6$）且工作得很好——差异在哪？

**量化对象不同**：SewingLDM 量化的不是版型坐标，而是**训练学到的潜变量**。自编码器与量化器联合训练（量化在训练回路里，直通梯度反传），编码器学会把信息放进量化网格**能保住的方向**——潜空间里 5 级 = 每维 ~2.2 bits，6 维共 ~13 bits 的离散瓶颈，但每个"bit 组合"对应的解码输出是被优化的。GG 08 会看到这个设计的消融：256×32/256×12 的更大隐空间反而**生成失败**（codebook 利用率塌缩），256×6 是 Goldilocks 点——**瓶颈不是越小越好，是要匹配数据的有效自由度**。

---

## 7. 谱系总图（本系列核心叙事线）

```
边几何:  4 维 (v,c)  ──────────────►  29 维全信息（SewingLDM）
              │                              ▲
缝合:    边对 JSON → stitch tags/成对分类器 → 拼进边向量
              │                              ▲
载体:    序列 (LSTM/Transformer)  ─┬─► 潜空间扩散 (SewingLDM)
         程序 DSL (GarmentCode)    └─► LLM 程序合成 (GarmentoPIA)
         栅格 (GarmentImage) ────────► unseen 拓扑翻倍
         隐式场 (ISP) ─────────────► latent flow matching (SIG'26)
```

三个断层（每个断层都是一篇博客的主角）：

1. **几何 → 语义**（4 维 → 29 维）：边向量逐步吞噬 stitch/位姿/约束信息——GG 06/07/08；
2. **序列 → 空间/隐式**：拓扑泛化的需求把表示从 token 顺序搬到空间叠加与连续场——GG 09；
3. **坐标 → 潜变量**：生成模型的离散瓶颈（FSQ/VQ）要求"先压缩再量化"——GG 08。

---

## 8. 与系列后续的连接

| 后续篇目 | 用到本篇的哪个结论 |
|---------|------------------|
| GG 03/04 | 隐式场条目（ISP 是服装 UDF 的版型版）；GG 01 rest shape 桥接 |
| GG 05 | §5.2 DSL 全文展开（Interface/递归组件/made-to-measure） |
| GG 06 | §3.1 4 维边编码 + §4.2 缝合分类器（NeuralTailor/SewFormer 的核心） |
| GG 07 | §5.1 序列代价（18k/30k token 危机，两文口径不一致）；DressCode 的量化机制（乘 50/1000 取整） |
| GG 08 | §3.3 29 维分解 + §6 潜空间量化动机（直接对照 SewingLDM 论文） |
| GG 09 | §5.3 栅格与 §5.4 隐式场的两条新路线 |
| GG 10/11 | §1 "为什么版型"（可制造/可仿真/解耦）——无版型路线的对照基准 |

---

## 9. 勘误表（写进正文的依据）

调研中发现三处与坊间流传不符的事实，正文已按原文修正，此处集中列出：

1. **SewFormer 不是 Korosteleva 的论文**：作者为 Liu et al.（Sea AI Lab，arXiv:2311.04218），训练数据是自建的 SewFactory（约 100 万渲染对）而非 2021 版型数据集。边表示与 NeuralTailor 一致（4 维 $(v_x, v_y, c_x, c_y)$，局部坐标系），缝合匹配模块也与其类似；
2. **ISP 的 latent 不是 CVAE**：论文明确是 auto-decoding（DeepSDF 式，无 encoder）；$|z|=32$ 这个数字只见于官方代码而非论文正文。作者为 Ren Li, Benoît Guillard, Pascal Fua（EPFL）；
3. **GarmentImage 不是 Korosteleva 组**：作者是东京大学 Tatsukawa, Qi, Shen, Igarashi（arXiv:2505.02592）。另 GarmentCode 的作者是 Korosteleva & **Sorkine-Hornung**（"Korosteleva & Lee" 是其 2021 数据集论文的署名）。

---

## 10. 习题

**题 1**：证明二次 Bezier 曲线的控制点坐标 $(c_x, c_y)$ 中，$c_y$ 在 $(0, 1)$ 内时曲线的**最大曲率**出现在 $t = c_x$ 附近（即原文"$c_x$ 对应曲率峰值位置"的说法），并求 $c_y \to 0$ 时峰值的极限行为。
*提示*：二次 Bezier $B(t) = (1-t)^2P_0 + 2(1-t)tP_1 + t^2P_2$，取 $P_0=(0,0), P_1=(c_x, c_y), P_2=(1,0)$（边局部系），对 $t$ 求二阶导数构造曲率 $\kappa = \frac{|B'\times B''|}{|B'|^3}$，在 $c_y$ 小时展开。

**题 2**：NeuralTailor 的 panel 类别槽位（31 类，每类至多一个 panel）与 SewFormer 的 query 解码（23 类，动态数量）是两种"处理 panel 数量可变"的方案。分析两者在 (a) padding 浪费、(b) unseen 拓扑泛化、(c) 与栅格编码（§5.3）对比下的优劣。
*提示*：对照 §5.3 的 IoU 表——unseen 组合上槽位制 0.31 / query 制 0.43 / 栅格 0.81，差距的根源在"组合"这个自由度放在表示的哪个位置。

**题 3**：GarmentCode 的 `Interface` 允许一边集合对一边集合的缝合（自动展平细分），而 JSON 格式只存 1-1 边对。写出"接口边数 $m \ne n$"时展平算法的伪代码，并说明它为什么保持缝合总长守恒。
*提示*：把两边都按累计弧长比例参数化到 $[0,1]$，在两边的并集分割点处插入新顶点——每段 $[\frac{k}{K}, \frac{k+1}{K}]$ 上两边一一对应，$K = m + n - $（重合分割点数）。

**题 4**：实验 3 里实测 RMS 普遍高于理论值 step/√12。构造一个具体反例：坐标服从什么分布时，量化误差的 RMS 可以达到理论值的两倍以上？
*提示*：理论值假设误差在量化格内均匀分布。若坐标集中在格点边界附近（如 laplace 型分布跨在边界两侧），一半样本被推过边界、误差接近 step/2 而非 step/2√3。

**题 5**（思考）：如果把 SewingLDM 的 29 维边向量里的 $R_i, T_i$（panel 位姿，共 7 维）从边向量中拿出来单独用一路分支编码（类似 SewFormer 的 panel 级输出），边向量降为 22 维。这个改动对 (a) 自编码器重建、(b) 扩散生成质量、(c) 训练难度 各有什么影响？
*提示*：(a) 几乎无损——位姿本来就是 panel 级量，重复 39 次不增加信息；(b) 可能更好也可能更坏：去掉了"边向量内部位姿与几何的一致性"这个约束，扩散模型少学一类耦合，但也失去了单一张量的简洁；(c) 两路分支的条件注入复杂度上升。这类"信息放边级还是 panel 级"的设计选择是 SewingLDM 论文消融可以直接验证的问题——GG 08 精读时回来对照。

---

*下一篇：[GG 03：人体上的布料——从子网格到隐式场](../GG_03_body_conditioned_garments/)（TailorNet 的 pose×shape×style 分解 + SMPLicit 的 UDF 隐式场与拓扑自由）*
