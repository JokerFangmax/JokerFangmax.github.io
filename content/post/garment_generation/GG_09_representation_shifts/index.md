---
title: "GG 09: 版型生成的表示革新——栅格、隐式场与 LLM 程序合成（GarmentImage + Latent Flow Matching + GarmentoPIA）"
date: 2026-09-29
math: true
tags:
  - Notes
  - Garment Generation
  - Sewing Pattern
  - Implicit Representations
  - Flow Matching
  - LLM Agents
categories:
  - Garment Generation
description: "SewingLDM 之后版型表示的三条新路线：GarmentImage 把版型画成 (34,16,16) 栅格——缝合交互从 O(P²) 配对表变成空间局部共享规则，unseen 组合 IoU 0.81 vs 序列 0.31/0.43；Latent Flow Matching 把 panel 变成 SDF+UDF 隐式场、用 DiT 学 flow matching 速度场（无 CFG），ISO-surface gradient 让'生成→网格→仿真损失'整条链可微；GarmentoPIA 用三角色 LLM agent 把 1942 年打版文献翻译成尺规 DSL 参数化程序——有效性由构造保证而非学习目标。本篇三个数值实验分别还原：局部规则 vs 配对表的组合泛化机理（unseen 误差 0.0000 vs 0.5796）、SDF 拓扑分裂的鞍点连续性 + 隐函数定理可微网格化（1e-13 有限差分验证）、DSL valid-by-construction（闭合残差 1e-16 vs 量化表示 19/20 超差）"
---

# GG 09：版型生成的表示革新

> 📖 服装生成系列第九篇（前篇：[GG 08：SewingLDM 深度精读](../GG_08_sewingldm_deep_dive/)）。GG 07/08 走完了"向量序列 → 连续 token 并行扩散 → 压缩潜空间"的主干——那条路把延迟打进了 1 秒量级、把 FLOPs 预算换成了模型能力，但**表示本身没变**：版型仍然是"panel 槽 × 边向量"的定长矩阵。本篇讲三条**换表示**的路线：栅格（GarmentImage）、隐式场 + flow matching（Latent Flow Matching）、LLM 程序合成（GarmentoPIA）——分别对着向量表示剩下的三个瓶颈。三篇论文全部对照 arXiv HTML 全文 / 会议正式 PDF / 官方仓库核实（含 8 处流行误传的勘误，见 §8.2）。

---

## 1. 这组论文解决什么问题？

GG 08 结尾留下的状态：SewingLDM 用 29 维边向量 + 256×6 FSQ 潜空间 + T5 条件，把"文本 → 复杂版型"做到了 clothes-to-body 2.20cm。但向量 token 表示有三个**结构性**瓶颈，加数据、加参数都治不了：

1. **拓扑组合泛化**。版型的合法空间里，"panel 怎么组合"是一个自由度。向量序列把这个自由度摊在全局 token 相关性上——每种组合方式都要训练数据里见过才学得会。unseen 组合一来（每个 panel 都见过、组合没见过），序列方法直接崩（SewFormer unseen IoU 0.4260，见下）；
2. **可微性断裂**。下游任务（布料仿真、refitting、视频对齐）对**几何**求损失。向量表示里"边端点在哪"由离散的边数与连续的参数共同决定——拓扑动一下，梯度就断了。GG 07 实验 2 已经量化过一侧（量化破坏闭合），另一侧（拓扑跳变）连损失都没法定义；
3. **分布外即非法**。学习路线学的是训练分布的投影——分布外采样给出的是"看起来像 token 序列但不是版型"的东西（GG 08 实验 1 的 codebook 0% 利用率就是极端形态）。有效性（闭合、缝合一致、可裁剪）从来不是学习目标，只是被数据分布"顺带"满足。

三条路线各攻一个瓶颈，而且**互不引用**（LFM 与 SewingLDM/DressCode 零定量对比；GarmentoPIA 明言"问题不同无法直接比较"）——这是三场单打，不是一场擂台：

| 论文 | venue | 任务 | 表示 | 生成/求解机制 | 攻的瓶颈 |
|---|---|---|---|---|---|
| GarmentImage | SIGGRAPH 2025 Conf. | 表示研究（潜空间探索 / 文本编辑 / 图→版型） | (34,16,16) 栅格张量 | CNN-VAE + SDS + 三级联 U-Net | ① 组合泛化 |
| Latent Flow Matching (LFM) | SIGGRAPH 2026 Conf. | 图→版型估计 + 无条件生成 + refitting/补全 | 每 panel SDF+UDF 隐式场 | 标准 flow matching + DiT | ② 可微性 |
| GarmentoPIA | CVM 2026 | 文献 → 参数化版型程序 | 尺规原语 DSL | 三角色 LLM agent + SQL 状态机 | ③ 分布外有效性 |

一个先说清的定位（勘误 1 的正身）：**GarmentImage 是表示论文，不是生成模型论文**。它证明的是"栅格编码让组合泛化成为卷积的局部性"，生成侧用的全部是现成的朴素组件；把它写成"新生成模型打爆扩散基线"就是误传。三篇的真正公共点是：**把瓶颈从"学更大的分布"挪到"换一个让瓶颈消失的表示"**。

### 方法流程鸟瞰

三条路线各是一条完整管线（细节在 §3–4）：

| 论文 | 流程 | 每步在买什么 |
|---|---|---|
| GarmentImage | 版型 → 画上 16×16 栅格（占用 + 边类型 + 变形矩阵，34 通道）→ 确定性三步解码回版型；应用侧全用现成组件（CNN-VAE 潜空间 / SDS 文本编辑 / 三级联 U-Net 图→版型） | 缝合从"全局配对表"变成"读邻格"——unseen 组合泛化成为卷积局部性的红利 |
| Latent Flow Matching | 每 panel 学 SDF + UDF 两个场 → VAE 压成潜码 + 位姿 token（无位置编码）→ DiT 学 flow matching 速度场（无 CFG）→ 可微网格化接下游损失 | 拓扑变成连续场：生成 → 网格 → 仿真损失整条链可微；缝合作为独立下游任务预测 |
| GarmentoPIA | 人工预处理文献指令 → 三角色 LLM agent（翻译/生成/校验）经 SQL 状态机产出尺规 DSL → SymPy 依赖排序 → 发射自包含 Python 程序 | 有效性由构造保证：LLM 随机性被 waiting list → 校验 → 晋升的机制挡在正式状态之外 |

---

## 2. 背景最小集

| 需要 | 出处 |
|---|---|
| 四种版型编码总览（向量/程序/栅格/隐式）、GarmentImage 张量形状、unseen IoU 表 | [GG 02](../GG_02_pattern_representations/) §5.3–5.4, §6 |
| ISP 的隐式版型（SDF panel + UV 映射 + 物理损失）——LFM 的直接前作 | [GG 04](../GG_04_self_supervised_draping_isp/) |
| GarmentCode DSL 的 OOP 设计与 GarmentCodeData | [GG 05](../GG_05_garmentcode_data/) |
| 向量 token 化的两代方案（离散 AR / 连续扩散）与各自的结构约束损失 | [GG 07](../GG_07_text_to_pattern_ar_diffusion/) §5.2 |
| FSQ 潜空间、CFG 模式压制、条件注入方式 | [GG 08](../GG_08_sewingldm_deep_dive/) |
| Flow matching 的速度场 / 直线路径 ODE | [VG 01](../../video_generation/VG_01_diffusion_basics/) §4 |

---

## 3. 表示法详解

### 3.1 GarmentImage：把版型画到网格上

张量形状 **(34, 16, 16)**：16×16 网格，34 通道 = front/back 两层各 17 通道。每个被占用的 cell 携带：

- **1 个占用 flag**：这个格子上有没有版型材料；
- **8 个边类型通道**：cell 的 **bottom 与 left 两条边各一个长度 4 的 one-hot**（注意不是"8 种边类型"——勘误 2）。4 类是：NON-BOUNDARY（非边界）/ NON-STITCH（边界但不缝合）/ **FRONT-TO-BACK**（缝到另一层同位置的边）/ **SIDE-BY-SIDE**（缝到同层相邻 cell）；
- **8 个变形矩阵通道**：$F \in \mathbb{R}^{2\times4}$，四列对应 cell 的四条边，每列 $f = \bar v_j - \bar v_i$（原版面上四边形边的终点减起点）。$F$ 把单位方格 cell 映射到版面上的任意四边形——**占用说"在哪"，$F$ 说"长什么样"**。

**解码是确定性三步**（全自动，无学习）：① 按"被边界边包围的 inside cell 簇"切分 panel；② 最小二乘恢复形状——$\min \sum \|\hat v_j - \hat v_i - f_{ij}\|^2$，同时约束网格顶点靠近格点位置（把变形场拉回一致几何）；③ 边类型映射到变形后的 panel 边，得到缝合关系。cell 在网格上的位置本身就编码了 panel 围绕身体的摆放。

**与 GG 02 表格的对照**：这正实现了 GG 02 §5.3 说的"缝合信息被空间化"——两条要缝合的边在网格上就是同位置（FRONT-TO-BACK）或相邻（SIDE-BY-SIDE），**匹配机制从"学出来的 3D 标签比对"变成了"读邻格"**。

### 3.2 Latent Flow Matching：panel 变成两个标量场

LFM（Cao, Li, Dumery, Li）把每个 panel 表示成一个连续函数 $f(x, z)$，$x \in [-1,1]^2$ 是版面坐标，输出**两个场**：

- **SDF $d_c$**：到 panel 闭合边界的符号距离（内负外正），零水平集 = 轮廓；
- **UDF $d_p$**：到边端点集合 $\{o_1, o_1', \dots, o_{N_e}, o_{N_e}'\}$ 的无符号距离，零根 = 边端点——**用它把轮廓环切分成一条条边**，缝合匹配才有"边"这个单位。

注意**没有 stitch 标签场**（对比 SewingLDM 的 3 维 stitch tag、GarmentImage 的边类型通道）：缝合关系不进表示，由独立的缝合预测模块（§4.2）从边几何推断——这是"表示只管形状、关系作为下游任务"的设计选择。

编码器是 **point transformer**（边界采样点 → 潜码 $z$），解码器是 MLP 实现 $f$；VAE 损失 = SDF 重建 + UDF 重建（各带 eikonal 型梯度惩罚 $(\|\nabla d\|^2 - 1)^2$）+ $\lambda_{KL}\cdot$KL。与 ISP（GG 04）的关键区别：ISP 用 auto-decoding（无编码器）、**每种拓扑单独建模**；LFM 有编码器、统一框架支持任意 panel 数 / 边结构 / 缝合配置。

### 3.3 GarmentoPIA：版型 = 可执行的尺规程序

GarmentoPIA（KAIST 文化技术研究生院）的输入不是 prompt 也不是图像，而是**服装打版文献的文本**（主源：H. Pepin《Modern Pattern Design》1942 年版）——目标产物是**参数化版型模型**：吃身体尺寸 + 服装尺寸的独立 Python 程序，可反复运行、零 LLM 成本。

它定义的是**自家尺规（ruler/compass）风格 DSL**（勘误 6 的正身——不是 GarmentCode）：

```text
make_a_point_by_relative_distance(anchor, dx, dy)   # 相对定位
make_a_circle(...)                                  # 圆规
make_a_french_curve(...)                            # 法式曲线（Bézier 参数化）
```

配合命名规范（`PointC`、`Segment PointA Point3`）与依赖关系（点依赖点、线依赖点）。GarmentCode（GG 05）仅作为"第一个版型 DSL"被引用；最终产物是**不依赖 LLM 的自包含程序**（论文 Algorithm 1：递归解析依赖 → 拓扑排序 → 发射 Python 代码）。

三种表示放在 GG 02 的谱系里看：栅格与隐式场是"把拓扑变成空间/场的连续几何"，DSL 是"把拓扑变成类型系统"——前两者为了**可学与可微**，后者为了**有效与可复用**。

---

## 4. 方法核心与数学推导

### 4.1 GarmentImage：三个朴素应用证明一个表示

表示论文的论证方式：拿现成组件搭三个应用，证明同一份编码通吃。

1. **潜空间探索**：CNN-VAE 压到 32 维潜码（基线是 Motionformer 风格的向量 VAE）；
2. **文本编辑**：在 VAE 潜空间优化，图像解码器输出 64×64 模拟图，用 **Stable Diffusion v1.5 的 SDS loss** 对齐文本（VG 02 §4 的 SDS 在版型域的实例——注意**没有直接生成版型的扩散模型**）；
3. **图→版型预测**：**三个级联 U-Net**，依次预测占用 flag（L2 损失）→ 边类型（以 image+flag 为输入，交叉熵）→ 变形矩阵（以 image+flag+边类型为输入，L2）——级联的理由是任务难度递增、输入递增。

**没有消融实验**（论文自无此项）——论证全部压在 D2 的 unseen 组合数字上（§5.1）。

### 4.2 LFM：标准 flow matching + 两个关键工程决策

**生成模型是教科书级标准**（勘误 4 的正身）：Lipman et al. (ICLR 2023) 的 flow matching，线性插值路径，DiT 骨干学速度场：

$$
\mathcal{L} = \mathbb{E}_{X_0 \sim \mathcal N,\ t \sim \mathcal U}\big\| V_t - U_\theta(X_t, t) \big\|^2, \qquad X_t = (1-t)X_0 + tX_1,
$$

**全文无 classifier-free guidance**（0 处提及）、无 OT 条件路径、无 rectified flow。"guidance"一词只出现在补全任务语境：按 Patel et al. (2025) 的控制策略，对 $X_t$ 做梯度下降，使估计的去噪样本 $\hat X_1 = X_t + (1-t)U_\theta$ 匹配用户给定的 panel 潜码（式 15）——**这是轨迹控制，不是 CFG**。

两个值得记的工程决策：

- **移除位置编码**。每个 panel 一个 token $\tilde z = [z,\ T \in \mathbb{R}^3,\ R \in SO(3)]$（潜码 + 平移 + 四元数旋转），self-attention 联合处理、训练时 zero-pad 到最大 panel 数。去掉位置编码使 token 集合**顺序无关**——补全任务（给一半 panel 潜码生成另一半）才成立。对照 GG 07：AR 的位置即结构、扩散的位置编码是归纳偏置——LFM 证明版型生成可以两者都不要（panel 集合是无序的）；
- **图像条件双通路**：**DINO-v3** patch tokens 与噪声潜 token 拼接过 self-attention（几何对应），**CLIP** 视觉语义嵌入过 cross-attention（语义对齐）——GG 08 讲过的"注入方式是设计空间"再添一例。

**缝合预测**是独立模块：沿每条边采点 → point-transformer 式编码 → self-attention → 边描述子 $f_e$；分类头判"是否缝合边"；成对互补性 logit 用 Jigsaw 式特征解耦（Lu et al. 2023）：

$$
f_c^{i,j} = G'(f_e^i)^\top A\, G(f_e^j) \to \sigma(\cdot), \qquad \mathcal{L}_{stitch} = \text{BCE},
$$

后处理做 GarmentCode 式 stitch flattening（按缝线边长细分边），把 many-to-many 匹配转成 one-to-one（布料模拟器的要求）。

**可微网格化**（本文的技术核心，勘误 5 的正身——不是 marching squares）是 ISP 式流程：初始化方形网格，各顶点求 SDF，全正面的三角形丢弃，正值顶点**投影**到零水平集：

$$
v \leftarrow v - d_c(v, z)\, \nabla d_c(v, z) \qquad \text{(式 6)}
$$

可微性经 iso-surface gradient 恢复（式 8，承 MeshSDF / Remelli 2020 与 Guillard 2024）：

$$
\frac{\partial v}{\partial z} = -\nabla d_c(v, z) \cdot \frac{\partial d_c}{\partial z}(v, z),
$$

端点检测：$d_p < \epsilon$ 的稠密网格点梯度下降到零根，DBSCAN 聚类中心近似端点，端点切分边界环。**这条链的价值在 refitting 实验**：目标服装距离 $\mathcal{L}_{CD}$ + Laplacian 平滑 + 缝长一致性，经 ContourCraft 神经布料模拟**可微反传**回潜码——"生成 → 网格 → 仿真损失"端到端（§5.4 的数字）。

### 4.3 GarmentoPIA：三角色 agent + SQL 状态机

流程三阶段（LangGraph + SymPy + SQLite3 实现）：

1. **文本预处理（人工！）**：把文献指令转成五段式 [起点, 动作, 方向, 距离/条件, 结果点]——论文承认当前 LLM 无法直接读原始打版书（歧义、省略、跨章节引用）。**人在回路是明说的，不是隐藏的**；
2. **组件生成模块**：三个 prompt 角色智能体，全部经 SQL 操作 Pattern Component DB（Point/Line/Constant 表 + Waiting List 表，字段 {Name, Equation, Step Number}）——
   - **Draft2DSL Translator**（GPT-o4mini）：逐条指令 → DSL 代码，RAG 检索已有组件；
   - **Component Entity Generator**（GPT-4.1）：生成 SQL 插入 waiting list，查重；
   - **Component Entity Reviewer**（GPT-4.1）：校验；发现错误则触发 **prompt 自精炼**（更新单句 supporting rules/examples，清空 waiting list 重跑）；通过则晋升入正式表；
3. **后处理**：语法/语义校验、依赖排序（SymPy 解符号依赖）、输出 Python 模型。

设计上最有信息量的一点：**用数据库而非对话上下文做状态**。LLM 的每次产出先进 waiting list、校验通过才晋升——LLM 随机性被挡在正式状态之外。这直接回应了它的局限 2（"LLM 随机性沿流程传播"）：Reviewer 拦不住的（错误的但合法的 DSL）会传播，但**语法层面**的错误被 SQL schema 与校验拦死（§5.3 的语法错误率 4.8% vs baseline 37.1%）。

---

## 5. 实验与关键数字

实验脚本：`scripts/garment_generation_experiments/gg09_experiments.py`（numpy/scipy，无学习框架）。三个实验分别给三条路线的**机理**做最小还原。

### 5.1 实验 1：局部共享规则 vs 全局配对表——unseen 组合泛化的机理

**设定**：玩具世界里 6 种 panel、21 种组合（两 panel 服装），留出 (0,5) 作 unseen。同一个 pairwise 缝合交互写进两种表示：

- **栅格侧**：缝合通道落在两 panel 的空间边界上，行 $y$ 的值 = $\varphi(\text{dir}_R[p,y],\ \text{dir}_L[q,y])$——一个**与 panel 身份无关的局部规则**（朝向对 → 缝合强度的 4×4 表）；
- **序列侧**：stitch tag 挂在 panel token 上，值 = $\alpha_p + \beta_q + \gamma_{pq}$——含 **pair 专属交互项** $\gamma$（配对表，21 对各一个）。

两侧都用最小二乘从 20 个 seen 组合学，测 unseen 组合 (0,5)：

| | seen 组合 | unseen 组合 |
|---|---|---|
| 栅格（局部规则 $\varphi$） | 0.0000 | **0.0000** |
| 序列（配对表 tag） | 0.2189 | **0.5796** |

unseen 误差 0.5796 ≈ 留出对的 $|\gamma| = 0.475$ 量级——**加性最优预测对交互项原理上不可预测**，因为 $\gamma_{05}$ 在训练数据里一行都没有。而栅格侧精确外推：seen pairs 里 (0,1)…(4,5) 的缝合通道**全部都是** $\varphi$ 的样本（4×4 表的每一格被反复填充），unseen 对 (0,5) 只是把学到的 $\varphi$ 用在新输入上——误差 0.0000 不是"泛化得好"，是**同一张表的新索引**。

端到端口径（占用通道 90% 精确 + 缝合通道 10% 由交互决定）：栅格 unseen 组合准确率 ~1.000 vs 序列 ~0.907——**和论文数字的量级对上了**：

| 表示 | seen 组合 | unseen 组合 | 论文出处 |
|---|---|---|---|
| NeuralTailor 向量 | 0.8390 | 0.3131 | GarmentImage Table 1（D2，质心对齐 IoU） |
| SewFormer 向量 | 0.9311 | 0.4260 | 同上 |
| **GarmentImage 栅格** | 0.9304 | **0.8127** | 同上 |

D2 协议：训练只见 3 种组合（单片连衣裙 / 带袖连衣裙 / 上衣+裤），测试第 4 种（**带袖上衣+裤**）——每个 panel 单独都见过，组合从未见过。机理读数：**"组合方式"这个自由度，栅格把它空间化为局部邻接规则（O(P) 个模式），序列把它摊在全局 token 相关性上（O(P²) 个模式）**——unseen 组合数占总组合数的比例随 P 增长（P=6 时 1/21，P=20 时过半），序列方法在这个自由度上是在背乘法表。

### 5.2 实验 2：SDF 的拓扑平滑插值 + 可微网格化

**(a) 拓扑经鞍点连续分裂**。两个版型：T 恤前片（1 大 panel，SDF 光滑并）与左右襟（2 panel，中缝分开）。SDF 线性插值 $s_t = (1-t)\,s_{\text{shirt}} + t\,s_{\text{doublet}}$：

| $t$ | 内部占比 | 零集 cell 数 | 与 $t=0$ 的 IoU |
|---|---|---|---|
| 0.00 | 0.215 | 576 | 1.000 |
| 0.25 | 0.188 | 520 | 0.876 |
| 0.50 | 0.170 | 344 | 0.789 |
| 0.75 | 0.169 | 356 | 0.787 |
| 1.00 | 0.160 | 528 | 0.743 |

**相邻插值步（Δt=0.01）的 IoU 最小值 = 0.978**——1 panel → 2 panel 的中缝是场从鞍点分裂出来的，全程连续，不需要任何离散决策。序列表示里同样的 morph 需要 presence flag 跳变（中间态"半个 panel"不可解码）。**隐式场把"拓扑"变成了场的连续几何**——这正是 LFM 能对潜码 $z$ 做连续优化（refitting、补全）的前提。

**(b) 零水平集点的可微性（隐函数定理）**。取 $y = y_0$ 切片，参数 $\theta = (t, s)$（插值参数 + 尺寸），零点 $x^\ast(\theta)$ 满足 $s(x^\ast;\theta) = 0$：

$$
\frac{d x^\ast}{d\theta} = -\frac{\partial s / \partial \theta}{\partial s / \partial x} \qquad \text{（隐函数定理）}
$$

解析梯度 vs 有限差分（对 brentq 求出的 $x^\ast$ 直接差分）：

```
零点 x* = 0.347000
隐函数定理 dx*/dθ = [-0.010000, +0.293000]
有限差分           = [-0.010000, +0.293000]
相对误差           = 9.96e-13
```

这就是 LFM 式 8 的 $\partial v/\partial z = -\nabla d_c \cdot \partial d_c/\partial z$ 的标量版机理：**网格化不是梯度的死胡同**——marching-cubes-style 提取的顶点位置对场参数可微，"生成 → 网格 → 布料仿真损失"（ContourCraft 反传）才能端到端。论文的定量佐证：refitting 实验 triangle quality **0.875（潜空间优化）vs 0.781（边控制点直接优化）**——可微链不但通，还比直接优化几何参数更好（潜码的参数化自带正则）。

### 5.3 实验 3：DSL 的 valid-by-construction + 组合爆炸

**(a) 构造即闭合**。玩具 DSL（6 组件类 × 4 变体，程序 = 组件列表，求值 = 每 component 一个闭合边环），2000 个随机程序求值 → 3989 个边环：

```
闭合残差: max = 8.88e-16（IEEE 双精度，构造即闭合）
对照 GG 07 实验 2: 量化 token 表示 h=20mm 时 19/20 版型闭合超差
```

**有效性是类型系统性质，不是学习目标**——求值器对任意程序输出闭合版型，与训练分布无关。

**(b) 程序空间计数**。组件库 6 类 × 4 变体 = 24 种组件，长度 1..d 的程序数：

| depth | 程序数 | GarmentCodeData 115,195 件覆盖率 |
|---|---|---|
| ≤ 2 | 600 | 全覆盖 |
| ≤ 3 | 14,424 | 全覆盖 |
| ≤ 4 | 346,200 | 33.3% |
| ≤ 5 | 8,308,824 | 1.4% |

玩具组件库已如此；真实 GarmentCode 组件库（GG 05 的 Edge/Panel/Interface/Stencil 层级）远大于此，覆盖率更低。**学习路线学的是程序空间的投影**——这给了 GarmentoPIA 动机的定量形状：程序求值器对任意程序有效（泛化性来自求值规则），学习模型只在分布内有效（泛化性来自数据）。

**论文自报数字**（7 个品类 × 每类 5 次；成功标准极严：SVG 与人工参考点一一对应、xy 坐标完全一致、path 命令完全一致）：

| 系统 | 单 panel 成功率 | 双 panel | 总成功率 | 语法错误 | 数值错误 | mIoU |
|---|---|---|---|---|---|---|
| Baseline（单 LLM 直接出 SVG） | 30.0% | 0.0% | 17.1% | 37.1% | 45.8% | 0.825 |
| GarmentoPIA−−（仅 DSL、单轮） | 40.0% | 20.0% | 28.6% | 5.7% | 65.7% | 0.528 |
| GarmentoPIA−（无 prompt 精炼） | 50.0% | 33.3% | 42.9% | 54.2% | 2.9% | 0.979 |
| **GarmentoPIA** | **90.0%** | **80.0%** | **85.7%** | **4.8%** | 9.5% | 0.975 |

三行消融读出来的机理：DSL 把**语法**错误从 37% 压到 6%（但数值错误反升到 66%——DSL 语法比 Python 难，值算错照样非法）；SQL 查重 + Reviewer 把**数值**错误压到 2.9%（但语法错误回升到 54%——翻译层压力变大）；prompt 自精炼把两者同时压住（4.8% / 9.5%）。**每道防线各拦一类错误，缺一层漏一类**——这是"agent + 状态机"结构本身的消融，不是换更强的 LLM。换骨干的表也支持这个读法：Qwen3 Coder Plus 68.6%（每步 ~2s）、GPT-5 85.0%（~70s）、GPT-o4mini+4.1 85.7%（~224s）——骨干决定速度与下限，**结构决定上限**。

### 5.4 其余关键数字

**GarmentImage**：自动编码（版型→栅格）成功率 **89.88%**（NeuralTailor 数据集连衣裙）——编码方向也有 10% 失败，且论文自列"编码不唯一：同一版型可对应不同 GarmentImage"。

**LFM**（单视图图→版型；Sewformer 在作者 split 上重训、AIpparel 用公开预训练）：

| 数据集 | 方法 | IoU↑ | Trans L2↓ | #Panel↑ | #Edge↑ | DSR↑ |
|---|---|---|---|---|---|---|
| SewFactory | Sewformer | 0.821 | 0.367 | 86.06% | 97.35% | 63.1% |
| SewFactory | **LFM** | **0.847** | **0.296** | **88.92%** | **98.63%** | **81.0%** |
| GCD | AIpparel | 0.834 | 1.783 | 91.39% | 86.53% | 62.6% |
| GCD | **LFM** | **0.892** | **1.274** | **91.99%** | **93.26%** | **69.0%** |

（Rot L2 略差于基线，论文称对悬垂无实际影响。）DSR = 可成功组装并悬垂模拟的版型比例——**"仿真就绪"第一次进了主表**（GG 07 说过的评测盲区，这里补上了一角）。缝合精度 Table 2：SewFactory F1 **0.993** vs Sewformer 0.956；GCD **0.981** vs AIpparel 0.821。

**关键空白**：LFM 全文与 SewingLDM / DressCode / GarmentImage **零定量对比**（基线只有 Sewformer / AIpparel）——本篇不做"谁更强"的跨表比较，各表只在自己协议内读。

---

## 6. 论文间关系

```
                    SewingLDM (ICCV'25)：向量 token + FSQ 潜空间 + IDDPM
                        │ 剩余瓶颈：① 组合泛化 ② 可微性 ③ 分布外有效性
        ┌───────────────┼───────────────────┐
        ↓               ↓                   ↓
  GarmentImage      Latent FM            GarmentoPIA
  (SIG'25, 东大)     (SIG'26, MBZUAI)      (CVM'26, KAIST)
  栅格 (34,16,16)    SDF+UDF 隐式场         尺规 DSL
  CNN-VAE/SDS/U-Net  Lipman FM + DiT        LLM agent + SQL
  攻 ① 组合泛化      攻 ② 可微性            攻 ③ 有效性
```

**谱系定位**（对照 GG 02 §6 的四表示谱系）：

- GarmentImage 是"栅格"列的第一篇**生成侧**论文（此前栅格只做可视化），且证明的是表示性质而非模型性质；
- LFM 是 ISP（GG 04）的精神续作，但**主体移到了 MBZUAI**（勘误 3：Ren Li——ISP 一作——现属 MBZUAI/SUSTech，作者里没有 Guillard 与 Fua，仅 Dumery 一人在 EPFL）。承接关系在技术上清晰：SDF panel → SDF+UDF；auto-decoding → point transformer 编码器；单拓扑建模 → 统一任意拓扑；GG 04 的物理损失 → ContourCraft 可微仿真；
- GarmentoPIA 与 GarmentCode（GG 05）同属"程序 DSL"列，但方向相反：GarmentCode 是**人写 DSL → 机器造数据**（11.5 万件喂学习路线），GarmentoPIA 是**机器读文献 → 写 DSL**——学习路线的数据源头，反过来成了 LLM 的输出端。

**三篇互不引用的读法**：不是失礼，是问题确实不同——GarmentImage 做表示与编码质量、LFM 做估计与可微下游、GarmentoPIA 做文献数字化与程序资产。它们与 GG 07/08 生成主干的关系是**补集**而非竞争：主干解决"从条件到分布内版型"，三条新路线分别解决"分布的组合结构""梯度的通路""分布外的合法性"。

**旁注**（GG 12 会展开）：SewFusion（arXiv:2609.23548，2026-09）用"自回归拓扑 + flow matching 面板几何"的混合路线——**FM 处理连续几何、离散结构另行处理**正在成为小趋势，与 LFM 的"FM 连续潜码 + 离散缝合分类头"同构。

---

## 7. 与 Work 3 的连接

Work 3（视频先验蒸馏 → 缝纫版型生成，SewingLDM P0 baseline）从本篇拿走四件事：

1. **蒸馏接口的表示选择**。视频先验（Wan2.1 blocks 8/14/17/21/29 的 3D 对应 + 遮挡补全信号，P2 实测）是**空间栅格特征**。GG 08 的注入口吃的是序列潜码（视频特征要池化/再投影），GarmentImage 的实验 1 机理说明：**空间对齐的表示让"对应关系"成为局部规则**——如果 Work 3 的下游表示是栅格/隐式场，视频特征的空间结构与版型的空间结构直接同构，蒸馏信号不需要经过"序列化"这道有损变换。P3 设计蒸馏头时，这是一个真实的架构选项（栅格版型头 vs SewingLDM 潜码头 + 视频条件）；
2. **可微链是蒸馏的第二通道**。GG 08 §7 讲过"§4.3 注入口做条件蒸馏"（方案 A）；LFM 证明另一条路：**损失端蒸馏**——视频先验出服装 mesh，经可微网格化 + 仿真损失反传到版型潜码。这条路不依赖 SewingLDM 的注入设计，但要过 FSQ 直通梯度（离散码）——**如果蒸馏目标改成 LFM 式连续潜码，梯度链干净得多**。代价是 LFM 没有"文本→全版型生成"的证据（只有无条件生成 + 图像条件估计）——P3 若走这条路等于换 baseline，需要单独的最小验证；
3. **评测自建清单 +2**。本篇确认了两个 SewingLDM/DressCode 评测里都没有、但决定可用性的指标：**unseen 组合 IoU**（D2 协议：训练按组合类型留出，Work 3 的 20 类组合可以照样切）与 **DSR**（可组装 + 可悬垂比例）。P0 的复现评测该把这两项与 GG 07 实验 2 的闭合/缝长一致性一起补上——这四项是我们能claim"视频先验带来结构收益"的证据位；
4. **GarmentoPIA 的模板思路**。它自述生成的参数化模型"可作为模板引导已有服装预测方法"——Work 3 的视频先验恰好能提供这个模板：视频给出的 panel 拓扑 + 摆放先验，可以把 SewingLDM 的生成空间从"全分布"收缩到"拓扑正确的子分布"。这与 GG 08 实验 2 的结论（提高 $\pi_c/\pi_u$ 分离度唯一可行）在机制上一致——**先验的价值是分离度，不是额外数据**。

一个诚实的边界：三条路线都没在"文本→复杂版型"主任务上与 SewingLDM 交过手，Work 3 的 baseline 地位没有被本篇动摇——被动摇的是"向量 token 表示是终点"这个隐含假设。

---

## 8. 局限与勘误

### 8.1 局限

**GarmentImage 自列**：边界不平滑（重建版型可能扭曲）；**编码不唯一**（同一版型可对应不同 GarmentImage，需统一转换策略——解码端 89.88% 成功率的另一面）；自动编码只支持简单品类；预测可能产生非法边类型组合（简单可自动修复、复杂需人工）；数据只有正面模拟渲染图、缺真实照片；两层结构不支持 tucks（需新增 TUCK 边类型），领口/袖口/口袋等分层特征需新层。**方法层面**：无消融；生成侧全是朴素组件（CNN-VAE / SDS / U-Net），栅格表示在强生成模型（扩散/FM）下的上限未知——**表示的证明与生成的证明之间有一段没人走过的空白**。

**LFM 自列**：拓扑仍是离散的（panel 数不可微——实验 2(a) 的连续分裂只在插值路径上，生成时 panel 数仍由 token 数决定）；只建模外边界，不含内部缝线/口袋/装饰；数据驱动生成可能产生分布外潜码 → 不规则场/无效 panel，缝合分类误差会传播；单视图估计本质歧义 + 合成数据训练，真实图像有域差。**方法层面**：消融只在补充材料（正文无）；与 SewingLDM/DressCode 零对比——"可微性收益"的全部证据来自 refitting/补全两个下游，主任务（文本/图像→版型质量）没有跨路线数字。

**GarmentoPIA 自列**：需人工预处理文献（建议未来做半自动翻译器）；LLM 随机性在初始阶段引入不一致并沿流程传播（建议自部署模型 + 固定温度/种子）；prompt 精炼积累的 rules/examples 越来越大，逼近 token 上限，长指令序列会出错/遗漏（计划符号计算混合架构、prompt 压缩、记忆增强）。**方法层面**：成功标准是"与 1942 年人工参考逐点坐标一致"——衡量的是复刻保真而非设计空间；7 个品类都是基础原型（前衣身/袖子/裙子/裤子），离"生成任意服装"还有整本打版书的距离。

### 8.2 勘误表（写作前逐一核实、修正的流行误传）

| 误传 | 事实 | 依据 |
|---|---|---|
| GarmentImage 是新的生成模型（扩散/AR） | **表示论文**：CNN-VAE 潜空间探索 + SDS 文本编辑 + 三级联 U-Net 图→版型，无版型扩散模型、无消融 | 论文 §4 |
| 每 cell 有"8 种边类型" | **bottom 与 left 两条 cell 边各一个长度 4 的 one-hot**（4 类边），共 8 通道 | 论文 §3；GG 02 §5.3 已同步精确化 |
| LFM 是 EPFL Fua 组（ISP 团队）续作 | **主体 MBZUAI**（Ren Li 现通讯单位 MBZUAI/SUSTech），作者无 Guillard/Fua，仅 Dumery 在 EPFL | arXiv:2601.17740 作者页 |
| LFM 用 OT 条件路径 / rectified flow / CFG | **标准 Lipman FM + DiT**，全文 0 处 classifier-free guidance；补全用 Patel 2025 轨迹控制 | 论文 §3.2 |
| LFM 的可微网格化是 marching squares 可微版 | **ISP 式网格投影**（$v \leftarrow v - d_c\nabla d_c$）+ MeshSDF 式 iso-surface gradient，端点用 UDF 零根 + DBSCAN | 论文 §3.1 |
| GarmentoPIA 让 LLM 写 GarmentCode | **自家尺规原语 DSL**，输出独立 Python 程序；GarmentCode 仅作相关工作被引用 | 论文 §3；正式 PDF |
| GarmentoPIA 与 SewingLDM/DressCode 对比过 | 作者明言**问题不同无法直接比较**，对比对象全是自家消融变体 | 论文 §5 |
| GarmentoPIA 有 arXiv / 开源代码 | **均无**；只有 CVM 2026 会议 PDF + GitHub landing page（"待正式发表后放出"） | iccvm.org/2026、GitHub |

**本篇实验自注**（记录在 `scripts/garment_generation_experiments/README.md`）：实验 2 的 smooth union（quilez smin）有两个方向——`h` 的差分方向与混合方向——**写反任何一个都静默变成 smooth max**：T 恤场整体为空、脚本不报错、表格外观看只是"内部占比 0.000"。与 GG 07 的 DDIM σ 被 `max(0,·)` 钳 0 同属"静默退化"家族：**任何"插值出的隐式场内部占比为 0"或"两个采样器逐位相同"都是 bug 信号**。

---

## 9. 习题

**题 1**：实验 1 里栅格侧的 $\varphi$ 是 4×4 表（16 格），序列侧的 $\gamma$ 是 21 对配对表。设 panel 种类 $P$、朝向类 $K$，写出两种表示需要"背下来"的模式数（$O(K^2)$ vs $O(P^2)$），并证明：当训练数据覆盖所有**含 panel 1 的组合**时，栅格侧 unseen 组合 $(p, q)$（$p, q \neq 1$）仍可精确预测，序列侧不行。
*提示*：$\varphi$ 的样本来自**边的朝向对**——任何 pair 都产生全部 $K^2$ 格的样本；$\gamma_{pq}$ 的样本只来自 pair $(p,q)$ 本身。这是"组合泛化 ⟹ 把交互写成 partner 无关的局部函数"的最小形式。

**题 2**：实验 2(a) 的插值 $s_t = (1-t)s_{\text{shirt}} + t\,s_{\text{doublet}}$ 中，拓扑分裂发生在场出现新零点的时刻。(a) 用鞍点条件（$\nabla s = 0$、Hessian 一个负特征值）说明分裂点的 $t^\ast$ 如何确定；(b) 两 SDF 的**线性插值**不是任何真实版型的 SDF（距离性质 $|\nabla s| = 1$ 被破坏）——这对 LFM 的 eikonal 惩罚 $(\|\nabla d\|^2-1)^2$ 意味着什么？生成模型为什么必须学"潜码 → SDF"而不是"潜码插值"？
*提示*：(b) eikonal 惩罚保证**解码出的**每个场都是合法 SDF——插值发生在潜码空间（两个合法 SDF 的潜码之间），解码后仍是合法场。这正是"latent"的几何意义（GG 08 同款论证）。

**题 3**：LFM 的 iso-surface gradient $\partial v/\partial z = -\nabla d_c \cdot \partial d_c/\partial z$ 与实验 2(b) 的 $dx^\ast/d\theta = -(\partial s/\partial\theta)/(\partial s/\partial x)$ 形式不同。(a) 从隐函数定理推出两者各自的适用条件（后者是一维切片、前者是法向投影）；(b) 证明当 $\nabla d_c$ 单位化时，法向分量 $\partial v/\partial z \cdot \hat n = -\partial d_c/\partial z \cdot \hat n$——即投影式只恢复**法向**导数，切向分量恒为 0。这对"网格顶点沿零面滑动"的优化行为意味着什么？
*提示*：切向滑动不改 SDF 值——损失对切向不敏感，这是 iso-surface 参数化的经典退化方向（GG 04 的 UV 映射正是在解决"切向在哪里"）。

**题 4**：GarmentoPIA 的消融表里，GarmentoPIA−−（仅 DSL）语法错误 5.7% 但数值错误 65.7%；Baseline（直接出 SVG）语法 37.1%、数值 45.8%。解释为什么"引入 DSL"会让语法错误大降、数值错误反升，而"SQL 查重 + Reviewer"让数值大降、语法回升——每一道防线拦住的是哪一层错误（token 序列合法性 / 符号依赖正确性 / 数值计算正确性）？
*提示*：DSL 把"合法性"从 LLM 的生成分布（软）挪到 parser + schema（硬）；但 DSL 的指令更长、依赖链更深，数值计算从"一步出坐标"变成"多步符号推导"——错误率是按**层**分流的，不是按总量下降的。结构化从来不是免费的。

**题 5**（思考）：Work 3 的视频先验蒸馏有两条通路（GG 09 §7）：A = 栅格/隐式场表示做下游、空间对齐注入（利用实验 1 的局部规则机理）；B = 连续潜码 + FM、损失端经可微仿真反传（利用实验 2 的可微链）。SewingLDM baseline 的哪个实测特性（GG 08）对 A 更友好、哪个对 B 是障碍？如果 P3 只够做一条，你会选哪条，用什么最小实验先证伪？
*提示*：对 A 友好的是 n_panels {2:18, 4:2} 的 CFG 压制问题——栅格表示的组合泛化直接扩大合法模式集合；对 B 的障碍是 FSQ 离散码（直通梯度）与两步训练（Step 2 只微调 attn proj，容量不足以承载损失端信号）。证伪 A 的最小实验：把 P0 的 20 seeds 从序列解码换成"预筛合法拓扑 + 栅格头"重训小模型，看 unseen 组合（按 D2 留出）IoU 是否显著高于序列基线；证伪 B 的最小实验：冻结 SewingLDM，仅对 scaling 后的潜码加视频 mesh 对齐损失（绕过 FSQ），看梯度方差是否可训练。

---

*下一篇：[GG 10：单图到仿真就绪——版型 + 扩散先验 + 可微物理（Dress-1-to-3 + DMap）](../GG_10_single_view_simulation_ready/)。*
