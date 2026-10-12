---
title: "GG 11: 无版型对照——3DGS 与 SDS 路线能走多远（GarmentDreamer）"
date: 2026-09-29
math: true
tags:
  - Notes
  - Garment Generation
  - 3D Gaussian Splatting
  - Score Distillation
  - Developability
  - Cloth Simulation
categories:
  - Garment Generation
description: "系列唯一的'反面教材'篇：GarmentDreamer（3DV 2025）不经过缝纫版型，用「UDF 模板扩散 → 3DGS + SDS 引导 → 显式 mesh 变形 + NeTF/VSD 纹理」三阶段管线直接从文本生成可穿的一体式服装。本篇不做立场先行：论文自己列出的版型优势清单（制造无缝衔接、仿真-真实一致、设计师工作流）与'一体式服装对多数图形学下游任务已经够用'的辩护同框呈现，然后用三个数值实验把这场争论里可量化的部分量出来——①可展性：版型↔3D 的桥只在 Gauss 曲率 K=0 时免费（Gauss–Bonnet/测地圆周亏损/最小二乘平展三重验证；非可展残差地板与版型畸变都 ∝a²）；②rest shape 缺失的下游代价：同等 RMS 位移下，可展扰动（褶皱）在'平展→版型→仿真'管线上机器零损耗、在'3D 边长当 rest'管线上落进皱纹亚稳态，非可展扰动（内在拉伸）则两条管线一起受伤；③SDS 机理 toy：guidance λ 的不动点先有偏（偏移源 = 先验混叠残差 |m₀|≈σ_c）、后在 λ*≈1.71 处整个分支消亡——论文的 guidance=100 深入'无吸引子'区，1200 步固定预算是截停而非收敛；同查询预算下 SDS 每查询效率 ≈6.2× 重采样伪 GT。含 10 条常见误记勘误（最终资产既非 3DGS 也非 UDF、SDS 只在中间阶段、template 扩散权重未发布等）"
---

# GG 11：无版型对照——3DGS 与 SDS 路线能走多远

> 📖 服装生成系列第十一篇（前篇：[GG 10：单图到仿真就绪](../GG_10_single_view_simulation_ready/)）。GG 05–10 走的全部是**版型路线**：DSL、观测恢复、生成三连、可微物理——版型作为中间表示被反复加码。本篇是系列唯一的**对照篇**：GarmentDreamer（UCLA + Utah + HKU + Style3D + Amazon，3DV 2025）代表另一条路——**不经过版型**，从文本直接生成可穿、可仿真的一体式（one-piece）3D 服装。它不是靶子：它是这条路线当前完成度最高的作品之一，而且它的作者群里同时有 GG 10 Dress-1-to-3 的核心成员——**同一个团队在两条路线上各发一篇**，这本身就是这个领域最诚实的一组对照实验。本篇全部事实对照 arXiv HTML 全文（v1，唯一版本）、3DV 2025 论文页、官方代码仓库逐文件核实（含 10 条常见误记，见 §8.2）。

---

## 1. 这组论文解决什么问题？

版型路线的卖点是"输出即制造/仿真就绪"，代价是论文自己承认的那句：

> "necessitates a vast amount of paired training data"（§1，批评版型生成路线）

——GarmentCodeData 的 11.5 万件（GG 05）尚且是参数化程序算出来的，真实"文本-版型"配对数据更稀缺。无版型路线的赌注正好相反：**借用现成的 3D 生成工具链**（3DGS、SDS、多视角蒸馏），绕开配对数据，直接产出 3D 服装。GarmentDreamer 给这条路线补上了此前缺失的一块：**服装是独立于身体生成的**。它对 avatar 类方法（衣服长在人身上、不可分离）的批评与 GG 03 一脉相承；它的目标是：

> "full-piece garment generation without relying on 2D sewing patterns"（摘要）

产出的资产是**非水密（non-watertight）的一体式网格**，带领口/袖口/腰口/裤脚开口，可直接穿到 SMPL-X 身体上做动力学仿真（§3.5，4 段 Mixamo 舞蹈演示）。

本篇的组织方式与前面各篇不同：先把 GarmentDreamer 的方法完整讲透（§3–4），然后**让它自己的结论说话**——论文 §5 自列的第一条局限就是一段教科书级的"版型价值清单"，而同一篇论文又辩护说一体式服装"对多数图形学下游任务已经够用"。这个张力不是矛盾，是**任务边界的划分**：本篇三个实验就是把这条边界上可量化的三段——版型↔3D 互换的数学条件、rest shape 缺失的仿真代价、SDS 先验蒸馏的统计性质——各自量出来（§5.2–5.4）。

### 方法流程鸟瞰

GarmentDreamer 三阶段，每阶段换一次表示——每次更换都是"上一阶段的表示顶不住了"（细节在 §3–4）：

| 阶段 | 表示 | 做什么 | 目的 |
|---|---|---|---|
| 1. 模板 | 隐式 UDF + 潜空间扩散 | 服装网格编成 UDF 潜码，按文本-类别条件扩散生成粗模板 | 拓扑自由（开口结构天然可表示），扩散管款式 |
| 2. 引导 | 3DGS + SDS | 模板转高斯初始化，冻结 Stable Diffusion 用 SDS 把文本语义蒸进高斯参数 | 借 2D 先验补几何/语义细节，绕开版型路线的配对数据 |
| 3. 资产 | 显式 mesh + NeTF/VSD 纹理 | 两阶段逐顶点变形对齐 3DGS（开口保持损失防封死），纹理对多视角 RGB 回归再蒸馏精修 | 高斯进不了仿真器——回到可仿真的通用格式 |

一句话：UDF 管拓扑、3DGS 管先验蒸馏、mesh 管落地。版型路线把结构约束放进**表示**（stitch 边天然开口），这条路线把它们全部挪进**损失**（每处开口手工保、纹理细节靠 2D 先验"见过多少版型"）——§5 的三个实验量的就是这笔账。

## 2. 所需背景最小集

- **SDS（Score Distillation Sampling）**：[VG 02](../../video_generation/VG_02_video_3d_priors/) 的 DreamFusion 推导——$\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}\left[w(t)\left(\hat\varepsilon(x_t; y, t) - \varepsilon\right)\frac{\partial x}{\partial\theta}\right]$，冻结 2D 扩散先验对可微 3D 表示做梯度蒸馏。本篇 §5.4 会把它的梯度结构、guidance 代价、查询预算全部拆开。
- **3DGS（3D Gaussian Splatting）**：VG 02——各向异性高斯的显式辐射场，光栅化可微、训练快，但**不是网格**，不能直接进布料仿真器。
- **版型与四种编码**：[GG 02](../GG_02_pattern_representations/)——panel/edge/stitch 的定义；本篇实验 1 的"平展"就是版型的数学本质。
- **UDF 与隐式服装**：[GG 04](../GG_04_self_supervised_draping_isp/)——DrapeNet 的自编码器损失（GarmentDreamer 模板阶段直接复用这类设计）；UDF 允许非水密开口表面。
- **XPBD 与 rest shape**：[GG 01](../GG_01_smpl_cloth_basics/)——布料仿真的 rest length 概念；实验 2 的全部结论都建立在"rest shape 是仿真的因、drape 是果"上。
- **布料测地 vs 3D 欧氏**：[GG 10](../GG_10_single_view_simulation_ready/) 实验 1——折叠是等距映射，UV/参数距离 = 布料内蕴距离；本篇实验 1 的手风琴是同一个构造。

## 3. 表示法详解：一个管线，三次换表示

GarmentDreamer 最有教学价值的地方是：**它没有单一表示**。三个阶段各用一个表示，每次更换都是一次"上一阶段的表示顶不住了"的承认。

**阶段 1：模板 = 隐式 UDF + 潜空间扩散。** 训练数据是 Cloth3D 与 SewFactory（Liu et al. 2023b，即 SewFormer 的数据论文）的服装网格，先经 FEM 准静态仿真预平滑（零静止弯曲 + Neo-Hookean 弹性）去掉仿真噪声。每件网格编码为隐式 **UDF**（unsigned distance field，非水密开口的天然载体），DGCNN 编码器把它压到 **64 维潜码 $\gamma$**，CBN-MLP 解码，MeshUDF 提取网格；训练用 DrapeNet 式自编码器重建损失。然后在 $\gamma$ 上跑 **LDM 式潜空间扩散**，按文本-类别条件生成新模板。

> ⚠️ 这一步常被误记成"用了 SewingLDM"。两者配方同构（潜空间 + 扩散 + 服装域），但 GarmentDreamer 全文**零引用** SewingLDM，潜码是 UDF 的 64 维而非版型的 256×6 FSQ——这是两条独立演化出的相似配方，不是继承（§8.2 误记 3）。

**阶段 2：引导 = 3DGS + SDS。** 模板网格转成 3DGS 初始化（GaussianDreamer 框架），冻结的 Stable Diffusion 通过 **SDS** 梯度把文本语义"蒸"进高斯参数。论文正文未点名扩散模型，但官方代码配置是 **stable-diffusion-2-1-base、guidance_scale 100、1200 步、1024² 渲染、24 视角**——guidance 100 这个数字会在 §5.4 变成主角。

**阶段 3：资产 = 显式 mesh 逐顶点变形 + NeTF 纹理。** 3DGS 语义好但**不是布料**——高斯没有连通表面、没有开口结构、进不了仿真器。最终资产回到模板 mesh：两阶段逐顶点变形（coarse：nvdiffrast 轮廓 mask MSE + 法向一致性 + Laplacian，保持开口；fine：神经延迟着色 RGB L1 + Metric3D 法向 + 开口保持损失），纹理用 **NeTF**（Instant-NGP 哈希编码、仅位置输入）对 3DGS 多视角 RGB 做 L1 回归，再用 **VSD**（ProlificDreamer）微调（SD-2.1-base v_pred，200+600 迭代）补高频细节，最后烘焙到 UV。

**读法**：UDF（拓扑自由）→ 3DGS（可微渲染 + 先验蒸馏的通用接口）→ 显式 mesh（仿真/下游的通用格式）。三次换表示的每一次都是有损的——阶段 3 的 L1 回归把 3DGS 的偏移原样继承（§5.4 实验 (c) 会精确量化"继承偏移、不再付方差"这句话）。

## 4. 方法核心与数学推导

### 4.1 SDS 引导（阶段 2）

冻结扩散先验 $\phi$，对 3DGS 参数 $\theta$（高斯的协方差/不透明度/颜色）做 DreamFusion 式蒸馏：

$$
\nabla_\theta \mathcal{L}_{\text{SDS}}(\phi, g(\theta)) = \mathbb{E}_{t, \varepsilon}\left[ w(t)\left(\hat\varepsilon_\phi(x_t; y, t) - \varepsilon\right)\frac{\partial x}{\partial\theta} \right], \quad x_t = \alpha_t g(\theta) + \sigma_t \varepsilon
$$

其中 $g$ 是可微高斯光栅化渲染，$y$ 是文本条件。代码里 guidance_scale=100——即 CFG 合成去噪器

$$
\hat\varepsilon_{\text{cfg}} = \hat\varepsilon_{\text{uncond}} + w\left(\hat\varepsilon_{\text{cond}} - \hat\varepsilon_{\text{uncond}}\right)
$$

取 $w=100$。VG 02 讲过 CFG 的幂指数几何（$p_{\text{cfg}} \propto p_c^{1+w}/p_u^w$），本篇 §5.4 实验 (b) 从另一个角度算这笔账：**这么大的 $w$ 下，蒸馏动力学还有没有可收敛的目标**。

### 4.2 两阶段 mesh 变形（阶段 3 coarse）

轮廓对齐（论文 Eq. 1）：渲染 mask 与 3DGS 渲染 mask 的 MSE，nvdiffrast 可微光栅化。正则（Eq. 2）：

$$
\mathcal{L}_{\text{coarse}} = \mathcal{L}_{\text{mask}} + \lambda_{NC}\mathcal{L}_{NC} + \lambda_L\mathcal{L}_L
$$

$\mathcal{L}_{NC}$（法向一致性）约束变形后法向与 3DGS 表面法向对齐，$\mathcal{L}_L$（Laplacian）保持局部光滑。**开口在此阶段被显式保护**——顶点只沿允许的方向移动。

### 4.3 开口保持损失（阶段 3 fine，Eqs. 5–7）

这是全文最"服装特化"的一处设计。开口（领口/袖口/腰口）边界顶点 $v$ 带法向 $\tilde n$，相机在 $c$，视线方向 $\tilde v - c$。开口不能被封死，约束写成：

$$
\mathcal{L}_H = \sum_{v \in \text{holes}} \max\left(0,\; \tilde n \cdot (\tilde v - c)\right)^2
$$

即**开口平面不得正对相机**（正对 = 视觉上被压成一条缝 = 后续纹理阶段会把它"补上"变成封口）。梯度里包含一个手写的 boolean 跳过：不满足可见性条件的顶点不回传梯度。消融显示去掉 $\mathcal{L}_H$ 后开口闭合、服装不可穿——这是"无版型路线也逃不开服装结构约束"的第一个证据：它只是把约束从**表示**（版型的 stitch 边天然开口）挪到了**损失**（每处开口都要手工保）。

### 4.4 NeTF 纹理 + VSD（阶段 3 纹理）

NeTF 损失（Eq. 8）：$\mathcal{L}_T = \sum_v \| \text{render}_{\text{NeTF}}(v) - I_{\text{3DGS}}(v) \|_1$——对 3DGS 的多视角 RGB 直接 L1 回归；随后 VSD 微调（ProlificDreamer 的变分得分蒸馏，SD-2.1-base 的 v_pred 版）把 2D 先验里的高频细节（布纹、省道、门襟、卷边、侧缝）补进纹理。消融显示这一步带来了"清晰的褶皱、布料纹理、缝头（seam allowance）、门襟（fly piece）、下摆与侧缝"——这些**全是版型概念**：无版型管线的纹理质量，最终仍由 2D 先验里"见过多少版型细节"决定。

## 5. 实验与关键数字

### 5.1 论文的数字（Table 1，全文唯一表格）

FCSS/CSS 两个分数（36 视角渲染、21 类文本提示；越高越好）：

| 方法 | FCSS | CSS | 耗时 | 备注 |
|---|---|---|---|---|
| Text2Mesh | 0.3396 | 0.2655 | ~20 min | |
| TextDeformer | 0.2367 | 0.1657 | ~35 min | 无纹理 |
| Wonder3D | 0.3402 | 0.2509 | ~4 min | 输入是 DALLE-3 生成的多视角图（非纯文本） |
| **GarmentDreamer** | **0.3413** | **0.2731** | ~15 min | |

三个要点：① **baseline 里没有任何版型路线方法**（SewingLDM/DressCode/GarmentDiffusion/NeuralTailor/SewFormer 一个都没比——grep 全文为零）；② FCSS 上与 Text2Mesh/Wonder3D 的差距在第三位小数，CSS 才拉开——文本对齐容易、语义细节难；③ Wonder3D 的输入不是文本而是 DALLE-3 图像，对照并不完全公平。消融全部定性：去 $\mathcal{L}_H$ 开口封死、去 $\mathcal{L}_N$ 表面污染、去 NeTF+VSD 纹理平掉。**整个评测体系里没有可制造性、没有仿真精度**——这两个维度版型路线的论文（GG 07–10）也大多没测，但版型路线至少**原则上**能测（版型是仿真器的输入格式），无版型路线连原则上都要先解决 §5.2 的桥。

### 5.2 实验 1：可展性——版型↔3D 的桥只在 K=0 时免费

版型 ↔ 3D 的互换不是信息压缩问题，是**微分几何问题**：平布（K=0）到 3D 服装的映射若保持布料不拉伸，必须是**等距映射**，而等距嵌入只对可展曲面（Gauss 曲率 $K\equiv 0$）存在。三个互补的数值验证（`scripts/garment_generation_experiments/gg11_experiments.py` 实验 1）：

**(a) 离散恒等式先立住**：icosphere（642 顶点）角缺陷总和 = 12.566370614359 = $2\pi\chi = 4\pi$，偏差 1.1e-13——Gauss–Bonnet 在离散网格上机器精度成立，后续测量可信。

**(b) Bertrand–Puiseux 测地圆周亏损**：测地半径 $s$ 的圆周长落后平面值多少，$\Delta(s) = 2\pi s - C(s) \to \frac{\pi}{3}K s^3$。球冠（$K=1$）网格上用加权 Dijkstra 测测地距离，$\Delta$ 与解析 $2\pi(s - \sin s)$ 相对偏差从 5.7e-2（s=0.06）收敛到 4.8e-4（s=0.66）；小圆极限序列 $\Delta/((\pi/3)s^3) = 1.128 \to 1.005 \to K=1$。**圆柱（K=0）同一机械量 $|\Delta|/(2\pi s)$ 最大 5.9e-5**——周长不落后，展开即平面。

**(c) 最小二乘平展**（本实验主体）：把 3D 网格的所有边长当约束，找 2D 顶点位置使 $\sum_e (\ell_{2D,e} - \ell_{3D,e})^2$ 最小——这就是"从生成的 3D 服装反推版型"的数学本质。两个对照扰动，**RMS 位移精确对齐**（0.0175）：

- **手风琴褶**（可展，$K=0$）：等距断言 max $|\Delta\ell|$ = 6e-17，面积比 = 1.000000000000。平展残差压到 **max 1.1e-07**（机器零级）——而且**复原的版型与真值平铺逐边一致**（max 相对偏差 4.3e-06，Procrustes 对齐后位置偏差 2.5e-07）。
- **鼓包**（非可展，$K\neq 0$）：面积比 1.00807（解析 $1+0.00814$，$\propto a^2$）。平展残差有**地板**：max 1.1e-04 压不下去；复原版型边长偏差 **max 2.52%、中位 0.24%**。幅度扫描 $a \propto 2\times$：残差 2.87e-05 → 1.15e-04 → 4.58e-04，版型畸变 0.64% → 2.52% → 9.73%——**全部 $\propto a^2$，无阈值，无安全区**。

结论：**版型↔3D 这座桥的过路费由 Gauss 曲率支付**。褶皱（可展扰动）免费往返；任何非可展几何（内在拉伸、球状鼓包）都要按 $K$ 的平方付费，且这笔费**在平展那一刻就交定了**，与后续用什么仿真器无关。无版型生成器没有任何机制保证输出 $K=0$——SDS 蒸馏出的"好看形状"在流形意义下几乎必然非可展。

### 5.3 实验 2：rest shape 缺失的下游代价

版型在仿真里的角色是 **rest shape**（无应力构型）。无版型输出只有 3D 悬垂态，rest 未知——常见 hack 是直接拿 3D 边长当 rest（管线 B）。控制变量实验：真衣 = 平铺布条钉顶边受张悬挂；"生成的服装" = 同等 RMS 位移的两种扰动；对照管线 A（平展→版型→仿真，版型路线的机械化）与管线 B（3D 边长当 rest）：

| 扰动 | 管线 A 偏差 (mean/max, m) | 管线 B 偏差 (mean/max, m) |
|---|---|---|
| 手风琴（可展, RMS z=0.0175） | **4.3e-08 / 2.6e-07** | 3.4e-04 / 4.0e-03 |
| 鼓包 a=0.0182 | 4.3e-04 / 1.4e-03 | 6.3e-04 / 4.4e-03 |
| 鼓包 a=0.0363 | 1.7e-03 / 5.6e-03 | 1.8e-03 / 5.7e-03 |
| 鼓包 a=0.0726 | 6.6e-03 / 2.2e-02 | 6.5e-03 / 2.2e-02 |

四个结构性的读数：

1. **可展扰动 + 版型管线 = 零损耗**。手风琴在管线 A 上机器零偏差——只要扰动是可展的，"平展→重仿真"无损还原真值 drape。**版型路线对褶皱免疫**。
2. **可展扰动 + rest hack = 皱纹亚稳态**。管线 B 在手风琴上的 3.4e-04 残差不是未收敛——速度容差收紧到 1e-5 再跑 2400 步，残差纹丝不动：系统收敛进了**皱纹亚稳态**（等距零能陷阱：rest 长度与真值完全一致（断言 6e-17），但初始构型在错误的势能盆地）。3D 形状**不决定** rest：同一度量下有无穷多平铺方式，挑错一个就锁死在错误的褶皱构型里。
3. **非可展扰动两条管线一起受伤**（$\propto a^2$，每翻倍 4×），管线 A 没有豁免权——版型不是万能药，它只对**可展分量**免费。解析锚验证：底边下垂过剩 $\approx H \cdot \mathbb{E}[(\partial w/\partial y)^2]/2 = 0.0033$，实测 0.0032 ✓。
4. **仿真的因果方向**：版型（因）→ drape（果）。无版型输出把果当资产，要仿真就得反推因——这正是 GG 06 证明过的病态逆问题，只是这次的"观测"是生成器自己编的。

### 5.4 实验 3：SDS 机理——不落地的先验蒸馏

GarmentDreamer 的 guidance_scale=100 来自代码配置。这个数字值得单独做一个实验。设置：冻结先验 = 解析 GMM（$K{=}3$ 模式、$\sigma_c=0.5$，一个"服装类型先验"），Tweedie 公式给出精确最优去噪器，"3D 表示" $\theta\in\mathbb{R}^2$，render = 恒等，conditioning 选中 $\mu_0$。

**(a) 单步梯度的精确分解。** $\lambda=1$ 条件去噪器下，SDS 梯度 $g = \hat\varepsilon_{\text{cond}}(\theta+\sigma\varepsilon) - \varepsilon$ 有闭式：

$$
g = \frac{\sigma}{v}(\theta - \mu_0) - \frac{\sigma_c^2}{v}\,\varepsilon, \qquad v = \sigma_c^2 + \sigma^2
$$

信号 $|\mathbb{E}g| = \sigma r/v$（峰值在 $\sigma = \sigma_c$），噪声 2D-RMS $= \sqrt{2}\,\sigma_c^2/v$，故单步 SNR $= \sigma r/(\sqrt2\sigma_c^2)$。数值验证（20 万样本）：五个 $\sigma$ 档位比值全部落在 0.997–1.005。两个副产品：信号最大与噪声最小**不可兼得**（前者要 $\sigma=\sigma_c$，后者要 $\sigma\to\infty$）；24 视角平均让 SNR 升 $\sqrt{24}\approx4.9\times$。

**(b) guidance 的代价：偏移 → 分支消亡 → 逃逸。** $\lambda>1$ 时 CFG 合成去噪器的期望更新 $h(\theta;\lambda) = \mathbb{E}_{\sigma,\varepsilon}[\hat\varepsilon_{\text{cfg}}]$（Gauss–Hermite 求积，确定性）满足 $h \approx \lambda\bar a_1\theta + (1-\lambda)(m_0 + B\theta)$，其中

$$
m_0 = \mathbb{E}[\hat\varepsilon_{\text{mix}}](\mu_0) = [-0.037, -0.481], \quad |m_0| = 0.483 \approx 0.97\,\sigma_c
$$

是**先验混叠残差**——无条件先验在目标真值处仍被其他模式拉扯。不动点 $x^* = (\lambda-1)m_0/(\lambda\bar a_1 - (\lambda-1)\beta)$：

| λ | 精确根 ‖x*‖ | 线性化 ‖x*‖ |
|---|---|---|
| 1.00 | 0.0000 | 0.0000 |
| 1.10 | 0.0603 | 0.0603 |
| 1.30 | 0.2367 | 0.2339 |
| 1.50 | 0.5933 | 0.5516 |
| 1.70 | 3.0501 | 1.3208 |

偏移方向 $= -m_0$（**远离干扰模式质量**——guidance 放大的是"条件比无条件更相信什么"，在模式重叠区这恰好是推离竞争模式的方向），幅度近似 $\propto(\lambda-1)$，斜率 $|m_0|/\bar a_1 \approx 0.54$。然后是崩塌：$\lambda\to\lambda^*$ 时根范数发散（1.705→3.66、1.712→7.58），**主支在 $\lambda^*\approx1.71$ 处跑向无穷远后消失**（线性化预测 1.98，非线性提前）；远场曲率 $\bar a(\lambda) = \mathbb{E}_\sigma[-(\lambda-1)/\sigma + \lambda\sigma/v]$ 在 $\lambda \gtrsim 1.8$ 后转负（**反恢复**——离目标越远推得越远），数值/解析比值 0.87–1.00。随机动力学（梯度裁剪 ‖g‖≤3）：$\lambda=1$ 终点 $\|\bar\theta-\mu_0\| = 0.0022$（落在机器零根上）；$\lambda=2/5/100$ 全部逃逸（$\lambda=5$ 轨迹：t=500 时已漂到 −18.9）。

**结论**：guidance $\geq 2$ 的 SDS 在这个设置里**没有可收敛的目标**——不动点先是偏的（偏移源 $|m_0|\approx\sigma_c$，即先验把模式混叠的脏水直接灌进结果），后来干脆不存在（分支消亡 + 远场反恢复）。论文的 **guidance=100 + 1200 步固定预算是截停，不是收敛**：前期沿 $-m_0$ 的快速漂移（恰是"风格化传输"效应）之后，剩余迭代在无吸引子的场里游走直到预算耗尽。真实 SD 的先验比 GMM 复杂得多，临界值不会是 1.71——但**机制是普适的**：guidance 放大条件-无条件差值，凡先验有模式混叠（"A-line 裙"与"直筒裙"在潜空间重叠），差值里就混进"推离邻类"的分量，$\lambda$ 大到一定程度这个分量主导动力学。

**(c) 同冻结先验、同查询预算的三个估计器。** 1 查询 = 1 次去噪器前向；伪 GT 样本 = 一条 50 步反扩散链 = 50 查询；SDS 步 = 24 视角 = 24 查询。方差最优步长（Robbins–Monro，$\eta_t = 1/(\bar a_1 t)$）下：

| 预算 B（查询） | SDS 实测 | SDS 理论 $\sqrt{V}/(\bar a_1\sqrt T)$ | 重采样伪 GT | PG 理论 $\sigma_c\sqrt{2\cdot 50/B}$ | 固定伪 GT |
|---|---|---|---|---|---|
| 2,400 | 0.0167 | 0.0166 | 0.1020 | 0.1021 | 0.1439 |
| 9,600 | 0.0080 | 0.0083 | 0.0528 | 0.0510 | 0.1439 |
| **28,800**（= 论文 24×1200） | **0.0048** | 0.0048 | 0.0298 | 0.0295 | 0.1439 |
| 153,600 | 0.0019 | 0.0021 | 0.0126 | 0.0128 | 0.1439 |

三列各自贴着理论线。读数：① **SDS 每查询效率 ≈ 6.2× 重采样伪 GT**（$\sqrt{50}\approx7.07$ 链成本优势 ÷ 1.15 Tweedie 噪声代偿）——SDS 贵在每步都要渲染 24 视角，但省了整条采样链，**方差意义上是划算的**；② **固定伪 GT**（GG 10 MagicMan 式：生成一次、确定性回归）有**预算无关的地板** $\sigma_c\sqrt{2/K} = 0.144$——回归阶段不再付方差，但采样噪声锁死在资产里；③ GarmentDreamer 的架构 = 两者串联：便宜的含噪 SDS 探索（3DGS 阶段）+ 干净的确定性回归出最终资产（mesh 阶段 L1 对 3DGS 渲染）——**回归继承 SDS 的全部偏移（guidance 偏置 + 先验混叠），只是不再付它的方差**。对照 GG 10：Dress-1-to-3 用 MagicMan 伪 GT 做回归目标，继承的是伪 GT 的幻觉（GG 10 实验 2 的 0.172 > 0.120）——两种中间表示，两种偏移，同一个"回归不产生新信息"的定理。

## 6. 论文间关系

- **与 Dress-1-to-3（GG 10）：同一个团队的对照实验。** Tianyi Xie、Yin Yang、Chenfanfu Jiang 同时在两篇作者列表里。Dress-1-to-3 走版型 + 可微物理（TOG 2025），GarmentDreamer 走无版型 3DGS + SDS（3DV 2025）——**两条路线的最优实践出自同一批人**，这不是分裂，是这个团队对"任务边界"的划分：要 simulation-ready 就用版型，要"文本到可穿 3D 资产"且不要求制造，3DGS 路线更快更自由。两篇互相引用与否不重要，重要的是它们**不比同一个指标**。
- **与 SewingLDM（GG 08）：零引用的平行演化。** 模板阶段的"潜空间 + 扩散 + 服装域"配方几乎同构，但潜码（UDF 64 维 vs 版型 256×6 FSQ）、条件（文本-类别 vs T5+体型+草图）、目标（3D 模板 vs 2D 版型）全部不同。同一时期两个社区独立收敛到 LDM 范式（VG 01 §4 的预测），又一次印证"表示决定上限、范式决定工程"。
- **与 SewFactory/SewFormer**：Liu et al. 2023b 作为训练数据源被引用（SewFactory 数据集论文），但全文从未出现 "SewFormer" 这个名字——无版型路线用版型社区的**数据**，不用版型社区的**表示**。
- **与 DressCode（GG 07）**：以 "He et al. 2024" 出现在 related work 里，作为版型路线代表被引用批评（配对数据依赖），**无任何定量对比**。
- **与 GaussianDreamer / DreamFusion / ProlificDreamer**：纯工具关系——3DGS 初始化框架、SDS 损失、VSD 微调。GarmentDreamer 的增量全在"服装特化"上：UDF 模板（开口拓扑）、开口保持损失（§4.3）、NeTF 纹理里的版型细节（§4.4）。
- **系列内定位**：GG 03（人体上的布料，批评对象）→ GG 07–09（版型生成）→ GG 10（观测恢复版型 + 可微物理）→ 本篇（无版型对照）→ GG 12（版图总结）。本篇补上谱系里缺席的最后一格：**3DGS/SDS 生成路线**在服装域的形态。

## 7. 与 Work 3 的连接

Work 3（视频先验蒸馏 → 缝纫版型服装生成，SewingLDM baseline）从本篇拿三样东西：

1. **选型论证**。视频先验蒸馏的中间表示候选：3DGS/mesh（无版型）vs 版型/UV 场（版型路线）。本篇实验 3 给出了无版型中间表示的**结构性风险**：若用 SDS 式蒸馏，guidance 高则无收敛目标（分支消亡 + 远场反恢复），guidance 低则模式混叠偏移 $|m_0|\approx\sigma_c$ 级——视频先验的文本-图像 conditioning 同样有模式重叠（同一段视频对应多件版型）。若用伪 GT 回归（GG 10 路线），继承采样幻觉。**版型/UV 空间的蒸馏落在 range-null 框架里更干净**（GG 10 实验 2 的机理：视频是真观测，不是生成先验——观测子空间是硬约束，不是蒸馏目标）。
2. **rest shape 论证**。Work 3 的输出是版型 → rest 天然正确（实验 2 管线 A：对可展扰动零损耗）。若 Work 3 曾考虑"先生成 3D 服装再平展回版型"的备选路线：实验 1 说明平展的畸变 $\propto a^2$ 且无阈值——**生成器的非可展误差会按平方律渗进版型**，而版型空间的生成（SewingLDM）从头就不产生这个问题。这是"表示选择"压倒"后处理修复"的又一例（GG 09 的主题）。
3. **评测维度**。本篇确认无版型路线的评测体系里没有（也不能有）可制造性与仿真精度维度。Work 3 的自建评测（GG 09 §7 提的 unseen 组合 IoU、DSR 之外）应加上**实验 1 的平展残差**作为生成版型的"可制造性"代理指标——它是纯几何量，不需要仿真器，一条 Dijkstra + 一次 least_squares 就能算。

反方向的诚实：无版型路线的"文本到资产 ~15 min、零配对数据"对 Work 3 也是一种校准——版型路线若在推理速度或数据效率上落后太多，"可制造"这个卖点只对制造业客户成立。论文自己划的界——"Although one-piece 3D clothing is not suitable for manufacturing, it is still sufficient for many graphics downstream tasks"——就是 Work 3 陈述自己价值主张时应该复用的句式：**先把下游任务说清楚，再谈表示**。

## 8. 局限与勘误

### 8.1 论文自列局限（§5，四条）与我们的增注

1. **效率**：分钟级（~15 min/件）而非秒级——SDS 的 1200 步 × 24 视角渲染是硬开销（实验 3 (c) 的预算账：28,800 次前向）。
2. **物理**：仿真演示是手工粗对齐 + 高刚度 PBD 推出身体的 pipeline，不是端到端可仿真资产——论文自己建议 differentiable physics（正是 GG 10 Dress-1-to-3 做的事）。
3. **版型**（原文照录，本篇立论之锚）：
   > "parameterizing the geometry with 2D sewing patterns could offer multiple benefits: they facilitate a seamless connection to the manufacturing process, ensure a closer match between real and simulated clothing, and provide a more intuitive design workflow for traditional fashion designers"
   
   三项好处——制造无缝衔接、仿真-真实一致、设计师工作流——本篇实验 1/2 分别给前两项做了数值注脚（第三项是 HCI 命题，本系列不测）。
4. **光照**：高光/阴影烘进纹理，PBR 是未竟方向。

### 8.2 常见误记（10 条，全部对照原文/代码核实）

1. **最终资产既不是 3DGS 也不是 UDF**——是显式 mesh + UV 纹理；3DGS 只是中间引导表示，UDF 只是模板阶段表示。
2. **SDS 只在 3DGS 阶段**；纹理阶段是 NeTF L1 回归 + VSD 微调，不是 SDS。
3. **不是 SewingLDM**——模板扩散配方相似但零引用、潜码不同。
4. **代码不是端到端**：推理管线已发布（CC BY-NC 4.0），但 template 生成扩散（AutoEncoder_dgcnn + Garment_Diffusion）只有训练代码、**预训练权重未发布**——用户须自备模板 mesh。
5. **arXiv 只有 v1**（2024-05-20），无修订版。
6. **标题没有 "Layered"**——是 "GarmentDreamer: 3DGS Guided Garment Synthesis with Diverse Geometry and Texture Details"。
7. **作者 8 人、前三人共同一作**（Boqian Li, Xuan Li, Ying Jiang），UCLA/Utah/HKU/Style3D/Amazon 混合团队；Feng Gao（Amazon）带"职位无关"声明。
8. **3DV 2025 pp. 1416–1426，DOI 10.1109/3DV66043.2025.00134，无 track（oral/spotlight）概念**。
9. **~15 min/件**（Table 1），GPU 型号未说明。
10. **baseline 只有 3 个**（Text2Mesh/TextDeformer/Wonder3D），零版型路线方法；Garment3DGen 与 WordRobe 仅定性比较。

### 8.3 本篇实验自勘误（写作过程的实现坑，已修）

- SDS 更新符号最初写成 $\theta \leftarrow \theta + \eta g$——DreamFusion 是**最小化**损失，目标瞬间变成斥子（$\lambda=1$ 终点 $5\times10^{12}$）；
- 高 $\lambda$ 下用固定 MC 样本集求不动点——MC 噪声 $\sim\lambda/\sqrt N$ 制造了一堆"伪不动点"，改 Gauss–Hermite 求积后真根才浮出；
- 多起点求根会**跳支**（λ=2 找到的"根"是近零 Jacobian 方向的浅谷），主支必须从小 $\lambda$ 延拓追踪；
- 固定统计量（实验 3(c) 固定伪 GT 地板）200 种子的 RMS 有 ±1.6σ 抽样波动（0.1524 vs 理论 0.1443），加到 20000 种子才贴线；
- 实验 2 的挂钉边从底边换成**顶边**：钉底边时受压直立平衡不稳定，对称初值停在假稳定态、非对称初值翻折，A/B 对比失效——受张悬挂才有唯一平衡；
- 平展求解器的软规范权重须与残差**同量级**（权重 1e3 vs 残差 1e-3 时 trf 信任域停摆，残差卡 5.9e-3 伪影；健全性检查 = 平铺真值本身也压不下去，才能定位是求解器问题）；非可展面有残差地板，容差必须设在地板之上。

## 9. 习题

**题 1**：实验 1(c) 中手风琴的平展残差是 1.1e-07 而非 1e-16。(a) 这 residual 来自哪里（提示：least_squares 的收敛容差 vs 几何的残差地板是两回事）？(b) 为什么它足以证明"可展"而 1.1e-04 足以证明"非可展"？(c) 若把判定阈值设在 1e-5，鼓包幅度多小的时候会被误判为可展？这如何转化为"用平展残差当可制造性指标"时的校准建议？
*提示*：(c) $a=0.0182$ 时残差 2.87e-05，$\propto a^2$ 外推到 $a\approx0.0128$ 就越过 1e-5——指标必须与幅度归一化（如残差/RMS z 的比值），否则小幅度非可展几何会静默通过。

**题 2**：实验 2 管线 B 在手风琴上收敛进皱纹亚稳态，而 rest 长度与真值完全一致（断言 6e-17）。(a) 用势能盆地的语言解释"度量无损 + 平衡态错误"如何同时成立；(b) 为什么延长仿真时间、收紧速度容差都救不了它？(c) 这个现象与 GG 04 的"准静态 + 好模板初值 = 物理损失方法的生存条件"是什么关系？
*提示*：(b) 亚稳态是能量极小值而非驻振荡——它没有出路，除非加扰动跳出盆地。(c) 同一枚硬币的两面：初值/中间表示错在盆地里，物理再正确也只会诚实地收敛到错误的解。

**题 3**：实验 3(b) 的混叠残差 $|m_0| \approx 0.97\sigma_c$。(a) 证明对于两个等权模式 $\mu_0=0, \mu_1=d$、各向同性 $\sigma_c$ 的 GMM，$|m_0| = \frac{1}{2}\frac{d}{v}\sigma\cdot\sigma_c^2 \cdot [\text{ sigmoid 项}]$（写出精确形式），并说明 $d \gg \sigma_c$ 时它指数消失；(b) 由此解释为什么实验 3 前一版用 $d=2.2\sigma_c$ 的"分离模式"测不出 guidance 偏移；(c) 真实文本-图像先验里，"模式分离度"对应什么？这对"guidance 多大安全"的直觉意味着什么？
*提示*：(c) 条件分布的锐度——prompt 越具体（"红色 A 字裙"vs"一件衣服"），模式分离越大，可用的 guidance 上限越高；guidance 安全线是 prompt 的函数，不是全局常数。

**题 4**：实验 3(c) 的效率比 6.2× = $\sqrt{50}$ ÷ 1.15。(a) 推导 Tweedie 噪声代偿因子 $\frac{\sqrt{24}\sqrt{V}}{\bar a_1\sqrt2\sigma_c}$（即 SDS 每查询方差劣势的来源）；(b) 反扩散链长 $L$ 变化时（DDPM 1000 步 vs DDIM 50 步），两个估计器的相对效率如何移动？(c) 固定伪 GT 的地板 $\sigma_c\sqrt{2/K}$ 对 $K$ 的依赖说明：把 24 视角预算改成"8 个样本 × 3 次重采"，地板怎么变？什么时候值得这么换？
*提示*：(b) $\sqrt{L}$ 缩短直接抬高 SDS 的优势——蒸馏的意义随采样器加速而衰减。(c) 地板 $\propto 1/\sqrt K$ 与重采次数无关；$K$ 小地板高，但换来的预算可以投进优化——这是 GG 10 "买来的先验"的预算分配问题。

**题 5**（思考）：综合本篇与 GG 09/10：服装生成的"表示谱系"两端是版型（结构先验、制造就绪、可组合性差）与 3D 显式资产（自由度高、先验零、下游受限）。GarmentDreamer 的三阶段管线实质是"UDF（拓扑）→ 3DGS（先验）→ mesh（下游）"的表示接力。(a) 每次交接损失什么、保住什么？(b) Work 3 的"视频先验 → 版型"若也做成两阶段接力（如 3D 中间表示 → 版型回归），本篇三个实验分别对哪一棒的损失发出警告？(c) 有没有"单表示"方案能同时吃到结构先验与扩散先验？结合 GG 09 的 SDF 版型 + 可微网格化与 GG 11 实验 1 的可展性约束给出你的设计草案。
*提示*：(a) UDF→3DGS 损失开口结构保住形状；3DGS→mesh 损失概率先验保住语义（继承偏移）。(b) 实验 1/2 警告 3D→版型那棒（非可展 $\propto a^2$、rest 歧义），实验 3 警告先验→3D 那棒（guidance/混叠）。(c) 例如"SDF 版型 + 等距约束的生成"——在版型 SDF 的隐式场里加可展性正则（对 $\det$ of metric 的惩罚），表示端直接编码 $K=0$；这正是本篇实验 1 的判据可以直接当损失用的地方。

---

*下一篇：[GG 12：版图总结](../GG_12_series_summary/)——19 篇论文的表示谱系（向量→程序→栅格→隐式→FM）× 任务矩阵（drape/重建/生成/仿真就绪），加上与本系列互链的 VG/RL 视角，收束整个系列。*
