# 服装生成（Garment Generation）论文系列博客规划（活文档）

> 论文来源：用户指定的 19 篇 garment generation 论文（2020–2026，CVPR/SIGGRAPH/TOG/ECCV/NeurIPS/ICCV/IJCAI/3DV/TPAMI/CVM）。
> 系列目标：**把服装生成领域的原理彻底搞懂**——重点服务于 Work 3（视频先验蒸馏 → 缝纫版型服装生成，SewingLDM baseline），版型生成链路写得最深（含完整数学推导）。
> 本文件是系列总体规划，**每发布一篇更新一次**。
> **✅ 系列已于 2026-09-30 完结**：12 篇全部发布（GG 01–12），覆盖全部 19 篇论文、累计 90+ 条勘误、12 个实验脚本（`scripts/garment_generation_experiments/`）。后续增量（Work 3 P3 推进中的新读数、新论文）按需追加新行。
> 调研来源：三路 subagent 已逐篇读取 arXiv abstract + HTML 全文方法部分（2026-09-28），arXiv ID 均已核实。

---

## 一、系列目录（12 篇，按主题分组）

组织逻辑：**背景（01–02）→ 表示演化第一幕：人体上的布料（03–04）→ 版型成为主角（05–09：DSL/观测恢复/生成三连）→ 仿真就绪与无版型对照（10–11）→ 版图总结（12）**。

| # | 标题 | 论文（合并） | 核心问题 | 前置 |
|---|------|-------------|---------|------|
| 01 | SMPL 人体模型与布料仿真基础 | — | 身体怎么参数化（β/θ/LBS）？布料物理（应变/弯曲/碰撞/XPBD/IPC）怎么仿真？自监督 draping 的物理损失长什么样？ | 线性代数、MLP |
| 02 | 缝纫版型表示法专题 | — | panel/edge/stitch 是什么？向量序列（NeuralTailor 式）、参数化程序（GarmentCode 式）、栅格（GarmentImage 式）、隐式场（ISP 式）四种编码各怎么工作？Korosteleva JSON 格式生态 | 01 |
| 03 | 人体上的布料：从子网格到隐式场 | TailorNet (CVPR'20) + SMPLicit (CVPR'21) | 服装 3D 形状怎么参数化为 pose×shape×style 的函数？拓扑变化（T恤→开衫）怎么处理？ | 01 |
| 04 | 自监督 draping 与隐式缝纫版型 | DrapeNet (CVPR'23) + ISP (NeurIPS'23) | 不用仿真 GT，物理损失自监督 draping？隐式版型（2D panel SDF + 缝合标签 + UV 映射）为什么是版型与隐式的最优结合？ | 03 |
| 05 | 版型 DSL 与数据基座 | GarmentCode (SIGA'23) + GarmentCodeData (ECCV'24) | 版型怎么被"编程"（Edge/Panel/Interface/Stitch 的 OOP 设计）？11.5 万 made-to-measure 数据怎么生产？ | 02 |
| 06 | 从观测恢复版型 | NeuralTailor (SIG'22) + SewFormer (SIGA'23) + Garment Recovery (CVPR'24) | 点云/单图 → 版型逆问题：edge 序列解码（LSTM/Transformer）、缝合匹配、双先验拟合 | 02 |
| 07 | 文本到版型 I：自回归与并行扩散 | DressCode (SIG'24) + GarmentDiffusion (IJCAI'25) | 版型怎么 token 化（量化/边 token）？SewingGPT 自回归为何慢（串行依赖墙，非 FLOPs 墙；18k/30k token 两文口径不一致）？DiT 并行去噪怎么快 100×？50 步跨步 DDPM 够不够？ | 02, VG01 |
| 08 | 文本到版型 II：SewingLDM 深度精读 | SewingLDM (ICCV'25，上交大 MoE AI Lab, arXiv:2412.14453)【Work 3 baseline】 | 29 维边向量（19 连续+10 离散，代码权威）、256×6 FSQ 潜空间（codebook 0%→100% Goldilocks 的计数机理）、两步 IDDPM-DiT 训练、T5+体型+草图条件注入（统计归一化残差，非 cross-attn）——逐模块推导 + 我们 P0 复现的实际经验（6.5s/件、panel_order None bug、n_panels {2:18,4:2} 的 CFG 机理） | 07 |
| 09 | 版型生成的表示革新 | GarmentImage (SIG'25) + Latent Flow Matching (SIG'26) + GarmentoPIA (CVM'26) | 栅格化版型（unseen 拓扑 IoU 翻倍）、隐式场 + latent flow matching（可微网格化）、LLM agent 程序合成——表示的三条新路线 | 07, 08 |
| 10 | 单图到仿真就绪：版型 + 扩散先验 + 可微物理 | Dress-1-to-3 (SIG'25) + DMap 静态/动态 (SIG'25/TPAMI'26) | "simulation-ready" 怎么达成：版型初估 + 多视角扩散伪 GT + 可微 CIPC 联合优化；pattern coordinates 的 UV↔3D 对应；时序扩展 | 06, VG01 |
| 11 | 无版型对照：3DGS 与 SDS 路线 | GarmentDreamer (3DV'25) | 无版型 mesh/UDF 生成能走多远？其"无版型"局限如何反证版型路线的价值（可制造性/可仿真性） | VG02 |
| 12 | 版图总结：表示谱系与未来 | 全部 19 篇 | 表示谱系（向量→程序→栅格→隐式→FM）×任务矩阵（drape/重建/生成/仿真就绪）+ 与 Work 3 视频先验蒸馏方案的连接 | 01–11 |

**总计 12 篇**，覆盖全部 19 篇论文。EPFL-Fua 系（DrapeNet→ISP→Garment Recovery→DMap×2）与 Korosteleva 系（2021 版型数据集→NeuralTailor→GarmentCode→GarmentCodeData）两条作者主线在分篇中保持连贯；注意 SewFormer 是 Sea AI Lab（Liu et al., arXiv:2311.04218），不属于 Korosteleva 系，但边表示与 NeuralTailor 同构（复用 BasicPattern 代码）。

---

## 二、依赖关系树

```
背景 01 (SMPL/布料仿真)
   ↓
背景 02 (版型表示法) ──────────────────────┐
   ↓              ↓                       │
03 TailorNet+    05 GarmentCode+          │
   SMPLicit         GarmentCodeData       │
   ↓              ↓        ↓              │
04 DrapeNet+   06 NeuralTailor+   07 DressCode+     ←─ VG01 (扩散基础)
   ISP            SewFormer+         GarmentDiffusion
   ↓              GarmentRecovery        ↓
   └────→ 10 Dress-1-to-3 +        08 SewingLDM ★（最深）
          DMap（版型+扩散+可微物理）      ↓
   ↓                                    09 GarmentImage +
11 GarmentDreamer                        LatentFM + GarmentoPIA
（无版型对照，需 VG02 SDS）               ↓
        └──────── 12 版图总结 ←── 全部 ──┘
```

- **02 是全系列地基**：四种版型编码在后续每篇都被引用；
- **07→08 是版型生成链路主干**（AR → 并行扩散 → latent diffusion），数学推导最深；
- **08 直接服务 Work 3**：SewingLDM 精读会结合 P0 复现经验（6.5s/件推理、序列化 `panel_order` None padding bug、同 prompt 结构多样性低等实测观察）；
- **10 与 Work 3 的"simulation-ready"目标最近**：Dress-1-to-3 的可微物理联合优化 + DMap 的 UV↔3D 对应，都是视频先验蒸馏方案的可借鉴组件；
- **11 是反面教材篇**：GarmentDreamer 的无版型路线及其自列局限，用来论证"为什么要版型"。

---

## 三、缝纫版型表示谱系（本系列核心叙事线）

```
显式向量序列          参数化程序            栅格图像           隐式场
NeuralTailor'22  →   GarmentCode'23   |   GarmentImage'25   ISP'23 / LatentFM'26
  (edge 序列+          (OOP DSL,          (16×16×34,          (2D SDF + stitch
   stitch 分类)         Interface 配对)     通道=边类型+         标签 / SDF+UDF+
       ↓               GarmentCodeData     形变矩阵)            可微网格化)
       │                    ↓                   │                  │
  SewFormer'23       (11.5万数据基座)            │            DMap'25/'26
  DressCode'24                                          (pattern coordinates
  (离散 token AR)                                             UV↔3D)
  GarmentDiffusion'25
  (边 token 并行)
  SewingLDM'25 ★
  (29维向量→潜空间)
```

**谱系的三个断层**：①向量→程序（从"预测几何"到"生成规则"）；②序列→栅格/隐式（解决拓扑变化时的空间不连续）；③DDPM→flow matching（生成模型的代际切换，与 VG01 §4 呼应）。

---

## 四、易混淆概念辨析地图（随篇更新）

| 概念对 | 辨析要点 | 出现于 |
|--------|---------|--------|
| 准静态 vs 动态物理损失 | PBNS/DrapeNet/ISP 丢惯性项（pose→形状映射唯一、稳但抓不住摆动）；SNUG 加 $\mathcal{L}_{\text{inertia}}$（backward-Euler-即-损失的完整形式）。准静态 + 好模板初值 = 物理损失方法的生存条件（GG 04 实验：1mm 噪声即永久抬高能量） | 01, 04 |
| 显式版型 vs 隐式版型 | 曲线列表可直接渲染/制造 vs SDF/UDF 可微、拓扑平滑变化 | 02, 04, 09 |
| panel 边的四种编码 | NeuralTailor/DressCode/SewFormer 4 维 (vₓ,v_y,cₓ,c_y)（位移+二次 Bezier 控制点，cₓ 移峰位、c_y 曲深）/ GarmentDiffusion 13 维边 token（e3⊕c3⊕a3⊕s3⊕f1，BRepGen 式）/ SewingLDM 29 维（2+2+4+3+3 几何 +4+3 位姿 +3+2+1+3+2 flag，19 连续+10 离散）——已按原文核对 | 02, 06, 07, 08 |
| stitch tag vs 缝合 flag vs Interface | 可学 3D 标签匹配 / 二值分类 / DSL 级组件配对 | 02, 05, 06 |
| 自回归生成 vs 扩散生成 vs flow matching | token 级条件依赖（慢、序列长）/ 并行去噪（快）/ 直线路径 ODE（更快更稳） | 07, 08, 09 |
| 串行依赖墙 vs FLOPs 墙 | AR 慢在"每步只喂 1 个 token 的访存受限串行链"，不在浮点量——DiT 并行去噪浮点量反而更多（GG 07 实验 1：等 FLOPs 串行链 vs 批量并行 10×；A10 屋顶线 58–337×）。加速 = 序列内并行 × 步数缩减（1000→50）两个正交因子 | 07 |
| 离散量化 vs 连续向量（跨 token 结构约束） | 逐分量量化的损失不在单坐标 RMS（h/√12）而在跨 token 全局约束：闭合 Σv=0 与缝合长度一致性按随机游走累积（~h√m/√12，h=20mm 时 19/20 版型闭合超差）——两代方法评测指标里都没有这一项。DDIM σ 比值方向写反会被 max(0,·) 静默钳 0，采样器无声退化 η=0（"两个采样器逐位相同 = bug 信号"） | 07, 08 |
| DDPM 跨步采样 vs DDIM (η=0) | 精确 score 下两者 50 步都在噪声地板附近、都可用——"DDIM 会模式漂移"是 toy 实现 VE/VP score 混淆的伪象（GG 07 §5.3 自勘误）。真机理：score 误差对确定性采样器的破坏远大于随机采样器（同一偏差 DDIM 恶化 45×、跨步 DDPM 几乎无感——噪声重注入能量抵消 drift，ODE 无校正）。退化顺序"先模内 σ 爆、后模式权重漂"可诊断步数墙。GarmentDiffusion 实践 = 跨步 DDPM 50 步 | 07, 08 |
| 有限标量量化 vs VQ-VAE 码本 | tanh 连续量化 5 级网格（无码本查找）vs 离散码本 | 08 |
| FSQ 潜空间的 Goldilocks | 大潜空间两侧都亏：重建侧每加一维 = 加一个 ±h/2 量化噪声通道（饱和甚至回升）；先验侧码字复用崩塌（256×32/12 → codebook 利用率 0% 无法生成——不是码字没用，是分布无重复结构可学）。选点 = 重建饱和 ∩ 复用尚存 ≈ 码本大小匹配数据有效自由度（5⁶=15,625 ≈ 19,683 样本） | 08 |
| CFG 的单旋钮几何衰减 | p_cfg ∝ p_c^{1+w}/p_u^w：模式质量按 π_c·(π_c/π_u)^w 衰减，压非法模式必压少数合法模式（w=2 时非法 <0.3% 且合法熵 0.96→0.50）——多样性只能靠提高 π_c/π_u 分离度（更好的条件模型），guidance 调不出来。P0 实测 n_panels {2:18, 4:2} 的机理 | 08 |
| 局部共享规则 vs 全局配对表 | 同一个 pairwise 缝合交互：栅格表示把它空间化为 φ(朝向对)——4×4 表、任何 seen pair 都提供样本、unseen 组合 = 同一张表的新索引（精确外推）；序列表示把它摊成 γ_pq 配对表——O(P²) 个模式、unseen pair 的 γ 原理上不可预测（误差 ≈ \|γ\|）。组合泛化 ⟹ 交互写成 partner 无关的局部函数（GG 09 实验 1：0.0000 vs 0.5796，还原 GarmentImage 0.8127 vs 0.3131/0.4260） | 09 |
| valid-by-construction vs 学习分布 | DSL 求值器对任意程序输出闭合版型（有效性是类型系统性质，闭合残差 ~1e-16），泛化性来自求值规则；学习模型只在分布内有效，115k 样本对玩具 depth≤4 程序空间覆盖率仅 33%（真实组件库更低）。学习路线有分布内平滑插值（FSQ），程序路线有分布外组合泛化——互补不是竞争 | 05, 09 |
| 拓扑的连续几何 vs 离散决策 | SDF 隐式场里 1 panel→2 panel 的拓扑分裂经鞍点连续发生（插值路径相邻步 IoU≥0.978，无离散决策）；序列表示同样 morph 需 presence flag 跳变（中间态不可解码）。零水平集点对参数可微（隐函数定理，LFM 的 iso-surface gradient ∂v/∂z 机理）——"生成→网格→仿真损失"端到端可微的基础。但注意 panel **数**仍是离散的（token 数决定），可微的是固定拓扑下的几何 | 04, 09 |
| 表示论文 vs 生成论文 | GarmentImage 是表示论文：朴素组件（CNN-VAE/SDS/U-Net）+ 无消融，证明的是编码的组合泛化性质而非生成质量——"表示的证明与生成的证明之间有空白"；LFM 亦然（与 SewingLDM 零定量对比，证据来自 refitting/补全下游）。引用三路线数字时不可与生成主干拼表 | 09 |
| 论文口径 vs 代码口径 | SewingLDM 论文 Eq.1 加总只有 28 维（R 写 ℝ³，代码四元数 4 维）、Sec 6 线型编码表撞码、Eq.4 σ 写反、train.sh 拼写笔误——baseline 论文不读到代码级等于没读。另：SewingLDM 全文未引用 GarmentDiffusion（同期互不引用），两文数字不可拼表 | 08 |
| 物理仿真 GT 监督 vs 物理损失自监督 | 前者需数据（TailorNet 55k 帧）、后者零 GT 但准静态（PBNS/SNUG/DrapeNet/ISP） | 01, 03, 04 |
| PBS vs XPBD vs CIPC vs Warp | 传统隐式求解 / 位置动力学（GarmentCodeData 用）/ codimensional IPC 无穿透（Dress-1-to-3 用） | 01, 05, 10 |
| drape 预测 vs 版型重建 vs 版型生成 | 给服装+身体→变形 / 给观测→版型 / 给条件→版型 | 03/06/07 |
| made-to-measure vs free-size | 参数依赖身体测量（跨体型有效）vs 平均版型；GarmentCode 约束的真实机制是归一化参数线性插值重映射（参数空间即合法空间），不是 clamp | 05 |
| 数据集官方统计 vs 下游预处理上限 | GarmentCodeData 论文无 panel/边数统计、无 benchmark、正文无 split；"39 panel × 37 edge" 是 SewingLDM padding 上限（39 = 语义 panel 类别数），常被记反成 37×39 | 05, 08 |
| 学出来的先验 vs 写下来的先验 vs 买来的先验 | SewFormer 把先验烧进权重（百万渲染对端到端，推理快、不可解释）/ Garment Recovery 保留显式优化（ISP + 变形网络 + 测试时物理项，simulation-ready、慢且依赖初始化）。Dress-1-to-3 走完来回：初估用前者、精修用后者；两条路线的公共新形态是**买来的先验**——直接调用预训练扩散（MagicMan 多视角伪 GT / DMap 自训练三套扩散）补单视角信息 | 06, 10 |
| UV 空间 vs pattern coordinates | 规则网格载体（ISP 的 CNN/碰撞天然载体）vs 像素↔UV↔3D 三方对应（DMap 的对齐机制） | 04, 10 |
| 内蕴度量 vs 3D 欧氏跨层混淆 | 布料的"邻近"由测地距离定义；褶皱/自遮挡让 3D 欧氏感受野 80% 的点是隔层远邻（最远 5.7×r）、带内 13× 全谱混淆——在 pattern/UV 坐标里卷积的局部性 = 布料的局部性（折叠是等距映射，UV 距离 = 布料测地，图测地独立验证无空气捷径） | 04, 10 |
| 低通先验 vs range-null 投影 | 平滑/物理先验按**频率**分配信任（低频偏差 98% 放行、高频无论真假一起压——"滤的是频率不是对错"，幻觉比无知糟 0.172>0.120）；DDPM_p 式投影按**是否被观测**分配信任（观测子空间硬约束、零空间走先验）——观测掩码与频段不对齐时（视频时空补丁）投影是唯一正确结构 | 10 |
| 忽略时间维的频域代价 | 准静态/逐帧独立的误差 = \|H(ω)−1\| 随频率单调增（0.6ω₀ 时 ~58%）；速度+加速度一阶修正由数据拟合自动恢复动力学系数（α→−c/k, −m/k）覆盖常规频段；共振段泰勒展开发散、修正反而放大——学习映射必有失准频段，"simulation-ready"（版型可进仿真器兜底）的存在理由 | 04, 10 |
| simulation-ready vs 视觉可用 | 可分离/可仿真/可制造 vs 看起来像就行 | 10, 11, 12 |
| 可展（K=0）扰动 vs 非可展扰动 | 版型↔3D 的桥由 Gauss 曲率收费：等距映射（褶皱/折叠）平展往返机器零、复原版型逐边还原；非可展几何（内在拉伸）残差地板 + 版型畸变 ∝a² 无阈值无安全区。生成器无机制保证 K=0——SDS "好看形状"在流形意义下几乎必然非可展。平展残差可当无仿真器的可制造性代理指标（须与幅度归一化） | 11 |
| 皱纹亚稳态 vs rest-shape 歧义 | 拿 3D 边长当 rest 的 hack：度量无损（rest=真值）却收敛进错误的褶皱盆地（等距零能陷阱，延长仿真/收紧容差救不了）——3D 悬垂态不决定 rest，同一度量下无穷多平铺。版型在仿真里的角色是 rest（因），drape 是果；无版型输出把果当资产 | 11 |
| 先验蒸馏 vs 先验回归（中间表示的偏移账本） | SDS：每查询方差便宜（省反扩散链，同预算 6.2× 伪 GT）但 guidance 有代价——不动点先偏（偏移源 = 先验混叠残差 \|m₀\|≈σ_c，guidance 放大"推离邻类"分量）、后亡（主支分岔消亡 + 远场反恢复，无吸引子 regime 里固定步数 = 截停）；伪 GT 回归：干净无方差但继承采样幻觉（幻觉比无知糟）。回归不产生新信息——两种中间表示，两种偏移。guidance 安全线是 prompt 锐度的函数，不是全局常数 | 10, 11 |
| 结构损伤面：线性约束 vs 非线性泛函 | 表示对约束的破坏是**选择性的**：逐分量量化直接破坏线性约束（闭合 Σv=0 按随机游走累积 ~h√m/√12），而 PCA 线性子空间对线性约束**精确保持**（训练数据在约束子空间内 → μ 与每个主成分都在子空间内 → 任意量化系数组合仍闭合，机器零），被 FSQ 破坏的是非线性泛函（缝长差 0.015→0.246cm）——两种表示的损伤面互补，选表示前先问"我的应用在乎哪个约束"。神经网络解码器放弃"免费保持"换非线性表达力，但可输出后投影回约束子空间两头都要 | 12 |
| 同网格精度 vs 同比特效率 | 表示对照必须双口径：SDF 对直线边沿任意方向仿射、网格线上线性插值零交叉精确落在边上 → 同网格下亚网格精度赢占用栅格 5–6×；但每格 8 bit vs 1 bit，同比特下 SDF 反输（23² SDF 6.00mm vs 64² 占用 1.04mm）。且粗网格（特征<Δx）SDF 反而略差——零交叉插值被小特征带偏，占用的多数表决更稳。"表示选择"里混着"量化深度选择"（占用 = SDF 的 1 bit 符号通道），实验要分开记账 | 12 |
| guidance 病理类型由条件锐度决定 | 完美（单模式）conditioning：分支在 λ*≈1.70 消亡 + 远场反恢复（ā(100)=−123<0）——"没有可收敛的目标"；混合（现实）conditioning：分支永不消亡、远场恒恢复（系数→E[1/σ]>0），代价换成**有界但单调增长的偏置**（w=1→100 偏 0.10→2.74，方向远离干扰模式）——无声的系统性偏移比显式发散更危险。统一结算：生成侧熵崩塌先发生（w≈3）、蒸馏侧偏置永不消亡，安全窗 w≲2–3，唯一拉宽途径 = 提高 π_c 分离度（蒸馏分离度，不是蒸馏均值） | 08, 11, 12 |

---

## 五、单篇统一模板（8 节，与 VG 系列一致）

1. **这组论文解决什么问题**：动机、前人缺陷
2. **所需背景最小集**：引用 01/02 或前篇链接
3. **表示法详解**：本组论文的版型/服装编码逐字段拆解（本系列最重的维度）
4. **方法核心与数学推导**：架构、损失、训练目标完整推导（07–09 最深）
5. **实验与关键数字**：只写调研/全文核对过的数字
6. **论文间关系**：组内对比 + 谱系定位
7. **与 Work 3 的连接**：可借鉴组件、baseline 差异点（08/10 篇重点）
8. **局限与思考**：失败模式、开放问题

写作约定：

- 术语首次出现：中文（English）；
- `math: true` LaTeX；
- **每篇写作前必须重读该论文 arXiv HTML 全文核对细节**（调研笔记只是骨架，正文数字以全文为准）；
- 发布路径：`content/post/garment_generation/GG_<NN>_<slug>/index.md`；
- 每篇头部互链前篇；每发布一篇更新本文件进度表。

---

## 六、与既有系列的互链分配

| 本系列 | 互链目标 | 连接点 |
|--------|---------|--------|
| 07–09（生成链路） | VG 01（DDPM/FM/CFG） | DiT、IDDPM、flow matching 的扩散基础 |
| 08 SewingLDM | VG 01 §4 | latent diffusion = Stable Diffusion 范式在版型域的实例 |
| 10 Dress-1-to-3 | VG 02 §4（SDS） | 扩散先验蒸馏到 3D 的服装版 |
| 11 GarmentDreamer | VG 02（3DGS + SDS） | 无版型对照 |
| 10/12 | RL 系列（按需） | 可微物理优化 = 基于梯度的决策；Dress-1-to-3 的多轮仿真优化 |
| 01 | learning_to_dress 系列 | 布料仿真的落地实例（穿衣智能体复现项目） |
| 12 | Work 3 记忆 | 视频先验蒸馏方案定位 |

---

## 七、进度追踪

| # | 状态 | 发布路径 |
|---|------|---------|
| 01 | ✅ 已发布 | `content/post/garment_generation/GG_01_smpl_cloth_basics/index.md`（2026-09-29 增补：§2 按"三层结构"（统计形状模型/运动学骨架/蒙皮算子）重写为小白友好精讲——模板稠密对应、PCA 体型基直观、关节回归与 24=1+23 算术、R−I 姿态特征的两个刻意设计、运动树/inverse bind/矩阵平均读法、twist 与 bend 同源推导（cos(φ/2)、180° 行列式 0、弦 vs 圆弧）、DQS 与 SMPL-LBS 取舍（B_P = LBS 专属逆补偿，官方 FAQ）、forward 计算图 + tensor 清单 + 单公式自检、SMPL 非物理边界（不知道力/skinning≠皮肤/服装-LBS=壳/正确角色=碰撞体条件）；Rodrigues 原论文 errata 提示；习题 1–6 新增 R−I 特征设计题） |
| 02 | ✅ 已发布 | `content/post/garment_generation/GG_02_pattern_representations/index.md`（04 发布时勘正 §5.4：ISP 的物理在 drape 网络而非版型网络） |
| 03 | ✅ 已发布 | `content/post/garment_generation/GG_03_body_conditioned_garments/index.md` |
| 04 | ✅ 已发布 | `content/post/garment_generation/GG_04_self_supervised_draping_isp/index.md` |
| 05 | ✅ 已发布 | `content/post/garment_generation/GG_05_garmentcode_data/index.md`（零神经网络篇：26 项测量分类/参数重映射机制/GarmentCodeData 三个"没有"均按代码与 v2 发布物核实；GG 02 §5.2 同步勘正 26 项分类与 clamp 说法） |
| 06 | ✅ 已发布 | `content/post/garment_generation/GG_06_pattern_recovery/index.md`（病态性三实验：敏感度谱 0.05–0.36mm/7.7×、类内 14–33 > 类间 2–5、内域盲区 Chamfer 精确 0；两大标题勘误：NeuralTailor 与 Garment Recovery 真名均与流行说法不同——后者从不是 NeRF/多视角方法，是 "Garment Recovery with Shape and Deformation Priors" 单图双先验；SewFormer 无旋转不变性贡献、指标无 IoU、训练数据是自建 SewFactory） |
| 07 | ✅ 已发布 | `content/post/garment_generation/GG_07_text_to_pattern_ar_diffusion/index.md`（三实验：串行墙 10×/屋顶线 58–337×、量化破坏闭合 19/20·吃势 17/20、50 步消融；勘误 12 条：DressCode 作者 Kai He 非 Korosteleva/CLIP 非 LLM/训练数据是 2021 数据集非 GarmentCodeData、18,135 vs "over 30k" 两文不一致、GarmentDiffusion 仓库无代码/100× 无耗时表。**发布当日自勘误**：初版实验 3 误用 VE 约定 score（VP 均值未缩放），"DDIM 50 模式漂移、差 7×" 是伪结论——正确 VP score 下 DDIM 50 在噪声地板（4.21e-04）；真结论是 score 误差对确定性采样器破坏远大于随机采样器（同一偏差：DDIM 恶化 45×、跨步 DDPM 几乎无感）。另两个实现坑：DDIM σ 比值方向写反被 max(0,·) 钳 0。GG 02 §5.1 与 GG 05 §1/§6 已同步勘正 DressCode 数据来源） |
| 08 | ✅ 已发布 | `content/post/garment_generation/GG_08_sewingldm_deep_dive/index.md`（Work 3 P0 baseline 精读：调研 agent 穷尽 arXiv v1/v2 + 官方代码逐文件 + HF 权重 + ICCV openaccess；三实验：FSQ Goldilocks（大潜空间两侧都亏：k=32 重建回升且码字零复现）、CFG 模式压制（解析+采样双验证，对接 P0 {2:18,4:2}）、潜压缩计算预算（1443→256 token，FLOPs 同量级但 DiT 大 4.5×）。论文自身 7 处错误勘误（Eq.1 28 维、Sec 6 编码撞码、Eq.4 σ 反、直通梯度未提、train_fisrt 拼写、supp panel 表 29 重复、125,865 vs 120,000 口径）；我们笔记 3 处修正（T5 非 CLIP、29 维算式、39 panel×37 边方向）；GG 02 §3.3/§5.1 已同步勘正） |
| 09 | ✅ 已发布 | `content/post/garment_generation/GG_09_representation_shifts/index.md`（三路线全核实：GarmentImage = **表示论文**非生成论文（CNN-VAE/SDS/三级联 U-Net，无消融），边类型 8 通道 = bottom/left 各一个 4 类 one-hot；LFM = **MBZUAI 非 EPFL Fua 组**（Ren Li 通讯），标准 Lipman FM + DiT、全文无 CFG、可微网格化是 ISP 式投影 + MeshSDF iso-surface gradient 非 marching squares；GarmentoPIA = **自家尺规 DSL 非 GarmentCode**、KAIST、CVM 2026 首尔、无 arXiv 无代码、对比对象全是自家消融。三实验：栅格局部规则 φ(4×4) vs 序列配对表 γ(O(P²)) 的组合泛化（unseen 误差 0.0000 vs 0.5796≈|γ|，还原 0.8127 vs 0.3131/0.4260）、SDF 拓扑分裂鞍点连续性（相邻步 IoU≥0.978）+ 隐函数定理可微网格化（vs 有限差分 9.96e-13）、DSL valid-by-construction（闭合残差 8.88e-16 vs 量化 19/20 超差）+ 组合空间计数（115k 样本对玩具 depth≤4 空间覆盖率仅 33%）。GarmentoPIA 消融的"每道防线各拦一类错误"读法；LFM 与 SewingLDM 零定量对比（不做跨表比较）；Work 3 §7：视频先验的空间栅格特征与栅格/隐式场表示同构（蒸馏不经序列化）、损失端蒸馏需绕 FSQ、自建评测 +2（unseen 组合 IoU、DSR）、模板先验=分离度。8 条勘误 + 本篇自注（smin 双方向任一写反静默变 smooth max）。GG 02 §5.3 边类型通道已同步精确化） |
| 10 | ✅ 已发布 | `content/post/garment_generation/GG_10_single_view_simulation_ready/index.md`（两篇全核实：Dress-1-to-3 = TOG 44(4) 非 Conf. track、多视角扩散是 **MagicMan 非 Zero123++**、可微物理 = CIPC + Warp AD + adjoint（与本构无关，无摩擦保路径无关）、优化静态悬垂、无 DSR 类指标、官方代码未发布；DMap 静态版**论文正文无 "DMap" 名字**（名字来自仓库，TPAMI 版才命名）、任务是图像/视频→重建非 pose→变形、DDPM_p = range-null 投影（观测硬约束 + 零空间走先验）、动态版时空解耦（空间权重冻结）+ 测试时引导、两版数据不同**不可拼表**、74.1GB 显存。**两篇与 GG 07–09 生成系零定量对比**（GaRec 是唯一叙述锚点但不能拼表）。三实验：UV≈布料测地（手风琴折叠 + 图测地独立验证等距，偏差全为正 = 无空气捷径）vs 3D 欧氏跨层混淆（r 感受野内 80% 点布料距离超 r、最远 5.7×r、带内 13× 全谱）；单视角零空间 + 伪 GT 误差结构 + 先验低通（幻觉比无知糟 0.172>0.120；低频偏差 98% 放行 vs 高频同幅压到 5%——滤的是频率不是对错，论文 Limitation 3 过度平滑即实证）；忽略时间维的频域代价（准静态 |H−1| 单调增、速度+加速度修正自动恢复 α→−c/k·−m/k、共振段泰勒发散无解→simulation-ready 兜底的必要性）。12 条勘误；Work 3 §7：视频先验 = 观测子空间真约束（DDPM_p 式 range-null 损失）、pattern coordinates 是蒸馏天然载体（少一道 3D→UV 有损变换）、adjoint 可微物理只够离线蒸馏用） |
| 11 | ✅ 已发布 | `content/post/garment_generation/GG_11_pattern_free_garmentdreamer/index.md`（对照篇全部事实核实：标题无 "Layered"、8 作者 3 共同一作、3DV 2025 pp.1416–1426、arXiv v1 唯一版本、SD-2.1-base/guidance 100/1200 步/24 视角来自代码配置、template 扩散权重未发布须自备 mesh、~15 min/件、baseline 仅 3 个且零版型路线、与 SewingLDM 零引用。三实验：①可展性三重验证（Gauss–Bonnet 1.1e-13、Bertrand–Puiseux 球/柱对照、最小二乘平展：手风琴机器零 + 复原版型 4.3e-06 vs 鼓包残差地板 + 版型畸变 2.52% ∝a²）——版型↔3D 的桥只在 K=0 免费；②等 RMS 扰动下游代价（可展+管线 A 零损耗、可展+rest hack 皱纹亚稳态（速度 1.3e-7 残差不动 = 等距零能陷阱）、非可展两管线齐伤 ∝a²、解析锚 0.0033 vs 0.0032）；③SDS toy（单步梯度精确分解 SNR=σr/(√2σ_c²)、guidance 不动点先偏后亡：混叠残差 |m₀|≈0.97σ_c、线性化 x*∝(λ−1)、主支 λ*≈1.71 消亡 + 远场反恢复 → guidance 100 是截停非收敛；同预算 SDS 6.2× 重采样伪 GT、固定伪 GT 地板 σ_c√(2/K) 预算无关）。§6 团队对照：Tianyi Xie/Yin Yang/Chenfanfu Jiang 同时在 Dress-1-to-3 与本篇——同团队两条路线划任务边界。§7 Work 3：版型/UV 蒸馏落在 range-null 框架、平展残差当可制造性代理指标、版型空间生成从头不产生非可展误差。§8.3 自勘误 6 条：SDS 符号反、MC 伪根→GH 求积、多起点跳支→延拓、200 种子 ±1.6σ、钉边选择、软权重同量级） |
| 12 | ✅ 已发布（**系列完结** 🎉 2026-09-30） | `content/post/garment_generation/GG_12_series_summary/index.md`（终篇收束，无新论文：①19 篇落 5×4 表示×任务矩阵，占有率 10/20=50%，空白格有结构——仿真就绪列最稀 2/19、drape 全在 3D/无版型桶、栅格行孤篇、版型生成列最密 8 篇，每个空白格配本系列量化过的屏障；②同一版型族五路线统一对照：程序 73bit→0.5mm 闭合缝合零损伤（误差=δ×∂版型/∂参数，系数 0.125–1.0）/ 量化序列闭合≈h 且 h=20mm 时量化吞最短边→闭合跳 1.4h 非线性爆炸 / 占用 vs SDF 同网格 5–6×、同比特占用反胜（23² SDF 6.00mm vs 64² 占用 1.04mm）、粗网格 SDF 略差（小特征<Δx 零交叉插值被带偏）/ PCA+FSQ 闭合机器零（线性约束被线性子空间精确编码，结构性为零）但 FSQ 伤缝长（非线性泛函）、同比特效率 6.5×（332 vs 51 bit @8.6mm）——**两种表示的结构损伤面互补**；域外省道：序列域内量级、SDF 靠网格、程序结构失败（无原语）、定维潜码维度失败（39×37 padding 买的正是这个自由度）；③CFG 两笔账统一：生成侧重叠模式 401² 精确积分——组件权重近似高估崩塌但 w=4.5（SewingLDM 实际值）少数合法模式 2.1% 成立，对接 P0 {2:18,4:2}；蒸馏侧混合条件不动点延拓至 w>100 偏置单调增长**分支不灭**（远场系数 2.095≈E[1/σ]=2.143>0 恒恢复）vs 完美条件 λ*≈1.70 分支消亡复算吻合——GG 11 分支消亡是完美 conditioning 特有病理，现实版是"有界但越来越偏"；安全窗 w≲2–3，拉宽唯一途径=提高条件分离度（**蒸馏分离度，不是蒸馏均值** = P3 第一优先级）。§6 互不引用三角（SewingLDM↔LatentFM、Dress-1-to-3/DMap↔生成系、GarmentDreamer↔版型系）+ 同团队对照（Xie/Yang/Jiang 两篇）+ **19 篇零家报告可制造性指标**；90 条勘误统计（自错近 1/3，全部由第二种独立验证暴露）。§8.3 自勘误：GG 07 打版器三沉睡 bug（领点全身宽度/腰线高于肩 100cm 每件自交/臀胸同摆 y=0——GG 07 只用边向量从未暴露）、镜像量化退化、PCA 秩退化两连、log-sum-exp 模式质量反向。实验脚本 `scripts/garment_generation_experiments/gg12_experiments.py` + `gg12_output.txt`） |
