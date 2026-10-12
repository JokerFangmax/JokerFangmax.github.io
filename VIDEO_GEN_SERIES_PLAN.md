# 视频生成 × 3D 论文系列博客规划（活文档）

> 论文来源：Hanyang Wang (汪晗阳， 清华大学电子工程系， 导师段岳岐) 的全部 9 篇论文 —— <https://hanyang-21.github.io>。
> 系列目标：**通过逐篇精读这 9 篇论文，把视频生成及相关领域的原理彻底搞懂**——每个核心公式从零推导，配直觉解释。
> 本文件是系列总体规划，**每发布一篇更新一次**。

---

## 一、系列目录（12 篇）

组织逻辑：**背景铺垫（01–02）→ 生成什么（03–08：对象→场景→语义→物理）→ 如何变好（09–11：对齐→搜索→引导，三个正交杠杆）→ 版图总结（12）**。

| # | 标题 | 论文/来源 | 核心问题 | 发表处 | 前置 |
|---|------|----------|---------|--------|------|
| 01 | 扩散模型基础：从 DDPM 到 Flow Matching 与 CFG | — | 扩散模型在训什么？Flow Matching 与 DDPM 是一回事吗？CFG 为什么能"引导"？ | — | 概率论、微积分 |
| 02 | 视频扩散与 3D 先验：架构、渲染与蒸馏 | — | 视频扩散模型怎么在帧间保持一致？NeRF/3DGS 怎么渲染？SDS 梯度从哪来？ | — | 01 |
| 03 | Unique3D：单图到高质量 3D 网格 | arXiv:2405.20343 | SDS 慢、前馈模型糙——高质量+高效率+一致性能同时要到吗？ | NeurIPS 2024 | 01, 02 |
| 04 | Make-Your-3D：主体驱动 3D 生成 | arXiv:2403.09625 | 给一张主体照片+文本，如何生成"既是它、又听话"的 3D 内容？ | ECCV 2024 | 03 |
| 05 | ReconX：视频扩散重建任意稀疏视图场景 | arXiv:2408.16767 | 稀疏视图重建是病态问题——把"重建"改写成"生成"为什么可行？ | IEEE TIP 2025 | 02 |
| 06 | VideoScene：视频扩散蒸馏为一步 3D 场景生成 | arXiv:2504.01956 | 视频扩散先验太慢（几十步去噪）——怎么蒸馏到一步？ | CVPR 2025 Highlight | 05 |
| 07 | LangScene-X：TriMap 视频扩散与 3D 语言场 | arXiv:2507.02813 | 稀疏视图下如何得到"能对话"的 3D 场景（几何+外观+语义）？ | ICCV 2025 | 05 |
| 08 | Physics3D：视频扩散学习 3D 高斯物理属性 | arXiv:2406.04338 | 生成的 3D 只有颜色和形状——怎么让它"懂物理"（黏弹性/塑性）？ | arXiv 2024 | 05 |
| 09 | DreamReward-X：3D 生成的人类偏好对齐 | arXiv:2403.14613 + TPAMI 扩展 | RLHF 能搬进 3D 生成吗？偏好数据怎么标、reward model 怎么训、怎么微调？ | IEEE TPAMI 2025 | 03, RL系列 |
| 10 | Video-T1：视频生成的测试时扩展 | arXiv:2503.18942 | 推理时多花算力能换来更好的视频吗？ verifier 怎么引导搜索？ | ICCV 2025 | 01, RL系列 |
| 11 | CFG-Ctrl：无分类器引导的控制论重构 | arXiv:2603.03281 | CFG 本质是什么控制器？为什么大 guidance scale 会崩？ | CVPR 2026 Highlight | 01 |
| 12 | 技术版图总结：引导 · 蒸馏 · 对齐 · 搜索 | 全部 9 篇 | 对预训练生成模型的四类干预如何构成 2024–2026 的研究范式？ | — | 03–11 |

---

## 二、依赖关系树

```
扩散基础 (01)
   ↓
视频扩散+3D先验 (02) ─────────────────┐
   ↓              ↓                   │
Unique3D (03)   ReconX (05)          │
   ↓        ↓         ↓               │
Make-Your-3D  VideoScene(06)          │
   (04)          ↓      ↓             │
             LangScene-X Physics3D    │
                (07)      (08)        │
                 │          │         │
   DreamReward-X (09) ← 03+RL系列     │
                 ↓                    │
   Video-T1 (10) ← 01+RL系列          │
                 ↓                    │
   CFG-Ctrl (11) ← 01 (采样理论)      │
                 ↓                    │
   版图总结 (12) ←────────── 全部 ─────┘
```

关键依赖说明：

- **01 是全系列地基**：DDPM/Flow Matching/CFG 推导被所有论文篇引用；
- **05 (ReconX) 是核心枢纽**：确立"视频扩散 = 3D 场景的生成先验"范式，06/07/08 都是该范式向效率/语义/物理的延伸；
- **09/10 与 RL 系列强互链**：DreamReward-X ↔ RLHF/Bradley-Terry/DPO（RL 系列 Blog 14-16 语境）；Video-T1 ↔ 测试时搜索/RLVR（RL 系列 Blog 21 语境）；
- **11 (CFG-Ctrl) 是理论终章**：把"采样时的引导"重构为控制论问题（P 控制 → 滑模控制 → Lyapunov 稳定性），与 FA 系列的收敛性思想呼应；
- **03/04 相对独立**（image→3D 路线），可与 05–08 并行阅读。

---

## 三、技术主题地图：四类干预 × 三个输出域

9 篇论文 = 对同一个预训练生成底座（扩散模型）的四类干预手段，落在不同输出域：

| 干预杠杆 | 思想 | 本系列论文 |
|---------|------|-----------|
| **引导 (Guidance)** | 推理时改 velocity/噪声预测的合成方式 | CFG-Ctrl（滑模控制）；背景：CFG 全家桶 |
| **蒸馏 (Distillation)** | 把多步先验压进少步/单步、或跨模态迁移 | VideoScene（Leap Flow 蒸馏）；Physics3D（视频扩散先验→物理参数蒸馏）；Unique3D（多视图+法线扩散的级联） |
| **对齐 (Alignment)** | 用人类偏好信号微调生成模型 | DreamReward-X（Reward3D + DreamFL） |
| **测试时搜索 (Search)** | 推理时用 verifier 选噪声/分支 | Video-T1（linear search vs Tree-of-Frames） |
| （跨域迁移） | 视频扩散先验作为通用 3D/4D 监督 | ReconX、LangScene-X、Physics3D、VideoScene |

---

## 四、易混淆概念辨析地图（随篇更新）

| 概念对 | 辨析要点 | 出现于 |
|--------|---------|--------|
| DDPM vs Flow Matching / Rectified Flow | 训练目标形式不同（噪声预测 vs velocity/流匹配），采样统一为 ODE/SDE 视角；SD3/Flux 属后者 | 01, 11 |
| CFG vs Classifier Guidance | 前者无需分类器（条件/无条件差），后者需分类器梯度；引导强度的副作用（过饱和） | 01, 11 |
| P 控制与滑模控制 | CFG = 固定增益 P 控制（超调/不稳定）；滑模 = 非线性切换 + 流形收敛 | 11 |
| SDS vs 多视图扩散 vs 前馈重建 | SDS：逐案例优化慢；多视图扩散：一次生成多视图再重建（Unique3D）；前馈：直接回归 | 02, 03 |
| 联合先验蒸馏 vs 采样+重建 | 同一联合先验：走 score 蒸馏会 stall 在全局均值（好初值守不住、加倍步数逃不掉——高噪声项的均值回归是结构性的）；走"采一个一致样本+确定性重建"1 次前向精确恢复。**模式选择交给生成、几何交给重建**——蒸馏路线的生存条件是条件锐度（均值≈众数） | 03 |
| 随机旋转积分平均 vs Frankot–Chellappa 投影 | 非无旋伪法线的两条可积化路线：单路径积分误差 = 随机游走 O(σ√N)；旋转平均按 1/√R 压低、R→∞ 收敛到 L2 投影同级地板——trick 是投影的蒙特卡洛近似（R 次积分换一次频域变换，换来 mask 任意形状的适配） | 03 |
| 几何置信度 vs 语义置信度 | cos²(法线,视线) 加权 = σ∝1/cos 隐含模型下的逆方差最优权（对异方差噪声 MMSE）；但对"自信地错"的正面视图给满权重——几何权重只能分配信任、不能验证信任，语义一致性必须由上游联合注意力保证 | 03 |
| NeRF vs 3DGS | 隐式体积渲染 vs 显式高斯基元光栅化；效率与可编辑性差异 | 02 |
| 视频扩散的 3D 一致性从哪来 | temporal attention？3D structure conditioning？epipolar 约束？各论文答案不同 | 05, 06, 07 |
| 逐步蒸馏 vs 跳步蒸馏 | consistency/Leap Flow：跳过冗余 timestep；策略网络自适应选步 | 06 |
| RLHF vs DPO vs 直接 reward 反传 | reward model + PPO / 偏好对直接优化 / reward loss 直接微调扩散模型（DreamFL） | 09 |
| linear search vs Tree-of-Frames | 全量噪声候选并行去噪（贵） vs 自适应扩展剪枝（省）；LLM 测试时扩展的类比 | 10 |
| 表面属性 vs 物理属性 | 颜色/形状 vs 杨氏模量/泊松比/黏弹性参数；可微仿真回路 | 08 |
| 逐场景优化 vs 可泛化 | per-scene fine-tune（NeRF 系）vs feed-forward / 量化压缩（LangScene-X 的 LQC） | 07 |

---

## 五、单篇统一模板（8 节）

1. **这篇论文解决什么问题**：具体动机、前人方法的缺陷（为什么难）
2. **所需背景最小集**：引用 01/02 或前篇链接，只补该篇特有前置
3. **方法总览**：pipeline 逐模块描述（文字版流程图）
4. **核心原理完整推导**：关键公式从零推（训练目标、损失、梯度形式），配每步直觉
5. **训练与推理细节**：数据、条件注入方式、推理流程、超参要点
6. **实验与消融**：关键数字、每个设计选择的验证（只写核对过的数字）
7. **与其他论文的联系**：系列内互链 + RL/FA 系列连接
8. **局限与思考**：失败模式、开放问题、面试可能被问的点

写作约定：

- 术语首次出现：中文（English）；
- 数学公式用 LaTeX（`math: true`）；
- **每篇写作前必须 fetch 论文全文（arxiv.org/html/<ID>）核对细节**——摘要级信息不够支撑"完整推导"；论文实验数字只写核对过的；DreamReward-X 的 TPAMI 增量内容需在写作时核对 IEEE 版本（arXiv:2403.14613 是基础版 DreamReward：Junliang Ye 等，Reward3D + DreamFL）；
- 可复现的数值实验写代码验证后放正文（如 01 的 2D toy DDPM/Flow Matching、11 的 CFG vs 滑模对比模拟）；
- 发布路径：`content/post/video_generation/VG_<NN>_<slug>/index.md`；
- 每篇头部互链前篇；每发布一篇更新本文件进度表。

---

## 六、与既有系列的互链分配

| 本系列 | 互链目标 | 连接点 |
|--------|---------|--------|
| 09 DreamReward-X | RL 系列（RLHF/DPO 博客） | 偏好建模 Bradley-Terry、reward model 训练、生成模型微调 vs PPO/DPO |
| 10 Video-T1 | RL 系列（RLVR/测试时博客） | verifier-guided search ↔ LLM 测试时扩展；reward 信号来源 |
| 11 CFG-Ctrl | FA 系列 03（Banach 不动点） | 迭代收敛性、Lyapunov 稳定性与压缩映射的亲缘 |
| 01/02/11/12 | MITS6184 系列（Generative Models 栏目） | 课程版地基互链：MITS_2（Flow/Diffusion 统一）↔ VG 01；MITS_5（CFG 原理，§16 大 scale 副作用停在 heuristic）↔ VG 01 §5.3 数值实验 + VG 11（控制论答案）；MITS_6（SD3 工程栈）↔ VG 02（视频侧延伸）；VG 12 §2 总索引 |
| 05–08 | algorithms/RL 按需 | 优化器选择、蒸馏即分布匹配等轻连接 |

---

## 七、进度追踪

| # | 状态 | 发布路径 |
|---|------|---------|
| 01 | ✅ 已发布 | `content/post/video_generation/VG_01_diffusion_basics/index.md` |
| 02 | ✅ 已发布 | `content/post/video_generation/VG_02_video_3d_priors/index.md` |
| 03 | ✅ 已发布 | `content/post/video_generation/VG_03_unique3d/index.md`（2026-09-30。论文 v3 + 官方代码双路核对，12 条勘误：标题无 "Single Stage"、Hanyang Wang 第五作者非一作、无 row-wise/unshuffle 注意力（实为 4 视图 self-attention 序列拼接）、无 icosahedron 初始化（法线积分 + screened Poisson）、论文 "SGD" 实为 Adam 变体、"300+100 步" 实为 200+100、训练 4 天 vs 项目页 5 天、SyncDreamer 不在定量表、超分主路径 Real-ESRGAN 非 ControlNet 等。两实验：①Janus 机制化三路线——单视角 SDS+CFG 的 J 随 w 单调升且 x̄ 贴 CFG 放大均值锚（w≥2 过饱和越过 1）、**联合先验蒸馏 stall 在对称中心**（好初值守不住 23.3%≈随机、1200 步逃不掉——均值回归结构性）、联合采样+重建 1 次前向精确恢复（94.0% vs 解析 93.6%）——"采样-重建分工"而非"更好先验"才是机制；②ISOMER 两机制——法线积分随机游走解析锚精确验证（0.8519 vs 0.8419）、旋转平均 1/√R 收敛到 Frankot–Chellappa L2 投影地板（蒙特卡洛近似）、cos² 加权 = σ∝1/cos 逆方差最优（0.0706 ≈ oracle 0.0705）、错误视图正面 bin 吃满 δ（几何置信度防不了"自信地错"——issue #3 断裂 bug 的候选机制）。脚本 `vg03_janus.py` / `vg03_isomer.py` 含 v1/v2 实现教训） |
| 04 | ✅ 已发布 | `content/post/video_generation/VG_04_make_your_3d/index.md`（2026-10-11。论文 ECCV 2024 全文核对（正文 + 附录 A 超参/A.2 时间账/A.3 方向 prompt/B 附加结果/C GPT-4V 评估）。核心：分布调和 Eq.3–5（目标分布按视图因子分解，两个预训练模型各逼近一半）、co-evolution 两步（identity-aware：冻 p_m 生成多视图+增广→CLIP→只调 p_c image cross-attn 30 步；subject-prior：原始 p_c 带属性多样本+Omnidata 法线→调 p_m cross-domain self-attn ~100 步，Eq.6 参数保持 λ‖θ−θ₀‖₁/N_θ）。toy 实验 v4（`vg04_v4.py`，线性高斯世界，2000 wild 主体）五行对照：朴素级联身份 0.378/属性 0.25 → co-evolution 0.199/0.94；C/D 消融独立复现两失败机制（属性断链=架构无通路、身份收缩=微调数据源污染）；E 消融验证参数保持（视图结构 0.185→0.893）；id-aware 系数重校准 0.40→0.98。v1–v3 教训：属性通道须独立维度（v2 属性落身份空间内指标耦合）、视图偏移按纯几何估计（属性经 delta 重复注入会出 1.11 系统偏差）、lam 须与数据量匹配（lam=1 时 B 视图结构 0.579）。实验数字：R-Precision 0.817/0.764/0.826 vs DreamBooth3D 0.783/0.710/0.797；user study 9.01/8.91/8.70/9.05 vs 5.33–6.89；36× 快、5 分钟、单卡 3090） |
| 05 | ✅ 已发布 | `content/post/video_generation/VG_05_reconx/index.md`（2026-10-12。arXiv v4（CVPR 2025）全文核对：方法三模块（DUSt3R 全局点云 Eq.9 → FPS+cross-attn 编码 Eq.10 → dual cross-attn 注入 Eq.11 λ_F tanh 门控 → 置信度 3DGS Eq.13–14）+ Prop.1 证明（Jensen + 函数空间嵌套，严格号论证为定性——博文做了两点保留的批判性读法）+ 全部表格数字（Easy/Hard/Cross、Mip360/TnT/DL3DV、CAT3D/ReconFusion 对比、Table V 七行消融）+ 增量式 360° 外推。toy v2（`vg05_toy.py`，2000 场景，48 维线性世界 16 可见/32 未见）：A 0.813 / B 图像条件独立幻觉 0.662（帧间不一致 1.19）/ C 3D 锚定上限 0.339（0.15）/ D 坏帧均匀 LS 0.534 / E 置信度加权 0.357——B→C 差距=一致性+DUSt3R 单目先验信息增量两笔账；D→E 验证置信度机制（诚实标注：坏帧设定偏悲观，论文消融该项仅 +0.5 dB）。v1 教训：视图不能以零填充进未见维的 LS（稀释帧内容）；坏帧须为共享偏置（系统性错误）而非零均值噪声——线性世界零均值噪声对均匀 LS 无害，置信度收益全部来自偏置部分） |
| 06 | ✅ 已发布 | `content/post/video_generation/VG_06_videoscene/index.md`（2026-10-13。arXiv:2504.01956v2 全文核对：方法（§3.1 一致性蒸馏 Eq.1–3 背景、§3.3 3D-aware leap flow Eq.4–6、§3.4 DDPNet Eq.7–9 contextual bandit + stopgrad + 策略不回传梯度、附录 6.3 bandit 论证、8.6 failure case 穿门）+ 全部表格数字（Table 1 一步 FVD 103.42 vs CogVideoX 一步 753.02/50 步 521.04；Table 2 Cross ACID 121.93 vs 微调 CogVideoX 一步 464.87/50 步 114.04；Table 3 消融 171.38/543.53/106.28/97.53——leap flow 为最大单项；Table 6 runtime 2.98 s vs 179–934 s；Table 7 显存 66+10+0.02≈76 GB；900+20K 步、8×A100 两天、DDPNet 前 4000 步）。toy（`vg06_toy.py`，2000 场景，32 维 8 维流形，50% 流形外伪影幅度 2.0，一步模型=保留+投影+0.5λ 重画方差）：A 纯噪声起点 1.490 / B 固定小步长 0.341/3.435/1.888 / C 固定大步长 1.060/1.101/1.081 / D 学习策略 0.320/1.109/0.715 / E oracle 0.323/1.111/0.717——λ 双刃、最优 λ 二元结构（干净≈0/伪影≈1）、学习策略追平 oracle。三个 bug 教训：① numpy 广播 (1,)×(200,)；② 信号维不可被 (1−λ) 缩放（保留+投影插值语义）；③ oracle 公式须含零维重画噪声项（此前学习版反超 oracle 的原因）） |
| 07 | ✅ 已发布 | `content/post/video_generation/VG_07_langscene_x/index.md`（2026-10-14。arXiv:2507.02813v1 全文核对：方法（TriMap 视频扩散四阶段渐进训练：web 一周→3D 一致 2000 步→StableNormal 200 段 800 步→SAM2 300 段 1000 步，Eq.1 多域扩散损失，i2v 改首尾帧插值；LQC Eq.2–6 VQ K=2048/D=3、straight-through+字典+伪 mask 监督 λ=1/0.2/0.5、COCO 500K 步；表面场 Eq.7–8 渐进法线正则（角度 τ 过滤离群）+ Eq.9 2D/3D 语义聚类（公式排版有瑕疵已注明））+ Table 1/2（LERF-OVS 80.85/50.52 vs LangSplat 35.72/15.37；ScanNet 94.14/66.54）+ 消融 Table 3（渐进 44.25/LQC 44.59/L_s2d 40.05/完整 51.68）。**勘误：论文称 ScanNet mIoU +14.92 与表格不符（实为 +10.46），已在博文指出**。toy v2（`vg07_toy.py`，64 类线性语义世界，4×4×4 码本，共享理想编码器只比解码端）：A 0.764 / E 连续 0.935/0.788 / F VQ snap 0.998/0.858 / G 超半码距崩 0.226 / H 码本不足 0.369 / I 对等混合 snap 第三方 0.235——离散码本=噪声信道纠错解码 + 三笔明码标价（半径/容量/平局）；量化抹平类内噪声（F>A 余弦）→ 实例级查询被封顶。v1 教训：① phi 展开一次性 (500,3600,515) 7.4GB 须分块；② 边界随机对 5:5 混合使中点距双父均超半码距（snap 全到第三方）、且像素级 argmax 对 VQ 确定性特征有平局偏见——改 85:15 alpha 合成 + 区域级平均激活指标） |
| 08 | ✅ 已发布 | `content/post/video_generation/VG_08_physics3d/index.md`（2026-10-15。arXiv:2406.04338v3 全文核对（正文 + 附录 A.2 附加消融/A.3 user study/B.1 MPM/B.2 SDS 推导/B.3 连续介质力学）：方法（Eq.1 黏弹性分解 F=F_E F_P=F_N F_V、F_P=I、σ=σ_E+σ_N **两支并联——正文未点破、由 B.3 两条直觉（弹塑性支=纯弹簧、黏弹性支=弹簧串联阻尼器「应变蓄水池」）拼出标准线性固体/Zener 结构**；Eq.8–10 fixed corotated；Eq.11–13+B.3 log 主应变 + trial-and-correction、耗散势 ψ_V 偏量/体量两路 ν_d=ν_v、**直接学更新规则系数 A,B 而非 ν_d/ν_v**；Eq.14 SVD-SDS 固定相机 + Eq.15 L_Fill 两损失分开优化；50³ 网格、每帧间隔 400 子步 Δt=1e-4、每物体约 5 分钟单卡 A6000）+ Table 1 五行（14.72/0.59/0.49/0.59 vs PhysDreamer 13.89、PhysGaussian 13.86；消融 w/o 黏弹 14.13/w/o 弹塑 13.53）+ user study（30 人 20 轮：8.95/8.57/9.05，运动真实性差距最大）。toy（`vg08_toy.py`，1D Zener 集总系统闭式解（线性 ODE 特征分解；Routh–Hurwitz 可证正参数恒稳定），21 帧带噪观测拟合 + 20 帧外推）：A 纯弹性单参数最优拟合外推 RMSE 0.626=oracle 25 倍、外推段平均|x| 0.670 vs 真实 0.090（能量永不耗散）且最优 k=27.1 落在静态 25/瞬时 33 刚度之间（Zener 刚度频率依赖） / B 手设静态刚度→低频（PhysGaussian 定性失败机制版） / C Zener 三参数贴噪声地板 0.021/0.025 / D 噪声×3 与 E 短窗 1.2s：刚度恢复稳（8%/7%）但阻尼 c 误差 55%/54%——**阻尼只经衰减包络进观测、需多个衰减周期才可辨识（视频先验时长的隐性预算）** / G 结构对照：GT 换纯弹性则单参数完美恢复（25.07 vs 25）——PhysDreamer 失败错在模型族不在参数维度。脚本教训：log 空间无界 LM 会走病态区（正参数系统恒稳定但 V 奇异→exp 溢出），须加边界 trf；RK4 纯 Python 太慢须闭式解；异常静默吞掉导致 None 调试绕路） |
| 09 | ✅ 已发布 | `content/post/video_generation/VG_09_dreamreward_x/index.md`（2026-10-16。arXiv:2403.14613v1 全文核对（正文 + 附录 0.A 推导/0.B 伪代码与实现/0.C Elo 表与消融）；DreamReward-X TPAMI 版 IEEE 付费墙无 arXiv，增量（reward-aware noise sampling 释放多样性）单列 §6.4 摘要级引用并诚实标注。方法：标注管线（Cap3D 图聚类 5000 prompt → MVDream 4-10 资产 → 过滤 2530 组/25,304 对 → 三维 1-6 分 + 排名 + 冲突实时检测 + 二次校验）→ Reward3D（ImageReward 底座 cross-attn+MLP、冻结 80%、BT 排名损失 Eq.1、4090 batch 8 AdamW 1e-5）→ DreamFL（Eq.9-14：δϵ=−∂r/∂ϵ_φ **启发式构造非定理**——λ 吸收所有常数 "to ensure the equation holds"，博文批判性指出；L_Reward≈L_SDS−λ_r·r；λ_r=t(L_SDS,r)·μ 量级配平，μ "0.25 until 0.6" 论文自相矛盾已指出；t<t_threshold 低噪声段才计 reward；收官 200 步纯 reward；40 分钟 ≈ MVDream、已有资产 2 分钟微调）+ Table 1 全表核对（478/1480/2.594/1.833/0.274；CLIP 无区分度 0.222-0.274；Spearman 1.00/1.00/0.80/0.60）+ Table 2 user study（4.88/5.03/5.30/5.07，**一致性第二输 MVDream 5.38**、65% 偏好）+ Elo Table 3（六维 1567/1305/1465/1479/1622/1442；MVDream 非每列次席——Text-Asset 列 ProlificDreamer 1262 更高）+ Table 4（1221 vs 779）+ Fig.8 λ_r 消融三态 + Fig.7 骨干无关 + Fig.3 负例测试。toy（`vg09_toy.py`，R^4 世界 800 资产、25k 对 β=1.5、M=8 ReLU 特征 BT 拟合、x* 距 2.12σ）：A λ=0 停先验最优 −4.500 / B 纯 reward 无锚 **hacking 纯形态**——分布内 corr 0.969 代理走上扬 +53、真实崩 −1287.5、38σ 外推（hacking 不是学错是优化把输入推出标注域）/ C 配平 λ=1 锚定 −0.480（消掉 89% 偏好差距；L_SDS 就是 KL 锚——DreamFL=无显式 KL 的 PPO）/ C′ λ=3 未配平 −4.683 / D 收官纯 reward 放大版 −1403.6（悬崖边跳舞，安全余量是步数给的）/ E 10% 数据 −0.588（toy 瓶颈在外推非标注方差——现实瓶颈是覆盖缺口，同一曲线两个轴）/ F oracle 0（锚的代价：最后 0.66σ 走不完=对齐 vs 先验质量的显式交易）；λ 扫描完整倒 U 曲线（0/0.1/0.3/0.5/1/2/3/5 → −4.50/−3.51/−2.83/−1.94/**−0.48**/−0.68/−4.68/−24.21）对应论文 Fig.8 三态。脚本教训：M=64 代理太好 B 行不崩须 M=8+bump；归一化梯度上升平衡点随机游走永不停须真梯度+步长范数上限让平衡点成吸引子；λ=3 冲过平衡点→把 λ 扫描做成主实验） |
| 10 | ✅ 已发布 | `content/post/video_generation/VG_10_video_t1/index.md`（2026-10-17。arXiv:2503.18942v2 全文核对（正文 + 附录 A.1 image-level alignment/A.2 hierarchical prompting/A.3 复杂度推导/B.1-B.4 含失败案例）；ICCV 2025 经项目页 🔥 确认（arXiv 页未标注）。作者：Fangfu Liu 与 Hanyang Wang 共同一作（清华+腾讯）。方法：TTS 重构为噪声空间轨迹搜索（§3.1 G/V/f 三元组、视频=噪声空间的退化树）；Random Linear Search（Alg.1，N 全量去噪+verifier argmax，O(TN)，两限制=线性须穷举+候选孤立）；Tree-of-Frames（Alg.2，优先队列+段边界 b=2 分支+剪枝保 k_t；A.3：单层 O(kb+blog(kb))、worst O(TN) 但 b_t=1 居多+k_t 几何收缩 → **O(N+T)**；三技术件：image-level alignment 二值预筛/DeepSeek-R1-8B 三段式子 prompt（首帧静态/中间动作/结尾状态）/多 verifier 秩平均 î=argmax(1/|M|)Σc_v·Rank_v；**本质=提高早段观测对成片的预测力**）；实验：6 模型（扩散 CogVideoX-2B/5B、OpenSora-v1.2；自回归 NOVA、Pyramid-Flow SD3/FLUX；ToF 仅自回归——架构决定可搜索性）×VBench 全涨（Table 2 全表核对：Pyramid-Flow FLUX 81.61→86.51、Multiple Objects 61.08→88.93 +45.6%；NOVA 仅 +1.58%）；Table 1 GFLOPs ToF 省 2.9-3.2×（5.22e7 vs 1.62e7）；B.3 2B+TTS 多维追平 13B HunyuanVideo；B.4 拍手失败案例。toy（`vg10_toy.py`，S=8 段路径生成器、总方差归一、MC 3000）：实验1 N 扫描——linear 贴无噪 best-of-N 上限按 √(2lnN) 对数增长（N=16→32 只 +0.31σ）、ToF(16) 同质量 62 vs 128 帧一半代价；实验2 世界结构——**剪枝风险完全由 ρ_1 决定**：front ρ=0.666 ToF 2.219>linear 1.717（又便宜又好）、back ρ=0.088 ToF 崩 0.797（linear 三世界不变=完美对照），hierarchical prompting=把世界从 back 推向 front；实验3 池结构——高方差池 +1.743/低方差池 +0.350（NOVA 机制版）/能力上限池推到 cap 即停（拍手机制版：搜索买不到池里没有的能力）；实验4 verifier——集成=降噪 σ_v/√3、**偏置 β=1.0 仍有界误选 1.234>0**（对照 VG 09 B/D 行 −1288：梯度可被 hack 出分布、选择不能——搜索只能选择不能创造）。§7 新观点：VG 06 蒸馏压平采样步数是否也压平样本方差→一步模型搜索空间坍缩（效率与搜索的硬币两面）） |
| 11 | ✅ 已发布 | `content/post/video_generation/VG_11_cfg_ctrl/index.md`（2026-10-18。arXiv:2603.03281v2 全文核对（正文 + 附录 6.2 CFG 变体/6.3 Theorem 1 与稳定性走廊/7 实现细节/8.1–8.4/9.1–9.2）；CVPR 2026 Highlight 经项目页 🔥 确认。作者：Hanyang Wang 与 Yiyang Liu 共同一作（清华，Yueqi Duan 通讯）。方法：CFG-Ctrl 重构——采样 ODE 为控制仿射系统 dx/dt = v+u（Eq.8-9，G=I_d 全驱动）、控制律 u=K_t·Π_t(e)（Eq.10）、**K=w、Π=I 代回恰好是 CFG（Eq.11-13）→ CFG ≡ 增益恒定的 P 控制，w 即比例增益**；Table 1 统一框架（权重调度=增益调度、APG=投影反馈 P_t=v(c)v(c)ᵀ/|v(c)|²、CFG-Zero⋆=投影到 v(∅) 正交补 K=[wI, s_t/(1−s_t)I]、Rectified-CFG++=MPC α(t)=λ_max(1−t)^γ）；SMC-CFG（Eq.18-28：滑模面 s=ė+λe、Lyapunov V=½‖s‖²、误差动力学 Eq.22 控制仿射 ė=Φ+w(∂v(c)/∂x−∂v(∅)/∂x)GΔe、切换律 Δe=−k·sign(s)、V̇≤−(kb_min−δ)‖s‖、有限时间收敛 t≤‖s(0)‖/η）；Theorem 1（k>δ/(w−ρD)+ε、‖ΔΓ‖≤ρ、w>ρD）；**稳定性走廊 Eq.45：δ_est/w < k < 2‖s_t‖/(wΔt)**；Alg.1 每步重测 e、修正经速度合成不持久改状态。实验：MS-COCO 5000 对、SD3.5(8B)/Flux-dev(12B, guidance-distilled 嵌入引导设 1)/Qwen-Image(20B)、A100 40GB、网格搜索 λ∈{2..8} k∈{0.01..0.8} 200 例辅助集→三模型全选 λ=6、k=0.1/0.7/0.1。Table 2 八指标（SMC-CFG 几乎全最优：SD3.5 FID 21.421→20.044 ImgRwd 0.8889→0.9486；Flux 27.323→26.398/1.0558；Qwen 35.431→33.371/1.2035；**诚实注明：SD3.5 Aesthetic 输 CFG-Zero⋆ 5.5790 vs 5.6124、Qwen 无引导 FID 最低 24.894=引导用自然图像统计换语义对齐**）；Table 3 消融（λ 平缓 λ=5 k=0.1 FID 25.951 最优；k 行 U 型：小 k CLIP 低 FID 好/大 k CLIP 0.3741 拉满但美感差=论文 abrupt pulls/vibrations；消融最优 (5,0.1) 与 Flux 部署 (6,0.7) 的多指标错位已点破）；Table 5 T2I-CompBench 四维全涨 Spatial 最大（Qwen 0.2968→0.4085）；Table 6 VQAScore 0.83/0.64/0.72→0.89/0.68/0.77；Table 7 Wan2.2-TI2V-5B Total 0.5594→0.5839 Color 0.9087→0.9818（跨模态迁移=修动力学非图像统计的最强证据）；Table 8 FLOPs 逐位相同（1024²: 3590870.89 vs .93）。§9.1 Flux 默认 w=2 各方法最优、更大快退、SMC 持续升；§9.2 自认两超参按模型手调。toy（`vg11_toy.py`，e∈R² 相平面，被控对象=饱和反阻尼 m(w,r)+旋转耦合 (w−1)κJ+有界漂移，SMC 按 Alg.1 差分 ė+s=ė+λe+Δe=−k·sign(s)、控制经 w 进 ė）：实验1 w 扫描——**CFG 过 w_c 后极限环且半径精确等于闭式平衡解 r²=ρ²(α(w−1)(w−w_c)−λ₀)/λ₀（w=8/10/12 预测 2.20/3.91/5.35 vs 实测 2.197/3.906/5.345 三位吻合）**、SMC 全段 0.005–0.291、w=8 时 157×差距；SMC 优势恰在失稳后出现（w≤6 对象稳定时 SMC 纯付 2-3× TV 颤振代价）；实验2 k 走廊完整 U 型（k=0.05 权限 wk=0.4<增长率 0.54 收敛慢、0.1-0.3 走廊、≥0.5 颤振主导 k=5 TV 反超 CFG）= Eq.45 两界的机制版；实验3 λ≥1 全平（流形上衰减速率在 T=20 内全收敛）→ 解释三模型全选 λ=6、k 才是按模型分化的参数（0.1/0.7/0.1）；实验4 匹配漂移不变性（δ≤2.0≈权限 wk=2.4 不动）越界软失败（δ=12 mean 0.753 非爆炸）vs CFG 全程趴极限环。**脚本教训三条：① 显式 Euler 对纯旋转恒数值不稳定（|1+Δtλ_rot|>1）高 w 段爆 1e20 是积分器发散非动力学——线性部分须 2×2 精确指数 e^{mΔt}R(θ)；② ė 差分方向写反 (e_prev−e)/Δt 时控制器反向泵能量、SMC 全面劣于 CFG——滑模 toy 第一检查项是符号；③ β 三次正反馈无饱和同样全线 NaN——非线性 toy 的失败机制必须手工造出有界版本（饱和反阻尼）否则数值问题淹没机制故事**。§7 新观点：引导三部合流（09 进损失=无界风险/10 进选择=对数回报有界误选/11 进动力学=颤振有界走廊受限——干预杠杆率与失控半径同源）；VG 06 交叉问题：走廊上界 ∝1/Δt，蒸馏压步数与滑模在极限处冲突（与 10 的「蒸馏压平搜索空间」同构）；Lyapunov vs Banach 压缩映射=收敛性证明两家族（有限时间 vs 渐近，非光滑 vs 度量）） |
| 12 | ✅ 已发布 | `content/post/video_generation/VG_12_summary/index.md`（2026-10-19。系列终章，综合 01–11 全部内容，无单篇论文/toy。结构：§2 地基两公理（扩散=流匹配=向量场——VG 11 控制仿射重构的前提；SDS 灌 3D 的起点与病灶——VG 03 蒸馏 stall 23.3% vs 采样-重建 94.0% 的范式分水岭）；§3 输出域扩张表（03–08 六篇各一行：推进/机制/核对数字）+ 三条机制结论（蒸馏生存条件=条件锐度/量化=噪声信道纠错/可辨识性由激励决定）；§4 四类干预地图（对齐/搜索/引导/蒸馏 × 干预位置/信号来源/收益形状/失控半径）——对齐改权重、引导改每步动力学、搜索改走完挑哪个；§5 **核心综合：调制方式决定收益形状与失控半径**——梯度（无界/38σ 外推 −1287.5）vs 选择（对数 σ√(2lnN)/有界误选 +1.23σ）vs 切换（失稳区最大/颤振走廊）vs 压平（效率无界/方差-搜索空间-走廊同时收窄）+ 四条推论（无免费方向/混合必然/蒸馏与三杠杆极限冲突——效率与可干预性同一硬币/架构决定可干预性）；§6 易混淆概念辨析总表 16 组（按采样与引导/3D 生成与重建/训练与效率/推理与评估四组）+ reward model vs verifier 的 Goodhart 分界；§7 依赖演化图 + 作者轨迹（Hanyang Wang 九篇：从「生成什么」到「怎么变好」，四根杠杆各占一篇的路线图意识）；§8 全场通病总清算（评估循环性 09/10/11 三连——当评估信号与优化信号同源增益有自我兑现成分）+ 五点思考（干预自动化=自适应控制/为可干预性设计架构/reward 基础设施三坑「贵偏弱」/科研面试复用清单四条/版图缺角：长程一致性、多主体物理、可编辑性）；附录 A 十二篇全目录。数字全部回引自已发布各篇核对过的数字（修正过一处跨表混用：VG 06 的 103.42 是 Table 1 FVD、114.04 是 Table 2 Cross ACID，不可直接对比，已改为 103.42 vs 50 步 521.04）。Hugo 验证八项全过（h2=10 与历篇一致、11 个系列链接全部存在）。**系列 12/12 完结**） |
