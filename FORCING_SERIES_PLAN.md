# Forcing 系列论文精读博客规划（活文档）

> 论文来源：`/Users/fanghojui/Downloads/forcing` 下的 21 篇 forcing 及相关论文（2015–2026.10，自回归视频扩散训练协议主线）。
> 系列目标：**把「训练时看见什么历史、生成时依赖什么历史」这条 forcing 主线读透**——从 Scheduled Sampling 到 Self Forcing、长视频缓存、少步蒸馏与上下文梯度，每篇完整推导 + 实验数字核对。
> 已有基础：`content/post/forcing/`（总览）+ `forcing_01`–`forcing_21`（导读存根，各 46 行）。本任务把 21 篇存根**全部扩写成精读文章**（对齐 WM/VG 系列规格），总览随之更新。
> 论文全文已提取：`tmp/forcing_texts/<arXivID>.txt`（pypdfium2，含 PAGE 分隔符）。
> 本文件是系列总体规划，每完成一篇更新一次进度表。

---

## 一、系列目录（00 总览 + 21 篇精读）

组织逻辑（总览篇已定，五部分）：**训练-推理差异（01–03）→ Diffusion/Self Forcing（04–07）→ 长视频与上下文（08–12）→ 少步蒸馏与初始化（13–17）→ 效率与上下文梯度（18–21）**。

| # | slug | 论文 | arXiv | 部分标题 | 全文 |
|---|------|------|-------|---------|------|
| 00 | `forcing` | 系列总览与阅读路线 | — | 五个观察维度 + 四条阅读路线 | — |
| 01 | `forcing_01` | Scheduled Sampling | 1506.03099v3 | 从 Teacher Forcing 到模型自己的历史 | ✅ 34K |
| 02 | `forcing_02` | Scheduled Sampling for Transformers | 1906.07651v1 | 为什么 Transformer 需要两遍解码 | ✅ 22K |
| 03 | `forcing_03` | Parallel Scheduled Sampling | 1906.04331v2 | 让历史替换适应并行训练 | ✅ 31K |
| 04 | `forcing_04` | Diffusion Forcing | 2407.01392v4 | 每个 token 都有自己的噪声等级 | ✅ 123K |
| 05 | `forcing_05` | History-Guided Video Diffusion | 2502.06764v2 | 把历史帧作为可调节的 guidance | ✅ 129K |
| 06 | `forcing_06` | CausVid | 2412.07772v4 | 从双向教师走向流式因果学生 | ✅ 66K |
| 07 | `forcing_07` | Self Forcing | 2506.08009v2 | 在自己的生成历史上训练 | ✅ 70K |
| 08 | `forcing_08` | Rolling Forcing | 2509.25161v1 | 滚动联合去噪如何抑制误差传播 | ✅ 56K |
| 09 | `forcing_09` | LongLive | 2509.22622v2 | 长视频训练与实时提示切换 | ✅ 65K |
| 10 | `forcing_10` | Self-Forcing++ | 2510.02283v1 | 用短片段监督长 rollout | ✅ 71K |
| 11 | `forcing_11` | Deep Forcing | 2512.05081v1 | 无需微调的历史缓存管理 | ✅ 69K |
| 12 | `forcing_12` | Context Forcing | 2602.06028v1 | 让教师也看见长历史 | ✅ 48K |
| 13 | `forcing_13` | Causal Forcing | 2602.02214v5 | 师生因果结构为何影响 ODE 初始化 | ✅ 85K |
| 14 | `forcing_14` | Diagonal Distillation | 2603.09488v2 | 沿时间与噪声两条轴分配计算 | ✅ 106K |
| 15 | `forcing_15` | Causal Forcing++ | 2605.15141v4 | 用在线一致性蒸馏初始化因果学生 | ✅ 65K |
| 16 | `forcing_16` | One-Forcing | 2605.23458v2 | 单步生成的清晰度与动态性 | ✅ 75K |
| 17 | `forcing_17` | Causal-rCM | 2606.25473v1 | 统一一致性初始化与分布匹配细化 | ✅ 105K |
| 18 | `forcing_18` | Separable Causal Diffusion | 2602.10095v1 | 把时间推理与逐帧去噪拆开 | ✅ 88K |
| 19 | `forcing_19` | Sparse Forcing | 2604.21221v1 | 持久记忆与局部稀疏注意力一起训练 | ✅ 76K |
| 20 | `forcing_20` | Self Gradient Forcing | 2607.20368v2 | 未来损失能否监督历史记忆写入 | ✅ 85K |
| 21 | `forcing_21` | SGF+ | 2610.10429v2 | 分离上下文写入与去噪的梯度职责 | ✅ 72K |

## 二、单篇统一模板（8 节，对齐 WM/VG 系列）

1. **这篇论文解决什么问题**：具体动机、前人方法的缺陷（为什么难）；引用总览/前篇链接
2. **最小背景**：只补该篇特有前置（扩散/蒸馏基础引用 VG 01/VG 06）
3. **方法总览**：pipeline 逐模块文字描述（严禁 ASCII 流程图）
4. **核心原理完整推导**：训练目标、损失、梯度形式从零推，每步配直觉；引用论文公式必须 `\tag{Eq.N}`
5. **训练与推理细节**：数据、条件注入、采样流程、超参要点（数字标本收折叠块）
6. **实验与消融**：关键数字 + 每个设计选择的验证（只写从论文全文核对过的数字）
7. **与系列其他篇目的关系**：系列内互链 + WM/VG 系列连接
8. **局限与个人评价**：失败模式、开放问题

## 三、写作约定

- **frontmatter 照抄现有存根**（title/date/math/tags/categories/series/weight/paper_id 全部不变），仅 description 更新为精读版摘要（含 2–3 个核对过的关键数字）
- 文末保留 `{{< forcing_nav >}}`；「论文信息」节（原题/版本/原文链接）保留
- 正文主线 + `<details><summary><b>展开：…</b></summary>` 折叠块（语法照抄 WM 系列）；每篇 200–320 行
- 术语首现中文（English）；`$$` 块前后必须空行
- **数字红线**：实验数字只写 `tmp/forcing_texts/<ID>.txt` 里核对过的；论文公式编号 `\tag{Eq.N}` 与原文一致
- 互链格式：系列内 `../forcing_XX/`；总览 `../forcing/`；VG 系列 `../../video_generation/VG_01_diffusion_basics/` 等；WM 系列 `../../world_memory/WM_03_memory_forcing/` 等
- 不做 toy 脚本（本轮以论文精读为主；如某篇有极简可算机制说明可用小例子，但不强制）
- 诚实记录：论文含糊处、版本差异、附录勘误如实写

## 四、与既有系列的互链分配

| 本系列 | 互链目标 | 连接点 |
|--------|---------|--------|
| 01–03（SS 家族） | WM 03（Memory Forcing 的 TF/CFT 两难） | teacher forcing 的训练-推理错配是同一问题 |
| 04（Diffusion Forcing）/ 05（HGVD） | WM 系列 DFoT 基线多处引用 | FloWM/RELIC 的基线即 DFoT |
| 05–07、12（蒸馏/上下文） | VG 06（VideoScene 少步蒸馏）、VG 10（测试时扩展语境） | 蒸馏压步数、AR rollout 的错配 |
| 07（Self Forcing） | WM 03/04（self-forcing 训练协议）、VG 11（CFG 控制律） | 训练时看见自己的错误 |
| 12（Context Forcing） | WM 05（WorldPlay 用 Context Forcing 蒸馏） | 同名技术在长视频记忆上的应用 |
| 全系列 | VG 01（扩散基础） | 加噪/去噪/score/flow matching 前置 |

## 五、并行分工（8 个 agent）

| Agent | 篇目 | 论文 |
|-------|------|------|
| A | 01–03 | SS 三部曲（短文，快） |
| B | 04–05 | DF + HGVD（两篇长文） |
| C | 06–07 | CausVid + Self Forcing |
| D | 08–10 | Rolling + LongLive + SF++ |
| E | 11–13 | Deep + Context + Causal Forcing |
| F | 14–16 | Diagonal + CF++ + One-Forcing |
| G | 17–18 | Causal-rCM + SCD |
| H | 19–21 | Sparse + SGF + SGF+ |

## 六、进度追踪

| # | 状态 | 备注 |
|---|------|------|
| 00 | ✅ 总览已发布 | 21 篇精读全部完成后同步更新：五维问题地图、五部分目录、四条阅读路线、8 节统一骨架 |
| 01 | ✅ 289 行 | SS 三调度曲线 + 混合历史目标有偏性（Jensen 视角，折叠块）；CIDEr 89.5→92.1、TIMIT 46.0→34.5；「换历史来源=换优化目标」警示 |
| 02 | ✅ 217 行 | Transformer 两遍解码（Embedding Mix）；桶内并行采样的收敛性分析 |
| 03 | ✅ 289 行 | 并行采样-混合（K 轮、概率 p）；实测 0.13 vs 47 步/秒；K≥T、p=1 收敛到真采样解码 |
| 04 | ✅ 319 行 | 逐 token 独立噪声等级；任意窗口训练 + Full/Incremental/Stabilized 采样；stabilization 是采样技术非损失项（按论文原文纠正任务书） |
| 05 | ✅ 320 行 | 历史帧 guidance（帧条件加权分数差）；按论文原文写，任务书中「stabilized conditional/逆向 guidance」措辞不实（agent 诚实记录） |
| 06 | ✅ 307 行 | CausVid 双向→因果蒸馏；异步噪声方案 + Fig.5 GRPO 数字经主会话复核（w/o: 6.89/24.52; w: 20.82/2.00） |
| 07 | ✅ 320 行 | Self Forcing：next-token rollout + 梯度截断；hindsight rollout；系列枢纽篇 |
| 08 | ✅ 254 行 | Rolling Forcing 滚动联合去噪 |
| 09 | ✅ 204 行 | LongLive 长视频训练 + 实时提示切换 |
| 10 | ✅ 234 行 | Self-Forcing++ 短片段监督长 rollout |
| 11 | ✅ 319 行 | Deep Forcing：无需微调的 sink 缓存管理（token 0–15,600 sink）；5s→60s+ 外推 12×；Dynamic Degree 57.56 vs SF 36.62 |
| 12 | ✅ 266 行 | Context Forcing：预训练长上下文教师；上下文 20+s vs FramePack 9.2s；与 WM_05 WorldPlay 同名组件撞名辨析（折叠块） |
| 13 | ✅ 310 行 | Causal Forcing：ODE 初始化完整推导链（MSE→条件期望→单射性→二阶矩恒等式→因果 ODE）；TF VR 3.343 vs DF 1.583；吞吐超 Wan2.1 2079% |
| 14 | ✅ 317 行 | Diagonal Distillation：时间×噪声双轴计算分配 |
| 15 | ✅ 291 行 | Causal Forcing++：在线一致性蒸馏初始化 |
| 16 | ✅ 319 行 | One-Forcing：单步生成清晰度/动态性 |
| 17 | ✅ 318 行 | Causal-rCM：一致性初始化 + 分布匹配细化统一 |
| 18 | ✅ 254 行 | Separable Causal Diffusion：时间推理与逐帧去噪解耦 |
| 19 | ✅ 319 行 | Sparse Forcing：持久记忆 + 局部稀疏注意力（PBSA 峰值加速 1.16–11.11×）；KV 44.9 GB = 参数 17.26× 动机 |
| 20 | ✅ 274 行 | SGF：未来损失监督历史写入（可微 cache 重建 RelL2 1.41%）；GSB 1900+ 判断 10 组全正 |
| 21 | ✅ 290 行 | SGF+：梯度冲突定量（Attention 104.2°/FFN 106.3°、512 组配对余弦全负）→ 角色分离；末篇含五维地图全系列收束表 |

全部 21 篇 + 总览已于 2026-10-14 完成扩写并通过核验（行数 204–320、折叠块开闭配对、`\tag{Eq.N}` 与原文一致、零 `../../` 链接残留、关键实验数字逐篇对照 tmp/forcing_texts 原文抽查通过）。description 长度统一对齐仓库既有风格（约 300–420 字符），未压缩到任务书 100–200 字。
