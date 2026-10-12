---
title: "VG 02: 视频扩散与 3D 先验——架构、渲染与蒸馏"
date: 2026-09-28
math: true
tags:
  - Notes
  - Video Generation
  - Diffusion Model
  - 3D Reconstruction
categories:
  - Video Generation
description: "3D VAE 与 DiT 的数学图景、NeRF 体积渲染与 3DGS 光栅化推导、SDS 梯度的完整来源、跨模态蒸馏的统一视角——ReconX/VideoScene/Physics3D 的公共前置"
---

# VG 02：视频扩散与 3D 先验——架构、渲染与蒸馏

> 📖 本系列精读 [Hanyang Wang](https://hanyang-21.github.io) 的 9 篇论文。本篇是第二篇背景地基（前篇：[VG 01：DDPM → Flow Matching → CFG](../VG_01_diffusion_basics/)）。
> 工程视角的姊妹篇：本栏目 [WAN-Video](../WAN-Video/)（Wan 报告精读，覆盖数据/训练/推理工程的完整版）、[CogVideoX](../CogVideoX/)。

---

## 1. 这一篇解决什么问题？

VG 01 结尾说：ReconX / VideoScene / LangScene-X / Physics3D 四篇论文的共同底牌是"**预训练视频扩散模型里存着世界先验**"。这句话要成立，必须回答三个问题：

1. **视频扩散模型到底在算什么**？——3D VAE + DiT + 时间注意力的数学图景（为什么这套架构能"看懂"运动）；
2. **3D 表示和渲染是什么**？——NeRF 的体积渲染、3DGS 的光栅化，两条技术路线的渲染方程与优劣（论文里"重建 3D 场景"的那一半）;
3. **怎么把 2D/视频先验"灌进" 3D 表示**？——SDS（Score Distillation Sampling）梯度的完整推导，以及它的各种变体——这是 DreamFusion 开创、贯穿本系列 5 篇论文的核心机制。

一句话总览：**生成（视频扩散）× 表示（NeRF/3DGS）× 蒸馏（SDS）= 2024–2026 的 3D AIGC 范式**。

---

## 2. 视频扩散架构：三个部件的数学

（WAN 笔记有完整工程版；这里只讲"数学上在算什么"。）

### 2.1 3D VAE：把视频压进可计算的隐空间

原始视频张量 $(F, H, W, 3)$ 太大——比如 121 帧 × 720p ≈ 10⁸ 维，直接在上面做扩散不现实。**3D VAE** 学一个编解码对

$$
\mathcal{E}: \mathbb{R}^{F\times H\times W\times 3} \to \mathbb{R}^{f\times h\times w\times c}, \qquad \mathcal{D} \approx \mathcal{E}^{-1},
$$

时空联合下采样（典型 $4\times8\times8$：时间压 4 倍、空间压 8 倍），把扩散搬到隐空间做，最后 $\mathcal{D}$ 解码回像素。数学上就是训练一个自编码器（重建损失 + 感知损失 + GAN 损失 + KL 正则），没有太多神秘——**关键约束是因果性**（causal conv：时间维只看过去帧），让编解码支持流式与任意长度（WAN §8.2 详述）。

<details>
<summary><b>展开：$4\times8\times8$ 到底压了多少？——「只是缩小」的三个错觉</b></summary>

**错觉一：压缩比是 $1/(4\cdot8\cdot8)=1/256$。** 漏了通道维：RGB 3 通道 → 隐通道典型 16（通道**变多**）。以 121 帧 720p 为例：

| | 形状 | 元素数 |
|---|---|---|
| 像素视频 | $121\times720\times1280\times3$ | $\approx3.3\times10^8$ |
| 隐视频 | $30\times90\times160\times16$ | $\approx6.9\times10^6$ |
| **实际压缩比** | | $\approx48\times$（不是 $256\times$） |

通道 3→16 不是浪费：若只有 3 个通道，每个隐「像素」要用 3 个数概括原位 $4\times8\times8=256$ 个像素的全部信息——不可恢复，解码必糊。通道扩张是把挤掉的空间信息摊到通道维存。**VAE 压的是感知冗余，不是硬丢分辨率**；16 通道是率失真平衡点，压得更狠字幕和人脸纹理就救不回来。

**错觉二：压 48 倍只是「小了一点」。** 扩散的开销不在元素数，在 **token 数的平方**（§2.3：注意力矩阵 $(NM)\times(NM)$）。token 账（patchify $1\times2\times2$）：

| | token 数 | 注意力开销 $\propto(\text{token})^2$ |
|---|---|---|
| 像素空间直接做 | $121\times640\times360\approx2.8\times10^7$ | $\approx7.7\times10^{14}$ |
| 隐空间 | $30\times45\times80\approx1.1\times10^5$ | $\approx1.2\times10^{10}$ |
| **比值** | $\approx258\times$ | $\approx\mathbf{6.6\times10^4\times}$ |

元素数降 48 倍、注意力账单降 6 万倍——平方放大器才是「解决高维」的真正机制。

**错觉三：VAE 只是「缩小」。** 它是学出来的近感知无损压缩，同时干两件事（LDM 论文的原始框架）：

| 任务 | 谁负责 | 内容 |
|---|---|---|
| **感知压缩** | VAE | 去掉像素级冗余（相邻像素高度相关、相邻帧近乎重复）——对语义毫无贡献的部分 |
| **语义生成** | 扩散模型 | 在冗余已除的空间里学「什么画面配什么条件」 |

第二行有数学理由，拆成四点（尽量说人话版）：

**(a) 加噪像兑水，但图像的浓淡太悬殊。** VG 01 的前向是「信号保留 $\sqrt{\bar\alpha_t}$ 倍、噪声补到 $\sqrt{1-\bar\alpha_t}$」——对图像的每个「方面」一视同仁（方面 = 一种变化方式：整体调亮是一种，细小纹理抖动是另一种）。问题在于**各方面的强弱差得离谱**：「整体明暗」的起伏幅度可以是「细小纹理」的一百倍。兑同样的水，浓的还剩味道、淡的瞬间没味。拿两个幅度悬殊的方面做数字标本（幅度 $10$ 与 $0.1$），看「兑到一半」（信号量 $=$ 噪声量）分别需要什么时刻：

| 想让谁「半干净」 | 需要的 $\bar\alpha_t$ | 此时另一个方面的处境 |
|---|---|---|
| 浓的（幅度 $10$） | $\bar\alpha\approx0.01$ | 淡的信号只剩噪声的万分之一——**早就全糊** |
| 淡的（幅度 $0.1$） | $\bar\alpha\approx0.99$ | 浓的信号还是噪声的近万倍——**原封未动** |

**不存在任何时刻让两方面同时「半干净半噪声」**：去噪器每一步都被迫同时面对「这块还清楚、那块已糊」。日程表怎么调都顾此失彼——这正是像素空间里 $\beta_t$ 日程（线性/余弦之争）成为敏感超参的原因。VAE 要干的事之一，就是**训练时先把各方面的浓淡拉成一样**，于是一张日程表就能服务全部方面。

**(b) 拉浓淡的手正是 KL 正则——先看四件套各管什么。** VAE 训练时四个损失同时压着，各管一摊：

| 损失 | 形式 | 管什么 | 不加会怎样 |
|---|---|---|---|
| 重建 | $\|\mathcal D(\mathcal E(x))-x\|$ | 编了还能解回原视频 | 压缩根本不成立 |
| 感知（LPIPS） | 深度特征间的距离 | 只罚「看得见」的失真，放过「看不见」的 | 逐像素都对，但看着糊 |
| GAN | 判别器分真假 | 逼出高频细节与纹理 | 输出向「平均图」靠拢、发糊 |
| KL | $\mathrm{KL}\big(q(z\mid x)\,\|\,\mathcal N(0,\mathbf I)\big)$ | 把隐空间**拉向标准正态**：各通道的数值范围拉平、通道之间不串味 | 隐空间分布歪斜、浓淡不齐，扩散难做 |

（$q(z\mid x)$ = 编码器对单张图 $x$ 输出的隐变量分布；「标准正态」就是 VG 01 里 $\varepsilon$ 服从的那个分布。）KL 这行最容易被误读成「只是为了让隐空间可以采样」——它同时是在**给扩散机制整形数据**：(a) 里需要的「拉平浓淡」正是它干的；GAN 项则反着拉（保住细节纹理，防止拉平过程把纹理抹掉），两者对抗着落成「浓淡均匀 + 细节还在」的隐空间。

**(c) 网络的学习预算被像素补丁活吃掉。** 网络容量有限，花在「天空渐变怎么接着渐变、草地纹理怎么铺」这类局部补丁活上的预算，就不能花在「这里该有一个人」这种语义上。像素级损失又不区分「看得见的错」和「看不见的错」——模型被迫分心去修感知上无所谓的像素对齐。VAE 解码器把补丁活整个承包了（它就是个专门的补丁机器），扩散网络的预算就全留给语义。

**(d) 不做 VAE 行不行？行，但不值。** 像素空间直接做扩散**能做**（图像侧 DDPM 本来就是像素空间的；视频侧 VDM 论文做过）——不是数学不可能，而是同样算力下预算被冗余吃掉、日程难调、收敛慢。隐空间方案相当于「一次 VAE 预训练，之后所有训练都在干净分布上」。

所以「$10^8\to10^6$」只是账面，**换分布**才是隐空间扩散的根本动机。

</details>

### 2.2 DiT：去噪器是个 Transformer

VG 01 的 $\varepsilon_\theta(x_t, t, c)$ 在视频模型里是一个 **Diffusion Transformer**：

1. 隐视频张量 patchify 成 token 序列：$(f, h, w, c) \to N = f\cdot\frac{h}{p}\cdot\frac{w}{p}$ 个 token（$p$ = patch size）；
2. 每层做自注意力（这些 token 互相 attend）+ MLP；
3. 条件注入：
   - **时间步 $t$**：正弦嵌入后经 MLP，通过 **adaLN**（自适应 LayerNorm——把 $t$ 的嵌入变成 scale/shift 参数调制每层激活）注入；
   - **文本 $c$**：T5/CLIP 编码成 token 序列，通过 **cross-attention**（每个视频 token attend 文本 token）注入。

**为什么 Transformer 取代了 U-Net？** 没有深刻的数学原因——是**缩放律**：Transformer 的堆叠更可预测、训练更稳定、硬件亲和（大核卷积的显存/访存特性差）。视频生成领域 2024 年后全面 DiT 化（Sora、CogVideoX、WAN 均是）。

<details>
<summary><b>展开：DiT 取代 U-Net 的四条理由——性能之外的经济账</b></summary>

背景：U-Net 与 DiT 干同一件事——都是 $\varepsilon_\theta(x_t,t,c)$ 的实现。DDPM/SD1–2/SDXL 的去噪器是 U-Net（卷积 + 下采样-上采样 + 跳连）；Peebles & Xie 2023（DiT 论文）证明纯 Transformer 也能做去噪器。注意「取代」不是性能碾压——以下四条里没有一条是「U-Net 原理上做不到」，拼的是**工程经济学**：

| # | 维度 | U-Net（卷积） | DiT（Transformer） |
|---|---|---|---|
| ① 缩放律（主因） | 堆大后的行为 | 超参多、每个尺寸近似新模型，外推不可靠 | 损失随算力/参数/数据**幂律下降且光滑可外推**：训小的、拟合曲线、外推预算训大的 |
| ② 统一处理 token | 输入怎么进网络 | 图像走卷积、文本走 cross-attention、时间步走嵌入——**三套机制** | 全部 token 化后同一种层处理，位置嵌入直接标注时空坐标 |
| ③ 硬件亲和 | 算力利用率 | 大核卷积显存/访存特性差，优化 kernel 靠手工 | 计算主体是矩阵乘（GEMM），张量核吃得满，生态（FlashAttention 等）全在为它优化 |
| ④ 长程依赖 | 远处信息怎么传 | 感受野靠堆层数缓慢生长，浅层只看局部 | 任意两层间 token 直连，全局一步到位 |

**① 为什么是主因**：缩放律 = 「损失随算力幂律下降」的规律。它改变的是**训前决策方式**——训几个小模型，拟合 loss–算力关系，就能外推出「$10\times$ 算力该配多少参数多少数据」再训大的。视频生成一次预训练动辄数月、数百万美元，**训坏了不能重来**，这个外推能力是生命线。U-Net 的等效关系也有人拟合过，但「每个尺寸都要重新调超参」使外推不可靠。

**② 为什么对视频特别关键**：视频张量天然时空混合。U-Net 的卷积是为 2D 图像设计的，搬上视频要改 3D 卷积或插伪时间层——每个改动都是新的设计自由度、都要重调；DiT 只需把 patchify 从 2D 改 3D，其余不动。§2.3 说「视频 DiT 的本质增量只有一个」，这句话成立的前提正是**架构本身是内容无关的 token 处理器**。

**③④ 是助推**，单独哪条都不足以换架构。还有一条历史反证：DiT 论文自己的「Gflops–性能」曲线显示，**同等算力下精心设计的 U-Net 并不落下风**——DiT 赢的是「相同人力下能可靠堆到的规模」，不是单点效率。

一句话：U-Net 是为图像手工调的精装机器，DiT 是通用零件的堆叠机器；视频生成拼的是「谁能被可靠地堆大」，而堆大的可预测性是 Transformer 十年生态攒下的老本。

</details>

### 2.3 时间注意力：视频模型唯一的新部件

对比图像 DiT，视频 DiT 的本质增量只有一个：**token 里混着不同帧**，全注意力天然覆盖帧间关系。数学上，一个 $N$ 帧、每帧 $M$ 个 token 的视频，注意力矩阵是 $(NM)\times(NM)$——**序列长度平方爆炸**就是视频扩散贵的根源（WAN §12 的 2D context parallel 就在解这个）。

**但注意**：时间注意力学到的是**数据驱动的统计一致性**，不是几何保证。两个视角的对应像素错位、物体瞬移，只要在训练数据里少见，模型就可能生成出来——**"视频扩散的 3D 一致性是软的"**。这句判断是 ReconX（VG 05）论文动机的直接来源：它要**用显式 3D 结构条件把这个软一致性变硬**。

---

## 3. 3D 表示与渲染：NeRF 与 3DGS

"重建/生成 3D 场景"的第一问题是：**3D 场景在计算机里怎么存？** 两条主流路线。

### 3.1 NeRF：隐式函数 + 体积渲染

**表示**：一个 MLP $f_\theta: (\mathbf{x}, \mathbf{d}) \mapsto (\sigma, \mathbf{c})$，把空间点坐标 $\mathbf{x}\in\mathbb{R}^3$ 和观察方向 $\mathbf{d}$ 映射到体密度 $\sigma$ 和颜色 $\mathbf{c}$。场景 = 网络权重（隐式、连续）。

**渲染**：体积渲染方程（沿光线积分）。相机射线 $\mathbf{r}(s) = \mathbf{o} + s\mathbf{d}$，像素颜色是

$$
C(\mathbf{r}) = \int_0^\infty T(s)\,\sigma\big(\mathbf{r}(s)\big)\,\mathbf{c}\big(\mathbf{r}(s), \mathbf{d}\big)\,ds, \qquad T(s) = \exp\left(-\int_0^s \sigma(\mathbf{r}(u))\,du\right),
$$

$T(s)$ 是**透射率**：光线到 $s$ 处都没被挡住的概率（密度沿途累积吸收）。直觉：把光线想成在雾里穿行，每一点以密度 $\sigma$ 概率"撞上"一点颜色，撞上就按吸收律衰减后面的贡献。

**离散化**（实际计算形式）：沿光线采 $N$ 个点 $s_i$，

$$
\hat C(\mathbf{r}) = \sum_{i=1}^{N} T_i\,\alpha_i\,\mathbf{c}_i, \qquad T_i = \exp\left(-\sum_{j<i}\sigma_j \delta_j\right),\quad \alpha_i = 1 - \exp(-\sigma_i \delta_i),
$$

$\alpha_i$ 是第 $i$ 段的"不透明度"。**训练**：光度损失 $\sum_{\mathbf{r}} \|\hat C(\mathbf{r}) - C^{\text{gt}}(\mathbf{r})\|^2$，对每条已知相机参数的射线反传——纯"从 2D 照片学 3D"的魔法。

**代价**：渲染一个像素要采几百个点、每个点过一遍 MLP——每帧几百毫秒，无法实时；且是逐场景优化（per-scene overfit），没有泛化。

### 3.2 3D Gaussian Splatting：显式基元 + 光栅化

**表示**：场景 = 一组 3D 高斯基元 $\{(\mu_i, \Sigma_i, \alpha_i, \mathbf{c}_i)\}_{i=1}^M$——位置 $\mu$、协方差 $\Sigma$（存四元数+尺度，保证正定）、不透明度 $\alpha$、球谐系数 $\mathbf{c}$（视角相关颜色）。**显式、离散、可编辑**。

**渲染**：把每个高斯**投影**（splatt）到屏幕上变成 2D 高斯（2D 投影协方差有闭式：$\Sigma' = JW\Sigma W^{\top}J^{\top}$，$W$ 是视角变换、$J$ 是投影的局部仿射近似），然后按深度排序做 **alpha blending**：

$$
C(\mathbf{r}) = \sum_{i \in \text{排序}} \mathbf{c}_i\,\alpha_i' \prod_{j<i}(1-\alpha_j'),
$$

与 NeRF 的离散体积渲染**形式完全同构**（$T_i = \prod_{j<i}(1-\alpha_j')$ 就是透射率的离散版）——区别在于：NeRF 每像素**按需**查询网络几百次，3DGS 把基元**预先**摆好、光栅化一次成型。速度差 2–3 个数量级（实时 100+ FPS vs 每帧秒级）。

**训练**：还是光度损失反传，但梯度直接到每个高斯的参数；配合自适应密度化（分裂/克隆/剪枝高斯）。**可微光栅化**是整个体系能训练的唯一原因。

### 3.3 对照表 + 本系列怎么选

| | NeRF | 3DGS |
|---|------|------|
| 表示 | 隐式 MLP（连续） | 显式高斯集合（离散） |
| 渲染 | 逐射线采样积分（慢） | 投影 + 排序混合（快 100×） |
| 每场景优化 | 分钟–小时级 | 分钟级 |
| 可编辑性 | 差（黑盒） | 好（每个高斯是对象） |
| 动态/物理扩展 | 难 | **好**（高斯可携带动学属性） |

本系列论文的选型：**ReconX / VideoScene / LangScene-X 全部用 3DGS**（速度对"从生成的视频恢复场景"的闭环至关重要）；**Physics3D 用 3DGS 有更深的原因**——物理仿真需要"离散质点"携带物理属性（杨氏模量、速度），高斯天然是质点（VG 08 详述）。

---

## 4. SDS：把 2D 先验灌进 3D 表示

### 4.1 问题设置

想生成 3D（比如一只 3D 狗），手里只有：一个 3D 表示 $\theta$（NeRF 或 3DGS，$\theta$ 是它的参数、**待优化**）+ 一个预训练的图像扩散模型 $\varepsilon_\phi$（它只见过 2D 图像；下标 $\phi$ 是它的权重、**全程冻结**——它在这里只当批评家，不被训练）。**没有 3D 数据、没有多视角数据。**

DreamFusion 的思路：渲染 $\theta$ 的一个随机视角 $\hat x = \text{render}(\theta, \mathbf{d})$，问扩散模型"这图像不像狗的概率梯度是多少"，用这个信号更新 $\theta$。**图像扩散模型变成了可微的、3D 感知的批评家。**

<details>
<summary><b>展开：§3 不是已经有 3D 表示了吗，为什么还要「生成」？</b></summary>

**关键区分：3D 表示是容器，不是内容。** §3 的 NeRF/3DGS 回答的是「3D 场景在计算机里**怎么存**」（格式），但容器里得装东西——NeRF 存的是 MLP 权重 $\theta$，而 §3.1 的训练方式（光度损失、逐场景优化）意味着：**必须先有一只真实的 3D 狗（或至少一组多视角照片），才训得出它的 $\theta$**。所以本节的真实处境是：

| 已有 | 缺失 |
|---|---|
| 容器（NeRF/3DGS：怎么存） | **内容**（$\theta$ 该取什么值：存什么） |
| 2D 扩散模型（见过几十亿张图，懂「狗长什么样」） | 3D 数据、多视角数据 |

「生成 3D」指的就是**凭空得到 $\theta$**：DreamFusion 里初始 $\theta$ 是随机初始化的，渲染出来一团噪声云，SDS 一步步把它拉成一只狗。

**「不是直接渲染就好了吗」——如果 $\theta$ 已经装好了内容，对，直接渲染就好**（那正是 §3 的用途：重建/表示）。但渲染 $\text{render}(\theta,\mathbf d)$ 本身不创造任何东西，只是把已有内容投到某个视角；问题恰恰是手里没有装好内容的 $\theta$，也没有数据能训出它。3D 数据的两条来源都断了：

| 来源 | 为什么断 |
|---|---|
| 3D 扫描（实物建模） | 设备贵、覆盖率极低——互联网上有几十亿张狗照片，没有几百万只扫好的 3D 狗 |
| 多视角视频重建（§3 的管线） | 需 controlled 多机位拍摄，数据规模比单目视频小多个数量级，且大多限特定类别 |

而 2D 图像数据便宜到泛滥。SDS 的全部聪明之处：**用可微渲染当翻译官，让只会看 2D 的扩散模型给 3D 参数当教练**——每次渲染一个随机视角、让 2D 批评家打分、梯度经渲染雅可比搬回 3D。等价读法：不是「造 3D 数据」，而是「**2D 监督经渲染可微地漏进 3D**」。

</details>

### 4.2 完整推导（三步）

**先对账本节要用的符号**（下标是分工的关键：**$\theta$ 归学生、$\phi$ 归老师**——$\theta$ 是被优化的 3D 参数，$\phi$ 是冻结的扩散模型权重，DreamFusion 里两者从不互换角色）：

| 符号 | 是什么 | 对应 VG 01 里的 |
|---|---|---|
| $\theta$ | 3D 场景参数（NeRF 的 MLP 权重 / 3DGS 的高斯参数）——**待优化** | 无对应（VG 01 没有 3D） |
| $\phi$ | 预训练扩散模型的权重——**冻结不动**，只前向 | $\theta$（VG 01 里 $\varepsilon_\theta$ 的下标是被训的；这里换成 $\phi$ 就是为了提醒「冻结」） |
| $\varepsilon_\phi(x_t,t,c)$ | 那个冻结网络（去噪器）的输出 | $\varepsilon_\theta(x_t,t,c)$——同一个函数形状 |
| $\hat x = \text{render}(\theta,\mathbf d)$ | 当前 3D 参数在视角 $\mathbf d$ 下的**渲染图** | $x_0$（干净图的角色由它扮演） |
| $\varepsilon$ | 你亲手注入的标准高斯噪声（一次一颗） | $\varepsilon$（VG 01 前向那颗骰子） |
| $x_t=\sqrt{\bar\alpha_t}\hat x+\sqrt{1-\bar\alpha_t}\varepsilon$ | 渲染图**加噪后**的版本 | $x_t$（同一配方，$x_0$ 换成 $\hat x$） |
| $p_\phi(x\mid c)$ | **扩散模型隐式定义的图像分布**：「给定文本 $c$，真图长什么样」——它不是哪个显式公式，是这个冻结模型所相信的分布（网络见过什么数据、就信什么）。$\mathbb E_{t,\varepsilon}\log p_\phi$ 大 = 渲染图落在模型认为的「真图高分区」 | $q(x_t)$（数据的真实分布 → 换成模型相信的分布） |
| $s_\phi(x_t)=\nabla_{x_t}\log p_\phi(x_t\mid c)$ | 上面那个分布的 **score**（坡向） | score $\nabla\log q$（式 2.4 左端） |

（$t$、$\bar\alpha_t$、$c$、$\mathbf d$ 沿用前文：噪声水平、信号保留率、文本条件、相机视角。）

**然后把设计逻辑摆在公式前面**。整个 4.2 要解决的是：**「像不像真图」是个黑盒判断，不可微，没法直接对 $\theta$ 求梯度**——三步推导就是把这个判断一步步翻译成梯度。每步各回答一个问题：

| 步 | 回答的问题 | 用的工具 |
|---|---|---|
| ① 概率目标 | 「像真图」怎么变成**可爬的坡**？ | 加噪 + 取 $\log p_\phi$（海拔） |
| ② 关键恒等式 | 坡的**方向**（score）怎么用网络读出来？ | VG 01 §2.4 的 score 关系，换个主体 |
| ③ 链式法则 | 图像空间的方向怎么**搬回 3D 参数**？ | 渲染雅可比 $J_\theta^\top$（翻译官）+ 方差缩减 |

与 VG 01 的角色对照（先记住这张表，第二步会回来用它）：VG 01 里 $\varepsilon_\theta$ 是**被训练的**学生，score 是训练的副产品；这里 $\varepsilon_\phi$ 是**冻结的**批评家，score 是被消费的读数。**同一个网络、同一个数学，换个角色。**

**第一步：写成概率目标。** 对渲染图加噪得 $x_t = \sqrt{\bar\alpha_t}\hat x + \sqrt{1-\bar\alpha_t}\varepsilon$，希望它接近扩散模型在文本 $c$ 下的分布，即最大化

$$
\mathbb{E}_{t,\varepsilon}\left[\log p_\phi(x_t \mid c)\right] \ \text{（对 } \theta \text{ 求梯度，} x_t \text{ 依赖 } \theta）.
$$

**为什么非要加噪**：先分清两个梯度——**渲染图对 3D 参数的梯度一直都有**（可微渲染的本职，第三步的 $J_\theta$ 就是它）；缺的是**批评家的意见**：图像空间里「这张图往哪改更像真图」，即 score $\nabla\log p_\phi$。而这个意见恰好在干净图上**读不到**：网络受训的输入全是带噪图（它会的唯一技能是「指出带噪图里的噪声」），且读数公式 $s_\phi=-\varepsilon_\phi/\sqrt{1-\bar\alpha_t}$ 要除以噪声幅值——$t\to0$ 处分母归零、网络也没在零噪声处受过训，仪表失灵。**加噪 = 把你的图搬到批评家受训的地盘上，用它听得懂的语言提问**：注入 $\varepsilon$ 得到 $x_t$（出一道它做过的题型），它的回答经第二步恒等式换算出坡向；之后才轮到 $J_\theta^\top$ 把坡向搬回 3D。（期望对 $t$ 取平均则是另一层设计：不同 $t$ 的批评家看不同「频率段」——高噪声管整体构图、低噪声管细节纹理。）

<details>
<summary><b>展开：我们怎么知道 $x_t$ 在 $p_\phi$ 下的概率是多少？——不知道，也不需要</b></summary>

自然的疑问：目标写的是最大化 $\log p_\phi(x_t\mid c)$，那 $x_t$ 在这个分布下的概率值怎么查？答案是**从头到尾不查**——梯度上升从来不问「这里海拔多少」，只问「脚下往哪边是上坡」：概率值（海拔）是 $\log p$ 本身，梯度（坡向）是 $\nabla\log p$。从第二步起，$\log p$ 的**值**再没出现过，出现的全是它的导数。

**概率值不是不想算，是算不出来。** 按定义展开：

$$
p_\phi(x_t\mid c)=\int p_\phi(x_t\mid x_0,c)\,p_\phi(x_0\mid c)\,dx_0
$$

要对**所有可能的干净图 $x_0$** 积分——整个图像空间、百万维起步，还带归一化常数，无闭式、数值不可行。扩散模型能给的只是对数似然的**变分下界**（ELBO）：每算一次要跑完整个去噪链，贵，且只是下界。**「查概率」这条路在原理上就堵死。**

**但梯度恰好免费——这是整个 trick 的核心。** 关键恒等式（VG 01 §2.4 折叠块证过同一条，换主体）：

$$
\nabla_{x_t}\log p_\phi(x_t\mid c)=\mathbb E_{p_\phi(x_0\mid x_t,c)}\big[\nabla_{x_t}\log p(x_t\mid x_0)\big]
$$

（$p_\phi(x_0\mid x_t,c)$ = 后验：模型看了带噪图后对「干净图是什么」的猜测分布；里层的条件 score 是高斯的、有闭式。）读法：**边缘 score = 各条件 score 的后验加权平均**。这个平均看起来也要做那个不可能的积分——但去噪器的训练目标恰好就是把这个平均装进网络权重：$\varepsilon_\phi(x_t,t,c)\approx\mathbb E[\varepsilon\mid x_t,c]$，**网络输出本身就是那个「不可积的平均」的化身**，一次前向 = 一次平均的完成。

| 量 | 能不能要 | 怎么得到 |
|---|---|---|
| $\log p_\phi(x_t\mid c)$ 的**值** | ✗ 原理性堵死（对全图像空间积分） | ——（ELBO 只是昂贵下界） |
| $\nabla_{x_t}\log p_\phi(x_t\mid c)$（score） | ✓ 免费 | 网络一次前向 + 乘系数（第二步恒等式） |

类比：雾中登山——等高线图（概率值）永远看不见，但脚下每步都能感到坡向（score）；扩散模型是台**「只报坡向、不报海拔」的仪表**，SDS 从设计上就只消费坡向。

（诚实的补充：连「梯度」也是近似——完整的 $\nabla_\theta\,\mathbb E\log p$ 还有一项含 score 对 $x_t$ 的导数，DreamFusion 把它扔了，这是 §4.3 病表第一行的病根之一。）

</details>

**第二步：读出 score——VG 01 式 (2.4) 的换装。** 目标：把坡向 $\nabla_{x_t}\log p_\phi(x_t\mid c)$ 用网络一次前向表示出来。四行，**每一行都是 VG 01 已证的结论换个主体，没有新数学**（②③④ 的补全留作习题 1）：

| 行 | 结论 | 用的性质 | VG 01 出处 |
|---|---|---|---|
| ① | $p_\phi(x_t\mid c)=\displaystyle\int p(x_t\mid x_0)\,p_\phi(x_0\mid c)\,dx_0$ | 「模型相信的干净图分布，卷上已知的前向核」——两个分布的含义见上一折叠块 | ——（定义） |
| ② | $\nabla_{x_t}\log p_\phi(x_t\mid c)=\mathbb E_{p_\phi(x_0\mid x_t,c)}\big[\nabla_{x_t}\log p(x_t\mid x_0)\big]$ | 混合分布的 score = 各分量 score 的**后验加权平均**（$\nabla\log\int=\mathbb E[\nabla\log\text{被积式}]$） | §2.4 第一步（同一条） |
| ③ | $=-\dfrac{x_t-\sqrt{\bar\alpha_t}\,\mathbb E[x_0\mid x_t,c]}{1-\bar\alpha_t}$ | 里层前向核是高斯，$\log p(x_t\mid x_0)$ 只剩二次式、梯度闭式；代入后期望线性、取出已知量 $\mathbb E[x_t\mid x_t]=x_t$ | §2.4 第二、三步（同一条） |
| ④ | $=-\dfrac{\mathbb E[\varepsilon\mid x_t,c]}{\sqrt{1-\bar\alpha_t}}\ \approx\ -\dfrac{\varepsilon_\phi(x_t,t,c)}{\sqrt{1-\bar\alpha_t}}$ | 分子换 ε 语言（条件期望版恒等式）；再「MSE 回归收敛到条件均值」把期望换成网络读数 | §4.3 折叠块 + §2.4 |

$$
\boxed{\ s_\phi(x_t)=\nabla_{x_t}\log p_\phi(x_t\mid c)=-\frac{\mathbb E[\varepsilon\mid x_t,c]}{\sqrt{1-\bar\alpha_t}}\ \approx\ -\frac{\varepsilon_\phi(x_t,t,c)}{\sqrt{1-\bar\alpha_t}}\ }
$$

两处诚实标注：④ 的「≈」是第一处近似（网络只是 $\mathbb E[\varepsilon\mid x_t,c]$ 的有限训练近似）；第二处在第三步之后（完整梯度还含 score 对 $x_t$ 的导数项，DreamFusion 丢弃了，§4.3 病表第一行）。**注意 $\varepsilon$（你注入的骰子）不在这条链里**——score 只是 $(x_t,t,c)$ 的函数，与哪颗骰子造出这个 $x_t$ 无关；$\varepsilon$ 出现在最终公式里是第三步方差缩减的手笔。结论与 VG 01 的关系，一张表对照（开头角色表的展开）：

| | VG 01 §2.4（训练 DDPM） | 这里（SDS 用模型） |
|---|---|---|
| 那个网络 | $\varepsilon_\theta$，**正在训练** | $\varepsilon_\phi$，**冻结**、只前向 |
| 它收敛到什么 | $\mathbb E[\varepsilon\mid x_t,c]$（后验平均意见） | 同一个事实，反过来**用**：拿输出当这个量的读数 |
| score 关系 | $\nabla\log q = -\mathbb E[\varepsilon\mid x_t]/\sqrt{1-\bar\alpha_t}$（式 2.4） | 换个主体：$\nabla\log p_\phi\approx-\varepsilon_\phi/\sqrt{1-\bar\alpha_t}$ |
| score 的角色 | 训练的**副产品** | 被消费的**仪表读数**（批评家的打分） |

（$\varepsilon-\varepsilon_\phi$ 的读法也接 VG 01：$\varepsilon_\phi$ 是**平均要去的噪声**——同一 $x_t$ 多种来历的平均意见；$\varepsilon$ 是你亲手注入的那颗**具体骰子**。「具体骰子 − 平均意见」= 这张特定带噪图的修正方向。物理比喻：$\log p$ 是海拔、score 是坡向，扩散模型是买来的地形图——SDS 站在 $x_t$ 读坡向、沿坡爬。）

**第三步：链式法则——把 2D 方向搬回 3D。** 前两步只回答了「**图像空间**里往哪改」，待更新的却是 $\theta$。由 $x_t = \sqrt{\bar\alpha_t}\,\hat x(\theta) + \sqrt{1-\bar\alpha_t}\,\varepsilon$ 依赖 $\theta$（经渲染函数，$\partial x_t/\partial\theta = \sqrt{\bar\alpha_t}\,J_\theta$，$J_\theta = \partial\hat x/\partial\theta$ 为渲染雅可比，详见下方折叠块）：

$$
\nabla_\theta\, \mathbb{E}\left[\log p_\phi(x_t\mid c)\right] = \mathbb{E}\left[\sqrt{\bar\alpha_t}\,J_\theta^{\top}\, s_\phi(x_t)\right] = \mathbb{E}\left[-\sqrt{\tfrac{\bar\alpha_t}{1-\bar\alpha_t}}\, J_\theta^{\top}\,\varepsilon_\phi(x_t, t, c)\right],
$$

（第二等号就是第二步的 $s_\phi$：网络读数除以噪声幅值 $\sqrt{1-\bar\alpha_t}$）。$J_\theta^\top$ 就是「翻译官」矩阵：图像空间的修正方向（列向量）左乘它，得到 3D 参数空间的修正方向——**可微渲染的全部意义就是让 $J_\theta$ 算得出来**（§4.1 折叠块说的「翻译官」严格指这个矩阵）。最后是工程收尾：注入的噪声 $\varepsilon$ 边际期望为零（且 $J_\theta$ 不依赖 $\varepsilon$），把它加进去做**方差缩减**（$\varepsilon_\phi - \varepsilon$ 的方差远小于 $\varepsilon_\phi$ 单独——类比 MC 估计减基线；这就是 $\varepsilon$ 进入最终公式的唯一途径），得到最终形式：

$$
\boxed{\ \nabla_\theta^{\text{SDS}} = \mathbb{E}_{t, \mathbf{d}, \varepsilon}\left[w(t)\, J_\theta^{\top}\left(\varepsilon - \varepsilon_\phi(x_t, t, c)\right)\right]\ }, \qquad w(t) = \sqrt{\tfrac{\bar\alpha_t}{1-\bar\alpha_t}}
$$

（这是对 $\mathbb{E}\log p$ 做**梯度上升**的写法。DreamFusion 原文把 $w(t)(\varepsilon_\phi-\varepsilon)$ 定义为损失 $L_{\text{SDS}}$ 的梯度、做梯度下降——两种约定差一个符号，本质相同。）

<details>
<summary><b>展开：渲染雅可比 $J_\theta$——形状、含义、从哪来</b></summary>

**定义与形状**：$J_\theta=\dfrac{\partial\hat x}{\partial\theta}\in\mathbb{R}^{D_x\times D_\theta}$——行数 $D_x$ = 图像像素数（百万级），列数 $D_\theta$ = 3D 参数个数（NeRF 是 MLP 权重数、3DGS 是所有高斯的属性数，百万到千万级）。$(i,j)$ 元素 = **第 $j$ 个 3D 参数动一点点，第 $i$ 个像素变多少**——「3D 世界 → 2D 照片」这条映射的全部一阶敏感度信息。

**它做什么**：$\nabla_\theta=J_\theta^\top\,\nabla_{x_t}$ 数学上是链式法则，几何上是一次**投影/搬运**——$J_\theta^\top\in\mathbb{R}^{D_\theta\times D_x}$ 把图像空间的批评方向（每个像素一个修正量）投到参数空间（每个参数一个修正量）。直觉读法：**第 $j$ 个参数的更新量 =「每个像素想要的修正」按「这个参数对每个像素的影响」加权求和**——参数收到的是「自己造成的像素变化」的加权和；影响的像素越多、对批评方向贡献越大的参数，被推得越狠。

**从哪来**：不需要显式构造这个天文数字级的矩阵——反向传播自动算 $J_\theta^\top v$（向量-雅可比积 vJP，恰好是公式需要的形状），工程上就是「批评 loss 对 $\theta$ 做一次 backward」。**可微渲染这个领域的存在意义就是让这条 backward 走得通**：NeRF 的体渲染方程天然可微（§3.1 的 $\hat C=\sum_iT_i\alpha_i\mathbf c_i$，每项都光滑）；3DGS 的光栅化含排序等离散操作，需要专门设计的可微光栅化器（软化处理）。

</details>

<details>
<summary><b>展开：被加噪、被批评的「图像」到底是谁的图？——SDS 从不生成目标图</b></summary>

本节的「图像」二字容易读混，先钉死：**被加噪、被批评的始终是 3D 表示的渲染图 $\hat x$，SDS 从头到尾没有生成过任何「目标图像」。**

| 候选 | 是/否 | 说明 |
|---|---|---|
| 扩散模型**生成**的目标图（「狗该长这样」的标准答案图） | **否** | 扩散模型只前向一次、当批评家，**从不采样**——$p_\phi(\cdot\mid c)$ 只用来打分，没有任何一张具体的「目标图」被画出来 |
| 3D 表示在视角 $\mathbf d$ 下的**渲染图** $\hat x$ | **是** | 每轮优化的主角：渲染 → 加噪 → 批评 → 更新 $\theta$ → 再渲染…… |

容易混淆的原因：$\log p_\phi(\cdot\mid c)$「给定文本 $c$ 下真图的分布」听起来像背后藏着一个隐形的「目标图集合」——但那个分布**只以坡向的形式存在**（上一折叠块：批评家只报坡向、不报海拔、也不画地图）。「想让渲染图靠近的目标」没有具体形态，它是数百万张可能的狗图的**平均意见场**；这也解释了 SDS 的一个病：平均意见指向的常是「典型的、各图折中的」样子（VG 01 §5.1「平均关联」的亲戚），而不是任何一张具体的图——§4.3 的「模糊/过饱和」病与这层结构性原因有关。

</details>

**直觉读法（按执行顺序合一段）**：SDS 的每一轮迭代 = **把 3D 表示渲染成图（$\hat x=\text{render}(\theta,\mathbf d)$，$\mathbf d$ 随机抽）→ 加噪把它当作 $x_t$，喂给冻结的 2D 图像扩散模型 → 用 $s_\phi\approx-\varepsilon_\phi(x_t,t,c)/\sqrt{1-\bar\alpha_t}$ 读出「这张渲染图往哪改会更像真图」的图像空间梯度 → 经渲染雅可比 $J_\theta^\top$ 把这个梯度传回 3D 参数 → 更新 $\theta$**。如此循环，直到渲染图在批评家眼中足够像真图。三个补充读法：$\varepsilon - \varepsilon_\phi(x_t)$（方差缩减后的批评信号）是"真实噪声比预测噪声多出来的部分"——指向"更像真图像"的修正方向；期望对所有噪声水平 $t$、视角 $\mathbf{d}$、噪声实例 $\varepsilon$ 取（不同 $t$ 的批评家看不同「频率段」）；**每个 3D 参数都被推向"让你的渲染在扩散模型眼中更像真图"的方向**——没有具体的目标图，「目标」始终只以坡向的形式存在（见上文折叠块）。

**一个容易搞错的地方（我推导时真踩了）**：符号。$\varepsilon-\varepsilon_\phi$ 与 $\varepsilon_\phi-\varepsilon$ 一字之差，方向完全相反——玩具实验里用错符号参数会指数飞走（见 §5）。记住锚点：**score 指向概率上升的方向，$\varepsilon_\phi$ 是 score 的负缩放，所以"去掉多余预测噪声"（$\varepsilon-\varepsilon_\phi$）才是爬坡方向。**

### 4.3 SDS 的病与药（本系列论文的对照）

SDS 开创了范式，但缺陷明显，后续工作各有药方：

| 病 | 症状 | 药方（本系列内） |
|----|------|----------------|
| 真实梯度 ≠ SDS 梯度（丢 $\varepsilon$ 项 + 用 U-Net Jacobian-free 近似） | 过饱和、模糊、细节少 | DreamReward-X（VG 09）：不蒸 score，蒸**人类偏好** |
| 每次更新要过一遍大 U-Net | 逐案例优化小时级 | Unique3D（VG 03）：干脆**前馈**，不优化；VideoScene（VG 06）：把 U-Net 蒸馏成单步 |
| 单视角独立蒸馏，无 3D 一致性硬约束 | 多视角矛盾、Janus 脸 | ReconX（VG 05）：改用**视频扩散**（时间注意力天然耦合多视角）；Make-Your-3D（VG 04）：多视图扩散与 2D 先验**分布调和** |
| SDS 只管"像图像"，不管"像什么物理" | 生成物无物理行为 | Physics3D（VG 08）：视频扩散先验蒸给**物理参数**（可微仿真回路） |

**跨模态蒸馏的统一视角**：SDS 的本质是"**用预训练模型的评分函数（score）作为不可微目标的局部线性化**"。换掉评分函数就得到整个家族：

$$
\text{蒸馏信号} = \begin{cases} \varepsilon_\phi - \varepsilon & \text{SDS（图像扩散先验）} \\ \text{视频扩散先验的多帧版本} & \text{ReconX / Physics3D} \\ \text{偏好模型的 reward 梯度} & \text{DreamReward-X} \\ \text{教师模型的逐步输出} & \text{VideoScene 的 Leap Flow} \end{cases}
$$

---

## 5. 数值实验：SDS 梯度真的能"拉"分布吗？

设置：先验 = 单高斯 $\mathcal{N}((2,0), 0.6^2\mathbf{I})$（闭式 score，即"条件扩散模型"），把一个**可优化参数** $\mu\in\mathbb{R}^2$（代表 3D 场景参数，渲染恒等映射 $J = \mathbf{I}$）当作渲染图，按 boxed 公式更新

$$
\mu_{k+1} = \mu_k + \eta\, w(t)\big(\varepsilon - \varepsilon_\phi(x_t)\big)\Big|_{x_t = \sqrt{\bar\alpha_t}\mu_k + \sqrt{1-\bar\alpha_t}\varepsilon}, \qquad t \sim \mathcal{U}\big[t_{\min}, t_{\max}\big].
$$

理论预期可以**先算出来**：对高斯先验，$\mathbb{E}[\varepsilon-\varepsilon_\phi] = \sqrt{\bar\alpha_t}\sqrt{1-\bar\alpha_t}\,(\mu_c - \mu)/v_t$（$v_t = (1-\bar\alpha_t)+\bar\alpha_t\sigma^2$），所以期望更新方向恒指向先验均值 $\mu_c$，且强度 $\propto \bar\alpha_t$。

**实验结果**（$\eta=0.05$，300 步，30 个种子）：

```text
t ∈ [0.02, 0.98]（全区间）: 30/30 收敛到右模式，μ₁ = +2.06 ± 0.32（目标 +2.0）✓
t ∈ [0.80, 0.98]（高噪声）: μ₁ = +0.01 ± 0.00 —— 完全不动 ✗
t ∈ [0.02, 0.20]（低噪声）: μ₁ = +1.98 ± 0.33 —— 闭式 score 下照样收敛 ✓

单条轨迹（每 60 步采样）: +0.00 → +1.31 → +2.00 → +2.07 → +2.06 → +1.69
```

**三个读数**：

1. **SDS 是把参数拉向先验高密度区的力** ✓——期望动力学就是朝 $\mu_c$ 的收缩流，轨迹干净利落（300 步内从 0 拉到 2.06）。
2. **高噪声区间是死区**：信号强度 $\propto \bar\alpha_t$，$t\to1$ 时 $\bar\alpha_t\to0$——先验"看不清"你的渲染图（$x_t$ 几乎纯噪声，与 $\mu$ 无关），自然没有批评信号。**DreamFusion 把 $t$ 上限设在 0.98 就是在砍掉这段无用的计算**。（低噪声区间在闭式 score 下信号完好；真实学习到的 U-Net 在极低 $t$ 时行为不同——其去噪输出倾向于重建输入自身，信号质量下降——这是实践中也设下限 $t\approx0.02$ 的原因，toy 里看不到这一层。）
3. **收敛点有 $0.32$ 的散布而非精确到众数**——参数最终停在先验均值附近的小邻域，这个散布来自随机 $t$、随机 $\varepsilon$ 的梯度噪声。真实 SDS 的"过饱和/坍缩"病根（拉向众数而非分布）在这个确定性更强的 toy 里表现为**所有种子都收敛到同一个模式**（条件先验下没有多样性可言）——VG 01 §5.3 的 CFG 收缩与它是同一家族的病：**都在放大"指向高密度"的方向**。

**踩坑记录**（正文 §4.2 的预告）：第一版实现把符号写成 $\varepsilon_\phi - \varepsilon$，参数指数飞走（30 步内 $\mu_1$ 冲到 $-10^9$）；第二版没控制步长，同样发散。**SDS 公式的符号与权重是血泪坑**——DreamFusion 原文的"损失下降"约定与直觉上的"梯度上升"约定差一个符号，读不同论文实现时要先对齐约定再搬代码。

---

## 6. 与系列后续的连接

| 后续论文 | 用到本篇的哪个结论 |
|---------|------------------|
| VG 03 Unique3D | §4.3 表（SDS 病 → 前馈药方）；多视图扩散 = §2.3 时间注意力的多视角版 |
| VG 04 Make-Your-3D | §4.3（SDS 的多视角矛盾 → 分布调和） |
| VG 05 ReconX | §2.3（"视频一致性是软的"→ 3D 结构条件）；§3.2（3DGS 恢复 + 置信度加权） |
| VG 06 VideoScene | §3.2（3DGS 快，但卡在视频扩散慢 → 蒸馏动机）；§4.3（蒸馏家族） |
| VG 07 LangScene-X | §3.2（语言特征也要"贴"到 3DGS 表面——渲染方程的额外输出通道） |
| VG 08 Physics3D | §3.3（3DGS = 离散质点 → 携带物理属性）；§4.3（蒸馏信号 = 视频扩散先验） |
| VG 09 DreamReward-X | §4.3（蒸馏信号换成偏好） |
| VG 12 总结 | §4.3 的蒸馏家族表是核心素材 |

课程版前置：本篇 §2 的三部件（VAE / DiT / 时间注意力）是 SD3 工程栈（[MITS.6184 第 6 篇](../../MITS6184/MITS6184_6/)：U-Net → DiT → VAE）在视频侧的延伸——3D VAE 把压缩搬进时间维，时间注意力是视频扩散唯一的新部件。

---

## 7. 习题

**题 1**：补全 §4.2 第二步表中的 ②③④ 行：证明 $\nabla_{x_t}\log p_\phi(x_t\mid c) = -\frac{\mathbb E[\varepsilon\mid x_t,c]}{\sqrt{1-\bar\alpha_t}}$。
*提示*：② 写 $p_\phi(x_t\mid c) = \int p(x_t\mid x_0)\,p_\phi(x_0\mid c)\,dx_0$，其中 $p(x_t\mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)\mathbf{I})$；用 $\nabla\log\int = \mathbb{E}[\nabla\log\text{被积式}]$（对后验 $p_\phi(x_0\mid x_t,c)$ 取期望）。③ 里层梯度 $\nabla_{x_t}\log p(x_t\mid x_0) = -\frac{x_t - \sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}$（高斯二次式求梯度），代入后线性拆期望、取出 $\mathbb E[x_t\mid x_t]=x_t$。④ 分子用条件期望版恒等式 $x_t-\sqrt{\bar\alpha_t}\,\mathbb E[x_0\mid x_t,c]=\sqrt{1-\bar\alpha_t}\,\mathbb E[\varepsilon\mid x_t,c]$（VG 01 §4.3 折叠块），约去一个 $\sqrt{1-\bar\alpha_t}$；最后的「≈」一步换 $\mathbb E[\varepsilon\mid x_t,c]\to\varepsilon_\phi(x_t,t,c)$ 用的是「MSE 回归收敛到条件均值」（VG 01 §2.2/§2.4）。

**题 2**：证明 §5 的断言：对高斯先验 $\mathcal{N}(\mu_c, \sigma^2\mathbf{I})$，$\mathbb{E}[\varepsilon - \varepsilon_\phi(x_t)] = \sqrt{\bar\alpha_t}\sqrt{1-\bar\alpha_t}(\mu_c - \mu)/v_t$，从而 SDS 的期望更新强度 $\propto \bar\alpha_t$（高噪声死区的来源）。
*提示*：$\varepsilon_\phi(x_t) = \sqrt{1-\bar\alpha_t}(x_t - \sqrt{\bar\alpha_t}\mu_c)/v_t$，代入 $x_t$ 的表达式分两步取期望：先对 $\varepsilon$（$x_t$ 线性依赖它），再乘回 $w(t)$。

**题 3**：3DGS 的投影协方差 $\Sigma' = JW\Sigma W^\top J^\top$ 中，为什么需要 $J$（投影的雅可比）而不是直接用投影矩阵？什么时候这个近似失效？
*提示*：透视投影是非线性的，$J$ 是局部仿射近似；相机离高斯太近（投影非线性剧烈）时误差大——这是 3DGS 训练中"先收敛粗略位再细化"的原因之一。

**题 4**：比较 NeRF 的 $T_i$ 与 3DGS 的 $\prod_{j<i}(1-\alpha_j')$：两者是什么关系？证明它们是同一个透射率概念的离散化。
*提示*：$\exp(-\sum\sigma_j\delta_j) \approx \prod_j \exp(-\sigma_j\delta_j) \approx \prod_j (1-\alpha_j')$，其中 $\alpha_j' = 1-\exp(-\sigma_j\delta_j)$——泰勒展开的一阶近似。

**题 5**（思考）：ReconX 选择"视频扩散"而不是"图像扩散 + 多视角一致性损失"作为 3D 先验，本质优势在哪？
*提示*：§2.3——图像扩散对每个视角独立采样，一致性要靠外部损失硬拉（优化难）；视频扩散的时间注意力在**生成时**就耦合了所有帧——先验本身是"运动一致"的。代价是视频扩散模型的算力与控制难度。

---

*下一篇：[VG 03：Unique3D——单图到高质量 3D 网格](../VG_03_unique3d/)（多视图扩散 + 法线扩散 + ISOMER 网格重建的完整 pipeline；SDS 路线 vs 前馈路线的正面交锋）*
