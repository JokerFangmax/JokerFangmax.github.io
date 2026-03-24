---
title: "NLP Week 3: Transformer"
date: 2026-03-22
math: true
tags:
  - Notes
  - NLP
  - Transformer
categories:
  - Transformer
description: "NLP第三周课堂笔记"
---

# Transformer 笔记：从 Attention 到现代大模型基础架构

## 1. 引言

Transformer 是现代自然语言处理和大模型中最核心的基础架构之一。  
在它出现之前，序列建模主要依赖 RNN、LSTM、GRU 等递归神经网络；而 Transformer 的出现，几乎彻底改变了这一范式。

Transformer 最重要的思想并不是“设计了一个更复杂的网络”，而是：

> 用 attention 作为核心计算单元，替代 RNN 的递归式序列建模方式。

这使得模型能够：

- 更容易并行训练
- 更好地建模长距离依赖
- 更适合大规模扩展
- 成为后续 BERT、GPT、T5、LLaMA 等模型的基础

这篇笔记将从传统 Seq2Seq 的问题出发引出**第一章主题Attention**，逐步整理 Attention、Self-Attention，接着介绍其核心应用即**第二章的Transformer 结构**以及在**第三章介绍若干重要改进方法**。

---

## 2. Attention

## 2.1 从 Seq2Seq 到 Attention
#### 2.1.1 传统 Seq2Seq 的基本思路

在早期机器翻译等任务中，经典做法是 encoder-decoder 结构：

- **Encoder**：把输入序列逐步编码成隐藏状态
- **Decoder**：根据 encoder 的输出逐步生成目标序列

例如翻译任务中：

- 输入：`I love Beijing`
- 输出：`我 喜欢 北京`

常见方式是：

1. 编码器读完整个输入句子
2. 用最后一个隐藏状态表示整句语义  ***什么意思***
3. 解码器从这个固定向量中逐步生成输出

这种方式虽然自然，但有明显局限。

### 2.1.2 信息瓶颈问题

输入序列长度是可变的，但编码器最终输出给解码器的通常是一个**固定维度向量**。  
这意味着：

- 无论句子多长，最终都要压缩进一个固定大小表示
- 句子越长，信息越容易丢失
- 长距离依赖更难保留

也就是说，encoder 被迫做一件很困难的事：

> 把整个输入序列的全部关键信息压缩到单个向量里。

这就是经典 seq2seq 的 **information bottleneck**。

---

## 2.2. Attention 机制

### 2.2.1 Attention 的核心思想

Attention 的出发点非常直接：

> 解码器不应该只依赖一个固定长度向量，而应该能够在生成每一步时，动态访问输入序列中所有位置的信息。

也就是说，在生成目标句子的第 $t$ 个 token 时，模型可以查看输入序列中所有 token(***所有的K是吗***)，并自动决定当前最应该关注哪些位置(***这是由Softmax知道的吧***)。***插入示意图***

### 2.2.2 Attention 的一般形式

Attention 可以抽象为如下过程：

给定：

- 一个 **query**
- 一组 **keys**
- 一组 **values**

attention 会先用 query 和每个 key 计算相似度，再把这些相似度归一化成权重，最后对 values 做加权求和，得到输出。

其一般形式可以写成：

$
\text{Attention}(q, K, V) = \sum_i \alpha_i v_i
$

其中 $\alpha_i$ 表示 query 对第 $i$ 个 value 的注意力权重。

***详细解释QKV的计算流程，以及为什么除以根号d可以保持QK的方差稳定***

### 2.2.3 Query / Key / Value 的直观理解

可以把它理解成一个检索过程：

- **Query**：当前我想找什么信息
- **Key**：每条候选信息的索引
- **Value**：候选信息本身的内容

模型先判断 query 和每个 key 的匹配程度，再根据匹配程度对 value 加权，得到一个“与当前任务最相关的信息摘要”。

***PPT里的fixed size representation是什么意思***

### 2.2.4 Attention 的主要优势

#### 2.2.4.1 缓解固定向量瓶颈

解码器不再只依赖单个压缩向量，而是能直接访问整个输入序列的信息。

#### 2.2.4.2 更好建模长距离依赖
***Vanish Gradient Problem补充***
RNN 中远距离信息需要跨越很多时间步传播，而 attention 可以直接让当前位置与远距离位置建立联系。

#### 2.2.4.3 便于并行计算

RNN 是按时间步串行计算的，而 attention 的许多操作可以一次性矩阵化计算，更适合 GPU。

#### 2.2.4.4 更具可解释性

attention map 可以展示模型在当前时刻重点关注了哪些位置，因此具有一定可视化价值。

---

## 2.3. Self-Attention

### 2.3.1 为什么需要 Self-Attention

前面的 attention 更多是指 decoder 去关注 encoder 的输出。  
但如果我们想用 attention 完全替代 RNN(***即把QKV那一套用在Encoder和Decoder内部，现在更多的是Decoder的Q到Encoder的KV，而两部分实际还是RNN***)，仅仅建模“输入和输出之间的关系”还不够，还需要建模：

- 输入句子内部 token 之间的关系
- 输出句子内部 token 之间的关系

于是就有了 **self-attention**。

### 2.3.2 Self-Attention 的定义

> Self-attention 就是 Query、Key、Value 都来自同一个序列。

也就是说，一个 token 不仅可以“看输入”，还可以“看同一句子里的其他 token”。

例如句子：

> The animal didn’t cross the street because **it** was too tired.

模型在理解 `it` 时，可以通过 self-attention 去关注前面的 `animal`，从而学到指代关系。

### 2.3.3 Naive Self-Attention 的问题

如果在 decoder 中直接对整句目标序列做 self-attention，会出现一个问题：

> 当前 token 会看到未来 token。

而语言建模的目标是：

$
P(x_t \mid x_{< t})
$

也就是第 $t$ 个位置只能依赖前面的 token。  
如果第 $t$ 个 token 能直接看到第 $t+1$、$t+2$ 个 token，那预测就变成“偷看答案”了。

这就是 **information leakage**。PPT 中也明确指出，naive self-attention 对 encoder 没问题，但在 decoder 中会让 next-token prediction 变得 trivial。:contentReference[oaicite:3]{index=3}

### 2.3.4 Causal Mask

#### 2.3.4.1 为什么需要 Mask

为了解决 decoder 中 self-attention 的信息泄漏问题，需要引入 **causal mask**。

在第 $t$ 个位置：

- 可以看自己
- 可以看前面的 token
- 不能看后面的 token

#### 2.3.4.2 Mask 的形式

对于长度为 4 的序列，mask 可以写成：

$
\begin{bmatrix}
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1
\end{bmatrix}
$

这样 attention 只能使用当前及历史信息，从而满足自回归生成的因果约束。

### 2.3.5 Self-Attention 与 Cross-Attention 的区别

#### 2.3.5.1 Self-Attention

- Query 来自当前序列
- Key 来自当前序列
- Value 来自当前序列

作用是建模序列内部依赖。

#### 2.3.5.2 Cross-Attention

- Query 来自 decoder
- Key 来自 encoder
- Value 来自 encoder

作用是让 decoder 在生成过程中访问源序列信息。PPT 中也强调，正是 self-attention 和 cross-attention 的结合，使得 RNN 可以被完全替代。:contentReference[oaicite:4]{index=4}

---

## 3 Transformer
## 3.1 Transformer 的提出

### 3.1.1 RNN 的并行化瓶颈

即使 attention 已经能帮助建模，如果整体框架还是 RNN，那么依然存在一个核心问题：

> RNN 的计算是沿时间维串行展开的。

这意味着第 $t$ 步必须等第 $t-1$ 步计算完成。  
这种结构天然不适合现代 GPU 的大规模并行。PPT 中也指出，RNN 的 sequential computation prevents parallelization。:contentReference[oaicite:5]{index=5}

### 3.1.2 Transformer 的关键突破

2017 年，Vaswani 等人在论文 *Attention Is All You Need* 中提出：

> 完全使用 attention 来建模序列，不再依赖 RNN。

这就是 Transformer。:contentReference[oaicite:6]{index=6}

### 3.1.3 Transformer 的总体意义

Transformer 的重要性在于：

- 用 attention 完成 token 间的信息交互
- 允许整个序列并行计算
- 更适合大规模训练和扩展

---

## 3.2 Transformer 的输入表示

### 3.2.1 输入层的两个核心问题

Transformer 接收的不是原始字符串，而是 token embedding 序列。  
因此输入层至少包含两个关键问题：

1. 怎么切 token
2. 怎么表示位置信息

### 3.2.2 Tokenization

#### 3.2.2.1 为什么不能直接按单词切分

最直观的方法是：

- 根据空格切分单词
- 每个单词作为一个 token
- 在训练语料上建立词表

但这样会遇到一个问题：**OOV（Out-of-Vocabulary）**。

如果测试时出现训练词表中不存在的词，就无法直接编码。  
PPT 中举的例子是：

- 训练集中有 `low`
- 训练集中有 `highest`
- 但新语料中出现了 `lowest`

如果按整词建表，`lowest` 就可能成为未知词。:contentReference[oaicite:7]{index=7}

#### 3.2.2.2 子词级别建模

为了解决 OOV 问题，更合理的方法是：

> 不把 token 固定成整词，而是构建子词（subword）词表。

这样：

- 常见词可以作为整体出现
- 生僻词可以拆成多个子词
- 未见过的新词也能由已有子词组合出来

#### 3.2.2.3 Byte Pair Encoding（BPE）

PPT 重点介绍了 BPE。:contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}

BPE 的大致流程是：

1. 初始词表由字符组成
2. 统计训练语料中最常见的相邻字符对
3. 把最频繁的 pair 合并成一个新 token
4. 重复这个过程，逐步形成更大的子词单位

#### 3.2.2.4 BPE 例子

假设训练语料中有：

- low
- lower
- newest
- widest

一开始词表可能只有字符：

- l, o, w, e, r, n, s, t, i, d

然后不断合并高频 pair，最终可能形成：

- `low`
- `est`

于是即使 `lowest` 没在训练集中完整出现过，也可以被切分成：

- `low`
- `est`

这就是 BPE 解决 OOV 的关键思想。

### 3.2.3 Position Encoding

#### 3.2.3.1 为什么必须有位置编码

Attention 本身只关注 token 与 token 之间的两两关系。  
如果不给位置信息，它并不知道：

- 哪个 token 在前
- 哪个 token 在后
- 两个 token 的距离多远

因此 Transformer 必须显式加入位置信息。

#### 3.2.3.2 Sinusoidal Position Encoding

PPT 中介绍的是经典的正弦位置编码。:contentReference[oaicite:10]{index=10}

其定义为：

$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$

$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$

其中：

- $pos$ 表示位置
- $i$ 表示维度索引
- $d$ 表示 embedding 维度

#### 3.2.3.3 为什么用正弦和余弦

这种设计的好处在于：

1. 不同位置会得到不同编码
2. 不同维度对应不同频率
3. 模型更容易从中学习相对位置信息

可以把它理解成：每个位置都被映射成一组多频率波形上的坐标，从而在高维空间中可区分。

---

## 3.3 Transformer 的核心结构

### 3.3.1 Transformer Block 的组成

一个标准 Transformer block 通常由以下部分组成：

1. Multi-Head Attention
2. Feed-Forward Network
3. Residual Connection
4. Layer Normalization

PPT 中在 encoder layer 部分也明确给出了这一结构。:contentReference[oaicite:11]{index=11}

### 3.3.2 Self-Attention 的计算过程

#### 3.3.2.1 线性映射得到 Q、K、V

对于输入表示 $X$，先通过线性映射得到：

$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$

也就是说，每个 token 会被映射成三种不同角色的表示：

- Query：当前 token 想找什么
- Key：当前 token 提供什么索引信息
- Value：当前 token 真正携带的信息

#### 3.3.2.2 Scaled Dot-Product Attention

Transformer 中最经典的 attention 形式是：

$
\text{Attention}(Q, K, V)
=
\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$

其中：

- $QK^\top$ 表示相关性分数
- 除以 $\sqrt{d_k}$ 是为了避免点积过大导致 softmax 过于尖锐
- softmax 把分数转成权重
- 最后再对 $V$ 做加权求和

### 3.3.3 Multi-Head Attention

#### 3.3.3.1 定义

如果只有一个 attention head，模型只能从一种视角学习关系。  
Transformer 采用的是 **Multi-Head Attention**：

$
\text{MHA}(Q,K,V)=\text{Concat}(head_1,\dots,head_h)W_O
$

其中每个 head 都独立做一次 attention：

$
head_i = \text{Attention}(Q_i, K_i, V_i)
$

#### 3.3.3.2 为什么需要多头

不同 head 可以关注不同类型的信息，例如：

- 局部邻近关系
- 长距离依赖
- 语法关系
- 共指与修饰关系

因此多头机制能显著增强表示能力。

### 3.3.4 Feed-Forward Network（FFN）

#### 3.3.4.1 FFN 的形式

在 attention 之后，每个 token 还会通过一个前馈网络：

$
\text{FFN}(x) = W_2 \sigma(W_1x+b_1)+b_2
$

#### 3.3.4.2 FFN 的作用

FFN 有两个重要特点：

1. 它是逐位置独立应用的
2. 它不负责 token 间交互，只负责对每个 token 的特征做进一步非线性变换

可以简单理解为：

- attention 负责信息交换
- FFN 负责特征加工

### 3.3.5 Residual Connection 与 Layer Normalization

#### 3.3.5.1 Residual Connection

残差连接的形式通常是：

$
x + \text{Sublayer}(x)
$

作用是让信息与梯度更容易传播，减轻深层网络训练困难。

#### 3.3.5.2 Layer Normalization

LayerNorm 会在特征维度上做归一化，使训练更稳定。  
原始 Transformer 中，attention 和 FFN 两个子层都配有残差连接和 layer normalization。:contentReference[oaicite:12]{index=12}

---

## 3.4 Encoder 与 Decoder

### 3.4.1 Encoder Layer

Encoder layer 包含：

1. Self-Attention
2. FFN

这里的 self-attention 不需要 mask，因为输入序列整体都是已知的。

### 3.4.2 Decoder Layer

Decoder layer 和 encoder 类似，但有两个关键变化。:contentReference[oaicite:13]{index=13}

#### 3.4.2.1 Masked Self-Attention

由于 decoder 是自回归生成的，当前位置不能看到未来 token，因此必须加 causal mask。

#### 3.4.2.2 Cross-Attention

decoder 还需要访问 encoder 的输出，因此在 masked self-attention 之后，还会加入一个 cross-attention 层：

- Query 来自 decoder
- Key / Value 来自 encoder

### 3.4.3 输出预测

经过多层 decoder 后，每个位置会得到一个隐藏表示。  
然后通过线性层映射到词表维度，再接 softmax，得到下一个 token 的概率分布：

$
P(y_t \mid y_{<t}, x)
$

这就是标准的自回归生成过程。

---

## 3.5 Transformer 的优势与局限

### 3.5.1 Transformer 的优势

#### 3.5.1.1 更适合并行

RNN 必须逐时间步计算，而 Transformer 可以一次性处理整段序列的表示更新。

#### 3.5.1.2 更容易建模长距离依赖

attention 可以直接连接任意两个位置，不必像 RNN 那样经过多步传递。

#### 3.5.1.3 更适合大规模扩展

Transformer 的核心是矩阵乘法、归一化和逐位置前馈网络，非常适合 GPU / TPU。

#### 3.5.1.4 表达能力强

多头 attention 让模型能从多个子空间学习不同关系，因此通常比传统递归结构更强。  
PPT 中也提到，Transformer 在机器翻译上优于先前架构，同时训练 FLOPs 更少。:contentReference[oaicite:14]{index=14}

### 3.5.2 Transformer 的局限

PPT 中指出：

- architecture is hard to optimize
- sensitive to model modifications

也就是说，Transformer 虽然强大，但训练并不总是稳定，而且对结构改动比较敏感。:contentReference[oaicite:15]{index=15}

此外，标准 self-attention 的时间和空间复杂度通常是序列长度 $n$ 的平方级：

$
O(n^2)
$

这在长上下文场景下会成为瓶颈。

---

## 4. Transformer 的若干重要改进

### 4.1 Rotary Position Embedding（RoPE）

#### 4.1.1 基本思想

传统位置编码通常是把位置向量加到 token embedding 上。  
RoPE 的思路不同：

> 把位置信息直接融入 Query 和 Key 的变换中。

PPT 中列出了 relative position encoding、RoPE、高维 attention 中的 generalized RoPE，以及 RoPE 的成功应用。:contentReference[oaicite:16]{index=16}

#### 4.1.2 为什么 RoPE 重要

可以把它理解为：

- 普通位置编码：把位置信息附加在输入表示上
- RoPE：把位置信息写进 attention 的相似度计算规则中

因此它通常更适合建模相对位置关系，也是现代大模型中非常常见的位置编码方式。

### 4.2 Gated Linear Units（GLU）

#### 4.2.1 核心思想

PPT 中提到 GLU 及其变体能改进 Transformer。:contentReference[oaicite:17]{index=17}

普通 FFN 大致是：

$
\text{FFN}(x)=W_2\sigma(W_1x)
$

而 GLU 会引入门控结构，大致可以理解为：

$
\text{GLU}(x)=A(x)\odot \sigma(B(x))
$

其中：

- $A(x)$ 给出候选特征
- $B(x)$ 生成门值
- $\odot$ 表示逐元素乘法

#### 4.2.2 GLU 的意义

门控机制使模型可以学习：

- 哪些信息应该通过
- 哪些信息应该被抑制

因此通常能提升表达能力和实际性能。现代模型里的 SwiGLU、GEGLU 都属于这一类思路。

### 4.3 Grouped-Query Attention（GQA）

#### 4.3.1 MHA、MQA、GQA 的区别

PPT 中提到了 GQA 及其与 MHA、MQA 的对比。:contentReference[oaicite:18]{index=18}

- **MHA**：每个 head 都有独立的 Q/K/V，表达能力强，但开销大
- **MQA**：多个 query head 共享同一组 key/value，推理更省资源
- **GQA**：若干个 query head 共享一组 key/value，在二者之间折中

#### 4.3.2 GQA 的意义

GQA 的核心价值是：

> 在推理效率和模型性能之间取得更平衡的折中。

这也是为什么它在现代大模型推理中非常重要。

### 4.4 Pre-LN 与 RMSNorm

#### 4.4.1 Pre-LN

PPT 中指出，原始 Transformer 使用的是 **Post-LN**，而后续研究发现 **Pre-LN** 往往更稳定。:contentReference[oaicite:19]{index=19}

- Post-LN：
$
\text{LN}(x + \text{Sublayer}(x))
$

- Pre-LN：
$
x + \text{Sublayer}(\text{LN}(x))
$

Pre-LN 的优势在于每个子层的输入先被规范化，因此梯度传播更稳定。

#### 4.4.2 RMSNorm

RMSNorm 是对 LayerNorm 的进一步简化。PPT 中也提到了 Improved Normalization 与 RMSNorm。:contentReference[oaicite:20]{index=20}

其直观形式可以写成：

$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}
$

与 LayerNorm 相比，RMSNorm 更简单，也常被认为在大模型中更高效、更稳定。

---

## 5. 总结

### 5.1 知识主线回顾

把整份内容串起来，可以得到 Transformer 的一条清晰发展逻辑：

#### 5.1.1 Seq2Seq 遇到瓶颈

RNN-based encoder-decoder 需要把整个输入压缩成单个向量，造成信息瓶颈。

#### 5.1.2 Attention 被提出

生成每一步时，不再只看一个固定向量，而是动态访问整段输入。

#### 5.1.3 Self-Attention 被引入

不仅能建模输入与输出之间的关系，还能建模序列内部 token 之间的依赖。

#### 5.1.4 Transformer 诞生

用纯 attention 替代 RNN，实现并行的序列建模框架。

#### 5.1.5 后续不断改进

为了更稳定、更高效、更适合长上下文和大规模推理，又发展出：

- RoPE
- GLU / SwiGLU
- GQA
- Pre-LN
- RMSNorm

### 5.2 一句话理解 Transformer

> Transformer 的本质，是用 attention 取代递归，把“按时间步传递信息”改成“全局两两交互建模”，从而为现代大模型奠定了基础。