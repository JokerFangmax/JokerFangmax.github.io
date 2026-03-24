# 预训练语言模型笔记：从 ELMo、BERT、GPT 到 Scaling Law

## 1. 引言

在理解了 Transformer 的基本结构之后，一个更关键的问题是：

> Transformer 为什么能成为大语言模型时代的统一基础架构？

这个问题的答案，不只在于 Transformer 本身的表达能力，还在于它非常适合**大规模预训练**。  
现代大语言模型的能力，并不是单靠某个特定任务训练出来的，而是在海量文本上进行预训练，再配合后续对齐和强化学习逐步形成的。

这一讲的重点可以概括为四条主线：

1. 大语言模型的核心训练目标是什么  
2. 语言模型通常经历哪些训练阶段  
3. 预训练模型的发展脉络是怎样的  
4. 为什么“规模化”会成为大模型时代最重要的方法论之一

---

## 2. 大语言模型的核心任务：Next Token Prediction

### 2.1 什么是 Next Token Prediction

大语言模型最核心的训练目标，通常可以概括成一句话：

> 给定前面的上下文，预测下一个 token。

也就是说，如果当前上下文是：

- `Tsinghua University is a`

模型的目标就是预测下一个最合理的 token，例如：

- `public`
- `university`

这种训练方式被称为 **next token prediction**，而逐个 token 生成文本的过程，则叫做 **autoregressive generation**。:contentReference[oaicite:1]{index=1}

---

### 2.2 为什么 Next Token Prediction 很重要

课件里强调了一个很核心的观点：

> 几乎所有自然语言任务，都可以统一为 next token prediction。

这是因为很多任务，本质上都可以写成“给定上下文，继续生成后续文本”的形式。例如：

#### 2.2.1 世界知识类任务

> The capital city of China is ___.

模型需要补出 `Beijing`。

#### 2.2.2 推理类任务

> If A then B, if B then not C. Now A = 1, so C = __.

模型需要完成逻辑推断。

#### 2.2.3 编程类任务

```python
def quick_sort(x: list) -> list:
    ____