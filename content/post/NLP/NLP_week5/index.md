---
title: "NLP Week 5: Context Engineering"
date: 2026-05-15
math: true
tags:
  - Notes
  - NLP
  - LLM
categories:
  - NLP Notes
description: "NLP第五周课堂笔记"
---


# NLP Week 5 — Context Engineering Lecture Notes

> 本文根据课件 **NLP 2026 w5 - Context Engineering** 整理，目标是把原本偏课堂提纲式的 slides 重写成适合直接放到 blog 上的 lecture notes。整体不按页逐条翻译，而是围绕一条更清晰的主线展开：**chat template 与 base model 的区别、in-context learning 与 CoT、解码参数与 reasoning/tool use、API 成本与 prefix caching、以及 context engineering 的核心思想与最佳实践**。

---

## 1. 这份课件真正想讲什么？

从标题上看，这节课叫 **Context Engineering**，但实际上它覆盖了三块紧密相关的内容：

1. **Prompt / Chat 基础机制**：LLM 到底是如何理解聊天输入的；
2. **Decoding 与推理能力扩展**：为什么不同任务要用不同的解码参数、reasoning、tool use；
3. **Context Engineering**：当上下文变长、系统变复杂时，如何管理 context 才能兼顾性能、成本与效率。

也就是说，这节课并不只是告诉你“怎么写 prompt”，而是在讲一个更深的问题：

> **LLM 的输入上下文本质上就是它当前可见的全部工作记忆，因此，如何构造、压缩、缓存、组织这段上下文，会直接决定模型的质量、速度和价格。**

课件第一页的 goal 也很明确：它想让你理解：

- context window 到底由什么组成；
- 如何把预训练 LLM 用到下游任务；
- 各种 decoding 参数是什么意思；
- API 成本如何计算；
- context engineering 的最佳实践是什么。

所以这节课的真正主线是：

> **从“LLM 看到了什么”出发，一直讲到“如何工程化地管理这些可见内容”。**

---

## 2. 什么是 Prompt Engineering？为什么它不只是“换个说法”？

课件第 4 页先给出 prompt engineering 的定义：

> Prompt engineering is the discipline of developing and optimizing prompts to efficiently utilize LLMs.

同时也强调：prompt 不只是“问题文本”，它也可以叫：

- input
- user message
- context
- query

而且 prompt engineering 远不只是 rephrasing。

### 2.1 为什么“prompt engineering 不只是改写句子”？

很多初学者会把 prompt engineering 理解成：

- 把问题写得更礼貌；
- 加一点限定词；
- 把句子换个说法。

但课件这里特别强调，真正的 prompt engineering 需要理解 LLM 的底层工作方式。因为 prompt 影响的不只是表面措辞，还会影响：

- chat template 如何展开成 token 序列；
- 模型对结构的识别；
- demonstration 是否被正确利用；
- 解码空间是否被约束；
- 成本、延迟、缓存命中率；
- reasoning 与 tool use 是否顺利工作。

所以 prompt engineering 的真正对象其实不是“语言表面”，而是：

> **模型在当前输入序列上执行 next-token prediction 时，究竟会如何解释、组织和利用这些上下文信息。**

这也是为什么这节课后半段自然过渡到了 context engineering：prompt 只是上下文管理中的一个局部问题。

---

## 3. Chat Template：聊天模型到底如何理解“消息结构”？

课件第 5–10 页讲的是一个非常基础但又极其重要的问题：

> **聊天模型看到的并不是“一个 Python 字典列表”，而是最终展开成的一串文本 token。**

### 3.1 聊天界面只是表象，模型真正看到的是字符串

用户平时和 ChatGPT 一类系统交互时，看见的是：

- 一轮轮 user / assistant message；
- 甚至还有 system prompt、tool message 等角色。

但模型本身并不直接“懂”这些 JSON / Python 数据结构。课件第 6–7 页说明，聊天历史在进入模型前，会经过 `apply_chat_template` 被转换成一个字符串，例如：

- `<|im_start|>user ... <|im_end|>`
- `<|im_start|>assistant ... <|im_end|>`

最终再追加一个 assistant 起始标记，让模型继续生成直到输出 end token。

这意味着：

> **所谓“聊天”在模型内部，本质上仍然只是文本续写。**

只不过这段文本经过了某种带结构标记的模板化包装。

### 3.2 为什么 chat template 很重要？

课件第 10 页明确说：chat templates are model-specific。

这件事的重要性非常大：

- 不同模型在 SFT 阶段见到的模板不同；
- 它们学会的是某种具体的“消息格式”；
- 如果你给错模板，即使模型本体能力很强，也可能输出质量明显下降。

也就是说，模型并不是天然理解“user”“assistant”“system”这些角色，而是它在后训练中学会了：

- 某些特殊 token 或格式意味着用户输入；
- 某些格式意味着 assistant 回复区域；
- 某些格式意味着系统指令优先。

### 3.3 System Prompt 为什么常常优先级更高？

课件第 9 页提到，在多轮对话中还可以添加 system prompt，而系统提示里的要求通常在与用户 prompt 冲突时优先。

这里的“优先”不是硬编码规则，而是训练分布导致的：

- 在后训练数据中，system message 往往扮演更高层级的行为约束；
- 所以模型学会了更强地遵循它。

这也是为什么现代 agent / tool-use / product systems 都会大量依赖 system prompt：

> **它是整个上下文中最适合放“全局行为规则”的位置。**

---

## 4. 为什么 Base Model 不懂 Chat Template？

课件第 10–12 页强调了一个很容易被忽略的事实：

> **Base model 从未见过 chat templates。**

### 4.1 Base Model 的本质：纯文本续写器

在 SFT 之前，base model 的训练目标通常只是 next-token prediction。它见到的是大量自然文本，而不是：

- `<|im_start|>`
- `<|assistant|>`
- `{"role": "user"}` 这种结构化对话格式。

所以对于 base model 来说，它并不会把 chat template 看成“消息结构”，而只是把它们当作一些怪异字符串。

这就是课件第 10 页那句很重要的话：

> LLMs only see chat templates in the post-training (supervised fine-tuning, SFT) process.

### 4.2 直接拿 base model 做聊天为什么容易变成 gibberish？

因为它没有学过这种模板对应的行为模式。你给它一串 `<|im_start|>user ...`，它只会把这看成普通文本的一部分，然后做机械续写，而不是“进入 assistant 回复模式”。

所以：

- **chat model** = base model + SFT / alignment 后学会了 chat template；
- **base model** = 还只是一个文本续写器。

这是理解现代 LLM 行为差异的基础。

---

## 5. Base Model 仍然有什么用？

虽然 base model 不适合直接聊天，但课件第 11–14 页说明，它仍然很有价值。

### 5.1 把任务改写成文本续写

因为很多任务本质上都可以写成 completion：

- 问答：
  - `Question: ... Answer:`
- 翻译：
  - `English: ... Chinese:`
- 分类：
  - `Text: ... Label:`

只要 prompt 的格式合理，base model 仍然能完成很多任务。

### 5.2 多项选择题与条件似然

课件第 13–14 页讲了一个特别重要的用法：

> **对于 base LLM，很多 benchmark 评估并不是“让它直接生成答案”，而是比较不同候选答案的条件似然。**

例如一道多选题：

- 题干 + 选项都作为上下文；
- 分别计算候选 A、B、C、D 的 token likelihood；
- 选平均 token likelihood 最大的答案。

这类做法常用于 zero-shot evaluation，例如 MMLU、ARC、BoolQ 等。fileciteturn10file0

### 5.3 为什么要除以 token 数？

课件提到通常会对 likelihood 做长度归一化，也就是除以 token 数，避免模型天然偏爱更短候选。fileciteturn10file0

这说明评估 base model 并不是简单粗暴地“看它生成了什么”，而是：

> **利用它作为概率模型的本质，直接比较不同答案的条件概率。**

这和 chat model 的直接 generation 评估其实是两套不同范式。

---

## 6. In-Context Learning：为什么 few-shot 示例能“教会”模型？

课件第 15–18 页进入 **in-context learning（ICL）**。

### 6.1 核心思想

ICL 的做法是：

- 把若干带标注示例直接放进 prompt；
- 再在最后给一个新样本；
- 让模型根据前面的 pattern 推断应该怎么回答。

也就是说，不更新参数，只更新上下文。

课件把这些示例叫做：

- demonstrations
- exemplars
- in-context examples。fileciteturn10file0

### 6.2 为什么它有效？

GPT-3 之后，人们发现大模型可以在上下文中“临时适应”任务格式。这并不是说模型真的在内部做了梯度下降，而是：

- 它从 prompt 中识别出任务模式；
- 从示例中读出输入输出对应关系；
- 再把这种模式延续到最后一个 query。

所以 ICL 可以被看作一种“上下文内任务对齐”。

### 6.3 ICL vs Fine-Tuning

课件第 18 页列了 ICL 和 fine-tuning 的优缺点。关键点是：

#### ICL 的优点
- 简单；
- 不需要更新模型参数；
- 可直接通过 API 使用；
- 在 few-shot 场景下有时效果很好。

#### ICL 的缺点
- inference 成本高，因为示例每次都要重复放进上下文；
- 上下文变长会增加 token 成本；
- 某些任务上不如 fine-tuning 稳定。fileciteturn10file0

这也是 context engineering 会自然关心 ICL 的原因：

> **ICL 的“学习”发生在 context 里，因此它的效果和成本都直接取决于 context 的组织方式。**

---

## 7. Prompt Format Sensitivity：为什么一个空格都可能毁掉结果？

课件第 19–20 页特别强调：

> **LLMs are very sensitive to prompt format.**

有时一个多余空格、一个少掉的换行，甚至会让准确率掉到接近 0。fileciteturn10file0

### 7.1 为什么格式会这么敏感？

因为模型不是在理解一个“抽象任务定义”，而是在拟合训练分布中的 token pattern。

如果 prompt 格式和它训练中见过的高质量模式更接近，那么：

- 示例边界更清楚；
- 任务信号更强；
- 结构更容易被识别。

反之，如果格式混乱：

- 模型可能误判输入 / 输出边界；
- 把本来该预测的部分当作上下文；
- 甚至误解整个任务。

### 7.2 对评估意味着什么？

课件第 20 页给出的建议很实用：

- 尽量使用官方 prompt format；
- 使用通行的评估框架（如 LM-Eval、OpenCompass 等）；
- 不要用相同任务直接比较 base model 和 chat model。fileciteturn10file0

这背后的原因是：

> **prompt 本身就是评估协议的一部分。**

如果协议不一致，分数对比就不公平。

---

## 8. Chain-of-Thought：为什么“把思考过程写出来”会变强？

课件第 21–24 页讲了 CoT prompting。

### 8.1 基本想法

CoT 的核心不是让模型“更聪明”，而是让它：

> **把原本可能隐式进行的多步推理，显式展开成中间步骤。**

课件引用了 *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*，说明这种方法能显著提升 reasoning 任务表现。fileciteturn10file0

### 8.2 Zero-shot CoT

更有意思的是课件第 22 页讲的 zero-shot CoT：

- 只需要在 prompt 中加上 “Let’s think step by step”
- 对足够大的模型，就可能显著提升推理表现。fileciteturn10file0

这说明：

- 某些推理能力模型本来就有；
- 只是普通 prompt 没有把这种能力激活出来；
- CoT 其实是一种“行为诱导”。

### 8.3 APE：连 prompt 也可以自动优化

课件第 23–24 页提到 APE（Automatic Prompt Engineer），指出 LLM 甚至可以自动搜索更有效的 prompt instruction。fileciteturn10file0

这反过来也支持了一个观点：

> **prompt engineering 并不是简单的语言修辞，而是一个可以被系统搜索和优化的工程问题。**

---

## 9. Decoding：为什么生成本质上是一个搜索问题？

从第 27 页开始，课件进入 decoding。

它先给出一个非常本质的视角：

> **LLM decoding is a search process.**fileciteturn10file0

### 9.1 为什么生成是搜索？

因为在每个时间步，模型都会输出一个整个词表上的概率分布。生成一个长度为 \(n\) 的序列，理论上相当于要在巨大的 token 组合空间里找到高概率路径。

但暴力穷举所有路径显然不可能，所以 decoding 的问题就是：

- 如何在这个组合空间里找到“好”的输出；
- 又要兼顾效率、质量、多样性。

这就是 greedy、beam、sampling 等策略存在的原因。

---

## 10. Greedy、Beam、Top-k、Top-p：这些解码策略到底在权衡什么？

### 10.1 Greedy Search

最简单的方法是 greedy：每一步都选当前概率最大的 token。

优点：
- 简单；
- 快；
- 输出稳定。

缺点：
- 容易陷入局部最优；
- 经常过于保守；
- 可能导致重复或次优序列。fileciteturn10file0

### 10.2 Beam Search

课件第 28 页说明，beam search 会保留 top-k 条候选路径，而不是只保留一条。这样它能跳出 greedy 的局部选择失误。fileciteturn10file0

beam 的本质是：

> **用更大的搜索宽度，换取更高的序列整体概率。**

但它的缺点是：

- 计算更贵；
- 对开放式生成不一定更自然；
- 仍然偏 deterministic。

### 10.3 Top-k Sampling

课件第 29–30 页讲到 sampling：不再总是拿最大概率 token，而是按概率随机采样。

Top-k sampling 会：

- 只保留概率最高的 k 个 token；
- 把其他 token 概率置零；
- 再在这 k 个里采样。fileciteturn10file0

这样做的效果是：

- 输出更丰富；
- 但也更不稳定。

### 10.4 Top-p（Nucleus）Sampling

Top-p sampling 则不是固定保留 k 个 token，而是保留累计概率超过某阈值 \(p\) 的最小 token 集合。fileciteturn10file0

它通常比 top-k 更自适应：

- 分布很尖锐时，保留少数 token；
- 分布较平缓时，保留更多 token。

因此它往往是实践中很常见的选择。

### 10.5 这些方法到底在平衡什么？

如果把这些 decoding 方法放在一条轴上，大致是在平衡：

- **确定性 / 可控性**
- **多样性 / 创造性**

事实型问答通常更偏前者；
创意写作、诗歌、头脑风暴则更偏后者。

---

## 11. 最重要的解码参数：temperature、top-k、top-p、max tokens

课件第 32–33 页列出了最常用的参数。

### 11.1 Temperature

Temperature 会改变概率分布的尖锐程度：

- 温度低：分布更尖锐，更确定；
- 温度高：分布更平缓，更随机。fileciteturn10file0

因此：

- factual QA 通常适合低温；
- creative generation 通常可以提高一点温度。

### 11.2 top-k / top-p

它们控制采样空间的大小：

- \(k\) 或 \(p\) 增大，随机性增强；
- \(k\) 或 \(p\) 减小，生成更保守。

### 11.3 max_new_tokens

这个参数控制模型最多生成多少 token。它不仅影响输出长度，也直接影响：

- 成本；
- 延迟；
- reasoning token 开销。

### 11.4 stop sequences

课件第 33 页讲 stop sequences：当模型生成到某个字符串时，提前终止生成。fileciteturn10file0

这在结构化输出、列表生成、agent 交互中都非常有用，因为它可以把开放生成变成带边界的生成。

### 11.5 repetition / frequency penalty

它们用于抑制重复：

- repetition penalty：对已出现 token 降权；
- frequency penalty：出现越多降得越多。fileciteturn10file0

这些参数的意义在于：

> **解码策略本身也是对模型行为的一种“后验控制”。**

你没有改模型参数，但你改了它在搜索空间里怎么挑答案。

---

## 12. Structured Outputs：为什么“给我 JSON”经常不够？

课件第 34 页提到，在很多应用里，我们希望输出严格遵守某种格式，比如 JSON。但只靠 prompt 写“please output JSON”往往不够稳定。fileciteturn10file0

### 12.1 为什么纯 prompt 不够？

因为 prompt 只能“鼓励”模型输出 JSON，却不能硬性禁止非法 token。模型仍然可能：

- 少一个括号；
- 混入自然语言解释；
- 输出字段名不一致；
- JSON 后再加别的内容。

### 12.2 结构化输出 API 的本质

课件说 OpenAI 的 structured outputs 是通过：

> zeroing the probability of tokens that do not adhere to the specified format.

也就是说，它不是单纯“提醒模型”，而是在解码层面直接约束非法 token 不可生成。fileciteturn10file0

这体现了一个非常重要的思想：

> **prompt engineering 与 decoding control 可以结合，前者提供语义目标，后者提供硬约束。**

---

## 13. Reasoning Models：为什么现在 chat template 里还会多一层“思考”？

课件第 35–37 页进入 reasoning models。

### 13.1 现代 reasoning model 的新结构

很多 state-of-the-art 模型在输出最终答案前，会先生成 reasoning tokens，也就是内部“思考过程”。这会让 chat template 比普通 chat 模型多出一层结构。fileciteturn10file0

也就是说，现代聊天不再只是：

- 用户消息；
- 助手回答；

而可能变成：

- 用户消息；
- 助手隐藏 reasoning；
- 助手最终回答。

### 13.2 为什么后续轮次通常不保留 reasoning tokens？

课件第 36 页明确说，多数模型不会把 reasoning tokens 放进后续轮次上下文。原因有两个：

1. 数据工程更简单；
2. 节省 context 长度。fileciteturn10file0

这件事很重要，因为它说明：

> **context engineering 的一个核心原则，就是不是所有生成内容都值得永久留在上下文里。**

reasoning 可能对当前轮有用，但对后续轮未必值得保留。

### 13.3 reasoning effort / budget

课件第 37 页提到，一些 API 支持指定 reasoning effort，例如：

```python
reasoning={"effort": "low"}
```

这意味着 reasoning 不只是模型内部隐式行为，还变成了一个可调的成本—质量参数。fileciteturn10file0

换句话说：

- 高 reasoning effort：可能更强，但更贵、更慢；
- 低 reasoning effort：更便宜、更快，但未必够强。

这再次把 prompt / decoding / API cost / context engineering 连起来了。

---

## 14. Tool Use：工具调用为什么本质上也是上下文工程？

课件第 38–41 页讲 tool use。

### 14.1 工具调用是怎么启用的？

方法非常直接：把工具描述（tool descriptions）作为字符串加入 prompt，通常放在 system prompt 附近。模型读完这些描述后，决定是否调用工具。fileciteturn10file0

这说明：

> **工具不是“外挂 API magically 生效”，而是先被写进上下文，成为模型可见的行动空间描述。**

### 14.2 Tool call 本质上是什么？

课件第 41 页说明，模型会生成带特殊标记的 tool call，开发者解析这段输出，执行函数，再把结果作为 tool message 回填给模型。fileciteturn10file0

这本质上是在做一个循环：

1. 把工具说明放进 context；
2. 模型基于 context 生成工具调用；
3. 外部执行工具；
4. 再把工具结果放回 context；
5. 模型继续基于更新后的 context 回答。

所以 tool use 可以被看成 context engineering 的一种特例：

> **上下文不仅包含“语言信息”，还包含“可行动接口”的描述和结果。**

---

## 15. Multimodal Inputs：为什么图片不能简单拼进字符串？

课件第 42 页提到，多模态输入通常通过在 message content 中放 image URL 来实现，但这类对话不能再被简单转换为纯字符串，因为图像会先经过视觉编码器。fileciteturn10file0

这说明：

- 文本上下文最终还是 token 序列；
- 图像上下文则要先变成另一种神经表示；
- 多模态 context 已经不再是“纯文本拼接”这么简单。

从 context engineering 的角度看，这意味着未来的上下文管理不只是“删减文本”，还包括：

- 图片放不放；
- 哪些图片保留；
- 图片摘要是否代替原图；
- 多模态证据如何压缩。

这也是 context engineering 会越来越复杂的一个原因。

---

## 16. API 成本：为什么上下文管理直接影响花多少钱？

课件第 45–50 页进入一个非常工程化但极重要的话题：**API pricing**。

### 16.1 为什么 LLM 成本和 context 绑定得这么紧？

因为 API 计费往往与：

- 输入 token 数；
- 输出 token 数；
- reasoning token 数；
- cache hit 与否；

直接相关。课件第 47 页甚至专门举例计算：

- prompt 50K token；
- 输出 10K token；
- 前 40K 输入命中 prefix cache；
- 则调用 GPT-5.4 的总成本按不同单价加总计算。fileciteturn10file0

这页的真正重点不是某个价格数字，而是：

> **context engineering 不是抽象优化，而是会直接决定你烧多少钱。**

### 16.2 reasoning token 为什么也重要？

课件提醒：有些 API 不会把 reasoning tokens 返回给用户，但这些 token 依然计费。fileciteturn10file0

这意味着：

- 你看不到，不代表没花钱；
- 深度 reasoning 模型的真实输出成本，往往高于表面可见文本。

这对 agent / pipeline 设计尤其重要。

---

## 17. Prefix Caching：为什么“公共前缀”是大规模系统的关键？

课件第 46、49、50 页专门讲了 prefix caching。

### 17.1 什么是 prefix caching？

如果多个请求共享一段长前缀，那么系统可以缓存这段前缀的中间表示，下次遇到相同前缀时直接复用，而不用重新编码。fileciteturn10file0

这会带来两个直接好处：

1. **降低成本**：重复前缀不用重复付全价；
2. **降低 TTFT（time to first token）**：模型更快开始输出。

### 17.2 为什么它和 context engineering 关系这么大？

因为 prefix caching 的命中率强烈依赖于你的上下文怎么组织。

如果你每次都：

- 改 system prompt 的顺序；
- 动态删减工具描述；
- 把稳定内容和临时内容混在一起；

那共享前缀就很难命中。

所以课件第 48 页给出的省钱建议里，除了“删不必要内容”“让模型简洁”之外，还有一句非常关键的话：

> **Maximize prefix cache hitting.**fileciteturn10file0

这说明在产品级系统里，prompt 不是随便拼就行，而是要考虑它的**可复用前缀结构**。

### 17.3 工具不可用时，为什么不要从 context 里删掉？

课件第 50 页给了一个很有启发性的建议：当某些工具暂时不可用时，不要把它们从 context 中删除，而应该在 decoding 时阻止生成这些工具名。fileciteturn10file0

这背后的原因正是 prefix caching：

- 如果删掉工具描述，前缀整体改变；
- cache 命中率下降；
- 成本和 TTFT 都会变差。

这说明：

> **上下文工程有时要在“语义最干净”与“系统最经济”之间做折中。**

---

## 18. 什么是 Context Engineering？为什么它比 Prompt Engineering 更大？

课件第 51 页给出 context engineering 的定义：

> Context engineering refers to the discipline of LLM context management to optimize costs, performance, and efficiency.fileciteturn10file0

这比 prompt engineering 大得多。因为 prompt engineering 更偏向：

- 任务描述怎么写；
- instruction 怎么给；
- few-shot 示例怎么放；
- wording 如何优化。

而 context engineering 还要额外考虑：

- context 长度怎么控制；
- 哪些历史要保留；
- 哪些内容应压缩；
- prefix cache 怎么命中；
- 工具说明怎么放；
- 子代理怎么拆；
- 长期记忆放文件系统还是 RAG。

所以可以说：

> **prompt engineering 是 context engineering 的一个子问题。**

---

## 19. 为什么 Context Engineering 重要？

课件第 52 页给出的答案很直接：

> 长上下文会带来速度下降、成本上升，以及由于 context rot 和高认知负担导致的性能下降。fileciteturn10file0

### 19.1 上下文越长，为什么不一定越好？

很多人会有一种直觉：

- 既然模型能看更多上下文，
- 那我把所有信息都塞进去，应该更好。

但课件明确反对这种天真想法。长上下文带来的问题至少有三类：

1. **更慢**：注意力计算更重；
2. **更贵**：token 成本直接上升；
3. **更差**：重要信息可能被淹没，模型开始“看了也等于没看”。

这里提到的 **context rot** 可以理解成：

> 当上下文越来越长时，模型对其中早期或细碎信息的利用能力会下降，信息虽然“在那”，却未必真正被有效使用。

### 19.2 高 cognitive load 是什么意思？

课件用“高认知负担”这个词非常形象。因为对于模型来说，长上下文不只是计算量问题，也是选择注意什么的问题：

- 信息太多时，重要部分不突出；
- 多条指令互相干扰；
- 历史噪声压过当前任务；
- 示例太多反而让模式不清晰。

所以 context engineering 的核心目标，不是“让上下文尽量大”，而是：

> **让上下文尽量短，但仍然保留完成任务最关键的信息。**

---

## 20. Context Compaction：压缩上下文到底在压什么？

课件第 54 页提出 **context compaction**，指的是让上下文更紧凑的方法。fileciteturn10file0

### 20.1 它不是简单删字数

真正的 compaction 不是粗暴删减，而是：

- 删掉无关信息；
- 把长历史压缩成摘要；
- 保留任务目标与关键中间状态；
- 让上下文仍然对当前步骤足够有用。

因此，context compaction 的本质是：

> **信息压缩，而不是字面裁剪。**

### 20.2 为什么有时服务端 compaction 很有用？

课件提到有些 API host 会提供 server-side compaction。其意义在于：

- 让应用开发者不必自己手写复杂上下文摘要逻辑；
- 在系统层统一做上下文压缩；
- 兼顾一致性与缓存策略。

不过从研究和工程角度，自己理解 compaction 机制仍然很重要，因为你需要判断：

- 什么能压；
- 什么绝不能压；
- 压缩后的损失是否可接受。

---

## 21. File System as Context：为什么不能把一切都塞进 prompt？

课件第 55–56 页提出一个很现实的方法：

> **把部分 context 卸载到文件系统。**

### 21.1 为什么这很自然？

因为 agentic behavior 往往会持续很久，模型上下文窗口再大，也不适合一直把所有历史都塞在 prompt 里。

于是可以把：

- 历史记录；
- 中间结果；
- 搜索笔记；
- 长文档；
- 大段代码；

保存为文本文件，让模型需要时再查。

这实际上把上下文分成了两层：

1. **working context**：当前 prompt 中的短期工作记忆；
2. **external context storage**：文件系统 / 向量库 / 检索系统中的长期外部记忆。

这和人类的短时记忆 vs 外部笔记有点类似。

### 21.2 为什么会自然走向 RAG 或 agentic search？

一旦信息被放进文件系统，模型就不能直接“看到”全部内容了，于是需要搜索机制。

课件第 56 页给出两条路线：

#### RAG
- 把文档分块；
- 做 embedding；
- 建索引；
- 按语义相似性快速召回。

#### Agentic Search
- 让 LLM 自己与文件系统交互；
- 生成 grep / CLI 命令；
- 用关键词或模式主动搜索。fileciteturn10file0

这说明 context engineering 和 RAG 并不是分离话题，而是天然连接的：

> **当上下文放不下时，你就必须把一部分 context 外部化，而外部化之后就需要检索。**

---

## 22. Recitation：为什么“把重要信息放到最后”有时会有用？

课件第 57 页讲了一个很有意思但很实用的技巧：**recitation**。fileciteturn10file0

它的想法是：

- 长上下文中，模型可能开始忘记一些信息；
- 可以把目标、关键要求等重要信息再次移动到上下文末尾；
- 让这些内容离当前生成位置更近。

### 22.1 这背后的直觉是什么？

虽然 Transformer 理论上能看全上下文，但在实践中：

- 离当前生成位置越近的信息，常常更容易被有效利用；
- 长序列前部的信息可能衰减；
- 尤其在复杂 agent 上下文中，初始目标容易被后续噪声淹没。

所以 recitation 本质上是一种“重要信息再提醒”。

它并不优雅，但很有效。

---

## 23. Sub-Agents：为什么拆任务能减少上下文负担？

课件第 58 页提出 sub-agents。核心思想是：

> **把一个大任务拆成多个子任务，让多个较小上下文的 agent 分头完成，再汇总。**fileciteturn10file0

### 23.1 为什么这会有用？

因为一个单一 agent 面临的问题通常是：

- 所有任务说明都在一起；
- 所有中间状态都在一起；
- 所有工具说明、历史记录、搜索结果都在一起；
- 上下文迅速膨胀。

如果拆成多个 sub-agent：

- 每个 agent 只看与自己子任务相关的信息；
- 每个上下文更短、更聚焦；
- 并行处理还可能提高总吞吐。

这说明 context engineering 并不只是“压缩一个 prompt”，还包括：

> **重新设计任务分工，让每个 prompt 天生更小。**

这是更高层次的上下文优化。

---

## 24. API 成本优化与 Context Engineering 最佳实践

课件第 48 页给出了一组很实用的降成本建议：

- 用 Batch / Flex API 处理不敏感时延任务；
- 尽可能提高 prefix cache 命中；
- 删除不必要内容；
- 要求模型简洁；
- 在可能时降低 reasoning effort；
- 简单任务用更小模型。fileciteturn10file0

如果把它们整合成 context engineering 的 best practices，可以总结成下面几条。

### 24.1 永远先问：这段内容真的需要在当前上下文里吗？

不要因为“也许会用到”就把一切都塞进去。

### 24.2 把稳定前缀固定下来

system prompt、工具描述、通用规则尽量保持稳定，以利于 prefix caching。

### 24.3 把变化部分放后面

这样既符合缓存，也更利于 recitation 和当前任务聚焦。

### 24.4 让上下文结构清晰

例如明确区分：

- system rules
- tool descriptions
- task objective
- retrieved evidence
- working memory
- final user request

这会降低模型的认知负担。

### 24.5 长任务不要死撑一个 agent

适当外部化到文件系统、RAG、搜索或子代理，是更健康的上下文策略。

### 24.6 思考成本也是成本

reasoning token 虽然常常不可见，但会影响价格与时延，因此要按任务难度控制 reasoning effort。

---

## 25. 这节课真正想建立的主线

如果把整套 slides 压缩成一条清晰主线，大概可以写成下面这样：

### 25.1 第一步：理解 LLM 真正看到的不是“聊天对象”，而是展开后的 token 序列

于是 chat template 变成基础。

### 25.2 第二步：理解 base model 与 chat model 的本质区别

前者只会文本续写，后者在 SFT 后学会了处理 chat template。

### 25.3 第三步：理解 prompt-based adaptation

包括：

- text completion
- in-context learning
- CoT prompting
- format sensitivity

### 25.4 第四步：理解 generation 并不是“随便吐字”，而是带参数的搜索过程

于是 decoding strategies 和参数配置变得关键。

### 25.5 第五步：理解 reasoning、tool use、多模态都在增加上下文结构复杂度

模型要处理的不再只是单轮文本。

### 25.6 第六步：理解上下文本身就是资源

它消耗：

- token budget
- API 钱
- latency
- attention capacity

### 25.7 第七步：因此必须做 context engineering

也就是：

- 压缩 context；
- 设计缓存友好的前缀；
- 外部化长期记忆；
- 用检索 / 文件系统 / 子代理管理复杂任务。

所以，这节课真正要你学会的，并不是几个 prompt 技巧，而是：

> **如何把 LLM 的上下文看成一种需要被精心设计和管理的计算资源。**

---

## 26. 最值得记住的几个核心结论

### 26.1 Chat 模型本质上仍然是文本续写

只不过聊天结构是通过 chat template 注入到 token 序列中的。

### 26.2 Base model 和 chat model 不能混为一谈

base model 不理解 chat template，chat model 则在 SFT 后学会了这种结构。

### 26.3 In-context learning 是一种“在上下文里适应任务”的方式

它很强，但成本也直接由 context 长度决定。

### 26.4 Prompt format 是评估和部署中的重要变量

不是细枝末节，而会实质影响性能。

### 26.5 Decoding 参数是在控制搜索行为

不同任务需要不同的确定性—多样性权衡。

### 26.6 Prefix caching 是产品级系统的关键优化手段

它直接影响成本与 TTFT，因此 prompt 结构必须缓存友好。

### 26.7 Context engineering 的目标不是“让上下文尽量长”，而是“让上下文尽量有用”

长上下文会更贵、更慢，也可能更差。

### 26.8 复杂 agent 系统必须把部分上下文外部化

文件系统、RAG、agentic search、sub-agents 都是在做这一件事。

---

## 27. 一句话总结

如果把这份 lecture notes 压缩成一句话，可以写成：

> **Context Engineering 的核心，是把 LLM 的上下文窗口视为一种昂贵且有限的工作记忆资源，并通过模板设计、示例组织、解码控制、缓存复用、外部存储与任务拆分等手段，让模型在有限上下文中以更低成本、更高效率和更稳定的方式完成复杂任务。**