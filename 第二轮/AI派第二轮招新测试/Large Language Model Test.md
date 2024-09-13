# Large Language Model Test

从2022年底开始，ChatGPT3.5横空出世，~~几乎成为大学生必不可少的工具~~，希望通过本次测试你能了解大模型的前世今生

考虑到大家计算资源有限，本次我们选取的都是比较小的模型，你需要有搭载NVIDIA显卡的笔记本，至少4G显存，如果没有的话可以考虑使用colab，或者上autodl租卡

## Step 1. Transformer

这里假设你已经有了深度学习的基础知识（如mlp）

首先需要你熟悉掌握transformer模型的基本框架，它也是当今时代一切大模型的起源，以下是一些推荐的资源：

- [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) transformer原始论文，写的太high level了，如果对nlp不熟悉可以不看

- [Transformer论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.999.0.0) d2l伟大无需多言

另外推荐两个我非常喜欢的视频，能够帮助大家很好理解transformer的架构：

[But what is a GPT? Visual intro to transformers | Chapter 5, Deep Learning](https://www.youtube.com/watch?v=wjZofJX0v4M)

[Attention in transformers, visually explained | Chapter 6, Deep Learning](https://www.youtube.com/watch?v=eMlx5fFNoYc&t=438s)

本节的任务：

1. 利用numpy手搓多头注意力,为了保证尽量结果一致, 我已经给出了代码的一部分，剩下的需要你自己填空

   ```python
   import numpy as np
   np.random.seed(114514)
   
   def scaled_dot_product_attention(Q, K, V, mask=None):
     '''
     1. 需要完成调整 K 的转置来匹配 Q 的最后一个维度，
     task1. 计算attn_score并缩放，
     3. softmax 应用于最后一个轴计算attn_weight，
     4. 应用attn_weights输出output
     5. 带掩码mask的的注意力可以不用实现,但请记住encoder和decoder的transformer块是不一样的，很大一部分都在就在mask上
     '''
       return output, attention_weights
   
   def multi_head_attention(embed_size, num_heads, input, mask=None):
     '''
     1. embed_size 确保可以等分 num_heads 份， 否则输出错误
     task1. 随机初始化Wq,Wk,Wv,Wo矩阵，并对input做线性变换
     3. 利用scaled_dot_product_attention()输出的attn_output计算O
     4. 返回output, attN_weights
     '''
       return output, weights
   
   # test e.g.
   embed_size = 128
   num_heads = 8
   input = np.random.randn(10, 20, embed_size)
   output, weights = multi_head_attention(embed_size, num_heads, input)
   print(output.shape, weights.shape)
   output[0][0][:10], weights[0][0][0][:10]
   ```

   测试的输出如下：

   ```python
   (10, 20, 128) (10, 8, 20, 20)
   (array([-91.96555916, -19.40983534, -32.99740866, 113.35786088,
           138.22610441,  81.21040905, -30.81003178,  90.7098463 ,
           162.38724319, -40.72173619]),
    array([1.94810489e-189, 3.21476597e-151, 3.61314239e-103, 4.96644350e-219,
           3.90604112e-173, 3.46437823e-131, 4.72245009e-077, 2.66307289e-194,
           1.00000000e+000, 5.17103825e-098]))
   ```

## Step 2. 大模型的三种架构

大模型通常采用三种主要架构：Encoder-Only, Decoder-Only, 和 Encoder-Decoder。

### 1. Encoder-Only 架构

**特点**：Encoder-Only 架构仅包含编码器部分。它通常用于处理那些只需要理解输入数据而不需要生成新数据的任务。这种架构通过堆叠多层编码器（通常是自注意力层和前馈神经网络层）来处理和理解输入。

**应用场景**：**文本分类**：如情感分析、意图识别等。**实体识别**：从文本中识别和分类命名实体。**特征提取**：为下游任务提取有用的特征，比如在更复杂的模型中使用。

**典型模型**：

- BERT（Bidirectional Encoder Representations from Transformers）是最著名的 Encoder-Only 架构的例子，广泛用于各种文本理解任务。

本节希望你能了解BERT的基础架构[BERT 论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1PL411M7eQ?vd_source=827e9d926cec44ef6817b376d985aae5)，**尤其是要完全理解BERT是怎么做预训练的**， 论文地址[arXiv:1810.04805](https://arxiv.org/abs/1810.04805)，encoder-only在如今使用的并不多，所以不要求了解过多

### 2. Decoder-Only 架构

**特点**：Decoder-Only 架构仅包含解码器部分。它设计用于生成任务，通过自回归方式逐个生成输出序列的元素。每个解码器层通常包含掩蔽的自注意力层，确保预测当前元素时只使用之前的元素，从而保持生成过程的因果关系。

**应用场景**：**文本生成**：如语言模型、机器翻译、文本摘要。**代码生成**：自动编写程序代码。**对话系统**：自动生成用户交互响应。

**典型模型**：

- GPT（Generative Pre-trained Transformer）系列是 Decoder-Only 架构的代表，广泛用于各种生成任务。
- Llama,llama2,llama3
- Qwen,Qwen2
- ....

几乎现在市面上所有的模型都是Decoder-only架构，我希望你能了解GPT系列的发展历史[【GPT，GPT-2，GPT-3 论文精读【论文精读】](https://www.bilibili.com/video/BV1AF411b7xQ?vd_source=827e9d926cec44ef6817b376d985aae5)

，并且**完全理解gpt是怎么做预训练的**

### 3. Encoder-Decoder 架构

**特点**：Encoder-Decoder 架构结合了编码器和解码器两部分。编码器负责理解输入，而解码器则负责基于编码器的输出生成数据。这种架构通常在编码器和解码器之间有一种交互机制（如注意力机制），使得解码器可以更有效地利用编码器的信息。

**应用场景**：**机器翻译**：将一种语言翻译成另一种语言。**文本到语音（TTS）**：将文本转换为语音输出。**图像字幕**：生成描述图像内容的文字。

**典型模型**：
- ChatGLM 是为数不多的Encoder-Decoder 架构的生成模型

这一部分不过多赘述也不需要你了解太多，因为原始transformer确实用的没那么多

本节主要考察你对一些基础知识的理解，请解决如下问题：

1. 为什么bert不能像gpt一样做文本生成？

2. 对于decoder-only的模型，它是由tokenizer，embedding层， $N\times$transformer block，lm_head，请你简单说明一下这4个部分分别在做什么？token是一个什么东西？

3. 为什么decoder-only的模型的数量远远多于Encoder-Decoder模型？明明二者都可以做文本生成

4. 使用预训练好的bert/gpt以及它们对应的tokenizer在imdb任务上finetune，计算这两种模型在IMDB分类任务在测试集上的准确率（**请使用huggingface的transformer库完成，这也是最重要的库**），并比较二者在训练前后分类的准确性（如果计算资源实在不够的兄弟可以不做，仅展示代码即可）

   如果没有vpn可能无法下载模型和数据集，可以在python代码里考虑使用镜像站`os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"`

   数据集：[stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)，下载/加载代码：

   ```python
   from datasets import load_dataset
   
   ds = load_dataset("stanfordnlp/imdb")
   ```

   模型：[bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased), [gpt2](https://huggingface.co/openai-community/gpt2)，下载/加载代码

   ```python
   # Load model directly
   from transformers import AutoTokenizer, AutoModelForMaskedLM
   
   model_name = "google-bert/bert-base-uncased" # openai-community/gpt2
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForMaskedLM.from_pretrained(model_name)
   ```

   **本节需要你参考transformer库的[官方文档](https://huggingface.co/docs/transformers/index)**

## Step 3. 一个decoder-only的Generative LLM的前世今生

本节我们以最新发布的Llama3为例，希望你能了解一个可以商业使用的llm是如何训练出来的，本节我们主要使用llama3的技术报告来帮助你了解llm的完整训练[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)，bilibili上也有李沐老师的讲解可以参考

### 1. Pre-training

请你阅读参考文章的第三节，回答以下问题：

- 训练时有一个参数max_length，它是做什么的
- 在真正开始训练时有一个warm up，它是用来做什么的

### 2. Post-training

在经过预训练后，我们得到的模型还不能直接应用于对话，它本质上还是一个next token predictor，为了让其具有良好的qa的属性，往往还需要一下额外的步骤才能得到一个商业可用的llm

请你阅读参考文章的第四节

#### 2.1 Instruction Tuning

请你自行搜索资料，回答一下问题：

- 什么是instruction tuning？为什么需要instruction tuning?
- Llama3 的instruction tuning的格式是怎么样的？

提示，llama2的instruction tuning的格式如下：

```structured text
<s>[INST] <<SYS>>
{system_message}
<</SYS>>
{user_message} [/INST]
{assistant_message} 
```

#### 2.2 SFT

在预训练结束后，我们会用高质量的qa对去sft llm，让llm初步具有qa的能力，但仅仅这样是不够的，我们还要将llm与人类偏好对齐

问题：

- 为什么要将llm与人类偏好对齐？不这么做会出现什么问题？

#### 2.3 RLHF/PPO

PPO是非常经典的强化学习算法，可以用于将llm与人类价值观对齐，由于这不是我们需要深入考虑的问题，所以我希望你解决以下几个小问题：

- rlhf的偏好数据集是如何构造的？
- reward model是做什么的？它是如何被训练的？

#### 2.4 [DPO](https://arxiv.org/abs/2305.18290)

DPO是去年Stanford提出的新的rlhf算法，在数学上可以证明与ppo等价，我仍然不需要你深入了解，但我希望你能理解DPO解决了什么问题：

- DPO和PPO相比优势在哪里？请你详细阐述一下

一个视频供你参考[DPO (Direct Preference Optimization) 算法讲解](https://www.bilibili.com/video/BV1GF4m1L7Nt/?vd_source=827e9d926cec44ef6817b376d985aae5)

## Step 4. llm实战演练1

这部分我们本地使用llm，我们使用的是Gemma-2-2b-it的小模型，并使用INT4量化的版本,

首先需要下载模型，与之前类似

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "unsloth/gemma-task1-2b-it-bnb-4bit

# Load model directly
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,device_map='auto')
```

使用该模型计算IMDB分类任务在测试集上的准确率，由于生成模型计算成本较高，仅仅使用**测试集前200个**进行测试即可，计算三种情况：

- zero-shot下的accuracy
- 2-shot下的accuracy（不能和测试集前200重叠）
- 4-shot下的accuracy（不能和测试集前200重叠）

关于zero-shot, n-shot(few-shot)的概念需要你搜索资料自行理解

提示：你可能需要使用tokenizer.apply_chat_template()这个函数，另外你需要提供prompt让模型知道他要做什么

## Step 5. llm实战演练2

这部分我们使用llm的api，使用openai的gpt-4o-mini（因为最便宜）。所谓api就是通过代码的形式用gpt交互，下面是一个例子，具体可参照openai的[说明文档](https://platform.openai.com/docs/quickstart)

```python
from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)
```

考虑到openai国内无法访问，这里提供几个代理网站https://wildcard.com.cn/.  https://aigptx.top/. 第一个网站也可以买virtual visa card 直接绑定openai，在colab上用原生api也是可以的

使用gpt-4o-mini计算IMDB分类任务在测试集上的准确率，由于生成模型成本较高，仅仅使用**测试集前200个**进行测试即可，计算三种情况：

- zero-shot下的accuracy
- 2-shot下的accuracy（不能和测试集前200重叠）
- 4-shot下的accuracy（不能和测试集前200重叠）

关于zero-shot, n-shot(few-shot)的概念需要你搜索资料自行理解

## Step 6. Research 

恭喜你，如果你做到这里，说明你对llm已经有了一个基本的理解了，下面我会向你介绍几篇工作，浅浅接触一些前沿的工作（only 我了解的领域hhhh）吧。

1. 一篇关于越狱攻击：http://arxiv.org/abs/2403.07865

    要求在通过api复现他的效果

2. 一篇关于越狱攻击的防御：https://arxiv.org/abs/2407.09121 

   讲清楚几个问题：他研究了/发现了一个什么有意思的问题，用什么样的方式解决了这个问题（这也是research的一般过程）

## Presentation

###### 科研是一个很大的话题；在真正投身科研之前，往往需要大量的学习和探索，打下基础，思考方向。

本次测试更多的是带你入门llm这一个领域，更加深入的学习往往需要更多时间，日积月累。

对于这一个月的成果，我希望你能以电子笔记的形式记录下来，比如latex+typora的形式，如果能汇总成一个ppt也是更好
