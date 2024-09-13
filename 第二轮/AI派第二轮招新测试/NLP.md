## 自然语言处理组(NLP)

### 数学基础

+ 概率论基础（最大似然估计，贝叶斯法则，二项式分布）
+ 信息论基础（熵，困惑度，噪声信道模型）
+ 马尔可夫模型和条件随机场

### 课程资料与论文阅读

nlp领域并不如cv领域那样容易理解，模型学习难度会更大，学习过程中可以做好相关笔记

- 课程资料：
  

​		[吴恩达深度学习课程第五课](https://www.bilibili.com/video/BV1F4411y7BA)(讲到经典模型处会给出相应的论文名称)，你能够从中学会：

- RNN的BPTT手动推导（理解梯度爆炸和梯度消失的原因）

- RNN的两种经典变体：GRU与LSTM

- word embedding的基本知识

- seq2seq结构

- 论文阅读相关资料：

  - 最早提出Attention mechanism的论文: [ Neural Machine Translation by Jointly Learning to Align and Translate (arxiv.org)](https://arxiv.org/abs/1409.0473)
  - Transformer模型经典论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - BERT原始论文：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
  - Bert的应用：<https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html#%E7%94%A8-BERT-fine-tune-%E4%B8%8B%E6%B8%B8%E4%BB%BB%E5%8B%99>
  - 希望你能够从中学会BERT的奇妙之处以及其在情感分类任务上是如何进行fine-tune的

- 代码编写相关资料：

  - Pytorch框架的学习：https://pytorch.org/tutorials/beginner/basics/intro.html

  - 预训练模型权重：[Models - Hugging Face](https://huggingface.co/models)

### 学习路线, 工具网站，相关资料推荐

##### [AI Expert Roadmap (am.ai)](https://i.am.ai/roadmap/#note) 

##### [Mikoto10032/DeepLearning: Deep Learning Tutorial (github.com)](https://github.com/Mikoto10032/DeepLearning)

[Transformer非常经典好的blog](https://jalammar.github.io/illustrated-transformer/)

https://github.com/bentrevett/pytorch-sentiment-analysis 

https://github.com/FudanNLP/nlp-beginner 

https://github.com/graykode/nlp-tutorial/tree/master 

http://nlp.seas.harvard.edu/2018/04/03/attention.html

### 代码应用与实践任务

#### Task1.NLP的基本任务

你可以从分类任务和序列标注任务中选择其一完成代码实现(当然,你也可以都完成)

**分类任务**

分类任务是最简单也最易学习的应用场景，这个代码任务可以分类任务来帮助大家入门NLP这一领域。

IMDB情感分析任务基本上算是一个已经被刷烂掉了的任务，不过也是很好的入门学习任务

+ 你可以选择从[官网](https://ai.stanford.edu/~amaas/data/sentiment/)或者[stanfordnlp/imdb · Datasets at Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb)下载数据集。

+ 由于文本的离散特性，往往需要先利用一些库对文本进行tokenize

+ 根据tokenize之后的token得到对应的word embedding

+ 从word embedding开始接入常规的模型训练过程

+ 使用基本的textcnn、lstm实现，准确率不做要求

**序列标注**

序列标注（Sequence Labeling）也是NLP中一个重要的任务，实体命名识别(NER)是序列标注任务中的一种。你也可以选择完成实现代码完成序列标注任务。

+ 你可以选择使用中文序列标注的数据集：[ttxy/cn_ner · Datasets at Hugging Face](https://huggingface.co/datasets/ttxy/cn_ner) 这个数据集整理了22个NER任务的数据（[中文NER数据集整理 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/529541521)）。你可以选择其中一个任务并使用这部分的数据数据（例如选择task_name=people_dairy_1998的数据）
+ 在序列标注任务中，数据处理相比情感分析任务可能更加麻烦，希望你能够思考并妥善处理。
+ 为了测试你模型的实际效果, 推荐在其给出的训练集上自行划分训练集和测试集
+ 使用基础的LSTM+CRF的方法实现，准确率不做要求



#### Task2.基于Transformer的预训练模型

学习最基本的预训练模型，bert和gpt(你也可以选择学习其中的一个)

+ 查阅资料, 了解并掌握transformer模型原理([Transformer论文逐段精读【论文精读】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.999.0.0))
+ 尝试手动实现bert或者gpt

+ 使用预训练好的bert/gpt以及它们对应的tokenizer进行finetune，并且**利用预训练模型实现上一个任务中的分类任务或者序列标注任务**(你也可以选择使用自己实现的bert并进行对比)

* 调用的库不做要求，强烈建议使用transformers库
* [transformers库官方文档](https://huggingface.co/docs/transformers/training) ，官方文本分类[代码参考](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/)
* 比较bert和gpt的区别



#### Task3.NLP与大模型（选做）

当下，大语言模型的相关应用越来越多，对越来越多人来说，大语言模型也逐渐成为日常学习和生活的一个工具。

+ 你可以尝试使用api（任意选个一个可用api就好[获取 API 密钥 | Google AI Studio](https://aistudio.google.com/app/apikey)），如果你资源足够，也可以选择本地部署一个大模型，实现对模型的使用。
+ 你可以尝试分析使用大模型完成情感分析或者序列标注任务的可行性（[GPT-NER: Named Entity Recognition via Large Language Models](https://arxiv.org/pdf/2304.10428)）。由于时间和资源成本较高，如果你想进行相关实践，只需要选择少量数据进行测试以达成演示的效果即可。
+ 在Task3中你也可以尝试使用大模型实现任何你感兴趣的任务，你有很高的自由度，只需要展现出你的学习即可。



#### 说明

+ 对于 **数学基础** 和 **课程资料与论文阅读** 部分，我们不进行严格考核。我们提供了一些NLP领域初期非常经典的工作，建议未接触过NLP领域的同学还是看一下。如果你期间有重点学习这部分内容，请做好记录。

+ 我们**不对准确率作出硬性要求**，**请不要过于纠结调参**，重点在于你在实现过程中获得的知识和收获

+ 在学习的过程中，请做好相关笔记或者记录，最后**面试**时，你可以选择你喜欢的形式呈现你的学习和实践成果（ppt，markdown或者博客都是ok的）

- 对于**学习路线, 工具网站，相关资料推荐**部分，我们仅提供参考。

* **对于Task1**，如果你此前以学习过相关内容并且已完成过序列标注或者情感分析中的任意一个，请尽量选择另外一个你没完成的任务。使用gpt，copilot或者他人的代码框架辅助实现是ok的，如果你使用了他人的框架，你需要说明并确保你理解代码每个部分的内容。
* **对于Task2**，如果你只有4GB的显存资源，可能会遇到一些问题，可以尝试使用混合精度或者TinyBert（[huawei-noah/TinyBERT_General_4L_312D at main (huggingface.co)](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D/tree/main)）等方式进行解决。
* 如果其中某项没有完成，也请不要气馁，面试过程中并不会刻意为难大家，只要展现出你的学习即可

- 详细信息可以询问陈子旸（QQ：3257161455），对于资源要求较高的任务，相信大家都各有方法，提供参考获取方法[autodl资源解决方案](https://www.autodl.com/)。
