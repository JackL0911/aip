import nltk
import torch
import torch.nn as nn
import torch.optim as optim

# 准备数据
sentences = [
    "我 喜欢 学习",
    "机器学习 很有趣"
]
ret = nltk.word_tokenize("A's pivot is the pin or the central point on which something balances or turns")
# 构建词汇
word_to_idx = {}
for sentence in ret:
    for word in sentence.split():
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

# 超参数
embedding_dim = 5  # 嵌入向量维度
vocab_size = len(word_to_idx)

# 创建嵌入层
embedding = nn.Embedding(vocab_size, embedding_dim)

# 示例：将一个词转换为索引
word = "s"
word_idx = torch.tensor(word_to_idx[word])

# 获取该词的词向量
word_vector = embedding(word_idx)

# 打印结果
print(f"'{word}' 的词向量：", word_vector)