import numpy as np
import torch
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, num_layers):
        super(Model, self).__init__()
        # 输入：vocab_size -- embedding_dim -- hidden_dim -- output_dim
        # num_layers为循环层数
        self.hidden_dim = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.5,
                            bidirectional=False)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, token, hidden):
        batch_size = token.size(0)

        # embedding层
        out = self.embedding(token.long())
        # lstm层
        out, hidden = self.lstm(out, hidden)
        # 改变形状
        out = out.contiguous().view(-1, self.hidden_dim)
        # 全连接层
        out = self.fc(out)
        # 把batch换到前面    在python中每个单词都会进行最后的sigmoid分类，但我们取最后一个当结果
        out = out.view(batch_size, -1)

        # 获取最后一次的结果
        out = out[:, -1]

        return out

    def init_hidden(self, batch_size):  # 初始化lstm层的记忆参数
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))

# 填充序列，把所有评论都填充到seq_length
def text_padding(encoded_reviews, seq_length):
    reviews = []
    for review in encoded_reviews:
        if len(review) >= seq_length:  # 评论长度过长就截取给定长度前面的
            reviews.append(review[:seq_length])
        else:  # 评论过短就在前面加相应的0直到为给定的长度
            reviews.append([0] * (seq_length - len(review)) + review)

    return np.array(reviews)