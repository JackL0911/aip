import numpy as np
from string import punctuation
from collections import Counter
import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

# from IMDB情感分析.test5 import load_sentences
from model import Model, text_padding

# 读取文件
with open('data/reviews.txt', 'r', encoding='utf-8') as f:
    reviews = f.read()
# reviews = load_sentences('sentences.pkl')
with open('data/labels.txt', 'r') as f:
    labels = f.read()


# 按照punctuation里面的标点符号，删除review里面的标点符号
reviews = reviews.lower()
all_reviews = ''.join([c for c in reviews if c not in punctuation])

reviews_split = all_reviews.split('\n')
all_reviews = ' '.join(reviews_split)

# 删除标点符号之后，按照空格对其进行分割
words_reviews = all_reviews.split()

# 计数单词出现的次数，然后对其排序
word_counts = Counter(words_reviews)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)

# 排序之后给他一个对应的整数序号。生成一个字典存储
reviews_vocab = {vocab: idx for idx, vocab in enumerate(vocab, 1)}

# 把上面的数据集，即所有句子，按照上面生成的字典赋值
last_reviews = []
for review in reviews_split:
    last_reviews.append([reviews_vocab[vocab] for vocab in review.split()])

splitted_labels = labels.split("\n")
last_label = np.array([
    1 if label == "positive" else 0 for label in splitted_labels
])

seq_length = 250
padded_reviews = text_padding(last_reviews, seq_length)

# 建立数据集
# 7:3建立训练和测试集
rat = 0.8
train_length = int(len(padded_reviews) * rat)

X_train = padded_reviews[:train_length]
y_train = last_label[:train_length]

X_test = padded_reviews[train_length:]
y_test = last_label[train_length:]

batch_size = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
train_dataset = TensorDataset(torch.from_numpy(X_train).to(device), torch.from_numpy(y_train).to(device))
test_dataset = TensorDataset(torch.from_numpy(X_test).to(device), torch.from_numpy(y_test).to(device))


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# 定义模型
vocab_size = len(vocab) + 1
embedding_size = 400
hidden_size = 256
output_size = 1
num_layers = 3
epochs = 3

model = Model(vocab_size, embedding_size, hidden_size, output_size, num_layers).to(device)

# 损失函数采用二进制交叉熵函数
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_all_loss = []
test_all_loss = []
for epoch in range(epochs):
    hidden = model.init_hidden(batch_size)  # 初始化lstm的记忆
    train_losses = []

    for i, (review, label) in enumerate(train_loader):
        model.train()   # 训练模式
        review, label = review.to(device), label.to(device)

        optimizer.zero_grad()   # 梯度清零
        hidden2 = tuple([h.data for h in hidden])   # 复制hidden
        output = model(review, hidden2)

        loss = criterion(output.squeeze(), label.float())
        loss.backward()
        # 梯度裁剪防止梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        # 优化
        optimizer.step()

        train_losses.append(loss.item())    # 记录
        train_all_loss.append(loss.item())

        if (i + 1) % 100 == 0:  # 测试
            test_h = model.init_hidden(batch_size)
            test_losses = []

            model.eval()

            for review, label in test_loader:
                review, label = review.to(device), label.to(device)
                test_h = tuple([h.data for h in test_h])
                output = model(review, test_h)
                test_loss = criterion(output.squeeze(), label.float())

                test_losses.append(test_loss.item())

                test_all_loss.append(test_loss.item())

            print("Epoch {} Step {} Train {:.4f} Test {:.4f}".
                  format(epoch + 1, i + 1, np.mean(train_losses), np.mean(test_losses)))

print(1)

plt.xlabel("time")
plt.ylabel("train_loss")
x = np.arange(len(train_all_loss))
plt.plot(x, train_all_loss, label='train loss')
plt.show()

plt.xlabel("time")
plt.ylabel("test_loss")
x = np.arange(len(test_all_loss))
plt.plot(x, test_all_loss, label='train loss')
plt.show()
