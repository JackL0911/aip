import numpy as np
from string import punctuation
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from IMDB情感分析.test5 import load_sentences
from model import Model, text_padding

# read data from text files
reviews_split = load_sentences('sentences.pkl')
# reviews = load_sentences('sentences.pkl')
with open('data/labels.txt', 'r') as f:
    labels = f.read()
def convert_to_string(two_dim_list):
    # 将每一行的字符串用换行符连接，最终用两个换行符连接每一行
    return "\n".join([" ".join(row) for row in two_dim_list])

# 示例使用

reviews = convert_to_string(reviews_split)


# 按照punctuation里面的标点符号，删除review里面的标点符号
reviews = reviews.lower()
all_text = ''.join([c for c in reviews if c not in punctuation])

reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)

# 删除标点符号之后，按照空格对其进行分割
words = all_text.split()

# 计数单词出现的次数，然后对其排序
word_counts = Counter(words)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)

# 排序之后给他一个对应的整数序号。生成一个字典存储
vocab2idx = {vocab: idx for idx, vocab in enumerate(vocab, 1)}

# 把上面的数据集，即所有句子，按照上面生成的字典赋值
encoded_reviews = []
for review in reviews_split:
    encoded_reviews.append([vocab2idx[vocab] for vocab in review.split()])

splitted_labels = labels.split("\n")
encoded_labels = np.array([
    1 if label == "positive" else 0 for label in splitted_labels
])

seq_length = 250
padded_reviews = text_padding(encoded_reviews, seq_length)

# 建立数据集
# 将数据按8:1:1的比例拆分为训练集、验证集和测试集
ratio = 0.8
train_length = int(len(padded_reviews) * ratio)

X_train = padded_reviews[:train_length]
y_train = encoded_labels[:train_length]

remaining_x = padded_reviews[train_length:]
remaining_y = encoded_labels[train_length:]

test_length = int(len(remaining_x) * 0.5)

X_val = remaining_x[: test_length]
y_val = remaining_y[: test_length]

X_test = remaining_x[test_length:]
y_test = remaining_y[test_length:]


batch_size = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
train_dataset = TensorDataset(torch.from_numpy(X_train).to(device), torch.from_numpy(y_train).to(device))
valid_dataset = TensorDataset(torch.from_numpy(X_val).to(device), torch.from_numpy(y_val).to(device))
test_dataset = TensorDataset(torch.from_numpy(X_test).to(device), torch.from_numpy(y_test).to(device))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
data_iter = iter(train_loader)
X_sample, y_sample = data_iter.next()


# 定义模型

vocab_size = len(vocab) + 1  # +1 for the 0 padding + our word tokens
embedding_dim = 400
hidden_dim = 256
output_dim = 1
num_layers = 2
num_epochs = 5

model = Model(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers).to(device)
print(model)

# Loss function and Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(num_epochs):
    hidden = model.init_hidden(batch_size)
    train_losses = []

    for i, (review, label) in enumerate(train_loader):
        model.train()
        review, label = review.to(device), label.to(device)

        # Initialize Optimizer
        optimizer.zero_grad()

        hidden = tuple([h.data for h in hidden])

        # Feed Forward
        output = model(review, hidden)

        # Calculate the Loss
        loss = criterion(output.squeeze(), label.float())

        # Back Propagation
        loss.backward()

        # Prevent Exploding Gradient Problem
        nn.utils.clip_grad_norm_(model.parameters(), 5)

        # Update
        optimizer.step()

        train_losses.append(loss.item())

        # Print Statistics
        if (i + 1) % 100 == 0:

            ### Evaluation ###

            # initialize hidden state
            val_h = model.init_hidden(batch_size)
            val_losses = []

            model.eval()

            for review, label in valid_loader:
                review, label = review.to(device), label.to(device)
                val_h = tuple([h.data for h in val_h])
                output = model(review, val_h)
                val_loss = criterion(output.squeeze(), label.float())

                val_losses.append(val_loss.item())

            print("Epoch: {}/{} | Step {}, Train Loss {:.4f}, Val Loss {:.4f}".
                  format(epoch + 1, num_epochs, i + 1, np.mean(train_losses), np.mean(val_losses)))

print(1)
