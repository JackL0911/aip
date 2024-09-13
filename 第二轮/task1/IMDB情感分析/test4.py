import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import nltk
import json

# 读取文本文件
def read_files(directory):
    sentences = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    tokens = nltk.word_tokenize(line.strip())
                    sentences.append(tokens)
    return sentences

# 保存词汇表
def save_vocab(vocab, filepath):
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(vocab, file)

# 加载词汇表
def load_vocab(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        vocab = json.load(file)
    return vocab

# 自定义 Dataset
class Word2VecDataset(Dataset):
    def __init__(self, sentences, window_size=2):
        self.window_size = window_size
        self.pairs = []
        for sentence in sentences:
            for i, word in enumerate(sentence):
                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)
                for j in range(start, end):
                    if j != i:
                        self.pairs.append((word, sentence[j]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

# Word2Vec 模型
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embeddings(x)

# 主要逻辑
def main(directory, vocab_file='vocab.json', embedding_dim=100, window_size=2, epochs=10, lr=0.01):
    # 检查是否存在词汇表文件
    if os.path.exists(vocab_file):
        vocab = load_vocab(vocab_file)
        vocab_size = len(vocab)
        print("Loaded vocabulary from file.")
    else:
        sentences = read_files(directory)

        # 构建词汇表
        word_counts = Counter(word for sentence in sentences for word in sentence)
        vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
        vocab_size = len(vocab)

        # 保存词汇表
        save_vocab(vocab, vocab_file)
        print("Vocabulary saved to file.")

    # 准备数据
    dataset = Word2VecDataset(sentences, window_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 创建模型
    model = Word2Vec(vocab_size, embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 训练模型
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            context_idx = torch.tensor([vocab[word] for word in context], dtype=torch.long)
            target_idx = torch.tensor([vocab[word] for word in target], dtype=torch.long)

            optimizer.zero_grad()
            output = model(context_idx)
            loss = criterion(output, target_idx)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

    # 保存词向量
    word_vectors = model.embeddings.weight.detach().numpy()
    np.save('word_vectors.npy', word_vectors)

    # 打印词向量示例
    for word, idx in vocab.items():
        print(f"'{word}': {word_vectors[idx]}")

# 使用时调用
if __name__ == '__main__':
    main('aclImdb_v1/aclImdb/test/all')  # 替换为你的文件夹路径