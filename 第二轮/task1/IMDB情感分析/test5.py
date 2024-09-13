import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import nltk
import json
import pickle  # 用于序列化

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


# 保存句子
def save_sentences(sentences, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(sentences, file)


# 加载句子
def load_sentences(filepath):
    with open(filepath, 'rb') as file:
        sentences = pickle.load(file)
    return sentences


# 自定义 Dataset
class Word2VecDataset(Dataset):
    def __init__(self, sentences, vocab, window_size=2):
        self.window_size = window_size
        self.vocab = vocab
        self.unk_token = '<UNK>'
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
        context_word, target_word = self.pairs[idx]
        context_index = self.vocab.get(context_word, self.vocab[self.unk_token])
        target_index = self.vocab.get(target_word, self.vocab[self.unk_token])
        return context_index, target_index


# Word2Vec 模型
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embeddings(x)


# 主要逻辑
def main(directory, vocab_file='vocab.json', sentences_file='sentences.pkl', embedding_dim=100, window_size=4,
         epochs=10, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 检查是否存在词汇表文件
    if os.path.exists(vocab_file):
        vocab = load_vocab(vocab_file)
        vocab['<UNK>'] = len(vocab)  # 添加未知单词标记
        vocab_size = len(vocab)
        print("Loaded vocabulary from file.")

        # 检查是否存在句子文件
        if os.path.exists(sentences_file):
            sentences = load_sentences(sentences_file)
            print("Loaded sentences from file.")
        else:
            sentences = read_files(directory)  # 读取文件以获得句子
            save_sentences(sentences, sentences_file)
            print("Saved sentences to file.")
    else:
        sentences = read_files(directory)

        # 构建词汇表
        word_counts = Counter(word for sentence in sentences for word in sentence)
        vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
        vocab_size = len(vocab)
        vocab['<UNK>'] = len(vocab)  # 添加未知单词标记

        # 保存词汇表
        save_vocab(vocab, vocab_file)
        print("Vocabulary saved to file.")

        # 保存句子
        save_sentences(sentences, sentences_file)
        print("Saved sentences to file.")

    # 准备数据
    dataset = Word2VecDataset(sentences,vocab, window_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 创建模型并移动到 GPU
    model = Word2Vec(vocab_size, embedding_dim).to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 训练模型
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            # context_idx = torch.tensor([vocab[word] for word in context], dtype=torch.long).to(device)
            # target_idx = torch.tensor([vocab[word] for word in target], dtype=torch.long).to(device)

            context_idx = context
            target_idx = target

            optimizer.zero_grad()
            output = model(context_idx)
            loss = loss_fn(output, target_idx)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

    # 保存词向量
    word_vectors = model.embeddings.weight.detach().cpu().numpy()
    np.save('word_vectors.npy', word_vectors)

    # 打印词向量示例
    for word, idx in vocab.items():
        print(f"'{word}': {word_vectors[idx]}")


# 使用时调用
if __name__ == '__main__':
    main('aclImdb_v1/aclImdb/train/all')  # 替换为你的文件夹路径