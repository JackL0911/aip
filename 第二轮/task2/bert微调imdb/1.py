import numpy as np
import torch
from keras.src.utils import pad_sequences
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW

from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

import time
import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载bert模型，使用BertForSequenceClassification模型
tokenizer = AutoTokenizer.from_pretrained("./bert_tokenizer/")

# 使用之前做好的文件。
with open('E:/学习ppt和word/AI派/第二轮/task1/IMDB情感分析/data/reviews.txt', 'r', encoding='utf-8') as f:
    sentences = f.read()

with open('E:/学习ppt和word/AI派/第二轮/task1/IMDB情感分析/data/labels.txt', 'r') as f:
    labels = f.read()

sentences = sentences.split('\n')
labels = labels.split("\n")
labels = np.array([
    1 if label == "positive" else 0 for label in labels
])
labels = labels.tolist()
# input_ids是转换为id后的文本
max_len = 128

input_ids = [tokenizer.encode(sent, add_special_tokens=True, max_length=max_len, truncation=True) for sent in sentences]

input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long",
                          value=0, truncating="post", padding="post")

# 生成注意力掩码，把填充的内容遮住
attention_masks = []

for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)

# 训练测试9:1
train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels,
                                                                        random_state=2024, test_size=0.1)
# 对注意力掩码做相同操作
train_masks, test_masks, _, _ = train_test_split(attention_masks, labels,
                                                 random_state=2024, test_size=0.1)

# 转为tensor
train_inputs = torch.tensor(train_inputs)
test_inputs = torch.tensor(test_inputs)
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)
train_masks = torch.tensor(train_masks)
test_masks = torch.tensor(test_masks)

# 创建数据集和loder
batch_size = 32

# 训练集的数据集
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)  # 数据随机采样
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# 测试集的数据集
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)  # 按顺序采样
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# 获取与训练好的bert分类模型，分两类
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
)

# 同样使用gpu训练
model.cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# epoch选2
epochs = 2


def cal_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


train_loss = []
train_plt_loss = []
test_loss = []
for epoch_i in range(epochs):

    print("Epoch:", epoch_i)

    # 记录当前时间
    t0 = time.time()
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):

        if step % 100 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print(
                '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    loss={}.'.format(step, len(train_dataloader), elapsed,
                                                                                  train_loss[-1]))
        if step % 100 == 0 and not step == 0:
            train_plt_loss.append(train_loss[-1])
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        loss = outputs[0]
        total_loss += loss.item()
        train_loss.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print("")

    t0 = time.time()

    # 开始测试
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0


    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = cal_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))

print("")
print("Training complete!")

plt.xlabel("time")
plt.ylabel("train_loss")
x = np.arange(len(train_plt_loss))
plt.plot(x, train_plt_loss, label='train loss')
plt.show()

print(1)
