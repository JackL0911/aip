from keras.src.utils import pad_sequences
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments, AutoTokenizer
import torch
import numpy as np

# 加载bert模型，使用BertForSequenceClassification模型
tokenizer = AutoTokenizer.from_pretrained("./bert_tokenizer/")
with open('E:/学习ppt和word/AI派/第二轮/task1/IMDB情感分析/data/reviews.txt', 'r', encoding='utf-8') as f:
    reviews = f.read()

with open('E:/学习ppt和word/AI派/第二轮/task1/IMDB情感分析/data/labels.txt', 'r') as f:
    y_train = f.read()

reviews = reviews.split('\n')

sentences = reviews
labels = y_train

# input_ids是转换为id后的文本，但是注意的是BERT要用“[SEP]”符号来分隔两个句子，且要在每一个样本前加上"[CLS]"符号。
MAX_LEN = 128

input_ids = [tokenizer.encode(sent, add_special_tokens=True, max_length=MAX_LEN) for sent in sentences]
# test_input_ids=[tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN) for sent in test_sentences]


print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                          value=0, truncating="post", padding="post")
