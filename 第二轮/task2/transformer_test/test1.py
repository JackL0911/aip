import torch
device='cuda' if torch.cuda.is_available() else 'cpu'


import re
import os


def rm_tags(text):
    re_tags = re.compile(r'<[^>]+>')
    return re_tags.sub(' ', text)


def read_files(filetype):
    path = "aclImdb/"
    file_list = []

    positive_path = path + filetype + "/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]

    negative_path = path + filetype + "/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    print("read", filetype, "files:", len(file_list))

    all_labels = ([1] * 12500 + [0] * 12500)

    all_texts = []
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]

    return all_labels, all_texts


y_train, train_text = read_files("E:\学习ppt和word\AI派\第二轮\2\IMDB情感分析\aclImdb_v1\aclImdb\test")
y_test, test_text = read_files("test")
