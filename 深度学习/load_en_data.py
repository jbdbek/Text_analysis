import os
import torch
import pandas as pd
from torch import nn
from d2l import torch as d2l
"""
读取CSV文件中的文本序列和标签
参数:
- file_path: CSV文件路径
- is_train: 是否读取训练集，默认为True
 返回:
- ID：ID列表
- data: 评论内容列表
- labels: 标签列表
"""
def read_csv_data(file_path, is_train=True):
    df = pd.read_csv(file_path)

    ID = df['ID'].tolist
    data = df['Content'].tolist()
    labels = df['Label'].tolist()
    
    # 转换标签值为整数
    labels = [int(label) for label in labels]

    return ID, data, labels

"""返回数据迭代器和评论数据集的词表"""
def load_data_revi(batch_size, num_steps=500):
    trainfile_path = "./feature_processing/out/train_data/en_sample_data/sample.csv"
    testfile_path = "./feature_processing/out/test_label_data/test.label.en.csv"
    train_data = read_csv_data(trainfile_path, True)
    test_data = read_csv_data(testfile_path, False)
    #基于单词级别，对读取到的评论进行分词
    train_tokens = d2l.tokenize(train_data[1], token='word') 
    test_tokens = d2l.tokenize(test_data[1], token='word')
    #构建词汇表，过滤掉出现频率低于5的词
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    #将词语转换为索引，并进行截断或填充
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    #创建数据迭代器
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[2])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[2])),
                                batch_size,
                                is_train=False) #不用于训练
    print(f"词汇表大小：{len(vocab)}")
    return train_iter, test_iter, vocab