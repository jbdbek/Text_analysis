import os
import torch
import pandas as pd
from torch import nn
from d2l import torch as d2l

'''双向LSTM+全连接'''
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        #将输入的词索引转换成对应的词向量(长度为embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        #LSTM是一种常见的循环神经网络(RNN)变体，用于捕捉序列（文本）的长程依赖关系，并缓解普通RNN的梯度消失/爆炸问题
        #bidirectional=True表示从前往后和从后往前都做一次序列遍历，将两者的输出拼接起来，能更好地利用上下文信息
        #num_layers=2意味着将LSTM进行堆叠，多层结构可以捕捉更深层次的特征
        self.encoder = nn.LSTM(embed_size, 
                               num_hiddens, 
                               num_layers=num_layers,
                               bidirectional=True)
        #因为是双向LSTM，并且在forward中拿了第1个时间步(outputs[0])和最后1个时间步(outputs[-1])的隐状态来拼接
        #所以向量维度为4*num_hiddens，该拼接结果送入一个线性层，映射到2维，用于二分类
        self.decoder = nn.Linear(4 * num_hiddens, 2)
    def forward(self, inputs):
        #inputs(批量大小，时间步数)
        #因为长短期记忆网络要求其输入的第一个维度是时间维，
        #所以在获得词元表示之前，输入会被转置
        embeddings = self.embedding(inputs.T) #=>(时间步数，批量大小，词向量维度)
        self.encoder.flatten_parameters()
        #返回上一个隐藏层在不同时间步的隐状态
        outputs, _ = self.encoder(embeddings) #=>(时间步数，批量大小，2*隐藏单元数)
        #连结初始和最终时间步的隐状态，作为全连接层的输入，
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)#=>(批量大小，4*隐藏单元数)
        return outs