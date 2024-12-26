import os
import torch
from torch import nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        #输入(batch_size, num_tokens)
        #可训练嵌入层，随着模型训练动态更新
        self.embedding = nn.Embedding(vocab_size, embed_size)
        #不可训练嵌入层，用于加载预训练的静态词向量
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        #输出(batch_size, num_tokens, embed_size)
        #训练过程中随机丢弃50%的神经元，防止模型过拟合
        self.dropout = nn.Dropout(0.5)
        #将卷积层和池化层提取的特征拼接后输入全连接层，将其映射到2维
        self.decoder = nn.Linear(sum(num_channels), 2)
        #自适应平均池化操作，用于在时间维度上对卷积后的特征进行汇聚，提取全局特征
        self.pool = nn.AdaptiveAvgPool1d(1)
        #输出(batch_size, num_channels, 1)
        #激活函数
        self.relu = nn.ReLU()
        #构建多个一维卷积层(Conv1d)，输入为(batch_size, 2 * embed_size, num_tokens)
        self.convs = nn.ModuleList()
        #num_channels：决定了每个卷积层的输出通道数(特征数量)
        #kernel_sizes：指定每个卷积层的感受野(窗口大小)
        #2*embed_size：是输入通道数，因为嵌入层输出经过拼接后维度翻倍
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))
        #输出为(batch_size, num_channels, new_seq_len)
    #前向传播
    def forward(self, inputs):
        #嵌入层：生成动态和静态嵌入，拼接在最后一个维度上
        embeddings = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2) 
        #=>(batch_size, num_tokens, 2 * embed_size)
        #调整张量维度，以适配 Conv1d 的输入格式
        #(batch_size, 2*embed_size, seq_len)
        embeddings = embeddings.permute(0, 2, 1)
        #卷积+池化：对每个卷积核提取的特征进行池化和激活
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), 
                          dim=-1)
            for conv in self.convs], 
                             dim=1)
        #拼接所有卷积层的输出，(batch_size, sum(num_channels))
        #Dropout + 全连接层：生成分类结果
        outputs = self.decoder(self.dropout(encoding))
        return outputs
        
                                  