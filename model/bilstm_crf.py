'''

 Filename:  bilstm-crf.py
 Description:  bilstm-crf模型结构
 Created:  2022年05月2日 10时20分
 Author:  Li Shao

'''

'''
input_ids:                       (batch, seq)
--> Embedding:                   (batch, seq, embedding_dim)
--> LSTM:                        (batch, seq, hidden_dim)
--> Dropout
--> Linear:                      (batch, seq, num_tags)
--> CRF
训练时：模型直接调用log_likelihood方法计算损失，然后进行训练
此时模型内部，线性层的输出和目标 一起经过 CRF 的前向计算forward得到损失
预测时：模型 前向计算forward 获得输出序列
此时模型内部，线性层的输出经过 CRF 的解码方法计算得到预测标签序列
'''

import sys
sys.path.append('..')
import torch
import torch.nn as nn
from TorchCRF import CRF
from config.config import Config

class BiLSTM_CRF(nn.Module):
    def __init__(self, config):
        super(BiLSTM_CRF, self).__init__()
        
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.num_tags = config.num_tags
        # self.tag_to_ix = tag2id
        self.embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim // 2,
            num_layers=config.num_workers,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(self.hidden_dim, self.num_tags)
        # CRF层
        self.crf = CRF(self.num_tags)

    def forward(self, x, mask):
        embeddings = self.embeds(x)
        outputs, _ = self.lstm(embeddings)
        outputs = self.dropout(outputs) 
        outputs = self.linear(outputs)
        outputs = self.crf.viterbi_decode(outputs, mask)
        return outputs

    def log_likelihood(self, x, labels, mask):
        embeddings = self.embeds(x)
        outputs, _ = self.lstm(embeddings)
        outputs = self.dropout(outputs)
        outputs = self.linear(outputs)
        loss = -self.crf.forward(outputs, labels, mask)
        return torch.sum(loss)

if __name__ == '__main__':
    config = Config()
    config.embedding_dim = 64
    config.hidden_dim = 20
    config.vocab_size = 50
    config.num_tags = 5
    config.dropout = 0.1
    device = config.device

    model = BiLSTM_CRF(config).to(device)
    x = torch.LongTensor([[1, 12, 31, 4, 15], [2, 18, 39, 14, 0]]).to(device)
    mask = torch.BoolTensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]).to(device)
    labels = torch.LongTensor([[1, 2, 1, 3, 0], [2, 4, 2, 1, 0]]).to(device)
    print(model.forward(x, mask))
    print(model.log_likelihood(x, labels, mask))