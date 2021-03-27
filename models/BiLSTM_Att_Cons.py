# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random


class Config(object):
    def __init__(self, embedding):
        self.model_name = 'BiLSTM_Att_Cons'
        self.class_list = ['0', '1']
        self.vocab_path = 'data/vocab.pkl'
        self.save_path = 'saved_dict/' + self.model_name + '.ckpt'
        self.log_path = 'log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load('data/' + embedding)["embeddings"].astype('float32'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = 0.5
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.n_vocab = 0
        self.num_epochs = 30
        self.bidirectional = True
        self.pad_size = 100
        self.learning_rate = 1e-3
        self.embed = self.embedding_pretrained.size(1)
        self.hidden_size = 128
        self.num_layers = 2
        self.hidden_size2 = 64


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        torch.manual_seed(1)
        self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.randn(config.hidden_size * 2))
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        emb = self.embedding(x[0])  # [batch_size, seq_len, embeding]
        H, _ = self.lstm(emb)
        M = self.tanh1(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = H * alpha
        att = alpha.squeeze(-1)
        out = torch.sum(out, 1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc(out)
        return out, att
