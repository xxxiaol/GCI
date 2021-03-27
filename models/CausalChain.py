# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random


class Config(object):

    def __init__(self, embedding):
        self.model_name = 'Causalchain'
        self.vocab_path = 'data/vocab.pkl'
        self.save_path = 'saved_dict/' + self.model_name + '.ckpt'
        self.log_path = 'log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load('data/' + embedding)["embeddings"].astype('float32'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = 0.5
        self.require_improvement = 1000
        self.n_vocab = 0
        self.num_epochs = 30
        self.bidirectional = True
        self.pad_size = 8
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

        self.chain = config.chain
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=False, batch_first=True, dropout=config.dropout)
        self.embed = config.embed
        self.pad_size = config.pad_size
        self.hidden_size = config.hidden_size
        self.hidden_size2 = config.hidden_size2

        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        emb = self.embedding(x[0])
        emb = emb.view(-1, self.pad_size, self.embed)
        H, _ = self.lstm(emb)

        mask = x[1]
        mask = mask.view(-1, self.pad_size)
        mask = mask.unsqueeze(-1)
        H = H * mask

        out = torch.sum(H, dim=1)
        out = out.view(-1, self.chain, self.hidden_size)

        score = x[2]
        score = score.view(-1, self.chain, 1)
        out = torch.mul(score, out)
        out = torch.max(out, 1)[0]

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc(out)
        return out
