# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn


def positional_encoding(dim, max_len=5000):
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


class Embedding(nn.Module):
    def __init__(self, embedding_dim, vocab_size, padding_idx, dropout):
        self.word_padding_idx = padding_idx
        self.embedding_dim = embedding_dim
        pe = positional_encoding(embedding_dim)
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.embedding_dim ** -0.5)

    @property
    def padding_idx(self):
        return self.word_padding_idx

    def forward(self, x, timestep=0):
        embedding = self.embedding(x) * math.sqrt(self.embedding_dim) + self.pe[timestep:timestep + x.size(1)]
        return self.dropout(embedding)
