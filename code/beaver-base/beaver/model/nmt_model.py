# -*- coding: utf-8 -*-
from typing import Dict

import torch
import torch.nn as nn

from beaver.model.embeddings import Embedding
from beaver.model.transformer import Decoder, Encoder


class Generator(nn.Module):
    def __init__(self, hidden_size: int, tgt_vocab_size: int):
        self.vocab_size = tgt_vocab_size
        super(Generator, self).__init__()
        self.linear_hidden = nn.Linear(hidden_size, tgt_vocab_size)
        self.lsm = nn.LogSoftmax(dim=-1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_hidden.weight)

    def forward(self, dec_out):
        score = self.linear_hidden(dec_out)
        lsm_score = self.lsm(score)
        return lsm_score


class NMTModel(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, generator: Generator):
        super(NMTModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src, tgt):
        tgt = tgt[:, :-1]  # shift left
        src_pad = src.eq(self.encoder.embedding.word_padding_idx)
        tgt_pad = tgt.eq(self.decoder.embedding.word_padding_idx)

        enc_out = self.encoder(src, src_pad)
        decoder_outputs, _ = self.decoder(tgt, enc_out, src_pad, tgt_pad)
        scores = self.generator(decoder_outputs)
        return scores

    @classmethod
    def load_model(cls, model_opt,
                   pad_ids: Dict[str, int],
                   vocab_sizes: Dict[str, int],
                   checkpoint=None):
        src_embedding = Embedding(embedding_dim=model_opt.hidden_size,
                                  dropout=model_opt.dropout,
                                  padding_idx=pad_ids["src"],
                                  vocab_size=vocab_sizes["src"])

        if len(model_opt.vocab) == 2:
            tgt_embedding = Embedding(embedding_dim=model_opt.hidden_size,
                                      dropout=model_opt.dropout,
                                      padding_idx=pad_ids["tgt"],
                                      vocab_size=vocab_sizes["tgt"])
        else:
            # use shared word embedding for source and target
            tgt_embedding = src_embedding

        encoder = Encoder(model_opt.layers,
                          model_opt.heads,
                          model_opt.hidden_size,
                          model_opt.dropout,
                          model_opt.ff_size,
                          src_embedding)

        decoder = Decoder(model_opt.layers,
                          model_opt.heads,
                          model_opt.hidden_size,
                          model_opt.dropout,
                          model_opt.ff_size,
                          tgt_embedding)

        generator = Generator(model_opt.hidden_size, vocab_sizes["tgt"])

        model = cls(encoder, decoder, generator)

        if model_opt.train_from and checkpoint is None:
            checkpoint = torch.load(model_opt.train_from, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint["model"])
        elif checkpoint is not None:
            model.load_state_dict(checkpoint)
        return model
