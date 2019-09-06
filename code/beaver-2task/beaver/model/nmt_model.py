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

    def __init__(self, encoder: Encoder,
                 cn_decoder: Decoder,
                 en_decoder: Decoder,
                 cn_generator: Generator,
                 en_generator: Generator):
        super(NMTModel, self).__init__()

        self.encoder = encoder
        self.cn_decoder = cn_decoder
        self.en_decoder = en_decoder
        self.cn_generator = cn_generator
        self.en_generator = en_generator

    def forward(self, source, summary_cn, summary_en):
        summary_cn = summary_cn[:, :-1]  # shift left
        summary_en = summary_en[:, :-1]  # shift left
        source_pad = source.eq(self.encoder.embedding.word_padding_idx)
        summary_cn_pad = summary_cn.eq(self.cn_decoder.embedding.word_padding_idx)
        summary_en_pad = summary_en.eq(self.en_decoder.embedding.word_padding_idx)

        enc_out = self.encoder(source, source_pad)

        cn_decoder_outputs, _ = self.cn_decoder(summary_cn, enc_out, source_pad, summary_cn_pad)
        en_decoder_outputs, _ = self.en_decoder(summary_en, enc_out, source_pad, summary_en_pad)
        cn_scores = self.cn_generator(cn_decoder_outputs)
        en_scores = self.en_generator(en_decoder_outputs)
        return cn_scores, en_scores

    @classmethod
    def load_model(cls, model_opt,
                   pad_ids: Dict[str, int],
                   vocab_sizes: Dict[str, int],
                   checkpoint=None):
        source_embedding = Embedding(embedding_dim=model_opt.hidden_size,
                                     dropout=model_opt.dropout,
                                     padding_idx=pad_ids["source"],
                                     vocab_size=vocab_sizes["source"])

        summary_en_embedding = Embedding(embedding_dim=model_opt.hidden_size,
                                         dropout=model_opt.dropout,
                                         padding_idx=pad_ids["summary_en"],
                                         vocab_size=vocab_sizes["summary_en"])

        if model_opt.share_cn_embedding:
            summary_cn_embedding = source_embedding
        else:
            summary_cn_embedding = Embedding(embedding_dim=model_opt.hidden_size,
                                             dropout=model_opt.dropout,
                                             padding_idx=pad_ids["summary_cn"],
                                             vocab_size=vocab_sizes["summary_cn"])

        encoder = Encoder(model_opt.layers,
                          model_opt.heads,
                          model_opt.hidden_size,
                          model_opt.dropout,
                          model_opt.ff_size,
                          source_embedding)

        cn_decoder = Decoder(model_opt.layers,
                             model_opt.heads,
                             model_opt.hidden_size,
                             model_opt.dropout,
                             model_opt.ff_size,
                             summary_cn_embedding)

        en_decoder = Decoder(model_opt.layers,
                             model_opt.heads,
                             model_opt.hidden_size,
                             model_opt.dropout,
                             model_opt.ff_size,
                             summary_en_embedding)

        cn_generator = Generator(model_opt.hidden_size, vocab_sizes["summary_cn"])
        en_generator = Generator(model_opt.hidden_size, vocab_sizes["summary_en"])

        model = cls(encoder, cn_decoder, en_decoder, cn_generator, en_generator)
        if checkpoint is None and model_opt.train_from:
            checkpoint = torch.load(model_opt.train_from, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint["model"])
        elif checkpoint is not None:
            model.load_state_dict(checkpoint)
        return model
