# -*- coding: utf-8 -*-
from typing import List

import torch

EOS_TOKEN = "<eos>"
BOS_TOKEN = "<bos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"


class Field(object):
    def __init__(self, bos: bool, eos: bool, pad: bool, unk: bool):
        self.bos_token = BOS_TOKEN if bos else None
        self.eos_token = EOS_TOKEN if eos else None
        self.unk_token = UNK_TOKEN if unk else None
        self.pad_token = PAD_TOKEN if pad else None

        self.vocab = None

    def load_vocab(self, words: List[str], specials: List[str]):
        self.vocab = Vocab(words, specials)

    def process(self, batch, device):

        max_len = max(len(x) for x in batch)

        padded, length = [], []

        for x in batch:
            bos = [self.bos_token] if self.bos_token else []
            eos = [self.eos_token] if self.eos_token else []
            pad = [self.pad_token] * (max_len - len(x))

            padded.append(bos + x + eos + pad)
            length.append(len(x) + len(bos) + len(eos))

        padded = torch.tensor([self.encode(ex) for ex in padded])

        return padded.long().to(device)

    def encode(self, tokens):
        ids = []
        for tok in tokens:
            if tok in self.vocab.stoi:
                ids.append(self.vocab.stoi[tok])
            else:
                ids.append(self.unk_id)
        return ids

    def decode(self, ids):
        tokens = []
        for tok in ids:
            tok = self.vocab.itos[tok]
            if tok == self.eos_token:
                break
            if tok == self.bos_token:
                continue
            tokens.append(tok)
        return " ".join(tokens).replace("@@ ", "").replace("@@", "")

    @property
    def special(self):
        return [tok for tok in [self.unk_token, self.pad_token, self.bos_token, self.eos_token] if tok is not None]

    @property
    def pad_id(self):
        return self.vocab.stoi[self.pad_token]

    @property
    def eos_id(self):
        return self.vocab.stoi[self.eos_token]

    @property
    def bos_id(self):
        return self.vocab.stoi[self.bos_token]

    @property
    def unk_id(self):
        return self.vocab.stoi[self.unk_token]


class Vocab(object):
    def __init__(self, words: List[str], specials: List[str]):
        self.itos = specials + words
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

