# -*- coding: utf-8 -*-

from beaver.data.dataset import TranslationDataset
from beaver.data.field import Field


def build_dataset(opt, data_path, vocab_path, device, train=True):
    src = data_path[0]
    tgt = data_path[1]

    src_field = Field(unk=True, pad=True, bos=False, eos=False)
    tgt_field = Field(unk=True, pad=True, bos=True, eos=True)

    if len(vocab_path) == 1:
        # use shared vocab
        src_vocab = tgt_vocab = vocab_path[0]
        src_special = tgt_special = sorted(set(src_field.special + tgt_field.special))
    else:
        src_vocab, tgt_vocab = vocab_path
        src_special = src_field.special
        tgt_special = tgt_field.special

    with open(src_vocab, encoding="UTF-8") as f:
        src_words = [line.strip() for line in f]
    with open(tgt_vocab, encoding="UTF-8") as f:
        tgt_words = [line.strip() for line in f]

    src_field.load_vocab(src_words, src_special)
    tgt_field.load_vocab(tgt_words, tgt_special)

    return TranslationDataset(src, tgt, opt.batch_size, device, train, {'src': src_field, 'tgt': tgt_field})

