# -*- coding: utf-8 -*-

from beaver.data.dataset import SumTransDataset
from beaver.data.field import Field


def build_dataset(opt, data_path, vocab_path, device, train=True):
    source_path = data_path[0]
    summary_cn_path = data_path[1]
    summary_en_path = data_path[2]

    source_field = Field(unk=True, pad=True, bos=False, eos=False)
    summary_cn_field = Field(unk=True, pad=True, bos=True, eos=True)
    summary_en_field = Field(unk=True, pad=True, bos=True, eos=True)

    cn_vocab, en_vocab = vocab_path
    source_special = source_field.special
    summary_cn_special = summary_cn_field.special
    summary_en_special = summary_en_field.special

    if opt.share_cn_embedding:
        summary_cn_special = source_special = sorted(set(source_special + summary_cn_special))

    with open(cn_vocab, encoding="UTF-8") as f:
        cn_words = [line.strip() for line in f]
    with open(en_vocab, encoding="UTF-8") as f:
        en_words = [line.strip() for line in f]

    source_field.load_vocab(cn_words, source_special)
    summary_cn_field.load_vocab(cn_words, summary_cn_special)
    summary_en_field.load_vocab(en_words, summary_en_special)

    return SumTransDataset(source_path, summary_cn_path, summary_en_path, opt.batch_size, device, train,
                           {'source': source_field, 'summary_cn': summary_cn_field, 'summary_en': summary_en_field})

