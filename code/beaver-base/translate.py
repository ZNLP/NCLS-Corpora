# -*- coding: utf-8 -*-
import logging

import torch
import os

from beaver.data import build_dataset
from beaver.infer import beam_search
from beaver.model import NMTModel
from beaver.utils import parseopt, get_device, calculate_bleu

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
opt = parseopt.parse_translate_args()
device = get_device()


def translate(dataset, fields, model):

    already, hypothesis, references = 0, [], []

    for batch in dataset:
        if opt.tf:
            scores = model(batch.src, batch.tgt)
            _, predictions = scores.topk(k=1, dim=-1)
        else:
            predictions = beam_search(opt, model, batch.src, fields)

        hypothesis += [fields["tgt"].decode(p) for p in predictions]
        already += len(predictions)
        logging.info("Translated: %7d/%7d" % (already, dataset.num_examples))
        references += [fields["tgt"].decode(t) for t in batch.tgt]

    if opt.bleu:
        bleu = calculate_bleu(hypothesis, references)
        logging.info("BLEU: %3.2f" % bleu)

    origin = sorted(zip(hypothesis, dataset.seed), key=lambda t: t[1])
    hypothesis = [h for h, _ in origin]
    with open(opt.output, "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(hypothesis))
        out_file.write("\n")

    logging.info("Translation finished. ")


def main():
    logging.info("Build dataset...")
    dataset = build_dataset(opt, [opt.input, opt.truth or opt.input], opt.vocab, device, train=False)

    fields = dataset.fields
    pad_ids = {"src": fields["src"].pad_id, "tgt": fields["tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab), "tgt": len(fields["tgt"].vocab)}

    # load checkpoint from model_path
    logging.info("Load checkpoint from %s." % opt.model_path)
    checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)

    logging.info("Build model...")
    model = NMTModel.load_model(checkpoint["opt"], pad_ids, vocab_sizes, checkpoint["model"]).to(device).eval()

    logging.info("Start translation...")
    with torch.set_grad_enabled(False):
        translate(dataset, fields, model)


if __name__ == '__main__':
    main()

