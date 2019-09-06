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

    already, hypothesis_1, hypothesis_2 = 0, [], []

    for batch in dataset:
        predictions_1, predictions_2 = beam_search(opt, model, batch.source, fields)

        hypothesis_1 += [fields["summary_cn"].decode(p) for p in predictions_1]
        hypothesis_2 += [fields["summary_en"].decode(p) for p in predictions_2]

        already += len(predictions_1)
        logging.info("Finished: %7d/%7d" % (already, dataset.num_examples))

    origin = sorted(zip(hypothesis_1, hypothesis_2, dataset.seed), key=lambda t: t[2])
    hypothesis_1 = [h for h, _, _ in origin]
    hypothesis_2 = [h for _, h, _ in origin]
    with open(opt.output[0], "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(hypothesis_1))
        out_file.write("\n")
    with open(opt.output[1], "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(hypothesis_2))
        out_file.write("\n")
    logging.info("All finished. ")


def main():
    logging.info("Build dataset...")
    dataset = build_dataset(opt, [opt.input, opt.input, opt.input], opt.vocab, device, train=False)

    fields = dataset.fields

    pad_ids = {"source": fields["source"].pad_id,
               "summary_cn": fields["summary_cn"].pad_id,
               "summary_en": fields["summary_en"].pad_id}
    vocab_sizes = {"source": len(fields["source"].vocab),
                   "summary_cn": len(fields["summary_cn"].vocab),
                   "summary_en": len(fields["summary_en"].vocab)}

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

