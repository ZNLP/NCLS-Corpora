# -*- coding: utf-8 -*-
import logging

import torch
import torch.cuda

from beaver.data import build_dataset
from beaver.infer import beam_search
from beaver.loss import WarmAdam, LabelSmoothingLoss
from beaver.model import NMTModel
from beaver.utils import Saver
from beaver.utils import calculate_bleu
from beaver.utils import parseopt, get_device, printing_opt

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
opt = parseopt.parse_train_args()

device = get_device()

logging.info("\n" + printing_opt(opt))

saver = Saver(opt)


def valid(model, criterion, valid_dataset, step):
    model.eval()
    total_loss, total = 0.0, 0

    hypothesis, references = [], []

    for batch in valid_dataset:
        scores = model(batch.src, batch.tgt)
        loss = criterion(scores, batch.tgt)
        total_loss += loss.data
        total += 1

        if opt.tf:
            _, predictions = scores.topk(k=1, dim=-1)
        else:
            predictions = beam_search(opt, model, batch.src, valid_dataset.fields)

        hypothesis += [valid_dataset.fields["tgt"].decode(p) for p in predictions]
        references += [valid_dataset.fields["tgt"].decode(t) for t in batch.tgt]

    bleu = calculate_bleu(hypothesis, references)
    logging.info("Valid loss: %.2f\tValid BLEU: %3.2f" % (total_loss / total, bleu))
    checkpoint = {"model": model.state_dict(), "opt": opt}
    saver.save(checkpoint, step, bleu, total_loss / total)


def train(model, criterion, optimizer, train_dataset, valid_dataset):
    total_loss = 0.0
    model.zero_grad()
    for i, batch in enumerate(train_dataset):
        scores = model(batch.src, batch.tgt)
        loss = criterion(scores, batch.tgt)
        loss.backward()
        total_loss += loss.data

        if (i + 1) % opt.grad_accum == 0:
            optimizer.step()
            model.zero_grad()

            if optimizer.n_step % opt.report_every == 0:
                mean_loss = total_loss / opt.report_every / opt.grad_accum
                logging.info("step: %7d\t loss: %7f" % (optimizer.n_step, mean_loss))
                total_loss = 0.0

            if optimizer.n_step % opt.save_every == 0:
                with torch.set_grad_enabled(False):
                    valid(model, criterion, valid_dataset, optimizer.n_step)
                model.train()
        del loss


def main():
    logging.info("Build dataset...")
    train_dataset = build_dataset(opt, opt.train, opt.vocab, device, train=True)
    valid_dataset = build_dataset(opt, opt.valid, opt.vocab, device, train=False)
    fields = valid_dataset.fields = train_dataset.fields
    logging.info("Build model...")

    pad_ids = {"src": fields["src"].pad_id, "tgt": fields["tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab), "tgt": len(fields["tgt"].vocab)}

    model = NMTModel.load_model(opt, pad_ids, vocab_sizes).to(device)
    criterion = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["tgt"], pad_ids["tgt"]).to(device)

    n_step = int(opt.train_from.split("-")[-1]) if opt.train_from else 1
    optimizer = WarmAdam(model.parameters(), opt.lr, opt.hidden_size, opt.warm_up, n_step)

    logging.info("start training...")
    train(model, criterion, optimizer, train_dataset, valid_dataset)


if __name__ == '__main__':
    main()
