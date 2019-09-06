# -*- coding: utf-8 -*-

import torch

from beaver.infer.beam import Beam


def beam_search(opt, model, src, fields):
    batch_size = src.size(0)
    beam_size = opt.beam_size
    device = src.device
    num_words = model.generator.vocab_size

    encoder = model.encoder
    decoder = model.decoder
    generator = model.generator

    beams = [Beam(opt.beam_size, fields["tgt"].pad_id, fields["tgt"].bos_id, fields["tgt"].eos_id,
                  device, opt.length_penalty) for _ in range(batch_size)]

    src = src.repeat(1, beam_size).view(batch_size*beam_size, -1)
    src_pad = src.eq(fields["src"].pad_id)
    src_out = encoder(src, src_pad)

    beam_expander = (torch.arange(batch_size) * beam_size).view(-1, 1).to(device)

    previous = None

    for i in range(opt.max_length):
        if all((b.done for b in beams)):
            break

        # [batch_size x beam_size, 1]
        current_token = torch.cat([b.current_state for b in beams]).unsqueeze(-1)
        tgt_pad = current_token.eq(fields["tgt"].pad_id)
        out, previous = decoder(current_token, src_out, src_pad, tgt_pad, previous, i)
        previous_score = torch.stack([b.scores for b in beams]).unsqueeze(-1)
        out = generator(out).view(batch_size, beam_size, -1)

        if i < opt.min_length:
            out[:, :, fields["tgt"].eos_id] = -1e15

        # find topk candidates
        scores, indexes = (out + previous_score).view(batch_size, -1).topk(beam_size)

        # find origins and token
        origins = (indexes.view(-1) // num_words).view(batch_size, beam_size)
        tokens = (indexes.view(-1) % num_words).view(batch_size, beam_size)

        for j, b in enumerate(beams):
            b.advance(scores[j], origins[j], tokens[j])

        origins = (origins + beam_expander).view(-1)
        previous = torch.index_select(previous, 0, origins)

    return [b.best_hypothesis for b in beams]
