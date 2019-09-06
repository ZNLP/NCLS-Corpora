# -*- coding: utf-8 -*-

import torch

from beaver.infer.beam import Beam


def beam_search(opt, model, src, fields):
    batch_size = src.size(0)
    beam_size = opt.beam_size

    encoder = model.encoder

    src = src.repeat(1, beam_size).view(batch_size * beam_size, -1)
    src_pad = src.eq(fields["source"].pad_id)
    src_out = encoder(src, src_pad)

    p1 = beam_search_1(opt, batch_size, src, fields["summary_cn"], src_out, src_pad, model.cn_decoder, model.cn_generator)
    p2 = beam_search_1(opt, batch_size, src, fields["summary_en"], src_out, src_pad, model.en_decoder, model.en_generator)
    return p1, p2


def beam_search_1(opt, batch_size, src, field, src_out, src_pad, decoder, generator):
    beam_size = opt.beam_size
    device = src.device
    num_words = generator.vocab_size

    beams = [Beam(opt.beam_size, field.pad_id, field.bos_id, field.eos_id,
                  device, opt.length_penalty) for _ in range(batch_size)]

    beam_expander = (torch.arange(batch_size) * beam_size).view(-1, 1).to(device)

    previous = None

    for i in range(opt.max_length):
        if all((b.done for b in beams)):
            break

        # [batch_size x beam_size, 1]
        current_token = torch.cat([b.current_state for b in beams]).unsqueeze(-1)
        tgt_pad = current_token.eq(field.pad_id)
        out, previous = decoder(current_token, src_out, src_pad, tgt_pad, previous, i)
        previous_score = torch.stack([b.scores for b in beams]).unsqueeze(-1)
        out = generator(out).view(batch_size, beam_size, -1)

        if i < opt.min_length:
            out[:, :, field.eos_id] = -1e15

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
