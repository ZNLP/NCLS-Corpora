# -*- coding: utf-8 -*-
import torch


class Beam(object):

    def __init__(self, beam_size, pad, bos, eos, device, lp):

        self.size = beam_size
        self.alpha = lp

        self.scores = torch.full([beam_size], -1e20).float().to(device)
        self.scores[0] = 0.

        self.hypotheses = torch.full([1, beam_size], fill_value=pad).long().to(device)
        self.hypotheses[0][0] = bos
        self.eos = eos
        self.finished = []

    @property
    def current_state(self):
        return self.hypotheses[-1]

    def advance(self, scores, origin, tokens):
        self.scores = scores
        self.hypotheses = torch.index_select(self.hypotheses, 1, origin)
        self.hypotheses = torch.cat([self.hypotheses, tokens.unsqueeze(0)])

        for idx, tok in enumerate(self.hypotheses[-1]):
            if tok == self.eos:
                self.finished.append((self.scores[idx].clone(), self.hypotheses[1:, idx]))
                self.scores[idx] = -1e20

    @property
    def done(self):
        max_score = max([self.length_penalty(score, self.hypotheses.size(0)) for score in self.scores])
        max_finish = max([self.length_penalty(t[0], t[1].size(0)) for t in self.finished]) if self.finished else -1e20
        return bool(max_score < max_finish)

    @property
    def best_hypothesis(self):
        finished = sorted(self.finished, key=lambda t: self.length_penalty(t[0], t[1].size(0)), reverse=True)
        if not finished:
            return self.hypotheses[1:, 0]
        return finished[0][1]

    def length_penalty(self, score, length):
        return score * (6 ** self.alpha) / ((5 + length) ** self.alpha)

