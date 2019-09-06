# -*- coding: utf-8 -*-

import torch.cuda

from beaver.utils.metric import calculate_bleu, file_bleu
from beaver.utils.saver import Saver


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def printing_opt(opt):
    return "\n".join(["%15s | %s" % (e[0], e[1]) for e in sorted(vars(opt).items(), key=lambda x: x[0])])
