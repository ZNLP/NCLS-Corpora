import json

import torch
import os
import datetime


class Saver(object):
    def __init__(self, opt):
        self.ckpt_names = []
        self.model_path = opt.model_path + datetime.datetime.now().strftime("-%y%m%d-%H%M%S")
        self.max_to_keep = opt.max_to_keep
        os.mkdir(self.model_path)

        with open(os.path.join(self.model_path, "params.json"), "w", encoding="UTF-8") as log:
            log.write(json.dumps(vars(opt), indent=4) + "\n")

    def save(self, save_dict, step, bleu, loss):
        filename = "checkpoint-step-%06d" % step
        full_filename = os.path.join(self.model_path, filename)
        self.ckpt_names.append(full_filename)
        torch.save(save_dict, full_filename)

        with open(os.path.join(self.model_path, "log"), "a", encoding="UTF-8") as log:
            log.write("%s\t step: %6d\t loss: %.2f\t bleu: %.2f\n" % (datetime.datetime.now(), step, loss, bleu))

        if 0 < self.max_to_keep < len(self.ckpt_names):
            earliest_ckpt = self.ckpt_names.pop(0)
            os.remove(earliest_ckpt)
