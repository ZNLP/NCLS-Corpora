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

    def save(self, save_dict, step, loss_task1, loss_task2, bleu_task1, bleu_task2, rouge1_task1, rouge1_task2, rouge2_task1, rouge2_task2):
        filename = "checkpoint-step-%06d" % step
        full_filename = os.path.join(self.model_path, filename)
        self.ckpt_names.append(full_filename)
        torch.save(save_dict, full_filename)

        with open(os.path.join(self.model_path, "log"), "a", encoding="UTF-8") as log:
            log.write("%s\t" % datetime.datetime.now())
            log.write("step: %6d\t" % step)
            log.write("loss-task1: %.2f\t" % loss_task1)
            log.write("loss-task2: %.2f\t" % loss_task2)
            log.write("bleu-task1: %3.2f\t" % bleu_task1)
            log.write("bleu-task2: %3.2f\t" % bleu_task2)
            log.write("rouge1-task1: %3.2f\t" % rouge1_task1)
            log.write("rouge1-task2: %3.2f\t" % rouge1_task2)
            log.write("rouge2-task1: %3.2f\t" % rouge2_task1)
            log.write("rouge2-task2: %3.2f\t" % rouge2_task2)
            log.write("\n")

        if 0 < self.max_to_keep < len(self.ckpt_names):
            earliest_ckpt = self.ckpt_names.pop(0)
            os.remove(earliest_ckpt)
