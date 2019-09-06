# -*- coding: utf-8 -*-

import os

import torch

import sys


def main():
    if len(sys.argv) != 3:
        print("python model_average.py model_path n")
        exit()

    model_path = sys.argv[1]
    n = int(sys.argv[2])  # last n model to be averaged
    fs = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.startswith("checkpoint")]
    fs = sorted(fs, reverse=True)[:n]  # last n file
    n = len(fs)  # actual file count
    cks = [torch.load(f, map_location=lambda storage, loc: storage) for f in fs]
    first_model = cks[0]["model"]  # average all weights into first model and save it
    for k, _ in first_model.items():
        for ck in cks[1:]:
            first_model[k] = (first_model[k] + ck["model"][k])
        first_model[k] = first_model[k] / n
    torch.save(cks[0], os.path.join(model_path, "averaged-%s-%s" % (fs[-1].split("-")[-1], fs[0].split("-")[-1])))


if __name__ == '__main__':
    main()
