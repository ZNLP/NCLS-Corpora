import os
import re
import subprocess
import tempfile

from beaver.utils.rouge import F_1


def calculate_bleu(hypotheses, references, lowercase=False):
    hypothesis_file = tempfile.NamedTemporaryFile(mode="w", encoding="UTF-8", delete=False)
    hypothesis_file.write("\n".join(hypotheses) + "\n")
    hypothesis_file.close()
    reference_file = tempfile.NamedTemporaryFile(mode="w", encoding="UTF-8", delete=False)
    reference_file.write("\n".join(references) + "\n")
    reference_file.close()
    return file_bleu(hypothesis_file.name, reference_file.name, lowercase)


def calculate_rouge(hypotheses, references):
    rg1list = []
    rg2list = []
    for hypo, ref in zip(hypotheses, references):
        rouge1 = F_1(hypo, ref, beta=1)
        rouge2 = F_1(hypo, ref, beta=2)
        rg1list.append(rouge1)
        rg2list.append(rouge2)
    rg1 = sum(rg1list) / len(rg1list)
    rg2 = sum(rg2list) / len(rg2list)
    return rg1 * 100, rg2 * 100


def file_bleu(hypothesis, reference, lowercase=False):
    # ../../../tools/multi-bleu.perl, so take 3 levels up.
    beaver_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    multi_bleu_path = os.path.join(beaver_path, "tools", "multi-bleu.perl")
    with open(hypothesis, "r") as read_pred, open(os.devnull, "w") as black_hole:
        bleu_cmd = ["perl", multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=black_hole).decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
        except subprocess.CalledProcessError:
            bleu_score = -1.0
        return float(bleu_score)

