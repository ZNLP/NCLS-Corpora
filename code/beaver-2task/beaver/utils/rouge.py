# -*- coding: utf-8 -*-


def get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram = text_length - n
    for i in range(max_index_ngram + 1):
        ngram_set.add(tuple(text[i:i+n]))
    return ngram_set


def rouge_n(evaluated_sentences, reference_sentences, n=2):  #默认rouge_2
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        return 0

    evaluated_ngrams = get_ngrams(n, evaluated_sentences)
    reference_ngrams = get_ngrams(n, reference_sentences)
    reference_ngrams_count = len(reference_ngrams)
    if reference_ngrams_count == 0:
        return 0

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_ngrams_count = len(overlapping_ngrams)
    return overlapping_ngrams_count / reference_ngrams_count


def rouge_1(evaluated_sentences, reference_sentences):
    evaluated_sentences = evaluated_sentences.split()
    reference_sentences = reference_sentences.split()
    return rouge_n(evaluated_sentences, reference_sentences, n=1)


def rouge_2(evaluated_sentences, reference_sentences):
    evaluated_sentences = evaluated_sentences.split()
    reference_sentences = reference_sentences.split()
    return rouge_n(evaluated_sentences, reference_sentences, n=2)


def F_1(evaluated_sentences, reference_sentences, beta=1):
    evaluated_sentences = evaluated_sentences.split()
    reference_sentences = reference_sentences.split()
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        return 0

    evaluated_ngrams = get_ngrams(beta, evaluated_sentences)  # equal to retrieved set
    reference_ngrams = get_ngrams(beta, reference_sentences)  # equal to relevant set
    evaluated_ngrams_num = len(evaluated_ngrams)
    reference_ngrams_num = len(reference_ngrams)

    if reference_ngrams_num == 0 or evaluated_ngrams_num == 0:
        return 0

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_ngrams_num = len(overlapping_ngrams)
    if overlapping_ngrams_num == 0:
        return 0
    return 2*overlapping_ngrams_num / (reference_ngrams_num + evaluated_ngrams_num)
