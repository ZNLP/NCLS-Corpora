# -*- coding: utf-8 -*-

import sys
import collections
log = sys.stderr.write


def main():
    size = int(sys.argv[1])
    counter = collections.Counter()
    for line in sys.stdin:
        counter.update(line.strip().split())
    items = counter.most_common()
    for word, _ in items[:size]:
        print(word)
    total = sum([c for _, c in items])
    appear = sum([c for _, c in items[:size]])
    log("total words: %d\n" % total)
    log("words in vocab: %d\n" % appear)
    log("vocab coverage: %.2f%%\n" % (1.0 * appear / total * 100))
    log("total unique words: %d\n" % len(items))


if __name__ == '__main__':
    main()
