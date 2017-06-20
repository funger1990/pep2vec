#!/usr/bin/env python
from __future__ import division

__author__ = 'Fan Zhang'

import gzip
import itertools
from collections import Counter
import multiprocessing
import sys
import os
import time

import numpy as np
import scipy as sp
import itertools
from gensim.models import Word2Vec


# ---------------------------------------
alphabet = list('ACDEFGHIKLMNPQRSTVWY')


def load_corpus(infile, n_line=None):
    # load corpus
    corpus = []
    i = 0
    with gzip.open(infile) as f:
        for line in f:
            sentence = line.strip().split()
            corpus.append(sentence)
            i += 1
            if n_line is not None and i == n_line:
                break

    return corpus

def count_least_common(corpus):
    counter = Counter(list(itertools.chain(* corpus)))
    min_cnt = float('inf')
    for k, v in counter.items():
        if v < min_cnt:
            pep = k
            min_cnt = v

    return pep, min_cnt

def train_model(file_corpus, vec_size, win_size, outfile):
    beg = time.time()

    print 'loading corpus...'
    corpus = load_corpus(file_corpus)

    # train word2vec
    n_cpu = multiprocessing.cpu_count()
    print 'available CPU number:', n_cpu

    print 'training word2vec...'
    print 'vector size:', vec_size
    print 'windows size', win_size
    model = Word2Vec(corpus,
                     sg=1,  # skip-gram
                     hs=0,  # negative sampling
                     negative=5,
                     size=vec_size,
                     window=win_size,
                     alpha=0.025,   # initial learning rate
                     min_alpha=0.0001,  # linear decay to min
                     iter=5,    # epoch
                     workers=n_cpu  # multi-core
                     )
    model.save(outfile)

    end = time.time()
    print 'time in total:', (end - beg) / 60.

    return model


# --------------------------------------------
# outdir = sys.argv[1]
# vec_size = sys.argv[2]
# win_size = sys.argv[3]


file_corpus = '/mnt/g/data/uniprot/corpus/uniprot_sprot_word2vec.txt.gz'
# file_model = os.path.join(outdir, 'word2vec_v{}_w{}.model'.format(vec_size, win_size))

# print sum((len(i) for i in corpus))
# least_trimer = count_least_common(corpus)

# train model
# model = train_model(file_corpus, vec_size=200, win_size=1, outfile=file_model_v200w1)


# word ='ADA'
# print word
#
# for vec in (50, 100, 150, 200):
#     for win in [1]:
#         print 'V', vec, 'W', win
#         # print models[(vec, win)].wv.similarity('n' + word, 'c' + word)
#         print models[(vec, win)].wv.most_similar(positive = 'n' + word)[:5]

# print model.wv.most_similar_cosmul(positive='DDD')


