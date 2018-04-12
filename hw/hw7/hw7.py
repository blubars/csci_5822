#!/usr/bin/env python3

####################################################################
# FILE DESCRIPTION
####################################################################
# CSCI 5822 Assignment 7
#  4/11/2018
# ------------------------------------------------------------------
# TOPIC MODELING

####################################################################
#  IMPORTS
####################################################################
import sys
from math import log, exp
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import norm
from numpy.random import dirichlet, multinomial
from sklearn.decomposition import LatentDirichletAllocation

import pdb

####################################################################
#       VARIABLES AND CONSTANTS
####################################################################
random.seed(12)

vocab = [chr(ord('A') + i) for i in range(20)]
topics = [0, 1, 2]

####################################################################
#  FUNCTION DEFINITIONS
####################################################################

####################################################################
#  PART 1
###################################################################

class LdaGenerator:
    def __init__(self, alpha, beta):
        self._alpha = [alpha for _ in vocab]
        self._beta = [beta for _ in topics]

    def get_topic(self, arr):
        # utilty function, return the non-zero index
        for i,n in enumerate(arr):
            if n != 0:
                return i
        return -1

    def get_word(self, arr):
        # utility function, return letter of non-zero index
        for i,n in enumerate(arr):
            if n != 0:
                return vocab[i]
        return -1

    def print_doc(self, doc):
        for c in doc:
            sys.stdout.write(c)
        sys.stdout.write('\n')
        sys.stdout.flush()

    def to_string(self, doc):
        s = ""
        for c in doc:
            s += c + ' '
        return s

    def generate_docs(self, num_docs, length):
        ''' Generate "num_docs" documents using LDA's model.'''
        docs = []
        # draw a distribution of topics for each document: P(Z|D)
        p_z = dirichlet(self._beta, num_docs)
        self._doc_topic_prior = p_z
        # draw distribution of words for each topic: P(W|Z)
        p_w = dirichlet(self._alpha, len(topics))
        self._topic_word_prior = p_w
        for i in range(num_docs):
            doc = self.generate_doc(p_z[i], p_w, length)
            docs.append(doc)
        return docs

    def generate_doc(self, topic_dist, word_dists, length):
        ''' Generate a document using LDA. Document should be
            "length" words long. '''
        # for each doc:
        # - for each word: 
        #     - select (latent) topic P(Z|D) drawn from dirichlet 
        #       distribution with alpha prior: theta_j^(d_i)
        #     - select word_j: P(W|Z) = \thi_{w_i}^{j}, drawn from
        #       dirichlet distr with beta prior
        doc = []
        # draw a topic for each word in the document
        Z = multinomial(1, topic_dist, length)
        for j in range(length):
            # draw a topic for word_j
            z = self.get_topic(Z[j])
            # draw a word from the distribution of words for that
            # topic
            w = multinomial(1, word_dists[z])
            doc.append(self.get_word(w))
        return doc

class CountVectorizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def count(self, arr):
        vect = np.zeros(len(self._vocab))
        for i, w in enumerate(self._vocab):
            for token in arr:
                if w == token:
                    vect[i] += 1
        return vect

####################################################################
#  MAIN
###################################################################
if __name__ == "__main__":
    num_docs = 200
    doc_len = 50

    # PART 1: ------------------------------------------
    # Generate documents
    gen = LdaGenerator(0.1, 0.01)
    docs = gen.generate_docs(num_docs, doc_len)
    dtprior = gen._doc_topic_prior
    twprior = gen._topic_word_prior
    for i, doc in enumerate(docs):
        print("** DOC {} *************".format(i))
        gen.print_doc(doc)
        print("   -> Dist: {}".format(dtprior[i]))
    print("\n --- \n")
    for t in topics:
        print("** TOPIC {}: {}".format(t, twprior[t]))
        # print sorted topic words
        st = np.argsort(1-twprior[t])
        words = [vocab[st[j]] for j in range(20)]
        gen.print_doc(words)

    # PART 2: ------------------------------------------
    # Run LDA on documents, compare to generated distributions
    lda = LatentDirichletAllocation(n_components=len(topics), 
            doc_topic_prior=0.1, topic_word_prior=0.01,
            max_iter=10)
    # transform into matrix (samples x features)
    cv = CountVectorizer(vocab)
    X = np.zeros((num_docs, len(vocab)))
    for i, doc in enumerate(docs):
        X[i,:] = cv.count(doc)
    lda.fit(X)
    for i, topic in enumerate(lda.components_):
        print(topic)
        # sort by frequency
        st = np.argsort(-topic)
        # print top 5 letters
        words = [vocab[st[j]] for j in range(20)]
        gen.print_doc(words)

    print("Done!")



