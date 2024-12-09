#!/usr/bin/env python3
""" Data Structures and Algorithms for CL III, Project
    See <https://dsacl3-2023.github.io/project/> for detailed instructions.

    Nikita L. Beklemishev, Szymon T. Kossowski
    Honour code: We pledge that the code in the Scorer class represents our own work
"""
import numpy as np
from collections import Counter
from conllu import read_conllu


class BaselineScorer:
    """A simple baseline scoring class.

    train() method keeps two counters: one for childpos-headpos pairs,
    and another for childpos-deprel.

    score() returns the product of relevant counts for a given arc.
    """

    def __init__(self):
        self.deplabels = None
        self.heads = None
        pass

    def train(self, train_conllu, dev_conllu=None):
        self.deplabels = Counter()
        self.heads = Counter()
        for sent in read_conllu(train_conllu):
            for token in sent[1:]:
                self.deplabels[(token.upos, token.deprel)] += 1
                self.heads[(token.upos, sent[token.head].upos)] += 1

    def score(self, sent, i, j, deprel):
        if self.deplabels is None or self.heads is None:
            raise Exception("The model is not trained.")
        headpos, childpos = sent[i].upos, sent[j].upos
        return self.deplabels[(childpos, deprel)] * self.heads[(childpos, headpos)]


class Scorer:
    # Assignment part 3: implement a better scorer below
    # You should keep the same interface: train() updates all
    # necessary information needed for the scoring, and score()
    # returns a score that is higher for high-probability edges.
    # You can implement any method you see fit. See the assignment
    # description for the requirements and some ideas that may help
    # implementing a simple scorer better than the baseline provided
    # above.
    #
    def __init__(self):
        self.deplabels = None  # counter of labels for a certain POS
        self.heads = None  # counter of heads of a certain POS for a certain POS
        # counter for quadruplets pos_child - dependency -
        # pos_head - whether head is before or after the child
        self.arcs = None
        # self.distances = None
        pass

    def train(self, train_conllu, dev_conllu=None):
        self.deplabels = Counter()
        self.heads = Counter()
        self.arcs = Counter()
        # self.distances = Counter()
        for sent in read_conllu(train_conllu):
            for id, token in enumerate(sent[1:]):
                self.deplabels[(token.upos, token.deprel)] += 1
                self.heads[(token.upos, sent[token.head].upos)] += 1
                # we include the branching direction together with other features as it is dependent on them
                self.arcs[(token.upos, token.deprel, sent[token.head].upos, (id > token.head))] += 1
                # self.distances[(token.upos, abs(token.head - id))] += 1

    def score(self, sent, i, j, deprel):
        if self.deplabels is None or self.heads is None:
            raise Exception("The model is not trained.")
        headpos, childpos = sent[i].upos, sent[j].upos
        unsmoothed = self.arcs[(childpos, deprel, headpos, j > i)]
        return 0.7 * unsmoothed + 0.3 * self.heads[(childpos, headpos)]

