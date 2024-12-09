#!/usr/bin/env python3
""" Data Structures and Algorithms for CL III, Project
    See <https://dsacl3-2023.github.io/project/> for detailed instructions.

    Nikita L. Beklemishev, Szymon T. Kossowski
    Honour code: We pledge that the code in the methods
    _find_cycle(), mst_parse() and evaluate() represents our own work
    and we elicited no unauthorised aid in debugging the code.
"""
import numpy as np
import scorer
from conllu import read_conllu

UDREL = ("acl", "advcl", "advmod", "amod", "appos", "aux", "case",
         "cc", "ccomp", "clf", "compound", "conj", "cop", "csubj",
         "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat",
         "goeswith", "iobj", "list", "mark", "nmod", "nsubj", "nummod",
         "obj", "obl", "orphan", "parataxis", "punct", "reparandum", "root",
         "vocative", "xcomp")


class DepGraph:
    """A directed graph implementation for MST parsing.

    """

    def __init__(self, sent, add_edges=True):
        self.nodes = [tok.copy() for tok in sent]
        n = len(self.nodes)
        self.M = np.zeros(shape=(n, n))
        self.deprels = [None] * n
        self.heads = [None] * n
        if add_edges:
            for i in range(1, n):
                self.add_edge(sent[i].head,
                              i, 1.0, sent[i].deprel)

    def add_edge(self, parent, child, weight=0.0, label="_"):
        self.M[parent, child] = weight
        self.deprels[child] = label
        self.nodes[child].head = parent
        self.nodes[child].deprel = label

    def remove_edge(self, parent, child, remove_deprel=True):
        self.M[parent, child] = 0.0
        if remove_deprel:
            self.deprels[child] = None

    def edge_list(self):
        """Iterate over all edges with non-zero weights.
        """
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                if self.M[i, j] != 0.0:
                    yield (self.nodes[i], self.nodes[j],
                           self.M[i, j], self.deprels[j])

    def get_children(self, node):
        for i in range(self.M.shape[1]):
            if self.M[node, i] != 0.0:
                yield i, self.M[node, i], self.deprels[i]

    def get_parents(self, node):
        for i in range(self.M.shape[0]):
            if self.M[i, node] != 0.0:
                yield i, self.M[i, node], self.deprels[node]

    def _find_cycle(self, start=0):
        """Find a cycle from the start node using an iterative DFS.
        """
        #
        # BUG 1/3: fix the bug below that does not affect
        # MST parsing, but does not detect cycles properly for a directed
        # graph in general.
        #
        # suspected bug: inclusion in visited is not a good test for a directed graph,
        # since there is no cycle in {(a, b), (a, c), (c, b)};
        # in MST, however, we only look at one incoming edge for every node,
        # therefore such cases don't occur

        stack = [start]
        visited = {start: None}
        while stack:
            node = stack.pop()
            for child, _, _ in self.get_children(node):
                # checking visited
                if child not in visited:
                    visited[child] = node
                    stack.append(child)
                # checking if the visited is on the path, i.e. reachable from itself
                else:
                    curr, path = node, [node]
                    while curr != start:
                        curr = visited[curr]
                        path.append(curr)
                    # child is in the path => loop found
                    if child in path:
                        i = path.index(child)
                        return list(reversed(path[:i + 1])), visited
                    # child was in the side branch => mark that we found one in the current branch
                    else:
                        visited[child] = node

        return [], visited

    def find_cycle(self):
        """Find and return a cycle if exists."""
        checked = set()
        for node in range(len(self.nodes)):
            if node in checked: continue
            cycle, visited = self._find_cycle(node)
            checked.update(set(visited))
            if cycle:
                return cycle
        return []

    def todot(self):
        """Return a GraphViz Digraph - can be useful for debugging."""
        dg = graphviz.Digraph()  # graph_attr={'rankdir': 'LR'})
        for head, dep, weight, deprel in self.edge_list():
            dg.edge(head.form, dep.form, label=f"{deprel}({weight:0.2f})")
        return dg


def mst_parse(sent, score_fn, deprels=UDREL):
    """Parse a given sentence with the MST algorithm.

    Note that even though we are doing labeled parsing, dependency
    labels are determined at the beginning (our scorers do not
    always assign the same score to the same label between two
    words)

    Parameters:
    sent: The input sentence represented as a sequence of Tokens.
    score_fn: A callable (function) that takes a sentence, the indices
              of parent and child nodes and a dependency label to
              assign a score to given graph edge (dependency arc).
              Note that larger the scores the better. We are
              interested in maximizing the total weight of tree edges,
              rather than minimizing.
    """
    #
    # BUG 2/3: fix the bug below that may result in a wrong MST.
    #
    # suspected bug:
    # the loops are not broken - the new incoming edge to the loop should replace the old cycled edge
    # so the best edge should be removed
    #
    n = len(sent)
    mst = DepGraph(sent, add_edges=False)
    for child in range(1, n):
        maxscore, besthead, bestrel = 0.0, None, None
        for head in range(n):
            # if child != head:  # check that it is not a self-pointing node
            for rel in deprels:
                score = score_fn(sent, head, child, rel)
                if score > maxscore:
                    maxscore, besthead, bestrel = score, head, rel
        mst.add_edge(besthead, child, maxscore, bestrel)
    #
    cycle = mst.find_cycle()
    removed = set()
    while len(cycle):
        minloss, bestu, bestv, oldp = float('inf'), None, None, None
        for v in cycle:
            parent, _, _ = list(mst.get_parents(v))[0]
            deprel = mst.deprels[v]
            weight = score_fn(sent, parent, v, deprel)
            for u in range(n):
                if u == v or u in cycle or (u, v) in removed:
                    continue
                uw = score_fn(sent, u, v, deprel)
                if weight - uw < minloss:
                    minloss = weight - uw
                    oldhead = parent
                    bestu, bestv, bestw, bestrel = u, v, uw, deprel
        mst.remove_edge(oldhead, bestv, remove_deprel=True)  # added line
        removed.add((oldhead, bestv))
        mst.add_edge(bestu, bestv, bestw, bestrel)
        cycle = list(mst.find_cycle())
    return mst


def evaluate(gold_sent, pred_sent):
    """Calculate and return labeled and unlabeled attachment scores."""
    #
    # BUG 3/3: fix a trivial bug below that results in
    # wrong calculation of the evaluation metric(s).
    #
    # suspected bug: the metric LAS should also include wrongly assigned dependencies, not only the labels
    #
    assert len(gold_sent) == len(pred_sent)
    n = len(gold_sent) - 1
    uas, las = 0, 0
    for i, gold in enumerate(gold_sent[1:], start=1):
        pred = pred_sent[i]
        if gold.head == pred.head:
            uas += 1
            if gold.deprel == pred.deprel:  # changed line
                las += 1
    return uas / n, las / n


if __name__ == "__main__":
    # The following shows example usage:
    import os
    import glob
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('treebank', help="Directory that holds the UD treebank.")
    args = ap.parse_args()

    train_file = glob.glob(os.path.join(args.treebank, '*-ud-train.conllu'))[0]
    dev_file = glob.glob(os.path.join(args.treebank, '*-ud-dev.conllu'))[0]
    test_file = glob.glob(os.path.join(args.treebank, '*-ud-test.conllu'))[0]

    # Train the scored on the training file.
    # The baseline scorer does not use the development set. You may
    # want to use it in case you use a method that needs to tune
    # hyperparameters.
    sc = scorer.Scorer()
    sc.train(train_file, dev_file)

    # Calculate UAS / LAS on all sentences, report the average UAS and
    # LAS per sentence (macro average)
    uassum, lassum, n = 0, 0, 0
    for sent in read_conllu(test_file):
        mst = mst_parse(sent, score_fn=sc.score)
        parsed = mst.nodes
        uas, las = evaluate(sent, parsed)
        uassum += uas
        lassum += las
        n += 1

    print("Macro averaged UAS:", uassum / n)
    print("Macro averaged LAS:", lassum / n)

# Example usage of 'todot()' method:
#    mst.todot().save('pred.dot')
#    DepGraph(sent).todot().save('gold.dot')
#    mst.todot().savefig('pred.pdf')
