"""
Nikita L. Beklemishev, Szymon T. Kossowski
Honour code: We pledge that this code represents our own work
"""

from mst import *
from conllu import *
from scorer import *
import pytest


# Assignment part 2: write necessary tests that would at least detect
# the three bugs in the mst.py.
#
# Please try to make your tests concise and clearly understandable.
#

@pytest.mark.parametrize("edges, expected_cycle_length", [
    # Case I - graph with a cycle
    ([(0, 2, 1), (2, 1, 1), (2, 3, 1), (2, 4, 1), (4, 5, 1), (5, 6, 1), (6, 4, 1)], 3),
    # Case II - graph without a cycle
    ([(0, 2, 1), (2, 1, 1), (2, 3, 1), (2, 4, 1), (4, 5, 1), (5, 6, 1)], 0),
    # Case III - graph without a cycle but with different structure than the one in the case II
    # Didn't know how to explain this different
    ([(0, 2, 1), (2, 1, 1), (2, 3, 1), (2, 4, 1), (4, 5, 1), (5, 6, 1), (4, 6, 1)], 0),
])
def test_cycle_detection(edges, expected_cycle_length):
    sent = [Token('<ROOT>'), Token('Twój'), Token('stary'), Token('leży'), Token('pijany'), Token('pod'),
            Token('sklepem')]
    graph = DepGraph(sent, add_edges=False)
    for parent, child, weight in edges:  # For simplicity weight is everywhere 1
        graph.add_edge(parent, child, weight)
    cycle = graph.find_cycle()
    assert len(cycle) == expected_cycle_length, f"Expected cycle length: {expected_cycle_length}, actual: {len(cycle)}"


def test_parse():
    def mock_score_fn(sent, head, child, rel):
        if rel != 'advmod':
            return 0
        if (head, child) == (0, 1):
            return 12
        if (head, child) in [(0, 2), (0, 3)]:
            return 4
        if (head, child) in [(1, 2), (3, 1)]:
            return 5
        if (head, child) == (2, 1):
            return 6
        if (head, child) in [(1, 3), (3, 2)]:
            return 7
        if (head, child) == (2, 3):
            return 8
        else:
            return 0

    sent = [Token(form="<ROOT>"), Token(form='zabij'), Token(form='ten'), Token(form='test')]
    graph = DepGraph(sent, add_edges=False)
    graph.add_edge(0, 1, 12.0, "advmod")  # <ROOT> -> zabij
    graph.add_edge(1, 3, 7, "advmod")  # zabij -> test
    graph.add_edge(3, 2, 7, "advmod")  # test -> ten

    parsed = mst_parse(sent, mock_score_fn)
    assert np.array_equal(parsed.M, graph.M)


@pytest.mark.parametrize("gold_sent,pred_sent,expected_uas,expected_las", [
    ([Token(form='ROOT', head=-1, deprel=None),
      Token(form='loves', head=0, deprel='root'),
      Token(form='Mary', head=1, deprel='nsubj')],
     [Token(form='ROOT', head=-1, deprel=None),
      Token(form='loves', head=0, deprel='root'),
      Token(form='Mary', head=1, deprel='nsubj')],
     1.0, 1.0),  # Perfect match
    ([Token(form='ROOT', head=-1, deprel=None),
      Token(form='loves', head=0, deprel='root'),
      Token(form='Mary', head=1, deprel='nsubj')],
     [Token(form='ROOT', head=-1, deprel=None),
      Token(form='loves', head=0, deprel='root'),
      Token(form='Mary', head=1, deprel='dobj')],
     1.0, 0.5),  # One correct head, one incorrect label
])
def test_evaluate(gold_sent, pred_sent, expected_uas, expected_las):
    uas, las = evaluate(gold_sent, pred_sent)
    assert uas == expected_uas and las == expected_las
