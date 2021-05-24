import pytest
from ctrace.utils import pair_greedy, rescale, rescale_none


def test_pair_greedy_no_labels():
    pairs = [(10, 5), (10, 4), (4, 2), (3, 1), (2, 0), (1, 6), (0, 3)]
    label_budgets = {0: None, 1: None}
    budget = 4
    def mapper(x): return x % 2
    assert pair_greedy(pairs, label_budgets, budget,
                       mapper) == set([5, 4, 2, 1])


def test_pair_greedy_labels():
    pairs = [(10, 5), (10, 4), (4, 2), (3, 1), (2, 0), (1, 6), (0, 3)]
    label_budgets = {0: None, 1: 2}
    budget = 5
    def mapper(x): return x % 2
    assert pair_greedy(pairs, label_budgets, budget,
                       mapper) == set([5, 4, 2, 1, 0])


def test_pair_greedy_over_budget():
    pairs = [(10, 5), (10, 4), (4, 2), (3, 1), (2, 0), (1, 6), (0, 3)]
    label_budgets = {0: None, 1: None}
    budget = 1000
    def mapper(x): return x % 2
    assert pair_greedy(pairs, label_budgets, budget,
                       mapper) == set([v for k, v in pairs])


def test_pair_greedy_label_constrained():
    pairs = [(10, 5), (10, 4), (4, 2), (3, 1), (2, 0), (1, 6), (0, 3)]
    label_budgets = {0: 1, 1: 1}
    budget = 4
    def mapper(x): return x % 2
    assert pair_greedy(pairs, label_budgets, budget,
                       mapper) == set([5, 4])


def test_rescale_no_budget():
    assert rescale([10, 20], [2, 1]) == [15, 15]
    assert rescale([10, 10], [3, 1]) == [15, 5]


def test_rescale_budget():
    assert rescale([10, 20], [2, 1], 60) == [30, 30]
    assert rescale([10, 10], [3, 1], 40) == [30, 10]


def test_rescale_single():
    assert rescale([30], [1], 40) == [40]
    assert rescale([30], [2], 40) == [40]


def test_rescale_none():
    policy1 = dict(enumerate([2, None, 1, None]))
    policy2 = dict(enumerate([2, None, None, 1]))
    assert rescale_none([10, 100, 20, 50], policy1) == dict(
        enumerate([15, None, 15, None]))
    assert rescale_none([10, 100, 50, 20], policy2) == dict(
        enumerate(([15, None, None, 15])))


def test_rescale_single():
    policy3 = dict(enumerate([2, None, None, None]))
    assert rescale_none([10, 100, 20, 50], policy3) == dict(
        enumerate([10, None, None, None]))
