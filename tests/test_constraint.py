import random

from ctrace.constraint import ProbMinExposedRestricted
from ctrace.dataset import load_graph
from ctrace.utils import find_contours, find_excluded_contours
import networkx as nx
import numpy as np

from ctrace.utils import PQ_deterministic

EPSILON = 0.0001
def test_find_excluded_contours():
    np.random.seed(42)
    G: nx.Graph = nx.balanced_tree(4, 3)
    N = len(G.nodes)
    # Assert that excluded contours is equivalent to find_contours for empty excluded sets
    for i in range(10):
        I = set(np.random.choice(N, 4, replace=False))
        assert find_contours(G, I) == find_excluded_contours(G, I, set())

    # Exclude the infected set
    I = {0, 1}
    excluded = {0, 1}
    assert find_excluded_contours(G, I, excluded) == (set(), set())

    # Exclude V1 entirely except for one element
    I = {0, 1}
    excluded = {2, 3, 4, 5, 6, 7}
    assert find_excluded_contours(G, I, excluded) == ({8, }, {33, 34, 35, 36})


def test_find_excluded_contours_randomized():
    """Randomly generates graphs and tests by removing the actual nodes from the graphs"""
    np.random.seed(42)

    for i in range(10):
        N = 100
        G = nx.erdos_renyi_graph(N, 0.15)
        I = set(np.random.choice(N, 8, replace=False))
        excluded = set(np.random.choice(N, 20, replace=False))

        computed = find_excluded_contours(G, I, excluded)

        # Remove nodes from graphs and recompute
        G.remove_nodes_from(excluded)
        expected = find_contours(G, I - excluded)

        assert expected == computed


def test_restricted_respect_limits_small():
    np.random.seed(42)

    # Setup contact tracing graphs
    G = nx.balanced_tree(3, 3)
    I = {0, 1}

    # Set K value
    K = 5

    # Find contours
    contour1, contour2 = find_contours(G, I)

    # Find the infected probabilities
    p1, q = PQ_deterministic(G, list(I), contour1, 0.1)

    # Generate labels and limits for group restriction
    L = 3
    label_compact = list(np.random.randint(L, size=len(contour1))) # maps node id -> label id
    labels = dict(zip(contour1, label_compact))
    print(labels)
    label_limits = (2, 2, 1)

    # List of nodes by their label
    # maps label id -> list of nodes
    label_list = {i: list(filter(lambda x: labels[x] == i, contour1)) for i in range(len(label_limits))}

    result = ProbMinExposedRestricted(G, I, contour1, contour2, p1, q, K, labels, label_limits, costs=None, solver=None)
    result.solve_lp()

    # Assert that output respects label constraints
    label_counts = [0] * L
    for (node, value) in result.quarantined_solution.items():
        label_counts[labels[node]] += value

    for (i, (limit, count)) in enumerate(zip(label_limits, label_counts)):
        assert count <= limit


def test_restricted_respect_limits_montgomery():
    np.random.seed(42)

    G = load_graph("montgomery")
    n = len(G.nodes)
    # TODO: Make it paramterized test
    I = [i for i in range(n) if random.random() > 0.99]
    K = 500

    # Find contours
    contour1, contour2 = find_contours(G, I)

    # Find the infected probabilities
    p1, q = PQ_deterministic(G, list(I), contour1, 0.1)

    # Generate labels and limits for group restriction
    L = 3
    label_compact = list(np.random.randint(L, size=len(contour1))) # maps node id -> label id
    labels = dict(zip(contour1, label_compact))

    # print(labels)
    # TODO: Make it parameterized
    label_limits = (100, 50, 50)

    # List of nodes by their label
    # maps label id -> list of nodes
    label_list = {i: list(filter(lambda x: labels[x] == i, contour1)) for i in range(len(label_limits))}

    result = ProbMinExposedRestricted(G, I, contour1, contour2, p1, q, K, labels, label_limits, costs=None, solver=None)
    result.solve_lp()

    # Assert that output respects label constraints
    label_counts = [0] * L
    for (node, value) in result.quarantined_solution.items():
        label_counts[labels[node]] += value

    for (i, (limit, count)) in enumerate(zip(label_limits, label_counts)):
        assert count <= limit + EPSILON






