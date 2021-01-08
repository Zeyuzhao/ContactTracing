from ctrace.constraint import find_contours, find_excluded_contours, ProbMinExposedRestricted
import networkx as nx
import numpy as np


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
    """Randomly generates graphs and tests by removing the actual nodes from the graph"""
    np.random.seed(42)

    for i in range(10):
        N = 100
        G = nx.erdos_renyi_graph(N, 0.15)
        I = set(np.random.choice(N, 8, replace=False))
        excluded = set(np.random.choice(N, 20, replace=False))

        computed = find_excluded_contours(G, I, excluded)

        # Remove nodes from graph and recompute
        G.remove_nodes_from(excluded)
        expected = find_contours(G, I - excluded)

        assert expected == computed


def test_ProbMinExposedRestricted():
    ProbMinExposedRestricted(G, infected, contour1, contour2, p1, q, k, labels, label_limits, costs=None, solver=None)
