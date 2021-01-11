import numpy as np
import networkx as nx
import random
from ctrace.solve import to_quarantine
from ctrace.constraint import load_graph
import time
# test_greedy
def test_degree_weighted_tree():
    np.random.seed(42)

    # Setup contact tracing graph
    G = nx.balanced_tree(5, 5)
    I = {0, 1}

    # Set K value
    K = 5
    _, degreeSol = to_quarantine(G=G, I0=I, safe=[], cost_constraint=K, runs=20, p=1, P=None, Q=None, method="degree")
    _, weightedSol = to_quarantine(G=G, I0=I, safe=[], cost_constraint=K, runs=20, p=1, P=None, Q=None, method="weighted")

    sol1 = {k for k,v in degreeSol.items() if v == 1}
    sol2 = {k for k,v in weightedSol.items() if v == 1}

    diffs = sol1 ^ sol2
    assert len(diffs) < 0.05 * len(degreeSol)

def test_degree_weighted_montgomery():
    # Setup montgomery graph
    G = load_graph("montgomery")
    n = len(G.nodes)
    I = [i for i in range(n) if random.random() > 0.99]
    K = 50

    start = time.time()
    _, degreeSol = to_quarantine(G=G, I0=I, safe=[], cost_constraint=K, runs=20, p=1, P=None, Q=None, method="degree")
    end1 = time.time()
    _, weightedSol = to_quarantine(G=G, I0=I, safe=[], cost_constraint=K, runs=20, p=1, P=None, Q=None, method="weighted")
    end2 = time.time()

    time1 = end1 - start
    time2 = end2 - end1
    print(f"Degree time: {time1}")
    print(f"Weighted time: {time2}")

    # Collect all the 1 indicators
    sol1 = {k for k,v in degreeSol.items() if v == 1}
    sol2 = {k for k,v in weightedSol.items() if v == 1}

    diffs = sol1 ^ sol2
    print(f"Diffs: {diffs}")

    # Differ by at most 5% of the set
    tolerance = 0.05 * len(sol1)
    assert len(diffs) < tolerance
    assert False
