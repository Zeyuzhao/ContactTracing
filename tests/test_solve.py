import numpy as np
import networkx as nx
import random
from ctrace.solve import to_quarantine
from ctrace.constraint import load_graph
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
    assert set(degreeSol.keys()) == set(weightedSol.keys())

def test_degree_weighted_montgomery():
    # Setup montgomery graph
    G = load_graph("montgomery")
    n = len(G.nodes)
    I = [i for i in range(n) if random.random() > 0.99]
    K = 50

    _, degreeSol = to_quarantine(G=G, I0=I, safe=[], cost_constraint=K, runs=20, p=1, P=None, Q=None, method="degree")
    _, weightedSol = to_quarantine(G=G, I0=I, safe=[], cost_constraint=K, runs=20, p=1, P=None, Q=None, method="weighted")
    assert set(degreeSol.keys()) == set(weightedSol.keys())
