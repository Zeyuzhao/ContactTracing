import numpy as np
import networkx as nx
from ctrace.solve import to_quarantine
# test_greedy
def test_degree_weighted():
    np.random.seed(42)

    # Setup contact tracing graph
    G = nx.balanced_tree(6, 6)
    I = {0, 1}

    # Set K value
    K = 5
    _, degreeSol = to_quarantine(G=G, I0=I, safe=[], cost_constraint=K, runs=20, p=1, P=None, Q=None, method="degree")
    _, weightedSol = to_quarantine(G=G, I0=I, safe=[], cost_constraint=K, runs=20, p=1, P=None, Q=None, method="weighted")
    assert set(degreeSol.keys()) == set(weightedSol.keys())