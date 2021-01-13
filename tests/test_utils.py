import networkx as nx

def test_max_neighbors():
    G = nx.balanced_tree(5, 5)
    nx.draw(G, with_labels=True)
