"""
Utility Functions, such as contours and PQ
"""
import math
from collections import defaultdict
from typing import Set, Iterable

import networkx as nx
import numpy as np

np.random.seed(42)

def find_contours(G: nx.Graph, infected):
    """Produces contour1 and contour2 from infected"""
    N = G.number_of_nodes()

    I_SET = set(infected)
    # print(f"Infected: {I_SET}")

    # COSTS = np.random.randint(1, 20, size=N)
    COSTS = np.ones(N)
    # print(f"COSTS: {COSTS}")
    # Compute distances
    dist_dict = nx.multi_source_dijkstra_path_length(G, I_SET)

    # convert dict vertex -> distance
    # to distance -> [vertex]
    level_dists = defaultdict(set)
    for (i, v) in dist_dict.items():
        level_dists[v].add(i)

    # Set of vertices distance 1 away from infected I
    V1: Set[int] = level_dists[1]

    # Set of vertices distance 2 away from infected I
    V2: Set[int] = level_dists[2]

    return (V1, V2)


def union_neighbors(G: nx.Graph, initial: Set[int], excluded: Set[int]):
    """Finds the union of neighbors of an initial set and remove excluded"""
    total = set().union(*[G.neighbors(v) for v in initial])
    return total - excluded

def find_excluded_contours(G: nx.Graph, infected: Set[int], excluded: Set[int]):
    """Finds V1 and V2 from a graphs that does not consider the excluded set"""
    v1 = union_neighbors(G, set(infected) - set(excluded),
                         set(infected) | set(excluded))
    v2 = union_neighbors(G, v1, set(v1) | set(infected) | set(excluded))
    return (v1, v2)

def PQ_deterministic(G: nx.Graph, I: Iterable[int], V1: Iterable[int], p: float):
    # Returns dictionary P, Q
    # Calculate P, (1-P) ^ [number of neighbors in I]
    P = {v: 1 - math.pow((1 - p), len(set(G.neighbors(v)) & set(I))) for v in V1}
    Q = defaultdict(lambda: defaultdict(lambda: p))
    return P, Q