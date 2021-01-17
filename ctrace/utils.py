"""
Utility Functions, such as contours and PQ
"""
import math
from collections import defaultdict
from typing import Set, Iterable, Tuple, List, Dict
from statistics import mean

import EoN
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

def max_neighbors(G, V_1, V_2):
    return max(len(set(G.neighbors(u)) & V_2) for u in V_1)

def MinExposedTrial(G: nx.Graph, SIR: Tuple[List[int], List[int],
                        List[int]], contours: Tuple[List[int], List[int]], p: float, quarantined_solution: Dict[int, int]):
    """

    Parameters
    ----------
    G
        The contact tracing graph with node ids.
    SIR
        The tuple of three lists of S, I, R. Each of these lists contain G's node ids.
    contours
        A tuple of contour1, contour2.
    p
        The transition probability of infection
    to_quarantine
        The list of people to quarantine, should be a subset of contour1
    Returns
    -------
    objective_value - The number of people in v_2 who are infected.
    """
    _, I, R = SIR

    full_data = EoN.basic_discrete_SIR(G=G, p=p, initial_infecteds=I,
                                       initial_recovereds=R, tmin=0,
                                       tmax=1, return_full_data=True)

    # Update S, I, R
    I = set([k for (k, v) in full_data.get_statuses(
        time=1).items() if v == 'I'])

    R = set([k for (k, v) in full_data.get_statuses(
        time=1).items() if v == 'R'])

    to_quarantine = indicatorToSet(quarantined_solution)
    # Move quarantined to recovered
    R = list(R & to_quarantine)
    # Remove quarantined from infected
    I = [i for i in I if i not in to_quarantine]
    full_data = EoN.basic_discrete_SIR(G=G, p=p, initial_infecteds=I,
                                       initial_recovereds=R,
                                       tmin=0, tmax=1, return_full_data=True)

    # Number of people infected in V_2
    I = set([k for (k, v) in full_data.get_statuses(
        time=1).items() if v == 'I'])
    objective_value = len(set(I) & set(contours[1]))
    return objective_value

def min_exposed_objective(G: nx.Graph,
                          SIR: Tuple[List[int], List[int], List[int]],
                          contours: Tuple[List[int], List[int]],
                          p: float,
                          quarantined_solution: Dict[int, int],
                          trials=5):
    runs = [MinExposedTrial(G, SIR, contours, p, quarantined_solution) for _ in range(trials)]
    return mean(runs)

def indicatorToSet(quarantined_solution: Dict[int, int]):
    return {q for q in quarantined_solution if quarantined_solution[q] == 1}