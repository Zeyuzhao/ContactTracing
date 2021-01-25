"""
Utility Functions, such as contours and PQ
"""
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Set, Iterable, Tuple, List, Dict

import EoN
import networkx as nx
import numpy as np
import pandas as pd

from . import PROJECT_ROOT

np.random.seed(42)

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

def pq_independent(G: nx.Graph, I: Iterable[int], V1: Iterable[int], p: float):
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
    return mean(runs) #, np.std(runs, ddof=1)

def indicatorToSet(quarantined_solution: Dict[int, int]):
    return {q for q in quarantined_solution if quarantined_solution[q] == 1}



# ==================================== Dataset Functions ==========================================


"""
Handles loading of datasets
"""

def prep_labelled_graph(in_path, out_dir, num_lines=None, delimiter=","):
    """Generates a labelled graphs. Converts IDs to ids from 0 to N vertices

    Parameters
    ----------
    in_path:
        filename of graphs edge-list
    out_dir:
        path to the directory that will contain the outputs files
    num_lines:
        number of edges to parse. If None, parse entire file

    Returns
    -------
    None
        Will produce two files within out_dir, data.txt and label.txt
    """

    # ID to id
    ID = {}

    # id to ID
    vertexCount = 0

    # Input file
    if in_path is None:
        raise ValueError("in_path is needed")

    # Output path and files
    if out_dir is None:
        raise ValueError("out_dir is needed")

    # Create directory if needed
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_path = out_dir / "data.txt"
    label_path = out_dir / "label.txt"

    with open(in_path, "r") as in_file, \
            open(graph_path, "w") as out_file, \
            open(label_path, "w") as label_file:
        for i, line in enumerate(in_file):
            # Check if we reach max number of lines
            if num_lines and i >= num_lines:
                break

            split = line.split(delimiter)
            id1 = int(split[0])
            id2 = int(split[1])
            # print("line {}: {} {}".format(i, id1, id2))

            if id1 not in ID:
                ID[id1] = vertexCount
                v1 = vertexCount
                vertexCount += 1
                label_file.write(f"{id1}\n")
            else:
                v1 = ID[id1]

            if id2 not in ID:
                ID[id2] = vertexCount
                v2 = vertexCount
                vertexCount += 1
                label_file.write(f"{id2}\n")
            else:
                v2 = ID[id2]
            out_file.write(f"{v1} {v2}\n")


def human_format(num):
    """Returns a filesize-style number format"""
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'\
        .format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])\
        .replace('.', '_')


def prep_dataset(name: str, data_dir: Path=None, sizes=(None,)):
    """
    Prepares a variety of sizes of graphs from one input graphs

    Parameters
    ----------
    name
        The name of the dataset. The graphs should be contained as {data_dir}/{name}/{name}.csv
    data_dir
        The directory of graphs
    """
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data"
    group_path = data_dir / name
    for s in sizes:
        instance_folder = f"partial{human_format(s)}" if s else "complete"
        prep_labelled_graph(in_path=group_path / f"{name}.csv", out_dir=group_path / instance_folder, num_lines=s)


def load_graph(dataset_name, graph_folder=None):
    """Will load the complete folder by default, and set the NAME attribute to dataset_name"""
    if graph_folder is None:
        graph_folder = PROJECT_ROOT / "data" / "graphs" / dataset_name / "complete"
    G = nx.read_edgelist(graph_folder / "data.txt", nodetype=int)

    # Set name of graphs
    G.graph["name"] = dataset_name
    return G

def load_graph_cville(fp = "undirected_albe_1.90.txt"):
    graph_file = PROJECT_ROOT / "data" / "raw" / fp
    df = pd.read_csv(graph_file, delim_whitespace=True)
    col1, col2 = 'Node1', 'Node2'

    # Factorize to ids from 0..len(nodes)
    sort_items = sorted(list(df[col1]) + list(df[col2]))
    unique_items = list(set(sort_items))

    # maps from old number to new id
    num2id = {x: i for (i, x) in enumerate(unique_items)}
    df[col1] = df[col1].map(lambda x: num2id[x])
    df[col2] = df[col2].map(lambda x: num2id[x])

    G = nx.from_pandas_edgelist(df, col1, col2)
    G.graph["name"] = "cville"
    return G

def generate_random_absolute(G, num_infected: int = None, k: int = None, costs: list = None):
    N = G.number_of_nodes()
    if num_infected is None:
        num_infected = int(N * 0.05)
    rand_infected = np.random.choice(N, num_infected, replace=False)
    return generate_absolute(G, rand_infected, k, costs)


def generate_absolute(G, infected, k: int = None, costs: list = None):
    """Returns a dictionary of parameters for the case of infected, absolute infection"""
    N = G.number_of_nodes()

    if k is None:
        k = int(0.8 * len(infected))

    if costs is None:
        costs = np.ones(N)

    contour1, contour2 = find_contours(G, infected)

    # Assume absolute infectivity
    p1 = defaultdict(lambda: 1)

    q = defaultdict(lambda: defaultdict(lambda: 1))
    return {
        "G": G,
        "infected": infected,
        "contour1": contour1,
        "contour2": contour2,
        "p1": p1,
        "q": q,
        "costs": costs,
        "k": k,
    }

def load_sir(sir_name, sir_folder: Path=None, merge=False):
    if sir_folder is None:
        sir_folder = PROJECT_ROOT / "data" / "SIR_Cache"
    dataset_path = sir_folder / sir_name
    with open(dataset_path) as file:
        data = json.load(file)
        if merge:
            data["I"] = list(set().union(*data["I_Queue"]))
            del data["I_Queue"]
        return data

def load_sir_path(path: Path, merge=False):
    with open(path) as file:
        data = json.load(file)
        if merge:
            data["I"] = list(set().union(*data["I_Queue"]))
            del data["I_Queue"]
        return data