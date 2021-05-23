"""
Utility Functions, such as contours and PQ
"""
import json
import math
import random
import copy
import os
from os import path
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Set, Iterable, Tuple, List, Dict, Any, TypeVar, Optional
from enum import IntEnum
from collections import UserList

import EoN
import networkx as nx
import numpy as np
import pandas as pd

from . import PROJECT_ROOT

np.random.seed(42)

np.random.seed(42)

def edge_transmission(u:int, v:int, G:nx.Graph):
    #1) Does not transmit if either u or v are effectively quarantined
    #2) Otherwise: transmits with probability of transmission along edge
    
    if (G.nodes[u]['quarantine'] > 0 or G.nodes[v]['quarantine']>0):
        return 0
    else:
        if random.random() < G[u][v]['transmission']:
            return 1
        return 0

def allocate_budget(G: nx.Graph, V1: set, budget: int, labels: list, label_map: dict, policy: str):
    distribution = []
    budget_labels = []
    
    if policy == "none": return []
    
    for i in range(len(labels)):
        distribution.append(sum([1 for n in V1 if G.nodes[n]["age_group"] == i]))

    if (policy == "old"):
        distribution[label_map['g']] *= 2
    elif (policy == "young"):
        distribution[label_map['p']] *= 2
        distribution[label_map['s']] *= 2
    elif (policy == "adult"):
        distribution[label_map['a']] *= 0.5
        distribution[label_map['o']] *= 0.5

    distribution_sum = sum(distribution)
    if distribution_sum == 0:
        return [0 for i in range(len(labels))]
    
    for i in range(len(labels)):
        budget_labels.append(math.floor(budget*distribution[i]/distribution_sum))
    return budget_labels

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

def find_excluded_contours_edges_PQ(G: nx.Graph, infected: Set[int], excluded: Set[int], discovery_rate:float = 1, snitch_rate:float = 1):
    v1 = set().union(*[effective_neighbor(G, v, G.neighbors(v)) for v in set(infected)]) - (set(infected) | set(excluded))
    v1_k = {v for v in v1 if random.uniform(0,1) < discovery_rate}
    P = {v: (1 - math.prod(1-(G[i][v]["transmission"] if check_edge_transmission(G, i, v) else 0) for i in set(set(G.neighbors(v)) & set(infected)))) for v in v1_k}
    
    v2_k = set()
    Q = {}
    exclusion = (set(infected) | set(excluded) | set(v1_k) )
    for u in v1_k:
        for v in set(G.neighbors(u))-exclusion:
            if check_edge_transmission(G, u, v) and (random.uniform(0,1) < snitch_rate):
                if u in Q:
                    Q[u][v] = G[u][v]["transmission"]
                else:
                    Q[u] = {v: G[u][v]["transmission"]}
                v2_k.add(v)
            else:
                if u in Q:
                    Q[u][v] = 0
                else:
                    Q[u] = {v:0}
    return v1_k, v2_k, P, Q
    
'''def union_neighbors(G: nx.Graph, initial: Set[int], excluded: Set[int]):
    """Finds the union of neighbors of an initial set and remove excluded"""
    total = set().union(*[G.neighbors(v) for v in initial])
    return total - excluded'''

def effective_neighbor(G: nx.Graph, infected: int, target: list, compliance_edge_known:bool = False):
    """
    Filters out edges of no transmission for G.neighbors
    """
    effective_neighbors = set()
    for v in target:
        if check_edge_transmission(G, infected, v):
            effective_neighbors.add(v)
    return effective_neighbors

def check_edge_transmission(G: nx.Graph, infected: int, target: int) -> bool:
    """
    Given node u, infected and node v, neighbor
    Does not transmit when:
    1) v or u are effectively quarantined
    """
    return not (G.nodes[infected]["quarantine"]>0 or G.nodes[target]["quarantine"]>0)

#Only know average transmission, assume uniformity
#Zeroes out the edges between u in V1 and v in V2 if they are not found during snitch rate process
def pq_independent(G: nx.Graph, I: Iterable[int], V1: Iterable[int], V2: Iterable[int], Q_state, p: float):
    # Returns dictionary P, Q
    # Calculate P, (1-P) ^ [number of neighbors in I]
    P = {v: (1 - math.pow((1 - p), len(set(G.neighbors(v)) & set(I)))) for v in set(V1)}
    Q = {}
    for key, values in Q_state.items():
        Q[key] = {v: p if values[v]!=0 else 0 for v in values.keys()}
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

'''def load_graph(dataset_name, graph_folder=None):
    """Will load the complete folder by default, and set the NAME attribute to dataset_name"""
    if graph_folder is None:
        graph_folder = PROJECT_ROOT / "data" / "graphs" / dataset_name / "complete"
    G = nx.read_edgelist(graph_folder / "data.txt", nodetype=int)

    # Set name of graphs
    G.__name__ = dataset_name
    return G'''


def load_graph_montgomery_labels():
    G = nx.Graph()
    G.NAME = "montgomery"
    
    file = open(PROJECT_ROOT / "data/graphs/montgomery/montgomery_labels_all.txt", "r")
    lines = file.readlines()
    nodes = {}
    rev_nodes = []
    edges_to_duration = {}
    cnode_to_labels = {}
    cnode_to_comp = {}
    c_node=0
    
    for line in lines:
        a = line.split(",")
        u = int(a[0])
        v = int(a[1])
        duration = int(a[2])
        age_group_u = int(a[3])
        age_group_v = int(a[4])
        compliance_u = float(a[5])
        compliance_v = float(a[6])
        
        if u in nodes.keys():
            u = nodes[u]
        else:
            nodes[u] = c_node
            rev_nodes.append(u)
            u = c_node
            c_node+=1   
    
        if v in nodes.keys():
            v = nodes[v]
        else:
            nodes[v] = c_node
            rev_nodes.append(v)
            v = c_node
            c_node+=1
        
        G.add_edge(u,v)
        edges_to_duration[(u,v)] = duration
        cnode_to_labels[u] = age_group_u
        cnode_to_labels[v] = age_group_v
        cnode_to_comp[u] = compliance_u
        cnode_to_comp[v] = compliance_v
        
    nx.set_edge_attributes(G, edges_to_duration, 'duration')
    nx.set_node_attributes(G, cnode_to_labels, 'age_group')
    nx.set_node_attributes(G, cnode_to_comp, 'compliance_rate_og')
    
    return G

def load_graph_cville_labels():
    G = nx.Graph()
    G.NAME = "cville"
    nodes = {}
    rev_nodes = []
    cnode_to_labels = {}
    cnode_to_comp = {}
    edges_to_duration = {}
    file = open(PROJECT_ROOT / "data/raw/charlottesville_labels_all.txt", "r")
    file.readline()
    lines = file.readlines()
    c = 0
    c_node=0

    for line in lines:

        a = line.split(",")
        u = int(a[0])
        v = int(a[1])
        duration = int(a[2])
        age_group_u = int(a[3])
        age_group_v = int(a[4])
        compliance_u = float(a[5])
        compliance_v = float(a[6])

        if u in nodes.keys():
            u = nodes[u]
        else:
            nodes[u] = c_node
            rev_nodes.append(u)
            u = c_node
            c_node+=1        

        if v in nodes.keys():
            v = nodes[v]
        else:
            nodes[v] = c_node
            rev_nodes.append(v)
            v = c_node
            c_node+=1

        G.add_edge(u,v)
        edges_to_duration[(u,v)] = duration
        cnode_to_labels[u] = age_group_u
        cnode_to_labels[v] = age_group_v
        cnode_to_comp[u] = compliance_u
        cnode_to_comp[v] = compliance_v

    nx.set_edge_attributes(G, edges_to_duration, 'duration')
    nx.set_node_attributes(G, cnode_to_labels, 'age_group')
    nx.set_node_attributes(G, cnode_to_comp, 'compliance_rate_og')
    
    return G;

def read_extra_edges(G_o: nx.Graph, alpha):
    G = copy.deepcopy(G_o)
    
    nx.set_edge_attributes(G, {e:False for e in G.edges()}, "added")
    
    if G.NAME == "montgomery":
        G.NAME = "montgomery_extra"
        filename = "montgomery_extra_edges_" + str(alpha) + ".txt"
        directory_path = PROJECT_ROOT / "data"/"graphs"/"montgomery"/filename
        if not path.exists(directory_path):
            store_extra_edges(G_o, alpha)
        infile = open(directory_path, "r")
    else:
        G.NAME = "cville_extra"
        filename = "cville_extra_edges_" + str(alpha) + ".txt"
        directory_path = PROJECT_ROOT / "data"/"raw"/"cville"/filename
        if not path.exists(directory_path):
            store_extra_edges(G_o, alpha)
        infile = open(directory_path, "r")
    
    lines = infile.readlines()
    for line in lines:
        a = line.split(",")
        u = int(a[0])
        v = int(a[1])
        duration = int(a[2])
        G.add_edge(u, v)
        G[u][v]['duration'] = duration
        G[u][v]['added'] = True
    
    return G;

def store_extra_edges(G, alpha):
    
    if G.NAME == "montgomery":
        filename = "montgomery_extra_edges" + "_" + str(alpha) + ".txt"
        outfile = open(PROJECT_ROOT / "data"/"graphs"/"montgomery"/filename, "w")
    else:
        filename = "cville_extra_edges" + "_" + str(alpha) + ".txt"
        outfile = open(PROJECT_ROOT / "data"/"raw"/"cville"/filename, "w")
    
    file = open(PROJECT_ROOT / "data/raw/charlottesville.txt", "r")
    file.readline()
    lines = file.readlines()
    
    durations = []
    taken_edges = {}
    node_list = [i for i in range(0,len(G.nodes))]
    
    for line in lines:
        a = line.split()
        duration = int(a[3])
        durations.append(duration)
    
    for node in G.nodes:
        if node not in taken_edges:
            taken_edges[node] = set()
        ngbrs = set(G.neighbors(node))
        for i in range(max(0, int((alpha)*len(ngbrs))-len(taken_edges[node]))):
            e = node_list[random.randint(0, len(G.nodes)-1)]
            while e in (ngbrs|taken_edges[node]|{node}):
                e = node_list[random.randint(0, len(G.nodes)-1)]
            
            if e in taken_edges:
                taken_edges[e].add(node)
            else:
                taken_edges[e] = {node}
            
            duration = durations[random.randint(0, len(durations)-1)]
            outfile.write(str(node) + "," + str(e)+ "," + str(duration) + "\n")

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

SIR = IntEnum("SIR", ["S", "I", "R"])
SEIR = IntEnum("SEIR", ["S", "E", "I", "R"])

class Partition(UserList):
    """
    DO NOT USE DIRECTLY!
    An container representing a partition of integers 0..n-1 to classes 1..k
    Stored internally as an array.
    Supports querying as .[attr], where [attr] is specified in types
    Supports imports from integer list and list of sets
    """
    type = None

    def __init__(self, other=None, size=0):
        # Stored internally as integers
        self.data: List[int]
        if other is None:
            self.type = type(self).type
            self._types = [e.name for e in self.type]
            self.data = [1] * size
        else:
            self.type = other.type
            self._types = [e.name for e in self.type]
            self.data = other.data.copy()
    # <================== Relies on cls.type [START] ==================>

    @classmethod
    def from_list(cls, l):
        """
        Copies data from a list representation into Partition container
        """
        p = cls(size=len(l))
        p.data = l.copy()
        return p

    @classmethod
    def from_sets(cls, sets):
        """
        Import data from a tupe of set indices, labels 1..k
        Union of sets must be integers [0..len()-1]
        """
        assert len(sets) == len(cls.type)
        p = cls(size=sum(map(len, sets)))
        for label, collection in enumerate(sets):
            for i in collection:
                p.data[i] = label + 1
        return p

    @classmethod
    def from_dist(cls, size, dist, rng=None):
        """
        Samples labels uniformly from distribution [dist] with [size] elements
        """
        if rng is None:
            rng = np.random.default_rng()
        raw = rng.choice(range(1, len(cls.type) + 1), size, p=dist)
        return cls.from_list(raw)

    # <================== Relies on cls.type [END] ==================>

    def __getitem__(self, item: int) -> int:
        return self.data[item]

    def __setitem__(self, key: int, value: int) -> None:
        self.data[key] = value

    def __getattr__(self, attr):
        if attr in self._types:
            return set(i for i, e in enumerate(self.data) if e == self.type[attr])
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'. \nFilter attributes are: {self._types}")

    def to_dict(self):
        return {k: v for k, v in enumerate(self.data)}

class PartitionSIR(Partition):
    type = SIR

    @classmethod
    def from_json(cls, file_name):
        raise NotImplementedError

class PartitionSEIR(Partition):
    type = SEIR

    @classmethod
    def from_json(cls, file_name):
        raise NotImplementedError

def uniform_sample(l: List[Any], p: float, rg=None):
    """Samples elements from l uniformly with probability p"""
    if rg is None:
        rg = np.random
    arr = rg.random(len(l))
    return [x for i, x in enumerate(l) if arr[i] < p]

rng = np.random.default_rng()

def pct_to_int(amt, pcts):
    """
    Distributes amt according to pcts. Last element accumulates all fractions, may be off by (n - 1)
    """
    first = [int(amt * pct) for pct in pcts[:-1]]
    return first + [amt - sum(first)]


assert pct_to_int(10, [0.5, 0.5]) == [5, 5]
assert pct_to_int(11, [0.5, 0.5]) == [5, 6]
assert pct_to_int(20, [0.33, 0.33, 0.34]) == [6, 6, 8]

