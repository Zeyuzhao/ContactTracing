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

def edge_transmission(u:int, v:int, G:nx.Graph):
    #1) Does not transmit if either u or v are effectively quarantined
    #2) Otherwise: transmits with probability of transmission along edge
    
    if (G.nodes[u]['quarantine'] > 0 or G.nodes[v]['quarantine']>0):
        return 0
    else:
        if random.random() < G[u][v]['transmission']:
            return 1
        return 0

'''
Allocates budget according to the policy and demographic populations in V_1. If an invalid policy is passed, the
allocation defaults to the equal policy.
'''
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

def find_excluded_contours_edges_PQ2(G: nx.Graph, infected: Set[int], excluded: Set[int], transmission_rate: float, snitch_rate:float = 1, transmission_known: bool = True):
    P = {}
    v1_k = set()
    exclusion = (set(infected) | set(excluded))
    for v in infected:
        for nbr in effective_neighbor(G, v, G.neighbors(v)):
            if nbr not in exclusion and (random.uniform(0,1) < snitch_rate):
                v1_k.add(nbr)
                if transmission_known:
                    if nbr in P:
                        P[nbr] *= 1-G[v][nbr]["transmission"]
                    else:
                        P[nbr] = 1-G[v][nbr]["transmission"]
                else:
                    if nbr in P:
                        P[nbr] *= 1-transmission_rate
                    else:
                        P[nbr] = 1-transmission_rate
                    
    for key,value in P.items():
        P[key] = 1-value
    
    v2_k = set()
    Q = {}
    exclusion = (set(infected) | set(excluded) | set(v1_k) )
    for u in v1_k:
        for v in set(G.neighbors(u))-exclusion:
            if check_edge_transmission(G, u, v) and (random.uniform(0,1) < snitch_rate):
                if transmission_known:
                    if u in Q:
                        Q[u][v] = G[u][v]["transmission"]
                    else:
                        Q[u] = {v: G[u][v]["transmission"]}
                else:
                    if u in Q:
                        Q[u][v] = transmission_rate
                    else:
                        Q[u] = {v: transmission_rate}
                v2_k.add(v)
            else:
                if u in Q:
                    Q[u][v] = 0
                else:
                    Q[u] = {v:0}
    return v1_k, v2_k, P, Q

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

def pq_independent_simp(P_state, Q_state):
    P = {v:1 for v in P_state}
    Q = {u:{v:1 if Q_state[u][v] != 0 else 0 for v in Q_state[u].keys()} for u in Q_state.keys()}
    return P, Q

def max_neighbors(G, V_1, V_2):
    return max(len(set(G.neighbors(u)) & V_2) for u in V_1)


'''def indicatorToSet(quarantined_solution: Dict[int, int]):
    return {q for q in quarantined_solution if quarantined_solution[q] == 1}'''


# ==================================== Dataset Functions ==========================================


"""
Handles loading of datasets
"""

def load_graph_montgomery_labels():
    G = nx.Graph()
    G.NAME = "montgomery"
    
    file = open(PROJECT_ROOT / "data/graphs/montgomery_labels_all.txt", "r")
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
    file = open(PROJECT_ROOT / "data/graphs/charlottesville_labels_all.txt", "r")
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
        directory_path = PROJECT_ROOT / "data"/"graphs"/filename
        if not path.exists(directory_path):
            store_extra_edges(G_o, alpha)
        infile = open(directory_path, "r")
    else:
        G.NAME = "cville_extra"
        filename = "cville_extra_edges_" + str(alpha) + ".txt"
        directory_path = PROJECT_ROOT / "data"/"graphs"/filename
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
        outfile = open(PROJECT_ROOT / "data"/"graphs"/filename, "w")
    else:
        filename = "cville_extra_edges" + "_" + str(alpha) + ".txt"
        outfile = open(PROJECT_ROOT / "data"/"graphs"/filename, "w")
    
    file = open(PROJECT_ROOT / "data/graphs/charlottesville.txt", "r")
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


def load_graph(name, **args):
    if name == "montgomery":
        return load_graph_montgomery_labels()
    if name == "cville":
        return load_graph_cville_labels()

    if name == "montgomery_extra":
        G =  load_graph_montgomery_labels()
        return read_extra_edges(G, 0.15)
    if name == "cville_extra":
        G = load_graph_cville_labels()
        return read_extra_edges(G, 0.15)
    raise ValueError(f"{name} is not a recognized graph name.")
    

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

def segmented_allocation(sizes, budgets, carry):
    assert len(budgets) == len(sizes)
    surplus = 0

    budgets_prior = budgets[:]
    # Check for budget surpluses
    for i in range(len(sizes)):
        if budgets[i] > sizes[i]:
            surplus += (budgets[i] - sizes[i])
            budgets[i] = sizes[i]
    # Allocate excess budget
    if carry:
        for i in range(len(sizes)):
            if (budgets[i] >= sizes[i]):
                pass
            elif (sizes[i] - budgets[i]) < surplus: # Surplus not used up
                surplus -= (sizes[i] - budgets[i])
                budgets[i] = sizes[i]
            elif (sizes[i] - budgets[i]) >= surplus: # Surplus used up
                budgets[i] = budgets[i] + surplus
                surplus = 0
    assert sum(budgets_prior) == (sum(budgets) + surplus) # Surplus invariant
    return budgets
    
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

def calculateExpected(state, quarantine):
    P, Q = state.P, state.Q
    total = 0
    for v in state.V2:
        expected = 1
        for u in (set(state.G.neighbors(v)) & state.V1):
            if u not in quarantine:
                expected *= (1-P[u]*Q[u][v])
            else:
                expected *= (1-(1-state.G.nodes[u]['compliance_rate'])*P[u]*Q[u][v])
        total += (1-expected)
    return total


def calculateMILP(state, quarantine):
    P, Q = state.P, state.Q
    total = 0
    for v in state.V2:
        expected = 0
        for u in (set(state.G.neighbors(v)) & state.V1):
            if u not in quarantine:
                expected = max(expected, P[u]*Q[u][v])
            else:
                expected = max(expected, (1-state.G.nodes[u]['compliance_rate'])*P[u]*Q[u][v])
        total += expected
    return total


def calculateD(state):
    '''
    Calculates the avg(D_v) and max(D_v) for v in V2
    '''
    P, Q = state.P, state.Q
    D_arr = [len(set(state.G.neighbors(v)) & state.V1) for v in state.V2]
    if len(D_arr) == 0:
        return (0, 0)
    return (max(D_arr), sum(D_arr))


def pair_greedy(pairs, label_budgets, budget, mapper):
    """
    pairs: (obj_val, id)
    label_budgets: label -> budget
    budget: int
    mapper: id -> label
    """
    label_budgets = {k: v for k, v in label_budgets.items() if v is not None}

    result = set()
    for val, v in pairs:
        if len(result) >= budget:
            break
        label = mapper(v)
        if label not in label_budgets:
            result.add(v)
        else:
            # Add greedily until budget runs out for a particular label
            if label_budgets[label] > 0:
                result.add(v)
                label_budgets[label] -= 1
    return result

def rescale_none(pop_counts, policy, budget=None):
    """
    Rescales and allows for None in policy proportion factors.
    Filters out the nones, and performs rescaling on the remaining numbers.

    pop_counts: dict[pop_id] -> count
    policy: dict[pop_id] -> proportion factor
    """
    scales = [v for k, v in policy.items() if v is not None]
    indices = [k for k, v in policy.items() if v is not None]
    nulls = [k for k, v in policy.items() if v is None]
    total = sum(scales)
    if total == 0:
        return policy  # 0 -> 0, None -> None

    counts = rescale([pop_counts[i] for i in indices], scales, budget)
    counts = round_fracs(counts)

    # Expand filtered back into original from
    soln = {}
    for k in policy:
        if k in nulls:
            soln[k] = None
        else:
            soln[k] = counts[indices.index(k)]
    return soln

def rescale(nums, scales, budget=None):
    """
    Rescale nums with scales to budget. Budget defaults to sum(nums)

    Ex:
    nums = [10, 20]
    scale = [2, 1]
    budget = 30
    will return [15, 15]
    """
    raw = np.multiply(nums, scales)
    if np.sum(raw) == 0:
        raise ValueError("Rescaling denominator may not 0")
    if budget is None:
        budget = np.sum(nums)
    return list(raw * budget / np.sum(raw))

def round_fracs(amts):
    """
    Rounds array, ensuring that sum(result) == sum(amts)
    Using even-rounding (rounds to the nearest even for 0.5)
    All fractions are closed (accumulated) to last element.
    """
    if len(amts) == 0:
        return []
    if len(amts) == 1:
        return [round(amts[0])]
    first = [round(amt) for amt in amts[:-1]]
    return first + [int(sum(amts) - sum(first))]
