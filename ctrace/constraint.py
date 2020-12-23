import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Constraint, Solver, Variable, Objective

from typing import Set, Dict, Sequence
import pickle as pkl
from pathlib import Path

import pandas as pd

class ProbMinExposed:
    def __init__(self, graph: nx.Graph, infected, contour1, contour2, p1, q, k, costs=None, solver=None):
        """Generates the constraints given a graph. Assumes V1, V2 are 1,2 away from I"""
        self.G = graph
        self.I = infected
        self.V1 = contour1
        self.V2 = contour2
        # Set of intrinsic probabilities of infection
        self.p1 = p1
        # Dictionary: q[u][v] = conditional probability p(v is infected | u is infected)
        self.q = q
        self.k = k

        # Default costs of uniform
        if costs is None:
            costs = np.ones(len(self.V1))

        self.costs = costs
        if solver is None:
            solver: Solver = pywraplp.Solver.CreateSolver('GLOP')
        self.solver = solver

        # Partial Evaluation storage
        self.partials = {}
        self.init()

    def init(self):
        # V1 indicator set
        self.X1: Dict[int, Variable] = {}
        self.Y1: Dict[int, Variable] = {}

        # V2 indicator set
        self.X2: Dict[int, Variable] = {}
        self.Y2: Dict[int, Variable] = {}

        # Declare Variables
        for u in self.V1:
            self.X1[u] = self.solver.NumVar(0, 1, f"V1_x{u}")
            self.Y1[u] = self.solver.NumVar(0, 1, f"V1_y{u}")

        for v in self.V2:
            self.X2[v] = self.solver.NumVar(0, 1, f"V2_x{v}")
            self.Y2[v] = self.solver.NumVar(0, 1, f"V2_y{v}")

        # First set of constraints X + Y = 1
        # By definition, X and Y sum to 1

        # Quarantine (x) / Free (y) Indicators
        # Parameter indicators (we have control)
        for u in self.V1:
            self.solver.Add(self.X1[u] + self.Y1[u] == 1)

        # Safe (x) / Exposed (y) Indicators
        # Result indicators (we have no control)
        for v in self.V2:
            self.solver.Add(self.X2[v] + self.Y2[v] == 1)

        # Second set of constraints: k (cost) constraint
        # The cost of quarantine is a linear combination
        cost: Constraint = self.solver.Constraint(0, self.k)
        for u in self.V1:
            # For now, the coefficient of every variable is 1 (The cost is uniform)
            cost.SetCoefficient(self.X1[u], int(self.costs[u]))

        # Third set of constraints: specify who is considered "saved"
        # (anyone "not exposed" must have no contact)

        # or, once v in V1 is exposed (Y1 = 1),
        # all v's neighbors in V2 must be exposed (Y2 >= Y1 = 1)

        # We only examine edges between sets V1 and V2
        for u in self.V1:
            for v in self.G.neighbors(u):
                if v in self.V2:
                    coeff = (self.q[u][v] * self.p1[u])
                    self.solver.Add(self.Y2[v] >= coeff * self.Y1[u])

        # Set minimization objective
        # Number of people free in V1 and people exposed in V2
        numExposed: Objective = self.solver.Objective()
        for u in self.V1:
            numExposed.SetCoefficient(self.Y1[u], self.p1[u])

        for v in self.V2:
            numExposed.SetCoefficient(self.Y2[v], 1)

        numExposed.SetMinimization()

    def setVariable(self, i: int, value: int):
        """Sets the ith V1 indicator to value int"""
        if i in self.partials:
            raise ValueError(f"Index {i} is already set!")
        if value not in (0, 1):
            raise ValueError("Value must be 0 or 1")
        self.partials[i] = value
        self.solver.Add(self.X1[i] == value)

    def getVariables(self):
        return self.quaran_sol

    # Deprecated
    def raw_soln(self):
        return self.quaran_sol

    def solve_lp(self):
        """Solves the LP problem"""
        status = self.solver.Solve()
        if status == self.solver.INFEASIBLE:
            raise ValueError("Infeasible solution")
        # Indicators
        self.quaran_sol: Dict[int, float] = {}
        self.safe_sol: Dict[int, float] = {}

        for u in self.V1:
            self.quaran_sol[u] = self.X1[u].solution_value()

        for v in self.V2:
            self.safe_sol[v] = self.X2[v].solution_value()



def prep_labelled_graph(data_name, in_path=None, out_dir=None, num_lines=None):
    """Generates a labelled graph. Converts IDs to ids from 0 to N vertices

    Parameters
    ----------
    data_name:
        name of the dataset and directory
    in_path:
        filename of graph edge-list
    out_dir:
        path to the containing directory
    num_lines:
        number of edges to parse. If None, parse entire file
    """

    # ID to id
    ID = {}

    # id to ID
    vertexCount = 0

    # Input file
    if in_path is None:
        in_path = "../data/mont/montgomery.csv"

    # Output path and files
    if out_dir is None:
        out_dir = f"../data/mont/labelled/{data_name}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    graph_path = f"{out_dir}/data.txt"
    label_path = f"{out_dir}/label.txt"

    delimiter = ","
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

def prep_dataset(in_path=None, out_dir=None, sizes=(100, 1000, 5000, 10000, None)):
    """Prepares a variety of sizes of graphs from one input graph"""
    for s in sizes:
        name = f"mont{human_format(s)}" if s else "montgomery"
        prep_labelled_graph(data_name= name, in_path=in_path, out_dir=out_dir, num_lines=s)

def load_graph(dataset_name, in_dir="../data/mont/labelled"):
    return nx.read_edgelist(f"{in_dir}/{dataset_name}/data.txt", delimiter=",", nodetype=int)

def load_auxillary(directory):
    """loads in infected, contour1, contour2, p1, q, k, and costs from directory"""
    pass

def find_coutours(G: nx.Graph, infected):
    """Produces contour1 and contour2 from infected"""
    N = G.number_of_nodes()

    I_SET = set(infected)
    print(f"Infected: {I_SET}")

    # COSTS = np.random.randint(1, 20, size=N)
    COSTS = np.ones(N)
    print(f"COSTS: {COSTS}")
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

def generate_absolute(G, num_infected: int = None, k : int = None, costs : list = None):
    """Returns a dictionary of parameters for the case of infected, absolute infection"""
    N = G.number_of_nodes()
    infected = np.random.choice(N, size=num_infected, replace=False)
    contour1, contour2 = find_coutours(G, infected)

    # Assume absolute infectivity
    p1 = np.ones(len(contour1))

    q = defaultdict(lambda: defaultdict(int))

    if k is None:
        k = 0.8 * len(infected)
    if costs is None:
        costs = np.ones(N)
    return {
        "G": G,
        "infected": infected,
        "contour1": contour1,
        "contour2": contour2,
        "p1": p1,
        "q": q,
        "costs": costs,
    }

def draw_absolute(G: nx.Graph, I, V1, V2, quarantined, safe, name=""):
    status = []
    N = G.number_of_nodes()
    for i in range(N):
        if i in V1:
            c = "blue" if (i in quarantined) else "orange"
        elif i in V2:
            c = "green" if (i in safe) else "yellow"
        elif i in I:
            c = "red"
        else:
            c = "grey"
        status.append(c)
        G.nodes[i]["color"] = c

    # Small

    if N < 20:
        pos = nx.spring_layout(G)
        dist_params = {
            "pos": pos,
            "node_color": status,
            "with_labels": True,
        }
        nx.draw_networkx(G, **dist_params)
    else:
        # large
        pos = nx.nx_agraph.graphviz_layout(G, prog="sfdp", args="-Goverlap=false")

        dist_params = {
            "pos": pos,
            "node_color": status,
            "with_labels": False,
            "node_size": 1,
            "width": 0.3
        }
        nx.draw_networkx(G, **dist_params)
        nx.draw_networkx_labels(G, pos, font_size=1)

    if name is None:
        name = "graph"

    plt.savefig(f'../output/{name}.png', dpi=1000)
    plt.show()

if __name__ == '__main__':
    G = load_graph("mont100")
    params = generate_absolute(G)











