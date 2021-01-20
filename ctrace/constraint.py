from statistics import mean

import EoN
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Constraint, Solver, Variable, Objective

from typing import Set, Dict, Sequence, Tuple, List
import pickle as pkl
from pathlib import Path

import pandas as pd

from . import PROJECT_ROOT
np.random.seed(42)

# NOTES:
# Used variables, quaran_raw, quarantined_solution v1 -> fractional indicators
# Precondition: call solve_lp() -> ObjectiveVal
#

class ProbMinExposed:
    def __init__(self, G: nx.Graph, infected, contour1, contour2, p1, q, k, costs=None, solver=None):
        """Generates the constraints given a graph. Assumes V1, V2 are 1,2 away from I"""
        self.G = G
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
            costs = defaultdict(lambda: 1)

        self.costs = costs
        if solver is None:
            solver: Solver = pywraplp.Solver.CreateSolver('GLOP')
        self.solver = solver

        # Partial Evaluation storage
        self.partials = {}
        self.init_variables()
        self.init_constraints()

    @classmethod
    def from_dataframe(cls, G, I, contour1, contour2,
                       p1_df: pd.DataFrame, q_df: pd.DataFrame,
                       k, costs=None, solver=None):
        # Initialize p1
        p1 = {}
        for i, row in p1_df.iterrows():
            p1[row['v']] = row['p_v']

        # Initialize p2
        q = defaultdict(lambda: defaultdict(int))
        for i, row in q_df.iterrows():
            q[row['u']][row['v']] = row['q_uv']

        return cls(G, I, contour1, contour2, p1, q, k, costs, solver)

    def init_variables(self):
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

    def init_constraints(self):

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
                    coeff = self.q[u][v] * self.p1[u]
                    self.solver.Add(self.Y2[v] >= coeff * self.Y1[u])

        # Set minimization objective
        # Number of people free in V1 and people exposed in V2
        numExposed: Objective = self.solver.Objective()
        #for u in self.V1:
            # print(f"p1: {self.p1[u]}")
            #numExposed.SetCoefficient(self.Y1[u], self.p1[u])

        for v in self.V2:
            numExposed.SetCoefficient(self.Y2[v], 1)

        numExposed.SetMinimization()

    def setVariable(self, index: int, value: int):
        """Sets the ith V1 indicator to value int"""
        i = self.quaran_map[index]
        if i in self.partials:
            raise ValueError(f"in {i} is already set!")
        if value not in (0, 1):
            raise ValueError("Value must be 0 or 1")
        self.partials[i] = value
        self.solver.Add(self.X1[i] == value)

    def getVariables(self):
        return self.quaran_raw

    def filled(self):
        """Returns true if every variable is solved"""
        return set(self.partials.keys()) == set(self.V1)

    def solve_lp(self):
        """Solves the LP problem"""
        status = self.solver.Solve()
        if status == self.solver.INFEASIBLE:
            raise ValueError("Infeasible solution")
        # Indicators
        self.quarantined_solution: Dict[int, float] = {}
        self.saved_solution: Dict[int, float] = {}
        self.infected_v1: Dict[int, float] = {}
        self.infected_v2: Dict[int, float] = {}

        self.quaran_raw = np.zeros(len(self.X1))
        self.quaran_map = {}

        self.objectiveVal = 0
        for i, u in enumerate(self.V1):
            val = self.quaran_raw[i] = self.quarantined_solution[u] = self.X1[u].solution_value(
            )
            self.quaran_map[i] = u
            self.objectiveVal += (self.p1[u] * (1 - val))

        for v in self.V2:
            self.saved_solution[v] = self.X2[v].solution_value()
            self.objectiveVal += (1 - self.saved_solution[v])

class ProbMinExposedRestricted(ProbMinExposed):
    def __init__(self, G: nx.Graph, infected, contour1, contour2, p1, q, k, labels: Dict[int, int], label_limits, costs=None, solver=None):
        """
        Parameters
        ----------
        G
        infected
        contour1
        contour2
        p1
        q
        k
        labels
            A len(G.node) sized array with labels 0..L-1 for each node
        limits
            A L sized array restricting the number of people in each category.
            Must sum to less than to <= k (these constraints should be more strict than k)
        costs
        solver
        """
        self.L = len(label_limits)
        if (len(labels) != len(contour1)):
            raise ValueError("labels must match V1 size")

        if any(map(lambda x: x >= self.L or x < 0, labels.values())):
            raise ValueError("labels must correspond to label limits")

        if (sum(label_limits) > k):
            raise ValueError("sum of label_limits must be less than or equal to k to respect the k constraint")
        self.labels = labels
        self.label_limits = label_limits

        super().__init__(G, infected, contour1, contour2, p1, q, k, costs, solver)


    def init_constraints(self):
        super().init_constraints()

        # Add label constraints
        for (label, limit) in enumerate(self.label_limits):
            label_constraint: Constraint = self.solver.Constraint(0, limit)

            # Get all nodes in V1 who are label
            members = filter(lambda x: self.labels[x] == label, self.V1)
            for m in members:
                label_constraint.SetCoefficient(self.X1[m], 1)


class ProbMinExposedMIP(ProbMinExposed):
    def __init__(self, G: nx.Graph, infected, contour1, contour2, p1, q, k, costs=None, solver=None):
        if solver is None:
            solver = pywraplp.Solver.CreateSolver('SCIP')
        super().__init__(G, infected, contour1, contour2, p1, q, k, costs, solver)

    def init_variables(self):
        # V1 indicator set
        self.X1: Dict[int, Variable] = {}
        self.Y1: Dict[int, Variable] = {}

        # V2 indicator set
        self.X2: Dict[int, Variable] = {}
        self.Y2: Dict[int, Variable] = {}

        # Declare Variables (With Integer Constraints)
        for u in self.V1:
            self.X1[u] = self.solver.IntVar(0, 1, f"V1_x{u}")
            self.Y1[u] = self.solver.IntVar(0, 1, f"V1_y{u}")

        for v in self.V2:
            self.X2[v] = self.solver.NumVar(0, 1, f"V2_x{v}")
            self.Y2[v] = self.solver.NumVar(0, 1, f"V2_y{v}")


# TODO: Handle root paths
def prep_labelled_graph(in_path, out_dir, num_lines=None, delimiter=","):
    """Generates a labelled graph. Converts IDs to ids from 0 to N vertices

    Parameters
    ----------
    in_path:
        filename of graph edge-list
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


def prep_dataset(name, data_dir: Path=None, sizes=(None,)):
    """Prepares a variety of sizes of graphs from one input graph"""
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data"
    group_path = data_dir / name
    for s in sizes:
        instance_folder = f"partial{human_format(s)}" if s else "complete"
        prep_labelled_graph(in_path=group_path / f"{name}.csv", out_dir=group_path / instance_folder, num_lines=s)


def load_graph(dataset_name, graph_folder=None):
    """Will load the complete folder by default, and set the NAME attribute to dataset_name"""
    if graph_folder is None:
        graph_folder = PROJECT_ROOT / "data" / dataset_name / "complete"
    G = nx.read_edgelist(graph_folder / "data.txt", nodetype=int)

    # Set name of graph
    G.NAME = dataset_name
    return G

def load_able_graph(fp = "undirected_albe_1.90.txt"):
    graph_file = PROJECT_ROOT / "data" / fp
    df = pd.read_csv(graph_file, delim_whitespace=True)
    col1, col2 = 'Node1', 'Node2'

    # Factorize to ids from 0..len(nodes)
    factored = pd.factorize(sorted(list(df[col1]) + list(df[col2])))

    # maps from old number to new id
    num2id = dict(zip(factored[1], factored[0]))
    df[col1] = df[col1].map(lambda x: num2id[x])
    df[col2] = df[col2].map(lambda x: num2id[x])

    G = nx.from_pandas_edgelist(df, col1, col2)
    G.NAME = "albe"
    return G






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
    """Finds V1 and V2 from a graph that does not consider the excluded set"""
    v1 = union_neighbors(G, set(infected) - set(excluded),
                         set(infected) | set(excluded))
    v2 = union_neighbors(G, v1, set(v1) | set(infected) | set(excluded))
    return (v1, v2)


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

class SAA():
    def __init__(self, G: nx.Graph, infected: Set[int], p: float, k: int, numberOfSamples: int, recovered: Set[int], costs: Dict[int, int], solver=None):
        self.G = G
        self.I = infected

        # Set of intrinsic probabilities of infection
        self.p = p

        # The current budget
        self.k = k

        # Default costs of uniform
        if costs is None:
            costs = defaultdict(lambda: 1)
        self.costs = costs

        self.numberOfSamples = numberOfSamples

        if solver is None:
            solver: Solver = pywraplp.Solver.CreateSolver('GLOP')
        self.solver = solver

        # Set contours
        self.contours = find_excluded_contours(self.G, self.I, recovered)

    def init_variables(self):
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

def load_graph_cville():
    G2 = nx.Graph()
    G2.graph["name"] = "cville"
    nodes = {}
    rev_nodes = []

    file = open(PROJECT_ROOT / "data" / "undirected_albe_1.90.txt", "r")
    file.readline()
    lines = file.readlines()
    c = 0
    c_node=0

    for line in lines:

        a = line.split()
        u = int(a[1])
        v = int(a[2])

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

        G2.add_edge(u,v)
        
    return (G2, rev_nodes)


# TODO: Move MinExposed objective to ProbMinExposed class
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