# %%
import pstats
import io
import cProfile
import ipywidgets as widgets

import numpy as np
from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Variable, Constraint, Objective, Solver
import networkx as nx

from ctrace.simulation import *
from ctrace.dataset import *
from ctrace.recommender import *

from ctrace.problem import *
from ctrace.utils import *
from ctrace.drawing import *

import scipy

from enum import IntEnum
from typing import Dict, Set, List, Any, TypeVar
from collections import UserList, namedtuple, defaultdict
from tqdm import tqdm

# Declare SIR Enum


class SIR:
    S = 1
    I = 2
    R = 3


T = TypeVar('T', bound='PartitionSIR')


class PartitionSIR(UserList):
    """
    Stored internally as an array.
    Supports querying as .S, .I, .R
    Supports imports from different formats
    """

    def __init__(self, lst=None, size=0):
        # Stored internally as integers
        self._types = ["S", "I", "R"]
        self.type = IntEnum("type", self._types)

        self.data: List[int]
        if lst is not None:
            self.data = lst.data.copy()
        else:
            self.data = [SIR.S] * size

    @classmethod
    def from_list(cls, l):
        p = PartitionSIR()
        p.data = l.copy()
        return p

    @classmethod
    def from_dict(cls, n: Optional[int], d: Dict[int, int]) -> T:
        if n is None:
            n = len(d)
        p = PartitionSIR(n)
        for k, v in d.items():
            p[k] = v
        return p

    @classmethod
    def from_sets(cls, S, I, R):
        p = PartitionSIR(size=len(S) + len(I) + len(R))
        for v in S:
            p[v] = SIR.S
        for v in I:
            p[v] = SIR.I
        for v in R:
            p[v] = SIR.R
        return p

    @classmethod
    def from_I(cls, I, size):
        p = PartitionSIR(size=size)
        for v in I:
            p[v] = SIR.I
        return p

    def __getitem__(self, item: int) -> int:
        return self.data[item]

    def __setitem__(self, key: int, value: int) -> None:
        self.data[key] = value

    def to_dict(self):
        return {k: v for k, v in enumerate(self.data)}

    # Only use the following as generators
    # To check for individual status

    @property
    def S(self):
        return set(i for i, e in enumerate(self.data) if e == SIR.S)

    @property
    def I(self):
        return set(i for i, e in enumerate(self.data) if e == SIR.I)

    @property
    def R(self):
        return set(i for i, e in enumerate(self.data) if e == SIR.R)

# %%


def min_cut_solver(
    G,
    sir_map,
    budget=None,
    edge_costs=None,
    vertex_costs=None,
    privacy=None,
    partial=None,
    mip=False,
    debug=False,
):
    """
    Modes:
        classic_mip: G, I, budget, mip=True
        classic_frac: G, I, budget, mip=False
        dp_frac: G, I, budget, privacy, mip=False
        evaluate: G, I, partial
        throw Error otherwise
    """

    start_ts = time.perf_counter()
    solver: Solver = pywraplp.Solver.CreateSolver("GUROBI")
    vertex_vars = {}
    edge_vars = {}
    # 2d dictionary defaulting to 1

    # Costs
    if edge_costs is None:
        edge_costs = defaultdict(lambda: 1)
    if vertex_costs is None:
        vertex_costs = defaultdict(lambda: 1)

    # Declare 0-1 variables
    if mip:
        for n in G.nodes:
            vertex_vars[n] = solver.IntVar(0, 1, f"v_{n}")

        for e in G.edges:
            edge_vars[e] = solver.IntVar(0, 1, f"e_{e[0]}_{e[0]}")
    else:
        for n in G.nodes:
            vertex_vars[n] = solver.NumVar(0, 1, f"v_{n}")

        for e in G.edges:
            edge_vars[e] = solver.NumVar(0, 1, f"e_{e[0]}_{e[0]}")

    # Constrain infected (or add (e, d) noise)
    if privacy is not None:
        Delta = 1
        (epsilon, delta) = privacy
        m = len(G.nodes)
        b_constraints = []
        noise = []
        for i, n in enumerate(G.nodes):
            # DOUBLE CHECK THIS!!!
            if sir_map[n] == SIR.I:
                b = 1
            else:
                s = Delta / epsilon * \
                    np.log(m * (np.exp(epsilon) - 1) / delta + 1)
                eta = trunc_laplace(support=s, scale=Delta / epsilon)
                # Test here?
                noise.append(float(s - eta))
                b = min(1, float(s - eta))  # Constraint within 1
                assert b >= 0
            # print(vertex_vars[n])
            # print(b)
            # print(vertex_vars[n] >= b)
            solver.Add(vertex_vars[n] >= b)
            b_constraints.append(b)
        # print(f"Noise (s-eta): {noise}")
        # print(f"b constraints (constrained to 1): {b_constraints}")
    else:
        for n in G.nodes:
            if sir_map[n] == SIR.I:
                solver.Add(vertex_vars[n] == 1)

    # Constrain edge solutions if given partial (solutions)
    if partial is not None:
        for e in G.edges:
            if partial[e] == 1:
                solver.Add(edge_vars[e] == 1)
            else:
                solver.Add(edge_vars[e] == 0)

    # Constrain transmission along edges
    # Bottleneck???
    for e in G.edges:
        solver.Add(edge_vars[e] >=
                   vertex_vars[e[0]] - vertex_vars[e[1]])
        solver.Add(edge_vars[e] >=
                   vertex_vars[e[1]] - vertex_vars[e[0]])

    # Constrain budget for edges
    cost: Constraint = solver.Constraint(0, budget)
    for e in G.edges:
        cost.SetCoefficient(edge_vars[e], edge_costs[e])

    # Set objecttive for people saved
    objective = solver.Objective()
    for n in G.nodes:
        objective.SetCoefficient(vertex_vars[n], vertex_costs[n])
    objective.SetMinimization()

    init_ts = time.perf_counter()
    print(f"Init Time: {init_ts - start_ts}")

    status = solver.Solve()
    if status == solver.INFEASIBLE:
        raise ValueError("Infeasible solution")

    solve_ts = time.perf_counter()
    print(f"Solve Time: {solve_ts - init_ts}")

    if status == solver.OPTIMAL:
        is_optimal = True
    else:
        is_optimal = False

    vertex_solns = {}
    edge_solns = defaultdict(dict)

    for i, n in enumerate(G.nodes):
        vertex_solns[n] = vertex_vars[n].solution_value()

    for i, e in enumerate(G.edges):
        edge_solns[e] = edge_vars[e].solution_value()

    return vertex_solns, edge_solns


def trunc_laplace(support, scale, gen=np.random, size=1):
    """
    Generate laplacian with support [-support, +support], and scale=scale
    """
    return ((-1) ** np.random.randint(2, size=size)) * scipy.stats.truncexpon.rvs(b=support / scale, scale=scale, size=size)
