import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Constraint, Solver, Variable, Objective

from typing import Set, Dict
import pickle as pkl

class Constrain:
    """A constraint object represents the LP formulation of the problem"""
    def __init__(self):
        pass

    def __copy__(self):
        raise NotImplementedError

    def init_lp(self):
        """Initializes the LP problem from the given graph"""
        raise NotImplementedError

    def compute_lp(self):
        "Re-solves LP solutions"
    def fix_variable(self, index, value):
        raise NotImplementedError



class ProbMinExposed:
    def __init__(self, graph: nx.Graph, infected, contour1, contour2, p1, q, k, costs):
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
        self.costs = costs
        self.init()

    def init(self):
        self.solver: Solver = pywraplp.Solver.CreateSolver('GLOP')
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
                    self.solver.Add(self.Y2[v] >= (self.q[u][v] * self.p1[u]) * self.Y1[u])

        # Set minimization objective
        # Number of people free in V1 and people exposed in V2
        numExposed: Objective = self.solver.Objective()
        for u in self.V1:
            numExposed.SetCoefficient(self.Y1[u], self.p1[u])

        for v in self.V2:
            numExposed.SetCoefficient(self.Y2[v], 1)

        numExposed.SetMinimization()

    def setVariable(self, i: int, value: int):
        self.solver.Add(self.X1[i] == value)

    def solve(self):
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

    def rawlp(self):
        return self.quaran_sol


if __name__ == '__main__':
    pass










