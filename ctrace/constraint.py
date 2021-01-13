from collections import defaultdict
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Constraint, Solver, Variable, Objective

from contextlib import redirect_stdout
import io
np.random.seed(42)

class ProbMinExposed:
    def __init__(self, G: nx.Graph, infected, contour1, contour2, p1, q, k, costs=None, solver: str=None):
        """Generates the constraints given a graphs. Assumes V1, V2 are 1,2 away from I"""
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

        # Capture messages - Gurobi gives unnecessary log messages
        file_out = io.StringIO()
        with redirect_stdout(file_out):
            if solver is None:
                solver = pywraplp.Solver.CreateSolver('GLOP')
            else:
                solver = pywraplp.Solver.CreateSolver(solver)
        self.recorded_out = file_out.getvalue()

        self.solver: Solver = solver

        # Check if solution is optimal
        self.isOptimal = None

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
        for u in self.V1:
            # print(f"p1: {self.p1[u]}")
            numExposed.SetCoefficient(self.Y1[u], self.p1[u])

        for v in self.V2:
            numExposed.SetCoefficient(self.Y2[v], 1)

        numExposed.SetMinimization()

    def setVariable(self, index: int, value: int):
        """Sets the ith V1 indicator to value int. May only use after solve_lp"""
        i = self.quaran_map[index]
        if i in self.partials:
            raise ValueError(f"in {i} is already set!")
        if value not in (0, 1):
            raise ValueError("Value must be 0 or 1")
        self.partials[i] = value
        self.solver.Add(self.X1[i] == value)

    def getVariables(self):
        return self.quaran_raw

    def setVariableId(self, id: int, value: int):
        """Sets the ith V1 indicator to value int"""
        if id in self.partials:
            raise ValueError(f"in {id} is already set!")
        if value not in (0, 1):
            raise ValueError("Value must be 0 or 1")
        self.partials[id] = value
        self.solver.Add(self.X1[id] == value)

    def filled(self):
        # TODO: Does this method make sense?
        """Returns true if every variable is solved"""
        return self.partials == self.V1

    def solve_lp(self):
        """Solves the LP problem"""
        status = self.solver.Solve()
        if status == self.solver.INFEASIBLE:
            raise ValueError("Infeasible solution")

        if status == self.solver.OPTIMAL:
            self.isOptimal = True
        else:
            self.isOptimal = False
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
    def __init__(self, G: nx.Graph, infected, contour1, contour2, p1, q, k, labels, label_limits, costs=None, solver=None):
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
        label_limits
            A L sized array restricting the number of people in each category.
            Must sum to less than to <= k (these constraints should be more strict than k)
        costs
        solver
        """
        self.L = len(label_limits)
        if (len(labels) != len(G.nodes)):
            raise ValueError("labels must match graphs nodes")

        if (any(map(lambda x: x >= self.L or x < 0, labels))):
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
            solver = 'SCIP'
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

