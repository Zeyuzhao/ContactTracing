from collections import defaultdict
from typing import Dict

import networkx as nx
import numpy as np
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
    def __init__(self, G: nx.Graph, infected, contour1, contour2, p1, q, k, costs=None, solver: str=None):
        """
        Constructs the constraints of the LP relaxation of MinExposed.
        Provides an API to retrieve LP solutions and set LP variables for iterated rounding algorithms.
        Parameters
        ----------
        G
            The contact tracing graph.
            Assumes each node labelled consecutively from [0, N), where N is the number of nodes
        infected
            The set of infected nodes
        contour1
            The set of nodes distance one away from infected, excluding recovered nodes
        contour2
            The set of nodes distance two away from infected, excluding recovered nodes
        p1
            An dictionary p1[u] that maps node u in V1 to its probability of infection
        q
            An dictionary q[u][v] that maps edge (u, v) to conditional probability of infection from node u to v
        k
            The budget allotted to quarantine nodes in V1.
        costs
            An dictionary costs[u] that describes the cost associated quarantining node u.
            Defaults to 1 for this project.
        solver
            An string describing the solver backend or-tools will use.
            Defaults to GLOP, but we GUROBI_LP for our experiments.
        """
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
            solver = pywraplp.Solver.CreateSolver('GLOP')
        else:
            solver = pywraplp.Solver.CreateSolver(solver)

        if solver is None:
            raise ValueError("Solver failed to initialized!")

        self.solver: Solver = solver

        # Check if solution is optimal
        self.is_optimal = None

        # Track number of edges between v1 and v2
        self.num_cross_edges = 0

        # Partial Evaluation storage
        self.partials = {}
        self.init_variables()
        self.init_constraints()

        # Computed after calling solve_lp()
        self.objective_value = 0

        # Maps from node id -> dense index
        self.quarantine_map = {}
        # Dense representation of v1 indicators for iterated rounding
        self.quarantine_raw = np.zeros(len(self.X1))
        self.objective_value_with_constant = 0

        # X2 indicator variables
        self.saved_solution: Dict[int, float] = {}
        # X1 indicator variables
        self.quarantined_solution: Dict[int, float] = {}

    def init_variables(self):
        """Initializes the variables with LP variables (fractional solutions)"""
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
        """Initializes the constraints according to the relaxed LP formulation of MinExposed"""

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

                    # Tracking the number of constraints between V_1 and V_2
                    self.num_cross_edges += 1

        # Set minimization objective
        # Number of people exposed in V2
        numExposed: Objective = self.solver.Objective()
        for v in self.V2:
            numExposed.SetCoefficient(self.Y2[v], 1)
        numExposed.SetMinimization()

    def set_variable_id(self, id: int, value: int):
        """
        Sets the ith V1 indicator by node id to value

        Parameters
        ----------
        id
            Node Id of a node in V1
        value
            An integer of value 0 or 1
        Returns
        -------
        None
        """
        if id in self.partials:
            raise ValueError(f"in {id} is already set!")
        if value not in (0, 1):
            raise ValueError("Value must be 0 or 1")
        self.partials[id] = value
        self.solver.Add(self.X1[id] == value)

    def set_variable(self, index: int, value: int):
        """
        Sets the ith V1 indicator using dense array index to value int.
        May only use after solve_lp

        Parameters
        ----------
        index
            An array index of the dense representation
        value
            An integer of value 0 or 1
        Returns
        -------
        None
        """
        i = self.quarantine_map[index]
        self.set_variable_id(i, value)

    def get_variables(self):
        """
        Retrieve the dense array representation of v1 indicators
        Returns
        -------
        array
        """
        return self.quarantine_raw

    def solve_lp(self):
        """
        Solves the LP problem and computes the LP objective value
        Returns
        -------
        None
        Sets the following variables:
        self.objective_value
            The objective value of the LP solution
        self.is_optimal
            Whether the LP solver reached an optimal solution
        self.quarantined_solution
            An dictionary mapping from V1 node id to its fractional solution
        self.quarantine_raw
            An array (dense) of the LP V1 (fractional) fractional solutions
        self.quarantine_map
            Maps the array index to the V1 node id
        """

        # Reset variables
        self.objective_value = 0
        self.quarantine_map = {}
        self.quarantine_raw = np.zeros(len(self.X1))
        self.quarantined_solution = {}
        self.saved_solution = {}

        status = self.solver.Solve()
        if status == self.solver.INFEASIBLE:
            raise ValueError("Infeasible solution")

        if status == self.solver.OPTIMAL:
            self.is_optimal = True
        else:
            self.is_optimal = False
        # Indicators
        for i, u in enumerate(self.V1):
            self.quarantine_raw[i] = self.quarantined_solution[u] = self.X1[u].solution_value(
            )
            self.quarantine_map[i] = u

        # V2 portion of the objective value
        for v in self.V2:
            self.saved_solution[v] = self.X2[v].solution_value()
            self.objective_value += (1 - self.saved_solution[v])

        # # Add the constant to the objective value (the V1 portion of the objective value)
        # self.objective_value_with_constant = self.objective_value + sum(self.p1[u] for u in self.V1)

class ProbMinExposedRestricted(ProbMinExposed):
    def __init__(self, G: nx.Graph, infected, contour1, contour2, p1, q, k, labels: Dict[int, int], label_limits, costs=None, solver=None):
        """
        Uses the same LP as ProbMinExposed, but with the added constraint of labels.
        Labels (from [0,L-1]) are assigned to every member in contour1, and label_limits sets
        the limit for the number of people quarantined in a label group.
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
            A len(G.contour1) sized array with labels 0..L-1 for each node
        label_limits
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
    """
    Uses the same LP formulation as ProbMinExposed, but creates integer variables for contour1 nodes.
    """
    def __init__(self, G: nx.Graph, infected, contour1, contour2, p1, q, k, costs=None, solver=None):
        if solver is None:
            solver = pywraplp.Solver.CreateSolver('SCIP')
        super().__init__(G, infected, contour1, contour2, p1, q, k, costs, solver)

    def init_variables(self):
        """Initializes the variables with LP variables (fractional solutions)"""
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

