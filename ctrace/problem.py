import abc
from typing import Dict

import networkx as nx
import numpy as np
from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Variable, Constraint, Objective
from .recommender import *
from .utils import pq_independent, find_excluded_contours, min_exposed_objective


class InfectionInfo:
    """Encapsulates the current state of infection given to the agent"""

    def __init__(self, G: nx.graph, sir, p, budget):
        self.G = G
        self.sir = sir
        self.budget = budget
        self.p = p

    def contours(self):
        return NotImplementedError

# Is this abstraction even needed?
class Problem:
    def __init__(self, info: InfectionInfo):
        self.info = info
        self.result = None

    def recommend(self):
        """Returns an n-sized set of individuals to quarantine"""
        raise NotImplementedError

# TODO: Why do we have classes - add additional capability and to conform to standard?

# TODO: Is it good software practice to call methods after the fact?
class MaxSaveMixin:
    def max_save_objective(self):
        """May only be called after recommend"""
        if self.result is None:
            raise ValueError("Must call recommend() before retrieving objective value")
        raise NotImplementedError


class MinExposedMixin:
    def min_exposed_objective(self):
        """Simulate the MinExposed Objective outline in the paper. May only be called after recommend"""
        if self.result is None:
            raise ValueError("Must call recommend() before retrieving objective value")
        contours = find_excluded_contours(self.info.G, self.info.sir.I, self.info.sir.R)

        min_exposed_objective(self.info.G, self.info.sir, self.info.p, self.result)


# TODO: Add a Mixin to allow for MinExposed tracking ability?
class RandomSolver(MaxSaveMixin, MinExposedMixin, Problem):
    def __init__(self, info):
        super().__init__(info)
        self.result = None

    def recommend(self):
        v1, _ = find_excluded_contours(self.info.G, self.info.sir.I, self.info.sir.R)
        self.result = rand(v1, self.info.budget)
        return self.result


class DegreeSolver(MaxSaveMixin, MinExposedMixin, Problem):
    def __init__(self, info):
        super().__init__(info)
        self.result = None

    def recommend(self):
        v1, v2 = find_excluded_contours(self.info.G, self.info.sir.I, self.info.sir.R)
        self.result = degree(self.info.G, v1, v2, self.info.budget)
        return self.result


class WeightedSolver(MaxSaveMixin, MinExposedMixin, Problem):
    def __init__(self, info):
        super().__init__(info)
        self.result = None

    def recommend(self):
        v1, v2 = find_excluded_contours(self.info.G, self.info.sir.I,
                                        self.info.sir.R)  # Time impact of excluded_contours?
        P, Q = pq_independent(self.info.G, self.info.sir.I, v1, self.info.p)  # Time impact of pq?
        self.result = weighted(self.info.G, P, Q, v1, v2, self.info.budget)
        return self.result


# Should random, greedy, weighted go under Problem?
class MinExposedProgram(MinExposedMixin, Problem):
    def __init__(self, info: InfectionInfo, solver_id):
        super().__init__(info)
        self.G = info.G
        self.SIR = info.sir
        self.contour1, self.contour2 = info.contours()
        self.budget = info.budget
        self.p = info.p

        # Compute P, Q from SIR
        self.P, self.Q = pq_independent(self.G, self.SIR.I, self.contour1, self.p)

        self.solver = pywraplp.Solver.CreateSolver(solver_id)

        # Partial evaluation storage
        self.partials = {}

        # controllable - contour1
        self.X1: Dict[int, Variable] = {}
        self.Y1: Dict[int, Variable] = {}

        # non-controllable - contour2
        self.X2: Dict[int, Variable] = {}
        self.Y2: Dict[int, Variable] = {}
        self.init_variables()

        # Initialize constraints
        self.init_constraints()

    def init_variables(self):
        """Declare variables as needed"""
        raise NotImplementedError

    def init_constraints(self):
        """Initializes the constraints according to the relaxed LP formulation of MinExposed"""

        # X-Y are complements
        for u in self.contour1:
            self.solver.Add(self.X1[u] + self.Y1[u] == 1)
        for v in self.contour2:
            self.solver.Add(self.X2[v] + self.Y2[v] == 1)

        # cost (number of people quarantined) must be within budget
        cost: Constraint = self.solver.Constraint(0, self.budget)
        for u in self.contour1:
            cost.SetCoefficient(self.X1[u], 1)

        # Y2[v] becomes a lower bound for the probability that vertex v is infected
        for u in self.contour1:
            for v in self.G.neighbors(u):
                if v in self.contour2:
                    c = self.Q[u][v] * self.P[u]
                    self.solver.Add(self.Y2[v] >= c * self.Y1[u])

        # Objective: Minimize number of people exposed in contour2
        num_exposed: Objective = self.solver.Objective()
        for v in self.contour2:
            num_exposed.SetCoefficient(self.Y2[v], 1)
        num_exposed.SetMinimization()

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
        self.quarantine_map = []  # Maps from dense to id
        self.quarantine_raw = np.zeros(len(self.X1))
        self.quarantined_solution = {}
        self.is_optimal = False

        status = self.solver.Solve()
        if status == self.solver.INFEASIBLE:
            raise ValueError("Infeasible solution")

        if status == self.solver.OPTIMAL:
            self.is_optimal = True
        else:
            self.is_optimal = False

        # Indicators
        for i, u in enumerate(self.contour1):
            self.quarantine_raw[i] = self.quarantined_solution[u] = self.X1[u].solution_value()
            self.quarantine_map.append(u)

        # Number of people exposed in V2
        self.objective_value = 0
        for v in self.contour2:
            self.objective_value += self.Y2[v].solution_value()

        return self.quarantined_solution

    def get_variables(self):
        """Returns array representation of indicator variables"""
        return self.quarantine_raw

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

    def get_solution(self):
        return self.quarantined_solution

    def recommend(self):
        raise NotImplementedError


class MinExposedLP(MinExposedProgram):
    def __init__(self, info: InfectionInfo, solver_id, rounder):
        super().__init__(info, solver_id)
        self.init_variables()
        self.rounder = rounder

    def init_variables(self):
        # Declare Fractional Variables
        for u in self.contour1:
            self.X1[u] = self.solver.NumVar(0, 1, f"V1_x{u}")
            self.Y1[u] = self.solver.NumVar(0, 1, f"V1_y{u}")
        for v in self.contour2:
            self.X2[v] = self.solver.NumVar(0, 1, f"V2_x{v}")
            self.Y2[v] = self.solver.NumVar(0, 1, f"V2_y{v}")

    def recommend(self):
        return self.rounder(self)


class MinExposedIP(MinExposedProgram):
    def __init__(self, info: InfectionInfo, solver_id):
        super().__init__(info, solver_id)
        self.init_variables()

    def init_variables(self):
        # Declare Variables (With Integer Constraints)
        for u in self.contour1:
            self.X1[u] = self.solver.IntVar(0, 1, f"V1_x{u}")
            self.Y1[u] = self.solver.IntVar(0, 1, f"V1_y{u}")
        for v in self.contour2:
            self.X2[v] = self.solver.NumVar(0, 1, f"V2_x{v}")
            self.Y2[v] = self.solver.NumVar(0, 1, f"V2_y{v}")

    def recommend(self):
        return self.rounder(self)
