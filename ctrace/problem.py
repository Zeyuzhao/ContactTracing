import abc
import math
from abc import ABC
from collections import defaultdict, namedtuple
from typing import Dict

import networkx as nx
import numpy as np
from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Constraint, Variable, Objective
from ctrace.utils import *


SIR = namedtuple("SIR", ["S", "I", "R"])
class InfectionState:
    """Requires loading of graph"""
    def __init__(self, G, sir, p):
        # Requirements on G - needs the name attribute!
        # G.graph["name"] = {ACTUAL NAME}
        self.G: nx.Graph = G

        # visible sir
        self.sir = sir
        self.contours = [set(), set()]

        self.p: int = p
        # TODO: Is this code efficient?
        self.init_pq("independent")

    def init_pq(self, method="constant"):
        if method == "absolute":
            self.P = defaultdict(lambda: 1)
            self.Q = defaultdict(lambda: defaultdict(lambda: 1))
            return True
        if method == "independent": # TODO: Test!
            self.P = {}
            for u in self.contours[1]:
                # Count the number of neighbors in infected
                count = sum((v in self.contours[0]) for v in self.G.neighbors(u))
                self.P[u] = 1 - math.pow(1 - self.p, count)
            self.Q = defaultdict(lambda: defaultdict(lambda: self.p))
            return True
        if method == "uniform":
            pass
        raise ValueError(f'Method "{method}" is invalid')

    def init_contours(self):
        # Initialize known contours
        v1, v2 = find_excluded_contours(self.G, self.sir.I, self.sir.R)
        self.contours.append(v1)
        self.contours.append(v2)

    def solve(self):
        pass

class MaxSave(Problem):

    pass

    def save_sir(self):
        pass

class MinExposed(Problem):
    def __init__(self, budget, G, S, I, I_known, R, p, solver_id: str = "GUROBI"):
        super().__init__(G, S, I, I_known, R, p)
        self.G = infection_state.G
        self.contour1, self.contour2 = infection_state.contours()
        self.budget = budget

        # Compute P, Q from SIR
        self.P, self.Q = PQ_deterministic()

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

    @abc.abstractmethod
    def init_variables(self):
        """Declare variables as needed"""
        pass

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
        self.quarantine_map = [] # Maps from dense to id
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

    def solve(self):
        # Returns rounded solution
        self.solve_lp()
        return self._round(self)


class MinExposedProgram(ABC):
    pass

class MinExposedLP(MinExposed):
    def __init__(self, G: nx.Graph, SIR, budgets, labels, p=None, q=None, solver: str=""):
        pass

class MinExposedIP(MinExposed):
    def __init__(self, G: nx.Graph, SIR, budgets, p=None, q=None, solver: str = ""):
        pass
