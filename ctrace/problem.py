import abc
import random
import networkx as nx
import numpy as np

from typing import Dict, List, Tuple
from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Variable, Constraint, Objective

from ctrace.round import *
from ctrace.utils import *
from ctrace.simulation import *

class MinExposedProgram:
    def __init__(self, info: InfectionState, solver_id="GLOP", simp = False):
        
        self.result = None
        self.G = info.G
        self.SIR = info.SIR
        
        self.budget = info.budget
        self.policy = info.policy
        self.budget_labels = info.budget_labels
        self.labels = info.labels
        
        self.p = info.transmission_rate
        self.transmission_known = info.transmission_known
        self.compliance_known = info.compliance_known
        
        self.contour1, self.contour2 = info.V1, info.V2
        self.solver = pywraplp.Solver.CreateSolver(solver_id)
        
        if simp:
            self.P, self.Q = pq_independent_simp(info.P, info.Q)
        else:
            self.P = info.P
            self.Q = info.Q
        

        if self.solver is None:
            raise ValueError("Solver failed to initialize!")
    
        # Partial evaluation storage
        self.partials = {}

        # controllable - contour1
        self.X1: Dict[int, Variable] = {}
        self.Y1: Dict[int, Variable] = {}

        # non-controllable - contour2
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
        
        if self.policy == "none":
            cost: Constraint = self.solver.Constraint(0, self.budget)
            for u in self.contour1:
                cost.SetCoefficient(self.X1[u], 1)
        else:
            for label in self.labels:
                cost: Constraint = self.solver.Constraint(0, self.budget_labels[label])
                for u in self.contour1:
                    if (self.G.nodes[u]["age_group"]==label):
                        cost.SetCoefficient(self.X1[u], 1)
                    else:
                        cost.SetCoefficient(self.X1[u], 0)
        
        if self.compliance_known:
            for u in self.contour1:
                for v in self.G.neighbors(u):
                    if v in self.contour2:
                        #c = self.Q[u][v] * self.P[u] *(1-self.P[v])
                        c = self.Q[u][v] * self.P[u]
                        self.solver.Add(self.Y2[v] >= c* ((1-self.G.nodes[u]['compliance_rate'])*self.X1[u] + self.Y1[u]))
        else:
            for u in self.contour1:
                for v in self.G.neighbors(u):
                    if v in self.contour2:
                        #c = self.Q[u][v] * self.P[u] *(1-self.P[v])
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

        self.objective_value = self.lp_objective_value()

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
    
    def lp_objective_value(self):
        # Will raise error if not solved
        # Number of people exposed in V2
        objective_value = 0
        for v in self.contour2:
            objective_value += self.Y2[v].solution_value()
        return objective_value

    def min_exposed_objective(self):
        """Simulate the MinExposed Objective outline in the paper. May only be called after recommend"""
        if self.result is None:
            raise ValueError("Must call recommend() before retrieving objective value")
        min_exposed_objective(self.info.G, self.info.SIR, self.info.transmission_rate, self.result)


class MinExposedLP(MinExposedProgram):
    def __init__(self, info: InfectionState, solver_id="GLOP", simp = False):
        super().__init__(info, solver_id, simp)

    def init_variables(self):
        # Declare Fractional Variables
        for u in self.contour1:
            self.X1[u] = self.solver.NumVar(0, 1, f"V1_x{u}")
            self.Y1[u] = self.solver.NumVar(0, 1, f"V1_y{u}")
        for v in self.contour2:
            self.Y2[v] = self.solver.NumVar(0, 1, f"V2_y{v}")


class MinExposedIP(MinExposedProgram):
    def __init__(self, info: InfectionState, solver_id="SCIP", simp = False):
        super().__init__(info, solver_id, simp)

    def init_variables(self):
        # Declare Variables (With Integer Constraints)
        for u in self.contour1:
            self.X1[u] = self.solver.IntVar(0, 1, f"V1_x{u}")
            self.Y1[u] = self.solver.IntVar(0, 1, f"V1_y{u}")
        for v in self.contour2:
            self.Y2[v] = self.solver.NumVar(0, 1, f"V2_y{v}")

class MinExposedSAA(MinExposedProgram):
    def __init__(self, info: InfectionState, solver_id="GLOP", num_samples=10, seed=42):
        self.result = None
        self.info = info
        self.G = info.G
        self.SIR = info.SIR
        self.budget = info.budget
        self.p = info.transmission_rate
        self.contour1, self.contour2 = info.V1, info.V2
        self.solver = pywraplp.Solver.CreateSolver(solver_id)
        self.num_samples = num_samples

        random.seed(seed)

        # A list of sets of edges between v1 and v2 that are actually sampled for iteration i
        self.contour_edge_samples = [[] for i in range(self.num_samples)]

        if self.solver is None:
            raise ValueError("Solver failed to initialize!")

        # Compute P, Q from SIR
        self.P, self.Q = pq_independent(self.G, self.SIR.I, self.contour1, self.p)
    
        # Partial evaluation storage
        self.partials = {}

        # controllable - contour1
        self.X1: Dict[int, Variable] = {}
        self.Y1: Dict[int, Variable] = {}

        # non-controllable - contour2 (over i samples)
        self.Y2: List[Dict[int, Variable]] = [{} for i in range(self.num_samples)]
        self.init_variables()

        # Initialize constraints
        self.init_constraints()

    def init_variables(self):
        """Declare variables as needed"""
        for u in self.contour1:
            self.X1[u] = self.solver.NumVar(0, 1, f"V1_x{u}")
            self.Y1[u] = self.solver.NumVar(0, 1, f"V1_y{u}")
        
        for i in range(self.num_samples):
            for v in self.contour2:
                self.Y2[i][v] = self.solver.NumVar(0, 1, f"V2[{i}]_y{v}") 

    def init_constraints(self):
        """Initializes the constraints according to the relaxed LP formulation of MinExposed"""

        # X-Y are complements
        for u in self.contour1:
            self.solver.Add(self.X1[u] + self.Y1[u] == 1)

        # cost (number of people quarantined) must be within budget
        cost: Constraint = self.solver.Constraint(0, self.budget)
        for u in self.contour1:
            cost.SetCoefficient(self.X1[u], 1)

        # Y2[v] becomes a lower bound for the probability that vertex v is infected
        # SAMPLING

        for i in range(self.num_samples):
            for u in self.contour1:
                for v in self.G.neighbors(u):
                    if v in self.contour2 and random.random() < self.p: # Sample with uniform probability
                        self.contour_edge_samples[i].append((u, v))
                        self.solver.Add(self.Y2[i][v] >= self.Y1[u])

        # Objective: Minimize number of people exposed in contour2
        num_exposed: Objective = self.solver.Objective()

        for i in range(self.num_samples):
            for v in self.contour2:
                num_exposed.SetCoefficient(self.Y2[i][v], 1)
        num_exposed.SetMinimization()

    def lp_objective_value(self):
        # Will raise error if not solved
        # Number of people exposed in V2
        objective_value = 0
        for i in range(self.num_samples):
            for v in self.contour2:
                objective_value += self.Y2[i][v].solution_value()
        return objective_value / self.num_samples
        
    def lp_sample_objective_value(self, i):
        objective_value = 0
        for v in self.contour2:
            objective_value += self.Y2[i][v].solution_value()
        return objective_value