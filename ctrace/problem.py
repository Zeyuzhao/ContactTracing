import abc
from ctrace import PROJECT_ROOT
import random
import itertools
import networkx as nx
import numpy as np
import rich
import logging
import time

from typing import Any, Dict, List, Tuple, Union
from copy import deepcopy
from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Variable, Constraint, Objective, Solver
from statistics import mean
from rich.logging import RichHandler

from .round import D, D_prime
from .utils import pq_independent, find_excluded_contours, min_exposed_objective, uniform_sample
from .simulation import InfectionInfo, SIR_Tuple


import tracemalloc
tracemalloc.start()
import time


# Experimental logging features:
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler(PROJECT_ROOT / 'logs' / 'problem.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def debug_memory(label=""):
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    logger.debug(f"[{label}]: {top_stats[:5]}")

class MinExposedProgram:
    def __init__(self, info: InfectionInfo, solver_id="GLOP"):
        
        self.result = None
        self.info = info
        self.G = info.G
        self.SIR = info.SIR
        self.budget = info.budget
        self.p = info.transmission_rate
        self.contour1, self.contour2 = self.info.V1, self.info.V2

        # Check contours are sorted
        
        self.solver: Solver = pywraplp.Solver.CreateSolver(solver_id)

        if self.solver is None:
            raise ValueError("Solver failed to initialize!")
            
        # Compute P, Q from SIR
        self.P, self.Q = pq_independent(self.G, self.SIR.I, self.contour1, self.p)
    
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
        start = time.time()

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

        end = time.time()
        logger.debug(f'Elapsed solver_lp: {end-start}')

        self._post_solve_handler()
        return self.quarantined_solution

    def _post_solve_handler():
        """Called after solve executes"""
        pass
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
        Sets the ith X1 indicator by node id to value
        Quarantine: 1
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
    def __init__(self, info: InfectionInfo, solver_id="GLOP"):
        super().__init__(info, solver_id)

    def init_variables(self):
        # Declare Fractional Variables
        for u in self.contour1:
            self.X1[u] = self.solver.NumVar(0, 1, f"V1_x{u}")
            self.Y1[u] = self.solver.NumVar(0, 1, f"V1_y{u}")
        for v in self.contour2:
            self.Y2[v] = self.solver.NumVar(0, 1, f"V2_y{v}")

class MinExposedIP(MinExposedProgram):
    def __init__(self, info: InfectionInfo, solver_id="GUROBI"):
        super().__init__(info, solver_id)

    def init_variables(self):
        # Declare Variables (With Integer Constraints)
        for u in self.contour1:
            self.X1[u] = self.solver.IntVar(0, 1, f"V1_x{u}")
            self.Y1[u] = self.solver.IntVar(0, 1, f"V1_y{u}")
        for v in self.contour2:
            self.Y2[v] = self.solver.NumVar(0, 1, f"V2_y{v}")

class MinExposedSAA(MinExposedProgram):
    def __init__(self, 
        G: nx.Graph,
        SIR: SIR_Tuple,
        budget: int,
        transmission_rate=1, 
        compliance_rate=1, 
        structure_rate=0, 
        num_samples=10, 
        seed=42,
        aggregation_method="max", # "max" | "mean"
        solver_id="GUROBI", # "GUORBI" | "GLOP" | "SCIP"
    ):
        self.seed = seed
        self.result = None
        # Set problem structure
        self.G: nx.Graph = G
        self.SIR = SIR
        self.budget = budget
        # Compute contours
        self.contour1, self.contour2 = find_excluded_contours(G, SIR.I, SIR.R)

        # Set SAA parameters - affects the samples.
        self.num_samples = num_samples

        # Conditional probability that edge can carry disease
        self.p = transmission_rate
        # Probability that V1 node observes quarantine
        self.q = compliance_rate
        # Probability to include edge in (I x V1) and (V1 x V2)
        self.s = structure_rate
        
        # Sample Data
        self.sample_data = [{} for _ in range(self.num_samples)]
        
        # Partial evaluation storage - applies only to controllable variables (X1)
        self.partials = {}
        # Controllable - contour1
        self.X1: Dict[int, Variable] = {}
        self.Y1: Dict[int, Variable] = {}
        # Non-controllable variables - over series.
        self.sample_variables: List[Dict[str, Union[Variable, Dict[int, Variable]]]] = [{} for _ in range(self.num_samples)]
        # Aggregate Variable - either sum or max
        self.Z: Variable = None

        # Aggregation method
        self.aggregation_method = aggregation_method

        # Solver
        self.solver_id=solver_id
        self.solver: Solver = pywraplp.Solver.CreateSolver(solver_id)
        if self.solver is None:
            raise ValueError("Solver failed to initialize")
    
    @classmethod
    def create(cls, 
        G: nx.Graph,
        SIR: SIR_Tuple,
        budget: int,
        log=False,
        **args,
    ) -> "MinExposedSAA":
        """Creates a new MinExposedSAA problem with sampling"""
        problem = cls(G, SIR, budget, **args)
        s = time.time()
        problem.init_samples()
        f = time.time()
        if log:
            logger.debug(f"Sampling Complete. [{f - s}]")

        s = time.time()
        problem.init_variables()
        f = time.time()
        if log:
            logger.debug(f"Variable Initialization Complete. [{f - s}]")

        s = time.time()
        problem.init_constraints()
        f = time.time()
        if log:
            logger.debug(f"Constraint Initialization Complete. [{f - s}]")
            
        return problem

    @classmethod
    def from_infection_info(cls, info: InfectionInfo, **args) -> "MinExposedSAA":
        """Only use G and SIR from infection_info"""
        raise NotImplementedError
        # problem = cls.create(info.G, info.SIR, **args)
        # return problem

    @classmethod
    def load_sample(cls, 
        G: nx.Graph,
        SIR: SIR_Tuple,
        budget: int,
        sample_data: Dict[str, Any],
        solver_id: str = "GLOP",
    ) -> "MinExposedSAA":
        """
        Creates a new MinExposedSAA problem with sample data
        transmission, compliance, structure, num_samples and seed are not used       
        """
        # TODO: [transmission, compliance, structure, num_samples, seed] uniquely determine sample_data
        problem = cls(G, SIR, budget, solver_id=solver_id, num_samples=len(sample_data))
        problem.sample_data = sample_data
        problem.init_variables()
        problem.init_constraints()
        # TODO: assert sample_data does not change
        return problem

    def init_samples(self):
        # Transmission Sampling
        random.seed(self.seed)
        for i in range(self.num_samples):
            # STRUCTURAL
            # Structural edges are sampled into existance.
            # Currently - structural edges must be in (I x V1) and (V1 x V2)
            # structural_edges = [[], []]
            # TODO: UNDISABLE structural edges???
            # structural_edges[0] = uniform_sample(list(itertools.product(self.SIR[1], self.contour1)), self.s)
            # structural_edges[1] = uniform_sample(list(itertools.product(self.contour1, self.contour2)), self.s)
            # self.sample_data[i]["structural_edges"] = structural_edges
            
            # Expanded network (with sampled structural edges)
            GE = self.G
            # GE.add_edges_from(structural_edges[0] + structural_edges[1])

            # DIFFUSION
            # Implementation Notes:
            # 1) Will use the graph obtained from structural uncertainty sampling
            # 2) border_edges[1] indicates infectious edges from u in V1 -> v in V2. 
            # Any edge here represents a binding constraint: 
            # Infectious edges (_, u) and (u, v) are sampled

            self.sample_data[i]["border_edges"] = [[], []]
            # sampled independently from I_border with transmission probability
            I_border = compute_border_edges(GE, self.SIR[1], self.contour1)
            I_border_sample = uniform_sample(I_border, self.p)
            self.sample_data[i]["border_edges"][0] = I_border_sample

            # relevant_v1 -> set of v1 nodes "infected" by sampled edges
            relevant_v1 = {b for (_,b) in I_border_sample}
            self.sample_data[i]["relevant_v1"] = relevant_v1
            assert relevant_v1.issubset(self.contour1)

            # V1_border - conditional on relevant_v1s only (sampled with transmission probability)
            V1_border = compute_border_edges(GE, relevant_v1, self.contour2)
            V1_border_sample = uniform_sample(V1_border, self.p)
            self.sample_data[i]["border_edges"][1] = V1_border_sample

            # COMPLIANCE
            self.sample_data[i]["non_compliant"] = set(uniform_sample(self.contour1,  1 - self.q))

    def init_variables(self):
        if self.solver_id.upper() == "GUROBI":
            self.init_int_variables()
            print("Integer Solving ...")
        else:
            self.init_frac_variables()

    def init_frac_variables(self):
        for u in self.contour1:
            self.X1[u] = self.solver.NumVar(0, 1, f"V1_x{u}")
            self.Y1[u] = self.solver.NumVar(0, 1, f"V1_y{u}")
        
        for i in range(self.num_samples):
            variables = {}
            variables["Y1"] = {u: self.solver.NumVar(0, 1, f"V1[{i}]_y{u}") for u in self.contour1}
            variables["Y2"] = {v: self.solver.NumVar(0, 1, f"V2[{i}]_y{v}") for v in self.contour2}
            variables["z"] = self.solver.NumVar(0, self.solver.infinity(), f"z[{i}]")
            self.sample_variables[i] = variables

        self.Z = self.solver.NumVar(0, self.solver.infinity(), f"Z")

    def init_int_variables(self):
        for u in self.contour1:
            self.X1[u] = self.solver.IntVar(0, 1, f"V1_x{u}")
            self.Y1[u] = self.solver.IntVar(0, 1, f"V1_y{u}")
        
        for i in range(self.num_samples):
            variables = {}
            variables["Y1"] = {u: self.solver.NumVar(0, 1, f"V1[{i}]_y{u}") for u in self.contour1}
            variables["Y2"] = {v: self.solver.NumVar(0, 1, f"V2[{i}]_y{v}") for v in self.contour2}
            variables["z"] = self.solver.NumVar(0, self.solver.infinity(), f"z[{i}]")
            self.sample_variables[i] = variables

        self.Z = self.solver.NumVar(0, self.solver.infinity(), f"Z")

    def init_constraints(self):
        
        # Def of X-Y: complement booleans
        for u in self.contour1:
            self.solver.Add(self.X1[u] + self.Y1[u] == 1)

        # cost (number of people quarantined) must be within budget
        cost: Constraint = self.solver.Constraint(0, self.budget)
        for u in self.contour1:
            cost.SetCoefficient(self.X1[u], 1)

        # <============== TODO: =================> 

        # People not asked to quarantine would not quarantine
        for i in range(self.num_samples):
            for u in self.contour1:
                self.solver.Add(self.sample_variables[i]["Y1"][u] >= self.Y1[u])

        # non-compliant contour1
        for i in range(self.num_samples):
            for u in self.contour1:
                if u in self.sample_data[i]["non_compliant"]:
                    self.solver.Add(self.sample_variables[i]["Y1"][u] >= 1)

        # Link sample Y1s with sample Y2s
        for i in range(self.num_samples):
            for (u, v) in self.sample_data[i]["border_edges"][1]:
                self.solver.Add(self.sample_variables[i]["Y2"][v] >= self.sample_variables[i]["Y1"][u])
        
        # Definition of z
        for i in range(self.num_samples):
            self.solver.Add(sum(self.sample_variables[i]["Y2"].values()) == self.sample_variables[i]["z"])
        # Combine using given lp_objective
        self.lp_objective()
    
    def _post_solve_handler(self):
        """Set variable solutions by examining solution value"""

        # Compute variable_solutions
        self.variable_solutions: Dict[str, Any] = {}
        self.variable_solutions["X1"] = {u: self.X1[u].solution_value() for u in self.contour1}
        self.variable_solutions["Y1"] = {u: self.Y1[u].solution_value() for u in self.contour1}
        self.variable_solutions["sample_variables"] = [{} for _ in range(self.num_samples)]
        for i in range(self.num_samples):
            sample_solution = {}
            sample_solution["Y1"] = {u: self.sample_variables[i]["Y1"][u].solution_value() for u in self.contour1}
            sample_solution["Y2"] = {u: self.sample_variables[i]["Y2"][u].solution_value() for u in self.contour2}
            sample_solution["z"] = self.sample_variables[i]["z"].solution_value()
            self.variable_solutions["sample_variables"][i] = sample_solution
        self.variable_solutions["Z"] = self.Z.solution_value()

        # Compute target statistics
        self.exposed_v2 = [[] for _ in range(self.num_samples)]
        for i in range(self.num_samples):
            self.exposed_v2[i] = [u for u, v in self.variable_solutions["sample_variables"][i]["Y2"].items() if is_close(1, v, 0.05)]
            
            z = self.variable_solutions["sample_variables"][i]["z"]
            # if not is_close(z, len(self.exposed_v2[i]), frac=0.05):
            #     print(f'Warning: z: {z} | computed: {len(self.exposed_v2[i])}')
            # assert is_close(self.variable_solutions["sample_variables"][i]["z"], len(self.exposed_v2[i]), frac=0.05)
        
    # Delegation
    def lp_objective(self):
        # Sum across all z_i intermediate sample objectives
        if self.aggregation_method == "mean":
            return self.mean_lp_objective()
        elif self.aggregation_method == "max":
            return self.max_lp_objective()
        raise ValueError(f"Invalid Aggregation Method: {self.aggregation_method}")

    def lp_objective_value(self):
        if self.aggregation_method == "mean":
            return self.mean_lp_objective_value()
        elif self.aggregation_method == 'max':
            return self.max_lp_objective_value()
        raise ValueError(f"Invalid Aggregation Method: {self.aggregation_method}")

    # Max aggregation 
    def max_lp_objective(self):
        for i in range(self.num_samples):
            self.solver.Add(self.sample_variables[i]["z"] <= self.Z)
        num_exposed: Objective = self.solver.Objective()
        num_exposed.SetCoefficient(self.Z, 1)
        num_exposed.SetMinimization()

    def max_lp_objective_value(self, i = None):
        if i is not None:
            return self.sample_variables[i]["z"].solution_value()
        max_objective = self.Z.solution_value()
        # computed_max_objective = max([self.sample_variables[i]["z"].solution_value() for i in range(self.num_samples)])
        # if not is_close(max_objective, computed_max_objective):
        #     print(f"Warning: {abs(max_objective - computed_max_objective)}")
        return max_objective

    # Mean aggegation
    def mean_lp_objective(self):
        # Objective: Minimize number of people exposed in contour2
        self.solver.Add(sum(self.sample_variables[i]["z"] for i in range(self.num_samples)) <= self.Z)
        num_exposed: Objective = self.solver.Objective()
        num_exposed.SetCoefficient(self.Z, 1)
        num_exposed.SetMinimization()

    # Mean aggegation
    def mean_lp_objective_value(self, i = None):
        # Will raise error if not solved
        # Number of people exposed in V2
        if i is not None:
            return self.sample_variables[i]["z"].solution_value()
        mean_objective = self.Z.solution_value() / self.num_samples
        assert is_close(mean_objective, mean([self.sample_variables[i]["z"].solution_value() for i in range(self.num_samples)]))
        return mean_objective

class ObjectiveMixin():
    def lp_objective():
        raise NotImplementedError
    def lp_objective_value():
        raise NotImplementedError

def is_close(a, b, tol=0.001, frac=None):
    if tol:
        return abs(b - a) <= tol
    return abs(b - a) <= a * frac

def is_sorted(l: List):
    return all(l[i] <= l[i+1] for i in range(len(l)-1))
def compute_border_edges(G, src, dest):
    contour_edges = []
    for i in src:
        for j in G.neighbors(i):
            if j in dest:
                contour_edges.append((i, j))
    return contour_edges

def compute_contour_edges(G: nx.Graph, I, V1, V2):
    return compute_border_edges(G, I, V1), compute_border_edges(G, V1, V2)

def grader(
    G,
    SIR,
    budget,
    transmission_rate,
    compliance_rate,
    action,
    structure_rate=0,
    grader_seed=None,
    num_samples=1,
    solver_id="GUROBI_LP",
):
    gproblem = MinExposedSAA.create(
        G=G,
        SIR=SIR,
        budget=budget,
        transmission_rate=transmission_rate,
        compliance_rate=compliance_rate,
        structure_rate=structure_rate,
        num_samples=num_samples,
        seed=grader_seed,
        solver_id=solver_id,
    )

    # Pre-set the solveable parameters
    for node in action:
        gproblem.set_variable_id(node, 1)
    # Set the rest to zero
    for node in gproblem.contour1 - action:
        gproblem.set_variable_id(node, 0)

    _ = gproblem.solve_lp()
    return gproblem.objective_value

def optimal_baseline(
    G,
    SIR,
    budget,
    transmission_rate,
    compliance_rate,
    structure_rate=0,
    grader_seed=None,
    num_samples=1,
    solver_id="GUROBI_LP",
):
    gproblem = MinExposedSAA.create(
        G=G,
        SIR=SIR,
        budget=budget,
        transmission_rate=transmission_rate,
        compliance_rate=compliance_rate,
        structure_rate=structure_rate,
        num_samples=num_samples,
        seed=grader_seed,
        solver_id=solver_id,
    )
    _ = gproblem.solve_lp()
    return gproblem.objective_value
