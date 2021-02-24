import abc
import random
import itertools
import networkx as nx
import numpy as np

from typing import Dict, List, Tuple
from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Variable, Constraint, Objective
from statistics import mean

from .round import D_prime
from .utils import pq_independent, find_excluded_contours, min_exposed_objective, uniform_sample
from .simulation import InfectionInfo

class MinExposedProgram:
    def __init__(self, info: InfectionInfo, solver_id="GLOP"):
        
        self.result = None
        self.info = info
        self.G = info.G
        self.SIR = info.SIR
        self.budget = info.budget
        self.p = info.transmission_rate
        self.contour1, self.contour2 = self.info.V1, self.info.V2
        self.solver = pywraplp.Solver.CreateSolver(solver_id)

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
    def __init__(self, info: InfectionInfo, solver_id="GLOP", num_samples=10, compliance_rate=1, structure_rate=0, seed=42):
        random.seed(seed)
        
        self.result = None
        self.info = info

        # Set problem structure
        self.G: nx.Graph = info.G
        self.SIR = info.SIR
        self.budget = info.budget
        self.contour1, self.contour2 = self.info.V1, self.info.V2
        self.solver = pywraplp.Solver.CreateSolver(solver_id)
        # self.contour_unknown = set(self.G.nodes) - set(self.SIR[1]) - set(self.contour1) - set(self.contour2)
        
        # Set SAA parameters - affects the samples.
        self.num_samples = num_samples

        # Conditional probability that edge can carry disease
        self.p = info.transmission_rate
        # Probability that V1 node observes quarantine
        self.q = compliance_rate
        # Probability to include edge in ((V1 x V2) - E)
        self.s = structure_rate
        
        self.sample_data = [{} for _ in range(self.num_samples)]
        self.init_samples()
        
        # Partial evaluation storage - applies only to controllable variables (X1)
        self.partials = {}
        # Controllable - contour1
        self.X1: Dict[int, Variable] = {}
        self.Y1: Dict[int, Variable] = {}
        self.Z = None
        # Non-controllable variables - over series.
        self.sample_variables =[{} for _ in range(self.num_samples)]
        self.init_variables()
        self.init_constraints()
        

    def init_samples(self):
        # Transmission Sampling
        for i in range(self.num_samples):

            # STRUCTURAL
            # Structural edges are sampled into existance.
            # Currently - structural edges must be in (I x V1) and (V1 x V2)
            structural_edges = ([], [])
            structural_edges[0] = uniform_sample(list(itertools.product(self.SIR[1], self.contour1)), self.s)
            structural_edges[1] = uniform_sample(list(itertools.product(self.contour1, self.contour2)), self.s)
            self.sample_data[i]["structural_edges"] = structural_edges
            
            # Expanded network (with sampled structural edges)
            GE = self.G.copy()
            GE.add_edges_from(structural_edges[0] + structural_edges[1])

            # DIFFUSION
            # Implementation Notes:
            # 1) Will use the graph obtained from structural uncertainty sampling
            # 2) border_edges[1] indicates infectious edges from u in V1 -> v in V2. 
            # Any edge here represents a binding constraint: 
            # Infectious edges (_, u) and (u, v) are sampled

            self.sample_data[i]["border_edges"] = ([], [])
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
            self.sample_data[i]["non_compliant"] = set(uniform_sample(self.contour1,  1 - self.compliance_rate))

    def init_variables(self):
        for u in self.contour1:
            self.X1[u] = self.solver.NumVar(0, 1, f"V1_x{u}")
            self.Y1[u] = self.solver.NumVar(0, 1, f"V1_y{u}")
        
        for i in range(self.num_samples):
            variables = {}
            variables[i]["Y1"] = [self.solver.NumVar(0, 1, f"V1[{i}]_y{u}") for u in self.contour1]
            variables[i]["Y2"] = [self.solver.NumVar(0, 1, f"V2[{i}]_y{v}") for v in self.contour2]
            variables[i]["z"] = self.solver.NumVar(0, self.solver.infinity(), f"z[{i}]")
            self.sample_variables[i] = variables

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
                if u in self.non_compliant_samples[i]:
                    self.solver.Add(self.sample_variables[i]["Y1"][u] >= 1)

        # Link sample Y1s with sample Y2s
        for i in range(self.num_samples):
            for (u, v) in self.sample_data[i]["border_edges"][1]:
                self.solver.Add(self.sample_variables[i]["Y2"][v] >= self.sample_variables[i]["Y1"][u])
        
        # Definition of z
        for i in range(self.num_samples):
            self.solver.Add(sum(self.sample_variables[i].values()) == self.sample_variables[i]["z"])
        # Combine using given lp_objective
        self.lp_objective()
    
    def lp_objective(self):
        # Sum across all z_i intermediate sample objectives
        return self.mean_lp_objective()
    def lp_objective_value(self):
        return self.mean_lp_objective_value()

    def max_lp_objective(self):
        pass

    def max_lp_objective_value(self):
        pass

    def mean_lp_objective(self):
        # Objective: Minimize number of people exposed in contour2
        num_exposed: Objective = self.solver.Objective()
        for i in range(self.num_samples):
            num_exposed.SetCoefficient(self.sample_variables[i]["z"], 1)
        num_exposed.SetMinimization()

    def mean_lp_objective_value(self, i = None):
                # Will raise error if not solved
        # Number of people exposed in V2
        if i is not None:
            return self._mean_lp_objective_value(i)
        return mean([self._mean_lp_objective_value(i) for i in range(self.num_samples)])

    def _mean_lp_objective_value(self, i):
        objective_value = 0
        for v in self.contour2:
            objective_value += self.sample_variables[i]["Y2"][v].solution_value()
        return objective_value

class MinExposedSAADiffusion(MinExposedProgram):
    def __init__(self, info: InfectionInfo, solver_id="GLOP", num_samples=10, seed=42):
        self.result = None
        self.info = info
        self.G = info.G
        self.SIR = info.SIR
        self.budget = info.budget
        self.p = info.transmission_rate
        self.contour1, self.contour2 = self.info.V1, self.info.V2
        self.solver = pywraplp.Solver.CreateSolver(solver_id)
        self.num_samples = num_samples

        random.seed(seed)

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

        # A collection of edges ordered by sample id and layer.
        # There are two layers - I -> V1, and V1 -> V2 edges.
        self.edge_samples = [([], []) for i in range(self.num_samples)]

        # Collection of infected v1s ordered by sample id
        self.v1_samples = [set() for i in range(self.num_samples)]

        self.init_samples()

        self.init_variables()

        # Initialize constraints
        self.init_constraints()

    def init_samples(self):
        for i in range(self.num_samples):
            # I -> V1 sampling
            for infected in self.SIR[1]:
                for v1 in self.G.neighbors(infected):
                    if v1 in self.contour1 and random.random() < self.p: 
                        self.edge_samples[i][0].append((infected, v1))
                        self.v1_samples[i].add(v1)
            # V1 -> V2 (Conditional on V1 being infected) for i in range(self.num_samples):
            for v1 in self.contour1:
                if v1 in self.v1_samples[i]:
                    for v2 in self.G.neighbors(v1):
                        if v2 in self.contour2 and random.random() < self.p:
                            self.edge_samples[i][1].append((v1, v2))        

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
        # Using edge_samples from V1->V2
        for i in range(self.num_samples):
            for (u, v) in self.edge_samples[i][1]:
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

class MinExposedSAACompliance(MinExposedProgram):
    def __init__(self, info: InfectionInfo, solver_id="GLOP", compliance_rate=0.5, num_samples=10, seed=42):
        self.result = None
        self.info = info
        self.G = info.G
        self.SIR = info.SIR
        self.budget = info.budget
        self.p = info.transmission_rate
        self.contour1, self.contour2 = self.info.V1, self.info.V2
        self.solver = pywraplp.Solver.CreateSolver(solver_id)
        self.num_samples = num_samples
        self.compliance_rate = compliance_rate

        random.seed(seed)

        if self.solver is None:
            raise ValueError("Solver failed to initialize!")

        # Compute P, Q from SIR
        self.P, self.Q = pq_independent(self.G, self.SIR.I, self.contour1, self.p)
    
        # Partial evaluation storage
        self.partials = {}

        # controllable - contour1
        self.X1: Dict[int, Variable] = {}
        self.Y1: Dict[int, Variable] = {}

        # non-controllable - contour 1 and contour2 (over i samples)
        self.Y1_samples: List[Dict[int, Variable]] = [{} for i in range(self.num_samples)]
        self.Y2_samples: List[Dict[int, Variable]] = [{} for i in range(self.num_samples)]
        
        # Sampled nodes
        self.non_compliant_samples = []
        self.init_samples()
        self.init_variables()

        # Initialize constraints
        self.init_constraints()

    def init_samples(self):
        # A list of sets of edges between v1 and v2 that are actually sampled for iteration i
        self.non_compliant_samples = [set(uniform_sample(self.contour1,  1 - self.compliance_rate)) for i in range(self.num_samples)]

        # Temporary test:
        # contour1_list = sorted(list(self.contour1))
        # self.non_compliant_samples = [{contour1_list[i] for i in range(30)} for i in range(self.num_samples)]
    
    def init_variables(self):
        """Declare variables as needed"""
        for u in self.contour1:
            self.X1[u] = self.solver.NumVar(0, 1, f"V1_x{u}")
            self.Y1[u] = self.solver.NumVar(0, 1, f"V1_y{u}")

        for i in range(self.num_samples):
            for v in self.contour1:
                self.Y1_samples[i][v] = self.solver.NumVar(0, 1, f"V1[{i}]_x{v}")
            for v in self.contour2:
                self.Y2_samples[i][v] = self.solver.NumVar(0, 1, f"V2[{i}]_y{v}")

    def init_constraints(self):
        """Initializes the constraints according to the relaxed LP formulation of MinExposed"""

        # X-Y are complements in contour1
        for u in self.contour1:
            self.solver.Add(self.X1[u] + self.Y1[u] == 1)

        # cost (number of people quarantined) must be within budget
        cost: Constraint = self.solver.Constraint(0, self.budget)
        for u in self.contour1:
            cost.SetCoefficient(self.X1[u], 1)

        # People not asked to quarantine would not quarantine
        for i in range(self.num_samples):
            for u in self.contour1:
                self.solver.Add(self.Y1_samples[i][u] >= self.Y1[u])

        # non-compliant contour1
        for i in range(self.num_samples):
            for u in self.contour1:
                if u in self.non_compliant_samples[i]:
                    self.solver.Add(self.Y1_samples[i][u] >= 1)


        for i in range(self.num_samples):
            for u in self.contour1:
                for v in self.G.neighbors(u):
                    if v in self.contour2:
                        self.solver.Add(self.Y2_samples[i][v] >= self.Y1_samples[i][u])

        # Objective: Minimize number of people exposed in contour2

        # Optimizing for sum over different scenarios (or minimizing the average)
        num_exposed: Objective = self.solver.Objective()
        for i in range(self.num_samples):
            for v in self.contour2:
                num_exposed.SetCoefficient(self.Y2_samples[i][v], 1)
        num_exposed.SetMinimization()

    def lp_objective_value(self):
        # Will raise error if not solved
        # Number of people exposed in V2
        objective_value = 0
        for i in range(self.num_samples):
            for v in self.contour2:
                objective_value += self.Y2_samples[i][v].solution_value()
        return objective_value / self.num_samples
        
    def lp_sample_objective_value(self, i):
        objective_value = 0
        for v in self.contour2:
            objective_value += self.Y2_samples[i][v].solution_value()
        return objective_value
    
class MinExposedSAAStructure():
    pass



def compute_border_edges(G, src, dest):
    contour_edges = []
    for i in src:
        for j in G.neighbors(i):
            if j in dest:
                contour_edges.append((i, j))
    return contour_edges

def compute_contour_edges(G: nx.Graph, I, V1, V2):
    return compute_border_edges(G, I, V1), compute_border_edges(G, V1, V2)
