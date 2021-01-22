import networkx as nx
from ortools.linear_solver.pywraplp import Constraint


class MinExposed:
    def __init__(self, G: nx.Graph, SIR, p, q, k, solver: str=""):
        pass

    def init_variables(self):
        raise NotImplementedError

    def init_constraints(self):
        raise NotImplementedError

    def solve_lp(self):
        pass

    def get_array_variables(self):
        """Returns array representation of indicator variables"""
        raise NotImplementedError

    def set_array_variable(self, index: int, value: int):
        """Sets the the ith indicator variable in the array representation"""
        pass

    def set_variable_id(self, id: int, value: int):
        """Sets a V1 indicator by its id"""
        pass

    def get_solution(self):
        raise NotImplementedError


class MinExposedLP:
    def __init__(self, G: nx.Graph, SIR, budgets, labels, p=None, q=None, solver: str=""):
        pass


class MinExposedMIP:
    def __init__(self, G: nx.Graph, SIR, budgets, p=None, q=None, solver: str = ""):
        pass


class InfectionState:
    def __init__(self):
        self.G = None
        self.S = None
        self.I_known = None
        self.I = None
        self.R = None

    def load_graph(self, name=""):
        pass

    def load_sir(self):
        pass

    def save_sir(self):
        pass