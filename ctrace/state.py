
import json

from networkx import Graph

from . import PROJECT_ROOT


class SIR_State():
    def __init__(self, graph:Graph, initial_infections:int, shocks:int, initial_iterations:int):
        self.graph = graph
        self.initial_infections = initial_infections
        self.shocks = shocks
        self.initial_infections = initial_infections

        self.S = set()
        self.I = set()
        self.R = set()


    @staticmethod
    def from_cache(graph_path, cache_path, args_path):
        pass

    @staticmethod
    def from_simulation(initial_infections: int, shocks: int, initial_iterations: int):
        pass

