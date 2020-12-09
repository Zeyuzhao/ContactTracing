# Generate fractional solution from problem
from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Variable, Constraint, Solver, Objective

import networkx as nx
import matplotlib.pyplot as plt
with open("data/erdos_renyi/example1.txt", "rb") as f:
    G: nx.Graph = nx.read_edgelist(f)
    print(G)
    nx.draw(G)
    plt.show()

