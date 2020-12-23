import math

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Constraint, Solver, Variable, Objective

from typing import Set, Dict
import pickle as pkl

np.random.seed(42)
# ============================ Initialize Graph ============================

# Tree
# G = nx.balanced_tree(4, 2)

# Random Tree

# G = nx.barabasi_albert_graph(100, 2, seed=42)
# Load dataset

data_name = "mon10k"
data_dir = f"../data/mont/labelled/{data_name}"
G: nx.Graph = nx.read_edgelist(f"{data_dir}/data.txt", nodetype=int)

# Drawing Distances
# pos = nx.spring_layout(G, iterations=200, seed=42)
# dist_params = {
#     "pos": pos,
#     "with_labels": False,
# }
# nx.draw(G, **dist_params)
# plt.show()

# ============================ Level Contours ============================

# Set of initial infected
N = G.number_of_nodes()
NUM_INFECTED = int(N / 20)

rand_infected = np.random.choice(N, NUM_INFECTED, replace=False)
I_SET = set(rand_infected)
print(f"Infected: {I_SET}")

# COSTS = np.random.randint(1, 20, size=N)
COSTS = np.ones(N)
print(f"COSTS: {COSTS}")
# Compute distances
dist_dict = nx.multi_source_dijkstra_path_length(G, I_SET)

# convert dict vertex -> distance
# to distance -> [vertex]
level_dists = defaultdict(set)
for (i, v) in dist_dict.items():
    level_dists[v].add(i)

# print(level_dists)

# Obtain V_1 and V_2

# Set of vertices distance 1 away from infected I
V1: Set[int] = level_dists[1]

# Set of vertices distance 2 away from infected I
V2: Set[int] = level_dists[2]

print("============================ Level Contours ============================")
print(f"Distance 1: {V1}")
print(f"Distance 2: {V2}")

# convert dict of distances to array
dists = [0] * N
for (i, v) in dist_dict.items():
    dists[i] = v

# ============================ Constraints ============================
solver: Solver = pywraplp.Solver.CreateSolver('GLOP')

# Constants
k = int(0.8 * NUM_INFECTED) # the cost budget

# V1 indicator set
X1: Dict[int, Variable] = {}
Y1: Dict[int, Variable] = {}

# V2 indicator set
X2: Dict[int, Variable] = {}
Y2: Dict[int, Variable] = {}

# Declare Variables
for u in V1:
    X1[u] = solver.NumVar(0, 1, f"V1_x{u}")
    Y1[u] = solver.NumVar(0, 1, f"V1_y{u}")

for v in V2:
    X2[v] = solver.NumVar(0, 1, f"V2_x{v}")
    Y2[v] = solver.NumVar(0, 1, f"V2_y{v}")

# First set of constraints X + Y = 1
# By definition, X and Y sum to 1

# Quarantine (x) / Free (y) Indicators
# Parameter indicators (we have control)
for u in V1:
    solver.Add(X1[u] + Y1[u] == 1)

# Safe (x) / Exposed (y) Indicators
# Result indicators (we have no control)
for v in V2:
    solver.Add(X2[v] + Y2[v] == 1)


# Second set of constraints: k (cost) constraint
# The cost of quarantine is a linear combination
cost: Constraint = solver.Constraint(0, k)
for u in V1:
    # For now, the coefficient of every variable is 1 (The cost is uniform)
    cost.SetCoefficient(X1[u], int(COSTS[u]))


# Third set of constraints: specify who is considered "saved"
# (anyone "not exposed" must have no contact)

# or, once v in V1 is exposed (Y1 = 1),
# all v's neighbors in V2 must be exposed (Y2 >= Y1 = 1)

# We only examine edges between sets V1 and V2
for u in V1:
    for v in G.neighbors(u):
        if v in V2:
            solver.Add(Y2[v] >= Y1[u])

# Set minimization objective
# Number of people free in V1 and people exposed in V2
numExposed: Objective = solver.Objective()
for u in V1:
    numExposed.SetCoefficient(Y1[u], 1)

for v in V2:
    numExposed.SetCoefficient(Y2[v], 1)

numExposed.SetMinimization()

# Solve and display solution
status = solver.Solve()
if status == solver.OPTIMAL:
    print("============================ Optimal Solution ============================")
    # Indicators
    quaran_sol: Dict[int, float] = {}
    safe_sol: Dict[int, float] = {}

    # Implement Rounding?
    quarantined: Set[int] = set()
    safe: Set[int] = set()

    border_count = 0
    for u in V1:
        quaran_sol[u] = X1[u].solution_value()
        if abs(quaran_sol[u] - 0.5) < 0.4:
            border_count += 1
        if quaran_sol[u] > 0.5:
            quarantined.add(u)

    for v in V2:
        safe_sol[v] = X2[v].solution_value()
        if abs(safe_sol[v] - 0.5) < 0.4:
            border_count += 1
        if safe_sol[v] > 0.5:
            safe.add(v)

    print(f"Quarantined Solution: {quaran_sol}")
    print(f"Safe Solution: {safe_sol}")
    print(f"Border Count: {border_count}")
    print("============================ Basic Rounding ============================")
    # print(f"Quarantined: {quarantined}")
    # print(f"Safe: {safe}")
else:
    if status == solver.FEASIBLE:
        print("A potentially suboptimal solution was found.")
    else:
        print('The solver could not solve the problem.')

print('\nAdvanced usage:')
print('Problem solved in %f milliseconds' % solver.wall_time())
print('Problem solved in %d iterations' % solver.iterations())

# ============================ Drawing ============================

status = []
for i in range(N):
    if i in V1:
        c = "blue" if (i in quarantined) else "orange"
    elif i in V2:
        c = "green" if (i in safe) else "yellow"
    elif i in I_SET:
        c = "red"
    else:
        c = "grey"
    status.append(c)
    G.nodes[i]["color"] = c

# Regular
# pos = nx.spring_layout(G)
# dist_params = {
#     "pos": pos,
#     "node_color": status,
#     "with_labels": True,
# }
# nx.draw(G, **dist_params)
# plt.show()

# Large Graphs
# set layout

# Must load layout ahead of time from pickled library.
# pos = pkl.load(open(f"{data_dir}/pos.p", "rb"))
# dist_params = {
#     "pos": pos,
#     "node_color": status,
#     "with_labels": False,
#     "node_size": 1,
#     "width": 0.3
# }
#
# nx.draw_networkx(G, **dist_params)
# nx.draw_networkx_labels(G, pos, font_size=1)
#
# plt.savefig(f'../output/graph_{data_name}.png', dpi=1000)
# plt.show()










# # Drawing Distances
# dist_params = {
#     "pos": pos,
#     "node_color": dists,
#     "with_labels": True,
#     "cmap": plt.cm.Blues,
# }
# nx.draw(G, **dist_params)
# plt.show()