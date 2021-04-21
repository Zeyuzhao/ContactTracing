#%%
%load_ext autoreload

%autoreload 2
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math
import EoN
import seaborn as sns
import time

from ctrace.simulation import *
from ctrace.dataset import *
from ctrace.recommender import *

from ctrace.problem import *
from ctrace.utils import *
from ctrace.drawing import *

import networkx as nx
#%%
# Create graph (with diagonal connections) to experiment on

seed=42
G, pos = small_world_grid(10, max_norm=True, sparsity=0.1, local_range=1, num_long_range=0.5, r=2, seed=42)

#%%
# G, pos = grid_2d(8, seed=42, diagonals=True, sparsity=0.2, global_rate=0)
# G = nx.scale_free_graph(100)
# G = nx.generators.navigable_small_world_graph(10, p=1, q=1, r=2, dim=2, seed=None)
# G = G.to_undirected()

# mapper = {n : i for i, n in enumerate(G.nodes())}
# pos = {i:(y ,-x) for i, (x,y) in enumerate(G.nodes())}
# G = nx.relabel_nodes(G, mapper)

# pos = nx.nx_agraph.graphviz_layout(G, prog="sfdp", args="-Goverlap=false")
# pos = nx.nx_agraph.graphviz_layout(G, prog="sfdp", args="")
SIR = random_init(G, num_infected=10, seed=seed)
budget=15
transmission_rate=1
compliance_rate=1
structure_rate=0

# Create infection state
# # infection_info = InfectionInfo(G, SIR, budget=0, transmission_rate=0)
edges = list(G.edges.data("long", default=False))
# long_edges= list(filter(lambda x: x[2], edges))
# short_edges= list(filter(lambda x: not x[2], edges))
draw_single(G, pos=pos, sir=SIR, edges=edges, title="Graph Struct", figsize=(5,5))

#%%
# sigma = nx.algorithms.smallworld.sigma(G, niter=10, nrand=10)
# print(f"Smallworld {'TRUE' if sigma >= 1 else 'FALSE'} ({sigma})")

#%%
sample_dim = (2, 2)
num_samples = sample_dim[0] * sample_dim[1]
num_samples = 10
info = SAAAgentGurobi(
    G=G,
    SIR=SIR,
    budget=budget,
    num_samples=num_samples,
    transmission_rate=transmission_rate, 
    compliance_rate=compliance_rate, 
    structure_rate=structure_rate, 
    seed=seed,
    solver_id="GUROBI",
)
action = info["action"]
problem = info["problem"]

print(problem.get_variables())
print(action)
#%%
# Test for randomness

problem2 = MinExposedSAA.load_sample(G, SIR, budget, problem.sample_data, solver_id="GUROBI")
problem2.solve_lp()

probabilities2 = problem2.get_variables()
# Run assertion is Gurobi is used - checks if all values are 0 or 1
assert all(is_close(p, 0) or is_close(p, 1) for p in probabilities2)
action2 = set([problem2.quarantine_map[k] for (k,v) in enumerate(probabilities2) if v==1])

print(problem2.get_variables())
print(action2)
print("DIFF ACTION:", action ^ action2)

#%%
# Examine performance of greedy with respect to LP

# TODO: REDO THIS!!!

info = InfectionInfo(G, SIR, budget, transmission_rate)
# actions -> set of node ids
greed_action = DegGreedy(info)

# grader
gseed = 1011
gproblem_greedy = MinExposedSAA.create(
    G=G,
    SIR=SIR,
    budget=budget,
    transmission_rate=transmission_rate, 
    compliance_rate=compliance_rate, 
    structure_rate=structure_rate,
    num_samples=num_samples,
    seed=gseed,
    solver_id="GUROBI_LP",
)
for node in greed_action:   
    gproblem_greedy.set_variable_id(node, 1)
sol = gproblem_greedy.solve_lp()

gproblem_minex = MinExposedSAA.create(
    G=G,
    SIR=SIR,
    budget=budget,
    transmission_rate=transmission_rate, 
    compliance_rate=compliance_rate, 
    structure_rate=structure_rate,
    num_samples=num_samples,
    seed=gseed,
    solver_id="GUROBI_LP",
)
for node in action:
    gproblem_minex.set_variable_id(node, 1)
sol = gproblem_minex.solve_lp()

print(f"Evaluation Seed: {gseed}")
print(f"Greedy Objective: {gproblem_greedy.objective_value}")
print(f"SAA Objective: {gproblem_minex.objective_value}")


# Run evaluations over more instances:
def SAAEval(problem: MinExposedSAA, action, seed):
    evaluator = MinExposedSAA.create(
        G=problem.G,
        SIR=problem.SIR,
        budget=problem.budget,
        transmission_rate=problem.p, 
        compliance_rate=problem.q, 
        structure_rate=problem.s,
        num_samples=problem.num_samples,
        seed=seed,
        solver_id="GUROBI",
    )
    for node in action:
        evaluator.set_variable_id(node, 1)
    evaluator.solve_lp()
    return evaluator

[problem.variable_solutions["sample_variables"][i]["z"] for i in range(problem.num_samples)]


#%%

from ctrace.problem import grader
none_obj = grader(G,
    SIR,
    budget,
    transmission_rate,
    compliance_rate,
    set(),
    structure_rate=0,
    grader_seed=None,
    num_samples=50,
    solver_id="GUROBI_LP"
)
robust_obj = grader(G,
    SIR,
    budget,
    transmission_rate,
    compliance_rate,
    action,
    structure_rate=0,
    grader_seed=None,
    num_samples=50,
    solver_id="GUROBI_LP"
)

print(f"None: {none_obj}")
print(f"Robust: {robust_obj}")
#%%
# Visualization

def viz_saa(problem: MinExposedSAA):
    args = []
    for sample_num in range(num_samples):
        # Sample Data
        non_compliant_samples = problem.sample_data[sample_num]["non_compliant"]
        relevant_v1 = problem.sample_data[sample_num]["relevant_v1"]

        edges = problem.sample_data[sample_num]["border_edges"]
        edge_colors = ["black"] * len(edges[0]) + ["grey"] * len(edges[1])

        # Key statistics
        exposed_v2 = problem.exposed_v2[sample_num]
        z_value = problem.variable_solutions["sample_variables"][sample_num]["z"]

        args.append({
            "title": f"Graph[{sample_num}]: LP.{problem.aggregation_method}: {z_value:.3f}",
            "pos": pos, 
            "sir":SIR,
            "quarantined_nodes":action, 
            "non_compliant_nodes": non_compliant_samples, 
            "exposed_nodes": exposed_v2, 
            "edges": edges[0] + edges[1], 
            "edge_color":edge_colors
        })
    fig, ax = draw_multiple_grid(G, args, *sample_dim)
    return fig, ax

fig, ax = viz_saa(gproblem_greedy)
fig, ax = viz_saa(gproblem_minex)
# %%
fig.savefig("multi.png")
# %%
import seaborn as sns

[robust_obj.lp_objective_value(i) for i in range(40)]
# %%

# %%

# %%