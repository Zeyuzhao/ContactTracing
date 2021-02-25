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
G, pos = grid_2d(20, seed=seed)
SIR = random_init(G, num_infected=20, seed=seed)
budget=50
transmission_rate=0.8 
compliance_rate=0.8 
structure_rate=0


# Create infection state
# infection_info = InfectionInfo(G, SIR, budget=0, transmission_rate=0)
draw_single(G, pos=pos, sir=SIR, edges=G.edges, title="Graph Struct")

#%%
sample_dim = (2, 2)
num_samples = sample_dim[0] * sample_dim[1]
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
# Visualization
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
# %%
fig.savefig("seq_diag_seed_42.svg")
# %%

# %%

# %%

# %%
