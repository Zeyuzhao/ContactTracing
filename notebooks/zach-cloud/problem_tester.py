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
G, pos = grid_2d(20, seed=42)
sir = random_init(G, num_infected=30, seed=42)

# Create infection state
infection_info = InfectionInfo(G, sir, budget=50, transmission_rate=0)
draw_single(G, pos=pos, sir=sir, edges=G.edges, title="Graph Struct")

#%%
sample_dim = (2, 2)
num_samples = sample_dim[0] * sample_dim[1]
info = SAAAgent(
    infection_info, 
    debug=True,
    solver_id="GUROBI",
    num_samples=num_samples, 
    transmission_rate=0.75, 
    compliance_rate=0.8, 
    structure_rate=0, 
    seed=42,
)
action = info["action"]
problem = info["problem"]

#%%
# Test for randomness

problem
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
        "title": f"Graph[{sample_num}]: MinExposed {z_value:.3f}",
        "pos": pos, 
        "sir":sir, 
        "quarantined_nodes":action, 
        "non_compliant_nodes": non_compliant_samples, 
        "exposed_nodes": exposed_v2, 
        "edges": edges[0] + edges[1], 
        "edge_color":edge_colors
    })
fig, ax = draw_multiple_grid(G, args, *(3,3))
# %%
fig.savefig("seq_diag_seed_42.svg")
# %%

# %%

# %%

# %%
