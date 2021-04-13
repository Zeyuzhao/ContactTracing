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
width=20
# Graph Structure
r=2 # Parameter of long ranged decay
sparsity=0.1
num_long_range=0.1
# SIR
num_infected=20
# Transmission dynamics
transmission_rate=0.4
compliance_rate=.9
structure_rate=0
# Problem constraints,
budget=20
# Method 
method="SAA"
num_samples=500
aggregation_method="max"
# Evaluation
eval_num_samples=500
eval_aggregation_method="max"
# Seeding
seed=1000


#%%
# <================ Create Problem Instance ================>


G, pos = small_world_grid(
    width, 
    max_norm=True, 
    sparsity=sparsity, 
    local_range=1, 
    num_long_range=num_long_range, 
    r=r,
    seed=seed,
)

SIR = random_init(G, num_infected=num_infected, seed=seed)

# passing in edges with "long" attribute
edges = list(G.edges.data("long", default=False))
draw_single(G, pos=pos, sir=SIR, edges=edges, title="Graph Struct", figsize=(5,5))

#%%
# <================ SAA Agent ================>

# Group these attributes into problem dataclasses!
info = SAAAgentGurobi(
    G=G,
    SIR=SIR,
    budget=budget,
    num_samples=num_samples,
    transmission_rate=transmission_rate, 
    compliance_rate=compliance_rate, 
    structure_rate=structure_rate,
    aggregation_method=aggregation_method,
    seed=seed,
    solver_id="GUROBI",
)
print("SAA Complete")
action = info["action"]
problem: MinExposedSAA = info["problem"]
saa_objectives = [problem.lp_objective_value(i) for i in range(num_samples)]
saa_objective_value = problem.lp_objective_value()
# Evaluation
grader_seed = seed # arbitary value (should be different)

gproblem: MinExposedSAA = grader(
    G,
    SIR,
    budget,
    transmission_rate,
    compliance_rate,
    action,
    structure_rate=structure_rate,
    grader_seed=grader_seed,
    num_samples=eval_num_samples,
    aggregation_method=eval_aggregation_method,
    solver_id="GUROBI",
)
print("SAA Evaluation Complete")
grader_objective_value = gproblem.objective_value
grader_objectives = [gproblem.lp_objective_value(i) for i in range(eval_num_samples)]
# <================ Compute baselines ================>

info = InfectionInfo(G, SIR, budget, transmission_rate)

# Weighted Greedy
greedy_action = DegGreedy(info)
grader_greedy = grader(
    G,
    SIR,
    budget,
    transmission_rate,
    compliance_rate,
    greedy_action,
    structure_rate=structure_rate,
    grader_seed=grader_seed,
    num_samples=eval_num_samples,
    aggregation_method=eval_aggregation_method,
    solver_id="GUROBI",
)
print("Weighted Evaluation Complete")
greedy_objective_value = grader_greedy.objective_value
grader_greedy_objectives = [grader_greedy.lp_objective_value(i) for i in range(eval_num_samples)]


# Random
random_action = Random(info)
grader_random = grader(
    G,
    SIR,
    budget,
    transmission_rate,
    compliance_rate,
    random_action,
    structure_rate=structure_rate,
    grader_seed=grader_seed,
    num_samples=eval_num_samples,
    aggregation_method=eval_aggregation_method,
    solver_id="GUROBI",
)
print("Random Evaluation Complete")
random_objective_value = grader_random.objective_value
grader_random_objectives = [grader_random.lp_objective_value(i) for i in range(eval_num_samples)]

#%%
# sns.histplot(saa_objectives, color='orange')
# sns.histplot(grader_objectives, color='red')
# sns.histplot(grader_greedy_objectives, color='blue')
# sns.histplot(grader_random_objectives, color='green')

fig = plt.figure()
df = pd.DataFrame()
df["saa_objectives"] = saa_objectives
df["grader_objectives"] = grader_objectives
df["grader_greedy_objectives"] = grader_greedy_objectives
df["grader_random_objectives"] = grader_random_objectives
df.to_csv("dist2.csv")
ax = sns.histplot(df)
ax.set_xlabel("Objective Value (Number of V2 Exposed)")

fig.savefig('objective_dist.png')
#%%
g_min_idx = df["grader_objectives"].idxmax()
g_min = max(df["grader_objectives"])
assert g_min == df["grader_objectives"][g_min_idx]
g_min

g_min_idxs = df.index[df['grader_objectives'] == g_min].tolist()
#%%
df.describe()
#%%
# Visualization

def viz_saa(problem: MinExposedSAA, title="SAA samples", shift=0, sample_dim=(2,2)):
    args = []
    num_samples = sample_dim[0] * sample_dim[1]
    for sample_num in range(shift, shift+num_samples):
        # Sample Data
        non_compliant_samples = problem.sample_data[sample_num]["non_compliant"]
        relevant_v1 = problem.sample_data[sample_num]["relevant_v1"]

        edges = problem.sample_data[sample_num]["border_edges"]
        edge_colors = ["black"] * len(edges[0]) + ["grey"] * len(edges[1])

        # Key statistics
        exposed_v2 = problem.exposed_v2[sample_num]
        z_value = problem.variable_solutions["sample_variables"][sample_num]["z"]

        args.append({
            "title": f"Graph[{sample_num}]: LP: {z_value:.3f}",
            "pos": pos, 
            "sir":SIR,
            "quarantined_nodes":action, 
            "non_compliant_nodes": non_compliant_samples, 
            "exposed_nodes": exposed_v2, 
            "edges": edges[0] + edges[1], 
            "edge_color":edge_colors
        })
    fig, ax = draw_multiple_grid(G, args, *sample_dim)
    fig.suptitle(f'{title}: {problem.aggregation_method}')
    return fig, ax

# fig, ax = viz_saa(problem, shift=g_min_idx)
fig, ax = viz_saa(gproblem,shift=g_min_idx)

#%%
gproblem
# %%
fig.savefig("multi.png")
# %%

# %%

# %%
