# %%
from enum import Enum
import scipy
from ctrace.min_cut import min_cut_solver, SIR
from ctrace.drawing import *
from ctrace.utils import *
from ctrace.problem import *
from ctrace.recommender import *
from ctrace.dataset import *
from ctrace.simulation import *
import networkx as nx
from ortools.linear_solver.pywraplp import Variable, Constraint, Objective, Solver
from ortools.linear_solver import pywraplp
import numpy as np
import ipywidgets as widgets
import cProfile
import io
import pstats
%load_ext autoreload
%autoreload 2


# %%
# Attempt to load the montgomery dataset
G = load_graph('montgomery')

# Initial statistics for G
print(nx.info(G))
avg_degree = mean(dict(G.degree()).values())
# %%

# Sample the edges that actually transmit the disease
active_edges = set(uniform_sample(G.edges, 0.15))
aG = nx.subgraph_view(G, filter_edge=lambda x, y: (x, y) in active_edges)
# print(f"Num Edges: {len(aG.edges())}")
print(nx.info(aG))

# %%
actual_sir = PartitionSIR.from_I(
    uniform_sample(aG.nodes(), 0.005), size=len(aG))
print(len(set(actual_sir.I)))
# %%

base_vertex, base_edge = min_cut_solver(
    aG,
    actual_sir,
    budget=700,
    edge_costs=None,
    vertex_costs=None,
    privacy=None,
    partial=None,
    mip=False
)

# %%

component_sizes = pd.Series([len(c) for c in sorted(nx.connected_components(aG), key=len, reverse=True)])
component_sizes.mean()

# %%

plt.figure()
grid = sns.histplot(data=component_sizes)
grid.set(xscale="log", yscale="log")
plt.savefig("log_dist", dpi=400)

# %%
v_soln = pd.Series(base_vertex)
e_soln = pd.Series(base_edge)
#%%
v_soln.value_counts()
# %%
e_soln.value_counts()

# %%
