# %%
import pstats
import io
import cProfile
import ipywidgets as widgets
import numpy as np
from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Variable, Constraint, Objective, Solver
import networkx as nx
from ctrace.simulation import *
from ctrace.dataset import *
from ctrace.recommender import *
from ctrace.problem import *
from ctrace.utils import *
from ctrace.drawing import *
from ctrace.min_cut import min_cut_solver, SIR
import scipy
from enum import Enum
%load_ext autoreload
%autoreload 2


# %%

def randomize(sir, p=0.05):
    out = sir.copy()
    for n, status in enumerate(sir):
        rand = np.random.rand()
        if (status == SIR.S or status == SIR.I) and rand < p:
            out[n] = 3 - status
    return out


def reset_node_attrs(G, keep=['pos']):
    for n in G.nodes:
        data = G.nodes[n]
        d = {k: data.get(k) for k in keep if data.get(k) is not None}
        data.clear()
        data.update(d)


def reset_edge_attrs(G, keep=['long']):
    for e in G.edges:
        data = G.edges[e]
        d = {k: data.get(k) for k in keep if data.get(k) is not None}

        data.clear()
        data.update(d)


def reset_attrs(G):
    reset_node_attrs(G)
    reset_edge_attrs(G)

min_cut_node_style = {
    # Default styling
    "default": {
        "node_size": 20,
        "node_color": "black",
        "edgecolors": "black",
        "linewidths": 0.5,
    },
    # Attribute styling
    "visible_sir": {
        SIR.I: {"edgecolors": "purple", "linewidths": 1.5},
    },
    "actual_sir": {
        SIR.I: {"node_size": 50, "node_color": "red"},
    },
    "status": {
        # Is infected?
        False: {"node_color": "black"},
        True: {"node_color": "red"},
    },
}

min_cut_edge_style = {
    # connectionstyle and arrowstyle are function-wide parameters
    # NOTE: For limit the number of unique connectionstyle / arrowstyle pairs
    "default": {
        "edge_color": "black",
        "arrowstyle": "-",
    },
    "long": {
        False: {},
        True: {"connectionstyle": "arc3,rad=0.2"},
    },

    # Overriding (cut overrides transmission)
    "transmit": {
        False: {},
        True: {"edge_color": "red"},
    },
    "cut": {
        False: {},
        True: {"edge_color": "blue"},
    },
}


#%%
# <=========================== Graph/SIR Setup ===========================>
seed = 42
budget = 100

G, pos = small_world_grid(
    width=20,
    max_norm=True,
    sparsity=0.1,
    p=1,
    local_range=1,
    num_long_range=0.2,
    r=2,
    seed=42
)
actual_sir = random_init(G, num_infected=30, seed=seed)
long_edges = list(G.edges.data("long", default=False))


print(G.edges[list(G.edges)[0]])
visible_sir = randomize(actual_sir)

print(f"Actual Infected: {actual_sir.I}")
print(f"Visible Infected: {visible_sir.I}")

# draw_style(G, min_cut_node_style, min_cut_edge_style, ax=None, DEBUG=False)

# Randomize Response
#%%

nx.set_node_attributes(G, actual_sir.to_dict(), "actual_sir")

rand_G = G.copy()
nx.set_node_attributes(rand_G, visible_sir.to_dict(), "visible_sir")

base_G = G.copy()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
draw_style(base_G, min_cut_node_style, min_cut_edge_style, ax=ax, DEBUG=False)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
draw_style(rand_G, min_cut_node_style, min_cut_edge_style, ax=ax, DEBUG=False)

# %%
# <======================= Baseline =======================> #


base_vertex, base_edge = min_cut_solver(
    base_G,
    actual_sir,
    budget=budget,
    edge_costs=None,
    vertex_costs=None,
    privacy=None,
    partial=None,
    mip=True
)

base_sir = PartitionSIR.from_list([v + 1 for k, v in base_vertex.items()])

nx.set_node_attributes(base_G, base_vertex, "status")
nx.set_edge_attributes(base_G, base_edge, "cut")
transmit = {e: base_vertex[e[0]] or base_vertex[e[1]] for e in base_G.edges}
# nx.set_edge_attributes(G, transmit, "transmit")

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
draw_style(base_G, min_cut_node_style, min_cut_edge_style, ax=ax, DEBUG=False)

print(
    f"Baseline Exposed (Additional): {len(set(base_sir.I))} ({len(set(base_sir.I)) - len(set(actual_sir.I))})")


# %%

rec_vertex, rec_edge = min_cut_solver(
    rand_G,
    visible_sir,
    budget=budget,
    edge_costs=None,
    vertex_costs=None,
    privacy=None,
    partial=None,
    mip=True
)

sol_vertex, sol_edge = min_cut_solver(
    rand_G,
    actual_sir,
    budget=budget,
    edge_costs=None,
    vertex_costs=None,
    privacy=None,
    partial=rec_edge,
    mip=True
)

rec_sir = PartitionSIR.from_list([v + 1 for k, v in rec_vertex.items()])
sol_sir = PartitionSIR.from_list([v + 1 for k, v in sol_vertex.items()])
# %%
# Attribute Painter

nx.set_node_attributes(rand_G, actual_sir.to_dict(), "actual_sir")
nx.set_node_attributes(rand_G, visible_sir.to_dict(), "visible_sir")
nx.set_node_attributes(rand_G, sol_vertex, "status")
nx.set_edge_attributes(rand_G, sol_edge, "cut")

transmit = {e: sol_vertex[e[0]] or sol_vertex[e[1]] for e in rand_G.edges}
nx.set_edge_attributes(rand_G, transmit, "transmit")

draw_style(rand_G, min_cut_node_style,
           min_cut_edge_style, ax=None, DEBUG=False)

print(
    f"Predicted Exposed (Addl): {len(rec_sir.I)} ({len(rec_sir.I) - len(actual_sir.I)})")
print(
    f"Actual Exposed (Addl): {len(sol_sir.I)} ({len(sol_sir.I) - len(actual_sir.I)})")

# %%

active_edges = set(uniform_sample(G.edges, 0.15))
aG = nx.subgraph_view(G, filter_edge=lambda x, y: (x, y) in active_edges)
print(f"Num Edges: {len(aG.edges())}")
# %%

# %%
G, pos = small_world_grid(
    width=60,
    max_norm=True,
    sparsity=0.3,
    p=1,
    local_range=1,
    num_long_range=3,
    r=2,
    seed=42
)
actual_sir = random_init(G, num_infected=500, seed=seed)

component_sizes = pd.Series([len(c) for c in sorted(
    nx.connected_components(aG), key=len, reverse=True)])
component_sizes.max()


#%%




#%%
frac_vertex, frac_edge = min_cut_solver(
    G,
    actual_sir,
    budget=500,
    edge_costs=None,
    vertex_costs=None,
    privacy=None,
    partial=None,
    mip=False
)
print(pd.Series(frac_vertex).value_counts())
print(pd.Series(frac_edge).value_counts())
# %%
