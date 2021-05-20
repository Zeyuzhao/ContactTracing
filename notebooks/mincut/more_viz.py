# %%
from enum import Enum
import scipy
from ctrace.min_cut import min_cut_solver, SIR, min_cut_round
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
import seaborn as sns
%load_ext autoreload
%autoreload 2

# %%
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

# %%


def max_degree_edges(G, edges, budget):
    edge_max_degree = {(u, v): max(G.degree(u), G.degree(v))
                       for (u, v) in edges}
    edges_by_degree = sorted(
        edge_max_degree, key=edge_max_degree.get, reverse=True)
    return edges_by_degree[:budget]


def degree_solver(G, SIR, budget):
    edges = max_degree_edges(G, list(G.edges(SIR.I)), budget)
    return edges


def random_solver(G, SIR, budget, seed=None):
    edges = np.random.choice(list(G.edges(SIR.I)), size=budget, replace=False)
    return edges


def randomize(sir, p=0.05, seed=None):
    """
    Flips between S (1) and I (2) with probability p.
    """
    out = sir.copy()
    rg = np.random.default_rng(seed)
    for n, status in enumerate(sir):
        rand = rg.random()
        if (status == SIR.S or status == SIR.I) and rand < p:
            out[n] = 3 - status
    return out
# %%


def experiment(code_name="sparse", transmission=0.3, rand_resp_prob=0.1, ax=None):
    seed = 42
    budget = 400
    num_infected = 30
    method = "LP"
    methods = {
        "LP": min_cut_round,
        "greedy": degree_solver,
        "random": random_solver,
    }

    G, pos = small_world_grid(
        width=20,
        max_norm=True,
        sparsity=0.1,
        p=1,
        local_range=1,
        num_long_range=0.2,
        r=2,
        seed=seed
    )

    rg = np.random.default_rng(seed)
    # Sample the edges that actually transmit the disease
    active_edges = set(uniform_sample(G.edges, transmission, rg=rg))
    G_transmit = nx.subgraph_view(
        G, filter_edge=lambda x, y: (x, y) in active_edges)

    actual_sir = random_init(G, num_infected=num_infected, seed=seed)

    visible_sir = randomize(actual_sir, p=rand_resp_prob, seed=seed)

    solver = methods[method]
    edge_rec = solver(G, visible_sir, budget=budget)
    if method in ["greedy", "random"]:
        edge_rec = {e: 1 for e in edge_rec}
    # Evaluation
    vertex_soln, edge_soln = min_cut_solver(
        G_transmit,
        actual_sir,
        budget=budget,
        partial=edge_rec,
        mip=False
    )

    rand_G = G_transmit.copy()
    nx.set_node_attributes(rand_G, actual_sir.to_dict(), "actual_sir")
    nx.set_node_attributes(rand_G, visible_sir.to_dict(), "visible_sir")
    nx.set_node_attributes(rand_G, vertex_soln, "status")
    nx.set_edge_attributes(rand_G, edge_soln, "cut")

    transmit = {e: vertex_soln[e[0]] or vertex_soln[e[1]]
                for e in rand_G.edges}
    nx.set_edge_attributes(rand_G, transmit, "transmit")

    draw_style(rand_G, min_cut_node_style,
               min_cut_edge_style, ax=ax, DEBUG=False)

    return vertex_soln, edge_soln


v, e = experiment(transmission=0.3)
# %%

name = "transmission"
a = 3
b = 3
fig, axes = plt.subplots(a, b, figsize=(4 * a, 4 * b))

transmit = [0.1 * x for x in range(1, 11)]
for i, ax in tqdm(enumerate(axes.flatten()), total=a*b):
    ax.set_title(f"Transmission: {transmit[i]:.3f}", fontsize=8)
    experiment(transmission=transmit[i], ax=ax)

fig.savefig(f"figs/{name}.png")
fig.savefig(f"figs/{name}.svg")

plt.close(fig)

# %%

name = "privacy_high"
a = 3
b = 3
fig, axes = plt.subplots(a, b, figsize=(4 * a, 4 * b))

rand = [0.05 * x for x in range(1, 11)]
for i, ax in tqdm(enumerate(axes.flatten()), total=a*b):
    experiment(rand_resp_prob=rand[i], transmission=0.9, ax=ax)
    ax.set_title(f"RR Gamma: {rand[i]:.3f}", fontsize=8)

fig.savefig(f"figs/{name}.png")
fig.savefig(f"figs/{name}.svg")

plt.close(fig)

# %%

name = "privacy_low"
a = 3
b = 3
fig, axes = plt.subplots(a, b, figsize=(4 * a, 4 * b))

rand = [0.05 * x for x in range(1, 11)]
for i, ax in tqdm(enumerate(axes.flatten()), total=a*b):
    experiment(rand_resp_prob=rand[i], transmission=0.9, ax=ax)
    ax.set_title(f"RR Gamma: {rand[i]:.3f}", fontsize=8)

fig.savefig(f"figs/{name}.png")
fig.savefig(f"figs/{name}.svg")

plt.close(fig)
