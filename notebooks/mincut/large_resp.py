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


#%%


def max_degree_edges(G, edges, budget):
    edge_max_degree = {(u, v): max(G.degree(u), G.degree(v))
                       for (u, v) in edges}
    edges_by_degree = sorted(
        edge_max_degree, key=edge_max_degree.get, reverse=True)
    return edges_by_degree[:budget]

def degree_solver(G, SIR, budget):
    edges = max_degree_edges(G, G.edges(SIR.I))
    return edges


def random_solver(G, SIR, budget):
    edges = np.random.choice(G.edges(SIR.I), size=budget, replace=False)
    return edges


def randomize(sir, p=0.05):
    """
    Flips between S (1) and I (2) with probability p.
    """
    out = sir.copy()
    for n, status in enumerate(sir):
        rand = np.random.rand()
        if (status == SIR.S or status == SIR.I) and rand < p:
            out[n] = 3 - status
    return out

# %%


methods = {}

def runner(
    G, 
    num_infected: int, 
    transmission: float, 
    rand_resp_prob: float, 
    budget: int, 
    method: str, 
    seed=None
):
    # Sample the edges that actually transmit the disease
    active_edges = set(uniform_sample(G.edges, 0.15))
    G_transmit = nx.subgraph_view(G, filter_edge=lambda x, y: (x, y) in active_edges)

    actual_sir = random_init(G, num_infected=num_infected, seed=seed)
    visible_sir = randomize(actual_sir)

    solver = methods[method]
    edge_rec = solver(G, visible_sir, budget)

    # Evaluation

    vertex_soln, edge_soln = min_cut_solver(
        G,
        actual_sir,
        budget=budget,
        partial=edge_rec,
        mip=False
    )
    score = sum(vertex_soln.values())
    return score



