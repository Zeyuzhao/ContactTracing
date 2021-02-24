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
import networkx as nx
from dataclasses import dataclass

from typing import Dict
from numbers import Number


random.seed(42)
# %%

# Utility Functions
def random_sir(G):
    nodes = set(G.nodes)
    I = set(random.sample(nodes, 10))
    R = set(random.sample(nodes - I, 10))
    S = nodes - I - R
    return SIR_Tuple(list(S), list(I), list(R))

def all_sus(G):
    return SIR_Tuple(set(G.nodes), set(), set())

def random_init(G, num_infected=5):
    nodes = set(G.nodes)
    I = set(random.sample(nodes, num_infected))
    R = set()
    S = nodes - I - R
    return SIR_Tuple(list(S), list(I), list(R))

def grid_sir(G, 
    ax, 
    pos:Dict[int,Number], 
    sir=None, 
    quarantined_nodes:List[int]=None, 
    non_compliant_nodes: List[int]=None, 
    edges:List[int]=None, 
    edge_color=None
):
    # G should be in a 2d grid form!
    if sir is None:
        sir = all_sus(G)

    if quarantined_nodes is None:
        quarantined_nodes = []
        # marked = random.sample(set(G.nodes), 10)

    if non_compliant_nodes is None:
        non_compliant_nodes = []
    if edges is None:
        edges = []

    if edge_color is None:
        edge_color = ["black"] * len(edges)

    if len(edges) != len(edge_color):
        raise ValueError("edges must match edge_colors")

    if pos is None:
        pos = {x: x["pos"] for x in G.nodes}
    
    node_size = [None] * len(G.nodes)
    node_color = [None] * len(G.nodes)
    border_color = [None] * len(G.nodes)
    linewidths = [0] * len(G.nodes)
    for i in range(len(G.nodes)):
        # Handle SIR
        if i in sir.S:
            node_size[i] = 10
            if i in non_compliant_nodes:
                node_color[i] = "red"
            else:
                node_color[i] = "black"
        elif i in sir.I:
            node_size[i] = 50
            node_color[i] = "red"
        else:
            node_size[i] = 10
            node_color[i] = "silver"

        # Handle Quarantine
        if i in quarantined_nodes:
            border_color[i] = "lawngreen"
            linewidths[i] = 1
        else:
            border_color[i] = "black"
            linewidths[i] = 1
    
    # Draw edges that are from I, V1, and V2
    nodes = nx.draw_networkx_nodes(G, pos=pos, node_color=node_color, node_size=node_size, edgecolors=border_color, linewidths=linewidths, ax=ax)

    # TODO: Draw v1-v2 edges
    
    nx.draw_networkx_edges(G, pos=pos, edgelist=edges, edge_color=edge_color, width=[], ax=ax)

def draw_single(G, **args):
    fig, ax = plt.subplots(figsize=(4,4))
    grid_sir(G, ax, **args)

def draw_multiple(G, args):
    fig, ax = plt.subplots(1, len(args), figsize=(4 * len(args),4))
    for i, config in enumerate(args):
        grid_sir(G, ax[i], **config)
    return fig, ax

#%%
# Create graph (with diagonal connections) to experiment on
width=20
G = nx.grid_2d_graph(width, width)
G.add_edges_from([
    ((x, y), (x+1, y+1))
    for x in range(width-1)
    for y in range(width-1)
] + [
    ((x+1, y), (x, y+1))
    for x in range(width-1)
    for y in range(width-1)
])
G.remove_nodes_from(uniform_sample(G.nodes(), 0.2))
mapper = {n : i for i, n in enumerate(G.nodes())}
pos = {i:(y,-x) for i, (x,y) in enumerate(G.nodes())}
G = nx.relabel_nodes(G, mapper)

sir = random_init(G, num_infected=30)
# G.add_edges_from(uniform_sample(list(itertools.product(sir.I, sir.I)), 1/width))
draw_single(G, pos=pos, edges=G.edges)



# Agent Evaluation Time
# %%

# Create infection state
infection_info = InfectionInfo(G, sir, budget=50, transmission_rate=0.5)

# %%
num_samples = 3
info = SAAAgent(infection_info, debug=True, num_samples=num_samples, transmission_rate=1, compliance_rate=1, structure_rate=0, seed=42)
action = info["action"]
problem = info["problem"]


# %%
# sample_num = 0
# v1_infected = problem.v1_samples[sample_num]
# edges = problem.edge_samples[sample_num][0], problem.edge_samples[sample_num][1]
# edge_colors = ["grey"] * len(edges[0]) + ["black"] * len(edges[1])
# draw_single(G, pos=pos, sir=sir, marked_nodes=action, edges=edges[0] + edges[1], edge_color=edge_colors)

args = []
for sample_num in range(num_samples):
    non_compliant_samples = problem.sample_data[sample_num]["non_compliant"]
    relevant_v1 = problem.sample_data[sample_num]["relevant_v1"]
    edges = problem.sample_data[sample_num]["border_edges"]
    edge_colors = ["grey"] * len(edges[0]) + ["black"] * len(edges[1])
    args.append({
        "pos":pos, "sir":sir, "quarantined_nodes":action, "non_compliant_nodes": non_compliant_samples, "edges": edges[0] + edges[1], "edge_color":edge_colors
    })
fig, ax = draw_multiple(G, args)
# %%
fig.savefig("seq_diag_seed_42.svg")
# %%

# %%

# %%

# %%
