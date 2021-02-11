#%%
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
import networkx as nx
from dataclasses import dataclass


#%%
G = nx.grid_2d_graph(20, 20)
mapper = {n : i for i, n in enumerate(G.nodes())}
pos = {i:(y,-x) for i, (x,y) in enumerate(G.nodes())}
G = nx.relabel_nodes(G, mapper)
# %%

# Utility Functions
def random_sir(G):
    nodes = set(G.nodes)
    I = set(random.sample(nodes, 10))
    R = set(random.sample(nodes - I, 10))
    S = nodes - I - R
    return SIR_Tuple(list(S), list(I), list(R))

def random_init(G, num_infected=5):
    nodes = set(G.nodes)
    I = set(random.sample(nodes, num_infected))
    R = set()
    S = nodes - I - R
    return SIR_Tuple(list(S), list(I), list(R))

def grid_sir(G, ax, pos=None, sir=None, marked_nodes=None, edges=None):
    # G should be in a 2d grid form!
    if sir is None:
        sir = random_sir(G)

    if marked_nodes is None:
        marked_nodes = []
        # marked = random.sample(set(G.nodes), 10)
    if edges is None:
        edges = []

    if pos is None:
        pos = {(x, y): (y, -x) for (x,y) in G.nodes}
    
    node_size = [None] * len(G.nodes)
    node_color = [None] * len(G.nodes)
    edge_color = [None] * len(G.nodes)
    linewidths = [0] * len(G.nodes)
    for i in range(len(G.nodes)):
        # Handle SIR
        if i in sir.S:
            node_size[i] = 10
            node_color[i] = "black"
        elif i in sir.I:
            node_size[i] = 50
            node_color[i] = "red"
        else:
            node_size[i] = 50
            node_color[i] = "silver"
        
        # Handle Quarantine
        if i in marked_nodes:
            edge_color[i] = "green"
            linewidths[i] = 2
        else:
            edge_color[i] = "black"
            linewidths[i] = 1
    
    # Draw edges that are from I, V1, and V2
    nodes = nx.draw_networkx_nodes(G, pos=pos, node_color=node_color, node_size=node_size, edgecolors=edge_color, linewidths=linewidths, ax=ax)

    # TODO: Draw v1-v2 edges
    edgelist = []
    nx.draw_networkx_edges(G, pos=pos, edgelist=edges, width=[], ax=ax)

def draw_single(G, sir=None, marked=None, edges=None):
    fig, ax = plt.subplots(figsize=(4,4))
    grid_sir(G, ax, pos=pos, sir=sir, marked_nodes=marked, edges=edges)

# %%
sir = random_init(G, num_infected=10)
# Create infection state
infection_info = InfectionInfo(G, sir, budget=10, transmission_rate=0.5)

# %%
info = SAADiffusionAgent(infection_info, debug=True)
action = info["action"]
problem = info["problem"]

draw_single(G, sir=sir, marked=action)


# %%
edges = problem.contour_edge_samples[6]
draw_single(G, sir=sir, marked=action, edges=edges)
# %%

# %%
