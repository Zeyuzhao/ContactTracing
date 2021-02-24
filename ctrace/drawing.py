import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import seaborn as sns

from ctrace.simulation import *
from ctrace.dataset import *
from ctrace.recommender import *
from ctrace.problem import *
from ctrace.utils import *
import networkx as nx

from typing import Dict
from numbers import Number

random.seed(42)

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

def grid_sir(G: nx.Graph, 
    ax, 
    pos:Dict[int,Number], 
    sir=None, 
    quarantined_nodes:List[int]=[], 
    non_compliant_nodes: List[int]=[],
    exposed_nodes: List[int]=[],
    edges:List[int]=[], 
    edge_color=None,
    **args,
):
    # G should be in a 2d grid form!
    if sir is None:
        sir = all_sus(G)

    if quarantined_nodes is None:
        quarantined_nodes = []
        # marked = random.sample(set(G.nodes), 10)

    if non_compliant_nodes is None:
        non_compliant_nodes = []

    if exposed_nodes is None:
        exposed_nodes = []

    if edges is None:
        edges = G.edges

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
            elif i in exposed_nodes:
                node_color[i] = "yellow"
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
            border_color[i] = "tab:blue"
            linewidths[i] = 1
        else:
            border_color[i] = "black"
            linewidths[i] = 1
    
    # Draw edges that are from I, V1, and V2
    nodes = nx.draw_networkx_nodes(
        G, 
        pos=pos, 
        node_color=node_color, 
        node_size=node_size, 
        edgecolors=border_color, 
        linewidths=linewidths, 
        ax=ax
    )
    
    nx.draw_networkx_edges(
        G, 
        pos=pos, 
        edgelist=edges, 
        edge_color=edge_color, 
        width=[], 
        ax=ax
    )

def draw_single(G, **args):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_title(args["title"])
    grid_sir(G, ax, **args)

def draw_multiple(G, args):
    fig, ax = plt.subplots(1, len(args), figsize=(4 * len(args),4))
    for i, config in enumerate(args):
        ax[i].set_title(config["title"])
        grid_sir(G, ax[i], **config)
    return fig, ax

def draw_multiple_grid(G, args, a, b):
    fig, ax = plt.subplots(a, b, figsize=(4 * a, 4 * b))
    assert a * b == len(args)

    for i, ((x, y), config) in enumerate(zip(itertools.product(range(a), range(b)), args)):
        ax[x, y].set_title(config["title"])
        grid_sir(G, ax[x, y], **config)
    return fig, ax