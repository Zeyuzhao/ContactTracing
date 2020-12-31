from typing import Union, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import networkx as nx
from .constraint import *

# custom colormap
top = cm.get_cmap('Oranges_r', 128)
bottom = cm.get_cmap('Blues', 128)

newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')


def draw_prob(G: nx.Graph, I, V1, V2, quarantined, saved, p1, transition):
    """Generates graph visualization of the probabilistic p < 1 case

    Parameters
    ----------
    G
        Contact Tracing Graph
    I
        Initial Infected
    V1
        1st infected contour
    V2
        2nd infected contour
    quarantined
        V1 x indicators (blue x = 1, none x = 0)
    saved
        V2 x colormap (yellow => higher chance of infection, green => lower chance of infection)
    p1
        V1 original odds
    transition
        conditional probability of infection from V1 to V2

    Returns
    -------
    None
    """
    color = Tuple[float, float, float, float]
    optional_color = List[Union[color, None]]

    RED: color = (1, 0, 0, 1)
    BLUE: color = (0, 0, 1, 1)
    GREY: color = (0.5, 0.5, 0.5, 1)
    BLACK: color = (0, 0, 0, 1)

    oranges = plt.get_cmap("Oranges")
    blues = plt.get_cmap("Blues")
    reds = plt.get_cmap("Reds")
    yellow = plt.get_cmap("YlGn")

    N = G.number_of_nodes()

    # color the node filling and border
    fill: optional_color = [GREY] * N
    border: optional_color = [BLACK] * N

    for i in range(N):
        if i in I:
            fill[i] = RED
        elif i in V1:
            fill[i] = oranges(p1[i])
            if i in quarantined:
                border[i] = BLUE
        elif i in V2:
            fill[i] = yellow(saved[i])
        else:
            c = GREY

    # color the edges
    M = len(G.edges)
    edge_color: optional_color = [BLACK] * M
    widths = [0] * M
    for i, (a, b) in enumerate(G.edges()):
        if a in V1 and b in V2:
            edge_color[i] = reds(transition[a][b])
            widths[i] = 1

    pos = nx.spring_layout(G)
    dist_params = {
        "pos": pos,
        "node_color": fill,
        "edgecolors": border,
        "linewidths": widths
    }
    edge_params = {
        "pos": pos,
        "edge_color": edge_color,
    }

    nx.draw_networkx_nodes(G, **dist_params)
    nx.draw_networkx_edges(G, **edge_params)
    nx.draw_networkx_labels(G, pos=pos)


def draw_contours(G: nx.Graph, I, V1, V2, name=None):
    status = []
    N = G.number_of_nodes()
    for i in range(N):
        if i in V1:
            c = "blue"
        elif i in V2:
            c = "green"
        elif i in I:
            c = "red"
        else:
            c = "grey"
        status.append(c)
        G.nodes[i]["color"] = c

    pos = nx.nx_agraph.graphviz_layout(G, prog="sfdp", args="-Goverlap=false")

    dist_params = {
        "pos": pos,
        "node_color": status,
        "with_labels": False,
        "node_size": 1,
        "width": 0.3
    }
    nx.draw_networkx(G, **dist_params)
    nx.draw_networkx_labels(G, pos, font_size=1)

    if name is None:
        name = "graph"

    plt.savefig(f'../output/{name}.png', dpi=1000)
    plt.show()


def draw_absolute(G: nx.Graph, I, V1, V2, quarantined, safe, name=None):
    status = []
    N = G.number_of_nodes()
    for i in range(N):
        if i in V1:
            c = "blue" if (i in quarantined) else "orange"
        elif i in V2:
            c = "green" if (i in safe) else "yellow"
        elif i in I:
            c = "red"
        else:
            c = "grey"
        status.append(c)
        G.nodes[i]["color"] = c

    # Small

    if N < 100:
        pos = nx.spring_layout(G)
        dist_params = {
            "pos": pos,
            "node_color": status,
            "with_labels": True,
        }
        nx.draw_networkx(G, **dist_params)
    else:
        # large
        pos = nx.nx_agraph.graphviz_layout(G, prog="sfdp", args="-Goverlap=false")

        dist_params = {
            "pos": pos,
            "node_color": status,
            "with_labels": False,
            "node_size": 1,
            "width": 0.3
        }
        nx.draw_networkx(G, **dist_params)
        nx.draw_networkx_labels(G, pos, font_size=1)

    if name is None:
        name = "graph"

    plt.savefig(f'../output/{name}.png', dpi=1000)
    plt.show()


def draw(constraint: ProbMinExposed):
    # Primitive Rounding
    quarantined: Set[int] = set()
    safe: Set[int] = set()

    for u, value in constraint.quarantined_solution.items():
        if value > 0.5:
            quarantined.add(u)

    for v, value in constraint.saved_solution.items():
        if value > 0.5:
            safe.add(v)
    draw_absolute(constraint.G, constraint.I, constraint.V1, constraint.V2, quarantined, safe)
