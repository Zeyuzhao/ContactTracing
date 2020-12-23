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


def draw_prob(G: nx.Graph, I, V1, V2, quarantined, odds, p1, transition):
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
    odds
        V2 y colormap (yellow higher chance, green lower chance)
    p1
        V1 original odds
    transition
        conditional probability of infection from V1 to V2

    Returns
    -------
    None
    """
    status = []
    N = G.number_of_nodes()
    for i in range(N):
        if i in I:
            c = "blue" if (i in quarantined) else "orange"
        elif i in V1:
            c = "green" if (i in safe) else "yellow"
        elif i in V2:
            c = "red"
        else:
            c = "grey"
        status.append(c)
        G.nodes[i]["color"] = c
    pass


draw_prob()


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

    for u, value in constraint.quaran_sol.items():
        if value > 0.5:
            quarantined.add(u)

    for v, value in constraint.safe_sol.items():
        if value > 0.5:
            safe.add(v)
    draw_absolute(constraint.G, constraint.I, constraint.V1, constraint.V2, quarantined, safe)
