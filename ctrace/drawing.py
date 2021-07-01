# from ctrace.min_cut import PartitionSIR
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import random
import seaborn as sns
import math
import networkx as nx
import itertools

from ctrace.simulation import *
from ctrace.recommender import *
from ctrace.problem import *
from ctrace.utils import *

from numpy.random import default_rng
from typing import Dict, TypeVar
from numbers import Number
from tqdm import tqdm

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


def prob_round(num: float, rg=None):
    f, n = math.modf(num)
    if rg is None:
        rg = np.random
    return int(n + (rg.random() < f))


def random_init(G, num_infected=5, seed=42):
    random.seed(seed)
    nodes = set(G.nodes)
    I = set(random.sample(nodes, num_infected))
    R = set()
    S = nodes - I - R
    return PartitionSIR.from_sets(list(S), list(I), list(R))


def grid_2d(width, seed=42, diagonals=True, sparsity=0.2, global_rate=0):
    random.seed(seed)
    G = nx.grid_2d_graph(width, width)

    if diagonals:
        G.add_edges_from([
            ((x, y), (x+1, y+1))
            for x in range(width-1)
            for y in range(width-1)
        ] + [
            ((x+1, y), (x, y+1))
            for x in range(width-1)
            for y in range(width-1)
        ])
    G.remove_nodes_from(uniform_sample(G.nodes(), sparsity))
    mapper = {n: i for i, n in enumerate(G.nodes())}
    pos = {i: (y, -x) for i, (x, y) in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapper)
    # Add edges using the small world method

    G.add_edges_from(uniform_sample(
        list(itertools.product(G.nodes, G.nodes)), global_rate))
    return G, pos


def small_world_grid(width: int, max_norm=False, sparsity=0, p=1, local_range=1, num_long_range=1, r=2, seed=42):
    """
    Generates an undirected small_world_grid graph.
    Has two types of connection: local and long-ranged.
    Each node is connected to all of its neighbors within local_range.
    Each node establishes num_long_range long range connnections (could overlap with short-ranged), 
    choosing a length d connection with probability proportional to 1/d^(-r)


    Parameters
    ----------
    width : int
        The width of the square grid
    max_norm: bool, optional
        Whether to use the max_norm for computing distances. Defaults to L1 norm
    sparsity: float
        The independent probability of removing each node
    num_long_range

    Returns
    -------
    Undirected graph G, with long=true attributes for long-ranged edges.

    Raises
    ------
    ValueError

    References
    ----------
    .. [1] J. Kleinberg. The small-world phenomenon: An algorithmic
       perspective. Proc. 32nd ACM Symposium on Theory of Computing, 2000.
    """

    rg = np.random.default_rng(seed)
    if r < 0:
        raise ValueError("r must be >= 1")

    G = nx.Graph()
    nodes = list(itertools.product(range(width), repeat=2))

    def dist(p1, p2):
        diffs = (abs(b - a) for a, b in zip(p1, p2))
        if max_norm:
            d = max(diffs)
        else:
            d = sum(diffs)
        return d

    for p1 in tqdm(nodes, total=len(nodes)):
        probs = [0]
        for p2 in nodes:
            # O(n^2) - we can speed it up
            if p1 == p2:
                continue
            d = dist(p1, p2)
            if d <= local_range:
                G.add_edge(p1, p2, long=False)
            probs.append(d ** -r)

        # Normalization
        probs = np.array(probs) / np.sum(probs)
        rounded_num = prob_round(num_long_range, rg=rg)

        targets = rg.choice(len(probs), size=rounded_num, p=probs)
        for t in targets:
            p2 = nodes[t]  # Convert int -> real id
            G.add_edge(p1, p2, long=dist(p1, p2) > local_range)

    G.remove_nodes_from(uniform_sample(G.nodes(), sparsity, rg))
    G.remove_edges_from(uniform_sample(G.edges(), 1-p, rg))
    # Remap nodes to 0 - n^2-1
    mapper = {n: i for i, n in enumerate(G.nodes())}
    pos = {i: (y, -x) for i, (x, y) in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapper)
    nx.set_node_attributes(G, pos, 'pos')
    return G, pos

# Styling dictionaries
# Later attributes take precedence
# default is skipped as an attribute
min_cut_node_style = {
    # Default styling
    "default": {
        "node_size": 10,
        "node_color": "black",
        "edgecolors": "black",
    },
    # Attribute styling
    "patient0": {
        # Is patient 0?
        False: {},
        True: {"node_size": 50},
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


def draw_style(G, node_style, edge_style, ax=None, DEBUG=False):
    node_style = node_style.copy()
    edge_style = edge_style.copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    default_node_style = node_style.pop("default", {})
    default_edge_style = edge_style.pop("default", {})

    def apply_style(collection, style, default):
        """
        collection: dictionary like
        style: { attr: {value: style_dict}}
        default: style_dict
        """
        processed = {}
        for item in collection:
            # Iteratively merge in styles
            styles = default.copy()
            for attr in style:
                val = collection[item].get(attr)
                new_styles = style[attr].get(val, {})
                styles = {**styles, **new_styles}
            processed[item] = styles
        return processed

    # Handle any missing attributes
    # TODO: Allow for functionwide node attributes
    df_nodes = pd.DataFrame.from_dict(
        apply_style(G.nodes,
                    node_style, default_node_style),
        orient="index"
    ).replace({np.nan: None})
    styled_nodes = df_nodes.to_dict(orient="list")

    # Index: Edges
    df_edges = pd.DataFrame.from_dict(
        apply_style(G.edges,
                    edge_style, default_edge_style),
        orient="index",
    ).replace({np.nan: None})
    # styled_edges = df_edges.to_dict(orient="list")

    if DEBUG:
        print(df_nodes)
        print(df_edges)

    # Core non-style attributes
    pos = [G.nodes[node]["pos"] for node in G.nodes]

    # Drawing
    nodes = nx.draw_networkx_nodes(
        G,
        pos=pos,
        **styled_nodes,
        ax=ax
    )

    # Creating a fake NONE
    # Handling function-wide parameters
    functionwide_params = ["connectionstyle", "arrowstyle"]

    # Pandas can't handle None equality!!!
    NONE = -1  # No functionwide parameter can be a -1!
    for p in functionwide_params:
        if p not in df_edges:
            df_edges[p] = NONE
        df_edges[p] = df_edges[p].fillna(NONE)

    for name, group in df_edges.groupby(functionwide_params):
        styled_edges = group.drop(
            functionwide_params, axis=1).to_dict(orient="list")

        # Convert NONE back to None
        functionwide_styles = {
            k: None if v == NONE else v
            for k, v in zip(functionwide_params, name)
        }

        if DEBUG:
            print("<======= Group ========>")
            print(f"Functionwide: {functionwide_styles}")
            # print(f"Styled: {styled_edges}")
            print("<======= End Group ========>")
            print()
        draw_networkx_edges(
            G,
            edgelist=list(group.index),
            pos=pos,
            **styled_edges,
            **functionwide_styles,
            node_size=styled_nodes["node_size"],
            ax=ax,
            arrows=True,
        )


def fast_draw_style(G, node_style, edge_style, ax=None, DEBUG=False):
    node_style = node_style.copy()
    edge_style = edge_style.copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    default_node_style = node_style.pop("default", {})
    default_edge_style = edge_style.pop("default", {})

    def apply_style(collection, style, default):
        """
        collection: dictionary like
        style: { attr: {value: style_dict}}
        default: style_dict
        """
        processed = {}
        for item in collection:
            # Iteratively merge in styles
            styles = default.copy()
            for attr in style:
                val = collection[item].get(attr)
                new_styles = style[attr].get(val, {})
                styles = {**styles, **new_styles}
            processed[item] = styles
        return processed

    # Handle any missing attributes
    # TODO: Allow for functionwide node attributes
    df_nodes = pd.DataFrame.from_dict(
        apply_style(G.nodes,
                    node_style, default_node_style),
        orient="index"
    ).replace({np.nan: None})
    styled_nodes = df_nodes.to_dict(orient="list")

    # Index: Edges
    df_edges = pd.DataFrame.from_dict(
        apply_style(G.edges,
                    edge_style, default_edge_style),
        orient="index",
    ).replace({np.nan: None})
    # styled_edges = df_edges.to_dict(orient="list")

    if DEBUG:
        print(df_nodes)
        print(df_edges)

    # Core non-style attributes
    pos = [G.nodes[node]["pos"] for node in G.nodes]

    # Drawing
    nodes = nx.draw_networkx_nodes(
        G,
        pos=pos,
        **styled_nodes,
        ax=ax
    )

    # Creating a fake NONE
    # Handling function-wide parameters
    functionwide_params = ["connectionstyle", "arrowstyle"]

    # Pandas can't handle None equality!!!
    NONE = -1  # No functionwide parameter can be a -1!
    for p in functionwide_params:
        if p not in df_edges:
            df_edges[p] = NONE
        df_edges[p] = df_edges[p].fillna(NONE)

    for name, group in df_edges.groupby(functionwide_params):
        styled_edges = group.drop(
            functionwide_params, axis=1).to_dict(orient="list")

        # Convert NONE back to None
        functionwide_styles = {
            k: None if v == NONE else v
            for k, v in zip(functionwide_params, name)
        }

        if DEBUG:
            print("<======= Group ========>")
            print(f"Functionwide: {functionwide_styles}")
            # print(f"Styled: {styled_edges}")
            print("<======= End Group ========>")
            print()

        if all(map(lambda x: x[1] is None, functionwide_styles.items())):
            nx.draw_networkx_edges(
                G,
                edgelist=list(group.index),
                pos=pos,
                **styled_edges,
                node_size=styled_nodes["node_size"],
                ax=ax,
                arrows=False,
            )
        else:
            draw_networkx_edges(
                G,
                edgelist=list(group.index),
                pos=pos,
                **styled_edges,
                **functionwide_styles,
                node_size=styled_nodes["node_size"],
                ax=ax,
                arrows=True,
            )

# %%
# I1 = E
# I2 = I

# <==================== Style Registry ====================>

LARGE_NODE_RADIUS = 100
SMALL_NODE_RADIUS = 50
seir_node_style = {
    # Default styling
    "default": {
        "node_size": SMALL_NODE_RADIUS,
        "node_color": "black",
        "edgecolors": "black",
        "linewidths": 0.5,
    },
    "seir": {
        SEIR.E: {"node_size": LARGE_NODE_RADIUS, "node_color": "red"},
        SEIR.I: {"node_size": LARGE_NODE_RADIUS, "node_color": "darkred"},
        SEIR.R: {"node_size": LARGE_NODE_RADIUS, "node_color": "skyblue"},
    },
    "isolate": {
        True: {"edgecolors": "aqua", "linewidths": 1.5}
    },
    #     "V1": {
    #         True: {"node_size": 30, "node_color": "orange"},
    #     }
}

seir_edge_style = {
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

# <==================== Graph Registry ====================>
base_grid = {
    'max_norm': False,
    'sparsity': 0,
    'p': 1,
    'local_range': 1,
    'num_long_range': 0,
    'r': 2,
}

feature_grid = {
    'max_norm': True,
    'sparsity': 0.1,
    'p': 1,
    'local_range': 1,
    'num_long_range': 1,
    'r': 2,
}

complex_grid = {
    'max_norm': True,
    'sparsity': 0.1,
    'p': 0.3,
    'local_range': 1,
    'num_long_range': 2,
    'r': 2,
}

# %%

def grid_world_init(initial_infection=(0.05, 0.1), width=10, small_world=base_grid, labels=False, seed=42):
    rng = np.random.default_rng(seed)
    G, pos = small_world_grid(**small_world, width=width, seed=seed)
    G.pos = pos
    # Simulate compliance rates and age groups
    for n in G.nodes:
        G.nodes[n].update({
            'compliance_rate_og': 1,
            'age_group': (n % 2) + 1 if labels else 0,
        })
    for e in G.edges:
        G.edges[e].update({
            'duration': 10000,
        })

    i1_frac, i2_frac = initial_infection
    seir = PartitionSEIR.from_dist(
        len(G), [1 - i1_frac - i2_frac, i1_frac, i2_frac, 0], rng=rng)

    state = InfectionState(
        G=G,
        SIR=(list(seir.S), list(seir.E), list(seir.I), list(seir.R)),
        budget=10,
        policy="A",
        transmission_rate=0.20,
        transmission_known=True,
        compliance_rate=1,
        compliance_known=True,
        discovery_rate=1,
        snitch_rate=1
    )
    nx.set_node_attributes(G, seir.to_dict(), "seir")
    return state

# state.set_budget_labels()
# print(state.V1)
# print('Evens', [v for v in state.V1 if G.nodes[v]["age_group"] == 1])
# print('Odds', [v for v in state.V1 if G.nodes[v]["age_group"] == 2])
# print(state.budget_labels)

# %%


def pct_format(name, v1, v2):
    pct = f"{(v1 / v2) * 100:.1f}%" if v2 != 0 else "-%"
    return f"{name}: {v1}/{v2} ({pct})"


def run(state, debug=False):
    raw_history = []
    action_history = []
    while len(state.SIR.I2) + len(state.SIR.I1) != 0:
        action = DegGreedy_fair(state)
        action_history.append(action)
        raw_history.append(copy.deepcopy(state))

        if debug:
            print(f"Budgets: {state.budget_labels}")
            print(
                f"Size (I1, I2, I): {(len(state.SIR.I1), len(state.SIR.I2), len(state.SIR.I1) + len(state.SIR.I2))}")
            print(f"V1 Size: {len(state.V1)}")
            print(f"V2 Size: {len(state.V2)}")
            print(pct_format("Budget Utilization", len(action), state.budget))
            print(pct_format("V1 Quarantined (out of V1)", len(action), len(state.V1)))
            print(pct_format("I1 Quarantined (out of Q)", len(
                set(state.SIR.I1) & set(action)), len(action)))
            print(f"I1: {state.SIR.I1}")
            print(f"I2: {state.SIR.I2}")
            print(f"V1: {state.V1}")
            print(f"V2: {state.V2}")
            print(f"Q: {action}")
            print("-----------------------------")

        # Mutable state
        state.step(action)

    # Ending state
    raw_history.append(state)
    action_history.append(set())
    return raw_history, action_history

# Build plot
def draw_frame(num, raw_history=None, action_history=None, ax=None, labels=False):
    ax.clear()
    state = raw_history[num]

    G_draw = state.G.copy()

    seir = PartitionSEIR.from_sets(state.SIR)
    nx.set_node_attributes(G_draw, seir.to_dict(), "seir")

    if num > 0:
        action = {n: True for n in action_history[num - 1]}
        nx.set_node_attributes(G_draw, action, "isolate")

    transmit = {e: (seir[e[0]] == SEIR.E or seir[e[0]] == SEIR.I) or (seir[e[1]] == SEIR.E or seir[e[1]] == SEIR.I)
                for e in G_draw.edges}
#     nx.set_edge_attributes(G_draw, transmit, "transmit")

    nx.set_node_attributes(G_draw, {n: True for n in state.V1}, "V1")
    nx.set_node_attributes(G_draw, {n: True for n in state.V2}, "V2")
#     nx.set_node_attributes(rand_G, vertex_soln, "status")
#     nx.set_edge_attributes(rand_G, edge_soln, "cut")

#     transmit = {e: vertex_soln[e[0]] or vertex_soln[e[1]]
#                 for e in rand_G.edges}
#     nx.set_edge_attributes(rand_G, transmit, "transmit")

    fast_draw_style(G_draw, seir_node_style,
                    seir_edge_style, ax=ax, DEBUG=False)

    if labels:
        nx.draw_networkx_labels(G_draw, G_draw.pos, font_size=10, ax=ax,
                                verticalalignment='bottom', horizontalalignment="right")
    # Scale plot ax
    ax.set_title(
        f"Step {num}: ({len(seir.S)}, {len(seir.E)}, {len(seir.I)}, {len(seir.R)})", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


# %%
from functools import partial

# To video
def draw_anim(raw_history, action_history, out_file=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ani = matplotlib.animation.FuncAnimation(
        fig, partial(draw_frame, raw_history=raw_history, action_history=action_history, ax=ax), frames=len(raw_history),
        interval=500, repeat=True, repeat_delay=1,
    )
    html_out = ani.to_jshtml()
    plt.close(fig)
    if out_file is not None:
        with open(f'figs/{out_file}', "w") as f:
            f.write(html_out)
    return html_out


def draw_unrolled(raw_history, action_history, fig_title='SIR Sim', out_file=None):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle(fig_title, fontsize=16)
    for i, ax in enumerate(axes.flatten()):
        if i < len(raw_history):
            draw_frame(i, raw_history=raw_history,
                   action_history=action_history, ax=ax)
        else:
            ax.set_axis_off()
    if out_file:
        fig.savefig(f'figs/{out_file}', dpi=fig.dpi)


def draw_networkx_edges(
    G,
    pos,
    edgelist=None,
    width=1.0,
    edge_color="k",
    style="solid",
    alpha=None,
    arrowstyle=None,
    arrowsize=10,
    edge_cmap=None,
    edge_vmin=None,
    edge_vmax=None,
    ax=None,
    arrows=True,
    label=None,
    node_size=300,
    nodelist=None,
    node_shape="o",
    connectionstyle=None,
    min_source_margin=0,
    min_target_margin=0,
):
    """Draw the edges of the graph G.

    This draws only the edges of the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    edgelist : collection of edge tuples
       Draw only specified edges(default=G.edges())

    width : float, or array of floats
       Line width of edges (default=1.0)

    edge_color : color or array of colors (default='k')
       Edge color. Can be a single color or a sequence of colors with the same
       length as edgelist. Color can be string, or rgb (or rgba) tuple of
       floats from 0-1. If numeric values are specified they will be
       mapped to colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    style : string
       Edge line style (default='solid') (solid|dashed|dotted,dashdot)

    alpha : float
       The edge transparency (default=None)

    edge_ cmap : Matplotlib colormap
       Colormap for mapping intensities of edges (default=None)

    edge_vmin,edge_vmax : floats
       Minimum and maximum for edge colormap scaling (default=None)

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    arrows : bool, optional (default=True)
       For directed graphs, if True draw arrowheads.
       Note: Arrows will be the same color as edges.

    arrowstyle : str, optional (default='-|>')
       For directed graphs, choose the style of the arrow heads.
       See :py:class: `matplotlib.patches.ArrowStyle` for more
       options.

    arrowsize : int, optional (default=10)
       For directed graphs, choose the size of the arrow head head's length and
       width. See :py:class: `matplotlib.patches.FancyArrowPatch` for attribute
       `mutation_scale` for more info.

    connectionstyle : str, optional (default=None)
       Pass the connectionstyle parameter to create curved arc of rounding
       radius rad. For example, connectionstyle='arc3,rad=0.2'.
       See :py:class: `matplotlib.patches.ConnectionStyle` and
       :py:class: `matplotlib.patches.FancyArrowPatch` for more info.

    label : [None| string]
       Label for legend

    min_source_margin : int, optional (default=0)
       The minimum margin (gap) at the begining of the edge at the source.

    min_target_margin : int, optional (default=0)
       The minimum margin (gap) at the end of the edge at the target.

    Returns
    -------
    matplotlib.collection.LineCollection
        `LineCollection` of the edges

    list of matplotlib.patches.FancyArrowPatch
        `FancyArrowPatch` instances of the directed edges

    Depending whether the drawing includes arrows or not.

    Notes
    -----
    For directed graphs, arrows are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False. Be sure to include `node_size` as a
    keyword argument; arrows are drawn considering the size of nodes.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edges = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))

    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 3)])
    >>> arcs = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    >>> alphas = [0.3, 0.4, 0.5]
    >>> for i, arc in enumerate(arcs):  # change alpha values of arcs
    ...     arc.set_alpha(alphas[i])

    Also see the NetworkX drawing examples at
    https://networkx.github.io/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_labels()
    draw_networkx_edge_labels()
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import colorConverter, Colormap, Normalize
        from matplotlib.collections import LineCollection
        from matplotlib.patches import FancyArrowPatch
        import numpy as np
    except ImportError as e:
        raise ImportError("Matplotlib required for draw()") from e
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = list(G.edges())

    if len(edgelist) == 0:  # no edges!
        if not G.is_directed() or not arrows:
            return LineCollection(None)
        else:
            return []

    if nodelist is None:
        nodelist = list(G.nodes())

    # FancyArrowPatch handles color=None different from LineCollection
    if edge_color is None:
        edge_color = "k"

    if arrowstyle is None:
        if G.is_directed():
            arrowstyle = '-|>'
        else:
            arrowstyle = '-'

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    # Check if edge_color is an array of floats and map to edge_cmap.
    # This is the only case handled differently from matplotlib
    if (
        np.iterable(edge_color)
        and (len(edge_color) == len(edge_pos))
        and np.alltrue([isinstance(c, Number) for c in edge_color])
    ):
        if edge_cmap is not None:
            assert isinstance(edge_cmap, Colormap)
        else:
            edge_cmap = plt.get_cmap()
        if edge_vmin is None:
            edge_vmin = min(edge_color)
        if edge_vmax is None:
            edge_vmax = max(edge_color)
        color_normal = Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

    if not arrows:
        edge_collection = LineCollection(
            edge_pos,
            colors=edge_color,
            linewidths=width,
            antialiaseds=(1,),
            linestyle=style,
            transOffset=ax.transData,
            alpha=alpha,
        )

        edge_collection.set_cmap(edge_cmap)
        edge_collection.set_clim(edge_vmin, edge_vmax)

        edge_collection.set_zorder(1)  # edges go behind nodes
        edge_collection.set_label(label)
        ax.add_collection(edge_collection)

        return edge_collection

    arrow_collection = None

    if arrows:
        # Note: Waiting for someone to implement arrow to intersection with
        # marker.  Meanwhile, this works well for polygons with more than 4
        # sides and circle.

        def to_marker_edge(marker_size, marker):
            if marker in "s^>v<d":  # `large` markers need extra space
                return np.sqrt(2 * marker_size) / 2
            else:
                return np.sqrt(marker_size) / 2

        # Draw arrows with `matplotlib.patches.FancyarrowPatch`
        arrow_collection = []
        mutation_scale = arrowsize  # scale factor of arrow head

        # FancyArrowPatch doesn't handle color strings
        arrow_colors = colorConverter.to_rgba_array(edge_color, alpha)
        for i, (src, dst) in enumerate(edge_pos):
            x1, y1 = src
            x2, y2 = dst
            shrink_source = 0  # space from source to tail
            shrink_target = 0  # space from  head to target
            if np.iterable(node_size):  # many node sizes
                source, target = edgelist[i][:2]
                source_node_size = node_size[nodelist.index(source)]
                target_node_size = node_size[nodelist.index(target)]
                shrink_source = to_marker_edge(source_node_size, node_shape)
                shrink_target = to_marker_edge(target_node_size, node_shape)
            else:
                shrink_source = shrink_target = to_marker_edge(
                    node_size, node_shape)

            if shrink_source < min_source_margin:
                shrink_source = min_source_margin

            if shrink_target < min_target_margin:
                shrink_target = min_target_margin

            if len(arrow_colors) == len(edge_pos):
                arrow_color = arrow_colors[i]
            elif len(arrow_colors) == 1:
                arrow_color = arrow_colors[0]
            else:  # Cycle through colors
                arrow_color = arrow_colors[i % len(arrow_colors)]

            if np.iterable(width):
                if len(width) == len(edge_pos):
                    line_width = width[i]
                else:
                    line_width = width[i % len(width)]
            else:
                line_width = width

            arrow = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle=arrowstyle,
                shrinkA=shrink_source,
                shrinkB=shrink_target,
                mutation_scale=mutation_scale,
                color=arrow_color,
                linewidth=line_width,
                connectionstyle=connectionstyle,
                linestyle=style,
                zorder=1,
            )  # arrows go behind nodes

            # There seems to be a bug in matplotlib to make collections of
            # FancyArrowPatch instances. Until fixed, the patches are added
            # individually to the axes instance.
            arrow_collection.append(arrow)
            ax.add_patch(arrow)

    # update view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))

    w = maxx - minx
    h = maxy - miny
    padx, pady = 0.05 * w, 0.05 * h
    corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return arrow_collection
