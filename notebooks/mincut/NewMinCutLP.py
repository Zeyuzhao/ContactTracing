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

import scipy
from enum import Enum
# %%

# <=========================== Graph/SIR Setup ===========================>
seed = 42
G, pos = small_world_grid(8, max_norm=False, sparsity=0.1,
                          local_range=1, num_long_range=0.2, r=2, seed=42)
SIR = random_init(G, num_infected=10, seed=seed)
budget = 20

# Create infection state
# # infection_info = InfectionInfo(G, SIR, budget=0, transmission_rate=0)
edges = list(G.edges.data("long", default=False))
# long_edges= list(filter(lambda x: x[2], edges))
# short_edges= list(filter(lambda x: not x[2], edges))
draw_single(G, pos=pos, sir=SIR, edges=edges,
            title="Graph Struct", figsize=(5, 5))

# %%

# <=========================== LP Setup ===========================>


def min_cut_solver(
    G,
    I,
    budget=None,
    edge_costs=None,
    vertex_costs=None,
    privacy=None,
    partial=None,
    mip=False
):
    """
    Modes:
        classic_mip: G, I, budget, mip=True
        classic_frac: G, I, budget, mip=False
        dp_frac: G, I, budget, privacy, mip=False
        evaluate: G, I, partial
        throw Error otherwise
    """

    solver: Solver = pywraplp.Solver.CreateSolver("GUROBI")
    vertex_vars = {}
    edge_vars = {}
    # 2d dictionary defaulting to 1

    # Costs
    if edge_costs is None:
        edge_costs = defaultdict(lambda: 1)
    if vertex_costs is None:
        vertex_costs = defaultdict(lambda: 1)

    # Declare 0-1 variables
    if mip:
        for n in G.nodes:
            vertex_vars[n] = solver.IntVar(0, 1, f"v_{n}")

        for e in G.edges:
            edge_vars[e] = solver.IntVar(0, 1, f"e_{e[0]}_{e[0]}")
    else:
        for n in G.nodes:
            vertex_vars[n] = solver.NumVar(0, 1, f"v_{n}")

        for e in G.edges:
            edge_vars[e] = solver.NumVar(0, 1, f"e_{e[0]}_{e[0]}")

    # Constrain infected (or add (e, d) noise)
    if privacy is not None:
        Delta = 1
        (epsilon, delta) = privacy
        m = len(G.nodes)
        b_constraints = []
        noise = []
        for i, n in enumerate(G.nodes):
            # DOUBLE CHECK THIS!!!
            if n in SIR.I:
                b = 1
            else:
                s = Delta / epsilon * \
                    np.log(m * (np.exp(epsilon) - 1) / delta + 1)
                eta = trunc_laplace(support=s, scale=Delta / epsilon)
                # Test here?
                noise.append(float(s - eta))
                b = min(1, float(s - eta))  # Constraint within 1
                assert b >= 0
            # print(vertex_vars[n])
            # print(b)
            # print(vertex_vars[n] >= b)
            solver.Add(vertex_vars[n] >= b)
            b_constraints.append(b)
        # print(f"Noise (s-eta): {noise}")
        # print(f"b constraints (constrained to 1): {b_constraints}")
    else:
        for n in SIR.I:
            solver.Add(vertex_vars[n] == 1)

    # Constrain edge solutions if given partial (solutions)
    if partial is not None:
        for e in G.edges:
            if partial[e] == 1:
                solver.Add(edge_vars[e] == 1)
            else:
                solver.Add(edge_vars[e] == 0)

    # Constrain transmission along edges
    for e in G.edges:
        solver.Add(edge_vars[e] >=
                   vertex_vars[e[0]] - vertex_vars[e[1]])
        solver.Add(edge_vars[e] >=
                   vertex_vars[e[1]] - vertex_vars[e[0]])

    # Constrain budget for edges
    cost: Constraint = solver.Constraint(0, budget)
    for e in G.edges:
        cost.SetCoefficient(edge_vars[e], edge_costs[e])

    # Set objecttive for people saved
    objective = solver.Objective()
    for n in G.nodes:
        objective.SetCoefficient(vertex_vars[n], vertex_costs[n])
    objective.SetMinimization()

    status = solver.Solve()
    if status == solver.INFEASIBLE:
        raise ValueError("Infeasible solution")

    if status == solver.OPTIMAL:
        is_optimal = True
    else:
        is_optimal = False

    vertex_solns = {}
    edge_solns = defaultdict(dict)

    for i, n in enumerate(G.nodes):
        vertex_solns[n] = vertex_vars[n].solution_value()

    for i, e in enumerate(G.edges):
        edge_solns[e] = edge_vars[e].solution_value()

    return vertex_solns, edge_solns


vertex_solns, edge_solns = min_cut_solver(
    G,
    SIR.I,
    budget=20,
    edge_costs=None,
    vertex_costs=None,
    privacy=None,
    partial=None,
    mip=True
)

# vertex_solns, edge_solns = min_cut_solver(
#     G,
#     SIR.I,
#     budget=20,
#     edge_costs=None,
#     vertex_costs=None,
#     privacy=(1, 1),
#     partial=None,
#     mip=False
# )

# %%


def grid_cut(
    G: nx.Graph,
    ax,
    pos: Dict[int, Number],
    initial_infected=set(),
    vertex_solns=None,
    edge_solns=None,
    **args,
):
    if vertex_solns is None:
        vertex_solns = [0] * len(G.edges)

    if edge_solns is None:
        edge_solns = defaultdict(int)

    if pos is None:
        pos = {x: x["pos"] for x in G.nodes}

    node_size = [10] * len(G.nodes)
    node_color = ["black"] * len(G.nodes)
    border_color = ["black"] * len(G.nodes)

    edge_color = ["black"] * len(G.edges)
    for i, n in enumerate(G.nodes):
        # Handle SIR
        if vertex_solns[n] < 1:
            node_color[i] = "black"
        else:
            node_color[i] = "red"

        if n in initial_infected:
            node_size[i] = 50
        else:
            node_size[i] = 10

    for i, e in enumerate(G.edges):
        # Handle SIR
        if edge_solns[e] < 1:
            if vertex_solns[e[0]] == 1:
                edge_color[i] = "red"
            else:
                edge_color[i] = "black"
        else:
            edge_color[i] = "grey"

    # Draw edges that are from I, V1, and V2
    nodes = nx.draw_networkx_nodes(
        G,
        pos=pos,
        node_color=node_color,
        node_size=node_size,
        edgecolors=border_color,
        ax=ax
    )

    # REDO!!!
    short_edges, short_props = zip(
        *list(filter(lambda x: not G[x[0][0]][x[0][1]].get("long"), zip(G.edges, edge_color))))
    longs = list(zip(
        *list(filter(lambda x: G[x[0][0]][x[0][1]].get("long"), zip(G.edges, edge_color)))))

    if len(longs) == 2:  # Handle case with no edges (HACK)
        long_edges, long_props = longs
    else:
        long_edges, long_props = ([], [])
    draw_networkx_edges(
        G,
        pos=pos,
        edgelist=short_edges,
        edge_color=short_props,
        node_size=node_size,
        ax=ax,
        arrowstyle='-',
    )

    draw_networkx_edges(
        G,
        pos=pos,
        edgelist=long_edges,
        edge_color=long_props,
        node_size=node_size,
        ax=ax,
        connectionstyle="arc3,rad=0.2",
        arrowstyle='-',
    )

# %%


def draw_multiple_grid_cut(G, args, a, b):
    """
    Draws multiple grid_cut() graphs in a grid (a, b).
    Args:
        G: a networks 

    """
    fig, ax = plt.subplots(a, b, figsize=(4 * a, 4 * b))

    for i, ((x, y), config) in enumerate(zip(itertools.product(range(a), range(b)), args)):
        ax[x, y].set_title(config.get("title"), fontsize=8)
        grid_cut(G, ax[x, y], **config)
    return fig, ax


# Draw different budgets
height = 2
width = 2
N = height * width
num_edges = len(G.edges)
budgets = np.linspace(0, num_edges, N).astype(int)
configs = []

for b in np.nditer(budgets):
    # Different MIPs
    vertex_solns, edge_solns = min_cut_solver(
        G,
        SIR.I,
        budget=int(b),
        edge_costs=None,
        vertex_costs=None,
        privacy=None,
        partial=None,
        mip=True
    )
    configs.append({
        "pos": pos,
        "initial_infected": SIR.I,
        "vertex_solns": vertex_solns,
        "edge_solns": edge_solns,
        "title": f"MinCut MIP Budget={b}/{num_edges}"
    })
fig, ax = draw_multiple_grid_cut(G, configs, height, width)

fig.savefig("multi.png")
fig.savefig("multi.svg")

# %%
# <=========================== Rounders ===========================>


def min_cut_l_round(G, vertex_solns, l):
    rounded_edge_solns = defaultdict(dict)
    for e in G.edges:
        if (vertex_solns[e[0]] - l) * (vertex_solns[e[1]] - l) <= 0:
            rounded_edge_solns[e] = 1
        else:
            rounded_edge_solns[e] = 0
    return rounded_edge_solns


def min_cut_d_prime(G, edge_solns, budget=None):
    dense = []
    for i, e in enumerate(G.edges):
        dense.append(edge_solns[e])
    result = D_prime(dense)
    if budget is not None:
        summed = sum(result)
        if summed > budget:
            raise ValueError(
                f"Sum of result violated budget ({summed} > {budget})")
    rounded_edge_solns = defaultdict(dict)
    for i, e in enumerate(G.edges):
        rounded_edge_solns[e] = result[i]
    return rounded_edge_solns
# <=========================== Sampling utilities ===========================>


# b = support / scale
def trunc_laplace(support, scale, gen=np.random, size=1):
    """
    Generate laplacian with support [-support, +support], and scale=scale
    """
    return ((-1) ** np.random.randint(2, size=size)) * scipy.stats.truncexpon.rvs(b=support / scale, scale=scale, size=size)


fig, ax = plt.subplots(1, 1)

r = trunc_laplace(support=100, scale=1, size=1000)
ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()

# %%


vertex_solns, edge_solns = min_cut_solver(
    G,
    SIR.I,
    budget=int(b),
    edge_costs=None,
    vertex_costs=None,
    privacy=(1, 1),
    partial=None,
    mip=False
)
rounded_edge_solns = min_cut_l_round(G, vertex_solns=vertex_solns, l=0.5)

print(rounded_edge_solns)

display_vertex_solns, display_edge_solns = min_cut_solver(
    G,
    SIR.I,
    budget=10000000,
    edge_costs=None,
    vertex_costs=None,
    privacy=None,
    partial=rounded_edge_solns,
    mip=True
)

assert display_edge_solns == rounded_edge_solns

fig, ax = plt.subplots(1, 1)
grid_cut(
    G,
    ax,
    pos,
    SIR.I,
    display_vertex_solns,
    display_edge_solns,
)
# %%
# DP testing:

# %%


def compute_stats(m, epsilon, delta=None, Delta=2):
    if delta is None:
        delta = 1/m
    s = Delta / epsilon * np.log(m * (np.exp(epsilon) - 1) / delta + 1)
    return s, (Delta / epsilon)


def perturb(m, epsilon, delta=None, Delta=2):
    if delta is None:
        delta = 1/m
    s = Delta / epsilon * np.log(m * (np.exp(epsilon) - 1) / delta + 1)
    eta = trunc_laplace(support=s, scale=Delta / epsilon)
    # Test here?
    # b = min(1, s - eta)
    return s - eta


def trunc_laplace_pdf(x, support, loc, scale):
    return np.exp(-abs((x-loc)/scale)) / (2 * scale * (1 - np.exp(-support / scale)))


m_slider = widgets.IntSlider(100, 1, 1000, 1)
eps_slider = widgets.IntSlider(1, 0.01, 10, 0.01)
m = m_slider.value
epsilon = eps_slider.value
num_samples = 10000

noise_mean, noise_scale = compute_stats(m, epsilon)
print(f"Mean: {noise_mean}")
print(f"Scale: {noise_scale}")

samples = np.array([perturb(m, epsilon) for _ in range(num_samples)])
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

x = np.linspace(0, 2 * noise_mean, num_samples)
ax.set_title(
    f"Truncated Laplacian Noise (m={m}, epsilon={epsilon}, samples={num_samples})")
ax.plot(x, trunc_laplace_pdf(x, noise_mean, noise_mean, noise_scale),
        'r-', lw=1, alpha=0.6, label=f'trunc_laplace(support={noise_mean:.1f}, scale={noise_scale:.1f})')

ax.hist(samples, density=True, histtype='stepfilled', alpha=0.2, bins=50)
ax.legend(loc='best', frameon=False)
plt.show()


# %%


def max_degree_edges(G, budget):
    edge_max_degree = {(u, v): max(G.degree(u), G.degree(v))
                       for (u, v) in G.edges}
    edges_by_degree = sorted(
        edge_max_degree, key=edge_max_degree.get, reverse=True)
    return edges_by_degree[:budget]


def degree_solver(G, SIR, budget):
    edges = max_degree_edges(G, budget)
    return min_cut_solver(
        G,
        SIR.I,
        budget=budget,
        edge_costs=None,
        vertex_costs=None,
        privacy=None,
        partial=edges,
        mip=False
    )


def random_solver(G, SIR, budget):
    edges = np.random.choice(G.edges, size=budget, replace=False)
    return min_cut_solver(
        G,
        SIR.I,
        budget=budget,
        edge_costs=None,
        vertex_costs=None,
        privacy=None,
        partial=edges,
        mip=False
    )


# Create a graphical user interface???
# Define configuration files???
def visualizer(G, SIR, budget, solver):
    # Draw different budgets
    height = 2
    width = 2
    N = height * width
    num_edges = len(G.edges)
    budgets = np.linspace(0, num_edges, N).astype(int)
    configs = []

    for b in np.nditer(budgets):
        # Different MIPs
        vertex_solns, edge_solns = degree_solver(
            G,
            SIR.I,
            budget=int(b),
        )
        configs.append({
            "pos": pos,
            "initial_infected": SIR.I,
            "vertex_solns": vertex_solns,
            "edge_solns": edge_solns,
            "title": f"MinCut MIP Budget={b}/{num_edges}"
        })
    fig, ax = draw_multiple_grid_cut(G, configs, height, width)

    fig.savefig("multi.png")
    fig.savefig("multi.svg")


# Evaluate different budgets by degree solver
# %%
G_old = G.copy()
# %%


# TODO: Implement YAML Loader


# Styling dictionaries
# Later attributes take precedence
# default is skipped as an attribute
node_style = {
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

edge_style = {
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
        fig, ax = plt.subplots(figsize=(4, 4))

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
    NONE = -1  # No functionwide paramter can a -1!
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
        )


# Edges need to split by connection style
fig, ax = plt.subplots(figsize=(4, 4))
ax.set_title("Test Graph", fontsize=8)

G = G_old.copy()
G.nodes[0]["patient0"] = True
# G.nodes[1]["patient0"] = True
G.nodes[2]["patient0"] = True
# G.edges[(0,2)]["cut"] = True
G.edges[(0, 2)]["transmit"] = True
G.edges[(0, 3)]["cut"] = True
draw_style(G, node_style, edge_style, DEBUG=False)


# %%

# %%
# fig, ax = plt.subplots(figsize=(4, 4))
# ax.set_title("Test Graph", fontsize=8)


G, pos = small_world_grid(15, max_norm=False, sparsity=0.1,
                          local_range=1, num_long_range=0.2, r=2, seed=42)
SIR = random_init(G, num_infected=10, seed=seed)

vertex_solns, edge_solns = min_cut_solver(
    G,
    SIR.I,
    budget=20,
    edge_costs=None,
    vertex_costs=None,
    privacy=None,
    partial=None,
    mip=True
)

# Attribute Semantic Painter
transmit = {e: vertex_solns[e[0]] or vertex_solns[e[1]] for e in G.edges}
nx.set_node_attributes(G, {n: n in SIR.I for n in G.nodes}, "patient0")
nx.set_node_attributes(G, vertex_solns, "status")
nx.set_edge_attributes(G, edge_solns, "cut")
nx.set_edge_attributes(G, transmit, "transmit")

draw_style(G, node_style, edge_style, DEBUG=True)
# %%

with cProfile.Profile() as pr:
    draw_style(G, node_style, edge_style, DEBUG=False)

pr.dump_stats('draw.prof')


# %%

# %%
