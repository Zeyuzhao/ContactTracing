# %%
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

# %%

# <=========================== Graph/SIR Setup ===========================>
seed = 42
G, pos = small_world_grid(8, max_norm=False, sparsity=0.1,
                          local_range=1, num_long_range=0.2, r=2, seed=42)
SIR = random_init(G, num_infected=10, seed=seed)
budget = 20
transmission_rate = 1
compliance_rate = 0.8
structure_rate = 0

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
    edge_vars = defaultdict(dict)
    # 2d dictionary defaulting to 1

    # Costs
    if edge_costs is None:
        edge_costs = defaultdict(lambda: defaultdict(lambda: 1))
    if vertex_costs is None:
        vertex_costs = defaultdict(lambda: 1)

    # Declare 0-1 variables
    if mip:
        for n in G.nodes:
            vertex_vars[n] = solver.IntVar(0, 1, f"v_{n}")

        for e in G.edges:
            edge_vars[e[0]][e[1]] = solver.IntVar(0, 1, f"e_{e[0]}_{e[0]}")
    else:
        for n in G.nodes:
            vertex_vars[n] = solver.NumVar(0, 1, f"v_{n}")

        for e in G.edges:
            edge_vars[e[0]][e[1]] = solver.NumVar(0, 1, f"e_{e[0]}_{e[0]}")

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
                b = min(1, s - eta)  # Constraint within 1
                assert b >= 0
            solver.Add(vertex_vars[n] >= b)
            b_constraints.append(b)
        print(f"Noise (s-eta): {noise}")
        print(f"b constraints (constrained to 1): {b_constraints}")
    else:
        for n in SIR.I:
            solver.Add(vertex_vars[n] == 1)

    # Constrain edge solutions if given partial (solutions)
    if partial is not None:
        for e in G.edges:
            if e in partial:
                solver.Add(edge_vars[e[0]][e[1]] == 1)
            else:
                solver.Add(edge_vars[e[0]][e[1]] == 0)

    # Constrain transmission along edges
    for e in G.edges:
        solver.Add(edge_vars[e[0]][e[1]] >=
                   vertex_vars[e[0]] - vertex_vars[e[1]])
        solver.Add(edge_vars[e[0]][e[1]] >=
                   vertex_vars[e[1]] - vertex_vars[e[0]])

    # Constrain budget for edges
    cost: Constraint = solver.Constraint(0, budget)
    for e in G.edges:
        cost.SetCoefficient(edge_vars[e[0]][e[1]], edge_costs[e[0]][e[1]])

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
        edge_solns[e[0]][e[1]] = edge_vars[e[0]][e[1]].solution_value()

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
        edge_solns = [0] * len(G.nodes)

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
        if edge_solns[e[0]][e[1]] < 1:
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


fig, ax = plt.subplots(figsize=(5, 5))
ax.set_title("MinCut Spread", fontsize=8)
grid_cut(
    G,
    ax,
    pos,
    SIR.I,
    vertex_solns,
    edge_solns,
)

# %%


def draw_multiple_grid_cut(G, args, a, b):
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
            rounded_edge_solns[e[0]][e[1]] = 1
        else:
            rounded_edge_solns[e[0]][e[1]] = 0
    return rounded_edge_solns


def min_cut_d_prime(G, edge_solns, budget=None):
    dense = []
    for i, e in enumerate(G.edges):
        dense.append(edge_solns[e[0]][e[1]])
    result = D_prime(dense)
    if budget is not None:
        summed = sum(result)
        if summed > budget:
            raise ValueError(
                f"Sum of result violated budget ({summed} > {budget})")
    rounded_edge_solns = defaultdict(dict)
    for i, e in enumerate(G.edges):
        rounded_edge_solns[e[0]][e[1]] = result[i]
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
# Test the distribution

# <=========================== Conversion utilities ===========================>


def dict_to_set(G, edge_solns):
    pass


#%%


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
#%%
# DP testing:


s = Delta / epsilon * np.log(m * (np.exp(epsilon) - 1) / delta + 1)
eta = trunc_laplace(support=s, scale=Delta / epsilon)
# Test here?
noise.append(float(s - eta))
b = min(1, s - eta)

#%%
# # %%
# x = np.array([[True,  False,  False],
#               [{},  4,  6],
#               [3,  7,  8],
#               [1, 10, 12]])
# x[:, x[0, :] == 1]


# def array_split(*arr):
#     mat = np.array(arr)
#     x[:, x[0, :] == 1]


# # %%
# np.hsplit(x, 1)
# # %%
# data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# x, y, z = data.T

# %%

# %%

# %%

# %%
