#%%
import seaborn as sns
%load_ext autoreload
%autoreload 2
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
from ctrace.min_cut import min_cut_solver, SIR
import scipy
from enum import Enum



# %%

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


def reset_node_attrs(G, keep=['pos']):
    for n in G.nodes:
        data = G.nodes[n]
        d = {k: data.get(k) for k in keep if data.get(k) is not None}
        data.clear()
        data.update(d)


def reset_edge_attrs(G, keep=['long']):
    for e in G.edges:
        data = G.edges[e]
        d = {k: data.get(k) for k in keep if data.get(k) is not None}

        data.clear()
        data.update(d)


def reset_attrs(G):
    reset_node_attrs(G)
    reset_edge_attrs(G)


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

#%%
# <=========================== Graph/SIR Setup ===========================>

code_name = "sparse"

seed = 42
budget = 100
num_infected = 30
G, pos = small_world_grid(
    width=20,
    max_norm=True,
    sparsity=0.1,
    p=1,
    local_range=1,
    num_long_range=0.2,
    r=2,
    seed=42
)

#%%
component_sizes = pd.Series([len(c) for c in sorted(
    nx.connected_components(G), key=len, reverse=True)])

sns.histplot(component_sizes, log_scale=True,)
#%%
# SIR Randomization
actual_sir = random_init(G, num_infected=num_infected, seed=seed)
# visible_sir = randomize(actual_sir)
visible_sir = actual_sir

# print(f"Actual Infected: {actual_sir.I}")
# print(f"Visible Infected: {visible_sir.I}")

# G Sampling

# Randomized Response Copy
G_agent = G.copy()
nx.set_node_attributes(G_agent, visible_sir.to_dict(), "visible_sir")

# Randomized Evaluation Copy
G_eval = G.copy()
nx.set_node_attributes(G, actual_sir.to_dict(), "actual_sir")

# Baseline Evaluation
G_base = G.copy()
nx.set_node_attributes(G, actual_sir.to_dict(), "actual_sir")


# %%

# Min Cut Fractional Solutions

frac_values_wide = pd.DataFrame()

budgets = [25 * i for i in range(15)]
for b in budgets:
    rec_vertex, rec_edge = min_cut_solver(
        G_agent,
        visible_sir,
        budget=b,
        edge_costs=None,
        vertex_costs=None,
        privacy=None,
        partial=None,
        mip=False
    )
    frac_values_wide[b] = list(rec_edge.values())
    print(f"Budget: {b} Edge Sum: {sum(rec_edge.values())} ")
    print("-------------------------------------------------")
#%%

frac_values = frac_values_wide.melt(
    var_name='budget', value_name='fractional_value').round(3)
#%%

# sns.violinplot(y="budget", x="fractional_value", data=frac_values, palette="light:g", inner="points", orient="h")

g = sns.stripplot(x="fractional_value", y="budget",
              data=frac_values, dodge=True, alpha=.25, zorder=1)
g.set_xticklabels(g.get_xticklabels(), rotation=45,
                  horizontalalignment='right')

#%%

# Density Viz (Very Laggy)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
ax1.set_title("Vertex Distribution", fontsize=8)
ax2.set_title("Edge Distribution", fontsize=8)
ax1.set(xlim=(0, 1))
ax2.set(xlim=(0, 1))
sns.histplot(rec_vertex, ax=ax1)
sns.histplot(rec_edge, ax=ax2)
print(pd.Series(rec_vertex).round(3).value_counts())
print(pd.Series(rec_edge).round(3).value_counts())

non_zero_edges = {k: 1 if v > 0 else 0 for k, v in rec_edge.items()}
#%%
print(sum(non_zero_edges.values()))
#%%


sns.set_theme(style="darkgrid")
df = sns.load_dataset("penguins")
sns.displot(
    frac_values, x="fractional_value", row="budget",
    binwidth=0.01, height=3, facet_kws=dict(margin_titles=True), aspect=5
)

#%%

# <========= Min Cut Fractional Solutions Non-Zero Visualization =========>
G_temp = G.copy()
nx.set_node_attributes(G_temp, visible_sir.to_dict(), "actual_sir")
nx.set_edge_attributes(G_temp, non_zero_edges, "cut")
draw_style(G_temp, min_cut_node_style, min_cut_edge_style, ax=None, DEBUG=False)
           
# %%
rounded = D_prime(list(rec_edge.values()))
rec_edge_rounded = {k: v for k, v in zip(rec_edge, rounded)}
ax = sns.histplot(rounded)
ax.set(xlim=(0, 1))
# %%
# Evaluation of different objective values (LP Evaluation)
cols = frac_values_wide
# cols = [25]

G_vizs = []
TOL = 1
for b in cols:
    dense = D_prime(frac_values_wide[b].to_numpy())
    rec_edge_rounded = {k: v for k, v in zip(rec_edge, dense)}
    if sum(dense) >= sum(frac_values_wide[b].to_numpy()) + TOL:
        print(
            f"Round Sum: {sum(dense)} > Frac Sum: {sum(frac_values_wide[b].to_numpy())}")

    G_viz = G.copy()

    viz_vertex, viz_edge = min_cut_solver(
        G_viz,
        visible_sir,
        budget=b,
        edge_costs=None,
        vertex_costs=None,
        privacy=None,
        partial=rec_edge_rounded,
        mip=False
    )
    nx.set_node_attributes(G_viz, visible_sir.to_dict(), "actual_sir")
    nx.set_node_attributes(G_viz, viz_vertex, "status")
    nx.set_edge_attributes(G_viz, viz_edge, "cut")

    transmit = {e: viz_vertex[e[0]] or viz_vertex[e[1]] for e in G_viz.edges}
    nx.set_edge_attributes(G_viz, transmit, "transmit")

    G_vizs.append({
        "title": f"Budget: {b} Objective value: {sum(viz_vertex.values())}",
        "G": G_viz
    })
print(G_vizs)
#%%

def draw_multiple(args, a, b, name, heavy=True):
    """
    Draws multiple draw_style() graphs in a grid (a, b).
    Args:
        G: a networks 

    """
    fig, axes = plt.subplots(a, b, figsize=(4 * a, 4 * b))

    for i, ax in tqdm(enumerate(axes.flatten()), total=a*b):
        ax.set_title(args[i].get("title"), fontsize=8)
        draw_style(args[i].get("G"), min_cut_node_style,
                   min_cut_edge_style, ax=ax, DEBUG=False)
    fig.savefig(f"{name}.png")
    fig.savefig(f"{name}.svg")

    if heavy:
        plt.close(fig)
        return ()
    return fig, ax
    

draw_multiple(G_vizs, 3, 3, f"{code_name}_multi_round")
#%%

# %%

# Evaluation of different objective values (MILP Evaluation)
cols = frac_values_wide
for b in cols:
    G_viz = G.copy()
    viz_vertex, viz_edge = min_cut_solver(
        G_viz,
        visible_sir,
        budget=b,
        edge_costs=None,
        vertex_costs=None,
        privacy=None,
        partial=None,
        mip=True
    )
    nx.set_node_attributes(G_viz, visible_sir.to_dict(), "actual_sir")
    nx.set_node_attributes(G_viz, viz_vertex, "status")
    nx.set_edge_attributes(G_viz, viz_edge, "cut")

    transmit = {e: viz_vertex[e[0]] or viz_vertex[e[1]] for e in G_viz.edges}
    nx.set_edge_attributes(G_viz, transmit, "transmit")

    G_vizs.append({
        "title": f"Budget: {b} Objective value: {sum(viz_vertex.values())}",
        "G": G_viz
    })
print(G_vizs)

# %%
draw_multiple(G_vizs, 3, 3, f"{code_name}_multi_mip")

# %%

# %%
