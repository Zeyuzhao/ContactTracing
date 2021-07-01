# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import copy
from ctrace.drawing import *
from ctrace.utils import *
from ctrace.recommender import *
from ctrace.simulation import *
from IPython.display import HTML
import matplotlib.animation
import time
import seaborn as sns
import math
import random
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import networkx as nx
from IPython import get_ipython

# %%
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# New Imports
#from ctrace.contact_tracing import *
#from ctrace.constraint import *
#from ctrace.solve import *
#from ctrace.simulation import *
#from ctrace.restricted import *


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
        budget=8,
        policy="A",
        transmission_rate=0.30,
        transmission_known=True,
        compliance_rate=1,
        compliance_known=True,
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

# if trial_id == 0:  # Only record 1 entry
#     path = path / "data" / str(id)
#     path.mkdir(parents=True, exist_ok=True)
#     with open(path / "sir_history.json", "wb") as f:
#         json.dump(raw_history, f)


# %%
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
def draw_anim(raw_history, action_history, fig_title=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ani = matplotlib.animation.FuncAnimation(
        fig, partial(draw_frame, raw_history=raw_history, action_history=action_history, ax=ax), frames=len(raw_history),
        interval=500, repeat=True, repeat_delay=1,
    )
    html_out = ani.to_jshtml()
    plt.close(fig)
    if fig_title is not None:
        with open(f'viz/{fig_title}.png', "w") as f:
            f.write(html_out)
    HTML(html_out)
    return html_out


def draw_unrolled(raw_history, action_history, fig_title='SIR Model Visualization'):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle(fig_title, fontsize=16)
    for i, ax in enumerate(axes.flatten()):
        if i < len(raw_history):
            draw_frame(i, raw_history=raw_history,
                   action_history=action_history, ax=ax)
        else:
            ax.set_axis_off()
    fig.savefig(f'viz/{fig_title}.png', dpi=fig.dpi)

# %%


state = grid_world_init(
    initial_infection=(0, 0.25),
    small_world=base_grid,
    width=10,
    seed=42,
)

fig, ax = plt.subplots(figsize=(6, 6))

fast_draw_style(state.G, seir_node_style,
                seir_edge_style, ax=ax, DEBUG=False)
plt.show()

#%%


state.labels
#%%

from ctrace.utils import *
from ctrace.recommender import SegDegree, Milp, DepRound_fair

#%%
seg_action = SegDegree(state, k1=0.2, k2=0.8, carry=True,rng=rng, DEBUG=False)


#%%

action = DegGreedy_fair(state)
milp_action = Milp(state)
round_action = DepRound_fair(state)


# Test calculate methods
ce_val = calculateExpected(state, action)
cm_val = calculateMILP(state, action)

# Test for consistancy
greedy_val = calculateMILP(state, action)
milp_val = calculateMILP(state, milp_action)
round_val = calculateMILP(state, round_action)

# print(f'Direct Calculation Expected: {ce_val}')
# print(f'Direct Calculation MILP: {cm_val}')
# print(f'LP Evaluation: {lp_val}')
# print(f'LP Evaluation2: {lp_val2}')

print(f"Greedy Value: {greedy_val}")
print(f'MILP Value: {milp_val}')
print(f'Round Value: {round_val}')


# Check MILP Optimality

#%%
raw_history, action_history = run(state)
html_out = draw_anim(raw_history, action_history, 'anim')

# %%
draw_unrolled(raw_history, action_history, 'DegGreedy on GridWorld Graphs')
# %%


state = grid_world_init(
    initial_infection=(0, 0.25),
    small_world=feature_grid,
    width=10,
    seed=42,
)
raw_history, action_history = run(state)
draw_unrolled(raw_history, action_history, 'DegGreedy on SmallWorld Graphs')
# %%
