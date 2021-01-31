# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# %matplotlib notebook
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# %%

import random
import networkx as nx
from matplotlib import pyplot as plt

G = nx.grid_2d_graph(100,100)

plt.figure(figsize=(6,6))
pos = {(x,y):(y,-x) for x,y in G.nodes()}
nx.draw(G, pos=pos,
        node_color=[random.choice(('lightgreen', 'red')) for i in range(len(G))],
        with_labels=False,
        node_size=10)
# Try testing with a grid graph


# %%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation

# Create Graph
G = nx.grid_2d_graph(20, 20)
pos = {(x,y):(y,-x) for x,y in G.nodes()}

# Build plot
fig, ax = plt.subplots(figsize=(6,6))


def update(num):
    ax.clear()

    # Background nodes
    # nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="gray")
    # null_nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=set(G.nodes()) - set(path), node_color="white",  ax=ax)
    # null_nodes.set_edgecolor("black")

    # # Query nodes
    # query_nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=path, node_color=idx_colors[:len(path)], ax=ax)
    # query_nodes.set_edgecolor("red")
    # nx.draw_networkx_labels(G, pos=pos, labels=dict(zip(path,path)),  font_color="white", ax=ax)
    # edgelist = [path[k:k+2] for k in range(len(path) - 1)]
    
    # nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=idx_weights[:len(path)], ax=ax)
    
    nx.draw(G, pos=pos,
        node_color=[random.choice(('lightgreen', 'red')) for i in range(len(G))],
        with_labels=False,
        node_size=50,
        ax=ax
    )
    
    
    # Scale plot ax
    ax.set_title(f"Frame {num}", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


ani = matplotlib.animation.FuncAnimation(fig, update, frames=20, interval=500, repeat=True, repeat_delay=1)
plt.show()


# %%
from IPython.display import HTML


# %%
ani.to_html5_video()


# %%



