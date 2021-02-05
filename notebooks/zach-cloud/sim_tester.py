
#%%
import networkx as nx
from ctrace.simulation import PartitionSIR, InfectionEnv

G = nx.grid_2d_graph(8, 8)
mapper = {n : i for i, n in enumerate(G.nodes())}
pos = {i:(y,-x) for i, (x,y) in enumerate(G.nodes())}
G = nx.relabel_nodes(G, mapper)

env = InfectionEnv(G, transmission_rate = 1, delay=5)

# %%
env.SIR_History
# %%
