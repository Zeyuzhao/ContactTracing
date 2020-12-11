import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
G = nx.karate_club_graph()
# print("Degrees")
# for v in G:
#     print(f"{v:4} {G.degree(v): 6}")

I_SET = {14, 16}
# Compute distances
dist_dict = nx.multi_source_dijkstra_path_length(G, I_SET)

# convert dict vertex -> distance
# to distance -> [vertex]

level_dists = defaultdict(list)
for (i, v) in dist_dict.items():
    level_dists[v].append(i)

print(level_dists)

# Obtain V_1 and V_2
print(f"Distance 1: {level_dists[1]}")
print(f"Distance 2: {level_dists[2]}")

# convert dict of distances to array
N = G.number_of_nodes()
dists = [0] * N
for (i, v) in dist_dict.items():
    dists[i] = v

# flip by distances

# set layout
pos = nx.spring_layout(G, iterations=200, seed=42)

# set drawing parameters
params = {
    "pos": pos,
    "node_color": dists,
    "with_labels": True,
    "cmap": plt.cm.Blues,
}
nx.draw(G, **params)
plt.show()