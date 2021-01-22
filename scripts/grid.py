import networkx as nx
import matplotlib.pyplot as plt
g = nx.grid_2d_graph(m=10, n=10)


pos = nx.spring_layout(g, iterations=100)
nx.draw(g)
plt.show()