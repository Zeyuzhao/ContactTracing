
#%%

import networkx as nx
import random
from typing import *
import itertools

#%%
def uniform_sample(l: List[Any], p: float):
    """Samples elements from l uniformly with probability p"""
    return [x for x in l if random.random() < p]
%timeit uniform_sample(list(itertools.product([i for i in range(100)], [j for j in range(100)])), 0.5)
# %%

G = nx.grid_2d_graph(100, 100)
%timeit G.copy()
# %%
