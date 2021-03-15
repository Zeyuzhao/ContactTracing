
# %%
from multiprocessing import Process
import concurrent.futures
import os
import itertools
from collections import deque, OrderedDict
from joblib import Parallel, delayed
from multiprocessing import Queue
from pprint import pprint
# %%

# Initialize input schemas
# Last elements change frequently
in_schema = ["graph", "sir", "method"]
tasks = []

# Append schemas
compact1 = {
    'graph': ['montgomery'],
    'sir': ['a.json', 'b.json', 'c.json'],
    'method': ['robust', 'greedy', 'random'],
}
# When appending, validate input
tasks.extend(dict(zip(compact1, x)) for x in itertools.product(*compact1.values()))

compact2 = {
    'graph': ['alpine'],
    'sir': ['a.json', 'b.json', 'c.json'],
    'method': ['robust', 'greedy', 'random'],
}
tasks.extend(dict(zip(compact2, x)) for x in itertools.product(*compact2.values()))

tasks.extend([{'graph': 'test', 'sir': 'd.json', 'method': 'none'}])
pprint(tasks)

# Ensure 
# %%
