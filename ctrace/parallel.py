import concurrent.futures
import time
from simulation import *
import itertools
import pandas as pd

def runner(args):
    """Example parallel function (only accepts one argument)"""
    a, b = args
    print(f'Doing {a} and {b}')
    return a + b

def parallel(func, args, logging=True):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        start = time.perf_counter()
        results = executor.map(func, args)
    finish = time.perf_counter()
    if logging:
        print(f'Finished in {round(finish - start, 2)} seconds')
    return results

def dict_product(dicts):
    """Expands an dictionary of lists into a cartesian product of dictionaries"""
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

# <========================================== Main ==========================================>
# Generate arguments
G = load_graph("montgomery")
(S, I, R) = initial(G)

# Cartesian Product the parameters
compact_params = {
    "G": [G],
    "budget": [x for x in range(100, 1000, 100)],
    "S": [S],
    "I_t": [I],
    "R": [R],
    "iterations": [3],
    "method": ["degree", "random"],
    "visualization": [False],
    "verbose": [False],
}

params = list(dict_product(compact_params))

def simulate(param):
    (infected, peak) = MDP(**param)
    return (infected, peak)


results = list(parallel(simulate, params))

# Save results into CSV
rows = []
# Save only the graph name and append results
for param, (num, peak) in zip(params, results):
    rows.append({
        "G": param["G"].NAME,
        "budget": param["budget"],
        "iterations": param["iterations"],
        "method": param["method"],
        "num_infected": num,
        "peak": peak,
    })

print(rows)
df = pd.DataFrame(rows)
df.to_csv("../output/plots/output.csv")




