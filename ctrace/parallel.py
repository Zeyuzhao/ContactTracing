import concurrent.futures
import time
from simulation import *
import itertools

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
    "budget": [x for x in range(100, 1000, 10)],
    "S": [S],
    "I_t": [I],
    "R": [R],
    "iterations": 15,
    "method": ["none", "degree", "random", "dependent", "iterated", "optimized"],
    "visualization": [False],
    "verbose": [False],
}

params = dict_product(compact_params)

def simulate(param):
    return MDP(**param)

results = parallel(simulate, params)

