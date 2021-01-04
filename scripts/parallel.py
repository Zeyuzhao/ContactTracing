import sys
import os
import csv
# Set path to ContactTracing/
os.chdir('..')
sys.path.insert(0, '.')
print(f"Current working directory: {os.getcwd()}")
print(f"Current path {sys.path}")

import pandas as pd
import itertools
import time
import concurrent.futures
from ctrace.simulation import *


def runner(args):
    """Example parallel function (only accepts one argument)"""
    a, b = args
    print(f'Doing {a} and {b}')
    return a + b


def parallel(func, args, logging=True):
    """Parallelize the simulate function"""
    # TODO: Make this function generic
    entries = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        start = time.perf_counter()
        results = [executor.submit(func, arg) for arg in args]

        with open('output/plots/logging.csv', "w") as logging_file:
            attr = [
                "G",
                "SIR_file",
                "budget",
                "iterations",
                "method",
                "trial_id",
                "num_infected",
                "peak",
            ]
            writer = csv.DictWriter(logging_file, fieldnames=attr)
            writer.writeheader()
            for f in concurrent.futures.as_completed(results):
                ((infected, peak), param) = f.result()
                entry = {
                    "G": param["G"].NAME,
                    "SIR_file": SIR_file,
                    "budget": param["budget"],
                    "iterations": param["iterations"],
                    "method": param["method"],
                    "trial_id": param["trial_id"],
                    "num_infected": infected,
                    "peak": peak,
                }
                writer.writerow(entry)
                entries.append(entry)
                logging_file.flush()
                print(f"Finished => {entry}")

        finish = time.perf_counter()
        print(f'Finished in {round(finish - start, 2)} seconds')
    return entries


def dict_product(dicts):
    """Expands an dictionary of lists into a cartesian product of dictionaries"""
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


# <========================================== Main ==========================================>
# Generate arguments
print(os.getcwd())
G = load_graph("montgomery")
SIR_file = "Q4data.json"
(S, I, R) = initial(from_cache=SIR_file)
print("Loaded!")
# Cartesian Product the parameters
trials = 2
compact_params = {
    "G": [G],
    "budget": [x for x in range(1000, 2000, 1000)],  # k
    "S": [S],
    "I_t": [I],
    "R": [R],
    "SIR_file": [SIR_file],
    "iterations": [3],
    "method": ["random", "degree"],
    "visualization": [False],
    "verbose": [False],
    "trial_id": [x for x in range(trials)],
}

params = list(dict_product(compact_params))

# print(params)
# Runs multiple trials
def simulate(param):
    print(
        f"Launching => budget: { param['budget'] }, method: {param['method']}, trial: {param['trial_id']}")
    (infected, peak) = MDP(**param)
    return ((infected, peak), param)


results = list(parallel(simulate, params))

# # <=========================== Save results into CSV ===========================>
# rows = []
# # Save only the graph name and append results
# for param, (num, peak) in zip(params, results):
#     rows.append({
#         "G": param["G"].NAME,
#         "SIR_file": SIR_file,
#         "budget": param["budget"],
#         "iterations": param["iterations"],
#         "method": param["method"],
#         "num_infected": num,
#         "peak": peak,
#     })

# df = pd.DataFrame(rows)
# df.to_csv("output/plots/output.csv")
