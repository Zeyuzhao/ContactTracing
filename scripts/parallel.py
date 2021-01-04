import sys
import os
import csv
from tqdm import tqdm
import logging

import pandas as pd
import itertools
import time
import concurrent.futures

# Set path to ContactTracing/
os.chdir('..')
sys.path.insert(0, '.')

output_file = f'output/plots/results[{time.strftime("%H-%M-%S")}].csv'
logging_file = f'output/plots/results[{time.strftime("%H-%M-%S")}].log'

logging.basicConfig(filename=logging_file, level=logging.DEBUG)

logging.info(f"Current working directory: {os.getcwd()}")
logging.info(f"Current path {sys.path}")

from ctrace.simulation import *


def minimize_params(p):
    p = p.copy()
    # Reduce graph to graph name only
    p["G"] = p["G"].NAME

    # Remove unnecessary parameters not needed for logging
    del p["S"]
    del p["I_t"]
    del p["R"]
    del p["visualization"]
    del p["verbose"]
    return p

def parallel(func, args, output_file=None):
    """Parallelize the simulate function"""
    # TODO: Make this function generic
    entries = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        start = time.perf_counter()
        results = [executor.submit(func, arg) for arg in args]

        if not output_file:
            output_file = f'output/plots/results.csv'
        with open(output_file, "w") as logging_file:
            # Attributes to log to files
            attr = [
                "G",            # The parameters of the experiment
                "SIR_file",
                "budget",
                "iterations",
                "method",
                "p",
                "trial_id",     # The trial number
                "num_infected", # Output Parameters
                "peak",
            ]
            writer = csv.DictWriter(logging_file, fieldnames=attr)
            writer.writeheader()
            for f in tqdm(concurrent.futures.as_completed(results), total=len(args)):
                ((infected, peak), param) = f.result()
                # Reduce parameters to viewable ones only
                param_view = minimize_params(param)
                entry = {
                    **param_view,
                    "num_infected": infected,
                    "peak": peak,
                }
                writer.writerow(entry)
                entries.append(entry)
                logging_file.flush()
                logging.info(f"Finished => {entry}")

        finish = time.perf_counter()
        logging.info(f'Finished in {round(finish - start, 2)} seconds')
    return entries

def dict_product(dicts):
    """Expands an dictionary of lists into a cartesian product of dictionaries"""
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

# <========================================== Main ==========================================>
# Generate arguments
G = load_graph("montgomery")
SIR_file = "Q4data.json"
(S, I, R) = initial(from_cache=SIR_file)
logging.info("Loaded!")

# <========================================== Important Parameters ==========================================>

# Cartesian Product the parameters
TRIALS = 10
compact_params = {
    "trial_id": [x for x in range(TRIALS)],
    "G": [G],
    "budget": [1300],  # k
    "S": [S],
    "I_t": [I],
    "R": [R],
    "SIR_file": [SIR_file],
    "iterations": [-1],
    "method": ["dependent", "random", "degree"],
    "p": [x * 0.01 for x in range(1, 16)],
    "visualization": [False],
    "verbose": [False],
}

# <========================================== End of Parameters ==========================================>

params = list(dict_product(compact_params))

# print(params)
# Runs multiple trials
def simulate(param):
    logging.info(
        f"Launching => {minimize_params(param)}"
    )
    (infected, peak) = MDP(**param)
    return ((infected, peak), param)

results = list(parallel(simulate, params, output_file))