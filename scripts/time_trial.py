import concurrent.futures
import csv
import logging
import os
import sys
import time

import shortuuid
from tqdm import tqdm

from ctrace import PROJECT_ROOT
from ctrace.dataset import *
from ctrace.simulation import *

# <======================================== Output Configurations ========================================>
# Configure Logging Files => First 5 digits
RUN_LABEL = shortuuid.uuid()[:5]

# Setup output directories
output_path = PROJECT_ROOT / "output" / f'saved_timer[{RUN_LABEL}]'
output_path.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = output_path / 'results.csv'
LOGGING_FILE = output_path / 'run.log'

# <================================================== Logging Setup ==================================================>
# Setup up Parallel Log Channel
logger = logging.getLogger("Time_Trial")
logger.setLevel(logging.DEBUG)

# Set LOGGING_FILE as output
fh = logging.FileHandler(LOGGING_FILE)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

# Info current path
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Current path {sys.path}")

# Get all files from SIR_Cache

json_dir = PROJECT_ROOT / "data" / "SIR_Cache" / "time_trials"
cache_paths = [f for f in json_dir.iterdir()]
print(f"Number of json files: {len(cache_paths)}")
# MAX_WORKERS = 1
COMPACT_CONFIG = {
    "G": ["montgomery"], # Graph
    "p": [0.078], # Probability of infection
    "budget": [i for i in range(400, 1001, 100)], # The k value
    "method": ["dependent", "weighted"],
    "from_cache": cache_paths, # If cache is specified, some arguments are ignored
    "trials": 10, # Number of trials to run for each config
}

# Setup load graph:
COMPACT_CONFIG["G"] = [load_graph(x) for x in COMPACT_CONFIG["G"]]

# Attributes need to partition configuration! Do NOT have duplicate attributes
COMPLEX = ["G"] # Attributes that need to be processed before printing
HIDDEN = ["visualization", "verbose", "trials", "logging"] # These attributes will NOT be logged at all

# Anything not in PRIMITIVE or HIDDEN
PRIMITIVE = list(COMPACT_CONFIG.keys() - set(COMPLEX) - set(HIDDEN)) # Attributes that will be printed as is
RESULTS = ["value", "isOptimal", "maxD", "duration"]

# Utilities
def dict_product(dicts):
    """Expands an dictionary of lists into a list of dictionaries through a cartesian product"""
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def expand_configurations(compact_config: Dict):
    """Expands compact configuration into many runnable configs through a cartesian product"""
    compact_config = compact_config.copy()

    # Handle multiple trials
    compact_config["trial_id"] = [i for i in range(compact_config["trials"])]
    del compact_config["trials"]

    # Expand configuration
    return list(dict_product(compact_config))


def readable_configuration(config: Dict):
    """Takes in a instance of an expanded configuration and returns a readable object"""

    output = {}


    # Paste in PRIMITIVE attributes
    for p in PRIMITIVE:
        output[p] = config[p]
    
    # Handle COMPLEX attributes
    output["G"] = config["G"].NAME
    output["from_cache"] = config["from_cache"].name

    # Ignore HIDDEN attributes
    return output

def MDP_runner(param):
    """Takes in runnable parameter and returns a (Result tuple, Readable Params)"""
    readable_params = readable_configuration(param)
    logger.info(f"Launching => {readable_params}")

    SIR = load_sir_path(param["from_cache"], merge=True)
    processed_params = {
        "G": param["G"],
        "I0": SIR["I"],
        "safe": SIR["R"],
        "cost_constraint": param["budget"],
        "p": param["p"],
        "method": param["method"],
    }
    t0 = time.time()
    # TODO: Fix parameters?
    (value, sol, isOptimal, maxD) = trial_tracker(**processed_params)
    t1 = time.time()
    time_diff = t1-t0
    return (value, isOptimal, maxD, time_diff), readable_params

def parallel_MDP(args: List[Dict]):
    with concurrent.futures.ProcessPoolExecutor() as executor, open(OUTPUT_FILE, "w") as output_file:
        results = [executor.submit(MDP_runner, arg) for arg in args]
        writer = csv.DictWriter(output_file, fieldnames=COMPLEX + PRIMITIVE + RESULTS)
        writer.writeheader()
        for f in tqdm(concurrent.futures.as_completed(results), total=len(args)):
            (value, isOptimal, maxD, time_diff), readable = f.result()

            # Merge the two dictionaries, with results taking precedence
            entry = readable
            result_dict = {
                "value": value,
                "isOptimal": isOptimal,
                "maxD": maxD,
                "duration": time_diff,
            }
            entry.update(result_dict)

            # Write and flush results
            writer.writerow(entry)
            output_file.flush()
            logger.info(f"Finished => {entry}")

def linear_MDP(args: List[Dict]):
    with open(OUTPUT_FILE, "w") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=COMPLEX + PRIMITIVE + RESULTS)
        writer.writeheader()
        for arg in tqdm(args, total=len(args)):
            (value, isOptimal, maxD, time_diff), readable = MDP_runner(arg)

            # Merge the two dictionaries, with results taking precedence
            entry = readable
            result_dict = {
                "value": value,
                "isOptimal": isOptimal,
                "maxD": maxD,
                "duration": time_diff,
            }
            entry.update(result_dict)

            # Write and flush results
            writer.writerow(entry)
            output_file.flush()
            logger.info(f"Finished => {entry}")


print(f'Logging Directory: {LOGGING_FILE}')
expanded_configs = expand_configurations(COMPACT_CONFIG)
print(expanded_configs[0])
parallel_MDP(expanded_configs)
print('done')
# to_quarantine(G=GRAPH,
#               I0=I,
#               safe=RECOVERED_SET,
#               cost_constraint=K_VALUE,
#               p=P_VALUE,
#               method = "dependent",
#               runs=None,
#               P=None,
#               Q=None)