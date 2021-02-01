#%%
from ctrace.runner import *
from ctrace.dataset import load_graph
from collections import namedtuple
json_dir = PROJECT_ROOT / "data" / "SIR_Cache"

cache_paths = [f for f in json_dir.iterdir()][:10] # Taking the first 400 files
print(f"Number of json files: {len(cache_paths)}")

config = {
    "G": ["montgomery"], # Graph
    "p": [0.078], # Probability of infection
    "budget": [i for i in range(600, 1200 + 1, 100)], # The k value
    "from_cache": cache_paths,  # If cache is specified, some arguments are ignored
    "method": ["random", "dependent", "greedy_weighted", "mip_gurobi"],
}
config["G"] = [load_graph(x) for x in config["G"]]

# in_schema and out_schema MUST match the input arguments and namedtuple respectively!
in_schema = list(config.keys())
out_schema = ["test"]

TutorialOutput = namedtuple("TutorialOutput", out_schema)
def tutorial(G, p, budget, from_cache, method, **kwargs):
    return TutorialOutput(from_cache)

run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=tutorial, trials=1)
# Attempt at making schemas extensible - quite hacky right now
# run.track_duration()

#%%
run.exec()