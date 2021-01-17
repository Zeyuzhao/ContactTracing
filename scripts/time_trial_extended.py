from ctrace.experiments import time_trial_extended_tracker
from ctrace.runner import *
from ctrace.simulation import generalized_mdp
from ctrace.experiments import *
from ctrace.dataset import load_graph
json_dir = PROJECT_ROOT / "data" / "SIR_Cache" / "time_trials"

cache_paths = [f for f in json_dir.iterdir()][:300] # Taking the first 300 files
print(f"Number of json files: {len(cache_paths)}")

config = {
    "G": ["montgomery"], # Graph
    "p": [0.078], # Probability of infection
    "budget": [i for i in range(100, 1001, 100)], # The k value
    "method": ["dependent", "weighted", "gurobi", "random"],
    "from_cache": cache_paths, # If cache is specified, some arguments are ignored
}
config["G"] = [load_graph(x) for x in config["G"]]

# in_schema and out_schema MUST match the input arguments and namedtuple respectively!
in_schema = list(config.keys())
out_schema = list(TimeTrialExtendTrackerInfo._fields)
run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=time_trial_extended_tracker, trials=2)
# Attempt at making schemas extensible - quite hacky right now
# run.track_duration()
run.exec()
