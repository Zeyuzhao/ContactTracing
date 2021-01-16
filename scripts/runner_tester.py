from ctrace.dataset import load_graph
from ctrace.runner import *
from ctrace.simulation import generalized_mdp
from ctrace.solve import to_quarantine

# G should have a __str__ representation
G = load_graph("montgomery")

config = {
    "G": [G], # Graph
    "p": [0.078], # Probability of infection
    "budget": [500, 600], # The k value
    "method": ["gurobi"],
    "num_initial_infections": [5], # Initial Initial (DATA)
    "num_shocks": [8], # Size of shocks in initial (DATA)
    "initial_iterations": [7], # Number of iterations before intervention
    "MDP_iterations": [-1], # Number of iterations of intervention
    "iterations_to_recover": [1], # Number of iterations it takes for a infected node to recover (set to 1)
    "from_cache": ['t7.json'], # If cache is specified, some arguments are ignored
    "verbose": [False], # Prints stuff
}

# in_schema and out_schema MUST match the input arguments and namedtuple respectively!
in_schema = ["G", "p", "budget", "method", "from_cache"]
out_schema = ["objective_val", "peak_infected"]
run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=generalized_mdp, trials=2)
run.exec()
