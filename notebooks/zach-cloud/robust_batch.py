# %%
# %load_ext autoreload

# %autoreload 2
# %%
from ctrace.runner import GridExecutorLinear, GridExecutorParallel
from ctrace.simulation import *
from ctrace.dataset import *
from ctrace.recommender import *
from ctrace.problem import *
from ctrace.utils import *
from ctrace.drawing import *
from tqdm import tqdm

import networkx as nx

# %%

# G = load_graph("montgomery")
# # Parameterize in t[n].json
# raw_sir = load_sir("t7.json", merge=True)
# SIR = SIR_Tuple(raw_sir["S"], raw_sir["I"], raw_sir["R"])
# budget=500
# transmission_rate=0.078
# compliance_rate=0.9
# structure_rate=0
# num_samples = 20
# solver_id = "GUROBI_LP"


config = {
    "G": ['montgomery'],
    "from_cache": [f't{i}.json' for i in range(7, 10)],
    "transmission_rate": [0.078],
    "compliance_rate": [0.5, 0.6, 0.7, 0.8, 0.9],
    "budget": [500, 600, 700, 800],
    "method": ["robust", "greedy"],
    "num_objectives": [1],
    "num_samples": [10, 20, 40],
}

# config = {
#     "G": ['montgomery'],
#     "from_cache": [f't{i}.json' for i in range(7, 8)],
#     "transmission_rate": [0.078],
#     "compliance_rate": [0.9],
#     "budget": [800],
#     "method": ["robust", "greedy"],
#     "num_objectives": [1],
#     "num_samples": [10],
# }
config["G"] = [load_graph(g) for g in config["G"]]
in_schema = list(config.keys())
out_schema = ["infected_v2"]
TrackerInfo = namedtuple("TrackerInfo", out_schema)

def robust_experiment(
    G,
    from_cache,
    transmission_rate,
    compliance_rate,
    budget,  # number of nodes to quarantine in V1
    method,  # robust | greedy
    # Number of times to average grader (the LP objective sample)
    num_objectives=10,
    # Number of samples for SAA (Increasing samples -> more accuracy?)
    num_samples=10,
    **args,
):
    raw_sir = load_sir(from_cache, merge=True)
    SIR = SIR_Tuple(raw_sir["S"], raw_sir["I"], raw_sir["R"])

    # # TODO: TEMP SOLUTION!!!
    # SIR = random_init(G, num_infected=1000, seed=None)
    info = InfectionInfo(G, SIR, budget, transmission_rate)
    if method == "robust":
        action = SAAAgent(
            info=info,
            debug=False,
            num_samples=num_samples,
            transmission_rate=transmission_rate,
            compliance_rate=compliance_rate,
            structure_rate=0,
            solver_id="GUROBI_LP"
        )
        # Running the objective multiple times?
        objs = [grader(G, SIR, budget, transmission_rate,
                    compliance_rate, action) for _ in tqdm(range(num_objectives))]
    elif method == "greedy":
        # Generate Greedy action
        info = InfectionInfo(G, SIR, budget, transmission_rate)
        # actions -> set of node ids
        action = DegGreedy(info)
        objs = [grader(G, SIR, budget, transmission_rate,
                    compliance_rate, action) for _ in tqdm(range(num_objectives))]
    elif method == "none":
        action = set()
        objs = [grader(G, SIR, budget, transmission_rate,
                    compliance_rate, action) for _ in tqdm(range(num_objectives))]
    else:
        raise ValueError(f"Invalid method ({method}): must be one the values")
    return TrackerInfo(max(objs))



# config = {
#     "G": ['montgomery'],
#     "from_cache": [f't{i}.json' for i in range(7, 8)],
#     "transmission_rate": [0.078],
#     "compliance_rate": [0.5],
#     "budget": [500],
#     "method": ["robust"],
#     "num_objectives": [1],
#     "num_samples": [10],
# }
config = {
    "G": ['montgomery'],
    "from_cache": [f't{i}.json' for i in range(7, 8)],
    "transmission_rate": [0.078],
    "compliance_rate": [0.8],
    "budget": [800],
    "method": ["robust"],
    "num_objectives": [100],
    "num_samples": [400, 200, 100],
}


config["G"] = [load_graph(g) for g in config["G"]]
in_schema = list(config.keys())
out_schema = ["infected_v2"]
TrackerInfo = namedtuple("TrackerInfo", out_schema)

run = GridExecutorLinear.init_multiple(
    config, in_schema, out_schema, func=robust_experiment, trials=1)

# print(f"First 5 trials: {run.expanded_config[:10]}")
print(f"Number of trials: {len(run.expanded_config)}")
# %%
run.track_duration()
# 40 sample size -> 20 workers
# 20 sample size -> 60
run.exec()

exit()

# %%

# %%
