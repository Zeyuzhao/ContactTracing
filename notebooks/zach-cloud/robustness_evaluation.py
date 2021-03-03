# %%
# %load_ext autoreload

# %autoreload 2
# %%
from ctrace.runner import GridExecutorLinear
from ctrace.simulation import *
from ctrace.dataset import *
from ctrace.recommender import *
from ctrace.problem import *
from ctrace.utils import *
from ctrace.drawing import *

import networkx as nx


def grader(
    G,
    SIR,
    budget,
    transmission_rate,
    compliance_rate,
    action,
    structure_rate=0,
    grader_seed=None,
    num_samples=1,
    solver_id="GUROBI_LP",
):
    gproblem = MinExposedSAA.create(
        G=G,
        SIR=SIR,
        budget=budget,
        transmission_rate=transmission_rate,
        compliance_rate=compliance_rate,
        structure_rate=structure_rate,
        num_samples=num_samples,
        seed=grader_seed,
        solver_id=solver_id,
    )
    # Pre-set the solveable parameters
    for node in action:
        gproblem.set_variable_id(node, 1)
    _ = gproblem.solve_lp()
    return gproblem.objective_value

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


CONFIG = {
    "G": ['montgomery'],
    "from_cache": [f't{i}.json' for i in range(7, 17)],
    "transmission_rate": [0.078],
    "compliance_rate": [0.5, 0.6, 0.7, 0.8, 0.9],
    "budget": [500, 600, 700, 800],
    "method": ["robust", "greedy"],
    "num_objectives": [10],
    "num_samples": [5, 10, 20, 40, 80],
}

CONFIG = {
    "G": ['montgomery'],
    "from_cache": [f't{i}.json' for i in range(7, 8)],
    "transmission_rate": [0.078],
    "compliance_rate": [0.9],
    "budget": [800],
    "method": ["robust", "greedy"],
    "num_objectives": [1],
    "num_samples": [1],
}
CONFIG["G"] = [load_graph(g) for g in CONFIG["G"]]

# Truncate G
CONFIG["G"][0] = CONFIG["G"][0].remove_nodes_from([i for i in range(int(len(CONFIG["G"][0].nodes) * 9/10))])
in_schema = list(CONFIG.keys())
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
    info = InfectionInfo(G, SIR, budget, transmission_rate)
    if method == "robust":
        action = SAAAgent(
            info=info,
            debug=False,
            num_samples=num_samples,
            transmission_rate=transmission_rate,
            compliance_rate=compliance_rate,
            structure_rate=0,
        )
        # Running the objective multiple times?
        objs = [grader(G, SIR, budget, transmission_rate,
                       compliance_rate, action) for _ in range(num_objectives)]

    if method == "greedy":
        # Generate Greedy action
        info = InfectionInfo(G, SIR, budget, transmission_rate)
        # actions -> set of node ids
        action = DegGreedy(info)
        objs = [grader(G, SIR, budget, transmission_rate,
                       compliance_rate, action) for _ in range(num_objectives)]
    else:
        raise ValueError("Invalid method: must be one the values")
    return TrackerInfo(mean(objs))


run = GridExecutorLinear.init_multiple(
    CONFIG, in_schema, out_schema, func=robust_experiment, trials=5)

# print(f"First 5 trials: {run.expanded_config[:10]}")
print(f"Number of trials: {len(run.expanded_config)}")
# %%
run.exec()
# %%

# %%
