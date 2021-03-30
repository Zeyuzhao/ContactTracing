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

import networkx as nx

from scipy.stats import zipf

deg = [zipf.rvs(a=2)+1 for _ in range(1000)]
G = nx.expected_degree_graph(deg)

while len(G.edges) < 2500:
    deg = [zipf.rvs(a=2)+1 for _ in range(1000)]
    G = nx.expected_degree_graph(deg)

print(len(G.edges))

S = []
I = []
R = []

for i in range(len(G.nodes)):
    if random.random() > .15:
        S.append(i)
    else:
        I.append(i)
        
SIR = SIR_Tuple(S,I,R)

state = InfectionInfo(G,SIR,100,.5)
print(len(state.SIR.I))
print(len(state.V1))
print(len(state.V2))


in_schema = [
    "G",
    "from_cache",
    "transmission_rate",
    "compliance_rate",
    "budget",
    "method",
    "num_objectives",
    "num_samples"
]


def robust_experiment(
    G,
    from_cache,
    transmission_rate,
    compliance_rate,
    budget,  # number of nodes to quarantine in V1
    method,  # robust | greedy
    num_objectives=10,
    # Number of samples for SAA (Increasing samples -> more accuracy?)
    num_samples=1,
    **args,
):
    

    info = InfectionInfo(G, SIR, budget, transmission_rate)
    if method == "robust":
        action = SAAAgent(
            info=info,
            debug=False,
            num_samples=num_samples,
            transmission_rate=transmission_rate,
            compliance_rate=compliance_rate,
            structure_rate=0,
            solver_id="GLOP"
        )
        # Running the objective multiple times?
        
    elif method == "greedy":
        # Generate Greedy action
        info = InfectionInfo(G, SIR, budget, transmission_rate)
        # actions -> set of node ids
        action = DegGreedy(info)
    elif method == "none":
        objs = [grader(G, SIR, budget, transmission_rate,
                    compliance_rate, set()) for _ in range(num_objectives)]
    else:
        raise ValueError(f"Invalid method ({method}): must be one the values")
    print("hi")
    objs = [grader(G, SIR, budget, transmission_rate,
                    compliance_rate, action) for _ in range(num_objectives)]
    
    with open(PROJECT_ROOT / "output" / "robust1.json", 'r+') as infile:
        data = json.load(infile)
        name = method + str(num_samples)
        
        if name in data:
            data[name] = data[name] + objs
        else:
            data[name] = objs
        infile.seek(0)
        json.dump(data,infile)
        infile.truncate()
        
    return TrackerInfo(mean(objs),np.percentile(objs, 95), max(objs))

# Config for robust
config = {
    "G": [G],
    "from_cache": ["generated1000"],
    "transmission_rate": [1],
    "compliance_rate": [0.6],
    "budget": [100],
    "method": ["robust"],
    "num_objectives": [3000],
    "num_samples": [i for i in range(100,2001,100)],
}


#config["G"] = [load_graph(g) for g in config["G"]]
in_schema = list(config.keys())
out_schema = ["mean_infected_v2","nf_infected_v2","max_infected_v2"]
TrackerInfo = namedtuple("TrackerInfo", out_schema)

run = GridExecutorParallel.init_multiple(
    config, in_schema, out_schema, func=robust_experiment, trials=10)

# print(f"First 5 trials: {run.expanded_config[:10]}")
print(f"Number of trials: {len(run.expanded_config)}")
# %%
run.track_duration()

# Lower max_workers
# 40 sample size -> 20 workers
# 20 sample size -> 60
run.exec(70)

robust_experiment(G, "generated1000", .5, .8, 100, "greedy", num_samples=1, num_objectives=3000)

exit()

# %%

# %%
