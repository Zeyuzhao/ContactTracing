#%%
import networkx as nx
import pandas as pd
from ctrace.runner import *
from ctrace.utils import load_graph_cville_labels, load_graph_montgomery_labels, read_extra_edges
from ctrace.simulation import *
from ctrace.recommender import *
from collections import namedtuple
json_dir = PROJECT_ROOT / "data" / "SIR_Cache"

G = load_graph_cville_labels()

config = {
    "G" : [G],
    "budget": [1350],
    "policy": ["none"],
    "transmission_rate": [0.05],
    "transmission_known": [True],
    "compliance_rate": [-1],
    "compliance_known": [True],
    "snitch_rate": [1],
    "from_cache": ["albe.json"],
    "k_1":[i/100 for i in range(0, 101, 1)],
    "k_2":[0.8]
}

in_schema = list(config.keys())
out_schema = ["infection_count", "infections_step"]
TrackerInfo = namedtuple("TrackerInfo", out_schema)

def time_trial_tracker(G: nx.graph, budget: int, policy:str, transmission_rate: float, transmission_known: bool, compliance_rate: float, compliance_known:bool, snitch_rate: float, from_cache: str, k_1: float, k_2: float, **kwargs):

    with open(PROJECT_ROOT / "data" / "SIR_Cache" / from_cache, 'r') as infile:
            j = json.load(infile)

            (S, I1, I2, R) = (j["S"], j["I1"], j["I2"], j["R"])
            infections = j["infections"]

    state = InfectionState(G, (S, I1, I2, R), budget, policy, transmission_rate, transmission_known, compliance_rate, compliance_known, snitch_rate)
    
    while len(state.SIR.I1) + len(state.SIR.I2) != 0:
        to_quarantine = SegDegree(state, [k_1, 1-k_1], [k_2, 1-k_2])
        state.step(to_quarantine)
        infections.append(len(state.SIR.I2))
    
    return TrackerInfo(len(state.SIR.R), infections)


run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=time_trial_tracker, trials=10)
run.exec(max_workers=100)
