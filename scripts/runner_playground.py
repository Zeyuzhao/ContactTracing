#%%
import networkx as nx
import pandas as pd
from ctrace.runner import *
from ctrace.utils import load_graph_cville_labels, load_graph_montgomery_labels, read_extra_edges
from ctrace.dataset import load_sir
from ctrace.simulation import *
from ctrace.recommender import *
from collections import namedtuple
json_dir = PROJECT_ROOT / "data" / "SIR_Cache"

#G = load_graph("montgomery")
#G2 = load_graph_montgomery_labels()
G2 = load_graph_cville_labels()
G2 = read_extra_edges(G2, 0.15)
G2.centrality = nx.algorithms.eigenvector_centrality_numpy(G2)

#G2 = load_graph_cville_labels()
#G2.centrality = nx.algorithms.eigenvector_centrality_numpy(G2)
#G = read_extra_edges(G, 0.15)
#G = load_graph_hid_duration()
#be5 for cville w/ added edges, ce6 for montgomery w/ added edges
#b5 for cville, c7 for montgomery
'''config = {
    "G" : [G],
    "budget": [1000],
    #"budget": [i for i in range(400, 1260, 10)],
    #"budget": [i for i in range(720, 2270, 20)],
    "policy": ["none"],
    "transmission_rate": [0.05],
    "transmission_known": [True],
    #"compliance_rate": [0.8],
    "compliance_rate": [i/100 for i in range(50, 101, 1)],
    "compliance_known": [True],
    "discovery_rate": [1],
    "snitch_rate": [1],
    #"discovery_rate": [.8],
    #"snitch_rate":  [i/100 for i in range(50,101,1)],
    "from_cache": ["c7.json"],
    "agent": [DegGreedy2_fair, DepRound2_fair]
}'''

config = {
    "G" : [G2],
    "budget": [i for i in range(720, 2270, 20)],
    "policy": ["none"],
    "transmission_rate": [0.05],
    "transmission_known": [True],
    "compliance_rate": [0.8],
    "compliance_known": [True],
    "discovery_rate": [1],
    "snitch_rate": [1],
    "from_cache": ["be5.json"],
    "agent": [Random, EC, DegGreedy_fair, DepRound_fair]
}

in_schema = list(config.keys())
out_schema = ["infection_count", "infections_step"]
TrackerInfo = namedtuple("TrackerInfo", out_schema)

def time_trial_tracker(G: nx.graph, budget: int, policy:str, transmission_rate: float, transmission_known: bool, compliance_rate: float, compliance_known:bool, discovery_rate: float, snitch_rate: float, from_cache: str, agent, **kwargs):

    with open(PROJECT_ROOT / "data" / "SIR_Cache" / from_cache, 'r') as infile:
            j = json.load(infile)

            (S, I1, I2, R) = (j["S"], j["I1"], j["I2"], j["R"])
            infections = j["infections"]

    state = InfectionState(G, (S, I1, I2, R), budget, policy, transmission_rate, transmission_known, compliance_rate, compliance_known, discovery_rate, snitch_rate)
    
    while len(state.SIR.I1) + len(state.SIR.I2) != 0:
        to_quarantine = agent(state)
        state.step(to_quarantine)
        infections.append(len(state.SIR.I2))
    
    return TrackerInfo(len(state.SIR.R), infections)

run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=time_trial_tracker, trials=10)
run.exec(max_workers=40)

'''config = {
    "G" : [G2],
    "budget": [1000,2000],
    "transmission_rate": [0.06],
    "compliance_rate": [1],
    "global_rate":  [.05],        
    "discovery_rate": [i/100 for i in range(1,101)],
    "snitch_rate":  [.8],
    "from_cache": ["a5.json"],
    "agent": [DegGreedy]
}

run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=time_trial_tracker, trials=10)
run.exec()'''

#%%
