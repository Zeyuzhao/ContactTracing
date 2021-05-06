#%%
import networkx as nx
import pandas as pd
from ctrace.runner import *
from ctrace.utils import load_graph_hid_duration, load_graph_cville_labels, load_graph_montgomery_labels, read_extra_edges
from ctrace.dataset import load_sir
from ctrace.simulation import *
from ctrace.recommender import *
from collections import namedtuple
json_dir = PROJECT_ROOT / "data" / "SIR_Cache"

#G = load_graph("montgomery")
G = load_graph_montgomery_labels()
G = read_extra_edges(G, 0.15)
#G = load_graph_hid_duration()

config = {
    "G" : [G],
    "budget":[1000],
    "policy": ["none"],
    #"budget": [i for i in range(100, 5000, 10)], #[i for i in range(100, 451, 50)],#[i for i in range(100,3710,10)],
    "transmission_rate": [0.05],
    "compliance_rate": [1],#[i/100 for i in range(50, 101, 5)],#[i/100 for i in range(50,101,5)],
    "compliance_known": [True, False],
    "partial_compliance": [False],
    "I_knowledge": [1],
    "discovery_rate": [1],
    "snitch_rate":  [1],
    "from_cache": ["ce6.json"],
    "agent": [Degree2, DepRound2_comp, DegGreedy2_comp]
}

in_schema = list(config.keys())
out_schema = ["infection_count", "infections_step"]
TrackerInfo = namedtuple("TrackerInfo", out_schema)

def time_trial_tracker(G: nx.graph, budget: int, policy:str, transmission_rate: float, compliance_rate: float, compliance_known:bool, partial_compliance:bool, I_knowledge:float, discovery_rate: float, snitch_rate: float, from_cache: str, agent, **kwargs):

    with open(PROJECT_ROOT / "data" / "SIR_Cache" / from_cache, 'r') as infile:
            j = json.load(infile)
            
            #(S, infected_queue, R) = (j["S"], j["I_Queue"], j["R"])
            (S, I1, I2, R) = (j["S"], j["I1"], j["I2"], j["R"])
            infections = j["infections"]
            # Make infected_queue a list of sets
            #infected_queue = [set(s) for s in infected_queue]
            #I = I.union(*infected_queue)
            #I = list(I)

    state = InfectionState(G, (S, I1, I2, R), budget, policy, transmission_rate, compliance_rate, compliance_known, partial_compliance, I_knowledge, discovery_rate, snitch_rate)
    
    while len(state.SIR.I1) + len(state.SIR.I2) != 0:
        to_quarantine = agent(state)
        state.step(to_quarantine)
        infections.append(len(state.SIR.I2))
    
    return TrackerInfo(len(state.SIR.R), infections)

run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=time_trial_tracker, trials=10)
run.exec()

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
