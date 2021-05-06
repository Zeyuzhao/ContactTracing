#%%
import networkx as nx
import pandas as pd
from ctrace.runner2 import *
from ctrace.utils import load_graph_hid_duration, load_graph_montgomery, load_graph_portland, read_extra_edges
from ctrace.dataset import load_sir
from ctrace.simulation import *
from ctrace.recommender import *
from collections import namedtuple
json_dir = PROJECT_ROOT / "data" / "SIR_Cache"

#G = load_graph("montgomery")
#G = load_graph_hid_duration()
G2 = load_graph_montgomery()
#G3 = load_graph_portland()
#c7 goes with montgomery, b5 goes with cville, d27 goes with portland

config = {
    "G" : [G2],
    #"budget": [0],
    "budget": [i for i in range(500, 1260, 10)],
    #"budget":[i for i in range(10000, 18000, 500)],
    #"budget":[i for i in range(18000, 25500, 500)],
    #"budget":[i for i in range(2020, 2270, 20)],
    #"budget": [i for i in range(200, 2020, 20)], #[i for i in range(100, 451, 50)],#[i for i in range(100,3710,10)],
    "transmission_rate": [0.05],
    #"partition": [(0.04, 0.40, 1.0)],
    #"time_stage": [0],
    "compliance_rate": [0.8],#[i/100 for i in range(50, 101, 5)],#[i/100 for i in range(50,101,5)],
    "partial_compliance": [False],
    #"global_rate":  [0], 
    "I_knowledge": [1],
    "discovery_rate": [1],
    "snitch_rate":  [1],
    #"from_cache": ["d27.json"],
    "from_cache": ["c7.json"],
    #"agent": [Random]
    #"agent": [DepRound2, DegGreedy2]

    "agent": [Degree2, DegGreedy2bad, DepRound2bad]
}
#config["G"] = [load_graph(x) for x in config["G"]]

in_schema = list(config.keys())
out_schema = ["infection_count", "infections_step"]
#out_schema = ["infected_count_known", "infected_count_real", "information_loss_V1", "information_loss_V2", "information_loss_I", "information_loss_V1_iterative", "information_loss_V2_iterative", "information_loss_V2_nbrs_iterative"]
TrackerInfo = namedtuple("TrackerInfo", out_schema)

def time_trial_tracker(G: nx.graph, budget: int, transmission_rate: float, compliance_rate: float, partial_compliance:bool, I_knowledge:float, discovery_rate: float, snitch_rate: float, from_cache: str, agent, **kwargs):
    
    with open(PROJECT_ROOT / "data" / "SIR_Cache" / from_cache, 'r') as infile:
            j = json.load(infile)
            
            #(S, infected_queue, R) = (j["S"], j["I_Queue"], j["R"])
            (S, I1, I2, R) = (j["S"], j["I1"], j["I2"], j["R"])
            infections = j["infections"]
            # Make infected_queue a list of sets
            #infected_queue = [set(s) for s in infected_queue]
            #I = I.union(*infected_queue)
            #I = list(I)

    state = InfectionState(G, (S, I1, I2, R), budget, transmission_rate, compliance_rate, partial_compliance, I_knowledge, discovery_rate, snitch_rate)
    while len(state.SIR.I1) + len(state.SIR.I2) != 0:
        to_quarantine = agent(state)
        state.step(to_quarantine)
        infections.append(len(state.SIR.I2))
    
    return TrackerInfo(len(state.SIR.R), infections)
    #return TrackerInfo(len(state.SIR_known.SIR[2]), len(state.SIR_real.SIR[2]), information_loss_V1, information_loss_V2, information_loss_I, information_loss_V1_iterative, information_loss_V2_iterative, information_loss_V2_nbrs_iterative)

run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=time_trial_tracker, trials=10)
# Attempt at making schemas extensible - quite hacky right now
# run.track_duration()
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
