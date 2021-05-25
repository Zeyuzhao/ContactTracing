#%%
import networkx as nx
import pandas as pd
from ctrace.runner import *
from ctrace.utils import *
from ctrace.dataset import *
from ctrace.simulation import *
from ctrace.recommender import *
from collections import namedtuple
json_dir = PROJECT_ROOT / "data" / "SIR_Cache"

G = load_graph_montgomery_labels()
#G = load_graph_cville_labels()
#G = read_extra_edges(G, 0.15)
#G.centrality = nx.algorithms.eigenvector_centrality_numpy(G)
#G = load_graph_hid_duration()

#be5 for cville w/ added edges, ce6 for montgomery w/ added edges

config = {
    "G" : [G],
    #"budget":[i for i in range(400, 1260, 50)],
    "policy": ["none"],
    #"budget": [i for i in range(100, 5000, 10)], #[i for i in range(100, 451, 50)],#[i for i in range(100,3710,10)],
    "transmission_rate": [0.05],
    "transmission_known": [True],
    "compliance_rate": [0.8],                      #[i/100 for i in range(50,101,5)],
    "compliance_known": [True],
    "discovery_rate": [i/100 for i in range(50, 101, 1)],
    "snitch_rate":  [0.8],
    "from_cache": ["c7.json"],                                                  #be5 is cville with extra edges
    "agent": [DepRound_fair],
    "target": [33800]
}

in_schema = list(config.keys())
out_schema = ["equivalent_budget"]
TrackerInfo = namedtuple("TrackerInfo", out_schema)

def time_trial_tracker(G: nx.graph, policy:str, transmission_rate: float, transmission_known:bool, compliance_rate: float, compliance_known:bool, discovery_rate: float, snitch_rate: float, from_cache: str, agent, target: int, **kwargs):

    with open(PROJECT_ROOT / "data" / "SIR_Cache" / from_cache, 'r') as infile:
            j = json.load(infile)
            (S, I1, I2, R) = (j["S"], j["I1"], j["I2"], j["R"])
            infections = j["infections"]
            
    l = 0
    r = 10000 

    iters = 0

    while l <= r:
        m = l + (r-l)//2

        average = 0

        iters += 1
        for i in range(20):

            state = InfectionState(G, (S, I1, I2, R), m, policy, transmission_rate, transmission_known, compliance_rate, compliance_known, discovery_rate, snitch_rate)

            while len(state.SIR.I1) + len(state.SIR.I2) != 0:
                to_quarantine = agent(state)
                state.step(to_quarantine)

            average += len(state.SIR.R)/20

        if average >= target:
            l = m+1
        else: 
            r = m-1

    return TrackerInfo(m)

run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=time_trial_tracker, trials=1)
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
