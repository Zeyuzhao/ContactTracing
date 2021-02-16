#%%
import networkx as nx
from ctrace.runner import *
from ctrace.dataset import load_graph
from ctrace.dataset import load_sir
from ctrace.simulation import *
from ctrace.recommender import *
from collections import namedtuple
json_dir = PROJECT_ROOT / "data" / "SIR_Cache"

config = {
    "G" : ["montgomery"],
    "budget": [i for i in range(100,3010,10)],
    "transmission_rate": [0.078],
    "compliance_rate": [.65,.6,.55],
    "global_rate":  [1],        
    "discovery_rate": [1],
    "snitch_rate":  [1],
    "from_cache": ["t7.json"],
    "agent": [Degree]
}
config["G"] = [load_graph(x) for x in config["G"]]

in_schema = list(config.keys())
out_schema = ["infected_count_known", "infected_count_real"]
TrackerInfo = namedtuple("TrackerInfo", out_schema)

def time_trial_tracker(G: nx.graph, budget: int, transmission_rate: float, compliance_rate: float, global_rate: float,
                  discovery_rate: float, snitch_rate: float, from_cache: str, agent, **kwargs):

    I = set()
    with open(PROJECT_ROOT / "data" / "SIR_Cache" / from_cache, 'r') as infile:
            j = json.load(infile)
            
            (S, infected_queue, R) = (j["S"], j["I_Queue"], j["R"])

            # Make infected_queue a list of sets
            infected_queue = [set(s) for s in infected_queue]
            I = I.union(*infected_queue)
            I = list(I)

    state = SimulationState(G, (S, I, R), (S, I, R), budget, transmission_rate, compliance_rate, global_rate, discovery_rate, snitch_rate)
    
    while len(state.SIR_real.SIR[1]) != 0:
        to_quarantine = agent(state)
        state.step(to_quarantine)

    return TrackerInfo(len(state.SIR_known.SIR[2]), len(state.SIR_real.SIR[2]))

run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=time_trial_tracker, trials=10)
# Attempt at making schemas extensible - quite hacky right now
# run.track_duration()
run.exec()

"""
config = {
    "G" : [load_graph("montgomery"),],
    "budget": [i for i in range(100,1001,10)],
    "transmission_rate": [0.078],
    "compliance_rate": [1],
    "global_rate":  [0],
    "discovery_rate": [.5,.6,.7,.8,.9,1],
    "snitch_rate":  [1],
    #QUESTION: For taking the cartesian product, would unnecessary computation happen since snitch+discovery rates should be the same?
    "from_cache": ["t8.json"],
    "agent_name": [Degree]
}

run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=time_trial_tracker, trials=10)
run.exec()
"""

#%%
