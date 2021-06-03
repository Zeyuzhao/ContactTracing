import networkx as nx
import pandas as pd
from ctrace.runner import *
from ctrace.utils import *
from ctrace.dataset import *
from ctrace.simulation import *
from ctrace.recommender import *
from collections import namedtuple
json_dir = PROJECT_ROOT / "data" / "SIR_Cache"

G = load_graph_cville_labels()

config = {
    "G" : [G],
    "policy": ["none"],
    "transmission_rate": [0.05],
    "transmission_known": [True],
    "compliance_rate": [i/100 for i in range(50,101,1)],
    "compliance_known": [True],
    "snitch_rate":  [1],
    "from_cache": ["albe.json"],
    "agent": [SegDegree, DegGreedy_fair, DepRound_fair],
    "target": [64151.03]
}

in_schema = list(config.keys())
out_schema = ["equivalent_budget"]
TrackerInfo = namedtuple("TrackerInfo", out_schema)

def time_trial_tracker(G: nx.graph, policy:str, transmission_rate: float, transmission_known:bool, compliance_rate: float, compliance_known:bool, snitch_rate: float, from_cache: str, agent, target: int, **kwargs):

    with open(PROJECT_ROOT / "data" / "SIR_Cache" / from_cache, 'r') as infile:
            j = json.load(infile)
            (S, I1, I2, R) = (j["S"], j["I1"], j["I2"], j["R"])
            
    l = 0
    r = 20000 

    iters = 0

    while l <= r:
        m = l + (r-l)//2

        average = 0

        iters += 1
        for i in range(20):

            state = InfectionState(G, (S, I1, I2, R), m, policy, transmission_rate, transmission_known, compliance_rate, compliance_known, snitch_rate)

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
run.exec(max_workers=100)
