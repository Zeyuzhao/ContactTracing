#%%
import networkx as nx
import pandas as pd
import time
import os
from ctrace.runner import *
from ctrace.utils import load_graph_montgomery_labels, load_graph_cville_labels, read_extra_edges
from ctrace.dataset import load_sir
from ctrace.simulation import *
from ctrace.recommender import *
from collections import namedtuple
json_dir = PROJECT_ROOT / "data" / "SIR_Cache"

def calculateExpected(state: InfectionState, quarantine):
    P, Q = state.P, state.Q
    total = 0
    for v in state.V2:
        expected = 1
        for u in (set(state.G.neighbors(v)) & state.V1):
            if u not in quarantine:
                expected *= (1-P[u]*Q[u][v])
            else:
                expected *= (1-(1-state.G.nodes[u]['compliance_rate'])*P[u]*Q[u][v])
        total += (1-expected)
    return total

G = load_graph_montgomery_labels()

config = {
    "G" : [G],
    "budget": [750],
    "policy": ["none"],
    "transmission_rate": [i/100 for i in range(0, 100, 5)],
    "transmission_known": [True],
    "compliance_rate": [0.8],
    "compliance_known": [True],
    "discovery_rate": [1],
    "snitch_rate":  [1],
    "from_cache": [i for i in list(os.listdir(PROJECT_ROOT/"data"/"SIR_Cache"/"optimal_trials")) if i[0]=="m" and i[1]=="a"],
}

#config["G"] = [load_graph(x) for x in config["G"]]

in_schema = list(config.keys())
out_schema = ["infection_size", "V1_size", "V2_size", "edge_size" , "D", "ip_expect", "dep_expect", "deg_expect", "time_ip", "time_dep", "time_deg", "ratio_dep", "ratio_deg"]
TrackerInfo = namedtuple("TrackerInfo", out_schema)

def optimal_ratio(G: nx.graph, budget: int, policy: str, transmission_rate: float, transmission_known: bool, compliance_rate: float, compliance_known: bool, discovery_rate: float, snitch_rate: float, from_cache: str, **kwargs):

    with open(PROJECT_ROOT / "data" / "SIR_Cache"/"optimal_trials"/from_cache, 'r') as infile:
        j = json.load(infile)
        (S, I1, I2, R) = (j["S"], j["I2"], j["I1"], j["R"])
        infections = j["infections"]
    
    state = InfectionState(G, (S, I1, I2, R), budget, policy, transmission_rate, transmission_known, compliance_rate, compliance_known, discovery_rate, snitch_rate)

    state.set_budget_labels()
    
    start = time.time()
    optimal_obj = MinExposedIP2_label(state)
    q = optimal_obj.solve_lp()
    time_ip = time.time()-start
    quarantine = set([k for (k,v) in q.items() if v==1])
    ip_expect = calculateExpected(state, quarantine)
    
    start = time.time()
    quarantine = DepRound_fair(state)
    time_dep = time.time()-start
    dep_expect = calculateExpected(state, quarantine)
    
    start = time.time()
    quarantine = DegGreedy_fair(state)
    time_deg = time.time()-start
    deg_expect = calculateExpected(state, quarantine)
    
    d = 0
    edge_size = 0
    for node in state.V2:
        d_temp = len(set(state.G.neighbors(node))&state.V1)
        edge_size += d_temp
        d = max(d, d_temp)
    
    if ip_expect == 0:
        ip_expect_checked = 1
    else:
        ip_expect_checked = ip_expect
    
    return TrackerInfo(len(state.SIR.I2), len(state.V1), len(state.V2), edge_size, d, ip_expect, dep_expect, deg_expect, time_ip, time_dep, time_deg, dep_expect/ip_expect_checked, deg_expect/ip_expect_checked)
    #return TrackerInfo(len(state.SIR_known.SIR[2]), len(state.SIR_real.SIR[2]), information_loss_V1, information_loss_V2, information_loss_I, information_loss_V1_iterative, information_loss_V2_iterative, information_loss_V2_nbrs_iterative)

run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=optimal_ratio, trials=1)
run.exec()