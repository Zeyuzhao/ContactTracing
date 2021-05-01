#%%
import networkx as nx
import pandas as pd
import time
import os
from ctrace.runner import *
from ctrace.utils import load_graph_hid_duration, load_graph_montgomery, load_graph_portland
from ctrace.dataset import load_sir
from ctrace.simulation import *
from ctrace.recommender import *
from collections import namedtuple
json_dir = PROJECT_ROOT / "data" / "SIR_Cache"

def calculateExpected(state: InfectionState, quarantine):
    P,Q = pq_independent_edges(state.G, state.SIR.I2, state.V1, state.V2)
    total = 0
    for v in state.V2:
        expected = 1
        for u in (set(state.G.neighbors(v)) & state.V1):
            if u not in quarantine and v not in quarantine:
                expected *= (1-P[u]*Q[u][v])
        total += (1-P[v])*(1-expected)
    
    return total

    
    
G = load_graph_montgomery()

config = {
    "G" : [G],
    #"budget":[1000],
    "budget":[i for i in range(500, 1100, 100)],
    "transmission_rate": [0.05],
    "compliance_rate": [0.8],#[i/100 for i in range(50, 101, 5)],#[i/100 for i in range(50,101,5)],
    "partial_compliance": [False],
    "I_knowledge": [1],
    "discovery_rate": [1],
    "snitch_rate":  [1],
    #"from_cache": ["c10.json"]
    "from_cache": [i for i in list(os.listdir(PROJECT_ROOT/"data"/"SIR_Cache"/"optimal_trials")) if i[0]=="m"]
}
#config["G"] = [load_graph(x) for x in config["G"]]

in_schema = list(config.keys())
out_schema = ["infection_size", "V1_size", "V2_size", "edge_size" , "D", "ip_expect", "dep_expect", "deg_expect", "time_ip", "time_dep", "time_deg", "ratio_dep", "ratio_deg"]
TrackerInfo = namedtuple("TrackerInfo", out_schema)

def optimal_ratio(G: nx.graph, budget: int, transmission_rate: float, compliance_rate: float, partial_compliance:bool, I_knowledge:float, discovery_rate: float, snitch_rate: float, from_cache: str, **kwargs):

    with open(PROJECT_ROOT / "data" / "SIR_Cache"/"optimal_trials"/from_cache, 'r') as infile:
        j = json.load(infile)
        (S, I1, I2, R) = (j["S"], j["I2"], j["I1"], j["R"])
        infections = j["infections"]
    state = InfectionState(G, (S, I1, I2, R), budget, transmission_rate, compliance_rate, partial_compliance, I_knowledge, discovery_rate, snitch_rate)

    start = time.time()
    optimal_obj = MinExposedIP2(state)
    optimal_obj.solve_lp()
    probabilities = optimal_obj.get_variables()
    rounded = D_prime(np.array(probabilities))
    quarantine = set([optimal_obj.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])
    time_ip = time.time()-start
    ip_expect = calculateExpected(state, quarantine)
    
    start = time.time()
    quarantine = DepRound2(state)
    time_dep = time.time()-start
    dep_expect = calculateExpected(state, quarantine)
    
    start = time.time()
    quarantine = DegGreedy2(state)
    time_deg = time.time()-start
    deg_expect = calculateExpected(state, quarantine)
    
    d = 0
    edge_size = 0
    for node in state.V2:
        d_temp = len(set(state.G.neighbors(node))&state.V1)
        edge_size += d_temp
        d = max(d, d_temp)
    
    return TrackerInfo(len(state.SIR.I2), len(state.V1), len(state.V2), edge_size, d, ip_expect, dep_expect, deg_expect, time_ip, time_dep, time_deg, dep_expect/max(1,ip_expect), deg_expect/max(1,ip_expect))
    #return TrackerInfo(len(state.SIR_known.SIR[2]), len(state.SIR_real.SIR[2]), information_loss_V1, information_loss_V2, information_loss_I, information_loss_V1_iterative, information_loss_V2_iterative, information_loss_V2_nbrs_iterative)

run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=optimal_ratio, trials=1)
run.exec()
    