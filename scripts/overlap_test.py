#%%
import networkx as nx
import pandas as pd
from ctrace.runner import *
from ctrace.utils import load_graph_cville_labels, load_graph_montgomery_labels, read_extra_edges
from ctrace.dataset import load_sir
from ctrace.simulation import *
from ctrace.recommender import *
from ctrace.problem_label import *
from collections import namedtuple
json_dir = PROJECT_ROOT / "data" / "SIR_Cache"

G = load_graph_montgomery_labels()
#G = read_extra_edges(G, 0.15)
#G.centrality = nx.algorithms.eigenvector_centrality_numpy(G)

#G2 = load_graph_cville_labels()
#G2 = read_extra_edges(G2, 0.15)
#G2.centrality = nx.algorithms.eigenvector_centrality_numpy(G2)

#G2 = load_graph_cville_labels()
#G2.centrality = nx.algorithms.eigenvector_centrality_numpy(G2)

#be5 for cville w/ added edges, ce6 for montgomery w/ added edges
#b5 for cville, c7 for montgomery

#defaults: snitch = 0.8, discovery = 0.8, compliance = 0.8
#montgomery budget: 750
#cville budget: 1350

def DepRound_test(state: InfectionState, lambda_round):
    state.set_budget_labels()
    
    problem2 = MinExposedLP2_label(state)
    problem2.solve_lp()
    probabilities = problem2.get_variables()

    q = set([problem2.quarantine_map[k] for (k,v) in enumerate(probabilities) if v>=lambda_round])
    
    return q

config = {
    "G" : [G],
    "budget": [i for i in range(400, 1260, 10)],
    "policy": ["none"],
    "transmission_rate": [0.05],
    "transmission_known": [True],
    "compliance_rate": [0.8],
    "compliance_known": [True],
    "discovery_rate": [1],
    "snitch_rate":  [1],
    "from_cache": [i for i in list(os.listdir(PROJECT_ROOT/"data"/"SIR_Cache"/"optimal_trials")) if i[0]=="m"],
    "lambda_round": [0.5]
}

in_schema = list(config.keys())
out_schema = ["d", "edge_size", "ratio"]
TrackerInfo = namedtuple("TrackerInfo", out_schema)

def overlap_tracker(G: nx.graph, budget: int, policy:str, transmission_rate: float, transmission_known: bool, compliance_rate: float, compliance_known:bool, discovery_rate: float, snitch_rate: float, from_cache: str, lambda_round: int, **kwargs):

    with open(PROJECT_ROOT / "data" / "SIR_Cache" / "optimal_trials"/from_cache, 'r') as infile:
            j = json.load(infile)
            (S, I1, I2, R) = (j["S"], j["I1"], j["I2"], j["R"])
            infections = j["infections"]

    state = InfectionState(G, (S, I1, I2, R), budget, policy, transmission_rate, transmission_known, compliance_rate, compliance_known, discovery_rate, snitch_rate)
    
    quarantine_deg = DegGreedy_fair(state)
    
    quarantine_dep = DepRound_test(state, lambda_round)
    
    if len(quarantine_dep)!= 0:
        ratio = len(quarantine_dep&quarantine_deg)/len(quarantine_dep)
    else:
        ratio = -1
    
    d = 0
    edge_size = 0
    for node in state.V2:
        d_temp = len(set(state.G.neighbors(node))&state.V1)
        edge_size += d_temp
        d = max(d, d_temp)
    
    return TrackerInfo(d, edge_size, ratio)

run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=overlap_tracker, trials=1)
run.exec(max_workers=40)

#%%
