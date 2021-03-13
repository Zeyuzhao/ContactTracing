#%%
import networkx as nx
import pandas as pd
from ctrace.runner import *
from ctrace.dataset import load_graph
from ctrace.dataset import load_sir
from ctrace.simulation import *
from ctrace.recommender import *
from collections import namedtuple
json_dir = PROJECT_ROOT / "data" / "SIR_Cache"

def readData():
    G = nx.Graph()
    G.NAME = "cville"
    nodes = {}
    rev_nodes = []
    cnode_to_labels = {}

    file = open(PROJECT_ROOT / "data/raw/charlottesville.txt", "r")
    file.readline()
    lines = file.readlines()
    c = 0
    c_node=0
    #ma = 0
    #mi = 100000000
    
    labels_df = pd.read_csv(PROJECT_ROOT/"data/raw/cville/cville_labels.txt")
    labels_df = labels_df[["pid", "hid"]]
    labels_dict = {}
    for index, ids in labels_df.iterrows():
        labels_dict[ids["pid"].item()] = ids["hid"].item()
    
    for line in lines:

        a = line.split()
        u = int(a[1])
        v = int(a[2])

        if u in nodes.keys():
            u = nodes[u]
        else:
            nodes[u] = c_node
            rev_nodes.append(u)
            cnode_to_labels[c_node] = labels_dict[u];
            u = c_node
            c_node+=1        

        if v in nodes.keys():
            v = nodes[v]
        else:
            nodes[v] = c_node
            rev_nodes.append(v)
            cnode_to_labels[c_node] = labels_dict[v];
            v = c_node
            c_node+=1

        G.add_edge(u,v)
    
    nx.set_node_attributes(G, cnode_to_labels, 'hid')
    
    return G;


G2 = readData();


config = {
    "G" : [G2],
    "budget": [1000, 2000],#[i for i in range(100,3710,10)],
    "transmission_rate": [0.078],
    "compliance_rate": [1],#[i/100 for i in range(50,101,5)],
    "global_rate":  [0],        
    "discovery_rate": [i/100 for i in range(5, 105, 5)],
    "snitch_rate":  [1],
    "from_cache": ["a6.json"],
    "agent": [DegGreedy]
}
#config["G"] = [load_graph(x) for x in config["G"]]

in_schema = list(config.keys())
out_schema = ["infected_count_known", "infected_count_real", "information_loss_V1", "information_loss_V2", "information_loss_I", "information_loss_V1_iterative", "information_loss_V2_iterative", "information_loss_V2_nbrs_iterative"]
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
    
    information_loss_V1 = 0
    information_loss_V2 = 0
    information_loss_I = 0
    information_loss_V1_iterative = 0
    information_loss_V2_iterative = 0
    information_loss_V2_nbrs_iterative = 0
    
    while len(state.SIR_real.SIR[1]) != 0:
        to_quarantine = agent(state)
        state.step(to_quarantine)
        
        V1_real = state.SIR_real.V1
        V1_known = state.SIR_known.V1
        V2_real = state.SIR_real.V2
        V2_known = state.SIR_known.V2
        I_real = set(state.SIR_real.SIR.I)
        I_known = set(state.SIR_known.SIR.I)

        information_loss_V1 += len(V1_real-V1_known)
        information_loss_V2 += len(V2_real-V2_known)
        information_loss_I += len(I_real-I_known)
        information_loss_V1_iterative += state.SIR_known.il_v1
        information_loss_V2_iterative += state.SIR_known.il_v2
        information_loss_V2_nbrs_iterative += state.SIR_known.il_v2_nbrs

    return TrackerInfo(len(state.SIR_known.SIR[2]), len(state.SIR_real.SIR[2]), information_loss_V1, information_loss_V2, information_loss_I, information_loss_V1_iterative, information_loss_V2_iterative, information_loss_V2_nbrs_iterative)

run = GridExecutorLinear.init_multiple(config, in_schema, out_schema, func=time_trial_tracker, trials=10)
# Attempt at making schemas extensible - quite hacky right now
# run.track_duration()
run.exec()

"""
config = {
    "G" : ["montgomery"],
    "budget": [i for i in range(100,3410,10)],
    "transmission_rate": [0.078],
    "compliance_rate": [i/100 for i in range(50,101,5)],
    "global_rate":  [1],        
    "discovery_rate": [1],
    "snitch_rate":  [1],
    "from_cache": ["t7.json"],
    "agent": [DegGreedy]
}
config["G"] = [load_graph(x) for x in config["G"]]

run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=time_trial_tracker, trials=10)
run.exec()
"""
#%%
