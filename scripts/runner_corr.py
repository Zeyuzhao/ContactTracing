#%%
import networkx as nx
from ctrace.runner import *
from ctrace.dataset import load_graph
from ctrace.dataset import load_sir
from ctrace.simulation import *
from ctrace.recommender import *
from collections import namedtuple
json_dir = PROJECT_ROOT / "data" / "SIR_Cache"

G2 = nx.Graph()
G2.NAME = "cville"
nodes = {}
rev_nodes = []

file = open(PROJECT_ROOT / "data/raw/charlottesville.txt", "r")
file.readline()
lines = file.readlines()
c = 0
c_node=0
ma = 0
mi = 100000000

for line in lines:

    a = line.split()
    u = int(a[1])
    v = int(a[2])

    if u in nodes.keys():
        u = nodes[u]
    else:
        nodes[u] = c_node
        rev_nodes.append(u)
        u = c_node
        c_node+=1        
    
    if v in nodes.keys():
        v = nodes[v]
    else:
        nodes[v] = c_node
        rev_nodes.append(v)
        v = c_node
        c_node+=1

    G2.add_edge(u,v)

config = {
    "G" : ['montgomery'],
    "budget": [1000],
    "transmission_rate": [0.078],
    "compliance_rate": [1],
    "global_rate":  [0],        
    "discovery_rate": [1],
    "snitch_rate":  [1],
    "from_cache": ["t7.json"],
    "agent": [Random]
}
config["G"] = [load_graph(x) for x in config["G"]]

in_schema = list(config.keys())
out_schema = ["ME_objective", "total_degree"]
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

    state = SimulationState(G, (S, I, R), (S, I, R), budget, transmission_rate, compliance_rate, global_rate, discovery_rate, discovery_rate)
    
    arr = agent(state)

    q = {}

    for node in range(len(G.nodes)):
        if node in arr:
            q[node] = 1
        else:
            q[node] = 0
    
    obj = min_exposed_objective(G, (S, I, R), (state.SIR_known.V1, state.SIR_known.V2), transmission_rate, q, 100)
    sum_deg = sum([len(set(G.neighbors(i)).intersection(state.SIR_known.V2)) for i in arr])
    
    return TrackerInfo(obj, sum_deg)

run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=time_trial_tracker, trials=1000)
run.exec()

