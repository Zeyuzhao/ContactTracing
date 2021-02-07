#%%
import networkx as nx
from ctrace.runner import *
from ctrace.dataset import load_graph
from ctrace.dataset import load_sir
from ctrace.simulation import *
from ctrace.recommender import *
from collections import namedtuple
json_dir = PROJECT_ROOT / "data" / "SIR_Cache"

#Take only files that start with t, YES this is a terrible solution. Also, I only took 1 file.
cache_paths = [f for f in json_dir.iterdir() if str(f)[str(f).rindex("/")+1]=="t"][:1] # Taking the first 400 files

print(f"Number of json files: {len(cache_paths)}")

config = {
    "G" : ["montgomery"],
    "budget": [1000],
    "transmission_rate": [0.078],
    "compliance_rate": [i/100 for i in range(10, 101, 1)],
    "global_rate":  [1],         #ranges are not currently inclusive of upper bound
    "discovery_rate": [1],
    "snitch_rate":  [1],
    #QUESTION: For taking the cartesian product, would unnecessary computation happen since snitch+discovery rates should be the same?
    "from_cache": cache_paths,
    "agent_name": ["Degree"]
}
config["G"] = [load_graph(x) for x in config["G"]]

in_schema = list(config.keys())
out_schema = ["infected_count_known", "infected_count_real"]
TrackerInfo = namedtuple("TrackerInfo", out_schema)

#Wouldn't compile the nx for G: nx.graph + I don't know what type from_cache is
def time_trial_tracker(G: nx.graph, budget: int, transmission_rate: float, compliance_rate: float, global_rate: float,
                  discovery_rate: float, snitch_rate: float, from_cache, agent_name: str, **kwargs):

    #George asks: does this work? good question.
    SIR = load_sir(from_cache, merge=True)
    S = SIR["S"]
    I = SIR["I"]
    R = SIR["R"]

    # George's code that should work in reading the files but is 'Less Pretty'
    # (it would make the from_cache list comprehension less awful though)
    ''' I = set()
       with open(PROJECT_ROOT / "data" / "SIR_Cache" / from_cache, 'r') as infile:
           j = json.load(infile)
           (S, infected_queue, R) = (j["S"], j["I_Queue"], j["R"])

           # Make infected_queue a list of sets
           infected_queue = [set(s) for s in infected_queue]
           I = I.union(*infected_queue)
           I = list(I)'''

    state = SimulationState(G, (S, I, R), (S, I, R), budget, transmission_rate, compliance_rate, global_rate,
                                                                                    discovery_rate, snitch_rate)
    agent = Random
    if agent_name == "Degree":
        agent = Degree

    while len(state.SIR_real.SIR[1]) != 0:
        to_quarantine = agent(state)
        state.step(to_quarantine)

    return TrackerInfo(len(state.SIR_known.SIR[2]), len(state.SIR_real.SIR[2]))

#My laptop is weird and parallel doesn't work for me. Switch back to Parallel to actually test it.
run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=time_trial_tracker, trials=10)
# Attempt at making schemas extensible - quite hacky right now
# run.track_duration()
#run.exec()


config = {
    "G" : ["montgomery"],
    "budget": [i for i in range(100,1001,10)],
    "transmission_rate": [0.078],
    "compliance_rate": [1],
    "global_rate":  [1],         #ranges are not currently inclusive of upper bound
    "discovery_rate": [1],
    "snitch_rate":  [1],
    #QUESTION: For taking the cartesian product, would unnecessary computation happen since snitch+discovery rates should be the same?
    "from_cache": cache_paths,
    "agent_name": ["Degree"]
}
config["G"] = [load_graph(x) for x in config["G"]]

run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=time_trial_tracker, trials=10)
run.exec()


#%%
