import random
import math
import numpy as np
import networkx as nx

from .round import D_prime
from .utils import min_exposed_objective
from .simulation import *
#from .problem2 import *
from .problem_label import *
from .problem import *

def NoIntervention(state: InfectionState):
    return set()

def Random(state: InfectionState):
    return set(random.sample(state.V1, min(state.budget, len(state.V1))))

def Random_label(state: InfectionState):
    if (state.policy == "none"): return Random(state)
    
    #Distribute budget across age groups
    state.set_budget_labels()
    
    quarantine = set()
    for label in state.labels:
        V1_label = set(node for node in state.V1 if state.G.nodes[node]["age_group"]==label)
        quarantine = quarantine.union(set(random.sample(V1_label, min((state.budget_labels[label]), len(V1_label)))))
    return quarantine

def EC(state: InfectionState):
    
    eigens: List[Tuple[int, int]] = []

    for u in state.V1:
        eigens.append((state.G.centrality[u], u))
    
    eigens.sort(reverse=True)
    return {i[1] for i in eigens[:state.budget]}

'''def Degree(state: InfectionState):
    degrees: List[Tuple[int, int]] = []
    V2_only = state.V2-state.V1
    for u in state.V1:
        count = sum([1 for v in state.G.neighbors(u) if v in V2_only])
        degrees.append((count, u))
        
    degrees.sort(reverse=True)
    return {i[1] for i in degrees[:state.budget]}'''

#Accounts for edges between V1
'''def Degree2(state: InfectionState):
    degrees: List[Tuple[int, int]] = []
    for u in state.V1:
        count = sum([1 for v in state.G.neighbors(u) if v in state.V2])
        degrees.append((count, u))
        
    degrees.sort(reverse=True)
    return {i[1] for i in degrees[:state.budget]}'''

# TODO: Test code! V2 -> set V2
'''def DegGreedy(state: InfectionState):
    #P, Q = pq_independent(info.G, info.SIR.I, info.V1, info.transmission_rate[info.time_stage])
    #P, Q = pq_independent_edges(state.G, state.SIR.I2, state.V1, state.V2)
    
    #only weighs V2 not in V1
    V2_only = state.V2-state.V1
    weights: List[Tuple[int, int]] = []
    for u in state.V1:
        w_sum = sum([state.Q[u][v] for v in state.G.neighbors(u) if v in V2_only]) # V2 is a set!
        weights.append((state.P[u] * w_sum, u))

    weights.sort(reverse=True)
    return {i[1] for i in weights[:state.budget]}'''

#Only knows average compliance, assumes uniformity
'''def DegGreedy2_avg_compliance(state: InfectionState):
    
    weights: List[Tuple[int, int]] = []
    for u in state.V1:
        w_sum = sum([state.Q[u][v]*(1-state.P[v]) for v in state.G.neighbors(u) if v in state.V2]) # V2 is a set!
        weights.append((state.P[u] * (w_sum), u))
    
    weights.sort(reverse=True)
    return {i[1] for i in weights[:state.budget]}

#Average transmission rate
def DegGreedy2_avg_transmission(state: InfectionState):
    P, Q = pq_independent(state.G, state.SIR.I2, state.V1, state.V2, state.transmission_rate)
    
    weights: List[Tuple[int, int]] = []
    for u in state.V1:
        w_sum = sum([Q[u][v]*(1-P[v]) for v in state.G.neighbors(u) if v in state.V2 and state.Q[u][v]!=0]) # V2 is a set!
        weights.append((state.G.nodes[u]['compliance_rate']*(P[u] * (w_sum))**2, u))

    weights.sort(reverse=True)
    return {i[1] for i in weights[:state.budget]}

def DegGreedy2_avg_both(state: InfectionState):
    P, Q = pq_independent(state.G, state.SIR.I2, state.V1, state.V2, state.transmission_rate)
    
    weights: List[Tuple[int, int]] = []
    for u in state.V1:
        w_sum = sum([Q[u][v]*(1-P[v]) for v in state.G.neighbors(u) if v in state.V2 and state.Q[u][v]!=0]) # V2 is a set!
        weights.append((state.P[u] * (w_sum), u))

    weights.sort(reverse=True)
    return {i[1] for i in weights[:state.budget]}

def DegGreedy2(state: InfectionState):
    if not state.compliance_known and not state.transmission_known: return DegGreedy2_avg_both(state)
    elif state.compliance_known and not state.transmission_known: return DegGreedy2_avg_transmission(state)
    elif not state.compliance_known and state.transmission_known: return DegGreedy2_avg_compliance(state)
    
    weights: List[Tuple[int, int]] = []
    for u in state.V1:
        w_sum = sum([state.Q[u][v]*(1-state.P[v]) for v in state.G.neighbors(u) if v in state.V2]) # V2 is a set!
        weights.append((state.G.nodes[u]['compliance_rate']*(state.P[u] * (w_sum))**2, u))

    weights.sort(reverse=True)
    return {i[1] for i in weights[:state.budget]}'''

#Fairness
<<<<<<< HEAD
'''def DegGreedy2_fair(state: InfectionState):
=======
def DegGreedy2_fair(state: InfectionState):
>>>>>>> 5c28841870cbd94b5a8811b8e8a51d32f5d6917c
    if not state.transmission_known:
        P, Q = pq_independent(state.G, state.SIR.I2, state.V1, state.V2, state.Q, state.transmission_rate)
    else:
        P, Q = state.P, state.Q
    
    weights: List[Tuple[int, int]] = []
    
    if state.compliance_known:
        for u in state.V1:
            w_sum = sum([Q[u][v]*(1-P[v]) for v in state.G.neighbors(u) if v in state.V2])
            #weights.append((state.G.nodes[u]['compliance_rate']*(P[u] * (w_sum))**2, u))
            weights.append((state.G.nodes[u]['compliance_rate']*P[u]*(w_sum), u))
    else:
        for u in state.V1:
            w_sum = sum([Q[u][v]*(1-P[v]) for v in state.G.neighbors(u) if v in state.V2])
            weights.append((state.P[u] * (w_sum), u))
    
    weights.sort(reverse=True)
    if (state.policy == "none"):
        return {i[1] for i in weights[:state.budget]}
    
    quarantine = set()
    state.set_budget_labels()
    for label in state.labels:
        deg = [tup for tup in weights if state.G.nodes[tup[1]]["age_group"]==label]
        quarantine = quarantine.union({i[1] for i in deg[:min(state.budget_labels[label], len(deg))]})
<<<<<<< HEAD
    return quarantine'''

def DegGreedy_fair(state: InfectionState):
    if not state.transmission_known:
        P, Q = pq_independent(state.G, state.SIR.I2, state.V1, state.V2, state.Q, state.transmission_rate)
    else:
        P, Q = state.P, state.Q
    
    weights: List[Tuple[int, int]] = []
    
    if state.compliance_known:
        for u in state.V1:
            w_sum = sum([Q[u][v] for v in state.G.neighbors(u) if v in state.V2])
            weights.append((state.G.nodes[u]['compliance_rate']*P[u]*(w_sum), u))
    else:
        for u in state.V1:
            w_sum = sum([Q[u][v] for v in state.G.neighbors(u) if v in state.V2])
            weights.append((state.P[u] * (w_sum), u))
    
    weights.sort(reverse=True)
    if (state.policy == "none"):
        return {i[1] for i in weights[:state.budget]}
    
    quarantine = set()
    state.set_budget_labels()
    for label in state.labels:
        deg = [tup for tup in weights if state.G.nodes[tup[1]]["age_group"]==label]
        quarantine = quarantine.union({i[1] for i in deg[:min(state.budget_labels[label], len(deg))]})
=======
>>>>>>> 5c28841870cbd94b5a8811b8e8a51d32f5d6917c
    return quarantine

'''def DepRound(state: InfectionState):
    
    problem = MinExposedLP(state)
    problem.solve_lp()
    probabilities = problem.get_variables()
    rounded = D_prime(np.array(probabilities))

    return set([problem.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])'''

#Average compliance (uniformity)
'''def DepRound2_avg_compliance(state: InfectionState):
    
    problem2 = MinExposedLP2(state)
    problem2.solve_lp()
    probabilities = problem2.get_variables()
    rounded = D_prime(np.array(probabilities))

    return set([problem2.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])

#Only knows average transmission
def DepRound2_avg_transmission(state: InfectionState):
    
    problem2 = MinExposedLP2(state, comp = True, bad=True)
    problem2.solve_lp()
    probabilities = problem2.get_variables()
    rounded = D_prime(np.array(probabilities))

    return set([problem2.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])

def DepRound2_avg_both(state: InfectionState):
    
    problem2 = MinExposedLP2(state, comp = False, bad=True)
    problem2.solve_lp()
    probabilities = problem2.get_variables()
    rounded = D_prime(np.array(probabilities))

    return set([problem2.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])

#Nonuniform compliances
def DepRound2(state: InfectionState):
    if not state.compliance_known and not state.transmission_known: return DepRound2_avg_both(state)
    elif state.compliance_known and not state.transmission_known: return DepRound2_avg_transmission(state)
    elif not state.compliance_known and state.transmission_known: return DepRound2_avg_compliance(state)
    
    problem2 = MinExposedLP2(state, comp=True)
    problem2.solve_lp()
    probabilities = problem2.get_variables()
    rounded = D_prime(np.array(probabilities))

    return set([problem2.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])'''

#Fairness
<<<<<<< HEAD
def DepRound_fair(state: InfectionState):
=======
def DepRound2_fair(state: InfectionState):
>>>>>>> 5c28841870cbd94b5a8811b8e8a51d32f5d6917c
    state.set_budget_labels()
    
    problem2 = MinExposedLP2_label(state)
    problem2.solve_lp()
    probabilities = problem2.get_variables()
    
    if state.policy == "none":
        rounded = D_prime(np.array(probabilities))
        return set([problem2.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])
    
    rounded = np.array([0 for i in range(len(probabilities))])
    for label in state.labels:
        partial_prob = [probabilities[k] if state.G.nodes[problem2.quarantine_map[k]]["age_group"]==label else 0 for k in 
                        range(len(probabilities))]
        rounded = rounded + D_prime(np.array(partial_prob))
    
    return set([problem2.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])

def SAA_Diffusion(state: InfectionState, debug=False, num_samples=10):
    problem = MinExposedSAADiffusion(state, num_samples=num_samples)
    problem.solve_lp()
    probabilities = problem.get_variables()
    rounded = D_prime(np.array(probabilities))

    action = set([problem.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])
    if debug:
        return {
            "problem": problem,
            "action": action,
        }
    return action

# returns rounded bits and objective value of those bits
def iterated(problem: MinExposedLP, d: int):
    problem.solve_lp()
    probabilities = np.array(problem.get_variables())

    curr = 0

    # rounds d values at the time, and then resolves the LP each time
    while curr + d < len(probabilities):

        probabilities[curr:curr + d] = D_prime(probabilities[curr:curr + d])

        for i in range(d):
            problem.set_variable(curr + i, probabilities[curr + i])

        problem.solve_lp()
        probabilities = np.array(problem.get_variables())

        curr += d

    # rounds remaining values and updates LP
    probabilities[curr:] = D_prime(probabilities[curr:])

    for i in range(curr, len(probabilities)):
        problem.set_variable(i, probabilities[i])

    problem.solve_lp()

    return (problem.objective_value, problem.quarantined_solution)


# returns rounded bits and objective value of those bits
def optimized(problem: MinExposedLP, d: int):
    problem.solve_lp()
    probabilities = np.array(problem.get_variables())

    # creates mapping to avoid re-ordering of the array
    mapping = []

    for (i, value) in enumerate(probabilities):
        distance = min(abs(value), abs(value - 1))
        mapping.append((distance, i))

    mapping.sort()

    while len(mapping) >= d:

        # rounds the most confident d values
        to_round = []

        for i in range(d):
            to_round.append(probabilities[mapping[i][1]])

        rounded = D_prime(np.array(to_round))

        for i in range(d):
            problem.set_variable(mapping[i][1], rounded[i])

        # resolves the LP under new constraints
        problem.solve_lp()
        probabilities = np.array(problem.get_variables())

        # updates the mappings; only need to worry about previously unrounded values
        mapping = mapping[d:]

        for (i, (value, index)) in enumerate(mapping):
            new_value = min(abs(probabilities[index]), abs(probabilities[index] - 1))
            mapping[i] = (new_value, index)

        mapping.sort()

    # rounds all remaining (less than d) values
    to_round = []

    for (value, index) in mapping:
        to_round.append(probabilities[index])

    rounded = D_prime(np.array(to_round))

    for i in range(len(rounded)):
        problem.set_variable(mapping[i][1], rounded[i])
        probabilities[mapping[i][1]] = rounded[i]

    problem.solve_lp()

    return (problem.objective_value, problem.quarantined_solution)
