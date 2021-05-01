import random
import math
import numpy as np
import networkx as nx

from .round import D_prime
from .utils import pq_independent_edges, find_excluded_contours_edges, min_exposed_objective
from .simulation import *
from .problem2 import *
from .problem import *

def NoIntervention(state: InfectionState):
    return set()


def Random(state: InfectionState):
    return set(random.sample(state.V1, min(state.budget, len(state.V1))))


def Degree(state: InfectionState):
    degrees: List[Tuple[int, int]] = []
    V2_only = state.V2-state.V1
    for u in state.V1:
        count = sum([1 for v in state.G.neighbors(u) if v in V2_only])
        degrees.append((count, u))
        
    degrees.sort(reverse=True)
    return {i[1] for i in degrees[:state.budget]}

#Accounts for edges between V1
def Degree2(state: InfectionState):
    degrees: List[Tuple[int, int]] = []
    for u in state.V1:
        count = sum([1 for v in state.G.neighbors(u) if v in state.V2])
        degrees.append((count, u))
        
    degrees.sort(reverse=True)
    return {i[1] for i in degrees[:state.budget]}

# TODO: Test code! V2 -> set V2
def DegGreedy(state: InfectionState):
    #P, Q = pq_independent(info.G, info.SIR.I, info.V1, info.transmission_rate[info.time_stage])
    P, Q = pq_independent_edges(state.G, state.SIR.I2, state.V1, state.V2)
    
    #only weighs V2 not in V1
    V2_only = state.V2-state.V1
    weights: List[Tuple[int, int]] = []
    for u in state.V1:
        w_sum = sum([Q[u][v] for v in state.G.neighbors(u) if v in V2_only]) # V2 is a set!
        weights.append((P[u] * w_sum, u))

    weights.sort(reverse=True)
    return {i[1] for i in weights[:state.budget]}

#Accounts for edges between V1
def DegGreedy2(state: InfectionState):
    P, Q = pq_independent_edges(state.G, state.SIR.I2, state.V1, state.V2)
    #P, Q = pq_independent(state.G, state.SIR.I2, state.V1, state.V2, state.transmission_rate)
    
    weights: List[Tuple[int, int]] = []
    for u in state.V1:
        w_sum = sum([Q[u][v]*(1-P[v]) for v in state.G.neighbors(u) if v in state.V2]) # V2 is a set!
        weights.append((P[u] * (w_sum), u))

    weights.sort(reverse=True)
    return {i[1] for i in weights[:state.budget]}

def DegGreedy2_comp(state: InfectionState):
    P, Q = pq_independent_edges(state.G, state.SIR.I2, state.V1, state.V2)
    #P, Q = pq_independent(state.G, state.SIR.I2, state.V1, state.V2, state.transmission_rate)
    
    weights: List[Tuple[int, int]] = []
    for u in state.V1:
        w_sum = sum([Q[u][v]*(1-P[v]) for v in state.G.neighbors(u) if v in state.V2]) # V2 is a set!
        weights.append((state.G[u]["compliance"]*(P[u] * (w_sum))**2, u))

    weights.sort(reverse=True)
    return {i[1] for i in weights[:state.budget]}

def DepRound(state: InfectionState):
    
    problem = MinExposedLP(state)
    problem.solve_lp()
    probabilities = problem.get_variables()
    rounded = D_prime(np.array(probabilities))

    return set([problem.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])

def DepRound2(state: InfectionState):
    
    problem2 = MinExposedLP2(state)
    problem2.solve_lp()
    probabilities = problem2.get_variables()
    rounded = D_prime(np.array(probabilities))

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
