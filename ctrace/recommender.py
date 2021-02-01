import random
import math
import numpy as np
import networkx as nx

from .round import D_prime
from .utils import pq_independent, find_excluded_contours, min_exposed_objective
from .simulation import *
from .problem import MinExposedLP

def NoIntervention(state: SimulationState):
    return set()


def Random(state: SimulationState):
    return set(random.sample(state.SIR_known.V1, min(state.SIR_known.budget, len(state.SIR_known.V1))))


# TODO: Test code! (Removed set)
def Degree(state: SimulationState):
    info = state.SIR_known
    
    degrees: List[Tuple[int, int]] = []
    for u in info.V1:
        count = sum([1 for v in info.G.neighbors(u) if v in info.V2])
        degrees.append((count, u))
        
    degrees.sort(reverse=True)
    return {i[1] for i in degrees[:info.budget]}


# TODO: Test code! V2 -> set V2
def DegGreedy(state: SimulationState):
    info = state.SIR_known
    P, Q = pq_independent(info.G, info.SIR.I, info.V1, info.transmission_rate)
    
    weights: List[Tuple[int, int]] = []
    for u in info.V1:
        w_sum = sum([Q[u][v] for v in info.G.neighbors(u) if v in info.V2]) # V2 is a set!
        weights.append((P[u] * w_sum, u))

    weights.sort(reverse=True)
    return {i[1] for i in weights[:info.budget]}


def DepRound(state: SimulationState):
    
    problem = MinExposedLP(state.SIR_known)
    
    problem.solve_lp()
    probabilities = problem.get_variables()
    rounded = D_prime(np.array(probabilities))

    # TODO: Maybe include objective value? Or delete the next section
    # sets variables so objective function value is correct
#    for i in range(len(rounded)):
#        problem.set_variable(i, rounded[i])

#    problem.solve_lp()

    return set([k for (k,v) in enumerate(rounded) if v==1])


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
