import random
import math
import numpy as np
import networkx as nx

from .round import D_prime
from .utils import pq_independent, find_excluded_contours, min_exposed_objective

#change them so it takes in a simulation_state parameter, preferably called state

def rand(state):
    v1, _ = find_excluded_contours(state.SIR_known.G, state.SIR_known.SIR.I, state.SIR_known.SIR.R)
    sample = random.sample(v1, min(state.SIR_real.budget, len(v1)))
    sol = {v for v in v1 if v in sample}
    return sol


def degree(state):
    v1, v2 = find_excluded_contours(state.SIR_known.G, state.SIR_known.SIR.I, state.SIR_known.SIR.R)
    degrees = []
    for u in v1:
        count = sum([1 for v in set(state.SIR_known.G.neighbors(u)) if v in v2])
        degrees.append((count, u))
    degrees.sort(reverse=True)
    sol = {degrees[i][1] for i in range(len(v1)) if i < state.SIR_real.budget}
    return sol


def weighted(state):
    v1, v2 = find_excluded_contours(state.SIR_known.G, state.SIR_known.SIR.I,
                                    state.SIR_known.SIR.R)  # Time impact of excluded_contours?
    P, Q = pq_independent(state.SIR_known.G, state.SIR_known.SIR.I, v1, state.SIR_known.transmission_rate)
    weights: List[Tuple[int, int]] = []
    for u in v1:
        w_sum = sum([Q[u][v] for v in set(state.SIR_known.G.neighbors(u)) if v in v2])
        weights.append((P[u] * w_sum, u))
    # Get the top k (cost_constraint) V1s ranked by w_u = p_u * sum(q_uv for v in v2)
    weights.sort(reverse=True)
    topK = {i[1] for i in weights[:state.SIR_real.budget]}
    sol = set([u for u in v1 if u in topK])
    return sol

"""
# returns rounded bits and objective value of those bits
def dependent(problem: MinExposedLP):
    problem.solve_lp()
    probabilities = problem.get_variables()
    rounded = D_prime(np.array(probabilities))

    # sets variables so objective function value is correct
    for i in range(len(rounded)):
        problem.set_variable(i, rounded[i])

    problem.solve_lp()

    return (problem.objective_value, problem.quarantined_solution)


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
    """