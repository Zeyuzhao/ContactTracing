import random
import math
import numpy as np
import networkx as nx

from .round import D_prime
from .utils import min_exposed_objective, pct_to_int
from .simulation import *
#from .problem2 import *
from .problem_label import *
from .problem import *


def NoIntervention(state: InfectionState):
    return set()


def Random(state: InfectionState):
    return set(random.sample(state.V1, min(state.budget, len(state.V1))))


def Random_label(state: InfectionState):
    if (state.policy == "none"):
        return Random(state)

    # Distribute budget across age groups
    state.set_budget_labels()

    quarantine = set()
    for label in state.labels:
        V1_label = set(
            node for node in state.V1 if state.G.nodes[node]["age_group"] == label)
        quarantine = quarantine.union(
            set(random.sample(V1_label, min((state.budget_labels[label]), len(V1_label)))))
    return quarantine


def EC(state: InfectionState):

    eigens: List[Tuple[int, int]] = []

    for u in state.V1:
        eigens.append((state.G.centrality[u], u))

    eigens.sort(reverse=True)
    return {i[1] for i in eigens[:state.budget]}


def DegGreedy_fair(state: InfectionState):
    """
    state.budget_labels: a mapping from id -> budget (does support None budgets)

    """
    if not state.transmission_known:
        P, Q = pq_independent(state.G, state.SIR.I2, state.V1,
                              state.V2, state.Q, state.transmission_rate)
    else:
        P, Q = state.P, state.Q

    weights: List[Tuple[int, int]] = []

    if state.compliance_known:
        for u in state.V1:
            w_sum = sum([Q[u][v]
                        for v in state.G.neighbors(u) if v in state.V2])
            weights.append(
                (state.G.nodes[u]['compliance_rate']*P[u]*(w_sum), u))
    else:
        for u in state.V1:
            w_sum = sum([Q[u][v]
                        for v in state.G.neighbors(u) if v in state.V2])
            weights.append((state.P[u] * (w_sum), u))

    weights.sort(reverse=True)

    # weights: (greedy_val, node_id) sorted in descending order
    if (state.policy == "none"):
        return {i[1] for i in weights[:state.budget]}

    quarantine = set()
    state.set_budget_labels()
    quarantine = pair_greedy(weights, state.budget_labels,
                             state.budget, (lambda x: state.G.nodes[x]["age_group"]))
    return quarantine

# Fairness


def DepRound_fair(state: InfectionState):
    state.set_budget_labels()

    problem2 = MinExposedLP2_label(state)
    problem2.solve_lp()
    probabilities = problem2.get_variables()

    if state.policy == "none":
        rounded = D_prime(np.array(probabilities))
        return set([problem2.quarantine_map[k] for (k, v) in enumerate(rounded) if v == 1])

    rounded = np.array([0 for i in range(len(probabilities))])
    for label in state.labels:
        partial_prob = [probabilities[k] if state.G.nodes[problem2.quarantine_map[k]]["age_group"] == label else 0 for k in
                        range(len(probabilities))]
        rounded = rounded + D_prime(np.array(partial_prob))

    return set([problem2.quarantine_map[k] for (k, v) in enumerate(rounded) if v == 1])


def SAA_Diffusion(state: InfectionState, debug=False, num_samples=10):
    problem = MinExposedSAADiffusion(state, num_samples=num_samples)
    problem.solve_lp()
    probabilities = problem.get_variables()
    rounded = D_prime(np.array(probabilities))

    action = set([problem.quarantine_map[k]
                 for (k, v) in enumerate(rounded) if v == 1])
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
            new_value = min(abs(probabilities[index]), abs(
                probabilities[index] - 1))
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


def segmented_greedy(state: InfectionState, split_pcts=[0.75, 0.25], alloc_pcts=[.25, .75], carry=True, rng=np.random, DEBUG=False):
    """
    TODO: NEED TO REWRITE!!!
    pcts are ordered from smallest degree to largest degree
    split_pcts: segment size percentages
    alloc_pcts: segment budget percentages

    Overflow Mechanic: the budget may exceed the segment size.
    We fill from right to left 
    (greater chance of overflow: larger degree usually have fewer members but higher budget), 
    and excess capacity is carried over to the next category.
    """
    if not math.isclose(1, sum(split_pcts)):
        raise ValueError(
            f"split_pcts '{split_pcts}' sum to {sum(split_pcts)}, not 1")
    if not math.isclose(1, sum(alloc_pcts)):
        raise ValueError(
            f"alloc_pcts '{alloc_pcts}' sum to {sum(alloc_pcts)}, not 1")

    budget = state.budget
    G = state.G
    split_amt = pct_to_int(len(state.V1), split_pcts)
    alloc_amt = pct_to_int(budget, alloc_pcts)

    v1_degrees = [(n, G.degree(n)) for n in state.V1]
    v1_sorted = [n for n, d in sorted(v1_degrees, key=lambda x: x[1])]

    v1_segments = np.split(v1_sorted, np.cumsum(split_amt[:-1]))

    overflow = 0
    samples = []
    for segment, amt in reversed(list(zip(v1_segments, alloc_amt))):
        # Overflow is carried over to the next segment
        segment_budget = amt
        if carry:
            segment_budget += overflow

        # Compute overflow
        if segment_budget > len(segment):
            overflow = segment_budget - len(segment)
            segment_budget = len(segment)
        else:
            overflow = 0

        sample = rng.choice(segment, segment_budget, replace=False)
        samples.extend(sample)

        if DEBUG:
            print(f"{segment_budget} / {len(segment)} (overflow: {overflow})")
            print("segment: ", segment)
            print("sample: ", sample)
            print("--------------")
            assert len(samples) <= budget
    return samples
