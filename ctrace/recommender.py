import random
import math
import numpy as np
import networkx as nx

from ctrace.round import D_prime
from ctrace.utils import pct_to_int
from ctrace.simulation import *
from ctrace.problem import *

def NoIntervention(state: InfectionState, extra = False):
    if extra == True:
        return {'action': set()}
    return set()


def Random(state: InfectionState, extra = False):
    action = set(random.sample(state.V1, min(state.budget, len(state.V1))))
    if extra == True:
        return {'action': action}
    return action

def Degree(state: InfectionState):
    degrees: List[Tuple[int, int]] = []
    for u in state.V1:
        count = sum([1 for v in state.G.neighbors(u) if v in state.V2 and state.Q[u][v]!=0])
        degrees.append((count, u))
        
    degrees.sort(reverse=True)
    return {i[1] for i in degrees[:state.budget]}

def SegDegree(state: InfectionState, k1=.2, k2=.8, carry=True,rng=np.random, DEBUG=False, extra=False):
    """
    k1 - top proportion of nodes classified as "high" degree
    k2 - proportion of budget assigned to "high" degree nodes
    carry - whether to assign surplus budget to under-constrained segments
    """
    budget = state.budget
    G = state.G
    v1_degrees = [(n, G.degree(n)) for n in state.V1]
    # Large to small
    v1_sorted = [n for n, d in sorted(v1_degrees, key=lambda x: x[1], reverse=True)]

    # Rounding transaction
    top_size = int(k1 * len(v1_sorted))
    top_budget = int(k2 * budget)

    bottom_size = len(v1_sorted) - top_size
    bottom_budget = budget - top_budget

    # Size invariant
    assert (top_size + bottom_size) == len(v1_sorted)
    assert (top_budget + bottom_budget) == budget


    sizes = [top_size, bottom_size]
    budgets = [top_budget, bottom_budget]
    budgets = segmented_allocation(sizes, budgets, carry=True)

    # Size constraint
    assert budgets[0] <= top_size
    assert budgets[1] <= bottom_size

    samples = []
    samples.extend(
        rng.choice(v1_sorted[:top_size], budgets[0], replace=False).tolist()
    )

    samples.extend(
        rng.choice(v1_sorted[top_size:], budgets[1], replace=False).tolist()
    )

    if extra:
        return {'action': samples}
    return samples

def DegGreedy_fair(state: InfectionState):
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
    return quarantine

def DepRound_fair(state: InfectionState):
    state.set_budget_labels()
    
    problem = MinExposedLP(state)
    problem.solve_lp()
    probabilities = problem.get_variables()
    
    if state.policy == "none":
        rounded = D_prime(np.array(probabilities))
        return set([problem.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])
    
    rounded = np.array([0 for i in range(len(probabilities))])
    for label in state.labels:
        partial_prob = [probabilities[k] if state.G.nodes[problem.quarantine_map[k]]["age_group"]==label else 0 for k in 
                        range(len(probabilities))]
        rounded = rounded + D_prime(np.array(partial_prob))

    return set([problem.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])

'''
For realistic model conditions where transmission rates are completely unknown.
The policymaker assumes transmission rates of 1.
'''
def DepRound_simplified(state: InfectionState):
    state.set_budget_labels()
    
    problem = MinExposedLP(state, simp = True)
    problem.solve_lp()
    probabilities = problem.get_variables()
    
    if state.policy == "none":
        rounded = D_prime(np.array(probabilities))
        return set([problem.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])
    
    rounded = np.array([0 for i in range(len(probabilities))])
    for label in state.labels:
        partial_prob = [probabilities[k] if state.G.nodes[problem.quarantine_map[k]]["age_group"]==label else 0 for k in 
                        range(len(probabilities))]
        rounded = rounded + D_prime(np.array(partial_prob))

    return set([problem.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])



def Milp(state: InfectionState, extra=False):
    state.set_budget_labels()

    problem2 = MinExposedIP(state, solver_id="GUROBI")
    problem2.solve_lp()
    indicators = problem2.get_variables()
    
    if extra:
        return {'action': set([problem2.quarantine_map[k] for (k, v) in enumerate(indicators) if v == 1]), 'is_optimal': problem2.is_optimal}
    return set([problem2.quarantine_map[k] for (k, v) in enumerate(indicators) if v == 1])