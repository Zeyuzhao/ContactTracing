from collections import namedtuple
import math
import random
from typing import Tuple, Set, List

from .constraint import *
from .utils import find_excluded_contours, PQ_deterministic, max_neighbors
from .round import *

from io import StringIO
import sys


#returns rounded bits and objective value of those bits
def basic_non_integer_round(problem: ProbMinExposed):
    problem.solve_lp()
    probabilities = problem.get_variables()
    rounded = D_prime(np.array(probabilities))
    
    #sets variables so objective function value is correct
    for i in range(len(rounded)):
        problem.set_variable(i, rounded[i])
    
    problem.solve_lp()
    
    return (problem.objective_value, problem.quarantined_solution)

#returns rounded bits and objective value of those bits
def iterated_round(problem: ProbMinExposed, d: int):
    problem.solve_lp()
    probabilities = np.array(problem.get_variables())
    
    curr = 0
    
    #rounds d values at the time, and then resolves the LP each time
    while curr + d < len(probabilities):
        
        probabilities[curr:curr+d] = D_prime(probabilities[curr:curr+d])
        
        for i in range(d):
            
            problem.set_variable(curr + i, probabilities[curr + i])
        
        problem.solve_lp()
        probabilities = np.array(problem.get_variables())
        
        curr += d
    
    #rounds remaining values and updates LP
    probabilities[curr:] = D_prime(probabilities[curr:])
    
    for i in range(curr,len(probabilities)):
        problem.set_variable(i, probabilities[i])
    
    problem.solve_lp()
    
    return (problem.objective_value, problem.quarantined_solution)

#returns rounded bits and objective value of those bits
def optimized_iterated_round(problem: ProbMinExposed, d: int):
    problem.solve_lp()
    probabilities = np.array(problem.get_variables())
    
    #creates mapping to avoid re-ordering of the array
    mapping = []
    
    for (i,value) in enumerate(probabilities):
        distance = min(abs(value),abs(value-1))
        mapping.append((distance,i))
    
    mapping.sort()
        
    while len(mapping) >= d:
        
        #rounds the most confident d values
        to_round = []
        
        for i in range(d):
            to_round.append(probabilities[mapping[i][1]])
        
        rounded = D_prime(np.array(to_round))
        
        for i in range(d):
            problem.set_variable(mapping[i][1], rounded[i])
        
        #resolves the LP under new constraints
        problem.solve_lp()
        probabilities = np.array(problem.get_variables())
        
        #updates the mappings; only need to worry about previously unrounded values
        mapping = mapping[d:]
        
        for (i,(value,index)) in enumerate(mapping):
            new_value = min(abs(probabilities[index]),abs(probabilities[index]-1))
            mapping[i] = (new_value,index)
            
        mapping.sort()
        
    
    #rounds all remaining (less than d) values
    to_round = []
    
    for (value, index) in mapping:
        to_round.append(probabilities[index])
        
    rounded = D_prime(np.array(to_round))
    
    for i in range(len(rounded)):
        problem.set_variable(mapping[i][1], rounded[i])
        probabilities[mapping[i][1]] = rounded[i]
    
    problem.solve_lp()
    
    return (problem.objective_value, problem.quarantined_solution)

#returns a map for which nodes to quarantine
def to_quarantine(G: nx.graph, I0, safe, cost_constraint, p=.5, method="dependent"):
    """

    Parameters
    ----------
    G
        Contact Tracing Graph
    I0
        Initial infected
    safe
        Recovered nodes that will not be infected
    cost_constraint
        The k value - the number of people to quarantine
    p

    method

    Returns
    -------
    (LP_SCORE, QUARANTINE_MAP)
        LP_SCORE - the nuumber of people exposed
        QUARANTINE_MAP - a dict[int, int] of a map of V1 IDs to 0-1 indicator variables
    """
    costs = np.ones(len(G.nodes))
    V_1, V_2 = find_excluded_contours(G, I0, safe)
    
    #put here to save run time
    if method == "none":
        sol = {u:0 for u in V_1}    
        return (-1, sol)
    # TODO: Add weighted degree
    elif method == "degree":
        return degree_solver(G, V_1, V_2, cost_constraint)

    elif method == "random":
        return random_solver(V_1, cost_constraint)

    P, Q = PQ_deterministic(G, I0, V_1, p)
    if method == "weighted":
        return weighted_solver(G, I0, P, Q, V_1, V_2, cost_constraint, costs)

    prob = ProbMinExposed(G, I0, V_1, V_2, P, Q, cost_constraint, costs)
    if method == "dependent":
        return basic_non_integer_round(prob)
    elif method == "iterated":
        return iterated_round(prob, int(len(V_1)/20))
    elif method == "optimized":
        return optimized_iterated_round(prob, int(len(V_1)/20))
    elif method == "gurobi":
        prob = ProbMinExposedMIP(G, I0, V_1, V_2, P, Q, cost_constraint, costs, solver='GUROBI')
        prob.solve_lp()
        return (prob.objective_value), prob.quarantined_solution
    elif method == "dependent_gurobi":
        prob = ProbMinExposed(G, I0, V_1, V_2, P, Q, cost_constraint, costs, solver='GUROBI')
        return basic_non_integer_round(prob)
    else:
        raise Exception("invalid method for optimization")


def weighted_solver(G, I0, P, Q, V_1, V_2, cost_constraint, costs):
    weights: List[Tuple[int, int]] = []
    for u in V_1:
        w_sum = 0
        for v in set(G.neighbors(u)):
            if v in V_2:
                w_sum += Q[u][v]
        weights.append((P[u] * w_sum, u))
    # Get the top k (cost_constraint) V1s ranked by w_u = p_u * sum(q_uv for v in v2)
    weights.sort(reverse=True)
    topK = weights[:cost_constraint]
    topK = {i[1] for i in topK}
    sol = {}
    for u in V_1:
        if u in topK:
            sol[u] = 1
        else:
            sol[u] = 0
    return -1, sol

def random_solver(V_1, cost_constraint):
    sample = random.sample(V_1, min(cost_constraint, len(V_1)))
    sol = {}
    for v in V_1:
        if v in sample:
            sol[v] = 1
        else:
            sol[v] = 0
    return (-1, sol)


def degree_solver(G, V_1, V_2, cost_constraint):
    # calculate degree of each vertex in V_1
    degrees = []
    for u in V_1:
        count = 0
        for v in set(G.neighbors(u)):
            if v in V_2:
                count += 1

        degrees.append((count, u))
    degrees.sort()
    degrees.reverse()
    sol = {}
    for i in range(len(V_1)):
        if i < cost_constraint:
            sol[degrees[i][1]] = 1
        else:
            sol[degrees[i][1]] = 0
    return (-1, sol)