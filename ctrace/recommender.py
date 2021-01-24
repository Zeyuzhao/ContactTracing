import random
import numpy as np

from .problem import *

def random(V_1, cost_constraint):
    sample = random.sample(V_1, min(cost_constraint, len(V_1)))
    sol = {}
    for v in V_1:
        if v in sample:
            sol[v] = 1
        else:
            sol[v] = 0
    return (-1, sol)

def degree(G, V_1, V_2, cost_constraint):
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

def weighted(G, I0, P, Q, V_1, V_2, cost_constraint, costs):
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

#returns rounded bits and objective value of those bits
def dependent(problem: MinExposedLP):
    problem.solve_lp()
    probabilities = problem.get_variables()
    rounded = D_prime(np.array(probabilities))
    
    #sets variables so objective function value is correct
    for i in range(len(rounded)):
        problem.set_variable(i, rounded[i])
    
    problem.solve_lp()
    
    return (problem.objective_value, problem.quarantined_solution)

#returns rounded bits and objective value of those bits
def iterated(problem: ProbMinExposed, d: int):
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
def optimized(problem: ProbMinExposed, d: int):
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
