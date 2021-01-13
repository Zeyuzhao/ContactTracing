import math
import random
from typing import Tuple, Set, List

from .constraint import *
from .utils import find_excluded_contours, PQ_deterministic, max_neighbors

from io import StringIO
import sys

def simplify(alpha:float, beta:float):
    if (alpha<0) | (alpha>1) | (beta<0) | (beta>1):
        raise ValueError('Invalid alpha or beta')
    
    p = random.uniform(0,1)
    
    if alpha+beta==0:
        
        return (0,0,-1,-1)
    
    elif alpha+beta<1:
        
        if p<alpha/(alpha+beta):
            return (-1,0,alpha+beta,-1)
        else:
            return (0,-1,-1,alpha+beta)
        
    elif alpha+beta==1:
        
        if p<alpha:
            return (1,0,-1,-1)
        else:
            return (0,1,-1,-1)
        
    elif alpha+beta < 2:
        
        if p<(1-beta)/(2-alpha-beta):
            return (1,-1,-1,alpha+beta-1)
        else:
            return (-1,1,alpha+beta-1,-1)
        
    else:
        
        return (1,1,-1,-1)
    
    
def D(p):
    t = len(p)
    sample = np.full(t,-1)
    prob = np.array(p,copy=True)
    
    leaves = np.arange(t)
    np.random.shuffle(leaves)
    
    #each iteration represents a level in the tree
    while t > 1:
        
        new_leaves = []
        
        for i in range(0,t-1,2):            
            
            a = prob[leaves[i]]
            b = prob[leaves[i+1]]
            
            (x,y,new_a,new_b) = simplify(a,b)
            
            if x==-1:
                
                new_leaves.append(leaves[i])
                prob[leaves[i]] = new_a
                sample[leaves[i+1]] = y
                
            elif y==-1:
                
                new_leaves.append(leaves[i+1])
                prob[leaves[i+1]] = new_b
                sample[leaves[i]] = x

            else:
                
                new_leaves.append(leaves[i])
                prob[leaves[i]] = x
                sample[leaves[i+1]] = y
    
        if t%2 == 1:

            new_leaves.append(leaves[-1])
            
        t = len(new_leaves)
        leaves = np.array(new_leaves)
        np.random.shuffle(leaves)

    if t==1:
        
        p = random.uniform(0,1)
        
        if p < prob[leaves[0]]:
            sample[leaves[0]] = 1
        else:
            sample[leaves[0]] = 0
       
    return sample

def D_prime(p):
    l = np.sum(p)
    
    p_prime = np.append(p,[math.ceil(l)-l])
    
    return D(p_prime)[:len(p)]

#returns rounded bits and objective value of those bits
def basic_non_integer_round(problem: ProbMinExposed):
    problem.solve_lp()
    probabilities = problem.getVariables()
    rounded = D_prime(np.array(probabilities))
    
    #sets variables so objective function value is correct
    for i in range(len(rounded)):
        problem.setVariable(i,rounded[i])
    
    problem.solve_lp()
    
    return (problem.objectiveVal, problem.quarantined_solution)

#returns rounded bits and objective value of those bits
def iterated_round(problem: ProbMinExposed, d: int):
    problem.solve_lp()
    probabilities = np.array(problem.getVariables())
    
    curr = 0
    
    #rounds d values at the time, and then resolves the LP each time
    while curr + d < len(probabilities):
        
        probabilities[curr:curr+d] = D_prime(probabilities[curr:curr+d])
        
        for i in range(d):
            
            problem.setVariable(curr+i, probabilities[curr+i])
        
        problem.solve_lp()
        probabilities = np.array(problem.getVariables())
        
        curr += d
    
    #rounds remaining values and updates LP
    probabilities[curr:] = D_prime(probabilities[curr:])
    
    for i in range(curr,len(probabilities)):
        problem.setVariable(i,probabilities[i])
    
    problem.solve_lp()
    
    return (problem.objectiveVal, problem.quarantined_solution)

#returns rounded bits and objective value of those bits
def optimized_iterated_round(problem: ProbMinExposed, d: int):
    problem.solve_lp()
    probabilities = np.array(problem.getVariables())
    
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
            problem.setVariable(mapping[i][1], rounded[i])
        
        #resolves the LP under new constraints
        problem.solve_lp()
        probabilities = np.array(problem.getVariables())
        
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
        problem.setVariable(mapping[i][1],rounded[i])
        probabilities[mapping[i][1]] = rounded[i]
    
    problem.solve_lp()
    
    return (problem.objectiveVal, problem.quarantined_solution)

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
        # Returns a tuple for its optimal value
        return (prob.objectiveVal, prob.isOptimal), prob.quarantined_solution
    elif method == "dependent_gurobi":
        prob = ProbMinExposed(G, I0, V_1, V_2, P, Q, cost_constraint, costs, solver='GUROBI_LP')
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
    # Compute LP MinExposed Score
    # TODO: Migrate to specialized method?
    prob = ProbMinExposed(G, I0, V_1, V_2, P, Q, cost_constraint, costs)
    for k, v in sol.items():
        prob.setVariableId(k, v)
    prob.solve_lp()
    return (prob.objectiveVal, sol)


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


def trial_tracker(G: nx.graph, I0, safe, cost_constraint, p=.5, method="dependent"):
    """
    Runs to_quarantine and tracks various statistics
    Parameters
    ----------
    G
    I0
    safe
    cost_constraint
    p
    method

    Returns
    -------
    value, sol, isOptimal, maxD
    value: the MinExposed objective value (expected number of people exposed)
    sol: an dictionary mapping from V1 IDs to its indicator variables
    isOptimal: (-1, 0, 1) -> (does not apply, false, true)
    maxD: the maximum number of neighbors of V1 that are in V2
    """
    costs = np.ones(len(G.nodes))
    V_1, V_2 = find_excluded_contours(G, I0, safe)
    P, Q = PQ_deterministic(G, I0, V_1, p)
    maxD = max_neighbors(G, V_1, V_2)

    if method == "weighted":
        obj_val, sol = weighted_solver(G, I0, P, Q, V_1, V_2, cost_constraint, costs)
        return (obj_val, sol, -1, maxD)

    if method == "dependent":
        # Dependent LP Rounding
        prob = ProbMinExposed(G, I0, V_1, V_2, P, Q, cost_constraint, costs)
        obj_val, sol = basic_non_integer_round(prob)
        return (obj_val, sol, -1, maxD)
    elif method == "gurobi":
        # Gurobi MIP Rounding
        prob = ProbMinExposedMIP(G, I0, V_1, V_2, P, Q, cost_constraint, costs, solver='GUROBI')
        prob.solve_lp()
        # Returns a tuple for its optimal value
        obj_val = prob.objectiveVal
        sol = prob.quarantined_solution
        isOptimal = prob.isOptimal
        return (obj_val, sol, isOptimal, maxD)
    else:
        raise Exception("invalid method for optimization")



