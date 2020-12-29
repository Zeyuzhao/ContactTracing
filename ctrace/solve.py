import random
import numpy as np
import math
from constraint import *
from contact_tracing import *

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
    to_return = D_prime(np.array(probabilities))
    
    #sets variables so objective function value is correct
    for i in range(len(to_return)):
        problem.setVariable(i,to_return[i])
    
    problem.solve_lp()
    
    return (problem.objectiveVal, to_return)

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
    
    return (problem.objectiveVal, probabilities)

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
    
    return (problem.objectiveVal, probabilities)

def to_quarantine(G: nx.graph, I0, cost_constraint, method = "dependent"):
    costs = np.ones(len(G.nodes))
    V_1, V_2 = find_contours(G, I0)
    P, Q = PQ(G, I0, runs = 100)
    
    prob = ProbMinExposed.from_dataframe(G, I0, V_1, V_2, P, Q, cost_constraint, COSTS)
    
    val = -1
    rounded = []
    
    if method == "dependent":
        (val, rounded) = basic_non_integer_round(prob)
    elif method == "iterated":
        (val, rounded) = iterated_round(prob, len(V_1)/20)
    elif method == "optimized":
        (val, rounded) = optimized_iterated_round(prob, len(V_1)/20)
    else:
        raise Exception("invalid method for optimization")
    
    return (val, rounded)