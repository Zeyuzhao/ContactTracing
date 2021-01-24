import numpy as np

from .problem import *

#returns rounded bits and objective value of those bits
def basic_non_integer_round(problem: MinExposedLP):
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
