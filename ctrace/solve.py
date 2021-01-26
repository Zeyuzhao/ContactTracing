from collections import namedtuple
import math
import random
from typing import Tuple, Set, List

from .constraint import *
from .utils import find_excluded_contours, pq_independent, max_neighbors
from .recommender import *

from io import StringIO
import sys



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