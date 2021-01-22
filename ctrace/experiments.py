from time import perf_counter

from .dataset import load_sir
from .solve import *
from .utils import min_exposed_objective, indicatorToSet

TrackerInfo = namedtuple("TrackerInfo", ['value', 'sol', 'isOptimal', 'maxD', 'I_size', 'v1_size', 'v2_size', 'num_cross_edges'])
def time_trial_tracker(G: nx.graph, I0, safe, cost_constraint, p=.5, method="dependent"):
    """
    Runs to_quarantine and tracks various statistics. Used in conjunction with GridExecutor
    (GridExecutorParallel or GridExecutorLinear) to track run statistics.
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
    value, sol, isOptimal, maxD, I_size, v1_size, v2_size, num_cross_edges
    value: the MinExposed objective value (expected number of people exposed)
    sol: an dictionary mapping from V1 IDs to its indicator variables
    isOptimal: (-1, 0, 1) -> (does not apply, false, true)

    # Statistics
    maxD: the maximum number of neighbors of V1 that are in V2
    I_size: size of I
    v1_size: size of V_1
    v2_size: size of V_2
    num_cross_edges: number of edges between v1 and v2
    """
    costs = np.ones(len(G.nodes))
    V_1, V_2 = find_excluded_contours(G, I0, safe)
    P, Q = PQ_deterministic(G, I0, V_1, p)
    maxD = max_neighbors(G, V_1, V_2)

    if method == "weighted":
        obj_val, sol, info = weighted_solver(G, I0, P, Q, V_1, V_2, cost_constraint, costs)
        return TrackerInfo(obj_val, sol, -1, maxD, *info)

    elif method == "dependent":
        # Dependent LP Rounding
        prob = ProbMinExposed(G, I0, V_1, V_2, P, Q, cost_constraint, costs, solver="GUROBI")
        obj_val, sol = basic_non_integer_round(prob)
        return TrackerInfo(obj_val, sol, -1, maxD, len(prob.I), len(prob.V1), len(prob.V2), prob.num_cross_edges)

    elif method == "dependent_scip":
        prob = ProbMinExposed(G, I0, V_1, V_2, P, Q, cost_constraint, costs)
        obj_val, sol = basic_non_integer_round(prob)
        return TrackerInfo(obj_val, sol, -1, maxD, len(prob.I), len(prob.V1), len(prob.V2), prob.num_cross_edges)
    elif method == "gurobi":
        # Gurobi MIP Rounding
        prob = ProbMinExposedMIP(G, I0, V_1, V_2, P, Q, cost_constraint, costs, solver='GUROBI')
        prob.solve_lp()
        # Returns a tuple for its optimal value
        obj_val = prob.objective_value
        sol = prob.quarantined_solution
        isOptimal = prob.is_optimal
        return TrackerInfo(obj_val, sol, isOptimal, maxD, len(prob.I), len(prob.V1), len(prob.V2), prob.num_cross_edges)
    else:
        raise Exception("invalid method for optimization")


return_params = ['I_size', 'v1_size', 'v2_size', 'num_cross_edges', 'maxD', 'mip_value', 'min_exposed_value', 'duration', 'v1_objective', 'greedy_overlap']
TimeTrialExtendTrackerInfo = namedtuple("TrackerInfo", return_params)
def time_trial_extended_tracker(G: nx.graph, p, budget, method, from_cache, **kwargs):
    """
    Runs to_quarantine and tracks various statistics
    Parameters
    ----------
    G
    p
    budget
    method
    p
    method
    from_cache

    Returns
    -------
    min_exposed_value: MinExposed objective value (expected number of people exposed)
    mip_value: the MinExposed LP objective value
    greedy_intersection: the percentage of quarantined members shared with weighted greedy

    # Statistics
    maxD: the maximum number of neighbors of V1 that are in V2
    I_size: size of I
    v1_size: size of V_1
    v2_size: size of V_2
    num_cross_edges: number of edges between v1 and v2
    duration: the time it took to execute the method specified
    """
    SIR = load_sir(from_cache, merge=True)
    infected = SIR["I"]
    recovered = SIR["R"]

    costs = np.ones(len(G.nodes))
    contour1, contour2 = find_excluded_contours(G, infected, recovered)
    P, Q = PQ_deterministic(G, infected, contour1, p)
    maxD = max_neighbors(G, contour1, contour2)

    # The constant value contour1 contributes to the objective value
    v1_objective = sum(P[u] for u in contour1)

    # start time
    weighted_start = perf_counter()
    _, weighted_solution = weighted_solver(G, infected, P, Q, contour1, contour2, budget, costs)
    # end time
    weighted_end = perf_counter()

    if method == "greedy_weighted":
        prob = ProbMinExposed(G, infected, contour1, contour2, P, Q, budget, costs)
        for k, v in weighted_solution.items():
            prob.set_variable_id(k, v)
        prob.solve_lp()

        min_exposed_value = min_exposed_objective(G, (_, infected, recovered), (contour1, contour2), p, weighted_solution)
        return TimeTrialExtendTrackerInfo(
            len(infected),
            len(contour1),
            len(contour2),
            prob.num_cross_edges,
            maxD,
            prob.objective_value,
            min_exposed_value,
            weighted_end - weighted_start,
            v1_objective,
            -1,
        )
    elif method == "greedy_degree":
        _, method_solution = degree_solver(G, contour1, contour2, budget)
        prob = ProbMinExposed(G, infected, contour1, contour2, P, Q, budget, costs)

        # TODO: Quick hack for finding the MIP Objective Value
        for k, v in method_solution.items():
            prob.set_variable_id(k, v)
        prob.solve_lp()
        mip_value = prob.objective_value
        # Returns: mip_value, method_solution
    elif method == "random":
        _, method_solution = random_solver(contour1, budget)

        prob = ProbMinExposed(G, infected, contour1, contour2, P, Q, budget, costs)

        # TODO: Quick hack for finding the MIP Objective Value
        for k, v in method_solution.items():
            prob.set_variable_id(k, v)
        prob.solve_lp()
        mip_value = prob.objective_value
    elif method == "dependent":
        # Dependent LP Rounding
        prob = ProbMinExposed(G, infected, contour1, contour2, P, Q, budget, costs, solver="GUROBI_LP")
        mip_value, method_solution = basic_non_integer_round(prob)
        # Returns mip_value and method_solution

    elif method == "dependent_scip":
        prob = ProbMinExposed(G, infected, contour1, contour2, P, Q, budget, costs)
        mip_value, method_solution = basic_non_integer_round(prob)

    elif method == "mip_gurobi":
        # Gurobi MIP Rounding
        prob = ProbMinExposedMIP(G, infected, contour1, contour2, P, Q, budget, costs, solver='GUROBI')
        prob.solve_lp()
        # Returns a tuple for its optimal value
        mip_value = prob.objective_value
        method_solution = prob.quarantined_solution
    else:
        raise Exception("invalid method for optimization")

    method_end = perf_counter()
    # Round method solution?
    greedy_intersection = len(indicatorToSet(method_solution) & indicatorToSet(weighted_solution))
    # TODO: Encapsulate G, (_, infected, recovered), (contour1, contour2)
    min_exposed_value = min_exposed_objective(G, (_, infected, recovered), (contour1, contour2), p, method_solution)

    return TimeTrialExtendTrackerInfo(
        len(infected),
        len(contour1),
        len(contour2),
        prob.num_cross_edges,
        maxD,
        mip_value,
        min_exposed_value,
        method_end - weighted_end,
        v1_objective,
        greedy_intersection
    )
