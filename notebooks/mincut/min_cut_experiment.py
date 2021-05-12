# %%
import networkx as nx
from ctrace.min_cut import *
from ctrace.drawing import *
from ctrace.utils import *
from ctrace.problem import *
from ctrace.recommender import *
from ctrace.dataset import *
from ctrace.simulation import *
from ctrace.exec.runner import GridExecutorLinear, GridExecutorParallel
%load_ext autoreload

%autoreload 2
# %%


def max_degree_edges(G, edges, budget):
    edge_max_degree = {(u, v): max(G.degree(u), G.degree(v))
                       for (u, v) in edges}
    edges_by_degree = sorted(
        edge_max_degree, key=edge_max_degree.get, reverse=True)
    return edges_by_degree[:budget]


def degree_solver(G, SIR, budget):
    edges = max_degree_edges(G, G.edges(SIR.I), budget)
    return edges


def random_solver(G, SIR, budget, seed=None):
    edges = np.random.choice(G.edges(SIR.I), size=budget, replace=False)
    return edges


def randomize(sir, p=0.05, seed=None):
    """
    Flips between S (1) and I (2) with probability p.
    """
    out = sir.copy()
    for n, status in enumerate(sir):
        rand = np.random.rand()
        if (status == SIR.S or status == SIR.I) and rand < p:
            out[n] = 3 - status
    return out

# %%

methods = {
    "LP": min_cut_round,
    "greedy": degree_solver,
    "random": random_solver,
}


def runner(
    G,
    num_infected: int,
    transmission: float,
    rand_resp_prob: float,
    budget: int,
    method: str,
    seed=None,
    logging_dir=None,
):
    # Sample the edges that actually transmit the disease
    active_edges = set(uniform_sample(G.edges, transmission))
    G_transmit = nx.subgraph_view(
        G, filter_edge=lambda x, y: (x, y) in active_edges)

    actual_sir = random_init(G, num_infected=num_infected, seed=seed)
    visible_sir = randomize(actual_sir, p=rand_resp_prob, seed=seed)

    solver = methods[method]
    edge_rec = solver(G, visible_sir, budget=budget)

    # Evaluation
    dict(list(edge_rec.items())[0:2])
    vertex_soln, edge_soln = min_cut_solver(
        G_transmit,
        actual_sir,
        budget=budget,
        partial=edge_rec,
        mip=False
    )
    score = sum(vertex_soln.values())
    return score


config = {
    "G": ['montgomery'],
    "num_infected": [100, 200, 300, 400, 500],
    "transmission": [0.1],
    "rand_resp_prob": [0.1, 0.2, 0.3, 0.4],
    "budget": [500, 600, 700, 800],
    "method": ["LP", "greedy", "random"],
}


config["G"] = [load_graph(g) for g in config["G"]]
in_schema = list(config.keys())
out_schema = ["total_infected"]
TrackerInfo = namedtuple("TrackerInfo", out_schema)

run = GridExecutorLinear.init_multiple(
    config, in_schema, out_schema, func=runner, trials=5)

# print(f"First 5 trials: {run.expanded_config[:10]}")
print(f"Number of trials: {len(run.expanded_config)}")
# %%
run.track_duration()
# 40 sample size -> 20 workers
# 20 sample size -> 60
run.exec()

exit()

# %%

# %%
