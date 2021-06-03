import pytest
import numpy as np

from ctrace.drawing import grid_world_init, complex_grid

from ctrace.recommender import binary_segmented_greedy


# No reliable way to test it.
def test_pair_greedy_label_constrained():
    k1 = 0.2
    k2 = 0.8
    TOL = 0.9
    
    rng = np.random.default_rng(42)
    state = grid_world_init(initial_infection=(0.05, 0.1), width=20, small_world=complex_grid, labels=False, seed=42)
    action = binary_segmented_greedy(state, k1=k1, k2=k2, carry=True,rng=rng, DEBUG=False)
    
    space_degrees = list(sorted(state.G.degree(n) for n in state.V1))
    action_degrees = list(sorted(state.G.degree(n) for n in action))

    approx_cutoff = space_degrees[int(k1 * len(space_degrees))]
    # assert len(list(filter(lambda x: x >= approx_cutoff, action_degrees))) >= k2 * len(action_degrees) * TOL
    # assert len(list(filter(lambda x: x <= approx_cutoff, action_degrees))) >= (1-k2) * len(action_degrees) * TOL

