import json
import random
from enum import Enum, IntEnum

import EoN
import networkx as nx
import numpy as np

from typing import Set, List
from collections import namedtuple, defaultdict

# from .utils import find_excluded_contours
from ctrace import PROJECT_ROOT
import gym
from gym.utils import seeding
from gym import spaces

SIR_Tuple = namedtuple("SIR_Tuple", ["S", "I", "R"])


# %%

class SIR(IntEnum):
    S = 0
    I = 1
    R = 2


# %%

# TODO: Create wrapper that corrupts Q
# TODO: Create wrapper that tracks history?
# TODO: Create wrapper that masks observation space
# TODO: Build a statistics tracker wrapper
class InfectionEnv(gym.Env):
    def __init__(self, G: nx.Graph):
        self.G = G
        self.N = len(self.G)

        # Environment Parameters
        self.transmission_rate = 0.078
        self.stale = 1  # Delay of agent observation from real state

        # IO Schema
        self.action_space = spaces.MultiBinary(self.N)
        self.observation_space = spaces.Dict({
            'sir': spaces.MultiDiscrete([3] * self.N),  # An indicator variable for every vertex
            'contours': spaces.MultiDiscrete([self.stale + 1] * self.N)  # Vertices in V1 and V2 to consider
        })

        # State information
        self.SIR_History = None
        self.time_step = 0

        # Initialization parameters
        self.delay = 5  # Infection head start
        self.clusters = 3

        # Initialize SIR_Queue
        self.reset()

    def reset(self):
        if self.delay < self.stale:
            raise ValueError("Delay must be longer than stale time")

        # The state of infection over all time steps
        self.SIR_History = []

        # Generate the initial clusters
        obs = np.zeros(self.N)
        seeds = np.random.choice(self.N, self.clusters, replace=False)
        obs[seeds] = 1
        self.SIR_History.append(obs)

        # Give the infection a head start
        self.time_step = 0
        for i in range(self.delay):
            obs, _, _, _ = self.step([0] * self.N) # Let infection advance ahead delay steps
        return obs

    def step(self, action: List[int]):
        # Retrieve current state.
        state = self.SIR_History[-1].copy()
        q = [i for i in action if action[i] == 1]

        # Quarantine
        quarantine_dict = {i: state[i] for i in q}

        # Run simulation
        full_data = EoN.basic_discrete_SIR(
            G=self.G,
            p=self.transmission_rate,
            initial_infecteds=sir[SIR.I],
            initial_recovereds=sir[SIR.R],
            tmin=0,
            tmax=1,
            return_full_data=True
        )
        # TODO: Unquarantine Q dict -> SIR
        # TODO: Store results into SIR_History

        self.time_step += 1

        # Retrieve SIR that is stale
        obs = {
            'sir': self.SIR_History[-(self.stale + 1)],
            'contours': [0] * self.N  # Contours are ignored
        }

        reward = 0  # Difference in infected people?
        done = False  # If time steps reached or number infected?
        # Info for tracking progress of simulation
        info = {}
        return obs, reward, done, info

    def _quarantine(self):
        raise NotImplementedError

    def _unquarantine(self):
        raise NotImplementedError

    def seed(self, seed=None):
        # Set seeds???
        raise NotImplementedError

    def render(self, mode="human"):
        # Create an infection env for grid infection?
        raise NotImplementedError


# %%

class PartitionSet(List):
    """Provide fast iteration, but slower membership?"""
    def __init__(self, size, attrs):
        super().__init__()
        self.Types = IntEnum("_", attrs)

    def __getitem__(self, item):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

class Partition(List):
    def __init__(self, attrs, size=0):
        # Stored internally as integers
        pass

# %%
# TODO: Add testing
def listToSets(l):
    d = defaultdict(list)
    for i, e in enumerate(l):
        d[e].append(i)
    return d


# TODO: Check if s forms partition?
def setsToList(s, n=0, default=None):
    if n == 0:
        n = sum([len(v) for _, v in s.items()])
    arr = [default] * n
    for k, v in s.items():
        for x in v:
            arr[x] = k
    return arr


# %%
import time

#
labels = np.random.randint(3, size=100000)
start = time.time()
sets = listToSets(labels)
end = time.time()
arr = setsToList(sets)
end2 = time.time()
print(f"arr to set: {end - start}")
print(f"set to arr: {end2 - end}")
print(f"Total Time: {end2 - start}")
# arr to set: 0.02387404441833496
# set to arr: 0.005036115646362305
# Total Time: 0.028910160064697266
