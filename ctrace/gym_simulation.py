# %%
import json
import random
import time

import EoN
import networkx as nx
import numpy as np
import gym
from gym.utils import seeding
from gym import spaces

from enum import IntEnum
from typing import Dict, Set, List, Any, TypeVar
from collections import UserList, namedtuple, defaultdict
from ctrace import PROJECT_ROOT


# %%

# Declare SIR Enum
class SIR:
    S = 1
    I = 2
    R = 3


# %%
# TODO: Add testing?

# T = TypeVar('T', bound='PartitionSIR')


# class PartitionSIR(UserList):
#     def __init__(self, size=0):
#         # Stored internally as integers
#         self._types = ["S", "I", "R"]
#         self.type = IntEnum("type", zip(self._types, range(1,4)))
#         self.data = [0] * size

#     @classmethod
#     def from_list(cls, l):
#         p = PartitionSIR()
#         p.data = l.copy()
#         return p

#     @classmethod
#     def from_dict_letters(cls, n: int, d: Dict[int, str]) -> T:
#         mapper = {
#             "S": 1,
#             "I": 2,
#             "R": 3,
#         }
#         p = PartitionSIR(n)
#         for k, v in d.items():
#             p[k] = mapper[v]
#         return p

#     def __getitem__(self, item: int) -> int:
#         return self.data[item]

#     def __setitem__(self, key: int, value: int) -> None:
#         self.data[key] = value

#     # TODO: Properties hardcoded - hacky solution
#     @property
#     def S(self):
#         return (i for i, e in enumerate(self.data) if e == SIR.S)

#     @property
#     def I(self):
#         return (i for i, e in enumerate(self.data) if e == SIR.I)

#     @property
#     def R(self):
#         return (i for i, e in enumerate(self.data) if e == SIR.R)


# %%

# TODO: Create wrapper that corrupts Q
# TODO: Create wrapper that tracks history?
# TODO: Create wrapper that masks observation space
# TODO: Build a statistics tracker wrapper
# TODO: Create wrapper that tracks diffs in infection
# -> nodes can be infected longer than 1 timestep
class InfectionEnv(gym.Env):
    def __init__(self,
                 G: nx.Graph,
                 transmission_rate=0.1,
                 stale=1,
                 delay=5,
                 clusters=3,
                 ):
        self.G = G
        self.N = len(self.G)

        # Environment Parameters
        self.transmission_rate = transmission_rate
        self.stale = stale  # Delay of agent observation from real state

        # IO Schema
        self.action_space = spaces.MultiBinary(self.N)
        self.observation_space = spaces.Dict({
            # An indicator variable for every vertex
            'sir': spaces.MultiDiscrete([3] * self.N),
            # Vertices in V1 and V2 to consider
            'contours': spaces.MultiDiscrete([self.stale + 1] * self.N)
        })

        # State information
        self.SIR_History = []
        self.quarantine_history = []
        self.time_step = 0

        # Initialization parameters
        self.delay = delay  # Infection head start
        self.clusters = clusters

        # Initialize SIR_Queue
        self.reset()

    def reset(self):
        if self.delay < self.stale:
            raise ValueError("Delay must be longer than stale time")

        # The state of infection over all time steps
        self.SIR_History = []
        self.quarantine_history = []
        self.time_step = 0

        # Generate the initial clusters
        obs = np.zeros(self.N)
        seeds = np.random.choice(self.N, self.clusters, replace=False)
        obs[seeds] = 1
        self.SIR_History.append(obs)

        # Give the infection a head start
        # TODO: Refactor into wrapper
        self.time_step = 0
        no_op = [0] * self.N
        for i in range(self.delay):
            self._step(no_op)  # Let infection advance ahead delay steps
        return obs

    def step(self, action: List[int]):
        self._step(action)
        # Retrieve SIR that is stale
        obs = {
            'sir': self.SIR_History[-(self.stale + 1)],
            'contours': [0] * self.N,  # Contours are ignored
            'time': self.time_step - self.delay,
        }

        I_count = obs['sir'].count(SIR.I)
        reward = I_count  # Number of people in infected
        # TODO: Test done condition!!!
        done = (I_count == 0) or self.time_step > self.total_time
        # Info for tracking progress of simulation
        info = {}
        return obs, reward, done, info

    def _step(self, action: List[int]) -> None:
        # Retrieve current state.
        partition = PartitionSIR.from_list(self.SIR_History[-1])

        # Action -> a integer boolean array of quarantine members
        # Map quarantine members to its state
        quarantine_dict = {i: partition[i] for i in action if action[i] == 1}

        # Move members into R (temporarily)
        for q in quarantine_dict:
            partition[q] = SIR.R  # 2

        # Run simulation
        full_data = EoN.basic_discrete_SIR(
            G=self.G,
            p=self.transmission_rate,
            initial_infecteds=list(partition.I),
            initial_recovereds=list(partition.R),
            tmin=0,
            tmax=1,
            return_full_data=True
        )
        # Advance quarantined
        # TODO: Extend to Multi-step quarantine (keep track of infection batches)
        for q, status in quarantine_dict.items():
            # I -> R
            if status == SIR.I:
                quarantine_dict[q] = SIR.R
            # S -> S (nothing)

        result_partition = PartitionSIR.from_dict_letters(
            self.N, full_data.get_statuses(time=1))

        # Move quarantine back into graph (undo the R state)
        for q, status, in quarantine_dict.items():
            result_partition[q] = status

        # Store results into SIR_History
        self.SIR_History.append(np.array(result_partition.data))
        self.time_step += 1

    def seed(self, seed=42):
        # Set seeds???
        np.random.seed(seed)
        random.seed(seed)

    def render(self, mode="human"):
        # Create an infection env for grid infection?
        raise NotImplementedError


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

if __name__ == '__main__':
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
