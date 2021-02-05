#%%
import json
import random
from enum import Enum, IntEnum

import EoN
import networkx as nx
import numpy as np

from typing import Dict, Set, List, Any, TypeVar
from collections import UserList, namedtuple, defaultdict

# from .utils import find_excluded_contours
from ctrace import PROJECT_ROOT
import gym
from gym.utils import seeding
from gym import spaces

SIR_Tuple = namedtuple("SIR_Tuple", ["S", "I", "R"])


# %%

# Declare SIR Enum
class SIR:
    S = 0
    I = 1
    R = 2


#%%

from collections import UserList
# TODO: Add testing?

T = TypeVar('T', bound='PartitionSIR')
class PartitionSIR(UserList):
    def __init__(self, size=0):
        # Stored internally as integers
        self._types = ["S", "I", "R"]
        self.type = IntEnum("type", zip(self._types, range(3)))
        self.data = [0] * size

    @classmethod
    def from_list(cls, l):
        p = PartitionSIR()
        p.data = l.copy()
        return p
    
    @classmethod
    def from_dict_letters(cls, n: int, d: Dict[int, str]) -> T:
        mapper = {
            "S": 0,
            "I": 1,
            "R": 2,
        }
        p = PartitionSIR(n)
        for k, v in d.items():
            p[k] = mapper[v]

    def __getitem__(self, item: int) -> int:
        return self.data[item]
    
    def __setitem__(self, key: int, value: int) -> None:
        self.data[key] = value

    # TODO: Properties hardcoded - hacky solution
    @property
    def S(self):
        return (i for i, e in enumerate(self.data) if e == 0)
    @property
    def I(self):
        return (i for i, e in enumerate(self.data) if e == 1)
    @property
    def R(self):
        return (i for i, e in enumerate(self.data) if e == 2)


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
        self.SIR_History = []
        self.quarantine_history = []
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
        self.quarantine_history = []
        self.time_step = 0

        # Generate the initial clusters
        obs = np.zeros(self.N)
        seeds = np.random.choice(self.N, self.clusters, replace=False)
        obs[seeds] = 1
        self.SIR_History.append(obs)

        # Give the infection a head start
        self.time_step = 0
        no_op = [0] * self.N
        for i in range(self.delay):
            self._step(no_op) # Let infection advance ahead delay steps
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
            initial_infecteds=partition.I,
            initial_recovereds=partition.R,
            tmin=0,
            tmax=1,
            return_full_data=True
        )
        # Advance quarantined
        for q, status in quarantine_dict.items():
            # I -> R
            if status == SIR.I:
                quarantine_dict[q] = SIR.R
            # S -> S (nothing)
        
        result_partition = PartitionSIR.from_dict_letters(self.N, 
        full_data.get_statuses(time=1))

        # Move quarantine back into graph (undo the R state)
        for q, status, in quarantine_dict.items():
            result_partition[q] = status
        
        # Store results into SIR_History
        self.SIR_History.append(result_partition.data)
        self.time_step += 1


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
