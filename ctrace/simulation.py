import json
import random
from enum import Enum, IntEnum

import EoN
import networkx as nx
import numpy as np

from typing import Set
from collections import namedtuple

# from .utils import find_excluded_contours
from ctrace import PROJECT_ROOT
import gym
from gym.utils import seeding
from gym import spaces

SIR_Tuple = namedtuple("SIR_Tuple", ["S", "I", "R"])

#%%
class SIR(IntEnum):
    S = 0
    I = 1
    R = 2
#%%

# TODO: Create wrapper that corrupts Q
# TODO: Create wrapper that tracks history?
# TODO: Create wrapper that masks observation space
class InfectionEnv(gym.Env):
    def __init__(self, G: nx.Graph):
        self.G = G
        self.N = len(self.G)

        # How stale the information the agent receives
        self.stale_steps = 1
        self.action_space = spaces.MultiBinary(self.N)
        self.observation_space = spaces.Dict({
            'sir': spaces.MultiDiscrete([3] * self.N),  # An indicator variable for every vertex
            'contours': spaces.MultiDiscrete([self.stale_steps + 1] * self.N)  # Vertices in V1 and V2 to consider
        })
        self.SIR_History = None
        self.time_step = None

        # Ensure SIR_Queue is filled and initialized
        self.reset()

    def reset(self, delay: int = 5, clusters: int = 3):
        if delay < self.stale_steps:
            raise ValueError("Delay must be longer than stale time")

        # The state of infection over all time steps
        self.SIR_History = []

        # Always one less than SIR_History (Actual sim time, not agent time)
        self.time_step = -1
        for i in range(delay):
            obs, _, _, _ = self.step(np.zeros(self.N))  # Let infection advance ahead delay steps
        return obs

    def step(self, action):
        # TODO: Shift "quarantined" elements from SIR -> Q dict
        self.sir = []
        # Convert array to sets
        # TODO: Run EoN and obtain results
        # TODO: Unquarantine Q dict -> SIR
        # TODO: Store results into SIR_History

        self.time_step += 1

        # Retrieve SIR that is stale
        obs = {
            'sir': self.SIR_History[-(self.stale_steps + 1)],
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
        self.np_random, seed = seeding.np_random(seed)
        raise NotImplementedError
        return [seed]

def listToSet(l):
    d = {}
    raise NotImplementedError
    return {"S": [], "I": [], "R": []}