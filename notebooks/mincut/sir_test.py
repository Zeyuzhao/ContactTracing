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

# Declare SIR Enum
class SIR:
    S = 1
    I = 2
    R = 3


# %%
# TODO: Add testing?

T = TypeVar('T', bound='PartitionSIR')

class PartitionSIR(UserList):
    def __init__(self, size=0):
        # Stored internally as integers
        self._types = ["S", "I", "R"]
        self.type = IntEnum("type", self._types)
        self.data = [SIR.S] * size

    @classmethod
    def from_list(cls, l):
        p = PartitionSIR()
        p.data = l.copy()
        return p

    @classmethod
    def from_dict_letters(cls, n: int, d: Dict[int, str]) -> T:
        mapper = {
            "S": SIR.S,
            "I": SIR.I,
            "R": SIR.R,
        }
        p = PartitionSIR(n)
        for k, v in d.items():
            p[k] = mapper[v]
        return p

    def __getitem__(self, item: int) -> int:
        return self.data[item]

    def __setitem__(self, key: int, value: int) -> None:
        self.data[key] = value

    @property
    def S(self):
        return set(i for i, e in enumerate(self.data) if e == SIR.S)

    @property
    def I(self):
        return set(i for i, e in enumerate(self.data) if e == SIR.I)

    @property
    def R(self):
        return set(i for i, e in enumerate(self.data) if e == SIR.R)

# %%
