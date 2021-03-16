#%%

from ctrace import PROJECT_ROOT
import hashlib
import networkx as nx

from typing import *
from dataclasses import dataclass, field, make_dataclass, InitVar
from pathlib import Path
import itertools
import random


@dataclass
class SIR:
    S: Set[int] = field(repr=False)
    I: Set[int] = field(repr=False)
    R: Set[int] = field(repr=False)
    length: int = field(default=0)
    
def load_graph(fp: Union[str, Path]) -> nx.DiGraph():
    g = nx.grid_2d_graph(10,10)
    return g

def load_sir(fp: Union[str, Path]) -> SIR:
    return SIR({0,1}, {2,3}, {3,4})

def md5_hash(fp: Union[str, Path]) -> str:
    BUF_SIZE = 65536
    md5 = hashlib.md5()
    with open(fp, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()

class GraphEq(nx.Graph):
    def __eq__(self, other):
        return self.nodes == other.nodes \
        and self.edges == other.edges

@dataclass
class FileParam:
    """Class for keeping track of file parameters"""
    name: str = field(repr=True)
    file_path: Union[str, Path] = field(repr=True)
    loader: InitVar[Callable[[str], Any]] # Parameterized by loader

    # Generated Parameters
    data: Any = field(init=False, repr=True)
    file_hash: str = field(init=False, repr=True)

    def __post_init__(self):
        self.data = self.loader(self.file_path)
        self.file_hash = md5_hash(self.file_path) # Change hasher to object?

@dataclass
class SirParam(FileParam):
    """Class for tracking SIR params"""
    loader: load_sir

@dataclass
class GraphParam(FileParam):
    """Class for keeping track of file parameters"""
    name: str = field(repr=True)
    data: Any = field(init=False, repr=False)
    file_path: Union[str, Path] = field(repr=True)
    file_hash: str = field(init=False, repr=True)

    def __post_init__(self):
        self.data = load_graph(self.file_path)
        # Temp Hack
        self.file_hash = md5_hash(self.file_path / 'data.txt')

#%%


# Desired schema
in_schema = [
    ('graph', GraphParam),
    ('sir', SirParam),
    ('rate1', float),
    ('rate2', float),
    ('seed', int)
]

graph_root = PROJECT_ROOT / 'data' / 'graphs'
sir_root = PROJECT_ROOT / 'data' / 'SIR_Cache'
compact = {
    'graph': [
        GraphParam('montgomery', graph_root / 'montgomery'), # Allow for specification of name
    ],
    'sir': [
        SirParam('m1', sir_root / 't7.json'),
        SirParam('m2', sir_root / 't8.json'),
    ],
    'rate1': [0.1 * x for x in range(10)],
    'rate2': [0.01 * x for x in range(10)],
    'seed': [21, 42, 101, 9000]
}
# TODO: Validate compact schema.

tasks = []

# Check that schema matches
assert set(compact.keys()) == {x[0] for x in in_schema}
new_tasks = [dict(zip(compact, x)) for x in itertools.product(*compact.values())]
# TODO: Add seeds and id 

compact = {
    'graph': [
        GraphParam('alpine', graph_root / 'alpine'),
    ],
    'sir': [
        SirParam('a1', sir_root / 'a8.json'),
        SirParam('a2', sir_root / 'a9.json'),
    ],
    'rate1': [0.1 * x for x in range(10)],
    'rate2': [0.01 * x for x in range(10)],
    'seed': [21, 42, 101, 9000],
}
assert set(compact.keys()) == {x[0] for x in in_schema}
new_tasks = [dict(zip(compact, x)) for x in itertools.product(*compact.values())]


# lambda function accepts InSchema objects
# Must attach id object






# %%

# %%
