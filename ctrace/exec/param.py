# %%
import itertools
import random
import pickle
import hashlib
import networkx as nx
import json
import pytest

from typing import *
from dataclasses import dataclass, field, make_dataclass, InitVar
from pathlib import Path
from ctrace import PROJECT_ROOT
from ctrace.utils import load_graph
from collections import namedtuple

SIR_Tuple = namedtuple("SIR_Tuple", ["S", "I", "R"])


@dataclass
class SIR():
    S: Set[int] = field(default_factory=set)
    I: Set[int] = field(default_factory=set)
    R: Set[int] = field(default_factory=set)


def md5_hash_obj(obj) -> str:
    md5 = hashlib.md5()
    f = pickle.dumps(obj)
    md5.update(f)
    return md5.hexdigest()


def load_sir_path(path: Path, merge=True):
    with open(path) as file:
        # json to dict
        data = json.load(file)
        if merge:
            data["I"] = list(set().union(*data["I_Queue"]))
            del data["I_Queue"]
        # dict to sir_tuple
        return SIR_Tuple(data["S"], data["I"], data["R"])


@dataclass
class ParamBase:
    name: str = field(init=False, repr=True)
    data: Any = field(init=False, repr=False)


@dataclass
class Param(ParamBase):
    name: str = field(repr=True)
    data: Any = field(init=True, repr=False)


@dataclass
class LambdaParam(ParamBase):
    data: Any = field(init=True, repr=False)

    def __post_init__(self):
        self.name = self.data.__name__


@dataclass
class FileParam(ParamBase):
    """
    Class for keeping track of files as parameters
    (Abstract)
    Required: name
    Optional: path
    """
    name: str = field(repr=True)
    # Path relative to PROJECT_ROOT
    path: Optional[Union[str, Path]] = field(repr=True, default=None)

    # Generated Parameters
    data: Any = field(init=False, repr=False)
    # md5 hash of pickled data (could swap with directory hash???)
    obj_hash: str = field(init=False, repr=True)

    # Parameterized functions
    def finder(self) -> Union[str, Path]:
        # self.name -> path
        raise NotImplementedError

    def loader(self) -> Any:
        # self.path -> data object
        raise NotImplementedError

    def hasher(self) -> str:
        # self.data -> md5 hash
        return md5_hash_obj(self.data)

    def signature(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "obj_hash": self.obj_hash,
        }

    def __post_init__(self):
        if self.path is None:
            self.path = self.finder()
        self.data = self.loader()
        self.obj_hash = self.hasher()


@dataclass
class GraphParam(FileParam):
    """
    Class for keeping track of graph files as parameters

    Required: name
    Optional: path
    """
    # Parameterized functions

    def finder(self) -> Union[str, Path]:
        # name -> path
        return PROJECT_ROOT / ("data") / "graph" / self.name

    def loader(self) -> Any:
        # path -> data object
        return load_graph(self.name, path=PROJECT_ROOT / self.path)


@dataclass
class SIRParam(FileParam):
    """
    Class for keeping track of graph files as parameters

    Required: name
    Optional: parent OR file path
    """
    parent: Optional[GraphParam] = field(repr=True, default=None)

    # Parameterized functions
    def finder(self) -> Union[str, Path]:
        # self.name, self.parent -> Path
        return self.parent.path / "sir_cache" / f"{self.name}.json"

    def loader(self) -> Any:
        # self.path -> SIR_Tuple object
        return load_sir_path(self.path)

    def signature(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            # Assign an object id? # This is a foriegn key
            "parent": self.parent.name if self.parent else None,
            "obj_hash": self.obj_hash,
        }

    def __post_init__(self):
        if self.path is None and self.parent is None:
            raise ValueError(
                "Must need specified path or parent to resolve file path")
        super().__post_init__()


# Hash is consistant across runs
def test_hash_consistancy():
    assert GraphParam('montgomery').obj_hash == GraphParam(
        'montgomery').obj_hash


def test_graph_param_signature():
    param = GraphParam('montgomery')
    sig = param.signature()
    assert sig["name"] == 'montgomery'
    assert isinstance(sig["path"], Path)  # path rooted at project_root
    assert len(sig["obj_hash"]) == 32  # md5 has 32 hex digits


def test_sir_param_value_error():
    with pytest.raises(RuntimeError) as excinfo:
        _ = SIRParam('t7')
    assert "resolve" in str(excinfo.value)


# %%
