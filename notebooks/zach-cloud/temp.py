from enum import IntEnum
from collections import UserList

SIR = IntEnum("type", ["S", "I", "R"])
SEIR = IntEnum("type", ["S", "E", "I", "R"])

class Partition(UserList):
    """
    An container representing a partition of integers 0..n-1 to classes 1..k
    Stored internally as an array.
    Supports querying as .[attr], where [attr] is specified in types
    Supports imports from integer list and list of sets
    """

    def __init__(self, other=None, size=0, enum_type=SIR):
        # Stored internally as integers
        self.data: List[int]

        if other is None:
            self.type = enum_type
            self._types = [e.name for e in enum_type]
            self.data = [1] * size
        else:
            self.type = other.type
            self._types = [e.name for e in enum_type]
            self.data = other.data.copy()

    @classmethod
    def from_list(cls, l, types=None):
        """
        Copies data from a list representation into Partition container
        """
        p = cls(size=len(l))
        p.data = l.copy()
        return p

    @classmethod
    def from_sets(cls, sets):
        """
        Import data from a list of set indices. Union of sets must be integers [0..len()-1]
        """
        assert len(sets) == len(self.type)
        p = cls(size=sum(map(len, sets)))
        for i, vals in enumerate(sets):
            for v in vals:
                self.data[v] = i
        return p

    def __getitem__(self, item: int) -> int:
        return self.data[item]

    def __setitem__(self, key: int, value: int) -> None:
        self.data[key] = value

    def __getattr__(self, attr):
        if attr in self._types:
            return set(i for i, e in enumerate(self.data) if e == self.type[attr])
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'. \nFilter attributes are: {self._types}")

    def to_dict(self):
        return {k: v for k, v in enumerate(self.data)}


class PartitionSIR(Partition):
    def __init__(self, other=None, size=0):
        return super(enum_type=SIR)

class PartitionSEIR(Partition):
    def __init__(self, other=None, size=0):
        return super(enum_type=SEIR)
