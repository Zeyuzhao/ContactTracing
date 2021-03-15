from typing import Dict, List, Callable
from collections import NamedTuple

class MultiExecutor():
    def __init__(
        self, 
        config: Dict, 
        in_schema: List[str], 
        out_schema: List[str], 
        func: Callable[..., NamedTuple]
    ):
        pass

