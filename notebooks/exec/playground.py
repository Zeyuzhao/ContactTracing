#%%

# Experiment with pickle hashing
# Pickle -> hash w/ md5
import io
import pickle
import networkx as nx
import hashlib
from ctrace.utils import load_graph
from pathlib import Path
from typing import Union

# %%
g = load_graph('montgomery')



# %%
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

def md5_hash_obj(obj) -> str:
    BUF_SIZE = 65536
    md5 = hashlib.md5()

    # Init in-memory file to serialize object
    f = io.BytesIO()
    pickle.dump(obj, f)

    while True:
        data = f.read(BUF_SIZE)
        if not data:
            break
        md5.update(data)
    return md5.hexdigest()


# %%
md5_hash_mem(f)
# %%
