#%%
import time
from ctrace.utils import load_sir
from pathlib import Path
import json

(Path("") / 'json').mkdir(parents=True, exist_ok=True)
t1 = time.time()
sir = load_sir('t7.json')
print(f"Loading time: {time.time() - t1:.3f}")

t1 = time.time()
with open(Path("") / 'json' / 'test.json', "w+") as f:
    json.dump(sir, f)
print(f"Writing time: {time.time() - t1:.3f}")

# %%
