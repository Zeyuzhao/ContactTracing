#%%
from joblib import Memory

cachedir = "cache"
memory = Memory(cachedir, verbose=0)

@memory.cache
def f(x):
    print(f'Running f({x})')
    return x
#%%
print(f(1))

# %%
print(f(2))
# %%
print(f(1))
# %%
