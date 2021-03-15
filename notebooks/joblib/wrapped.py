#%%
import sys
import time
import traceback
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects

def func_async(i, *args):
    return 2 * i

print(Parallel(n_jobs=2)(delayed(func_async)(21) for _ in range(1))[0])
# %%

#%% 
# Large list

large_list = list(range(1000000))
t_start = time.time()
print(Parallel(n_jobs=2)(delayed(func_async)(21, large_list) for _ in range(1))[0])
print(f"With loky backend and cloudpickle serialization {time.time()-t_start:.3f}s")
# %%

with parallel_backend('multiprocessing'):
    t_start = time.time()
    Parallel(n_jobs=2)(delayed(func_async)(21, large_list) for _ in range(1))
    print(f"With multiprocessing backend and pickle serialization {time.time()-t_start:.3f}s")
# %%
with parallel_backend('multiprocessing'):
    t_start = time.time()
    Parallel(n_jobs=2)(
        delayed(func_async)(21, large_list) for _ in range(1))
    print("With multiprocessing backend and pickle serialization: {:.3f}s"
            .format(time.time() - t_start))
# %%
