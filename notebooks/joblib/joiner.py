#%%
import os
from multiprocessing import Process, Manager
from joblib import Parallel, delayed
from pathlib import Path
import functools
import time
def save(queue):
    with open(Path('') / 'json' / 'log.csv', "w") as out:
        while True:
            row = queue.get()
            if row is None: break
            out.write(row + '\n')
            out.flush()
            print(f'Writing: {row}')

def time_tracker(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        val = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        val['runtime'] = run_time
        return val
    return wrapper_timer


@time_tracker
def func(x):
    time.sleep(1)
    return {'id': str(os.getpid()), 'square': x**2}

# out_schema needs to match func
out_schema = ['id', 'square', 'runtime']

def worker(x, queue):
    print(f'[{x}] started')
    results = func(x)
    # Checks that results conform to out_schema
    queue.put(",".join(str(results[k]) for k in out_schema))
    print(f'[{x}] finished')


m = Manager()
queue = m.Queue()
p = Process(target=save, args=(queue,)) 
p.start()
Parallel(n_jobs=-1)(delayed(worker)(i, queue) for i in range(100))
queue.put(None) # terminate save loop
p.join()
            
# %%
