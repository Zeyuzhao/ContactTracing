import concurrent.futures
import time
from simulation import *



def runner(args):
    """Example parallel function (only accepts one argument)"""
    a, b = args
    print(f'Doing {a} and {b}')
    return a + b

def parallel(func, args):
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(func, args)
        for result in results:
            print(result)
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} seconds')


if __name__ == '__main__':
    args = []
    parallel(runner, [(1, 1), (2, 3), (4, 4)])