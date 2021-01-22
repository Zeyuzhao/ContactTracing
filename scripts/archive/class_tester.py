import concurrent.futures
import pickle

from tqdm import tqdm


class A:
    def __init__(self, name):
        self.name = name

    def runner(self, value):
        return f"{self.name} says: {value}"

    def exec(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            args = [1,2,3,4]
            results = [executor.submit(self.runner, arg) for arg in args]
            for f in tqdm(concurrent.futures.as_completed(results), total=len(args)):
                print(f.result())

a_test = A("Bob")

a_test.exec()
