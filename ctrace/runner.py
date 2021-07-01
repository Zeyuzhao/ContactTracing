import concurrent.futures
import csv
from ctrace.utils import max_neighbors
import functools
import itertools
import logging
import time
from collections import namedtuple
from typing import Dict, Callable, List, Any, NamedTuple
import traceback

import shortuuid
import tracemalloc
from tqdm import tqdm

from ctrace import PROJECT_ROOT

DEBUG = False

def debug_memory(logger, label=""):
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    logger.debug(f"[{label}]: {top_stats[:5]}")


class GridExecutor():
    """
    Usage: Create a new GridExecutor with config, in_schema, out_schema and func.
    GridExecutor is an abstract class for running a cartesian product of lists of arguments.
    Input and output arguments specified by schemas are assumed to have pretty __str__.
    """
    def __init__(self, config: Dict, in_schema: List[str], out_schema: List[str], func: Callable[..., NamedTuple]):
        """
        Parameters
        ----------
        config
            A dictionary mapping string attributes to arrays of different parameters.
            Each item of the dictionary must be an array of arguments
        in_schema
            A list describing what and the order input attributes would be printed
        out_schema
            A list describing what and the order output attributes would be printed
        func
            A function to execute in parallel. Input arguments must match config keys.
            Output arguments must be a namedtuple. namedtuple must encompass all attributes in out_schema
        """
        self.compact_config = config.copy()

        # Schemas need to be consistent with input_param_formatter and output_param_formatter
        self.in_schema = in_schema.copy()
        self.out_schema = out_schema.copy()
        self.func = func

        self.init_output_directory()
        print(f"Logging Directory Initialized: {self.output_directory}")

        # Expand configurations
        self.expanded_config = list(GridExecutor.cartesian_product(self.compact_config))

        # TODO: Hack Fix
        self._track_duration = False

    # TODO: Change post initialization method?
    @classmethod
    def init_multiple(cls, config: Dict[str, Any], in_schema: List[str],
                      out_schema: List[str], func: Callable, trials: int):
        """
        Runs each configuration trials number of times. Each trial is indexed by a "trial_id"s
        """
        compact_config = config.copy()
        compact_config["trial_id"] = list(range(trials))
        in_schema.append("trial_id")
        return cls(compact_config, in_schema, out_schema, func)


    # TODO: Find a workaround for decorations???
    # <================== Problem ====================>
    def track_duration(self):
        """Adds a wrapper to runner to track duration, and another column to out_schema for run_duration"""
        # raise NotImplementedError
        self.out_schema.append("run_duration")
        self._track_duration = True
        # self.runner = GridExecutor.timer(self.runner)

    @staticmethod
    def timer(func):
        """A decorator that adds an duration attribute to output of a runner"""
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start_time = time.perf_counter()  # 1
            formatted_param, formatted_output = func(*args, **kwargs)
            end_time = time.perf_counter()  # 2

            formatted_output["run_duration"] = str(end_time - start_time)
            return formatted_param, formatted_output
        return wrapper_timer
    # <================== Problem ====================>

    @staticmethod
    def cartesian_product(dicts):
        """Expands an dictionary of lists into a list of dictionaries through a cartesian product"""
        return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

    def input_param_formatter(self, in_param):
        """Uses in_schema and __str__ to return a formatted dict"""
        filtered = {}
        for key in self.in_schema:
            if key == "G":
                filtered[key] = in_param[key].NAME
            elif key == "agent":
                filtered[key] = in_param[key].__name__
            else:
                filtered[key] = str(in_param[key])
        return filtered

    def output_param_formatter(self, out_param):
        """Uses out_schema and __str__ to return a formatted dict"""

        filtered = {}
        for key in self.out_schema:
            filtered[key] = str(out_param[key])
        return filtered

    def init_output_directory(self):
        # Initialize Output
        self.run_id = shortuuid.uuid()[:5]

        # Setup output directories
        self.output_directory = PROJECT_ROOT / "output" / f'run_{self.run_id}'
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.result_path = self.output_directory / 'results.csv'
        self.logging_path = self.output_directory / 'run.log'

    def init_logger(self):
        # Setup up Parallel Log Channel
        self.logger = logging.getLogger("Executor")
        self.logger.setLevel(logging.DEBUG)

        # Set LOGGING_FILE as output
        fh = logging.FileHandler(self.logging_path)
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

    # TODO: Encapsulate writer and its file into one object
    # TODO: Find a way to move it to the constructor (use file open and close?)
    def init_writer(self, result_file):
        raise NotImplementedError

    # TODO: provide a single method write result and flush to file
    def write_result(self, in_param, out_param):
        raise NotImplementedError

    def _runner(self, param: Dict[str, Any]):
        """A runner method that returns a tuple (formatted_param, formatted_output)"""
        formatted_param = self.input_param_formatter(param)
        self.logger.info(f"Launching => {formatted_param}")

        try:
            out = self.func(**param)._asdict()
        except Exception as e:
            # Find a way to export culprit data?
            self.logger.error(traceback.format_exc())
            out = {x: None for x in self.out_schema}
            
        # TODO: Added as a hack to allow output_param_formatter not to crash
        if self._track_duration:
            out["run_duration"] = None
        # output_param_formatter assumes out to be consistent with out_schema
        formatted_output = self.output_param_formatter(out)
        return formatted_param, formatted_output

    def runner(self, param):
        """TODO: Temporary workaround because of multiprocessing issues with decorators and lambdas"""
        if self._track_duration:
            return GridExecutor.timer(self._runner)(param)
        else:
            return self._runner(param)

    def exec(self):
        raise NotImplementedError

class GridExecutorParallel(GridExecutor):
    # Override the exec
    def exec(self, max_workers=20):
        with concurrent.futures.ProcessPoolExecutor(max_workers) as executor, \
             open(self.result_path, "w+") as result_file: # TODO: Encapsulate "csv file"
            self.init_logger()

            # TODO: Encapsulate "initialize csv writer" - perhaps use a context managers
            row_names = self.in_schema + self.out_schema
            writer = csv.DictWriter(result_file, fieldnames=row_names)
            writer.writeheader()

            results = [executor.submit(self.runner, arg) for arg in self.expanded_config]

            for finished_task in tqdm(concurrent.futures.as_completed(results), total=len(self.expanded_config)):
                (in_param, out_param) = finished_task.result()

                # TODO: Encapsulate "writer"
                writer.writerow({**in_param, **out_param})
                result_file.flush()

                self.logger.info(f"Finished => {in_param}")
                # debug_memory(self.logger, "run")

class GridExecutorLinear(GridExecutor):
    # Override the exec
    def exec(self):
        with open(self.result_path, "w") as result_file: # TODO: Encapsulate "csv file"
            self.init_logger()

            # TODO: Encapsulate "initialize csv writer" - perhaps use a context managers
            writer = csv.DictWriter(result_file, fieldnames=self.in_schema + self.out_schema)
            writer.writeheader()

            for arg in tqdm(self.expanded_config):
                (in_param, out_param) = self.runner(arg)

                # TODO: Encapsulate "writer"
                writer.writerow({**in_param, **out_param})
                result_file.flush()

                self.logger.info(f"Finished => {in_param}")