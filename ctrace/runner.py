import concurrent.futures
import csv
import itertools
import logging
from collections import namedtuple
from typing import Dict, Callable, List, Any, NamedTuple

import shortuuid
from tqdm import tqdm

from ctrace import PROJECT_ROOT

DEBUG = False
class GridExecutorParallel():
    """
    Encapsulates the cartesian product of different parameters
    Specify the key "trials" to run the same method several times

    Usage: Create a new GridExecutorParallel and call exec()
    """
    def __init__(self, config: Dict, in_schema: List[str], out_schema: List[str], func: Callable[..., NamedTuple]):
        """
        Parameters
        ----------
        config
            A dictionary mapping string attributes to arrays of different parameters.
        in_schema
            A list describing what and the order input attributes would be printed
        out_schema
            A list describing what and the order output attributes would be printed
        func
            A function to execute in parallel. Input arguments must match
        """
        self.compact_config = config
        self.in_schema = in_schema
        self.out_schema = out_schema
        self.func = func

        # Initialize NamedTuple Schemas for logger
        # self.InputFormatType: namedtuple = namedtuple("InputSchema", in_schema)
        # self.OutputFormatType: namedtuple = namedtuple("OutputSchema", out_schema)

        self.init_output_directory()
        print(f"Logging Directory Initialized: {self.output_directory}")
        self.expand_config()

    @classmethod
    def init_multiple(cls, config: Dict[str, Any], in_schema: List[str],
                      out_schema: List[str], func: Callable, trials: int):
        compact_config = config.copy()
        # Add trials
        compact_config["trial_id"] = list(range(trials))
        in_schema.append("trial_id")
        return cls(compact_config, in_schema, out_schema, func)

    @staticmethod
    def cartesian_product(dicts):
        """Expands an dictionary of lists into a list of dictionaries through a cartesian product"""
        return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

    def expand_config(self):
        """Adds trial IDs to replace trials and generates expanded configurations"""
        # TODO: stick with generator?
        self.expanded_config = list(GridExecutorParallel.cartesian_product(self.compact_config))

    def input_param_formatter(self, in_param):
        """Uses schema and __str__ to return a formatted dict"""

        filtered = {}
        for key in self.in_schema:
            filtered[key] = str(in_param[key])
        return filtered

    def output_param_formatter(self, out_param):
        """Uses schema and __str__ to return a formatted dict"""

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

    def runner(self, param: Dict[str, Any]):
        """A runner method that returns a tuple (formatted_param, formatted_output)"""
        logger = self.logger
        func = self.func

        # if DEBUG:
        #     print(f"GUROBI_HOME: {os.environ.get('GUROBI_HOME')}")
        formatted_param = self.input_param_formatter(param)
        logger.info(f"Launching => {formatted_param}")
        out = func(**param) # out must be a named_tuple
        formatted_output = self.output_param_formatter(out._asdict())
        return formatted_param, formatted_output

    def exec(self):
        with concurrent.futures.ProcessPoolExecutor() as executor, \
             open(self.result_path, "w") as result_file:
            self.init_logger()

            # TODO: Encapsulate "initialize csv writer"
            writer = csv.DictWriter(result_file, fieldnames=self.in_schema + self.out_schema)
            writer.writeheader()

            results = [executor.submit(self.runner, arg) for arg in self.expanded_config]

            for finished_task in tqdm(concurrent.futures.as_completed(results), total=len(self.expanded_config)):
                (in_param, out_param) = finished_task.result()

                # TODO: Encapsulate "writer"
                writer.writerow({**in_param, **out_param})
                result_file.flush()

                self.logger.info(f"Finished => {in_param}")
