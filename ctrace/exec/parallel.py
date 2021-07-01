# %%
import shortuuid
from ctrace import PROJECT_ROOT
from ctrace.exec.param import GraphParam, SIRParam, FileParam, ParamBase
from pathlib import Path, PurePath
import multiprocessing as mp
import concurrent.futures
from typing import List, Tuple, Dict, Any, Callable
from zipfile import ZipFile
import itertools
import pandas as pd
import numpy as np
import tqdm
import csv
import random
import time
import pickle
import json
import shutil


class MultiExecutor():
    INIT = 0
    EXEC = 1

    def __init__(
        self,
        runner: Callable,
        schema: List[Tuple[str, type]],
        post_execution: Callable = lambda self: self,
        output_id: str = None,
        seed: bool = True,
        validation: bool = True,
        name_prefix = 'run',
        num_process=50,
    ):
        self.runner = runner
        self.schema = schema

        # Multiexecutor state
        self.tasks: List[Dict[str, Any]] = []  # store expanded tasks
        self.signatures = {}  # store signatures of any FileParam
        self.stage: int = MultiExecutor.INIT  # Track the state of executor
        self.workers = {}
        self.queues = {}

        self.manager = mp.Manager()

        # Filter FileParams from schema
        self.file_params = [l for (l, t) in schema if issubclass(t, ParamBase)]

        # Executor Parameters
        self.seed: bool = seed
        self.validation = validation
        self._schema = self.schema[:]
        self._schema.insert(0, ('id', int))
        if self.seed:
            self._schema.append(('seed', int))
        self.num_process = num_process

        self.name_prefix = name_prefix
        # Initialize functions
        self.output_id = output_id
        self.init_output_directory()

        self.post_execution = post_execution

    def init_output_directory(self):
        if self.output_id is None:
            self.output_id = f"{self.name_prefix}_{shortuuid.uuid()[:5]}"
        # Setup output directories
        self.output_directory = PROJECT_ROOT / "output" / self.output_id
        self.output_directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def cartesian_product(dicts):
        """Expands an dictionary of lists into a list of dictionaries through a cartesian product"""
        return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

    def add_cartesian(self, config: Dict[str, List[Any]]):
        if self.validation:
            # check stage
            if self.stage != MultiExecutor.INIT:
                raise Exception(
                    f"Adding entries allowed during INIT stage. Current stage: {self.stage}")
            # labels must match
            config_attr = set(config.keys())
            schema_attr = set(x[0] for x in self.schema)
            if config_attr != schema_attr:
                raise ValueError(
                    f"Given config labels {config_attr} does match specified schema labels {schema_attr}")

            # assert schema types
            for tup in self.schema:
                prop_name, prop_type = tup
                for item in config[prop_name]:
                    if not isinstance(item, prop_type):
                        raise ValueError(
                            f"Property [{prop_name}]: item [{item}] is not a [{prop_type}]")

        # Collect signatures from FileParams
        new_tasks = list(MultiExecutor.cartesian_product(config))
        self.tasks.extend(new_tasks)
        print(f"Added {len(new_tasks)} cartesian tasks.")

    def add_collection(self, collection: List[Dict[str, Any]]):
        if self.validation:
            # check stage
            if self.stage != MultiExecutor.INIT:
                raise Exception(
                    f"Adding entries allowed during INIT stage. Current stage: {self.stage}")

            # assert types
            schema_attr = set(x[0] for x in self.schema)
            for i, task in enumerate(collection):
                task_attr = set(task.keys())
                if task_attr != schema_attr:
                    raise ValueError(
                        f"task[{i}]: {task_attr} does match specified schema [{schema_attr}]")

                for tup in self.schema:
                    prop_name, prop_type = tup
                    item = task[prop_name]
                    if not isinstance(item, prop_type):
                        raise ValueError(
                            f"task[{i}] [{prop_name}]: item [{item}] is not a [{prop_type}]")
        self.tasks.extend(collection)

    def attach(self, worker):
        # Inject dependencies
        name = worker.name
        worker.run_root = self.output_directory
        worker.queue = self.manager.Queue()

        self.queues[name] = worker.queue
        self.workers[name] = worker

    def exec(self):
        self.stage = MultiExecutor.EXEC
        # Clean Up and pre-exec initialization

        # Attach seeds
        df = pd.DataFrame.from_dict(self.tasks, orient='columns')
        df["seed"] = np.random.randint(0, 100000, size=(len(df), 1))
        df.insert(0, "id", range(len(df)))

        # Label receives name (string), data receives data object
        label_df = df.copy()
        data_df = df.copy()
        for label in self.file_params:
            label_df[label] = label_df[label].map(lambda x: x.name)
            data_df[label] = data_df[label].map(lambda x: x.data)

        label_df.to_csv(self.output_directory / "input.csv", index=False)
        self.tasks = data_df.to_dict('records')

        processes = []
        # Start workers and attach worker queues
        for (_, worker) in self.workers.items():
            p = mp.Process(target=worker.start)
            p.start()
            processes.append(p)

        # Start and attach loggers
        self.loggers = {}  # TODO: Implement loggers
        for item in self.tasks:
            item["queues"] = self.queues
            item["loggers"] = self.loggers
            item["path"] = self.output_directory

        with mp.Pool(self.num_process) as pool:
            list(tqdm.tqdm(pool.imap_unordered(
                self.runner, self.tasks), total=len(self.tasks)))

        # Clean up workers
        for (_, q) in self.queues.items():
            q.put("done")

        for p in processes:
            p.join()

        # TODO: Join the main and auxillary csvs???
        self.post_execution(self)


class Worker():
    def __init__(self):
        self.queue = mp.Queue()

    def start():
        raise NotImplementedError()


class CsvSchemaWorker(Worker):
    # TODO: Inject destination?
    def __init__(self, name, schema, relpath: PurePath, run_root: Path = None, queue=None):
        """
        Listens on created queue for dicts. Worker will extract only data specified from dicts
        and fills with self.NA if any attribute doesn't exist.

        Dicts should contain id, preferrably as first element of schema

        Will stop if dict is passed a terminating object (self.term).

        """
        # Should be unique within a collection!
        self.name = name
        self.schema = schema
        self.queue = queue
        self.relpath = relpath
        self.run_root = run_root

        # Default value
        self.default = None
        # Terminating value
        self.terminator = "done"

    def start(self):
        """

        """
        # TODO: Replace prints with logging
        if self.run_root is None:
            raise ValueError('run_root needs to be a path')

        if self.queue is None:
            raise ValueError('need to pass a queue to run')

        self.path = self.run_root / self.relpath

        with open(self.path, 'w') as f:
            writer = csv.DictWriter(
                f, self.schema, restval=self.default, extrasaction='raise')
            writer.writeheader()
            print(f'INFO: CsvSchemaWorker {self.name} initialized @ {self.path}')
            while True:
                msg = self.queue.get()
                if (msg == self.terminator):
                    print(f"INFO: Worker {self.name} finished")
                    break
                # Filter for default
                # data = {l: (msg.get(l, self.default)) for l in self.schema}
                writer.writerow(msg)
                f.flush()
                # print(f'DEBUG: Worker {self.name} writes entry {msg.get("id")}')


class CsvWorker(Worker):
    # TODO: Inject destination?
    def __init__(self, name, relpath: PurePath, run_root: Path = None, queue=None):
        """
        Listens on created queue for dicts. Worker will extract only data specified from dicts
        and fills with self.NA if any attribute doesn't exist.

        Dicts should contain id, preferrably as first element of schema

        Will stop if dict is passed a terminating object (self.term).

        """
        # Should be unique within a collection!
        self.name = name
        self.queue = queue
        self.relpath = relpath
        self.run_root = run_root

        # Default value
        self.default = None
        # Terminating value
        self.terminator = "done"

    def start(self):
        """
        Each object piped to queue must be in form [id, val1, val2 ...]
        """
        # TODO: Replace prints with logging
        if self.run_root is None:
            raise ValueError('run_root needs to a path')

        if self.queue is None:
            raise ValueError('need to pass a queue to run')

        self.path = self.run_root / self.relpath

        with open(self.path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "vals"])
            print(f'INFO: CsvWorker {self.name} initialized @ {self.path}')
            while True:
                msg = self.queue.get()
                if (msg == self.terminator):
                    print(f"INFO: Worker {self.name} finished")
                    break
                # Filter for default
                # data = {l: (msg.get(l, self.default)) for l in self.schema}
                writer.writerow(msg)
                f.flush()
                # print(f'DEBUG: Worker {self.name} writes entry {msg.get("id")}')
# %%


if __name__ == '__main__':
    # Example Usage
    in_schema = [
        ('graph', GraphParam),
        ('sir', SIRParam),
        ('transmission_rate', float),
        ('compliance_rate', float),
        ('method', str),
    ]

    main_out_schema = ["id", "out_method"]
    aux_out_schema = ["id", "runtime"]

    main_handler = CsvSchemaWorker(
        name="csv_main", schema=main_out_schema, relpath=PurePath('main.csv'))
    aux_handler = CsvSchemaWorker(
        name="csv_aux", schema=aux_out_schema, relpath=PurePath('aux.csv'))

    def runner(data):

        queues = data["queues"]
        instance_id = data["id"]
        method = data["method"]
        graph = data["graph"]
        compliance_rate = data["compliance_rate"]
        sir = data["sir"]
        # Execute logic here ...

        # Output data to workers and folders
        main_obj = {
            "id": instance_id,
            "out_method": method,
        }

        aux_obj = {
            "id": instance_id,
            "runtime": random.random()
        }
        queues["csv_main"].put(main_obj)
        queues["csv_aux"].put(aux_obj)

        # Save large checkpoint data ("High data usage")
        save_extra = False
        if save_extra:
            path = data["path"] / "data" / str(data["id"])
            path.mkdir(parents=True, exist_ok=True)

            with open(path / "sir_dump.json", "w") as f:
                json.dump(sir, f)

    def post_execution(self):
        compress = False
        if self.output_directory / "data".exists() and compress:
            print("Compressing files ...")
            shutil.make_archive(
                str(self.output_directory / "data"), 'zip', base_dir="data")
            shutil.rmtree(self.output_directory / "data")

    run = MultiExecutor(runner, in_schema,
                        post_execution=post_execution, seed=True)

    # Add compact tasks (expand using cartesian)
    mont = GraphParam('montgomery')
    run.add_cartesian({
        'graph': [mont],
        'sir': [SIRParam(f't{i}', parent=mont) for i in range(7, 10)],
        'transmission_rate': [0.1, 0.2, 0.3],
        'compliance_rate': [.8, .9, 1.0],
        'method': ["robust"]
    })
    run.add_cartesian({
        'graph': [mont],
        'sir': [SIRParam(f't{i}', parent=mont) for i in range(7, 10)],
        'transmission_rate': [0.1, 0.2, 0.3],
        'compliance_rate': [.8, .9, 1.0],
        'method': ["none"]
    })

    # Add lists of tasks
    run.add_collection([{
        'graph': mont,
        'sir': SIRParam('t7', parent=mont),
        'transmission_rate': 0.1,
        'compliance_rate': 1.0,
        'method': "greedy"
    }])

    run.add_collection([{
        'graph': mont,
        'sir': SIRParam('t7', parent=mont),
        'transmission_rate': 0.1,
        'compliance_rate': 0.5,
        'method': "greedy"
    }])

    # main_out_schema = ["mean_objective_value", "max_objective_value", "std_objective_value"]

    run.attach(main_handler)
    run.attach(aux_handler)

    run.exec()

# %%
