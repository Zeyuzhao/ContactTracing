from param import GraphParam, SIRParam
import multiprocessing as mp
from typing import List, Tuple, Dict, Any
import itertools
class MultiExecutor():
    INIT = 0
    EXEC = 1
    def __init__(self, schema: List[Tuple[str, type]], seed: bool = True, validation: bool = True):
        self.schema = schema
        self.seed: bool = seed
        self.tasks: List[Dict[str, Any]] = [] # store expanded tasks
        self.signatures = {} # store signatures of any FileParam
        self.stage: int = MultiExecutor.INIT # Track the state of executor

        self._schema = self.schema.insert(0, ('id', int))
        if self.seed:
            self._schema.push(('seed', int))


    @staticmethod
    def cartesian_product(dicts):
        """Expands an dictionary of lists into a list of dictionaries through a cartesian product"""
        return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

    def add_cartesian(self, config: Dict[str, List[Any]]):
        # check stage
        if self.stage != MultiExecutor.INIT:
            raise Exception(f"Adding entries allowed during INIT stage. Current stage: {self.stage}")
        
        if self.validation:
            # labels must match
            config_attr = set(config.keys())
            schema_attr = set(x[0] for x in self.schema)
            if config_attr != schema_attr:
                raise ValueError(f"Given config labels {config_attr} does match specified schema labels {schema_attr}")
            
            # assert schema types
            for tup in self.schema:
                prop_name, prop_type = tup
                for item in config[prop_name]:
                    if not isinstance(item, prop_type):
                        raise ValueError(f"Property [{prop_name}]: item [{item}] is not a [{prop_type}]")

        # Collect signatures from FileParams

        self.tasks.extend(MultiExecutor.cartesian_product(config))

    def add_collection(self, collection: List[Dict[str, Any]]):

        if self.validation:
            # check stage
            if self.stage != MultiExecutor.INIT:
                raise Exception(f"Adding entries allowed during INIT stage. Current stage: {self.stage}")

            # assert types
            schema_attr = set(x[0] for x in self.schema)
            for i, task in enumerate(collection):
                task_attr = set(task.keys())
                if task_attr != schema_attr:
                    raise ValueError(f"task[{i}]: {task_attr} does match specified schema [{schema_attr}]")

                for tup in self.schema:
                    prop_name, prop_type = tup
                    item = task[prop_name]
                    if not isinstance(item, prop_type):
                        raise ValueError(f"task[{i}] [{prop_name}]: item [{item}] is not a [{prop_type}]")
        self.tasks.extend(collection)


# Example Usage
in_schema = [
    ('graph', GraphParam),
    ('sir', SIRParam),
    ('transmission_rate', float),
    ('compliance_rate', float),
    ('method', str),
]

run = MultiExecutor(in_schema, seed=True)

# Add compact tasks (expand using cartesian)
mont = GraphParam('montgomery')
run.add_cartesian({
    'graph': [mont],
    'sir' : [SIRParam(f't{i}', mont) for i in range(7, 10)],
    'transmission_rate': [0.1, 0.2, 0.3],
    'compliance_rate': [.8, .9, 1],
    'method': ["robust"]
})
run.add_cartesian({
    'graph': [mont],
    'sir' : [SIRParam(f't{i}', mont) for i in range(7, 10)],
    'transmission_rate': [0.1, 0.2, 0.3],
    'compliance_rate': [.8, .9, 1],
    'method': ["none"]
})

# Add lists of tasks
run.add_collection([{
    'graph': mont,
    'sir' : SIRParam('t7', mont),
    'transmission_rate': 0.1,
    'compliance_rate': 1,
    'method': ["greedy"]
}])

run.add_collection([{
    'graph': mont,
    'sir' : SIRParam('t7', mont),
    'transmission_rate': 0.1,
    'compliance_rate': 0.5,
    'method': ["greedy"]
}])

class Worker():
    def __init__(self):
        self.queue = mp.Queue()
    def start():
        raise NotImplementedError()

class CsvWorker(Worker):
    # TODO: Inject destination?
    def __init__(self, name, schema):
        self.name = name
        self.schema = schema
        self.queue = mp.Queue()
    def start():
        raise NotImplementedError()

main_out_schema = ["mean_objective_value", "max_objective_value", "std_objective_value"]
main_handler = CsvWorker("main", main_out_schema)

aux_out_schema = ["runtime"]
aux_handler = CsvWorker("main", aux_out_schema)

run.attach("csv_main", main_handler)
run.attach("csv_aux", aux_handler)

run.exec()