# %%
import os
import traceback
from ctrace.utils import calculateD, calculateExpected, calculateMILP
import numpy as np
from ctrace import PROJECT_ROOT
from ctrace.simulation import InfectionState
from ctrace.exec.param import GraphParam, SIRParam, FileParam, ParamBase, LambdaParam
from ctrace.exec.parallel import CsvWorker, MultiExecutor, CsvSchemaWorker
from ctrace.recommender import Milp, SegDegree, DepRound_fair, DegGreedy_fair, Random
import json
import shutil
import random
import copy
import pickle
from pathlib import PurePath
import time

from functools import wraps
import errno
import os
import signal

class TimeoutError(Exception):
    pass

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

# Example Usage
in_schema = [
    ('graph', ParamBase),  # nx.Graph
    ('agent', ParamBase),  # lambda
    ('from_cache', str),   # PartitionSEIR
    ('budget', int),
    ('policy', str),
    ('transmission_rate', float),
    ('transmission_known', bool),
    ('compliance_rate', float),
    ('compliance_known', bool),
    ('discovery_rate', float),
    ('snitch_rate', float),
    ('trial_id', int),
]
# Must include "id"
main_out_schema = [
    "id",
    "milp_obj",
    "expected_obj",
    "is_optimal",
    "I_size",
    "v1_size",
    "v2_size",
    "total_cross_edges",
    "D",
    "time", 
]

main_handler = CsvSchemaWorker(
    name="csv_main", schema=main_out_schema, relpath=PurePath('main.csv'))

def runner(
    queues,
    id,
    path,
    # User Specific attributes
    graph,
    agent,
    from_cache,
    budget,
    policy,
    transmission_rate,
    transmission_known,
    compliance_rate,
    compliance_known,
    discovery_rate,
    snitch_rate,
    trial_id,
    # Allow additonal args to be passed (to be ignored)
    **args,
):
    # Execute logic here ...
    with open(PROJECT_ROOT / "data" / "SIR_Cache" / "optimal_trials" / f"{graph.NAME}_trials" / from_cache, 'r') as infile:
        j = json.load(infile)

        (S, I1, I2, R) = (j["S"], j["I1"], j["I2"], j["R"])
        # An array of previous infection accounts
        infections = j["infections"]

    state = InfectionState(
        graph, (S, I1, I2, R), budget, policy, transmission_rate,
        transmission_known, compliance_rate, compliance_known, snitch_rate
    )

    try:
        # 1hr min limit
        with timeout(14400): 
            time1 = time.perf_counter()
            result = agent(state, extra=True)
            time2 = time.perf_counter()

        action = result['action']
        is_optimal = result.get('is_optimal')
        
        milp_obj = calculateMILP(state, action)
        expected_obj = calculateExpected(state, action)
        
        maxD, totalD = calculateD(state)
        # Output data to  and folders
        main_out = {
            "id": id,
            "milp_obj": milp_obj,
            "expected_obj": expected_obj,
            "is_optimal": is_optimal,
            "I_size": len(state.SIR.I2),
            "v1_size": len(state.V1),
            "v2_size": len(state.V2),
            "total_cross_edges": totalD,
            "D": maxD,
            "time": time2 - time1, 
        }
        queues["csv_main"].put(main_out)
    except TimeoutError as e:
        print(f'Skipping {id} due to timeout')
    except Exception as e:
        traceback.print_exc()


def runner_star(x):
    return runner(**x)


def post_execution(self):
    compress = False
    delete = False
    if (self.output_directory / "data").exists() and compress:
        print("Compressing files ...")
        shutil.make_archive(
            str(self.output_directory / "data"), 'zip', base_dir="data")
        if delete:
            shutil.rmtree(self.output_directory / "data")


run = MultiExecutor(
    runner_star, 
    in_schema,
    post_execution=post_execution, 
    seed=True, 
    num_process=20,
    name_prefix='opt'
)

# Add compact tasks (expand using cartesian)
montgomery = GraphParam('montgomery')
cville = GraphParam('cville')

# Schema
run.add_cartesian({
    "graph": [montgomery],
    "budget": [750],
    "agent": [LambdaParam(SegDegree), LambdaParam(DegGreedy_fair),  LambdaParam(DepRound_fair), LambdaParam(Random), LambdaParam(Milp)],
    # "agent": [LambdaParam(MILP_fair)],
    # "budget": [i for i in range(400, 1260, 50)],
    "policy": ["A"],
    "transmission_rate": [0.05],
    "transmission_known": [True],
    "compliance_rate": [-1.0],
    "compliance_known": [True],
    "discovery_rate": [1.0],
    "snitch_rate": [1.0],
    "from_cache": os.listdir(PROJECT_ROOT/"data"/"SIR_Cache"/"optimal_trials"/ 'montgomery_trials'),
    "trial_id": [i for i in range(1)]
})
run.add_cartesian({
    "graph": [cville],
    "budget": [1350],
    # "budget": [i for i in range(720, 2270, 20)],
    "agent": [LambdaParam(SegDegree), LambdaParam(DegGreedy_fair),  LambdaParam(DepRound_fair), LambdaParam(Random), LambdaParam(Milp)],
    # "agent": [LambdaParam(MILP_fair)],
    "policy": ["A"],
    "transmission_rate": [0.05],
    "transmission_known": [True],
    "compliance_rate": [-1.0],
    "compliance_known": [True],
    "discovery_rate": [1.0],
    "snitch_rate": [1.0],
    "from_cache": os.listdir(PROJECT_ROOT/"data"/"SIR_Cache"/"optimal_trials" / 'cville_trials'),
    "trial_id": [i for i in range(1)],
})

# main_out_schema = ["mean_objective_value", "max_objective_value", "std_objective_value"]

run.attach(main_handler)

# %%
run.exec()
