# %%
import numpy as np
from ctrace import PROJECT_ROOT
from ctrace.simulation import InfectionState
from ctrace.exec.param import GraphParam, SIRParam, FileParam, ParamBase, LambdaParam
from ctrace.exec.parallel import CsvWorker, MultiExecutor, CsvSchemaWorker
from ctrace.recommender import SegDegree
import json
import shutil
import random
import copy
import pickle
from pathlib import PurePath

# Example Usage
in_schema = [
    ('graph', ParamBase),  # nx.Graph
    ('agent', ParamBase),  # lambda
    ('agent_params', dict),
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
main_out_schema = ["id", "peak", "total"]
aux_out_schema = ["id", "sir_history"]

main_handler = CsvSchemaWorker(
    name="csv_main", schema=main_out_schema, relpath=PurePath('main.csv'))
aux_handler = CsvWorker(
    name="csv_inf_history", relpath=PurePath('inf_history.csv'))


def runner(
    queues,
    id,
    path,
    # User Specific attributes
    graph,
    agent,
    agent_params,
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

    with open(PROJECT_ROOT / "data" / "SIR_Cache" / from_cache, 'r') as infile:
        j = json.load(infile)

        (S, I1, I2, R) = (j["S"], j["I1"], j["I2"], j["R"])
        # An array of previous infection accounts
        infections = j["infections"]

    raw_history = []
    state = InfectionState(graph, (S, I1, I2, R), budget, policy, transmission_rate,
                           transmission_known, compliance_rate, compliance_known, snitch_rate)

    while len(state.SIR.I1) + len(state.SIR.I2) != 0:
        to_quarantine = agent(state, **agent_params)
        # Record full history
        if trial_id == 0:
            raw_history.append({
                'state': {
                    'S': list(state.SIR.S),
                    'I1': list(state.SIR.I1),
                    'I2': list(state.SIR.I2),
                    'R': list(state.SIR.R),
                },
                'action': list(to_quarantine),
            })
        state.step(to_quarantine)
        infections.append(len(state.SIR.I2))

    # Output data to workers and folders

    main_out = {
        "id": id,
        "peak": max(infections),
        # Total infections (number of people recovered)
        "total": len(state.SIR.R),
    }
    aux_out = [id, *infections]

    queues["csv_main"].put(main_out)
    queues["csv_inf_history"].put(aux_out)

    # if trial_id == 0:  # Only record 1 entry
    #     path = path / "data" / str(id)
    #     path.mkdir(parents=True, exist_ok=True)
    #     with open(path / "sir_history.json", "w") as f:
    #         json.dump(raw_history, f)


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
    num_process=80,
    name_prefix='seg_extra'
)

# Add compact tasks (expand using cartesian)
montgomery = GraphParam('montgomery_extra')
cville = GraphParam('cville_extra')

# Schema
run.add_cartesian({
    "graph": [montgomery],
    "budget": [750],
    "agent": [LambdaParam(SegDegree)],
    "agent_params": [{'k1': round(p, 3)} for p in np.arange(0, 1.01, 0.01)],
    # "budget": [i for i in range(400, 1260, 50)],
    "policy": ["A"],
    "transmission_rate": [0.05],
    "transmission_known": [False],
    "compliance_rate": [-1.0],
    "compliance_known": [False],
    "discovery_rate": [1.0],
    "snitch_rate": [1.0],
    "from_cache": ["mont.json"],
    "trial_id": [i for i in range(5)]
})
run.add_cartesian({
    "graph": [cville],
    "budget": [1350],
    # "budget": [i for i in range(720, 2270, 20)],
    "agent": [LambdaParam(SegDegree)],
    "agent_params": [{'k1': round(p, 3)} for p in np.arange(0, 1.01, 0.01)],
    "policy": ["A"],
    "transmission_rate": [0.05],
    "transmission_known": [False],
    "compliance_rate": [-1.0],
    "compliance_known": [False],
    "discovery_rate": [1.0],
    "snitch_rate": [1.0],
    "from_cache": ["albe.json"],
    "trial_id": [i for i in range(5)],
})

run.add_cartesian({
    "graph": [montgomery],
    "budget": [750],
    "agent": [LambdaParam(SegDegree)],
    "agent_params": [{'k2': round(p, 3)} for p in np.arange(0, 1.01, 0.01)],
    # "budget": [i for i in range(400, 1260, 50)],
    "policy": ["A"],
    "transmission_rate": [0.05],
    "transmission_known": [False],
    "compliance_rate": [-1.0],
    "compliance_known": [False],
    "discovery_rate": [1.0],
    "snitch_rate": [1.0],
    "from_cache": ["mont_star.json"],
    "trial_id": [i for i in range(5)]
})
run.add_cartesian({
    "graph": [cville],
    "budget": [1350],
    # "budget": [i for i in range(720, 2270, 20)],
    "agent": [LambdaParam(SegDegree)],
    "agent_params": [{'k2': round(p, 3)} for p in np.arange(0, 1.01, 0.01)],
    "policy": ["A"],
    "transmission_rate": [0.05],
    "transmission_known": [False],
    "compliance_rate": [-1.0],
    "compliance_known": [False],
    "discovery_rate": [1.0],
    "snitch_rate": [1.0],
    "from_cache": ["albe_star.json"],
    "trial_id": [i for i in range(5)],
})

# main_out_schema = ["mean_objective_value", "max_objective_value", "std_objective_value"]

run.attach(main_handler)
run.attach(aux_handler)

# %%
run.exec()
