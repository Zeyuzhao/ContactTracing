from ctrace import PROJECT_ROOT
from ctrace.simulation import InfectionState
from ctrace.exec.param import GraphParam, SIRParam, FileParam, ParamBase
from ctrace.exec.parallel import CsvWorker, MultiExecutor
from ctrace.recommender import segmented_greedy, DepRound_fair, DegGreedy_fair
import json
import shutil
import random
from pathlib import PurePath

if __name__ == '__main__':
    # Example Usage
    in_schema = [
        ('graph', ParamBase),  # nx.Graph
        ('agent', ParamBase),  # lambda
        ('from_cache', str),   # PartitionSEIR
        ('budget', int),
        ('policy', str),
        ('transmission_rate', float),
        ('transmission_known', float),
        ('compliance_rate', float),
        ('compliance_known', float),
        ('discovery_rate', float),
        ('snitch_rate', float),
    ]
    # Must include "id"
    main_out_schema = ["id", "out_method"]
    aux_out_schema = ["id", "runtime"]

    main_handler = CsvWorker("csv_main", main_out_schema, PurePath('main.csv'))
    aux_handler = CsvWorker("csv_aux", aux_out_schema, PurePath('aux.csv'))

    def runner(
        queues,
        id,
        path,
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
    ):
        # Execute logic here ...

        with open(PROJECT_ROOT / "data" / "SIR_Cache" / from_cache, 'r') as infile:
            j = json.load(infile)

            (S, I1, I2, R) = (j["S"], j["I1"], j["I2"], j["R"])
            infections = j["infections"] # An array of previous infection accounts

        state = InfectionState(graph, (S, I1, I2, R), budget, policy, transmission_rate,
                           transmission_known, compliance_rate, compliance_known, discovery_rate, snitch_rate)

        
        while len(state.SIR.I1) + len(state.SIR.I2) != 0:
            to_quarantine = agent(state)
            state.step(to_quarantine)
            infections.append(len(state.SIR.I2))

        # Output data to workers and folders

        peak = max()
        main_obj = {
            "id": id,
            "peak": peak,
            "total": state.SIR.R, # Total infections (number of people recovered)
        }

        aux_obj = {
            "id": id,
            "sir_history": infections, # List of infection count (I2) over time
        }

        queues["csv_main"].put(main_obj)
        queues["csv_aux"].put(aux_obj)

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
    montgomery = GraphParam('montgomery')
    cville = GraphParam('cville')

    run.add_cartesian({
        "G": [montgomery],
        "budget": [i for i in range(400, 1260, 10)],
        "agent": [DepRound_fair, DegGreedy_fair],
        # "budget": [i for i in range(400, 1260, 50)],
        "policy": ["none"],
        "transmission_rate": [0.05],
        "transmission_known": [True],
        "compliance_rate": [0.8],
        "compliance_known": [True],
        "discovery_rate": [1],
        "snitch_rate": [1],
        "from_cache": ["c7.json"],
    })
    run.add_cartesian({
        "G": [cville],
        "budget": [i for i in range(720, 2270, 20)],
        "agent": [DepRound_fair, DegGreedy_fair]
        "policy": ["none"],
        "transmission_rate": [0.05],
        "transmission_known": [True],
        "compliance_rate": [-1],
        "compliance_known": [False],
        "discovery_rate": [1],
        "snitch_rate": [1],
        "from_cache": ["b5.json"],
    })

    # main_out_schema = ["mean_objective_value", "max_objective_value", "std_objective_value"]

    run.attach(main_handler)
    run.attach(aux_handler)

    run.exec()
