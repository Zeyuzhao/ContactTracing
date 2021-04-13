
#%%
from ctrace.simulation import InfectionInfo
from ctrace.recommender import DegGreedy, Random, SAAAgentGurobi
from ctrace.drawing import random_init, small_world_grid
from ctrace import PROJECT_ROOT
from ctrace.exec.param import GraphParam, SIRParam, FileParam
from ctrace.exec.parallel import *
from pathlib import Path, PurePath
from ctrace.problem import MinExposedSAA, grader

in_schema = [
    # Graph Structure
    ("width", int),
    ("r", float), # Parameter of long ranged decay
    ("sparsity", float),
    ("num_long_range", float),
    # SIR
    ("num_infected", int),
    # Transmission dynamics
    ("transmission_rate", float),
    ("compliance_rate", float),
    ("structure_rate", float),
    # Problem constraints,
    ("budget", int),
    # Method 
    ("method", str),
    ("num_samples", int),
    # Evaluation
    ("eval_num_samples", int),
    ("eval_aggregation_method", str),
    # Seeding
    ("seed", int),
]

main_out_schema = [
    "id",
    "saa_objective_value",
    "grader_objective_value",
    "greedy_objective_value",
    "random_objective_value"
]

main_handler = CsvSchemaWorker("csv_main", main_out_schema, PurePath('main.csv'))
saa_handler = CsvWorker("csv_saa_vals", PurePath('saa.csv'))
grader_handler = CsvWorker("csv_grader_vals", PurePath('grader.csv'))
greedy_handler = CsvWorker("csv_greedy_vals", PurePath('greedy.csv'))
random_handler = CsvWorker("csv_random_vals", PurePath('random.csv'))


def runner(
    # Support info
    queues,
    id,
    path,
    width=10,
    # Graph Structure
    r=2, # Parameter of long ranged decay
    sparsity=0.1,
    num_long_range=1,
    # SIR
    num_infected=5,
    # Transmission dynamics
    transmission_rate=0.2,
    compliance_rate=.9,
    structure_rate=0,
    # Problem constraints,
    budget=50,
    # Method 
    method="SAA",
    num_samples=100,
    # Evaluation
    eval_num_samples=100,
    eval_aggregation_method="max",
    # Seeding
    seed=42,
    **args,
):
    # <================ Create Problem Instance ================>
    G, pos = small_world_grid(
        width, 
        max_norm=True, 
        sparsity=sparsity, 
        local_range=1, 
        num_long_range=num_long_range, 
        r=r,
        seed=seed,
    )

    SIR = random_init(G, num_infected=num_infected, seed=seed)

    # <================ SAA Agent ================>

    # Group these attributes into problem dataclasses!
    info = SAAAgentGurobi(
        G=G,
        SIR=SIR,
        budget=budget,
        num_samples=num_samples,
        transmission_rate=transmission_rate, 
        compliance_rate=compliance_rate, 
        structure_rate=structure_rate, 
        seed=seed,
        solver_id="GUROBI",
    )
    action = info["action"]
    problem: MinExposedSAA = info["problem"]
    saa_objectives = [problem.lp_objective_value(i) for i in range(num_samples)]
    saa_objective_value = max(saa_objectives)
    # Evaluation
    grader_seed = seed # arbitary value (should be different)

    gproblem = grader(
        G,
        SIR,
        budget,
        transmission_rate,
        compliance_rate,
        action,
        structure_rate=structure_rate,
        grader_seed=grader_seed,
        num_samples=eval_num_samples,
        aggregation_method=eval_aggregation_method,
        solver_id="GUROBI_LP",
    )
    grader_objective_value = gproblem.objective_value
    grader_objectives = [gproblem.lp_objective_value(i) for i in range(num_samples)]
    # <================ Compute baselines ================>

    info = InfectionInfo(G, SIR, budget, transmission_rate)

    # Weighted Greedy
    greedy_action = DegGreedy(info)
    grader_greedy = grader(
        G,
        SIR,
        budget,
        transmission_rate,
        compliance_rate,
        greedy_action,
        structure_rate=structure_rate,
        grader_seed=grader_seed,
        num_samples=eval_num_samples,
        aggregation_method=eval_aggregation_method,
        solver_id="GUROBI_LP",
    )
    greedy_objective_value = grader_greedy.objective_value
    grader_greedy_objectives = [grader_greedy.lp_objective_value(i) for i in range(num_samples)]


    # Random
    random_action = Random(info)
    grader_random = grader(
        G,
        SIR,
        budget,
        transmission_rate,
        compliance_rate,
        random_action,
        structure_rate=structure_rate,
        grader_seed=grader_seed,
        num_samples=eval_num_samples,
        aggregation_method=eval_aggregation_method,
        solver_id="GUROBI_LP",
    )
    random_objective_value = grader_random.objective_value
    grader_random_objectives = [grader_random.lp_objective_value(i) for i in range(num_samples)]
    # <================ Output data to workers and folders ================>

    main_obj = {
        "id": id,
        "saa_objective_value": saa_objective_value,
        "grader_objective_value": grader_objective_value,
        "greedy_objective_value": greedy_objective_value,
        "random_objective_value": random_objective_value,
    }
    queues["csv_main"].put(main_obj)

    # Store each individual SAA Objectives
    saa_objectives.insert(0, id)
    queues["csv_saa_vals"].put(saa_objectives)

    grader_objectives.insert(0, id)
    queues["csv_grader_vals"].put(grader_objectives)

    grader_greedy_objectives.insert(0, id)
    queues["csv_greedy_vals"].put(grader_greedy_objectives)

    grader_random_objectives.insert(0, id)
    queues["csv_random_vals"].put(grader_random_objectives)

    # Save large checkpoint data ("High data usage") 
    save_extra = id % 1 == 0
    if save_extra:
        path = path / "data" / str(id)
        path.mkdir(parents=True, exist_ok=True)
        # with open(path / "sir_dump.json", "w") as f:
        #     json.dump(sir, f)
        
        with open(path / "graph.pkl", "wb") as f:
            pickle.dump(G, f)
        with open(path / "pos.pkl", "wb") as f:
            pickle.dump(pos, f)
        with open(path / "sir.pkl", "wb") as f:
            pickle.dump(SIR, f)

        # with open(path / "problem.pkl", "wb") as f:
        #     pickle.dump(problem, f)

        # with open(path / "grader.pkl", "wb") as f:
        #     pickle.dump(gproblem, f)

def _runner(data):
    runner(**data)

def post_execution(self):
    compress=False
    if (self.output_directory / "data").exists() and compress:
        print("Compressing files ...")
        shutil.make_archive(str(self.output_directory / "data"), 'zip', base_dir="data")
        shutil.rmtree(self.output_directory / "data")

    # TODO: Merge different files

    

run = MultiExecutor(_runner, in_schema, post_execution=post_execution, seed=True)

# Add compact tasks (expand using cartesian)
mont = GraphParam('montgomery')
# run.add_cartesian({
#     # Graph Structure
#     "width": [50, 80, 100, 200],
#     "r":[1.1, 1.5, 2.0, 3.0, 4.0], # Parameter of long ranged decay
#     "sparsity":[0.1],
#     # SIR
#     "num_infected":[50,100,150,200],
#     # Transmission dynamics
#     "transmission_rate":[0.15],
#     "compliance_rate":[0.9],
#     "structure_rate":[0.0],
#     # Problem constraints,
#     "budget":[100, 200, 300, 400],
#     # Method 
#     "method":["SAA"],
#     "num_samples":[100],
#     # Evaluation
#     "eval_num_samples":[100],
#     "eval_aggregation_method":["max"],
#     # Seeding
#     "seed":[100,200,300,400,500],
# })

run.add_cartesian({
    # Graph Structure
    "width": [50],
    "r":[2.0], # Parameter of long ranged decay
    "sparsity":[0.1],
    "num_long_range": [5.0],
    # SIR
    "num_infected":[50, 100, 150, 200],
    # Transmission dynamics
    "transmission_rate":[0.2],
    "compliance_rate":[0.9],
    "structure_rate":[0.0],
    # Problem constraints,
    "budget":[100, 200, 300, 400],
    # Method 
    "method":["SAA"],
    "num_samples":[100],
    # Evaluation
    "eval_num_samples":[100],
    "eval_aggregation_method":["max"],
    # Seeding
    "seed":[200],
})

# main_out_schema = ["mean_objective_value", "max_objective_value", "std_objective_value"]

run.attach(main_handler)


run.attach(saa_handler)
run.attach(grader_handler)
run.attach(greedy_handler)
run.attach(random_handler)
run.exec()
# %%
