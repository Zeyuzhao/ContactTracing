import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ctrace
from tqdm import tqdm
from ctrace.simulation import *
from ctrace.utils import *

from ctrace.dataset import load_graph, load_sir_path
import concurrent.futures


G = load_graph("montgomery")

with open(ctrace.PROJECT_ROOT /"output"/"run_K2VWk" / "results.csv", "r") as csv:
    df = pd.read_csv(csv)

def compute_row_sum(from_cache, p=0.078):
    SIR = load_sir_path(from_cache, merge=True)
    I = SIR["I"]
    R = SIR["R"]
    v1, v2 = find_excluded_contours(G, I, R)
    P, Q = PQ_deterministic(G, I, v1, p)
    return sum(P.values())

cache_data = df["from_cache"]
results = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = tqdm(executor.map(compute_row_sum, cache_data), total=len(cache_data))
