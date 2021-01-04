import math
from collections import defaultdict
from typing import Set, List

import networkx as nx
import EoN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import concurrent.futures as cf


def V1(G, I):
    V1 = set([])
    for i in I:
        for j in G.neighbors(i):
            if j not in I:
                V1.add(j)
    return V1


def single_run_PQ(G, I, v1, p=0.5, runs=20):
    Q = pd.DataFrame()
    P = {}
    full_data = EoN.basic_discrete_SIR(
        G, p, tmax=2, initial_infecteds=I, return_full_data=True)
    df = pd.DataFrame(full_data.transmissions(), columns=['time', 'u', 'v'])
    df.time += 1
    df['u'] = df['u'].astype('str')
    df['v'] = df['v'].astype('str')
    df['edge'] = df['u'] + ' ' + df['v']
    df['q_uv'] = 1 / runs
    data = df[df.time == 2][['edge', 'q_uv']].reset_index(drop=True)
    for v in v1:
        P[v] = P.get(v, 0) + (1 / runs) * \
            int(full_data.node_status(v, 1) == 'I')
    PP = pd.DataFrame(columns=['v', 'p_v'])
    PP['v'] = P.keys()
    PP['p_v'] = P.values()
    PP = PP.sort_values('v')
    return [PP, data]


def PQ(G, I, p=0.5, runs=20):
    """Returns dataframe P, Q"""
    v1 = V1(G, I)
    Q = pd.DataFrame()
    P = pd.DataFrame()

    with cf.ProcessPoolExecutor() as executor:
        results = [executor.submit(single_run_PQ,
                                   G, I, v1, p=p, runs=runs)
                   for _ in range(runs)]

        for x in cf.as_completed(results):
            PP, QQ = x.result()
            Q = pd.concat([Q, QQ])
            P = pd.concat([P, PP])

    Q = Q.groupby('edge').sum().reset_index()
    Q['u'] = Q['edge'].apply(lambda x: x.split(
        ' ')[0]).astype('float').astype('int')
    Q['v'] = Q['edge'].apply(lambda x: x.split(
        ' ')[1]).astype('float').astype('int')
    Q['v1-v2'] = Q['v'].apply(lambda x: x not in v1)
    Q = Q.loc[Q['v1-v2']][['u', 'v', 'q_uv']]
    P = P.groupby('v')['p_v'].sum().reset_index()
    return [P, Q]


def PQ_deterministic(G: nx.Graph, I: List[int], V1: List[int], p: float):
    # Returns dictionary P, Q
    # Calculate P, (1-P) ^ [number of neighbors in I]
    P = {v: 1 - math.pow((1 - p), len(set(G.neighbors(v)) & set(I))) for v in V1}
    Q = defaultdict(lambda: defaultdict(lambda: p))
    return P, Q
