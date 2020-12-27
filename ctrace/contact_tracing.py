import networkx as nx
import EoN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


def grab_graph():
    montgomery = open('../data/mont/montgomery.csv')

    nodes = {}
    for line in montgomery:
         u,v = (line[:-1].split(','))
         u = int(u)
         v = int(v)
         for w in [u,v]:
             if nodes.get(w,True):
                nodes[w]= False

    sorted_ids = sorted(nodes.keys())
    node_maps = {str(sorted_ids[i]):i for i in range(len(sorted_ids))}
    montgomery = open('../data/mont/montgomery.csv')

    G = nx.Graph()
    nodes = {}

    for line in montgomery:
         u,v = (line[:-1].split(','))
         u = node_maps[u]
         v = node_maps[v]
         for w in [u,v]:
             if nodes.get(w,True):
                nodes[w]= False
                G.add_node(w)
         G.add_edge(u, v)
    return G

def V1(G,I):
    V1 = set([])
    for i in I:
        for j in G.neighbors(i):
            if j not in I:
                V1.add(j)
    return V1


def PQ(G,I,p=0.5,runs = 20):
    v1 = V1(G,I)
    Q = pd.DataFrame()
    k = runs
    P = {}
    for _ in range(k):
        full_data = EoN.basic_discrete_SIR(G, p,tmax=2, initial_infecteds = I,return_full_data=True)
        df = pd.DataFrame(full_data.transmissions(),columns =['time','u','v'])
        df.time += 1
        df['u'] = df['u'].astype('str')
        df['v'] = df['v'].astype('str')
        df['edge']= df['u'] + ' ' + df['v']
        df['q_uv'] = 1/k

        for v in v1:
            P[v] = P.get(v, 0) + (1/k)* int(full_data.node_status(v,1) == 'I')

        data  = df[df.time == 2][['edge','q_uv']].reset_index(drop=True)

        Q= pd.concat([Q,data])

    Q = Q.groupby('edge').sum().reset_index()
    Q['u'] = Q['edge'].apply(lambda x: x.split(' ')[0]).astype('float').astype('int')
    Q['v'] = Q['edge'].apply(lambda x: x.split(' ')[1]).astype('float').astype('int')
    Q['v1-v2'] = Q['v'].apply(lambda x: x not in v1)
    Q = Q.loc[Q['v1-v2']][['u','v','q_uv']]

    PP = pd.DataFrame(columns=['v','p_v'])
    PP['v'] = P.keys()
    PP['p_v'] = P.values()
    PP= PP.sort_values('v')
    return [PP,Q]
