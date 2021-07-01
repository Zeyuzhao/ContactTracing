import json
import random

import EoN
import networkx as nx
import numpy as np
import math

from typing import Set
from collections import namedtuple

from .utils import find_excluded_contours_edges_PQ2, edge_transmission, allocate_budget
from . import PROJECT_ROOT

SIR_Tuple = namedtuple("SIR_Tuple", ["S", "I1", "I2", "R"])
                
class InfectionState:
    def __init__(self, G:nx.graph, SIR: SIR_Tuple, budget:int, policy:str, transmission_rate:float, transmission_known: bool = False, compliance_rate:float = 1, compliance_known: bool = False, snitch_rate:float = 1):
        self.G = G
        self.SIR = SIR_Tuple(*SIR)
        self.budget = budget
        
        self.transmission_rate = transmission_rate
        self.transmission_known = transmission_known
        self.compliance_rate = compliance_rate
        self.compliance_known = compliance_known
        self.snitch_rate = snitch_rate
       
        #the policies are: none, old, adult, young, equal --> defaults to "equal", which is proportionate to distribution of V1
        self.policy = policy
        self.label_map = {"a": 0, "g": 1, "o": 2, "p": 3, "s": 4}
        self.labels = [0, 1, 2, 3, 4]                               #the labels used for fairness
        self.compliance_map = [.6, .8, .85, .75, .8]                #the compliance_rate average for each label demographic
        
        edge_to_transmission = {}
        self.labels = [0, 1, 2, 3, 4]
        self.compliance_map = [.6, .8, .85, .75, .8]
        
        #Convert duration times to transmission rates
        mean_duration = np.mean(list(nx.get_edge_attributes(G, "duration").values()))
        lambda_cdf = -math.log(1-transmission_rate)/mean_duration
        exponential_cdf = lambda x: 1-math.exp(-lambda_cdf*x)
        
        #Scale noncompliances such that the weighted average of compliances equals the parameter
        frequencies = list(nx.get_node_attributes(self.G, 'age_group').values())
        #If the compliance_rate parameter is negative, it defaults to the compliance mapping for the age groups
        if compliance_rate < 0:
            k = 1
        else:
            k = max(0, len(G.nodes)*(self.compliance_rate-1)/sum([frequencies.count(i)*(self.compliance_map[i]-1) for i in range(len(self.compliance_map))]))
        self.compliance_map = [(1-k*(1-self.compliance_map[i])) for i in range(len(self.compliance_map))]
        
        for node in G.nodes():
            G.nodes[node]['quarantine'] = 0
            
            new_compliance = 1-k*(1-G.nodes[node]['compliance_rate_og'])
            
            if new_compliance < 0:
                G.nodes[node]['compliance_rate'] = 0
            elif new_compliance > 1:
                G.nodes[node]['compliance_rate'] = 1
            else:
                G.nodes[node]['compliance_rate'] = new_compliance
            
            node_compliance_rate = G.nodes[node]['compliance_rate']
            
            for nbr in G.neighbors(node):
                order = (node,nbr)
                if node>nbr: 
                    order = (nbr, node)
                transmission_edge = exponential_cdf(G[node][nbr]["duration"])
                if order not in edge_to_transmission: 
                    edge_to_transmission[order] = transmission_edge
        
        nx.set_edge_attributes(G, edge_to_transmission, 'transmission')
        
        # initialize V1 and V2
        self.set_contours()

    def step(self, quarantine: Set[int]):
        # moves the SIR forward by 1 timestep
        full_data = EoN.discrete_SIR(G = self.G, test_transmission = edge_transmission, args = (self.G,), initial_infecteds=self.SIR.I1 + self.SIR.I2, initial_recovereds=self.SIR.R, tmin=0, tmax=1, return_full_data=True)
        
        S = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'S']
        I1 = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'I']
        I2 = self.SIR.I1
        R = self.SIR.R + self.SIR.I2  
        
        self.SIR = SIR_Tuple(S,I1,I2,R)
        
        #Update the quarantined nodes, each quarantined node is quarantined for 2 timesteps
        for node in self.G.nodes:
            self.G.nodes[node]['quarantine'] -= 1
            #Update labels of nodes that adhere to quarantine
            self.G.nodes[node]['quarantine'] = 2 if node in quarantine and random.random() < self.G.nodes[node]['compliance_rate'] else max(self.G.nodes[node]['quarantine'], 0)
        
        self.set_contours()
    
    def set_contours(self):
        (self.V1, self.V2, self.P, self.Q) = find_excluded_contours_edges_PQ2(self.G, self.SIR.I2, self.SIR.R, self.transmission_rate, self.snitch_rate, self.transmission_known)
    
    def set_budget_labels(self):
        self.budget_labels = allocate_budget(self.G, self.V1, self.budget, self.labels, self.label_map, self.policy)