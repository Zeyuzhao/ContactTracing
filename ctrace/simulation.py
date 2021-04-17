import json
import random

import EoN
import networkx as nx
import numpy as np
import math

from typing import Set
from collections import namedtuple

from .utils import find_excluded_contours_edges, edge_transmission, edge_transmission_hid, allocate_budget
from . import PROJECT_ROOT

SIR_Tuple = namedtuple("SIR_Tuple", ["S", "I1", "I2", "R"])
                
class InfectionState:
    
    def __init__(self, G:nx.graph, SIR: SIR_Tuple, budget:int, transmission_rate:float, compliance_rate:float = 1, partial_compliance:bool = False, I_knowledge:float = 1, discovery_rate:float = 1, snitch_rate:float = 1):
        self.G = G
        self.SIR = SIR_Tuple(*SIR)
        self.budget = budget
        self.transmission_rate = transmission_rate
        self.compliance_rate = compliance_rate
        self.partial_compliance = partial_compliance
        self.discovery_rate = discovery_rate
        self.snitch_rate = snitch_rate
        self.policy = "none"
        self.label_map = {0:"a", 1:"g", 2:"o", 3:"p", 4:"s"}
        self.labels = [0, 1, 2, 3, 4]
        
        node_to_compliance = {}
        edge_to_compliance = {}
        #edge_to_transmission = {}
        compliance_edge = 0
        
        mean_duration = np.mean(list(nx.get_edge_attributes(G, "duration").values()))
        lambda_cdf = -math.log(1-transmission_rate)/mean_duration
        exponential_cdf = lambda x: 1-math.exp(-lambda_cdf*x)
        
        #TODO: Figure out when to modify compliance edges
        for node in G.nodes():
            G.nodes[node]['quarantine'] = 0
            node_to_compliance[node] = compliance_rate
            
            if not partial_compliance: 
                compliance = 0 if random.random()>compliance_rate else 1
            
            for nbr in G.neighbors(node):
                order = (node,nbr)
                
                if node>nbr: 
                    order = (nbr, node)
                
                transmission_edge = exponential_cdf(G[node][nbr]["duration"])
                #edge_to_transmission[order] = transmission_edge
                if partial_compliance:
                    compliance_edge = (0 if random.random()>compliance_rate else 1, transmission_edge)
                else: 
                    compliance_edge = (compliance, transmission_edge)
                    
                if order not in edge_to_compliance: 
                    edge_to_compliance[order] = {node: compliance_edge}
                else: 
                    edge_to_compliance[order][node] = compliance_edge
        
        nx.set_node_attributes(G, node_to_compliance, 'compliance_rate')
        nx.set_edge_attributes(G, edge_to_compliance, 'compliance_transmission')
        
        # initialize V1 and V2
        self.set_contours()
        
    # returns a SimulationState object loaded from a file
    def load(self, G:nx.graph, file):
        with open(PROJECT_ROOT / "data" / "SIR_Cache" / file, 'r') as infile:
            j = json.load(infile)
            
            if G.name != j['G']:
                raise Exception("Graph and SIR not compatible")
                
            SIR_real = (j["S"], j["I"], j["R"])
            
            return SimulationState(G, SIR_real, j['budget'], j["transmission_rate"], j["compliance_rate"], j['I_knowledge'], j['discovery_rate'], j["snitch_rate"])
    
    # saves a SimulationState object to a file
    def save(self, file):
        
        to_save = {
            "G": self.G.name,
            "S": self.SIR[0],
            "I": self.SIR[1],
            "R": self.SIR[2],
            "budget": self.budget,
            "I_knowledge": self.I_knowledge,
            "transmission_rate": self.transmission_rate,
            "compliance_rate": self.compliance_rate,
            "discovery_rate": self.discovery_rate,
            "snitch_rate": self.snitch_rate
        }
        
        with open(PROJECT_ROOT / "data" / "SIR_Cache" / file, 'w') as outfile:
            json.dump(to_save, outfile)


    def step(self, quarantine: Set[int]):
        # moves the SIR forward by 1 timestep
        full_data = EoN.discrete_SIR(G = self.G, test_transmission = edge_transmission, args = (self.G,), initial_infecteds=self.SIR.I1 + self.SIR.I2, initial_recovereds=self.SIR.R, tmin=0, tmax=1, return_full_data=True)
        #full_data = EoN.basic_discrete_SIR(G=self.G, p=self.transmission_rate, initial_infecteds=self.SIR.I1 + self.SIR.I2, initial_recovereds=self.SIR.R, tmin=0, tmax=1, return_full_data=True)
        
        S = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'S']
        I1 = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'I']
        I2 = self.SIR.I1
        R = self.SIR.R + self.SIR.I2  
        
        self.SIR = SIR_Tuple(S,I1,I2,R)
        
        #Update the quarantined nodes, each quarantined node is quarantined for 2 timesteps
        for node in self.G.nodes:
            self.G.nodes[node]['quarantine'] -= 1
            self.G.nodes[node]['quarantine'] = 2 if node in quarantine and self.G.nodes[node]['quarantine']<=0 else max(self.G.nodes[node]['quarantine'], 0)
        
        self.set_contours()
    
    def set_contours(self):
        #For knowledge of which edges are complied along, add parameter compliance_known:bool
        (self.V1, self.V2) = find_excluded_contours_edges(self.G, self.SIR.I2, self.SIR.R, self.discovery_rate, self.snitch_rate)
    
    def set_budget_labels(self):
        self.budget_labels = allocate_budget(self.G, self.V1, self.budget, self.labels, self.label_map, self.policy)
        
        