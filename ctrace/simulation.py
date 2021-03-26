import json
import random

import EoN
import networkx as nx
import numpy as np

from typing import Set
from collections import namedtuple

from .utils import find_excluded_contours
from . import PROJECT_ROOT

SIR_Tuple = namedtuple("SIR_Tuple", ["S", "I", "R"])

class SimulationState:
    
    def __init__(self, G:nx.graph, SIR_real: SIR_Tuple, SIR_known: SIR_Tuple, budget: int, transmission_rate:float, compliance_rate: list, partial_compliance: bool, global_rate:float, discovery_rate:float, snitch_rate:float):
        self.G = G
        '''
        Compliance is an array of compliance rates per node (by index). Compliance edge weightings are binary {0, 1}. Partial_compliance indicates whether all edges of a given node should be the same.
        The edges are a dict of {u: (compliance, transmission), v:(compliance, transmission)} where u means u-->v and v means v-->u
        '''
        node_to_compliance = {}
        edge_to_compliance = {}
        compliance_edge = 0
        for node in G.nodes():
            G.nodes[node]['quarantine'] = 0
            node_to_compliance[node] = compliance_rate[node]
            if not partial_compliance: compliance_edge = (0 if random.random()>compliance_rate[node] else 1, transmission_rate)
            for nbr in G.neighbors(node):
                order = (node,nbr)
                if node>nbr: order = (nbr, node)
                if partial_compliance: compliance_edge = (0 if random.random()>compliance_rate[node] else 1, transmission_rate)
                if order not in edge_to_compliance: edge_to_compliance[order] = {node: compliance_edge}
                else: edge_to_compliance[order][node] = compliance_edge
        
        nx.set_node_attributes(G, node_to_compliance, 'compliance_rate')
        nx.set_edge_attributes(G, edge_to_compliance, 'compliance_transmission')
        
        
        self.SIR_real: InfectionInfo = InfectionInfo(G, SIR_real, budget, transmission_rate, 1, 1)
        self.SIR_known: InfectionInfo = InfectionInfo(G, SIR_known, budget, transmission_rate, discovery_rate, snitch_rate)

        self.compliance_rate = compliance_rate
        self.global_rate = global_rate
        
    # returns a SimulationState object loaded from a file
    def load(self, G:nx.graph, file):
        with open(PROJECT_ROOT / "data" / "SIR_Cache" / file, 'r') as infile:
            j = json.load(infile)
            
            if G.name != j['G']:
                raise Exception("Graph and SIR not compatible")
                
            SIR_real = (j["S_real"], j["I_real"], j["R_real"])
            SIR_known = (j["S_known"], j["I_known"], j["R_known"])
            
            return SimulationState(G, SIR_real, SIR_known, j["transmission_rate"], j["compliance_rate"], j["global_rate"])
    
    # saves a SimulationState object to a file
    def save(self, file):
        
        to_save = {
            "G": self.SIR_real.G.name,
            "S_real": self.SIR_real[0],
            "I_real": self.SIR_real[1],
            "R_real": self.SIR_real[2],
            "S_known": self.SIR_known[0],
            "I_known": self.SIR_known[1],
            "R_known": self.SIR_known[2],
            "budget": self.SIR_real.budget,
            "transmission_rate": self.SIR_real.transmission_rate,
            "compliance_rate": self.compliance_rate,
            "global_rate": self.global_rate
        }
        
        with open(PROJECT_ROOT / "data" / "SIR_Cache" / file, 'w') as outfile:
            json.dump(to_save, outfile)
    
    # TODO: Adapt to indicators over entire Graph G
    def step(self, to_quarantine: Set[int]):
        
        def edge_transmission(u, v, G):
            if (G.nodes[u]['quarantine'] == 1 and G.nodes[u]['hid'] != G.nodes[v]['hid']):
                return 1 if random.random() < ((1-G[u][v]['compliance_transmission'][u][0]) * G[u][v]['compliance_transmission'][u][1]) else 0
            else:
                return 1 if random.random() < G[u][v]['compliance_transmission'][u][1] else 0
        
        full_data = EoN.discrete_SIR(G = self.G, test_transmission = edge_transmission, args = (self.G,), initial_infecteds=self.SIR_real.SIR[1], initial_recovereds=self.SIR_real.SIR[2], tmin=0, tmax=1, return_full_data=True)
        
        S = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'S']
        I = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'I']
        R = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'R']
        
        self.SIR_real.SIR = SIR_Tuple(S,I,R)
        
        # moves the known SIR forward by 1 timestep
        I = [i for i in self.SIR_known.V1 if i in self.SIR_real.SIR.I]
        S = [i for i in self.SIR_known.SIR.S if i not in I]
        R = self.SIR_known.SIR.R + self.SIR_known.SIR.I
        
        self.SIR_known.SIR = SIR_Tuple(S,I,R)
        
        # implements the global rate
        difference = [i for i in self.SIR_real.SIR.I if i not in self.SIR_known.SIR.I]
        for node in difference:
            if random.uniform(0,1) <= self.global_rate:
                if node in self.SIR_known.SIR.S:
                    self.SIR_known.SIR.I.append(node)
                    self.SIR_known.SIR.S.remove(node)
        
        #updates quarantined
        for node in self.G.nodes:
            self.G.nodes[node]['quarantine'] = 1 if node in to_quarantine else 0
        
        # resets the V1 and V2
        self.SIR_known.set_contours()
        self.SIR_real.set_contours()
                
class InfectionInfo:
    
    def __init__(self, G:nx.graph, SIR: SIR_Tuple, budget:int, transmission_rate:float, discovery_rate:float, snitch_rate:float):
        self.G = G
        self.SIR = SIR_Tuple(*SIR)
        self.transmission_rate = transmission_rate
        self.budget = budget
        #self.quarantined = ([],[],[])
        self.discovery_rate = discovery_rate
        self.snitch_rate = snitch_rate
        
        # initialize V1 and V2
        self.set_contours()
        
    def update_quarantine(self, to_quarantine):
        # un-quarantine people from previous time step
        #for node in self.quarantined[0]:
        #    self.SIR.S.append(node)
        #    self.SIR.R.remove(node)    
        #self.quarantined = ([],[],[])
        #self.quarantine_list = [1 if node in to_quarantine else 0 for node in G.nodes]
        # quarantine the new group
        '''for node in to_quarantine:
            if node in self.SIR.S:
                self.SIR.S.remove(node)
                self.quarantined[0].append(node)
            elif node in self.SIR.I:
                self.SIR.I.remove(node)
                self.quarantined[1].append(node)
            else:
                self.SIR.R.remove(node)
                self.quarantined[2].append(node)'''
                
    def set_contours(self):
        #(self.V1, self.V2) = find_excluded_contours(self.G, self.SIR[1], self.SIR[2] + self.quarantined[0] + self.quarantined[1] + self.quarantined[2], self.discovery_rate, self.snitch_rate)
        (self.V1, self.V2) = find_excluded_contours(self.G, self.SIR[1], self.SIR[2], self.discovery_rate, self.snitch_rate)