import json
import random

import EoN
import networkx as nx
import numpy as np

from typing import Set
from collections import namedtuple

from .utils import find_excluded_contours
from . import PROJECT_ROOT

SIR_Tuple = namedtuple("SIR_Tuple", ["S", "I1", "I2", "R"])
                
class InfectionInfo:
    
    def __init__(self, G:nx.graph, SIR: SIR_Tuple, budget:int, transmission_rate:float, compliance_rate:float = 1, I_knowledge:float = 1, discovery_rate:float = 1, snitch_rate:float = 1):
        self.G = G
        self.SIR = SIR_Tuple(*SIR)
        self.budget = budget
        self.transmission_rate = transmission_rate
        self.compliance_rate = compliance_rate
        self.discovery_rate = discovery_rate
        self.snitch_rate = snitch_rate
        self.quarantined = ([],[],[])
        
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
        recovered = self.SIR[2] + self.quarantined[0] + self.quarantined[1] + self.quarantined[2]
        full_data = EoN.basic_discrete_SIR(G=self.G, p=self.transmission_rate, initial_infecteds=self.SIR_real.SIR[1], initial_recovereds=recovered, tmin=0, tmax=1, return_full_data=True)
        
        S = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'S']
        I = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'I']
        R = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'R']
        
        to_quarantine = [u for u in quarantine if
        update_quarantine(to_quarantine
        
    def update_quarantine(self, to_quarantine):
        # un-quarantine people from previous time step
        for node in self.quarantined[0]:
            self.SIR.S.append(node)
            self.SIR.R.remove(node)
            
        self.quarantined = ([],[],[])
        
        # quarantine the new group
        for node in to_quarantine:
            if node in self.SIR.S:
                self.SIR.S.remove(node)
                self.quarantined[0].append(node)
            elif node in self.SIR.I:
                self.SIR.I.remove(node)
                self.quarantined[1].append(node)
            else:
                self.SIR.R.remove(node)
                self.quarantined[2].append(node)
                
    def set_contours(self):
        (self.V1, self.V2) = find_excluded_contours(self.G, self.SIR[1], self.SIR[2] + self.quarantined[0] + self.quarantined[1] + self.quarantined[2], self.discovery_rate, self.snitch_rate)