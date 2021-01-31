import json
import random

import EoN
import gym
import networkx as nx
import numpy as np
from typing import Set
<<<<<<< HEAD
from collections import namedtuple

from .utils import find_excluded_contours
import random
from . import PROJECT_ROOT
SIR_Tuple = namedtuple("SIR_Tuple", ["S", "I", "R"])

class SimulationState(gym.Env):
    
    def __init__(self, G:nx.graph, SIR_real: SIR_Tuple, SIR_known: SIR_Tuple, budget: int, transmission_rate:float, compliance_rate:float, global_rate:float, discovery_rate:float, snitch_rate:float):
        self.G = G
        self.SIR_real: InfectionInfo = InfectionInfo(G, SIR_real, budget, transmission_rate, 1, 1)
        self.SIR_known: InfectionInfo = InfectionInfo(G, SIR_known, budget, 1, discovery_rate, snitch_rate)
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

    def save(self, file):
        
        to_save = {
            "G": self.SIR_real.G.name,
            "S_real": self.SIR_real.SIR[0],
            "I_real": self.SIR_real.SIR[1],
            "R_real": self.SIR_real.SIR[2],
            "S_known": self.SIR_known.SIR[0],
            "I_known": self.SIR_known.SIR[1],
            "R_known": self.SIR_known.SIR[2],
            "budget": self.SIR_real.budget,
            "transmission_rate": self.SIR_real.transmission_rate,
            "compliance_rate": self.compliance_rate,
            "global_rate": self.global_rate
        }
        
        with open(PROJECT_ROOT / "data" / "SIR_Cache" / file, 'w') as outfile:
            json.dump(to_save, outfile)
            
    # equivalent to our previous initial function 
    def generate(self, G:nx.graph, initial_infections:int):
        raise NotImplementedError

    # TODO: Adapt to indicators over entire Graph G
    def step(self, quarantine_known: Set[int]):
        # Sample each member independently with probability compliance_rate
        quarantine_real = {i for i in quarantine_known if random.random() < self.compliance_rate}
        
        # moves the timestep forward by 1
        self.SIR_real.step(quarantine_real)
        self.SIR_known.step(quarantine_known)
        
        # post processing of I_known
        to_remove = []
        for node in self.SIR_known.SIR.I:
            if node not in self.SIR_real.SIR.I:
                to_remove.append(node)
                
        for node in to_remove:
            self.SIR_known.SIR.I.remove(node)
            self.SIR_known.SIR.S.append(node)
        
        self.SIR_known.set_contours()
        self.SIR_real.set_contours()
        
        # implements the global rate
        """
        for node in self.SIR_real.SIR.I - self.SIR_known.I:
            if random.uniform(0,1) <= self.global_rate:
                self.SIR_known.I.append(node)
                self.SIR_known.S.remove(node)
           """     
            
class InfectionInfo:
    
    def __init__(self, G:nx.graph, SIR: SIR_Tuple, budget:int, transmission_rate:float, discovery_rate:float, snitch_rate:float):
        self.G = G
        self.SIR = SIR_Tuple(*SIR)
        self.transmission_rate = transmission_rate
        self.budget = budget
        self.quarantined = ([],[],[])
        self.discovery_rate = discovery_rate
        self.snitch_rate = snitch_rate
        
        # initialize V1 and V2
        self.set_contours()
        
    def step(self, to_quarantine):
                
        recovered = self.SIR[2] + self.quarantined[0] + self.quarantined[1] + self.quarantined[2]
        full_data = EoN.basic_discrete_SIR(G=self.G, p=self.transmission_rate, initial_infecteds=self.SIR[1], initial_recovereds=recovered, tmin=0, tmax=1, return_full_data=True)

        S = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'S']
        I = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'I']
        R = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'R']
        
        # un-quarantine people from previous time step
        for node in self.quarantined[0]:
            S.append(node)
            R.remove(node)
        
        # quarantine the new group
        for node in to_quarantine:
            if node in S:
                S.remove(node)
                self.quarantined[0].append(node)
            elif node in I:
                I.remove(node)
                self.quarantined[1].append(node)
            else:
                R.remove(node)
                self.quarantined[2].append(node)

        for node in I:
            if node not in self.V1:
                I.remove(node)
                S.append(node)
        
        self.SIR = SIR_Tuple(S,I,R)
        
    def set_contours(self):
        (self.V1, self.V2) = find_excluded_contours(self.G, self.SIR[1], self.SIR[2] + self.quarantined[0] + self.quarantined[1] + self.quarantined[2], self.discovery_rate, self.snitch_rate)