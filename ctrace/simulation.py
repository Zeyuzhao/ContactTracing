import json
import random

import EoN
import gym
import networkx as nx
import numpy as np
from typing import Set

from . import PROJECT_ROOT
from collections import namedtuple
SIR_Tuple = namedtuple("SIR_Tuple", ["S", "I", "R"])

class SimulationState(gym.Env):
    
    def __init__(self, G:nx.graph, SIR_real: SIR_Tuple, SIR_known: SIR_Tuple, budget: int, transmission_rate:float, compliance_rate:float, global_rate:float):
        self.G = G
        self.SIR_real: InfectionInfo = InfectionInfo(G, SIR_real, budget, transmission_rate)
        self.SIR_known: InfectionInfo = InfectionInfo(G, SIR_known, budget, transmission_rate)
        self.compliance_rate = compliance_rate
        self.global_rate = global_rate
        
    #idk how to use json loading stuff, this is on you Zach, make it pretty please
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
        size = np.random.binomial(len(quarantine_known),self.compliance_rate)
        quarantine_real = set(random.sample(quarantine_known, size))
        
        # moves the timestep forward by 1
        self.SIR_real.step(quarantine_real)
        self.SIR_known.step(quarantine_known)
        
        # need to do post processing of I_known, but we don't know how they want this yet 
        

class InfectionInfo:
    
    def __init__(self, G:nx.graph, SIR: SIR_Tuple, budget:int, transmission_rate:float):
        self.G = G
        self.SIR = SIR_Tuple(*SIR)
        self.transmission_rate = transmission_rate
        self.budget = budget
        self.quarantined = ([],[],[])

    def step(self, to_quarantine):
        
        #this might need to be edited slightly depending on if we are assuming SIR to be lists vs. sets
        full_data = EoN.basic_discrete_SIR(G=self.G, p=self.transmission_rate, initial_infecteds=self.SIR[1], initial_recovereds=self.SIR[2] + [item for sublist in self.quarantined for item in sublist], tmin=0, tmax=1, return_full_data=True)

        S = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'S']
        I = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'I']
        R = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'R']
        
        # un-quarantine people from previous time step
        for node in self.quarantined[0]:
            self.SIR[0].append(node)
        
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

        self.SIR = SIR_Tuple(S,I,R)