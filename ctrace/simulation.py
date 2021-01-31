import json
import random

import EoN
import gym
import networkx as nx
import numpy as np
from typing import Set
import random
from . import PROJECT_ROOT
from collections import namedtuple
from typing import Set
SIR_Tuple = namedtuple("SIR_Tuple", ["S", "I", "R"])

class SimulationState(gym.Env):
    
    def __init__(self, G:nx.graph, SIR_real: SIR_Tuple, SIR_known: SIR_Tuple, budget: int, transmission_rate:float, compliance_rate:float, global_rate:float):
        self.G = G
        self.SIR_real = SIR_real
        self.SIR_known = SIR_known

        self.compliance_rate = compliance_rate
        self.global_rate = global_rate
        self.budget = budget
        self.transmission_rate = transmission_rate
    
    @classmethod
    def from_generate(cls, G, budget, transmission_rate, compliance_rate, global_rate):
        SIR_real = cls.generate(G)
        SIR_known = SIR_Tuple(list(G), [], [])
        return cls(G, SIR_real, SIR_known, budget, transmission_rate, compliance_rate, global_rate)
        
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
            "S_real": self.SIR_real[0],
            "I_real": self.SIR_real[1],
            "R_real": self.SIR_real[2],
            "S_known": self.SIR_known[0],
            "I_known": self.SIR_known[1],
            "R_known": self.SIR_known[2],
            "budget": self.budget,
            "transmission_rate": self.transmission_rate,
            "compliance_rate": self.compliance_rate,
            "global_rate": self.global_rate
        }
        
        with open(PROJECT_ROOT / "data" / "SIR_Cache" / file, 'w') as outfile:
            json.dump(to_save, outfile)
            
    # Need to call init before this?
    @staticmethod
    def generate(G:nx.graph, steps = 5, initial_infection_frac=0.0001):
        full_data = EoN.basic_discrete_SIR(G=G, p=0.5, rho=initial_infection_frac,
        tmin = 0, tmax=steps, return_full_data=True)

        S = [k for (k, v) in full_data.get_statuses(
        time=1).items() if v == 'S']
        I = [k for (k, v) in full_data.get_statuses(
        time=1).items() if v == 'I']
        R = [k for (k, v) in full_data.get_statuses(
        time=1).items() if v == 'R']
        return SIR_Tuple(S, I, R)


    # TODO: Adapt to indicators over entire Graph G
    def step(self, quarantine_known: Set[int]):
        # Sample each member independently with probability compliance_rate
        quarantine_real = {i for i in quarantine_known if random.random() < self.compliance_rate}
        
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