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
    
    def __init__(self, G:nx.graph, SIR_real: SIR_Tuple, SIR_known: SIR_Tuple, budget: int, transmission_rate:tuple, partition: tuple, time_stage: int, compliance_rate:float, global_rate:float, discovery_rate:float, snitch_rate:float):
        self.G = G
        self.SIR_real: InfectionInfo = InfectionInfo(G, SIR_real, budget, transmission_rate, partition, time_stage, 1, 1)
        self.SIR_known: InfectionInfo = InfectionInfo(G, SIR_known, budget, transmission_rate, partition, time_stage, discovery_rate, snitch_rate)
        self.compliance_rate = compliance_rate
        self.global_rate = global_rate
        
    # returns a SimulationState object loaded from a file
    '''def load(self, G:nx.graph, file):
        with open(PROJECT_ROOT / "data" / "SIR_Cache" / file, 'r') as infile:
            j = json.load(infile)
            
            if G.name != j['G']:
                raise Exception("Graph and SIR not compatible")
                
            SIR_real = (j["S_real"], j["I_real"], j["R_real"])
            SIR_known = (j["S_known"], j["I_known"], j["R_known"])
            
            return SimulationState(G, SIR_real, SIR_known, j["transmission_rate"], j["compliance_rate"], j["global_rate"])'''
    
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
    def step(self, quarantine_known: Set[int]):
        
        previous_infected_real = len(self.SIR_real.SIR[1])
        previous_infected_known = len(self.SIR_known.SIR[1])
        # moves the real SIR forward by 1 timestep
        recovered = self.SIR_real.SIR[2] + self.SIR_real.quarantined[0] + self.SIR_real.quarantined[1] + self.SIR_real.quarantined[2]
        full_data = EoN.basic_discrete_SIR(G=self.G, p=self.SIR_real.transmission_rate[self.SIR_real.time_stage], initial_infecteds=self.SIR_real.SIR[1], initial_recovereds=recovered, tmin=0, tmax=1, return_full_data=True)
        
        S = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'S']
        I = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'I']
        R = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'R']
        
        self.SIR_real.SIR = SIR_Tuple(S,I,R)
        
        if(self.SIR_real.time_stage!=2 and len(self.SIR_real.SIR[2])/len(self.G.nodes) > self.SIR_real.partition[self.SIR_real.time_stage]):
            self.SIR_real.time_stage+=1
        '''if(self.SIR_real.time_stage==0 and len(self.SIR_real.SIR[1])/previous_infected_real > self.SIR_real.partition[self.SIR_real.time_stage]):
            self.SIR_real.time_stage+=1
        elif(self.SIR_real.time_stage==1 and len(self.SIR_real.SIR[1])/previous_infected_real < self.SIR_real.partition[self.SIR_real.time_stage]):
            self.SIR_real.time_stage+=1'''
        
        # moves the known SIR forward by 1 timestep
        I = [i for i in self.SIR_known.V1 if i in self.SIR_real.SIR.I]
        S = [i for i in self.SIR_known.SIR.S if i not in I]
        R = self.SIR_known.SIR.R + self.SIR_known.SIR.I + self.SIR_known.quarantined[0] + self.SIR_known.quarantined[1] + self.SIR_known.quarantined[2]
        
        self.SIR_known.SIR = SIR_Tuple(S,I,R)
        
        '''if(self.SIR_known.time_stage==0 and len(self.SIR_known.SIR[1])/previous_infected_known > self.SIR_known.partition[self.SIR_known.time_stage]):
            self.SIR_known.time_stage+=1
        elif(self.SIR_known.time_stage==1 and len(self.SIR_known.SIR[1])/previous_infected_known < self.SIR_known.partition[self.SIR_known.time_stage]):
            self.SIR_known.time_stage+=1'''
        if(self.SIR_known.time_stage!=2 and len(self.SIR_known.SIR[2])/len(self.G.nodes) > self.SIR_known.partition[self.SIR_known.time_stage]):
            self.SIR_known.time_stage+=1
        
        
        # implements the global rate
        difference = [i for i in self.SIR_real.SIR.I if i not in self.SIR_known.SIR.I]
        for node in difference:
            if random.uniform(0,1) <= self.global_rate:
                if node in self.SIR_known.SIR.S:
                    self.SIR_known.SIR.I.append(node)
                    self.SIR_known.SIR.S.remove(node)
                else:
                    self.SIR_known.quarantined[1].append(node)
                    self.SIR_known.quarantined[0].remove(node)
        
        
        # updates the quarantined people
        quarantine_real = {i for i in quarantine_known if random.random() < self.compliance_rate}
        self.SIR_real.update_quarantine(quarantine_real)
        self.SIR_known.update_quarantine(quarantine_known)
        
        
        # resets the V1 and V2
        self.SIR_known.set_contours()
        self.SIR_real.set_contours()
                
class InfectionInfo:
    
    def __init__(self, G:nx.graph, SIR: SIR_Tuple, budget:int, transmission_rate:tuple, partition: tuple, time_stage: int, discovery_rate:tuple, snitch_rate:float):
        self.G = G
        self.SIR = SIR_Tuple(*SIR)
        self.transmission_rate = transmission_rate
        self.time_stage = time_stage
        self.partition = partition
        self.budget = budget
        self.quarantined = ([],[],[])
        self.discovery_rate = discovery_rate
        self.snitch_rate = snitch_rate
        
        # initialize V1 and V2
        self.set_contours()
        
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
        (self.V1, self.V2, self.il_v1, self.il_v2, self.il_v2_nbrs) = find_excluded_contours(self.G, self.SIR[1], self.SIR[2] + self.quarantined[0] + self.quarantined[1] + self.quarantined[2], self.discovery_rate, self.snitch_rate)