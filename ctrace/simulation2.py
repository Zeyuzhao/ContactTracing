import networkx as nx
import numpy as np
import EoN
import random
import gym

from .problem import *


class SimulationState(gym.Env):
    
    def __init__(self, G:nx.graph, SIR_real, SIR_known, transmission_rate:float, compliance_rate:float, global_rate:float):
        
        self.SIR_real: InfectionState = InfectionState(G, SIR_real)
        self.SIR_known: InfectionState = InfectionState(G, SIR_known)
        self.transmission_rate = transmission_rate
        self.compliance_rate = compliance_rate
        self.global_rate = global_rate
        
    #idk how to use json loading stuff, this is on you Zach, make it pretty please
    def load(self, G:nx.graph, file):
        pass
        
        
    def save(self):
        pass
     
    # equivalent to our previous initial function 
    def generate(self, G:nx.graph, initial_infections:int, ):
        
    def step(self, quarantine_known):
        
        size = np.random.binomial(len(quarantine_known),compliance_rate)
        quarantine_real = set(random.sample(quarantine_known, size))
        
        # moves the timestep forward by 1
        SIR_real.step(transmission_rate, quarantine_real)
        SIR_known.step(transmission_rate, quarantine_known)
        
        # need to do post processing of I_known, but we don't know how they want this yet 
        

class InfectionState:
    
    def __init__(self, G:nx.graph, (S,I,R)):
        
        self.graph = G
        self.SIR = (S,I,R)
        self.quarantined = ([],[],[])
        
    def step(self, p, to_quarantine):
        
        #this might need to be edited slightly depending on if we are assuming SIR to be lists vs. sets
        full_data = EoN.basic_discrete_SIR(G=G, p=p, initial_infecteds=self.SIR[1], initial_recovereds=self.SIR[2]+[item for sublist in self.quarantined for item in sublist], tmin=0, tmax=1, return_full_data=True)

        S = [k for (k, v) in full_data.get_statuses(time=timesteps).items() if v == 'S']
        I = [k for (k, v) in full_data.get_statuses(time=timesteps).items() if v == 'I']
        R = [k for (k, v) in full_data.get_statuses(time=timesteps).items() if v == 'R']
        
        # un-quarantine people from previous time step
        for node in quarantined[0]:
            self.SIR[0].add(node)
        
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
        
        
        self.SIR = (S,I,R)
        
