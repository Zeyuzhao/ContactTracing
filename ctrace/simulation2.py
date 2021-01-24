import networkx as nx

from .problem import *




class SimulationState:
    
    def __init__(self, G:nx.graph, SIR_real, SIR_known, problem:Problem, transmission_rate:float, compliance_rate:float):
        
        self.real: InfectionState = InfectionState(G, SIR_real)
        self.known: InfectionState = InfectionState(G, SIR_known)
        self.strategy: Problem = problem
        self.transmission_rate = transmission_rate
        self.compliance_rate = compliance_rate

    def load(self, G:nx.graph, file):
        pass
        
        
    def save(self):
        pass
        
        
    def step(self):
        
        quarantine_known = strategy.recommend()
        size = np.random.binomial(len(quarantine_known),compliance_rate)
        quarantine_real = random.sample(quarantine_known, size)
        pass
        

class InfectionState:
    
    def __init__(self, G:nx.graph, (S,I,R)):
        
        self.graph = G
        self.SIR = (S,I,R)
        
    def step(self, p, quarantined):
        
        pass
        
