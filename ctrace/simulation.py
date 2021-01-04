import itertools
import json

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import EoN

from contact_tracing import *
from constraint import *
from solve import *

#returns the nodes in S, I, R after timesteps 
def initial(G: nx.graph= None, timesteps=5, p=0.1, cache=None, from_cache=None):
    """
    Loads initial SIR set. Either generate set from parameters, or load from cache

    Parameters
    ----------
    G
        The contact tracing graph
    timesteps
        Number of iterations to run for EON
    p
        The transition infection probability
    cache
        The file to save to cache
    from_cache
        Load S, I, R data from cache

    Returns
    -------
    (S: List[int], I: List[int], R: List[int])
        Where S is the list of susceptible nodes,
        I is the list of infected nodes,
        and R is the list of recovered nodes
    """
    if from_cache:
        with open(f'../data/SIR_Cache/{from_cache}', 'r') as infile:
            j = json.load(infile)
            return (j["S"], j["I"], j["R"])

    full_data = EoN.basic_discrete_SIR(G=G, p=p, rho=.0001, tmin=0, tmax=timesteps, return_full_data=True) 
    
    S = [k for (k,v) in full_data.get_statuses(time=timesteps).items() if v == 'S']
    I = [k for (k,v) in full_data.get_statuses(time=timesteps).items() if v == 'I']
    R = [k for (k,v) in full_data.get_statuses(time=timesteps).items() if v == 'R']
    
    print(full_data.I())

    if cache:
        save = {
            "S": S,
            "I": I,
            "R": R,
        }
        with open(f'../data/SIR_Cache/{cache}', 'w') as outfile:
            json.dump(save, outfile)

    return (S, I, R)

def MDP_step(G, S, I_t, R, Q1, Q2, p):
    
    full_data = EoN.basic_discrete_SIR(G=G, p=p, initial_infecteds=I_t, initial_recovereds=R+Q1+Q2, tmin=0, tmax=1, return_full_data=True)
    
    S = [k for (k,v) in full_data.get_statuses(time=1).items() if v == 'S']
    I = [k for (k,v) in full_data.get_statuses(time=1).items() if v == 'I']
    R = [k for (k,v) in full_data.get_statuses(time=1).items() if v == 'R']
    
    return (S, I, R)

def MDP(G: nx.graph, budget, S, I_t, R, p=0.5, iterations=10, method="dependent", visualization=False, verbose=False):
    """
    Simulates a discrete step SIR model on graph G. Infected patients recover within one time period.

    Parameters
    ----------
    G
        The graph G of disease spread
    budget
        The "k" value to quarantine per time step
    S
        The initial susceptible set of nodes
    I_t
        The initial infected set of nodes
    R
        The initial recovered set of nodes
    p
        The probability of transmission (parameter for EoN)
    iterations
        The number of initial iterations to run before quarantine begins
    method
        The type of method to run: none, degree, random, dependent, iterated, optimized
    visualization
        Whether to display matplotlib plots or not
    verbose
        Debugging information
    Returns
    -------
    (recovered, peak)
        recovered - the total number of patients that recovered
        peak - the maximum number of patients infected at one time period

    """
    peak = 0

    Q_infected = []
    Q_susceptible = []
    
    x=[]
    y1=[]
    y2=[]
    y3=[]
    
    if visualization:
        x.append(0)
        y1.append(len(R))
        y2.append(len(I_t))
        y3.append(len(S))

    if iterations == -1:
        iterator = itertools.count(start=0, step=1)
    else:
        iterator = range(iterations)

    for t in iterator:
        if (len(I_t) == 0):
            break

        if verbose:
            print(str(len(I_t)) + " " + str(len(S)) + " " + str(len(R)))

        (val, recommendation) = to_quarantine(G, I_t, R, budget, method=method, p=p)

        (S, I_t, R) = MDP_step(G, S, I_t, R, Q_infected, Q_susceptible, p=p)
        #after this, R will contain Q_infected and Q_susceptible

        #people from previous timestep get unquarantined (some of these people remain in R because they were infected before quarantine)
        #for node in Q_infected:
        #    R.append(node)

        for node in Q_susceptible:
            R.remove(node)
            S.append(node)

        #reset quarantined lists
        Q_infected = []
        Q_susceptible = []
        
        if visualization:
            x.append(t+1)
            y1.append(len(R))
            y2.append(len(I_t))
            y3.append(len(S))
        
        if len(I_t) > peak:
            peak = len(I_t)
        
        #people are quarantined (removed from graph temporarily after the timestep)
        for (k,v) in recommendation.items():
            if v == 1:
                if k in S:
                    S.remove(k)
                    Q_susceptible.append(k)
                elif k in I_t:
                    I_t.remove(k)
                    Q_infected.append(k)
        
    if visualization:
        colors = ["red", "limegreen", "deepskyblue"]
        labels = ["Infected", "Recovered", "Susceptible"]

        fig, ax = plt.subplots()
        ax.stackplot(x, y2, y1, y3, labels=labels, colors=colors)
        ax.legend(loc='upper left')
        ax.set_title("Epidemic Simulation; Quarantine Method: " + method)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Number of People")
        plt.show()
           
    return (len(R), peak)