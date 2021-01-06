import itertools
import json

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import EoN
from collections import namedtuple

from .contact_tracing import *
from .constraint import *
from .solve import *
from . import *

SIR_TYPE = namedtuple("SIR_TYPE", ["S", "I_QUEUE", "R", "label"])

# (len(R), peak, total_iterated)
SIM_RETURN = namedtuple("SIM_RETURN", ["num_contracted", "peak_infected", "simulation_length"])

def initial_shock(G: nx.graph, timesteps=5, p=0.1, num_shocks=7, verbose=False):
    full_data = EoN.basic_discrete_SIR(
        G=G, p=0.5, rho=.0001, tmin=0, tmax=1, return_full_data=True)

    for t in range(timesteps):
        # shock_I =  set([i for i in range(n) if random.random() < 0.001])

        # S = set([k for (k,v) in full_data.get_statuses(time=1).items() if v == 'S']).difference(shock_I)
        S = set([k for (k, v) in full_data.get_statuses(
            time=1).items() if v == 'S'])
        I = set([k for (k, v) in full_data.get_statuses(
            time=1).items() if v == 'I'])
        R = set([k for (k, v) in full_data.get_statuses(
            time=1).items() if v == 'R'])

        shock_I = random.sample(S, num_shocks)

        # update S, I to account for shocks
        S = S.difference(shock_I)
        I = I.union(shock_I)

        full_data = EoN.basic_discrete_SIR(
            G=G, p=p, initial_infecteds=I, initial_recovereds=R, tmin=0, tmax=1, return_full_data=True)
        print(len(S), len(R), len(I), len(shock_I), len(S) + len(I) + len(R))

    if verbose:
        print(full_data.I())
    S = set([k for (k, v) in full_data.get_statuses(time=1).items() if v == 'S'])
    I = set([k for (k, v) in full_data.get_statuses(time=1).items() if v == 'I'])
    R = set([k for (k, v) in full_data.get_statuses(time=1).items() if v == 'R'])

    return (list(S), list(I), list(R))


# returns the nodes in S, I, R after timesteps


def initial(G: nx.graph = None, timesteps=5, p=0.1, cache=None, from_cache=None):
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
        with open(PROJECT_ROOT / "data" / "SIR_Cache" / from_cache, 'r') as infile:
            j = json.load(infile)
            return (j["S"], j["I"], j["R"])

    full_data = EoN.basic_discrete_SIR(
        G=G, p=p, rho=.0001, tmin=0, tmax=timesteps, return_full_data=True)

    S = [k for (k, v) in full_data.get_statuses(
        time=timesteps).items() if v == 'S']
    I = [k for (k, v) in full_data.get_statuses(
        time=timesteps).items() if v == 'I']
    R = [k for (k, v) in full_data.get_statuses(
        time=timesteps).items() if v == 'R']

    # print(full_data.I())

    if cache:
        save = {
            "S": S,
            "I": I,
            "R": R,
        }
        with open(PROJECT_ROOT / "data" / "SIR_Cache" / cache, 'w') as outfile:
            json.dump(save, outfile)

    return (S, I, R)


def MDP_step(G, S, I_t, R, Q1, Q2, p):
    full_data = EoN.basic_discrete_SIR(
        G=G, p=p, initial_infecteds=I_t, initial_recovereds=R + Q1 + Q2, tmin=0, tmax=1, return_full_data=True)

    S = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'S']
    I = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'I']
    R = [k for (k, v) in full_data.get_statuses(time=1).items() if v == 'R']

    return (S, I, R)


def MDP(G: nx.graph, budget, S, I_t, R, p=0.5, iterations=10, method="dependent", visualization=False, verbose=False,
        **kwargs):
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
    (recovered, peak, iterations)
        recovered - the total number of patients that recovered
        peak - the maximum number of patients infected at one time period
        iterations - the number of iterations the simulation ran

    """
    peak = 0
    total_iterated = 0
    Q_infected = []
    Q_susceptible = []

    x = []
    y1 = []
    y2 = []
    y3 = []

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
        # Loop until no infected left.
        if len(I_t) == 0:
            total_iterated = t
            break

        if verbose:
            print(str(t) + " " + str(len(I_t)) + " " +
                  str(len(S)) + " " + str(len(R)))

        (val, recommendation) = to_quarantine(
            G, I_t, R, budget, method=method, p=p)

        (S, I_t, R) = MDP_step(G, S, I_t, R, Q_infected, Q_susceptible, p=p)
        # after this, R will contain Q_infected and Q_susceptible

        # people from previous timestep get unquarantined (some of these people remain in R because they were infected before quarantine)
        # for node in Q_infected:
        #    R.append(node)

        for node in Q_susceptible:
            R.remove(node)
            S.append(node)

        # reset quarantined lists
        Q_infected = []
        Q_susceptible = []

        if visualization:
            x.append(t + 1)
            y1.append(len(R))
            y2.append(len(I_t))
            y3.append(len(S))

        if len(I_t) > peak:
            peak = len(I_t)

        # people are quarantined (removed from graph temporarily after the timestep)
        for (k, v) in recommendation.items():
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

    return (len(R), peak, total_iterated)


def shock(S, I, num_shocks):
    shocks = random.sample(S, num_shocks)

    S = S.difference(shocks)
    I = I.union(shocks)

    return (S, I)


def generalized_mdp(G: nx.graph,
                    p: float,  # Required
                    budget: int,  # Required
                    method: str,  # Required
                    MDP_iterations: int,
                    num_shocks: int,  # Required
                    num_initial_infections: int,
                    initial_iterations: int,  # Data
                    iterations_to_recover: int = 1,  # Required
                    cache: str = None,  # Data
                    from_cache: str = None,
                    shock_MDP: bool = False,  # Required
                    visualization: bool = False,  # Required
                    verbose: bool = False,
                    **kwargs):  # Required
    S = set()
    I = set()
    R = set()
    infected_queue = []

    x = []
    y1 = []
    y2 = []
    y3 = []
    
    # Data set up
    if from_cache:
        with open(PROJECT_ROOT / "data" / "SIR_Cache" / from_cache, 'r') as infile:
            j = json.load(infile)
            (S, infected_queue, R) = (
                set(j["S"]), j["I_Queue"], set(j["R"]))

            # Make infected_queue a list of sets
            infected_queue = [set(s) for s in infected_queue]
            I = I.union(*infected_queue)
            if len(infected_queue) != iterations_to_recover:
                raise ValueError(
                    "Infected queue length must be equal to iterations_to_recover")

    else:
        # initialize S, I, R
        I = set(random.sample(range(len(G.nodes)), num_initial_infections))
        S = set([i for i in range(len(G.nodes))]).difference(I)
        R = set()

        # initialize the queue for recovery
        infected_queue = [set() for _ in range(iterations_to_recover)]
        infected_queue.pop(0)
        infected_queue.append(I)
        
        if visualization:
            x.append(0)
            y1.append(len(R))
            y2.append(len(I))
            y3.append(len(S))
    
        if verbose:
            print(0, len(S), len(I), len(R))

        for t in range(initial_iterations):

            full_data = EoN.basic_discrete_SIR(
                G=G, p=p, initial_infecteds=I, initial_recovereds=R, tmin=0, tmax=1, return_full_data=True)

            # update susceptible, infected, and recovered sets
            S = set([k for (k, v) in full_data.get_statuses(
                time=1).items() if v == 'S'])
            new_I = set([k for (k, v) in full_data.get_statuses(
                time=1).items() if v == 'I'])

            (S, new_I) = shock(S, new_I, num_shocks)

            to_recover = infected_queue.pop(0)
            infected_queue.append(new_I)

            I = I.difference(to_recover)
            I = I.union(new_I)
            R = R.union(to_recover)
            
            if visualization:
                x.append(t+1)
                y1.append(len(R))
                y2.append(len(I))
                y3.append(len(S))
        
            if verbose:
                print(t+1, len(S), len(I), len(R), len(new_I))

    if cache:
        save = {
            "S": list(S),
            # convert list of sets into list of queue
            "I_Queue": [list(s) for s in infected_queue],
            "R": list(R),
        }
        with open(PROJECT_ROOT / "data" / "SIR_Cache" / cache, 'w') as outfile:
            json.dump(save, outfile)

    # Running the simulation
    peak = 0
    total_iterated = 0
    Q_infected = []
    Q_susceptible = []

    if MDP_iterations == -1:
        iterator = itertools.count(start=0, step=1)
    else:
        iterator = range(MDP_iterations)

    if verbose:
        print("<======= SIR Initialization Complete =======>")

    for t in iterator:

        # get recommended quarantine
        (val, recommendation) = to_quarantine(
            G, I, R, budget, method=method, p=p)

        # go through one step of the disease spread
        # (S, I, R) = MDP_step(G, S, I, R, Q_infected, Q_susceptible, p=p)

        full_data = EoN.basic_discrete_SIR(G=G, p=p, initial_infecteds=I, initial_recovereds=list(
            R) + Q_infected + Q_susceptible, tmin=0, tmax=1, return_full_data=True)

        S = set([k for (k, v) in full_data.get_statuses(
            time=1).items() if v == 'S'])
        new_I = set([k for (k, v) in full_data.get_statuses(
            time=1).items() if v == 'I'])

        if shock_MDP:
            (S, new_I) = shock(S, new_I, num_shocks)

        to_recover = infected_queue.pop(0)
        infected_queue.append(new_I)

        I = I.difference(to_recover)
        I = I.union(new_I)
        R = R.union(to_recover)

        if visualization:
            x.append(len(x)+1)
            y1.append(len(R))
            y2.append(len(I))
            y3.append(len(S))
        
        if verbose:
            print(t+initial_iterations+1,len(S), len(I), len(R), len(new_I))
        
        if len(I) > peak:
            peak = len(I)
        
        # Loop until no infected left.
        if (MDP_iterations == -1) & (len(I) == 0):
            total_iterated = t + initial_iterations + 1
            break
        
        # people are quarantined (removed from graph temporarily after the timestep)
        for (k, v) in recommendation.items():
            if v == 1:
                if k in S:
                    S.remove(k)
                    Q_susceptible.append(k)
                elif k in I:  # I_t is undefined
                    I.remove(k)
                    Q_infected.append(k)
    
    #while 
    
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
        
    # TODO: Check return statement
    return SIM_RETURN(len(R), peak, total_iterated)
