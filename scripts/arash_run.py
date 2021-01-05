import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import concurrent.futures as cf
import random
import math
import EoN
import time

from ctrace.contact_tracing import *
from ctrace.constraint import *
from ctrace.solve import *
from ctrace.simulation import *

def parallel(G, S, I, R, method, name):


    with cf.ProcessPoolExecutor() as executor:
        results = [(k, executor.submit(MDP, G, k, S, I, R, p=0.1, iterations=15, method=method))                                   
                                       for k in range(1300,2001,10)]
        results1 = [(k, executor.submit(MDP, G, k, S, I, R, p=0.1, iterations=15, method=method))                                   
                                       for k in range(1300,2001,10)]
        results2 = [(k, executor.submit(MDP, G, k, S, I, R, p=0.1, iterations=15, method=method))                                   
                                       for k in range(1300,2001,10)]
        results3 = [(k, executor.submit(MDP, G, k, S, I, R, p=0.1, iterations=15, method=method))                                   
                                       for k in range(1300,2001,10)]
        results4 = [(k, executor.submit(MDP, G, k, S, I, R, p=0.1, iterations=15, method=method))                                   
                                       for k in range(1300,2001,10)]

    ind = []
    peak1 = []
    peak2 = []
    peak3 = []
    peak4 = []
    peak5 = []
    total1 = []
    total2 = []
    total3 = []
    total4 = []
    total5 = []

    for i in range(len(results)):
        ind.append(results[i][0])
        peak1.append(results[i][1].result()[0])
        peak2.append(results1[i][1].result()[0])
        peak3.append(results2[i][1].result()[0])
        peak4.append(results3[i][1].result()[0])
        peak5.append(results4[i][1].result()[0])

        total1.append(results[i][1].result()[1])
        total2.append(results1[i][1].result()[1])
        total3.append(results2[i][1].result()[1])
        total4.append(results3[i][1].result()[1])
        total5.append(results4[i][1].result()[1])

    peak1 = np.array(peak1)
    peak2 = np.array(peak2)
    peak3 = np.array(peak3)
    peak4 = np.array(peak4)
    peak5 = np.array(peak5)
    total1 = np.array(total1)
    total2 = np.array(total2)
    total3 = np.array(total3)
    total4 = np.array(total4)
    total5 = np.array(total5)
    avg_peak = (peak1+peak2+peak3+peak4+peak5)/5
    avg_total = (total1+total2+total3+total4+total5)/5
    
    #save dataframe as csv
    df = pd.DataFrame([ind,total1,total2,total3,total4,total5,peak1,peak2,peak3,peak4,peak5,avg_total,avg_peak]).transpose()
    df = df.rename(columns={0:"k",1:"peak1",2:"peak2",3:"peak3",4:"peak4",5:"peak5",6:"total1",7:"total2",8:"total3",9:"total4",10:"total5",11:"avg_peak",12:"avg_total"})
    
    path = "../output/Q4_csvs/" + name
    df.to_csv(path_or_buf=path, index=False)
    
def main():
    G = load_graph('montgomery')
    
    (S, I, R) = initial(G, from_cache="Q4data.json")
    
    #print(len(I))
    
    parallel(G, S, I, R, "random","arash_dependent_data1.csv")
    parallel(G, S, I, R, "dependent","arash_dependent_data2.csv")
    
    
if __name__ == "__main__":
    main()