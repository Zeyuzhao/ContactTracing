import math
import random
import numpy as np


def simplify(alpha:float, beta:float):
    if (alpha<0) | (alpha>1) | (beta<0) | (beta>1):
        raise ValueError('Invalid alpha or beta')
    
    p = random.uniform(0,1)
    
    if alpha+beta==0:
        
        return (0,0,-1,-1)
    
    elif alpha+beta<1:
        
        if p<alpha/(alpha+beta):
            return (-1,0,alpha+beta,-1)
        else:
            return (0,-1,-1,alpha+beta)
        
    elif alpha+beta==1:
        
        if p<alpha:
            return (1,0,-1,-1)
        else:
            return (0,1,-1,-1)
        
    elif alpha+beta < 2:
        
        if p<(1-beta)/(2-alpha-beta):
            return (1,-1,-1,alpha+beta-1)
        else:
            return (-1,1,alpha+beta-1,-1)
        
    else:
        
        return (1,1,-1,-1)
        
def D(p):
    t = len(p)
    sample = np.full(t,-1)
    prob = np.array(p,copy=True)
    
    leaves = np.arange(t)
    np.random.shuffle(leaves)
    
    #each iteration represents a level in the tree
    while t > 1:
        
        new_leaves = []
        
        for i in range(0,t-1,2):            
            
            a = prob[leaves[i]]
            b = prob[leaves[i+1]]
            
            (x,y,new_a,new_b) = simplify(a,b)
            
            if x==-1:
                
                new_leaves.append(leaves[i])
                prob[leaves[i]] = new_a
                sample[leaves[i+1]] = y
                
            elif y==-1:
                
                new_leaves.append(leaves[i+1])
                prob[leaves[i+1]] = new_b
                sample[leaves[i]] = x

            else:
                
                new_leaves.append(leaves[i])
                prob[leaves[i]] = x
                sample[leaves[i+1]] = y
    
        if t%2 == 1:

            new_leaves.append(leaves[-1])
            
        t = len(new_leaves)
        leaves = np.array(new_leaves)
        np.random.shuffle(leaves)

    if t==1:
        
        p = random.uniform(0,1)
        
        if p < prob[leaves[0]]:
            sample[leaves[0]] = 1
        else:
            sample[leaves[0]] = 0
       
    return sample

def D_prime(p):
    l = np.sum(p)
    
    p_prime = np.append(p,[math.ceil(l)-l])
    
    return D(p_prime)[:len(p)]