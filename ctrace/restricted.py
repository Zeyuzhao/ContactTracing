def r_get_prob(G:nx.graph, I, R, k, p, rev_nodes):
    df = pd.read_csv("../data/cville_dem.txt")
    V_1, V_2 = find_excluded_contours(G,I,R)

    unique = df['age_group'].unique()
    map_label = {}

    for i, group in enumerate(unique):
        map_label[group] = i

    #generate label dict for V_1 
    labels = {i: map_label[df[df.pid==rev_nodes[i]]['age_group'].item()] for i in V_1}
    
    #generate label limits by our definition of fairness
    label_limits = [0 for _ in range(len(map_label.keys()))]

    for i in V_1:
        label_limits[labels[i]]+=1
    
    if len(V_1) != 0:
        label_limits = k*np.array(label_limits)/len(V_1)
        label_limits = np.floor(label_limits)

    P, Q = PQ_deterministic(G,I,V_1,p)

    prob = ProbMinExposedRestricted(G=G,
                                    infected=I,
                                    contour1=V_1, 
                                    contour2=V_2,
                                    p1=P,
                                    q=Q,
                                    k=k,
                                    labels=labels,
                                    label_limits=label_limits,
                                    costs=[1 for _ in range(len(G.nodes))])

    return prob

def r_dependent(prob: ProbMinExposedRestricted):

    prob.solve_lp()
    probabilities = prob.getVariables()

    soln = np.array([0.0 for _ in range(len(probabilities))])

    for l in range(len(prob.label_limits)):
        partial_soln = np.array([probabilities[i] if prob.labels[prob.quaran_map[i]]==l else 0 for i in range(len(probabilities))])
        partial_soln = D_prime(partial_soln)

        soln += partial_soln
    
    for (k, v) in enumerate(soln):
        prob.setVariable(k,v)

    prob.solve_lp()
    
    return (prob.objectiveVal, prob.quarantined_solution)

def r_greedy(prob: ProbMinExposedRestricted):
    
    sol = {}
    
    for l in range(len(prob.label_limits)):
        
        members = list(filter(lambda x: prob.labels[x] == l, prob.V1))
        
        weights: List[Tuple[int, int]] = []
        for u in members:
            w_sum = 0
            for v in set(prob.G.neighbors(u)):
                if v in prob.V2:
                    w_sum += prob.q[u][v]
            weights.append((prob.p1[u] * w_sum, u))
        # Get the top k (cost_constraint) V1s ranked by w_u = p_u * sum(q_uv for v in v2)
        weights.sort(reverse=True)
        topK = weights[:int(prob.label_limits[l])]
        topK = {i[1] for i in topK}
        
        for u in members:
            if u in topK:
                sol[u] = 1
            else:
                sol[u] = 0
    
    return (-1, sol)
        
def r_random(prob: ProbMinExposedRestricted):
    
    sol = {}
    
    for l in range(len(prob.label_limits)):
        
        members = list(filter(lambda x: prob.labels[x] == l, prob.V1))
        
        topK = random.sample(members,int(prob.label_limits[l]))
        
        for u in members:
            if u in topK:
                sol[u] = 1
            else:
                sol[u] = 0
    
    return (-1, sol)
    
def to_q(G, I, R, budget, method, p, rev_nodes):
    
    prob = r_get_prob(G, I, R, budget, p, rev_nodes)
    
    if method == "rrandom":
        return r_random(prob)
    elif method == "rweighted":
        return r_greedy(prob)
    elif method == "rdependent":
        return r_dependent(prob)
    else:
        raise ValueError("invalid method")

#generalized_mdp except with new functions for to_quarantine and solvers
def r_mdp(G: nx.graph,
        p: float,  # Required
        budget: int,  # Required
        method: str,  # Required
        MDP_iterations: int,
        num_shocks: int,  # Required
        num_initial_infections: int,
        initial_iterations: int,  # Data
        rev_nodes, #whatever is the most intuitive but reversed
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

            full_data = EoN.basic_discrete_SIR(G=G, p=p, initial_infecteds=I, initial_recovereds=R, tmin=0, tmax=1, return_full_data=True)

            # update susceptible, infected, and recovered sets
            S = set([k for (k, v) in full_data.get_statuses(time=1).items() if v == 'S'])
            new_I = set([k for (k, v) in full_data.get_statuses(time=1).items() if v == 'I'])

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
        (val, recommendation) = to_q(G, I, R, budget, method=method, p=p, rev_nodes=rev_nodes)

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
    
    return SIM_RETURN(len(R), peak, total_iterated)