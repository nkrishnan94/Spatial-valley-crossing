import numpy as np
from numpy.random import choice
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint

##standing wave solution of Fisher wave
def standing_wave(y0,x,D,rw):
    w = y0[0]      ##initial value for wave profile at x =0, i.e. w(x=0)
    z = y0[1]      ##initial value for rate of change of profile w.r.t. position x , at x=0 i.e. dw/dx(x=0)
    dwdx = z
    dzdx =(-2*((rw*D)**.5)*dwdx -w*rw*(1-w))/D
    
    return [dwdx,dzdx]

def rand_neighbors(demes):
    ind_1 = np.random.choice(demes)
    left = demes[:ind_1][-1:]
    right = demes[ind_1:][1:2]
    neighb = np.append(left,right).flatten()
    ind_2=choice(neighb)
    neigh = [ind_1,ind_2]
    neigh.sort()
    return np.array(neigh)


def counts_to_cells(counts,n_allele):
    cells = np.repeat(np.arange(n_allele+1),counts)
    return cells
    
def cells_to_counts(cells,n_allele):
    counts = np.bincount(cells, minlength=n_allele+1)
    return counts

def initialize(K,n_allele,mu):
    stand = odeint(standing_wave,[1,-(2*2**.5)/K],np.arange(70),args=(2*2**.5,1))[:,0]
    w_0 = (K*stand).astype(int)
    w_0 = w_0[w_0>1]
    L = np.vstack(((K-w_0),w_0)).T
    L = np.append(L,np.zeros((len(w_0),n_allele-1)),axis=1)



    ##initialize array
    L_empty= np.append([K],np.zeros(n_allele,dtype=int))

    for i in range(50):
        L= np.append(L,[L_empty],axis=0)
    return L.astype(int), L_empty
    
def migration(cells,K):
    cells_1 = cells[0]
    cells_2 = cells[1]
    pick_ind=choice(np.arange(K),2,replace= True)
    picks = np.array([np.arange(K) == pick_ind[i] for i in [0,1]])
    keep =  ~picks
    cells_1 = np.append(cells_1[keep[0]], cells_2[picks[1]])
    cells_2 = np.append(cells_2[keep[1]], cells_1[picks[0]])
    return np.array([cells_1,cells_2])


def duplication(cells,K,P):
    pick_ind=choice(np.arange(K),2,replace= False)
    picks = np.array([np.arange(K) == pick_ind[i] for i in [0,1]])
    r= np.random.random()
    if P[tuple(cells[pick_ind])]> r:
        cells[pick_ind[1]] =cells[pick_ind[0]]
    return cells
    
    
def mutation(cells,mu,K,n_allele):
    pick_ind=choice(np.arange(K))
    r1= np.random.random()
    if mu>r1:
        if cells[pick_ind] != n_allele and cells[pick_ind] !=0:
            cells[pick_ind] = cells[pick_ind] +1
    return cells

def recenter(L,L_empty, K):
    shift = 0
    while L[0,0]<int(.02*K):
        L=L[1:,:]
        shift+=1
    for i in range(shift):
        L=np.append(L,[L_empty],axis=0)
    return L

def update(L,L_empty,P,K,n_allele,mu):

        demes = np.arange(len(L))
        #migration
        neighb = rand_neighbors(demes)
        cells = np.array(list(map(counts_to_cells, L[neighb],2*[n_allele] )))
        cells = migration(cells,K)
        counts =  np.array(list(map(cells_to_counts, cells,2*[n_allele] )))
        L[neighb] = counts

        #duplication
        dup_deme = choice(demes)
        cells = counts_to_cells(L[dup_deme],n_allele)
        cells = duplication(cells,K,P)
        counts = cells_to_counts(cells, n_allele)
        L[dup_deme] = counts

        ##mutation
        mut_deme = choice(demes)
        cells = counts_to_cells(L[mut_deme],n_allele)
        cells = mutation(cells,mu,K,n_allele)
        counts = cells_to_counts(cells, n_allele)
        L[mut_deme] = counts



        return L

def run_spatial(n_gen,K,landscape,mu):
    n_allele = len(landscape)
    func_args = [K,n_allele,mu]
    ##initialize probability matrix
    P = np.ones((n_allele+1,n_allele+1))
    P[0,1:] = 1 - landscape
    
    L , L_empty = initialize(*func_args)
    L_history=[L]
    #begin evolution
    for t in range(n_gen):
        for dt in range(K):
            L = update(L,L_empty,P,*func_args)
            L= recenter(L,L_empty,K)
        L_history.append(L)
    return L_history

def fix_time_spatial(K,landscape,mu):
    n_allele = len(landscape)
    func_args = [K,n_allele,mu]
    P = np.ones((n_allele+1,n_allele+1))
    P[0,1:] = 1 - landscape
    
    L,L_empty = initialize(*func_args)
    L_history=[L]
    #begin evolution
    fixed=False
    t = 0
    while not fixed:
        L = update(L,L_empty,P,*func_args)
        L = recenter(L,L_empty,K)
        fixed = np.sum(L[:,1:n_allele])==0
        t+=1
    return L,t




#Run the automaton
#Implements cell division. The division rates are based on the experimental data
def run_mixed(n_gen,fit_land,  # Fitness landscape
                  mut_rate=0.1,  # probability of mutation per generation
                  max_cells=10**5,  # Max number of cells 
                  init_counts=None,
                  carrying_cap=True,
                  thresh = .02
                  ):
    
    # Obtain transition matrix for mutations

    # Number of different alleles
    n_allele = len(fit_land)
    # Keeps track of cell counts at each generation
    
    P = np.ones((n_allele+1,n_allele+1))
    P[0,1:] = 1 - fit_land

    if init_counts is None:
        counts = np.zeros(n_allele+1)
        counts[1] = 10
    else:
        counts = init_counts
    counts= counts.astype(int)
    fixed = False
    t = 0
    count_history = []
    for gen in n_gen:
        for dt in range(max_cells):
            n_cells = np.sum( counts )

            # Scale division rates based on carrying capacity

            cell_types =  np.repeat(np.arange(n_allele+1),counts)
            cell_types = duplication(cell_types,K,P)
            cell_types = mutation(cell_types,mut_rate,K,n_allele)
            counts = np.bincount(cell_types, minlength=n_allele+1)


            count_history.append(counts)
    return counts
#Run the automaton
#Implements cell division. The division rates are based on the experimental data
def fix_time_mixed(fit_land,  # Fitness landscape
                  mut_rate=0.1,  # probability of mutation per generation
                  max_cells=10**5,  # Max number of cells 
                  init_counts=None,
                  carrying_cap=True,
                  thresh = .02
                  ):
    
    # Obtain transition matrix for mutations

    # Number of different alleles
    n_allele = len(fit_land)
    # Keeps track of cell counts at each generation
    
    P = np.ones((n_allele+1,n_allele+1))
    P[0,1:] = 1 - fit_land

    if init_counts is None:
        counts = np.zeros(n_allele+1)
        counts[1] = 10
    else:
        counts = init_counts
    counts= counts.astype(int)
    fixed = False
    t = 0
    while not fixed:

        n_cells = np.sum( counts )

        # Scale division rates based on carrying capacity
            
        cell_types =  np.repeat(np.arange(n_allele+1),counts)
        cell_types = duplication(cell_types,K,P)
        cell_types = mutation(cell_types,mut_rate,K,n_allele)
        counts = np.bincount(cell_types, minlength=n_allele+1)
       

        t+=1
        fixed = (np.sum(counts[1:n_allele]) ==0)
    return counts, t

    