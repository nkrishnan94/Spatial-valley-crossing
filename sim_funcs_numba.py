import numpy as np
from numpy.random import choice
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
from numba import jit
##standing wave solution of Fisher wave

def standing_wave(y0,x,D,rw):
    w = y0[0]      ##initial value for wave profile at x =0, i.e. w(x=0)
    z = y0[1]      ##initial value for rate of change of profile w.r.t. position x , at x=0 i.e. dw/dx(x=0)
    dwdx = z
    dzdx =(-2*((rw*D)**.5)*dwdx -w*rw*(1-w))/D ## fisher equation in comoving frame
    
    return [dwdx,dzdx]


def initialize(K,n_allele,mu):
    ## generate standing wave
    stand = odeint(standing_wave,[1,-(2*2**.5)/K],np.arange(70),args=(2*2**.5,1))[:,0]
    
    ## cuttoff non integer cell density based off of carry capacity K
    w_0 = (K*stand).astype(int)
    w_0 = w_0[w_0>1]
    
    ## subtract wild type cells from carrying capacity to get 'emopty' particle count
    L = np.vstack(((K-w_0),w_0)).T
    L = np.append(L,np.zeros((len(w_0),n_allele-1)),axis=1)

    ##array strucutre of an empty deme to be added on as needed 
    L_empty= np.append([K],np.zeros(n_allele,dtype=int))

    ## add on some number of empty demes
    for i in range(500):
        L= np.append(L,[L_empty],axis=0)
        
    return L.astype(int), L_empty



## given  ab array with counts and an array withprobabilities return index from first array
# faster than np.random.choice for smallish arrays
@njit
def choice(options,probs):
    x = np.random.rand()
    cum = 0
    for i,p in enumerate(probs):
        cum += p
        ##sum of probability must be 1
        if x < cum:
            break
    return options[i]

@njit
def run_spatial(n_gen,## nunmber of gnerations
                K, ## population size
                landscape, ## fitness landscape (Growthrates of each genotype (should be <<1))
                mu,
                L_init):  ## mutation rate
    
    ## number of alleles in system (based of off provided landscape)
    n_allele = len(landscape)

    ##initialize probability matrix
    #L = L_init
    L_empty = L_init[-1]
    L=L_init
    P = np.ones((n_allele+1,n_allele+1))
    P[0,1:] = 1 - landscape
    
    ## list of allele number - pre-established so array doesnt have to be regenerated
    alleles = np.arange(n_allele+1)
    
    #slots for picked alleles each iteration to be stored, so array doesnt havent to be regerated each time
    picks = np.array([0,0,0,0,0])
    
    #sstore trace history
    L_history= np.expand_dims(L,0)

        
    for t in range(n_gen):
        for dt in range(K):
            
            #number of demes with a non empty particle (+2)
            n_demes = np.where(L[:,0]!=K)[0][-1] +2

            #pick adjacent demes tobe swapped, pick demes for duplication and mutation
            ind_1 = np.random.randint(0, n_demes)
            if (np.random.random() < .5 and ind_1 != 0) or ind_1 == n_demes - 1:
                ind_2 = ind_1 - 1
            else:
                ind_2 = ind_1 + 1
            neighb = np.array([ind_1, ind_2])
            dup_deme, mut_deme = np.random.randint(0,n_demes,2)


            #dmigration: pick two cells from each of the selected demes, and swap them
            for i in range(2):
                picks[i] = choice(alleles, L[neighb][i]/K)
                
            for inds in [[0,1],[1,0]]:
                L[neighb[inds[0]],picks[inds[0]]] -= 1
                L[neighb[inds[0]],picks[inds[1]]] += 1


            #duplication: pick two cells from the selected deme and echange first with copy of second according to
            #3 probability matrix
            for i in range(2,4):
                picks[i] = choice(alleles,L[dup_deme]/K)

            if P[picks[2],picks[3]] > np.random.random():
                L[dup_deme,picks[2]] += 1
                L[dup_deme,picks[3]] -= 1


            ##mutation
            mut_deme = np.random.randint(n_demes)
            picks[4] = choice(alleles,L[mut_deme]/K)
            #picks.append(choice(alleles,L[mut_deme]/K))
            if mu>np.random.random():
                #3 only particles that are not empty spaces and are not the 'peak' in the landscape strucutre can mutate
                if picks[4] != n_allele and picks[4] != 0:
                    ## mutate cell and fromat in terms of cell counts i.e. [empty cell count,...,chosen cell count]

                    ##remove original cell and add mutated cell to cell counts
                    L[mut_deme,picks[4]] -=1
                    L[mut_deme,picks[4]+1] +=1
            ##track how many demes are to be omitted
            shift = 0
            while L[0,0]<int(.02*K):
                L=L[1:,:]
                shift+=1
            #if L[0,0]<int(.02*K):
            #    shift = np.where(L[:,0]<int(.02*K))[-1][0]
            #    L = L[shift:,:]
            for i in range(shift):
                L=np.append(L,np.expand_dims(L_empty,0),axis=0)


        L_history = np.concatenate((L_history, np.expand_dims(L,0)),axis=0)

    return L_history



## run simulation until a fixation event occurs (fixation to some threshold 'thresh')


@njit
def fix_time_spatial(K, ## population size
                     landscape, ## fitness landscape (Growthrates of each genotype (should be <<1))
                     mu,
                     L_init,
                     thresh):
     ## number of alleles in system (based of off provided landscape)
    n_allele = len(landscape)

    ##initialize probability matrix
    #L = L_init
    L_empty = L_init[-1]
    L=L_init
    P = np.ones((n_allele+1,n_allele+1))
    P[0,1:] = 1 - landscape
    
    ## list of allele number - pre-established so array doesnt have to be regenerated
    alleles = np.arange(n_allele+1)
    
    #slots for picked alleles each iteration to be stored, so array doesnt havent to be regerated each time
    picks = np.array([0,0,0,0,0])
    


        
    fixed=False
    muts_arise = False
    t = 0

    while not fixed:
            
        #number of demes with a non empty particle (+2)
        n_demes = np.where(L[:,0]!=K)[0][-1] +2

        #pick adjacent demes tobe swapped, pick demes for duplication and mutation
        ind_1 = np.random.randint(0, n_demes)
        if (np.random.random() < .5 and ind_1 != 0) or ind_1 == n_demes - 1:
            ind_2 = ind_1 - 1
        else:
            ind_2 = ind_1 + 1
        neighb = np.array([ind_1, ind_2])
        dup_deme, mut_deme = np.random.randint(0,n_demes,2)


        #dmigration: pick two cells from each of the selected demes, and swap them
        for i in range(2):
            picks[i] = choice(alleles, L[neighb][i]/K)

        for inds in [[0,1],[1,0]]:
            L[neighb[inds[0]],picks[inds[0]]] -= 1
            L[neighb[inds[0]],picks[inds[1]]] += 1


        #duplication: pick two cells from the selected deme and echange first with copy of second according to
        #3 probability matrix
        for i in range(2,4):
            picks[i] = choice(alleles,L[dup_deme]/K)

        if P[picks[2],picks[3]] > np.random.random():
            L[dup_deme,picks[2]] += 1
            L[dup_deme,picks[3]] -= 1


        ##mutation
        mut_deme = np.random.randint(n_demes)
        picks[4] = choice(alleles,L[mut_deme]/K)
        #picks.append(choice(alleles,L[mut_deme]/K))
        if mu>np.random.random():
            #3 only particles that are not empty spaces and are not the 'peak' in the landscape strucutre can mutate
            if picks[4] != n_allele and picks[4] != 0:
                ## mutate cell and fromat in terms of cell counts i.e. [empty cell count,...,chosen cell count]

                ##remove original cell and add mutated cell to cell counts
                L[mut_deme,picks[4]] -=1
                L[mut_deme,picks[4]+1] +=1
        ##track how many demes are to be omitted
        shift = 0
        while L[0,0]<int(.02*K):
            L=L[1:,:]
            shift+=1
        #if L[0,0]<int(.02*K):
        #    shift = np.where(L[:,0]<int(.02*K))[-1][0]
        #    L = L[shift:,:]
        for i in range(shift):
            L=np.append(L,np.expand_dims(L_empty,0),axis=0)
            
        ##check if beneficial mutant has arised, if it previously didnt exicst
        if (np.sum(L[:,-1]) !=0) ==True and muts_arise ==False:
            ## record time
            arise_time = t
        muts_arise = (np.sum(L[:,-1]) !=0)
        ## check if fixed
        fixed = np.sum(L[:,1:n_allele])==0
        t+=1



    return L, arise_time




#Run the automaton
#Implements cell division. The division rates are based on the experimental data

def run_mixed(n_gen,fit_land,  # number of genetaions Fitness landscape (growth rates - should be <<1)
                  mut_rate=0.1,  # probability of mutation per generation
                  max_cells=10**5,  # Max number of cells 
                  init_counts=None,
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
    for gen in range(n_gen):

        n_cells = np.sum( counts )

        # Scale division rates based on carrying capacity

        cell_types =  np.repeat(np.arange(n_allele+1),counts)
        cell_types = duplication(cell_types,max_cells,P)
        cell_types = mutation(cell_types,mut_rate,max_cells,n_allele)
        counts = np.bincount(cell_types, minlength=n_allele+1)


        count_history.append(counts) 
    return count_history
#Run the automaton
#Implements cell division. The division rates are based on the experimental data

def fix_time_mixed(fit_land,  # Fitness landscape
                  mut_rate=0.1,  # probability of mutation per generation
                  max_cells=10**5,  # Max number of cells 
                  init_counts=None,
                  thresh = .9
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
        if sum(init_counts) !=max_cells:
            raise 'sum of initial counts must be equal to carrying capactiy'
        counts = init_counts
    counts= counts.astype(int)
    fixed = False
    t = 0
    muts= 0
    muts_arise =False
    while not fixed:

        n_cells = np.sum( counts )

        # Scale division rates based on carrying capacity
            
        cell_types =  np.repeat(np.arange(n_allele+1),counts)
        cell_types = duplication(cell_types,max_cells,P)
        cell_types = mutation(cell_types,mut_rate,max_cells,n_allele)
        counts = np.bincount(cell_types, minlength=n_allele+1)
       

        t+=1
        if (counts[-1] !=0) ==True and muts_arise ==False:
            arise_time = t
        muts_arise = (counts[-1] !=0)
        fixed = counts[-1]> (thresh*max_cells)
    return counts, t,arise_time

    
