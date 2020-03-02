import numpy as np
from numpy.random import choice
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
from numba import jit
##standing wave solution of Fisher wave


## given  ab array with counts and an array withprobabilities return index from first array
# faster than np.random.choice for smallish arrays

def choice(options,probs):
    x = np.random.rand()
    cum = 0
    for i,p in enumerate(probs):
        cum += p
        ##sum of probability must be 1
        if x < cum:
            break
    return options[i]


##generate a standing wave solution of fisher equations - i.e. an 'established' wave front

def standing_wave(y0,x,D,rw):
    w = y0[0]      ##initial value for wave profile at x =0, i.e. w(x=0)
    z = y0[1]      ##initial value for rate of change of profile w.r.t. position x , at x=0 i.e. dw/dx(x=0)
    dwdx = z
    dzdx =(-2*((rw*D)**.5)*dwdx -w*rw*(1-w))/D ## fisher equation in comoving frame
    
    return [dwdx,dzdx]

### given a deme return that deme and a random neighbor

def rand_neighbors(n_demes):
    ind_1 = np.random.randint(0, n_demes)
    if (np.random.random() < .5 and ind_1 != 0) or ind_1 == n_demes - 1:
        ind_2 = ind_1 - 1
    else:
        ind_2 = ind_1 + 1
    return np.array([ind_1, ind_2])


##convert array of cell counts for each fitness
## to an array for each cell with its identity (fitness)

def counts_to_cells(counts,n_allele):
    cells = np.repeat(np.arange(n_allele+1),counts)
    return cells

##convert array of each cell with its identity (fitness)cell counts for each fitness
## to an array of cell counts for each fitness  

def cells_to_counts(cells,n_allele):
    counts = np.bincount(cells, minlength=n_allele+1)
    return counts


## initialize array of number of spaces and number wild type cells for each deme 
##using standing wave solution. 

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
    for i in range(140):
        L= np.append(L,[L_empty],axis=0)
    return L.astype(int), L_empty
 
## take two neighboring demes, pick a particle from each and exchange them    


def migration(cell_counts,n_allele,K):
    empty_cnt = np.zeros((2,n_allele+1)).astype(int)
    
    for i in range(2):
        empty_cnt[i,choice(np.arange(n_allele+1), cell_counts[i]/K)] =int(1)
        
        
    for i,j in enumerate(range(2)[::-1]):
        cell_counts[i]+=(empty_cnt[j] -empty_cnt[i])
        
    return cell_counts


#from one chosen deme pick two cells and exchange the first with second one with some proability

def duplication(cell_counts,K,P,n_allele):

    #pick two cells randomly from chosen deme. yes, i know i use list.append but np append was slower
    ## when i timed it 
    picks = []
    for i in range(2):

        picks.append(choice(np.arange(n_allele+1),
                          cell_counts/np.sum(cell_counts)))
            
    ## format chosen cells in terms of cell counts i.e. [empty cell count,...,chosen cell count]
    empty_cnt = np.zeros((2,n_allele+1)).astype(int)
    empty_cnt[:,picks] =1
    
    if all(picks)!=0:
        cell_counts += (empty_cnt[0] - empty_cnt[1])
    else:
            
        if P[tuple(picks)]> np.random.random():
            cell_counts += (empty_cnt[0] - empty_cnt[1])
    return cell_counts




    
## from randomly chosen deme pick random cell and give it a mutation (change its genotype) with
## some probability



def mutation(cell_counts,mu,n_allele):
    empty_cnt= np.zeros(n_allele+1).astype(int)
    empty_cnt[choice(np.arange(n_allele+1),
                          cell_counts/np.sum(cell_counts))] =1
    
    if mu>np.random.random():
        #3 only particles that are not empty spaces and are not the 'peak' in the landscape strucutre can mutate
        if any(np.arange(n_allele+1)[1:-1])==np.nonzero(np.array([0,1,0]))[0]:
            ## mutate cell and fromat in terms of cell counts i.e. [empty cell count,...,chosen cell count]

            ##remove original cell and add mutated cell to cell counts
            cell_counts += np.append(np.array([0]),empty_cnt[:-1],axis=0) - empty_cnt 
    return cell_counts


## shift simulation box

def recenter(L,L_empty, K):
    shift = 0
    ##track how many demes are to be omitted
    while L[0,0]<int(.02*K):
        L=L[1:,:]
        shift+=1

    #for each dropped deme, add one
    for i in range(shift):
        L=np.append(L,[L_empty],axis=0)
    return L


## one update setp includes on migration, duplication and mutation step 

def update(L, ## population
    L_empty, ## empty deme structure
    P, ## porbability matrix for mutation
    K, # population size
    n_allele, ## length of landsacpe
    mu): ##mutation rate
        #L_tip = np.where(L[:,0]!=K)[0][-1]
        

        return L
## run simulation for chosen number of generation

def run_spatial(n_gen,## nunmber of gnerations
    K, ## population size
    landscape ## fitness landscape (Growthrates of each genotype (should be <<1))
    ,mu):  ## mutation rate

    n_allele = len(landscape)

    ##initialize probability matrix
    stand = odeint(standing_wave,[1,-(2*2**.5)/K],np.arange(70),args=(2*2**.5,1))[:,0]
    P = np.ones((n_allele+1,n_allele+1))
    P[0,1:] = 1 - landscape
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
    L_history=[L]
    L_0=np.zeros(L.shape).astype(int)
    #begin evolution
    alleles = np.arange(n_allele+1)
    picks = np.zeros(5).astype(int)
    rands = np.random.random(10**6)
    rands1 =np.random.random(10**6)
    
    for t in range(n_gen):
        for dt in range(K):
            
            L_1 = L_0
            L_1[::2][:-1]=L[1::2]
            L_1[1::2]= L[0::2][:-1]

            for deme in range(len(L_1)):
                cells = np.repeat(alleles,L_1[deme])
                np.random.shuffle(cells)
                swap = P[cells[:int(K/2)],cells[int(K/2):]]> np.random.random()
                cells[int(K/2):][swap] = cells[:int(K/2)][swap]
                np.random.shuffle(cells)
                cells[:]
                L_1[deme] = np.append(np.bincount(cells),np.zeros(n_allele+1-len(np.bincount(cells))))
                for muts in alleles[1:-1]:
                    L_1[deme][[muts,muts+1]]+=np.repeat(np.random.poisson(L[deme][muts]*mu),2)*np.array([-1,1])
            L= L_1    
            

            ##track how many demes are to be omitted
            if L[0,0]<int(.02*K):
                shift = np.where(L[:,0]<int(.02*K))[-1][0]
                for i in range(shift):
                    L=np.append(L,[L_empty],axis=0)

                    ##recenter simulation box

                ##save

            L_history.append(L)
            
    return L_history

## run simulation until a fixation event occurs (fixation to some threshold 'thresh')

def fix_time_spatial(K,landscape,mu,thresh):
    n_allele = len(landscape)
    func_args = [K,n_allele,mu]
    ##initialize probability matrix
    P = np.ones((n_allele+1,n_allele+1))
    P[0,1:] = 1 - landscape
    ##initallize population
    L,L_empty = initialize(*func_args)
    init_pop = np.sum(L[:,1])
    L_history=[L]
    #mutant has not yet fixed
    fixed=False

    t = 0

    while not fixed:
        # perform on esimulation step
        L = update(L,L_empty,P,*func_args)
        ## move simulation box
        L = recenter(L,L_empty,K)

        ##check if beneficial mutant has arised, if it previously didnt exicst
        if (np.sum(L[:,-1]) !=0) ==True and muts_arise ==False:
            ## record time
            arise_time = t
        muts_arise = (np.sum(L[:,-1]) !=0)
        ## check if fixed
        fixed = np.sum(L[:,1:n_allele])<(thresh*init_pop)
        t+=1
    try:
        return L,t, arise_time
    except:
        return L,t




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

    
