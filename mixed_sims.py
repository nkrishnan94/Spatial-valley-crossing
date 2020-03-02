import numpy as np
from sim_funcs import*
import itertools
from datetime import datetime
from tqdm import tqdm 
from scipy.integrate import odeint

##start time
start = datetime.now()

#parameters
mut_rates = [.0001,.0005,.001]
pop_size = [100,500,1000]
fit_lands = [np.array([0.1,0.09999,0.15]),np.array([0.1,0.0999,0.15]),np.array([0.1,0.099,0.15]) ]
trials = 5

#empty list to save
results = []

for K in pop_size:
    for n in range(trials):
        for mu in mut_rates:
            for fl in tqdm(fit_lands):
                ##run simulaiton
                L,t_mix,est_time = fix_time_mixed(fl,  # Fitness landscape
                      mut_rate=mu,  # probability of mutation per generation
                      max_cells=K,  # Max number of cells
                      init_counts=np.array([0,K,0,0]),###sum must equal K
                      thresh = .5)
                #save as dictionary
                results_dict = {'pop. size':K, 
                                'mutation rate': mu, 'landscape':fl,
                                'mutant est. time': est_time,
                                'trial number':n }
                #apppend trial results to lsist
                results.append(results_dict)
                
    #convert results list to array, with the start time for record keeping
    results_arr = np.array([results,start])
    #save
    np.save('results/mixed_results_%s.npy' % start, results_arr)
    
    
#convert results list to array, with the start time for record keeping
results_arr = np.array([results,start])
#save
np.save('results/mixed_results_%s.npy' % start, results_arr)
          
            
            
            
        
    