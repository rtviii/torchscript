import random

import numpy as np


sampled_nodes = []
npop          = 20
n_eachpop     = 500
n_total       = n_eachpop * npop
Nsampled_ind  = 10
sampled_genes = 10000
n_eachpop     = 500
start_pop     = 0
end_pop       = n_eachpop - 1

for k in range(npop):
    smp = random.sample(range(start_pop,end_pop), Nsampled_ind)
    sampled_nodes.append(smp)
    start_pop = start_pop + n_eachpop
    end_pop = end_pop + n_eachpop
    
sampled_nodes = np.array(sampled_nodes).flatten()