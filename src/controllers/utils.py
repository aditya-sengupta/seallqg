import numpy as np
from scipy import stats, linalg

def log_likelihood(func, data):
    def get_ll(pars):
        pars_model, sd = pars[:-1], pars[-1]
        data_predicted = func(pars_model)
        LL = -np.sum(stats.norm.logpdf(data, loc=data_predicted, scale=sd))
        return LL

    return get_ll

def combine_matrices_for_klqg(base, addons, measure_once=False):
    matrices = []    
    for (i, (Mb, Ma)) in enumerate(zip(base, addons)):
        if measure_once and i == 2: # C
                matrices.append(np.hstack((Mb, Ma)))
        elif measure_once and i == 4: # V
                matrices.append(Ma)
        else:
            matrices.append(linalg.block_diag(Mb, Ma))
    return matrices
