# authored by Aditya Sengupta

import numpy as np
from .image import stack, getdmc, applydmc
from ..utils import joindata
# from compute_cmd_int import measure_tt

def refresh(verbose=True):
    bestflat = np.load(joindata("bestflats/bestflat.npy"))
    dmc = getdmc()
    applydmc(bestflat)
    imflat = stack(100)
    np.save(joindata("bestflats/imflat.npy"), imflat)
    if verbose:
        print("Updated the flat image.")
    applydmc(dmc)
    return bestflat, imflat

if __name__ == "__main__":
    refresh()
