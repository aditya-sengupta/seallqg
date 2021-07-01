from tt import *
import numpy as np

bestflat = np.load('/home/lab/blgerard/bestflat.npy') #load bestflat, which should be an aligned FPM
applydmc(bestflat)