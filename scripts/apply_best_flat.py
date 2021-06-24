from ancillary_code import *
import numpy as np

bestflat=np.load('bestflat.npy') #load bestflat, which should be an aligned FPM
applydmc(bestflat)