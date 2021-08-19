# authored by Aditya Sengupta

import numpy as np

def identity(measurement, **kwargs):
    return measurement

def make_kf_observer(klqg):
    def kfilter(measurement, **kwargs):
        klqg.update(measurement)
        klqg.predict()
        return klqg.x.astype(np.float32) # specific

    return klqg
