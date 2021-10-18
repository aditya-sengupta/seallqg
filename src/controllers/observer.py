# authored by Aditya Sengupta

import numpy as np

def identity(measurement, **kwargs):
    return measurement

def make_kf_observer(klqg):
    uzero = np.zeros((klqg.input_size,))
    def kfilter(measurement, **kwargs):
        u = kwargs.get("u")
        if u is None:
            u = uzero
        klqg.update(measurement[:2]) # adjustment for the # of zern modes
        klqg.predict(u)
        return klqg.measure()

    return kfilter
