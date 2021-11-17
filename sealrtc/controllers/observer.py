# authored by Aditya Sengupta

import numpy as np

def identity(measurement, **kwargs):
    return measurement
    
def kfilter(measurement, **kwargs):
    klqg = kwargs.get("klqg")
    u = kwargs.get("u")
    if u is None:
        u = np.zeros((klqg.input_size,))
    # biryani
    klqg.update(measurement[:2]) # adjustment for the # of zern modes
    klqg.predict(u[:2])
    return klqg.measure()
