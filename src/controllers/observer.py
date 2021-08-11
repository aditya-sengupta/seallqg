# authored by Aditya Sengupta

import numpy as np

def identity(measurement, **kwargs):
    return measurement

def kfilter(measurement, x, kf, **kwargs):
    return kf.predict(kf.update(x, measurement))
