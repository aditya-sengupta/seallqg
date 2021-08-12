# authored by Aditya Sengupta

import numpy as np

def identity(measurement, **kwargs):
    return measurement

def make_kf_observer(kf):
    def kfilter(measurement, **kwargs):
        kf.update(measurement)
        kf.predict()
        return kf.measure()

    return kfilter
