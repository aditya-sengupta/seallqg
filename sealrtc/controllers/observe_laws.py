"""
Observer laws.
These are all functions with a measurement as their first argument, 
and further arguments that are filled in by the Controller constructor.
"""

import numpy as np

def identity(measurement):
    return measurement
    
def kfilter(measurement, klqg):
    # check out order of operations here
    klqg.predict(klqg.curr_control)
    klqg.update(measurement[:2]) # TODO generalize
    return klqg.measure()
