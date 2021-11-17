"""
Observer laws.
These are all functions with a measurement as their first argument, 
and further arguments that are filled in by the Controller constructor.
"""

import numpy as np

def identity(measurement):
    return measurement
    
def kfilter(measurement, klqg):
    klqg.update(measurement[:2]) # TODO generalize
    klqg.predict(klqg.curr_control)
    return klqg.measure()
