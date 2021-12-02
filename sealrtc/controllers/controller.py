"""
Definitions of a generic controller, and two simple cases.
"""

from abc import ABC
from functools import partial

import numpy as np

from ..utils import joindata

class Controller(ABC):
    def reset(self):
        pass

    def __call__(self, measurement):
        self.observe_law(measurement)
        return self.control_law()

    def observe_law(self, measurement):
        pass

    def control_law(self):
        pass

class Openloop(Controller):
    def __init__(self, p=2):
        self.root_path = joindata("openloop", "ol")
        self.leak = 1
        self.name = "openloop"
        self.u = np.zeros((p,))
    
    def control_law(self):
        return self.u

class Integrator(Controller):
    def __init__(self, s=5, p=2, gain=0.1, leak=1.0):
        self.root_path = joindata("integrator", f"int_gain_{gain}_leak_{leak}")
        self.s = s
        self.p = p
        self.gain = gain
        self.leak = leak
        self.u = np.zeros((p,))
        self.state = np.zeros((s,))
        self.name = "integrator"

    def reset(self):
        self.u = np.zeros((self.p,))

    def observe_law(self, measurement):
        self.state = measurement[:self.p]
        
    def control_law(self):
        self.u = -(self.gain * self.state + (1-self.leak) * self.u)
        return self.u
