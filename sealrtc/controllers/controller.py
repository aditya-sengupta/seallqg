from abc import ABC
from functools import partial

import numpy as np

from ..utils import joindata

class Controller(ABC):
    def reset(self):
        pass

    def __call__(self, measurement):
        return self.control_law(self.observe_law(measurement))

    def observe_law(self, measurement):
        return measurement

    def control_law(self, state):
        return np.array([0, 0]), 1

class Openloop(Controller):
    pass

class Integrator(Controller):
    def __init__(self, gain=0.1, leak=1.0):
        self.root_path = joindata("integrator", f"int_gain_{gain}_leak_{leak}")
        self.gain = gain
        self.leak = leak
        self.curr_control = np.zeros((2,))

    def reset(self):
        self.curr_control = np.zeros((2,))

    def observe_law(self, measurement):
        return measurement
        
    def control_law(self, state):
        self.curr_control = (self.gain * state + (1-self.leak) * self.curr_control)
        return self.curr_control, self.leak
