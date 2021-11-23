from abc import ABC
from functools import partial

import numpy as np

from ..utils import joindata

class Controller(ABC):
    def __call__(self, measurement):
        return self.control_law(self.observe_law(measurement))

class Openloop(Controller):
    def __init__(self):
        self.root_path = joindata("openloop", "ol")
        self.observe = lambda measurement: measurement
        self.control_law = lambda state: np.array([0, 0]), 1

class Integrator(Controller):
    def __init__(self, gain=0.1, leak=1.0):
        self.root_path = joindata("integrator", f"int_gain_{gain}_leak_{leak}")
        self.gain = gain
        self.leak = leak
        self.observe = lambda measurement: measurement
        self.curr_control = np.zeros((2,))

    def reset(self):
        self.curr_control = np.zeros((2,))
        
    def control_law(self, meas):
        self.curr_control = -(self.gain * meas + self.leak * self.curr_control)
        return self.curr_control, self.leak
