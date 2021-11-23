import numpy as np

# I hate this

class Integrator:
    def __init__(self, gain=0.1, leak=1.0):
        self.gain = gain
        self.leak = leak
        self.curr_control = np.zeros((2,))

    def control(self, meas):
        self.curr_control = -(self.gain * meas) + self.leak * self.curr_control
        return self.curr_control

    def reset(self):
        self.curr_control = np.zeros((2,))
