# authored by Aditya Sengupta

import numpy as np
from abc import ABC, abstractmethod

from ..optics import tt_to_dmc, getdmc

class Controller(ABC):
    @abstractmethod
    def control(measurement, dmcurr=None):
        """
        A controller interacts with 

        Arguments
        ---------
        measurement : np.ndarray, (2,)
        The (tilt, tip) measurement to work off.

        dmcurr : np.ndarray, (ydim, xdim) = (32, 32) here
        The current DM command (if needed for, e.g., a leaky integrator).
        Optional in case of simulation mode.

        Returns
        -------
        command : np.ndarray, (ydim, xdim)
        The command to be put on the DM.

        Additional kwargs will be specified as needed.
        """
        raise NotImplementedError

class OpenLoop(Controller):
    def control(self, measurement):
        return getdmc()

class Integrator(Controller):
    """
    A simple integrator.
    """
    def __init__(self, gain=0.1, leak=1.0):
        self.gain = gain
        self.leak = leak

    def control(self, measurement):
        dmcn = tt_to_dmc(measurement)
        return self.gain * dmcn + self.leak * getdmc()

class KalmanIntegrator(Controller):
    """
    Integrator control, but based on an underlying Kalman filter's state prediction.
    """
    def __init__(self, gain=0.1, leak=1.0):
        self.gain = gain
        self.leak = leak
        self.x = None
        self.kf = None

    def control(self, measurement):
        self.x = self.kf.predict(self.kf.update(self.x, measurement))
        dmcn = tt_to_dmc(self.kf.measure(self.x))
        return self.gain * dmcn + self.leak * getdmc()

class LQGController(Controller):
    """
    Linear-quadratic-Gaussian control.
    """
    def __init__(self):
        pass

    def control(self, measurement):
        pass
