"""
Implementation of a *generic* Kalman-LQG observer and controller.
"""

import numpy as np
from copy import copy
from scipy.stats import multivariate_normal as mvn
from tqdm import trange, tqdm

from .dare import solve_dare
from ..utils import rms

# I don't really need to keep W, V, Q, R as state attributes
# but just want to be aware in case the DARE solution is not nice

class KalmanLQG:
    """
    Kalman filter information

    x - the state
    A - the time-evolution matrix
    B - the input-to-state matrix
    C - the state-to-measurement matrix
    W - the process noise matrix (covariance around Ax)
    V - the measurement noise matrix (covariance around Cx)
    Q - the state penalty (cost x'Qx)
    R - the input penalty (cost u'Ru)
    """
    def __init__(self, A, B, C, W, V, Q, R, verbose=True):
        self.A, self.B, self.C, self.W, self.V, self.Q, self.R = A, B, C, W, V, Q, R
        self.recompute()

    def recompute(self):
        self.x = np.zeros((self.state_size,))
        self.Pobs = solve_dare(self.A.T, self.C.T, self.W, self.V)
        self.Pcon = solve_dare(self.A, self.B, self.Q, self.R)
        self.K = self.Pobs @ self.C.T @ np.linalg.pinv(self.C @ self.Pobs @ self.C.T + self.V)
        self.L = -np.linalg.pinv(self.R + self.B.T @ self.Pcon @ self.B) @ self.B.T @ self.Pcon @ self.A
        self.process_dist = mvn(cov=self.W, allow_singular=True)
        self.measure_dist = mvn(cov=self.V, allow_singular=True)
        self.curr_control = np.zeros((self.input_size,))

    @property
    def state_size(self):
        return self.A.shape[0]

    @property
    def measure_size(self):
        return self.C.shape[0]

    @property
    def input_size(self):
        return self.B.shape[1]

    def __str__(self):
        return "Kalman-LQG observer and controller with state size " + str(self.state_size) + ", input size " + str(self.input_size) + " and measurement size " + str(self.measure_size)

    def predict(self, u):
        self.x = self.A @ self.x + self.B @ u

    def update(self, y):
        self.x = self.x + self.K @ (y - self.C @ self.x)

    def measure(self):
        return self.C @ self.x

    def control(self):
        self.curr_control = self.L @ self.x
        return self.curr_control

    def filter(self, measurements, x0):
        steps = len(measurements)
        assert len(measurements.shape) == 2 and measurements.shape[1] == self.measure_size, f"incorrect size for measurements in Kalman filter: expected (nsteps, {self.measure_size}) but got {measurements.shape}"
        states = np.empty((steps, self.state_size))
        self.x = x0
        uzero = np.zeros((self.input_size,))

        for (i, m) in enumerate(tqdm(measurements)):
            self.predict(uzero)
            self.update(m)
            states[i] = self.x
        
        return states

    def sim_control(self, nsteps=1000, x0=None):
        x_init = copy(self.x)
        if x0 is None:
            self.x = self.process_dist.rvs()
        else:
            self.x = x0
        states = np.zeros((nsteps, self.state_size))
        states[0] = self.x
        for i in trange(1, nsteps):
            u = self.control()
            self.predict(u)
            x = self.A @ states[i-1] + self.B @ u + self.process_dist.rvs()
            y = self.C @ x + self.measure_dist.rvs()
            self.update(y)
            states[i] = x
    
        self.x = x_init
        return states @ self.C.T

    def sim_process(self, nsteps=1000, x0=None):
        states = np.zeros((nsteps, self.state_size))
        if x0 is None:
            states[0] = self.process_dist.rvs()
        else:
            states[0] = x0
        states[0] = self.process_dist.rvs()
        for i in trange(1, nsteps):
            states[i] = self.A @ states[i-1] + self.process_dist.rvs()
            
        return states @ self.C.T

    def sim_control_nokf(self, nsteps=1000, x0=None):
        states = np.zeros((nsteps, self.state_size))
        if x0 is None:
            states[0] = self.process_dist.rvs()
        else:
            states[0] = x0
        for i in trange(1, nsteps):
            states[i] = (self.A + self.B @ self.L) @ states[i-1] + self.process_dist.rvs()
            
        return states @ self.C.T

    def improvement(self, kfilter=True, **kwargs):
        if kfilter:
            return rms(self.sim_process(**kwargs)) / rms(self.sim_control(**kwargs))
        else:
            return rms(self.sim_process(**kwargs)) / rms(self.sim_control_nokf(**kwargs))
            