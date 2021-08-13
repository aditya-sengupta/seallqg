# authored by Aditya Sengupta

import numpy as np
from copy import copy
from scipy import linalg

from .dare import solve_dare

class KFilter:
    def __init__(self, A, C, Q, R, verbose=True):
        self.A, self.C, self.Q, self.R = A, C, Q, R
        self.x = np.zeros((self.state_size,))
        self.P = solve_dare(self.A.T, self.C.T, self.Q, self.R, verbose=verbose)
        self.K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.R)

    @property
    def state_size(self):
        return self.A.shape[0]

    @property
    def measure_size(self):
        return self.C.shape[0]

    def __str__(self):
        return "Kalman filter with state size " + str(self.state_size) + " and measurement size " + str(self.measure_size)

    def __add__(self, other):
        if self.state_size == 0:
            return other
        elif other.state_size == 0:
            return self
        A = linalg.block_diag(self.A, other.A)
        C = np.hstack((self.C, other.C))
        Q = linalg.block_diag(self.Q, other.Q)
        R = self.R # this is a hacky workaround
        # really there's no read noise that is *uniquely* associated with one process or the other
        # so we just pick one, assuming that if we are adding KFilters
        # either one's R matrix would be representative of the read noise properties of the whole
        return KFilter(A, C, Q, R)

    def concat(self, other):
        """
        This differs from addition in that we don't combine the observed state variables.
        In this package, we'll use this only for combining a tip filter with a tilt filter.
        """
        if self.state_size == 0:
            return other
        elif other.state_size == 0:
            return self
        A = linalg.block_diag(self.A, other.A)
        C = linalg.block_diag(self.C, other.C)
        Q = linalg.block_diag(self.Q, other.Q)
        R = linalg.block_diag(self.R, other.R)
        return KFilter(A, C, Q, R)

    def predict(self):
        self.x = self.A @ self.x

    def update(self, y):
        self.x = self.x + self.K @ (y - self.C @ self.x)

    def measure(self):
        return self.C @ self.x

    def run(self, measurements, x0):
        steps = len(measurements)
        assert len(measurements.shape) == 2 and measurements.shape[1] == self.measure_size, "incorrect size for measurements in Kalman filter."
        states = np.empty((steps, self.state_size))
        self.x = x0

        for (i, m) in enumerate(measurements):
            self.predict()
            self.update(m)
            states[i] = self.x
        
        return states
