# authored by Aditya Sengupta

import numpy as np
from copy import copy
from scipy import linalg

class KFilter:
    def __init__(self, A, C, Q, R, verbose=True):
        self.A, self.C, self.Q, self.R = A, C, Q, R
        self.x = np.zeros((self.state_size,))
        self.compute_gain(verbose)

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

    def compute_gain(self, verbose=True):
        try:
            self.P = linalg.solve_discrete_are(self.A.T, self.C.T, self.Q, self.R)
            if verbose:
                print("Solved discrete ARE.")
        except (ValueError, np.linalg.LinAlgError):
            if verbose:
                print("Discrete ARE solve failed, falling back to iterative solution.")
            P = copy(self.Q)
            lastP = np.zeros_like(self.A)
            iters = 0
            while not np.allclose(lastP, P):
                P = self.A @ P @ self.A.T + self.Q
                K = P @ self.C.T @ np.linalg.inv(self.C @ P @ self.C.T + self.R)
                P = P - K @ self.C @ P
                iters += 1
            self.P = P
            if verbose:
                print("Solved iteratively in {} iterations".format(iters))
        self.K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.R)

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
