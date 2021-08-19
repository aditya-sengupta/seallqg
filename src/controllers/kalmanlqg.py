# authored by Aditya Sengupta

import numpy as np
from copy import copy
from scipy import linalg

from .dare import solve_dare

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
        self.x = np.zeros((self.state_size,))
        self.recompute()

    def recompute(self):
        self.Pobs = solve_dare(self.A.T, self.C.T, self.W, self.V)
        self.Pcon = solve_dare(self.A, self.B, self.Q, self.R)
        self.K = self.Pobs @ self.C.T @ np.linalg.inv(self.C @ self.Pobs @ self.C.T + self.V)
        self.L = -np.linalg.inv(self.R + self.B.T @ self.Pcon @ self.B) @ self.B.T @ self.Pcon @ self.A

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

    def __add__(self, other):
        if self.state_size == 0:
            return other
        elif other.state_size == 0:
            return self
        A = linalg.block_diag(self.A, other.A)
        B = np.hstack((self.B, other.B))
        C = np.hstack((self.C, other.C))
        W = linalg.block_diag(self.W, other.W)
        V = self.V # this is a hacky workaround
        # really there's no read noise that is *uniquely* associated with one process or the other
        # so we just pick one, assuming that if we are adding KalmanLQGs
        # either one's V matrix would be representative of the read noise properties of the whole
        Q = linalg.block_diag(self.Q, other.Q)
        R = self.R # hack
        return KalmanLQG(A, B, C, W, V, Q, R)

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
        B = linalg.block_diag(self.B, other.B)
        C = linalg.block_diag(self.C, other.C)
        W = linalg.block_diag(self.W, other.W)
        V = linalg.block_diag(self.V, other.V)
        Q = linalg.block_diag(self.Q, other.Q)
        R = linalg.block_diag(self.R, other.R)
        return KalmanLQG(A, B, C, W, V, Q, R)

    def predict(self, u):
        self.x = self.A @ self.x + self.B @ u

    def update(self, y):
        self.x = self.x + self.K @ (y - self.C @ self.x)

    def measure(self):
        return self.C @ self.x

    def control(self):
        return self.L @ self.x

    def filter(self, measurements, x0):
        steps = len(measurements)
        assert len(measurements.shape) == 2 and measurements.shape[1] == self.measure_size, "incorrect size for measurements in Kalman filter."
        states = np.empty((steps, self.state_size))
        self.x = x0
        uzero = np.zeros((self.input_size,))

        for (i, m) in enumerate(measurements):
            self.predict(uzero)
            self.update(m)
            states[i] = self.x
        
        return states
