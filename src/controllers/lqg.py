# authored by Aditya Sengupta

import numpy as np
from copy import copy
from scipy import linalg

from .dare import solve_dare

class LQGController:
    """
    Sufficient information to build an LQG controller:

    x - the state
    A - the time-evolution matrix
    B - the input-to-state matrix
    Q - the state penalty (cost x'Qx)
    R - the input penalty (cost u'Ru)

    This could be integrated with the KFilter 
    into one big "LQG state space description" class
    but for now it's separate to test observer-controller separation
    """
    def __init__(self, A, B, Q, R, verbose=True):
        self.A, self.B, self.Q, self.R = A, B, Q, R
        self.x = np.zeros((self.state_size,))
        self.P = solve_dare(A, B, Q, R, verbose=verbose)
        self.L = -np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

    @property
    def state_size(self):
        return self.A.shape[0]

    @property
    def input_size(self):
        return self.C.shape[0]

    def __str__(self):
        return "LQG controller with state size " + str(self.state_size) + " and input size " + str(self.input_size)

    def __add__(self, other):
        if self.state_size == 0:
            return other
        elif other.state_size == 0:
            return self
        A = linalg.block_diag(self.A, other.A)
        B = np.hstack((self.B, other.B))
        Q = linalg.block_diag(self.Q, other.Q)
        R = self.R # hack, see comment on KFilter.__add__ for why

    def concat(self, other):
        if self.state_size == 0:
            return other
        elif other.state_size == 0:
            return self
        A = linalg.block_diag(self.A, other.A)
        B = linalg.block_diag(self.B, other.B)
        Q = linalg.block_diag(self.Q, other.Q)
        R = linalg.block_diag(self.R, other.R)
        return LQGController(A, B, Q, R)

    def control(self):
        return self.L @ self.x
 