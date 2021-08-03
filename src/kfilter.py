import numpy as np
from copy import copy
from scipy import linalg

class KFilter:
    def __init__(self, A, C, Q, R):
        try:
            self.P = linalg.solve_discrete_are(A.T, C.T, Q, R)
            print("Solved discrete ARE.")
        except (ValueError, np.linalg.LinAlgError):
            print("Discrete ARE solve failed, falling back to iterative solution.")
            P = copy(Q)
            lastP = np.zeros_like(A)
            iters = 0
            while not np.allclose(lastP, P):
                P = A @ P @ A.T + Q
                K = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
                P = P - K @ C @ P
                iters += 1
            self.P = P
            print("Solved iteratively in {} iterations".format(iters))
        self.A, self.C, self.Q, self.R = A, C, Q, R
        self.K = self.P @ C.T @ np.linalg.inv(C @ self.P @ C.T + R)

    @property
    def s(self):
        return self.A.shape[0]

    @property
    def m(self):
        return self.C.shape[0]

    def __str__(self):
        return "Kalman filter with state size " + str(self.A.shape[0]) + " and measurement size " + str(self.H.shape[0])

    def __add__(self, other):
        if self.s == 0:
            return other
        elif other.s == 0:
            return self
        A = linalg.block_diag(self.A, other.A)
        C = linalg.block_diag(self.C, other.C)
        Q = linalg.block_diag(self.Q, other.Q)
        R = linalg.block_diag(self.R, other.R)
        return KFilter(A, C, Q, R)

    def predict(self, x):
        return self.A @ x

    def update(self, x, y):
        return x + self.K @ (y - self.C @ x)

    def run(self, measurements, inputs, x0):
        steps = len(measurements)
        states = np.empty((steps, self.s))
        x = copy(x0)
        
        for (i, (u, m)) in enumerate(zip(inputs, measurements)):
            x = self.update(self.predict(x, u), m)
            states[i] = x
        
        return states
