import numpy as np
from copy import deepcopy
from scipy import linalg

class KFilter:
    def __init__(self, A, B, C, Q, R):
        iters = 0
        P = deepcopy(Q)
        lastP = np.zeros_like(A)
        K = zeros(A.shape[0], C.shape[0])
        while not np.allclose(lastP == P):
            lastP = deepcopy(P)
            P = A @ P @ A.T + Q
            K = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
            P = P - K @ C @ P
            iters += 1
        print("Took %d iterations to reach steady-state covariance.")
        self.A, self.B, self.C, self.Q, self.R, self.K = A, B, C, Q, R, K

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
        B = linalg.block_diag(self.B, other.B).T
        A = linalg.block_diag(self.A, other.A)
        C = linalg.block_diag(self.C, other.C)
        Q = linalg.block_diag(self.Q, other.Q)
        R = linalg.block_diag(self.R, other.R)
        return KFilter(A, B, C, Q, R)

    def predict(self, x, u):
        return self.A @ x + self.B @ u

    def update(self, x, y):
        return x + self.K @ (y - self.C @ x)

    def dare(self):
        return linalg.solve_discrete_are(self.A, self.C, self.Q, self.R)

    def run(self, measurements, inputs, x0):
        steps = len(measurements)
        states = np.empty((steps, self.s))
        x = deepcopy(x0)
        
        for (i, (u, m)) in enumerate(zip(inputs, measurements)):
            x = self.update(self.predict(x, u), m)
            states[i] = x
        
        return states
