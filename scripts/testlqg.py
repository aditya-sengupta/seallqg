import numpy as np
import scipy.linalg as la

from src.controllers.dare import solve_dare

np.random.seed(5)

s, m = 4, 2

M = np.random.randn(s,s)
eigvals = np.random.uniform(-1, 1, s)
eigvals = eigvals / ((1 + 1e-6) * np.max(np.abs(eigvals)))
A = np.linalg.inv(M) @ np.diag(eigvals) @ M
B = np.random.randn(s, m)
Q = np.eye(s)
R = np.eye(m)

P = la.solve_discrete_are(A, B, Q, R)
assert np.allclose(P, solve_dare(A, B, Q, R))