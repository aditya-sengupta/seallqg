import numpy as np
import scipy.linalg as la

from src.utils import rms
from src.controllers.dare import solve_dare, check_dare, solve_dare_iter
from src.controllers.kalmanlqg import KalmanLQG

#np.random.seed(5)

s, m, p = 5, 1, 1

M = np.random.randn(s,s)
eigvals = np.random.uniform(-1, 1, s)
eigvals = eigvals / (1.01 * np.max(np.abs(eigvals)))
A = np.linalg.inv(M) @ np.diag(eigvals) @ M
B = np.random.randn(s, p)
C = np.random.randn(m, s)
Wr, Vr = np.random.randn(s, s), np.random.randn(m, m)
W, V = Wr @ Wr.T, Vr @ Vr.T
Q = 1 * np.eye(s)
R = 1e-6 * np.eye(p)

klqg = KalmanLQG(A, B, C, W, V, Q, R)
# Ppr = solve_dare(A.T, C.T * 0, W, V)
Pcon_true, iter = solve_dare_iter(A + B @ klqg.L, B, Q, R)

print(f"Improvement: {klqg.improvement()}")
print(f"Controlled process covariance: {C @ Pcon_true @ C.T}")
print(f"Controlled process actual cov: {np.cov(klqg.sim_control(nsteps=100)[iter:].T)}")