import numpy as np
import scipy.linalg as la
from scipy.stats import multivariate_normal as mvn
from tqdm import trange

from src.utils import rms
from src.controllers.dare import solve_dare, check_dare, solve_dare_iter
from src.controllers.kalmanlqg import KalmanLQG

s, m, p = 1, 1, 1

M = np.random.randn(s,s)
eigvals = np.random.uniform(-1, 1, s)
eigvals = eigvals / (1.01 * np.max(np.abs(eigvals)))
A = np.linalg.inv(M) @ np.diag(eigvals) @ M
Wr = np.random.randn(s, s)
W = Wr @ Wr.T
process = lambda: mvn(cov=W, allow_singular=True).rvs()

C = np.random.randn(m, s)
Vr = np.random.randn(m, m)
V = Vr @ Vr.T

Ppr = solve_dare(A.T, C.T * 0, W, V)

nsteps = 1000
states = np.zeros((nsteps, s))
states[0] = process()
for i in trange(1, nsteps):
    states[i] = A @ states[i-1] + process()

print(f"Predicted covariance: {Ppr}")
print(f"Naively-calculated covariance: {W / (1 - A ** 2)}")
print(f"Actual covariance: {np.cov(states[500:].T)}")
