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

nruns = 10
means = np.zeros((nruns,))
for j in trange(nruns):
    nsteps = 500
    half_nsteps = nsteps // 2
    states = np.zeros((nsteps, s))
    #variances = np.zeros((nsteps, s, s))
    for i in range(1, nsteps):
        states[i] = A @ states[i-1] + process()
        #variances[i] = A @ variances[i-1] @ A.T + W
    means[j] = np.mean(states ** 2)

print(f"Predicted covariance: {Ppr}")
print(f"Naively-calculated covariance: {W / (1 - A ** 2)}")
print(f"Actual mean-square: {np.mean(means)}")
print(f"Estimator for mean-square: {W / (1 - A ** 2) * (1 - (A ** 2) / (nsteps) * (1 - A ** (2 * nsteps))/(1 - A ** 2))}")
#print(f"Tracked variances: {np.mean(variances[half_nsteps:])} +/- {np.std(variances[half_nsteps:])}")
