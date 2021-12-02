"""
Implementation of a *generic* Linear-Quadratic-Gaussian observer (Kalman filter) and controller (LQR)
"""

import warnings

from copy import copy

import numpy as np

from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn
from tqdm import tqdm, trange

from .controller import Controller
from .dare import solve_dare

from ..utils import rms, joindata, genpsd, fs

# I don't really need to keep W, V as state attributes
# but just want to be aware in case the DARE solution is not nice

class LQG(Controller):
    """
    Kalman filter information

    x - the state
    A - the time-evolution matrix
    B - the input-to-state matrix
    C - the state-to-measurement matrix
    D - the input-to-measurement matrix
    W - the process noise matrix (covariance around Ax)
    V - the measurement noise matrix (covariance around Cx)
    """
    def __init__(self, A, B, C, D, W, V, dt=1/fs):
        self.name = "LQG"
        self.A, self.B, self.C, self.D, self.W, self.V = A, B, C, D, W, V
        self.dt = dt
        self.leak = 1
        obsrank, conrank = self.observability_rank(), self.controllability_rank()
        n = A.shape[0]
        if obsrank < n:
            print(f"WARNING: LQG system is not observable, observability matrix rank is {obsrank} against dimension {n}.")
        if conrank < n:
            print(f"WARNING: LQG system is not controllable, controllability matrix rank is {conrank} against dimension {n}.")
        self.recompute()
        self.root_path = joindata("lqg", f"lqg_nstate_{self.state_size}")

    def recompute(self):
        s = self.A.shape[0]
        assert self.A.shape == (s, s), "A must be a square matrix."
        p = self.B.shape[1]
        assert len(self.B.shape) == 2, "got wrong number of dimensions in B."
        assert self.B.shape == (s, p), f"B must have dimension matching A: got {self.B.shape[0]} whereas {s} was expected in dimension 0, or got wrong number of dimensions."
        m = self.C.shape[0]
        assert len(self.C.shape) == 2, "got wrong number of dimensions in C."
        assert self.C.shape == (m, s), f"C must have dimension matching A: got {self.C.shape[1]} whereas {s} was expected in dimension 1."
        assert len(self.D.shape) == 2, "got wrong number of dimensions in D."
        assert self.D.shape == (m, p), f"D must have dimensions matching B and C: got {self.D.shape} whereas {(m, p)} was expected."
        Q = self.C.T @ self.C
        R = self.D.T @ self.D
        S = self.C.T @ self.D
        self.x = np.zeros((self.state_size,))
        self.u = np.zeros((self.input_size,))
        self.Pobs = solve_dare(self.A.T, self.C.T, self.W, self.V)
        self.Pcon = solve_dare(self.A, self.B, Q, R, S=S)
        self.K = self.Pobs @ self.C.T @ np.linalg.pinv(self.C @ self.Pobs @ self.C.T + self.V)
        self.L = -np.linalg.pinv(R + self.B.T @ self.Pcon @ self.B) @ (S.T + self.B.T @ self.Pcon @ self.A)
        self.process_dist = mvn(cov=self.W, allow_singular=True)
        self.measure_dist = mvn(cov=self.V, allow_singular=True)

    def observability_rank(self):
        obs_matrix = np.vstack([self.C @ np.linalg.matrix_power(self.A, i) for i in range(self.state_size)])
        return np.linalg.matrix_rank(obs_matrix)

    def controllability_rank(self):
        con_matrix = np.hstack([np.linalg.matrix_power(self.A, i) @ self.B for i in range(self.state_size)])
        return np.linalg.matrix_rank(con_matrix)

    @property
    def state_size(self):
        return self.A.shape[0]

    @property
    def measure_size(self):
        return self.C.shape[0]

    @property
    def input_size(self):
        return self.B.shape[1]

    def __repr__(self):
        return f"LQG observer and controller with state size {self.state_size}, input size {self.input_size} and measurement size {self.measure_size}."

    def measure(self):
        return self.C @ self.x + self.D @ self.u

    def predict(self):
        self.x = self.A @ self.x + self.B @ self.u

    def update(self, y):
        self.x = self.x + self.K @ (y - self.measure())

    def control_law(self):
        self.u = self.L @ self.A @ self.x
        return self.u

    def observe_law(self, measurement):
        # check out order of operations here
        self.predict()
        self.update(measurement[:self.measure_size])

    def simulate(self, con, nsteps=1000, plot=False):
        # make sure nothing you pass in as "con" depends on "self" or things will get weird with the internal state
        # if you want to compare, e.g. KF + integrator to LQG, use a copy of this instance
        self.reset()
        controllers = []
        if not hasattr(con, "__iter__"):
            con = [con]
        controllers.extend(con)
        controllers.append(self)
        states_one = np.zeros((nsteps, self.state_size))
        states_one[0] = self.process_dist.rvs()
        sim = [
            [
                copy(states_one),
                np.zeros((nsteps, self.input_size)),
                np.zeros((nsteps, self.measure_size))
            ]
            for _ in controllers
        ] # sim for state-input-measure, I'm so clever 

        for j in trange(1, nsteps):
            process_noise, measure_noise = self.process_dist.rvs(), self.measure_dist.rvs()
            for (c, (s, i, m)) in zip(controllers, sim):
                m[j-1] = self.C @ s[j-1] + self.D @ i[j-1] + measure_noise
                i[j] = c(m[j-1])
                s[j] = self.A @ s[j-1] + self.B @ i[j] + process_noise
            
        for c in controllers:
            c.reset()

        if plot: 
            nsteps_plot = min(1000, nsteps)
            times = np.arange(nsteps_plot) * self.dt
            fig, axs = plt.subplots(1, 2, figsize=(10,6))
            plt.suptitle("Simulated LQG control results")
            meastoplot = lambda meas: np.linalg.norm(meas, axis=1)[:nsteps_plot]
            for (con, simres) in zip(controllers, sim):
                measurements = simres[2]
                rmsval = rms(measurements)
                axs[0].plot(times, meastoplot(measurements), label=f"{con.name}, rms = {round(rmsval, 3)}")
                freqs, psd = genpsd(np.linalg.norm(measurements, axis=1), dt=self.dt)
                axs[1].loglog(freqs, psd, label=f"{con.name} PSD")

            axs[0].set_title("Control residuals")
            axs[0].set_xlabel("Time (s)")
            axs[0].set_ylabel("Simulated RMS error")
            axs[0].legend()
            axs[1].set_title("Transfer functions")
            axs[1].set_xlabel("Frequency (Hz)")
            axs[1].set_ylabel("Simulated power")
            axs[1].legend()

        return sim

    def improvement(self, con, nsteps=1000):
        simres = self.simulate(con, nsteps=nsteps, plot=False)
        rms_lqg = rms(simres[0][2])
        return [rms(s[2]) / rms_lqg for s in simres[1:]]
