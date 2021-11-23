"""
Implementation of a *generic* Linear-Quadratic-Gaussian observer (Kalman filter) and controller (LQR)
"""

import numpy as np
from copy import copy
from scipy.stats import multivariate_normal as mvn
from tqdm import tqdm, trange

from .controller import Controller
from .dare import solve_dare
from ..utils import rms, joindata

# I don't really need to keep W, V, Q, R as state attributes
# but just want to be aware in case the DARE solution is not nice

class LQG(Controller):
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
        self.recompute()
        self.root_path = joindata("lqg", f"lqg_nstate_{self.state_size}")

    def recompute(self):
        self.x = np.zeros((self.state_size,))
        self.Pobs = solve_dare(self.A.T, self.C.T, self.W, self.V)
        self.Pcon = solve_dare(self.A, self.B, self.Q, self.R)
        self.K = self.Pobs @ self.C.T @ np.linalg.pinv(self.C @ self.Pobs @ self.C.T + self.V)
        self.L = -np.linalg.pinv(self.R + self.B.T @ self.Pcon @ self.B) @ self.B.T @ self.Pcon @ self.A
        self.process_dist = mvn(cov=self.W, allow_singular=True)
        self.measure_dist = mvn(cov=self.V, allow_singular=True)
        self.curr_control = np.zeros((self.input_size,))

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
        return f"LQG observer and controller with state size {self.state_size}, input size {self.input_size} and measurement size {self.measure_size}."

    def predict(self, u):
        self.x = self.A @ self.x + self.B @ u

    def update(self, y):
        self.x = self.x + self.K @ (y - self.C @ self.x)

    def measure(self):
        return self.C @ self.x

    def control_law(self, state):
        self.curr_control = self.L @ state
        return self.curr_control, 1

    def observe_law(self, measurement):
        # check out order of operations here
        self.predict(self.curr_control)
        self.update(measurement[:2]) # TODO generalize
        return self.x

    def filter(self, measurements, x0):
        steps = len(measurements)
        assert len(measurements.shape) == 2 and measurements.shape[1] == self.measure_size, f"incorrect size for measurements in Kalman filter: expected (nsteps, {self.measure_size}) but got {measurements.shape}"
        states = np.empty((steps, self.state_size))
        self.x = x0
        uzero = np.zeros((self.input_size,))

        for (i, m) in enumerate(tqdm(measurements)):
            self.predict(uzero)
            self.update(m)
            states[i] = self.x
        
        return states

    def loop_iter(self, meas):
        if meas is not None:
            self.predict(self.curr_control)
            self.update(meas)
        else:
            self.x = np.zeros((self.state_size,))
        return self.control()

    def simulate(self, *con, nsteps=1000):
        # TODO change this 
        # make sure nothing you pass in as "con" depends on "self" or things will get weird with the internal state
        # if you want to compare, e.g. KF + integrator to LQG, use a copy of this instance
        controllers = [self]
        controllers.extend(con)
        states_one = np.zeros((nsteps, self.state_size))
        states_one[0] = self.process_dist.rvs()
        states = [copy(states_one) for _ in controllers]
        for i in trange(1, nsteps):
            process_noise, measure_noise = self.process_dist.rvs(), self.measure_dist.rvs()
            measurements = [self.C @ state[i-1] + measure_noise for state in states] # the same one for each controller
            uvals = [c(m)[0] for c, m in zip(controllers, measurements)] # I think this is what (1, 1) tensor contraction is
            for (j, u) in enumerate(uvals):
                states[j][i] = self.A @ states[j][i-1] + self.B @ u + process_noise

        [c.reset() for c in controllers]
        return [state @ self.C.T for state in states]

    def improvement(self, *con, nsteps=1000):
        state_arrays = self.simulate(*con, nsteps=nsteps)
        return [rms(s) / rms(state_arrays[0]) for s in state_arrays[1:]]
