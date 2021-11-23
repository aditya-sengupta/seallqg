# could refactor to not be a class, but it's fine for now

import numpy as np
from scipy import optimize, signal, stats
from copy import copy

from .lqg import LQG
from .utils import log_likelihood, combine_matrices_for_klqg
from ..utils import genpsd, rms
from ..utils import fs

class SystemIdentifier:
    """
    Driver class to build LQG objects from open-loop data.
    Largely built off the Meimon 2010 system identification, and simulation/experimentation on top of that.
    """
    def __init__(
        self,
        ol,
        fs=fs, 
        f1=None, f2=None, fw=None, 
        N_vib_max=4, energy_cutoff=1e-6, 
        max_ar_coef=2, 
    ):

        self.ol = ol
        self.fs = fs # Hz
        if f1 is None:
            self.f1 = fs / 120 # lowest possible frequency of a vibration mode
        else:
            self.f1 = f1

        if f2 is None:
            self.f2 = fs / 3 # highest possible frequency of a vibration mode
        else:
            self.f2 = f2

        if fw is None:
            self.fw = fs / 3 # frequency above which measurement noise dominates
        else:
            self.fw = fw

        self.max_ar_coef = max_ar_coef
        self.N_vib_max = N_vib_max # number of vibration modes to be detected
        self.energy_cutoff = energy_cutoff # proportion of total energy after which PSD curve fit ends
        self.freqs, p1 = genpsd(self.ol[:,0], dt=1/self.fs)
        _, p2 = genpsd(self.ol[:,1], dt=1/self.fs)
        self.psds = [p1, p2]

    @property
    def times(self):
        return np.arange(0, len(self.ol) / self.fs, 1 / self.fs)

    def damped_harmonic(self, pars_model):
        A, f, k, p = pars_model
        return A * np.exp(-k * 2 * np.pi * f * self.times) * np.cos(2 * np.pi * f * np.sqrt(1 - k**2) * self.times - p)

    def make_psd(self, pars_model):
        s = self.damped_harmonic(pars_model)
        return genpsd(s, dt = 1 / self.fs)

    def psd_f(self, f):
        def get_psd_f(pars):
            k = pars[0]
            return self.make_psd([1, f, k, np.pi])[1]

        return get_psd_f

    def est_measurenoise(self, mode=0):
        return np.sqrt(np.mean(self.fs * self.psds[mode][self.freqs > self.fw]))

    def interpolate_psd(self, freqs, psd):
        mask = np.logical_and(self.f1 < freqs, freqs < self.f2)
        return stats.linregress(np.log(freqs[mask]), np.log(psd[mask]))

    def vibe_fit_freq(self, mode=0):
        # returns a 4xN np array with fit parameters, and a 1xN np array with variances.
        par0 = [1e-4, 1]
        PARAMS_SIZE = 2
        df = self.freqs[1] - self.freqs[0]
        width = max(1, int(np.ceil(1/df)))

        peaks = []
        psd = copy(self.psds[mode])
        # indices where the PSD peaks
        unsorted_peaks = signal.find_peaks(psd, height=self.energy_cutoff)[0]
        # indices in range ordered by energy
        freqs_energy = np.flip(np.argsort(psd))
        # we also want them to be in the range f1-f2
        freqs_energy = freqs_energy[np.logical_and(
            self.freqs[freqs_energy] >= self.f1, 
            self.freqs[freqs_energy] <= self.f2)
        ]
        # for our peaks we want unsorted_peaks, in the order in freqs_energy
        # or phrased differently, we want the subset of freqs_energy that's in unsorted_peaks
        peaks = freqs_energy[np.in1d(freqs_energy, unsorted_peaks)]

        params = -np.ones((self.N_vib_max, PARAMS_SIZE))
        variances = np.zeros(self.N_vib_max)

        i = 0
        for peak_ind in peaks:
            if i >= self.N_vib_max:
                break
            if not(np.any(np.abs(params[:,0] - self.freqs[peak_ind]) <= width * df)): 
                # this is to prevent peak overlapping/fitting to the same thing twice
                l, r = max(0, peak_ind - width), min(len(psd), peak_ind + width)
                windowed = psd[l:r]
                objective = lambda pars: self.psd_f(self.freqs[peak_ind])(pars)[l:r]
                psd_ll = log_likelihood(objective, windowed)
                res = optimize.minimize(psd_ll, par0, method='Nelder-Mead')
                k, sd = res.x
                params[i] = [(self.freqs[peak_ind] + self.freqs[peak_ind + 1])/2, k]
                # slight hack because i'm noticing the frequencies are getting underestimated
                variances[i] = sd ** 2
                i += 1

        return params[:i], variances[:i], psd

    def make_state_transition_vibe(self, params):
        STATE_SIZE = 2 * params.shape[0]
        A = np.zeros((STATE_SIZE, STATE_SIZE))
        for i in range(STATE_SIZE // 2):
            f, k = params[i]
            w0 = 2 * np.pi * f / np.sqrt(1 - k**2)
            A[2 * i][2 * i] = 2 * np.exp(-k * w0 / self.fs) * np.cos(w0 * np.sqrt(1 - k**2) / self.fs)
            A[2 * i][2 * i + 1] = -np.exp(-2 * k * w0 / self.fs)
            A[2 * i + 1][2 * i] = 1
        return A

    def make_klqg_vibe(self, params, variances, mode=0):
        # Make a Kalman-LQG object with which to control vibrations.
        A = self.make_state_transition_vibe(params)
        STATE_SIZE = 2 * params.shape[0]
        B = np.zeros((STATE_SIZE, 1))
        B[0,0] = -1 / (2 * STATE_SIZE)
        C = np.array([[1, 0] * (STATE_SIZE // 2)])
        W = np.zeros((STATE_SIZE, STATE_SIZE))
        for i in range(variances.size):
            W[2 * i][2 * i] = variances[i]
        V = self.est_measurenoise(mode)**2 * np.identity(1)
        
        Q = 10 * C.T @ C # only penalize the observables
        R = 100 * np.eye(1)
        return (A, B, C, W, V, Q, R)

    def make_2d_klqg_vibe(self):
        matrices = [np.zeros((0,0)) for _ in range(7)]
        for mode in range(2):
            params, variances, _ = self.vibe_fit_freq(mode)
            matrices = combine_matrices_for_klqg(matrices, self.make_klqg_vibe(params, variances * 1e12, mode))

        return matrices

    def make_klqg_ar(self, mode=0, ar_len=2):
        n = len(self.ol[:,mode])
        TTs_mat = np.empty((n - ar_len, ar_len))
        for i in range(ar_len):
            TTs_mat[:, i] = self.ol[ar_len - i : n - i, mode] 

        ar_coef, _, _, _ = np.linalg.lstsq(TTs_mat, self.ol[ar_len:, mode], rcond=None)
        ar_residual = self.ol[ar_len:, mode] - (TTs_mat @ ar_coef)
        
        A = np.zeros((ar_len, ar_len))
        A[0,:] = ar_coef
        for i in range(1, ar_len):
            A[i,i-1] += 1.0

        B = np.zeros((ar_len, 1))
        B[0,0] = -1
        C = np.zeros((1, ar_len))
        C[0,0] = 1

        W = np.zeros((ar_len, ar_len))
        W[0,0] = np.mean(ar_residual ** 2) + rms(self.ol)

        V = np.array([[self.est_measurenoise(mode) ** 2]])

        Q = C.T @ C
        R = 100 * np.eye(1)
        return (A, B, C, W, V, Q, R)

    def make_2d_klqg_ar(self, ar_len=0):
        if ar_len == 0:
            ar_len = self.max_ar_coef
        matrices = [np.zeros((0,0)) for _ in range(7)]
        for mode in range(2):
            matrices = combine_matrices_for_klqg(matrices, self.make_klqg_ar(mode, ar_len))

        return matrices

    def make_klqg_from_openloop(self, model_atm=True, model_vib=True):
        """
        Designs a LQG object based on open-loop data. 
        (Essentially stitches together a bunch of LQG objects.)

        Arguments
        ---------
        model_atm : bool
        Makes a filter that models atmospheric aberrations.

        model_vib : bool
        Makes a filter that models mechanical vibrations.

        Returns
        -------
        klqg : LQG
        A Kalman-LQG object that controls either tip or tilt.
        """
        A = np.zeros((0,0))
        B = np.zeros((0,2))
        C = np.zeros((2,0))
        W = np.zeros((0,0))
        V = np.zeros((0,0))
        Q = np.zeros((0,0))
        R = np.zeros((0,0))

        matrices = [A, B, C, W, V, Q, R]
        
        if model_vib:
            vib_matrices = self.make_2d_klqg_vibe()
            matrices = combine_matrices_for_klqg(matrices, vib_matrices, measure_once=True)

        if model_atm:
            atm_matrices = self.make_2d_klqg_ar()
            matrices = combine_matrices_for_klqg(matrices, atm_matrices, measure_once=True)

        matrices[1] /= (np.abs(np.sum(matrices[1])))
        # steering + delay model here
        
        """matrices = combine_matrices_for_klqg(matrices, [
            np.zeros((2,2)),
            -np.eye(2),
            np.eye(2),
            1e-6 * np.eye(2),
            1e-6 * np.eye(2),
            1e-6 * np.eye(2),
            np.eye(2)
            ],
        measure_once=True)"""

        return LQG(*matrices)
