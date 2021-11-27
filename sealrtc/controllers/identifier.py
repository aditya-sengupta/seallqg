from copy import copy

import numpy as np

from lmfit import Model, Parameters
from scipy import optimize, signal, stats, integrate, linalg

from .lqg import LQG
from ..utils import genpsd, rms
from ..utils import fs

def combine_matrices_for_lqg(base, addons, measure_once=False):
    matrices = []    
    for (i, (Mb, Ma)) in enumerate(zip(base, addons)):
        if measure_once and i == 1: # B
            matrices.append(np.vstack((Mb, Ma)))
        elif measure_once and i == 2: # C
            matrices.append(np.hstack((Mb, Ma)))
        elif measure_once and i == 4: # V
            matrices.append(Ma)
        elif measure_once and i == 6: # R
            if Mb.shape == Ma.shape:
                matrices.append(Mb + Ma)
            else:
                matrices.append(Ma)
        else:
            matrices.append(linalg.block_diag(Mb, Ma))
    return matrices

def damped_harmonic(t, w, k, A0, v0, sigma, dt=1/fs):
    """
    Generate a damped harmonic up until time t with sinusoidal parameters (w, k) and initial conditions defining (A, p).
    This is the solution to the ODE with stochastic driving noise

    y'' + 2kw y' + w^2 y = N(0, sigma^2)

    with initial conditions

    x(0) = A0
    x'(0) = v0
    """
    times = np.arange(0, t+dt, dt)
    y0 = np.array([A0, v0])
    w2 = w ** 2
    zeta = 2 * k * w

    def deriv(t, y):
        return np.array([y[1], -zeta * y[1] - w2 * y[0] - np.random.normal(0, sigma ** 2)])

    res =  integrate.solve_ivp(deriv, (0, t + dt), y0, t_eval=times)
    return res.t, res.y[0,:]

def multivib(t, ws, ks, sigmas, dt=1/fs):
    """
    Generate a multi-vibration signal with the given w, k, sigma values, and random initial conditions.
    """
    assert len(ws) == len(ks)
    assert len(ws) == len(sigmas)
    N = len(ws)
    A0s = np.random.uniform(-1, 1, size=(N,))
    v0s = np.random.uniform(-1, 1, size=(N,))
    times = np.arange(0, t+dt, dt)
    y = np.zeros_like(times)
    for (w, k, sigma, A0, v0) in zip(ws, ks, sigmas, A0s, v0s):
        y += damped_harmonic(t, w, k, A0, v0, sigma, dt=dt)[1]
        
    return times, y

def find_psd_peaks(freqs, psd, Nvib=3, f1=fs/120, f2=fs/3, energy_cutoff=1e-6):
    """
    scipy.signal.find_peaks, but with some extra conditions
    """
    # indices where the PSD peaks
    unsorted_peaks = signal.find_peaks(psd, height=energy_cutoff)[0]
    # indices in range ordered by energy
    freqs_energy = np.flip(np.argsort(psd))
    # we also want them to be in the range f1-f2
    freqs_energy = freqs_energy[np.logical_and(
        freqs[freqs_energy] >= f1, 
        freqs[freqs_energy] <= f2)
    ]
    peaks = freqs_energy[np.in1d(freqs_energy, unsorted_peaks)]
    return freqs[peaks[:Nvib]]

def vib_coeffs(f, k, fs=fs):
    w = 2 * np.pi * f
    a1 = 2 * np.exp(-k * w / fs) * np.cos(w * np.sqrt(1 - k**2) / fs)
    a2 = -np.exp(-2 * k * w / fs)
    return a1, a2

def powerfit_psd(freqs, psd, f1=fs/120, f2=fs/3, nstd=3):
    """
    Linear regression in log-log space with vibrations removed.
    """
    mask = np.intersect1d(np.where(f1 <= freqs), np.where(freqs <= f2))
    mask = np.intersect1d(np.where(f1 <= freqs), np.where(freqs <= f2))    
    reslin = stats.linregress(np.log(freqs[mask]), np.log(psd[mask]))
    fitpower = np.exp(reslin.slope * np.log(freqs) + reslin.intercept)
    deviation = np.log(psd) - (reslin.slope * np.log(freqs) + reslin.intercept)
    peak_mask = np.where(deviation > nstd * np.std(deviation))
    modified_psd_tilt = copy(psd)
    modified_psd_tilt[peak_mask] = fitpower[peak_mask]
    reslin_mod = stats.linregress(np.log(freqs[mask]), np.log(modified_psd_tilt[mask]))
    # modified_fitpower = np.exp(reslin_mod.slope * np.log(freqs) + reslin_mod.intercept)
    return reslin_mod.slope, reslin_mod.intercept

def model_psd(freqs, f, k, sigma):
    phase = 2 * np.pi * freqs / fs
    a1, a2 = vib_coeffs(f, k)
    denom = np.abs(1 - a1 * np.exp(-1j * phase) - a2 * np.exp(-2j * phase))
    return sigma ** 2 / fs / denom ** 2

def log_model_psd(freqs, f, k, sigma):
    return np.log(model_psd(freqs, f, k, sigma))

def fit_psd(freqs, psd, Nvib=3):
    df = np.max(np.diff(freqs))
    fcens = find_psd_peaks(freqs, psd, Nvib)
    fs, ks, sigmas = [], [], []
    for fcen in fcens:
        fit_params = Parameters()
        fit_params.add('f', value=fcen, min=fcen-df, max=fcen+df)
        fit_params.add('k', value=1e-4, min=1e-10, max=1e-3)
        fit_params.add('sigma', value=1e-1, min=1e-10, max=100)
        model = Model(log_model_psd)
        slope, intercept = powerfit_psd(freqs, psd)
        one_peak_psd = np.exp(slope * np.log(freqs) + intercept)
        one_peak_mask = np.abs(freqs - fcen) < 2 * df
        one_peak_psd[one_peak_mask] = psd[one_peak_mask]
        res = model.fit(np.log(one_peak_psd), fit_params, freqs=freqs).best_values
        fs.append(res['f'])
        ks.append(res['k'])
        sigmas.append(res['sigma'])

    return fs, ks, sigmas


def estimate_v(freqs, psd, fw=fs/3):
    return np.sqrt(np.mean(fs * psd[freqs > fw]))

def make_A_vibe(fs, ks):
    s = 2 * len(fs)
    A = np.zeros((s, s))
    for (i, (f, k)) in enumerate(zip(fs, ks)):
        A[2 * i][2 * i: 2 * i + 2] = vib_coeffs(f, k)
        A[2 * i + 1][2 * i] = 1
    return A

def make_lqg_vibe(fs, ks, sigmas, measurenoise):
    """
    Make an LQG controller for vibrations, in one dimension
    """
    A = make_A_vibe(fs, ks)
    s = 2 * len(fs)
    B = np.zeros((s, 1))
    B[0,0] = -1 / (2 * s)
    C = np.array([[1, 0] * (s // 2)])
    Wdiag = np.zeros(s)
    Wdiag[0::2] = sigmas
    W = np.diag(Wdiag)
    V = np.array([[measurenoise ** 2]])
    Q = 10 * C.T @ C # only penalize the observables
    R = 100 * np.eye(1)
    return (A, B, C, W, V, Q, R)

def make_2d_lqg_vibe(freqs, psds, Nvib=3):
    matrices = [np.zeros((0,0)) for _ in range(7)]
    fs, ks, sigmas = [], [], []
    for psd in psds:
        fcens = find_psd_peaks(freqs, psd, Nvib=Nvib)
        for fcen in fcens:
            f, k, sigma = fit_psd(freqs, psd, fcen)
            fs.append(f)
            ks.append(k)
            sigmas.append(sigma)
        matrices = combine_matrices_for_lqg(
            matrices, 
            make_lqg_vibe(fs, ks, sigmas, estimate_v(freqs, psd))
        )
    return matrices

def make_lqg_ar(ol_comp, freqs, psd, ar_len=2):
    n = len(ol_comp)
    TTs_mat = np.empty((n - ar_len, ar_len))
    for i in range(ar_len):
        TTs_mat[:, i] = ol_comp[ar_len - i : n - i] 

    ar_coef, _, _, _ = np.linalg.lstsq(TTs_mat, ol_comp[ar_len:], rcond=None)
    ar_residual = ol_comp[ar_len:] - (TTs_mat @ ar_coef)
    
    A = np.zeros((ar_len, ar_len))
    A[0,:] = ar_coef
    for i in range(1, ar_len):
        A[i,i-1] += 1.0

    B = np.zeros((ar_len, 1))
    B[0,0] = -1
    C = np.zeros((1, ar_len))
    C[0,0] = 1

    W = np.zeros((ar_len, ar_len))
    W[0,0] = np.mean(ar_residual ** 2) + rms(ol_comp)

    V = np.array([[estimate_v(freqs, psd) ** 2]])

    Q = C.T @ C
    R = 100 * np.eye(1)
    return (A, B, C, W, V, Q, R)

def make_2d_lqg_ar(ol, freqs, psds, ar_len=2):
    matrices = [np.zeros((0,0)) for _ in range(7)]
    for (ol_comp, psd) in zip(ol, psds):
        matrices = combine_matrices_for_lqg(
            matrices, 
            make_lqg_ar(ol_comp, freqs, psd, ar_len)
        )
    return matrices



def make_lqg_from_ol(ol, model_atm=True, model_vib=True):
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
    lqg : LQG
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

    freqs, psd_tilt = genpsd(ol[:,0], dt=1/fs)
    _, psd_tip = genpsd(ol[:,1], dt=1/fs)
    psds = [psd_tilt, psd_tip]

    if model_vib:
        vib_matrices = make_2d_lqg_vibe(freqs, psds)
        matrices = combine_matrices_for_lqg(matrices, vib_matrices, measure_once=True)

    if model_atm:
        atm_matrices = make_2d_lqg_ar(ol, freqs, psds)
        matrices = combine_matrices_for_lqg(matrices, atm_matrices, measure_once=True)

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
