"""
Integrator control with a vibration Kalman filter.
Starting this off as just the same as notebooks/notch_filter.ipynb.
"""

import sys
sys.path.append("..")

from src import *
from src.constants import dt

import numpy as np
from scipy.signal import windows
from scipy.stats import normaltest, chi2
from matplotlib import pyplot as plt

ol = np.load(joindata("openloop/ol_tt_dt_11_08_2021_10_22_17.npy"))[:,0]
f, p = genpsd(ol, dt=dt)

n_sigfig = 5

def truncate_impulse(impulse, N=3):
    return impulse[:N] / np.sum(impulse[:N])

baseline_err = round(rms(ol[1000:]), n_sigfig)
print("Baseline RMS error: {}".format(baseline_err))
x_start = design_filt(N=32, dt=dt, plot=False)
print("fractal_deriv RMS error with {0} impulse response elements: {1}".format(
    len(x_start), 
    round(rms(filt(x_start, dt=dt, u=ol[1000:], plot=False))), n_sigfig)
)

_, x = design_from_ol(ol[:1000], dt=dt)
x = truncate_impulse(x, N=10)
res = filt(x, dt=dt, u=ol[1000:], plot=False)
print("Designed filter RMS error with {0} impulse response elements: {1}".format(len(x), round(rms(res), n_sigfig)))