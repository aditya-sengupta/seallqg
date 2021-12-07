from functools import partial
from math import ceil

import numpy as np
from scipy.stats import multivariate_normal as mvn

from ..utils import dt
from ..utils import joindata

def make_air(dur, **kwargs):
    return np.zeros((ceil(dur / dt), 2))

def make_noise(dur, sd, **kwargs):
    cov = sd ** 2 * np.eye(2)
    dist = mvn(cov=cov)
    return np.array([dist.rvs() for _ in range(ceil(dur / dt))])

def make_ustep(dur, tilt_amp, tip_amp, **kwargs):
    nsteps = ceil(dur / dt)
    dist = np.zeros((nsteps, 2))
    dist[nsteps // 2, :] = [tilt_amp, tip_amp]
    return dist

def make_train(dur, n, tilt_amp, tip_amp, **kwargs):
    nsteps = int(np.ceil(dur / dt))
    dist = np.zeros((nsteps, 2))
    for i in range(n):
        dist[nsteps * (i + 1) // (n + 1), :] = [tilt_amp, tip_amp]

def make_sine(dur, amp, ang, f, **kwargs):
    times = np.arange(0, dur + dt, dt)
    sinusoid = np.diff(amp * np.sin(2 * np.pi * f * times))
    cosang, sinang = np.cos(ang), np.sin(ang)
    return np.array([[cosang * s, sinang * s] for s in sinusoid])

def make_atmvib(dur, atm, vib, scaledown, **kwargs):
    fname = joindata("sims", f"ol_atm_{atm}_vib_{vib}.npy")
    control_commands = np.diff(np.load(fname), axis=0) / scaledown
    control_commands = control_commands[:int(dt * dur), :]
    return control_commands

def make_sum(dur, *schedule_makers, **kwargs):
    return sum(schedule_makers(dur, **kwargs))