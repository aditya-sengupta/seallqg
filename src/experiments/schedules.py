import numpy as np
import time
import warnings

from functools import partial

from ..constants import dt
from ..utils import joindata, zeno
from ..optics import applytip, applytilt

def schedule(duration, logger, times, disturbances):
    """
    Base function for disturbance schedules.

    duration : scalar
        The duration in seconds; "times" must all be less than this duration.

    logger : logging.Logger
        The logger, to document when disturbances were applied.

    times : array_like (N,)
        Times at which commands are to be sent, normalized to the interval [0, duration].

    disturbances : array_like (N, 2)
        The (tilt, tip) disturbance to be sent.
    """
    times = np.array(times) / duration
    t0 = time.time()
    logger.info("init from command_thread")
    for (t, (z0, z1)) in zip(times, disturbances):
        zeno((t0 + t) - time.time())
        logger.info(f"Disturbance  : {[z0, z1]}")
        applytilt(z0)
        applytip(z1)

def make_noise():
    return partial(schedule, times=[], disturbances=[[]])

def make_ustep(tilt_amp, tip_amp):
    return partial(
        schedule, 
        times = [0.5], 
        disturbances = [[tilt_amp, tip_amp]]
    )

def make_train(n, tilt_amp, tip_amp):
    return partial(
        schedule,
        times = np.linspace(0, 1, n + 2)[1:-1],
        disturbances = [[tilt_amp, tip_amp] * n]
    )

def make_sine(amp, ang, f):
    times = np.arange(0, 1, dt)
    sinusoid = np.diff(amp * np.sin(2 * np.pi * f * times))
    cosang, sinang = np.cos(ang), np.sin(ang)
    return partial(
        schedule,
        times,
        disturbances = [[cosang * s, sinang * s] for s in sinusoid]
    )

def make_atmvib(atm, vib, scaledown, f):
    fname = joindata("sims", f"ol_atm_{atm}_vib_{vib}.npy")
    control_commands = np.diff(np.load(fname), axis=0) / scaledown
    return partial(
        schedule,
        times = np.linspace(0, 1, len(control_commands)),
        disturbances = control_commands
    )

def atmvib_schedule(t, atm, vib, scaledown, f, **kwargs):
    """
    Put on a custom disturbance signal with 'atm' HCIPy atmospheric layers and 'vib' vibrational modes.
    (for now precomputed, but it's not hard to extend this just by using src.controllers.make_atm_vib)
    """
    fname = joindata("sims", f"ol_atm_{atm}_vib_{vib}.npy")
    control_commands = np.diff(np.load(fname), axis=0) / scaledown
    nsteps = len(control_commands)
    if t / nsteps < 1/f:
        warnings.warn("atmvib_schedule may be sending DM commands faster than the camera readout, truncating")
    control_commands = control_commands[:int(f*t)]
    for cmd in control_commands:
        time.sleep(dt)
        applytiptilt(cmd[0], cmd[1])
