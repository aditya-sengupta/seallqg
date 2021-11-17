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
    times = np.array(times)
    t0 = time.time()
    assert np.all(np.diff(times) >= dt - 1e-8), f"can't send disturbances on timescales shorter than {dt}"
    logger.info("init from command_thread")
    for (t, (z0, z1)) in zip(times, disturbances):
        zeno((t0 + t) - time.time())
        logger.info(f"Disturbance  : {[z0, z1]}")
        applytilt(z0)
        applytip(z1)

def make_noise(duration):
    return partial(schedule, duration, times=[], disturbances=[[]])

def make_ustep(duration, tilt_amp, tip_amp):
    return partial(
        schedule, 
        duration,
        times = [0.5 * duration], 
        disturbances = [[tilt_amp, tip_amp]]
    )

def make_train(duration, n, tilt_amp, tip_amp):
    return partial(
        schedule,
        duration,
        times = np.linspace(0, duration, n + 2)[1:-1],
        disturbances = [[tilt_amp, tip_amp] for _ in range(n)]
    )

def make_sine(duration, amp, ang, f):
    times = np.arange(0, duration, dt)
    sinusoid = np.diff(amp * np.sin(2 * np.pi * f * times))
    cosang, sinang = np.cos(ang), np.sin(ang)
    return partial(
        schedule,
        duration,
        times = times,
        disturbances = [[cosang * s, sinang * s] for s in sinusoid]
    )

def make_atmvib(duration, atm, vib, scaledown, f):
    fname = joindata("sims", f"ol_atm_{atm}_vib_{vib}.npy")
    control_commands = np.diff(np.load(fname), axis=0) / scaledown
    control_commands = control_commands[:int(dt * duration)]
    return partial(
        schedule,
        duration,
        times = np.linspace(0, duration, len(control_commands)),
        disturbances = control_commands
    )
