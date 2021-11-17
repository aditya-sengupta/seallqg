import numpy as np
import time
import warnings

from functools import partial

from ..constants import dt
from ..utils import joindata, zeno
from ..optics import optics, applytip, applytilt

def schedule(dur, logger, times, disturbances):
    """
    Base function for disturbance schedules.

    dur : scalar
        The dur in seconds; "times" must all be less than this dur.

    logger : logging.Logger
        The logger, to document when disturbances were applied.

    times : array_like (N,)
        Times at which commands are to be sent, normalized to the interval [0, dur].

    disturbances : array_like (N, 2)
        The (tilt, tip) disturbance to be sent.
    """
    times = np.array(times)
    t0 = time.time()
    assert np.all(np.diff(times) >= dt - 1e-8), f"can't send disturbances on timescales shorter than {dt}"
    logger.info("Disturbance initialized.")
    for (t, (z0, z1)) in zip(times, disturbances):
        zeno((t0 + t) - time.time())
        logger.info(f"Disturbance  : {[z0, z1]}")
        applytilt(optics, z0)
        applytip(optics, z1)

def make_noise(dur):
    return partial(schedule, dur, times=[], disturbances=[[]])

def make_ustep(dur, tilt_amp, tip_amp):
    return partial(
        schedule, 
        dur,
        times = [0.5 * dur], 
        disturbances = [[tilt_amp, tip_amp]]
    )

def make_train(dur, n, tilt_amp, tip_amp):
    return partial(
        schedule,
        dur,
        times = np.linspace(0, dur, n + 2)[1:-1],
        disturbances = [[tilt_amp, tip_amp] for _ in range(n)]
    )

def make_sine(dur, amp, ang, f):
    times = np.arange(0, dur, dt)
    sinusoid = np.diff(amp * np.sin(2 * np.pi * f * times))
    cosang, sinang = np.cos(ang), np.sin(ang)
    return partial(
        schedule,
        dur,
        times = times,
        disturbances = [[cosang * s, sinang * s] for s in sinusoid]
    )

def make_atmvib(dur, atm, vib, scaledown, f):
    fname = joindata("sims", f"ol_atm_{atm}_vib_{vib}.npy")
    control_commands = np.diff(np.load(fname), axis=0) / scaledown
    control_commands = control_commands[:int(dt * dur)]
    return partial(
        schedule,
        dur,
        times = np.linspace(0, dur, len(control_commands)),
        disturbances = control_commands
    )
