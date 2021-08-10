import numpy as np
import time
import warnings

from ..utils import joindata
from ..optics import applytip, applytilt, applytiptilt

tsleep = 0.01

# disturbance schedules go here

def noise_schedule(t):
    """
    Do nothing (profile bench noise.)
    """
    time.sleep(t)

def ustep_schedule(t, tip_amp=0.1, tilt_amp=0.0):
    time.sleep(t/2)
    applytip(tip_amp)
    applytilt(tilt_amp)

def step_train_schedule(t, n=5, tip_amp=0.1, tilt_amp=0.0):
    print("Putting on {0} impulses over {1} seconds".format(n, t))
    for _ in range(n):
        time.sleep(t/(n+1))
        applytip(tip_amp)
        applytilt(tilt_amp)

def sine_schedule(t, amp, ang, f):
    times = np.arange(0.0, t, tsleep)
    sinusoid = np.diff(amp * np.sin(2 * np.pi * f * times))
    cosang, sinang = np.cos(ang), np.sin(ang)
    for s in sinusoid:
        t2 = time.time()
        applytip(cosang * s)
        applytilt(sinang * s)
        time.sleep(max(0, tsleep - (time.time() - t2)))

def atmvib_schedule(t, atm=0, vib=2, scaledown=10):
    fname = joindata("sims/ol_atm_{0}_vib_{1}.npy".format(atm, vib))
    control_commands = np.diff(np.load(fname), axis=0) / scaledown
    nsteps = len(control_commands)
    if t / nsteps < 0.01:
        warnings.warn("atmvib_schedule may be sending DM commands faster than the camera readout, truncating")
    control_commands = control_commands[:int(100*t)]
    for cmd in control_commands:
        time.sleep(tsleep)
        applytiptilt(cmd[0], cmd[1], verbose=False)
