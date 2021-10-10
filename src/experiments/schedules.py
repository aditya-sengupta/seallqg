import numpy as np
import time
import warnings

from ..constants import dt
from ..utils import joindata
from ..optics import applytip, applytilt, applytiptilt

# disturbance schedules go here

def noise_schedule(t, **kwargs):
    """
    Do nothing (profile bench noise.)
    """
    time.sleep(t)

def ustep_schedule(t, tip_amp=0.1, tilt_amp=0.0, **kwargs):
    """
    Put on a unit step.
    """
    time.sleep(t/2)
    applytip(tip_amp)
    applytilt(tilt_amp)

def step_train_schedule(t, n=5, tip_amp=0.1, tilt_amp=0.0, **kwargs):
    """
    Put on a train of unit steps.
    """
    print(f"Putting on {n} impulses over {t} seconds")
    for _ in range(n):
        time.sleep(t/(n+1))
        applytip(tip_amp)
        applytilt(tilt_amp)

def sine_schedule(t, amp=0.1, ang=np.pi, f=0.2, **kwargs):
    """
    Put on a sine wave.
    """
    times = np.arange(0.0, t, dt)
    sinusoid = np.diff(amp * np.sin(2 * np.pi * f * times))
    cosang, sinang = np.cos(ang), np.sin(ang)
    for s in sinusoid:
        t2 = time.time()
        applytip(cosang * s)
        applytilt(sinang * s)
        time.sleep(max(0, dt - (time.time() - t2)))

def atmvib_schedule(t, atm=0, vib=2, scaledown=10, f=100, **kwargs):
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
