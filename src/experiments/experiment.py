# authored by Aditya Sengupta
# my tip-tilt experiments

import numpy as np
import tqdm
import time
from functools import partial

from ..utils import joindata

from .exp_utils import record_experiment, control_schedule
from ..optics import get_expt, set_expt, applydmc, getdmc, stack, dmzero
from ..optics import applytip, applytilt, applytiptilt, funz, aperture
from ..optics import refresh
from ..controllers import OpenLoop, Integrator

bestflat, imflat = refresh()

def uconvert_ratio(amp=1.0):
    expt_init = get_expt()
    set_expt(1e-5)
    uconvert_matrix = np.zeros((2,2))
    for (mode, dmcmd) in enumerate([applytip, applytilt]):
        applydmc(bestflat)
        dmcmd(amp)
        dm2 = getdmc()
        cm2x = []
        while len(cm2x) != 1:
            im2 = stack(100)
            cm2x, cm2y = np.where(im2 == np.max(im2))

        applydmc(bestflat)
        dmcmd(-amp)
        dm1 = getdmc()
        cm1x = []
        while len(cm1x) != 1:
            im1 = stack(100)
            cm1x, cm1y = np.where(im1 == np.max(im1))

        dmdiff = aperture * (dm2 - dm1)
        
        dmdrange = np.max(dmdiff) - np.min(dmdiff)
        uconvert_matrix[mode] = [dmdrange /  (cm2y - cm1y), dmdrange / (cm2x - cm1x)]

    set_expt(expt_init)
    applydmc(bestflat)
    return uconvert_matrix

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
    for _ in range(n):
        time.sleep(t/(n+1))
        applytip(tip_amp)
        applytilt(tilt_amp)

def atmvib_schedule(atm, vib, scaledown, delay):
    fname = joindata("sims/ol_atm_{0}_vib_{1}".format(atm, vib))
    control_commands = np.diff(np.load(fname), axis=0) / scaledown
    for cmd in control_commands:
        applytiptilt(cmd[0], cmd[1], verbose=False)
        time.sleep(delay)

def record_openloop(t=10):
    path = "openloop/ol"
    return record_experiment(path, OpenLoop(), noise_schedule, t=t)

def record_usteps(t, tip_amp=0.1, tilt_amp=0.0):
    path = "usteps/ustep_amps_{0}_{1}".format(tip_amp, tilt_amp)
    return record_experiment(path, dist_schedule=lambda: ustep_schedule(t, tip_amp, tilt_amp))

def record_usteps_in_circle(niters=10, amp=0.1, nangles=12):
    timearrs, ttvalarrs = [], []
    for _ in tqdm.trange(niters):
        for ang in np.arange(0, 2 * np.pi, np.pi / nangles):
            times, ttvals = record_usteps(amp * np.cos(ang), amp * np.sin(ang), verbose=False)
            timearrs.append(times)
            ttvalarrs.append(ttvals)

    return timearrs, ttvalarrs

def record_sinusoids(delay=1e-2):
    timearrs, ttvalarrs = [], []
    for mode in [0, 1]:
        nsteps_per_osc = 50
        nosc = 50
        times = np.arange(0, nsteps_per_osc * nosc * delay, delay)
        f = 1
        amplitude = 1.0
        dmfn = lambda cmd: funz(1, 2*mode-1, cmd, bestflat=bestflat)
        path = "sinusoid/sinusoid_amp_{0}_nsteps_{1}_nosc_{2}_f_{3}_delay_{4}_mode_{5}".format(round(amplitude, 3), nsteps_per_osc, nosc, f, delay, mode)

        def sine():
            control_commands = amplitude * np.diff(np.sin(2 * np.pi * times * f))
            for cmd in control_commands:
                dmfn(cmd)
                time.sleep(delay)
            
        times, ttvals = record_experiment(path, dist_schedule=sine)
        timearrs.append(times)
        ttvalarrs.append(ttvals)

    return timearrs, ttvalarrs
    
def record_atm_vib(atm=0, vib=2, delay=1e-2, scaledown=10):
    """
    Plays vibrations/turbulence on the DM - scaled down in amplitude and in time by a factor of 10.
    """
    path = "atmvib/atm_{0}_vib_{1}".format(atm, vib)
    return record_experiment(path, dist_schedule=lambda: atmvib_schedule(atm, vib, scaledown, delay))

record_integrator = partial(record_experiment, control_schedule=Integrator().control)

def record_integrator_nodisturbance(t=1, delay=0.01, gain=0.1, leak=1.0):
    path = "closedloop/cl_gain_{0}_leak_{1}".format(gain, leak)
    return record_integrator(path, t=t, delay=delay, gain=gain, leak=leak)
    
def record_integrator_with_ustep(t=1, delay=0.01, gain=0.1, leak=1.0, tip_amp=0.0, tilt_amp=0.1):
    path = "closedloop/cl_gain_{0}_leak_{1}_disturb_tip_{2}_tilt_{3}".format(gain, leak, tip_amp, tilt_amp)
    def disturbance_schedule():
        time.sleep(t / 2)
        applytip(tip_amp)
        applytilt(tilt_amp)

    return record_integrator(path, t=t, delay=delay, gain=gain, leak=leak, dist_schedule=disturbance_schedule)

def record_integrator_with_impulse_train(t=1, delay=0.01, gain=0.1, leak=1.0, n=5, tip_amp=0.0, tilt_amp=0.1):
    path = "closedloop/cl_gain_{0}_leak_{1}_disturb_tip_{2}_tilt_{3}_n_{4}".format(gain, leak, tip_amp, tilt_amp, n)
    return record_integrator(path, t=t, delay=delay, gain=gain, leak=leak, dist_schedule=lambda: step_train_schedule(t, n, tip_amp, tilt_amp))

def record_integrator_with_sinusoid(t=1, delay=0.01, gain=0.1, leak=1.0, amp=0.05, ang=0, f=5):
    path = "closedloop/cl_gain_{0}_leak_{1}_disturb_sin_amp_{2}_ang_{3}_f_{4}".format(gain, leak, amp, ang, f)
    times = np.arange(0.0, t, delay)
    sinusoid = np.diff(amp * np.sin(2 * np.pi * f * times))

    def disturbance_schedule():
        cosang, sinang = np.cos(ang), np.sin(ang)
        for s in sinusoid:
            t2 = time.time()
            applytip(cosang * s)
            applytilt(sinang * s)
            time.sleep(max(0, delay - (time.time() - t2)))

    return record_integrator(path, t=t, delay=delay, gain=gain, leak=leak, dist_schedule=disturbance_schedule)

def record_integrator_with_atm_vib(delay=0.01, gain=0.1, leak=1.0, atm=0, vib=2, scaledown=10):
    path = "closedloop/cl_gain_{0}_leak_{1}_atm_{2}_vib_{3}_scaledown_{4}".format(gain, leak, atm, vib, scaledown)
    return record_integrator(path, t=10000*delay, delay=delay, gain=gain, leak=leak, dist_schedule=lambda: atmvib_schedule(atm, vib, scaledown, delay))

record_inttrain = record_integrator_with_impulse_train
record_intnone = record_integrator_nodisturbance
record_intustep = record_integrator_with_ustep
record_intsin = record_integrator_with_sinusoid
record_intatmvib = record_integrator_with_atm_vib