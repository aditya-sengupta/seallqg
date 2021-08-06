# my tip-tilt experiments
import numpy as np
import tqdm
import time

from tt import *
from compute_cmd_int import measure_tt
from exp_utils import record_experiment, tt_to_dmc

bestflat = np.load("../data/bestflats/bestflat.npy")
applydmc(bestflat)
imflat = stack(100)

def record_openloop(t=10):
    path = "../data/openloop/ol"
    command_schedule = lambda: None
    return record_experiment(command_schedule, path, t=t)

def record_usteps(tip_amp=0.1, tilt_amp=0.0):
    path = "../data/usteps/ustep_amps_{0}_{1}".format(tip_amp, tilt_amp)
    def command_schedule(tip_amp, tilt_amp):
        time.sleep(0.5)
        applytip(tip_amp)
        applytilt(tilt_amp)

    return record_experiment(lambda: command_schedule(tip_amp, tilt_amp), path)

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
        path = "../data/sinusoid/sinusoid_amp_{0}_nsteps_{1}_nosc_{2}_f_{3}_delay_{4}_mode_{5}".format(round(amplitude, 3), nsteps_per_osc, nosc, f, delay, mode)

        def command_schedule():
            control_commands = amplitude * np.diff(np.sin(2 * np.pi * times * f))
            for cmd in control_commands:
                dmfn(cmd)
                time.sleep(delay)
            
        times, ttvals = record_experiment(command_schedule, path)
        timearrs.append(times)
        ttvalarrs.append(ttvals)
    
def record_atm_vib(atm=0, vib=2, delay=1e-2, scaledown=10):
    """
    Plays vibrations/turbulence on the DM - scaled down in amplitude and in time by a factor of 10.
    """
    fname = "../data/sims/ol_atm_{0}_vib_{1}.npy".format(atm, vib)
    path = "../data/atmvib/atm_{0}_vib_{1}.npy".format(atm, vib)
    
    def command_schedule():
        control_commands = np.diff(np.load(fname), axis=0) / scaledown
        for cmd in control_commands:
            applytiptilt(cmd[0], cmd[1], verbose=False)
            time.sleep(delay)

    return record_experiment(command_schedule, path)

def record_integrator(path, t=1, delay=0.01, gain=0.1, leak=1.0, disturbance_schedule=lambda: None):
    """
    General engine for integrator control (probably to be generalized to any kind of control soon).
    Should never be directly called as the function signature is deeply weird.
    """
    from compute_cmd_int import make_im_cm
    _, cmd_mtx = make_im_cm()

    def command_schedule():
        imflat = np.load("../data/bestflats/imflat.npy")
        t1 = time.time()
        while time.time() < t1 + t:
            ti = time.time()
            frame = getim()
            tt = measure_tt(frame - imflat, cmd_mtx)
            dmcn = tt_to_dmc(tt)
            applydmc(leak * getdmc() + gain * dmcn) 
            time.sleep(max(0, delay - (time.time() - ti)))

    return record_experiment([command_schedule, disturbance_schedule], path, t)

def record_integrator_nodisturbance(t=1, delay=0.01, gain=0.1, leak=1.0):
    path = "../data/closedloop/cl_gain_{0}_leak_{1}".format(gain, leak)
    return record_integrator(path, t=t, delay=delay, gain=gain, leak=leak)
    
def record_integrator_with_ustep(t=1, delay=0.01, gain=0.1, leak=1.0, tip_amp=0.0, tilt_amp=0.1):
    path = "../data/closedloop/cl_gain_{0}_leak_{1}_disturb_tip_{2}_tilt_{3}".format(gain, leak, tip_amp, tilt_amp)
    def disturbance_schedule():
        time.sleep(t / 2)
        applytip(tip_amp)
        applytilt(tilt_amp)

    return record_integrator(path, t=t, delay=delay, gain=gain, leak=leak, disturbance_schedule=disturbance_schedule)

def record_integrator_with_sinusoid(t=1, delay=0.01, gain=0.1, leak=1.0, amp=0.05, ang=0, f=0.2):
    path = "../data/closedloop/cl_gain_{0}_leak_{1}_disturb_sin_amp_{2}_ang_{3}_f_{4}".format(gain, leak, amp, ang, f)
    times = np.arange(0.0, t, delay)
    sinusoid = np.diff(amp * np.sin(2 * np.pi * f * times))

    def disturbance_schedule():
        cosang, sinang = np.cos(ang), np.sin(ang)
        for s in sinusoid:
            t2 = time.time()
            applytip(cosang * s)
            applytilt(sinang * s)
            time.sleep(max(0, delay - (time.time() - t2)))

    return record_integrator(path, t=t, delay=delay, gain=gain, leak=leak, disturbance_schedule=disturbance_schedule)

record_intsin = record_integrator_with_sinusoid