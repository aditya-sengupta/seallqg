# my tip-tilt experiments
import numpy as np
import tqdm
import time

from tt import *
from compute_cmd_int import measure_tt
#from image import *
from compute_cmd_int import imflat
from exp_utils import record_experiment, tt_to_dmc

def record_usteps(tip_amp=0.1, tilt_amp=0.0):
    path = "../data/usteps/ustep_amps_{0}_{1}".format(tip_amp, tilt_amp)
    def command_schedule(tip_amp, tilt_amp):
        time.sleep(0.5)
        applytip(tip_amp)
        applytilt(tilt_amp)

    return record_experiment(lambda: command_schedule(tip_amp, tilt_amp), path)

def record_usteps_in_circle(niters=10, amp=0.1, nangles=12):
    for _ in tqdm.trange(niters):
        for ang in np.arange(0, 2 * np.pi, np.pi / nangles):
            record_usteps(amp * np.cos(ang), amp * np.sin(ang), verbose=False)

def record_sinusoids(delay=1e-2):
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
            
        record_experiment(command_schedule, path)
    
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

    record_experiment(command_schedule, path)

def record_integrator(t=1, delay=0.01, gain=0.1, leak=1.0,):
    path = "../data/closedloop/cl_gain_{0}_leak_{1}".format(gain, leak)

    def command_schedule():
        t1 = time.time()
        while time.time() < t1 + t:
            ti = time.time()
            frame = getim()
            tt = measure_tt(frame - imflat)
            dmcn = tt_to_dmc(tt)
            applydmc(leak * getdmc() + gain * dmcn) 
            time.sleep(max(0, delay - (time.time() - ti)))

    record_experiment(command_schedule, path)
    
