# authored by Aditya Sengupta
# my tip-tilt experiments

import numpy as np
from functools import partial

from .schedules import noise_schedule, ustep_schedule, step_train_schedule, sine_schedule, atmvib_schedule
from .exp_utils import record_experiment, control_schedule
from ..optics import get_expt, set_expt, applydmc, getdmc, stack
from ..optics import applytip, applytilt, aperture
from ..optics import refresh
from ..controllers import openloop, integrate

def uconvert_ratio(amp=1.0):
    bestflat, imflat = refresh()
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

# "record" functions: a bunch of combinations of a control_schedule and dist_schedule 
# with specific handling for parameters like amplitudes

def record_openloop(dist_schedule, t=10, **kwargs):
    path = "openloop/ol"
    v = True
    for k in kwargs:
        if k == "verbose":
            v = v and kwargs.get(k)
        else:
            path = path + k + "_" + str(kwargs.get(k))
    
    return record_experiment(
        path, 
        partial(control_schedule, control=openloop), 
        partial(dist_schedule, t, **kwargs),
        t=t,
        verbose=v
    )

record_oltrain = partial(record_openloop, step_train_schedule)
record_olnone = partial(record_openloop, noise_schedule)
record_olustep = partial(record_openloop, ustep_schedule)
record_olsin = partial(record_openloop, sine_schedule)
record_olatmvib = partial(record_openloop, atmvib_schedule)
# deleted record OL usteps in circle as it didn't seem too useful
# can add it back in to schedules.py later if desired

def record_integrator(dist_schedule, t=1, gain=0.1, leak=1.0, **kwargs):
    path = "closedloop/cl_gain_{0}_leak_{1}".format(gain, leak)
    v = True
    for k in kwargs:
        if k == "verbose":
            v = v and kwargs.get(k)
        else:
            path = path + "_" + k + "_" + str(kwargs.get(k))

    return record_experiment(
        path, 
        control_schedule=partial(control_schedule, control=partial(integrate, gain=gain, leak=leak)),
        dist_schedule=partial(dist_schedule, t, **kwargs),
        t=t,
        verbose=v
    )

record_inttrain = partial(record_integrator, step_train_schedule)
record_intnone = partial(record_integrator, noise_schedule)
record_intustep = partial(record_integrator, ustep_schedule)
record_intsin = partial(record_integrator, sine_schedule)
record_intatmvib = partial(record_integrator, atmvib_schedule)
