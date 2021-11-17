# authored by Aditya Sengupta
# my tip-tilt experiments

import numpy as np
from os import path
from functools import partial

from .schedules import make_noise, make_ustep, make_train, make_sine, make_atmvib
from .exp_runner import run_experiment, control_schedule_from_law
from ..optics import optics
from ..optics import applytip, applytilt, aperture
from ..controllers import openloop, integrate

# uconvert_ratio removed 2021-11-17 at commit just after 6a207e3

# "record" functions: a bunch of combinations of a control_schedule and dist_schedule
# with specific handling for parameters like amplitudes

# TODO make sure that the kwargs are getting passed into the controller!
record_openloop = partial(run_experiment, path.join("openloop", "ol"), openloop)

record_oltrain = partial(record_openloop, make_train)
record_olnone = partial(record_openloop, make_noise)
record_olustep = partial(record_openloop, make_ustep)
record_olsin = partial(record_openloop, make_sine)
record_olatmvib = partial(record_openloop, make_atmvib)
# deleted record OL usteps in circle as it didn't seem too useful
# can add it back in to schedules.py later if desired

record_integrator = partial(run_experiment, path.join("integrator", "int"), integrate)

record_inttrain = partial(record_integrator, make_train)
record_intnone = partial(record_integrator, make_noise)
record_intustep = partial(record_integrator, make_ustep)
record_intsin = partial(record_integrator, make_sine)
record_intatmvib = partial(record_integrator, make_atmvib)

make_record_lqg = lambda lqg: partial(run_experiment, path.join("lqg", "lqg"), lqg)
