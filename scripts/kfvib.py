"""
Integrator control with a vibration Kalman filter.
More of this is in this script than I'd like, but I'll fix it later, I promise
"""
import sys
sys.path.append("..")

from src import *
from src.controllers import make_kalman_controllers
from src.experiments.schedules import *
from src.experiments.exp_utils import record_experiment, control_schedule

import numpy as np
from matplotlib import pyplot as plt
from functools import partial

ol = np.load(joindata("openloop/ol_tt_stamp_12_08_2021_10_14_46.npy"))
ident = SystemIdentifier()
kf = ident.make_kfilter_from_openloop(ol)
kf.Q *= 1e12
kf.compute_gain()

Q = np.zeros((4,4)) # LQG state penalty matrix 
Q[0,0] = 1e4
Q[2,2] = Q[0,0]
R = 100 * np.identity(2) # LQG input penalty matrix

kalman_integrate, kalman_lqg = make_kalman_controllers(kf)

def record_kf_integ(dist_schedule, t=1, gain=0.1, leak=1.0, **kwargs):
    path = "kfilter/kf"
    for k in kwargs:
        path = path + "_" + k + "_" + str(kwargs.get(k))

    return record_experiment(
        path,
        control_schedule=partial(control_schedule, control=partial(kalman_integrate, gain=gain, leak=leak)),
        dist_schedule=partial(dist_schedule, t, **kwargs),
        t=t
    )

record_kinttrain = partial(record_kf_integ, step_train_schedule)
record_kintnone = partial(record_kf_integ, noise_schedule)
record_kintustep = partial(record_kf_integ, ustep_schedule)
record_kintsin = partial(record_kf_integ, sine_schedule)
record_kintatmvib = partial(record_kf_integ, atmvib_schedule)

def record_lqg(dist_schedule, t=1, **kwargs):
    path = "lqg/lqg"
    for k in kwargs:
        path = path + "_" + k + "_" + str(kwargs.get(k))

    return record_experiment(
        path,
        control_schedule=partial(control_schedule, control=kalman_lqg),
        dist_schedule=partial(dist_schedule, t, **kwargs),
        t=t
    )

record_kinttrain = partial(record_kf_integ, step_train_schedule)
record_kintnone = partial(record_kf_integ, noise_schedule)
record_kintustep = partial(record_kf_integ, ustep_schedule)
record_kintsin = partial(record_kf_integ, sine_schedule)
record_kintatmvib = partial(record_kf_integ, atmvib_schedule)

simulate = False

if simulate:
    ttvals = ol - kf.run(ol, kf.x) @ kf.C.T

else:
    times, ttvals = record_kintnone(t=10, gain=0.2)

f, p = genpsd(ttvals[:,0])
plt.loglog(f, p)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (DM units^2/Hz)")
plt.title("Closed loop results from Kalman integrator control")
if not simulate:
    plt.savefig("/home/lab/asengupta/plots/cl_kf_int.pdf")
plt.show()