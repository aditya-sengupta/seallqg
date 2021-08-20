"""
Integrator control with a vibration Kalman filter.
More of this is in this script than I'd like, but I'll fix it later, I promise
"""
import sys
sys.path.append("..")

from src import *
from src.utils import joindata
from src.controllers import make_kalman_controllers
from src.experiments.schedules import *
from src.experiments.exp_utils import record_experiment, control_schedule

import numpy as np
from matplotlib import pyplot as plt
from functools import partial

collect_new_ol = False
if collect_new_ol:
    _, ol = record_olnone(t=100)
else:
    ol = np.load(joindata("openloop/ol_tt_stamp_20_08_2021_09_45_48.npy"))
ident = SystemIdentifier(ol, fs=100)
klqg = ident.make_klqg_from_openloop()
def recompute_schedules(klqg):
    kalman_integrate, kalman_lqg = make_kalman_controllers(klqg)

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

    record_lqgtrain = partial(record_lqg, step_train_schedule)
    record_lqgnone = partial(record_lqg, noise_schedule)
    record_lqgustep = partial(record_lqg, ustep_schedule)
    record_lqgsin = partial(record_lqg, sine_schedule)
    record_lqgatmvib = partial(record_lqg, atmvib_schedule)

    return record_kintnone, record_lqgnone

record_kintnone, record_lqgnone = recompute_schedules(klqg)
times, ttvals = record_kintnone(t=1)

"""f, p = genpsd(ttvals[:,0])
plt.loglog(f, p)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (DM units^2/Hz)")
plt.title("Closed loop results from Kalman-LQG control")
if not simulate:
    plt.savefig(joindata("../plots/cl_kf_lqg.pdf"))
plt.show()
"""