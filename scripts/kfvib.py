"""
Integrator control with a vibration Kalman filter.
More of this is in this script than I'd like, but I'll fix it later, I promise
"""
import sys
sys.path.append("..")
from os import path

from src import *
from src.utils import joindata
from src.controllers import make_kalman_controllers
from src.experiments.schedules import *
from src.experiments.exp_utils import record_experiment, control_schedule_from_law

import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from datetime import datetime

ol = np.load(joindata("openloop", "ol_z_stamp_18_10_2021_08_40_07.npy"))
# update this with the latest 100-second openloop

ol_spectra = [genpsd(ol[:,i], dt=0.01) for i in range(2)]

ident = SystemIdentifier(ol, fs=100)
klqg = ident.make_klqg_from_openloop()
def recompute_schedules(klqg):
    kalman_integrate, kalman_lqg = make_kalman_controllers(klqg)

    def record_kf_integ(dist_schedule, t=1, gain=0.1, leak=1.0, **kwargs):
        record_path = path.join("kfilter", "kf")
        for k in kwargs:
            path = path + f"_{k}_{kwargs.get(k)}"

        return record_experiment(
            record_path,
            control_schedule=partial(control_schedule_from_law, control=partial(kalman_integrate, gain=gain, leak=leak)),
            dist_schedule=partial(dist_schedule, t, **kwargs),
            t=t
        )

    record_kinttrain = partial(record_kf_integ, step_train_schedule)
    record_kintnone = partial(record_kf_integ, noise_schedule)
    record_kintustep = partial(record_kf_integ, ustep_schedule)
    record_kintsin = partial(record_kf_integ, sine_schedule)
    record_kintatmvib = partial(record_kf_integ, atmvib_schedule)

    def record_lqg(dist_schedule, t=1, **kwargs):
        record_path = path.join("lqg", "lqg")
        for k in kwargs:
            path = path + "_" + k + "_" + str(kwargs.get(k))

        return record_experiment(
            record_path,
            control_schedule=partial(control_schedule_from_law, control=kalman_lqg),
            dist_schedule=partial(dist_schedule, t, **kwargs),
            t=t
        )

    record_lqgtrain = partial(record_lqg, step_train_schedule)
    record_lqgnone = partial(record_lqg, noise_schedule)
    record_lqgustep = partial(record_lqg, ustep_schedule)
    record_lqgsin = partial(record_lqg, sine_schedule)
    record_lqgatmvib = partial(record_lqg, atmvib_schedule)

    return record_kintnone, record_lqgnone

def run_experiment(klqg, t=10, i=1):
    klqg.recompute()
    assert klqg.improvement() >= 1, "Kalman-LQG setup does not improve in simulation."
    exp = recompute_schedules(klqg)[i]
    klqg.x = np.zeros(klqg.state_size,)
    times, zvals = exp(t=t)
    return times, zvals, datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

kint = partial(run_experiment, i=0)
lqg = partial(run_experiment, i=1)

def plot_cl_rtf(zvals, mode, dt=datetime.now().strftime("%d_%m_%Y_%H_%M_%S"), save=True):
    fig, axs = plt.subplots(2, figsize=(9,9))
    fig.tight_layout(pad=4.0)
    plt.suptitle("LQG rejection")
    for mode in range(2):
        cl = zvals[:,mode]
        olc = ol[:len(cl),mode]
        f_ol, p_ol = genpsd(olc, dt=0.01)
        f_cl, p_cl = genpsd(cl, dt=0.01)
        rms_ratio = rms(cl) / rms(olc)
        rms_ratio = str(np.round(rms_ratio, 4))[:7]
        axs[mode].loglog(f_ol, p_ol, label="Open-loop")
        axs[mode].loglog(f_cl, p_cl, label="Closed-loop")
        axs[mode].loglog(f_cl, p_cl / p_ol, label="Rejection TF")
        axs[mode].legend()
        axs[mode].set_xlabel("Frequency (Hz)")
        axs[mode].set_ylabel(r"Power (DM $units^2/Hz$)")
        axs[mode].set_title(f"Mode {mode}, CL/OL RMS {rms_ratio}")
        fname = "../plots/cl_lqg_" + dt + ".pdf"
        if save:
            plt.savefig(joindata(fname))
    plt.show()

# start ad hoc modifications to the observe/control matrices
klqg.R *= 1e6
# end modifications

if __name__ == "__main__":
    times, zvals, dt = lqg(klqg, t=10)
    plot_cl_rtf(zvals, dt)
