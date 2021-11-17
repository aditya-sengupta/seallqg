"""
Integrator control with a vibration Kalman filter.
More of this is in this script than I'd like, but I'll fix it later, I promise
"""
from os import path

from sealrtc import *
from sealrtc.utils import joindata
from sealrtc.controllers import kalman_lqg
from sealrtc.experiments.schedules import *
from sealrtc.experiments.exp_runner import run_experiment, control_schedule_from_law
from sealrtc.constants import fs, dt

import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from datetime import datetime

np.random.seed(5)

if optics.name == "Sim":
    optics.set_wait()

dmc2wf = np.load(joindata("bestflats", "lodmc2wfe.npy"))
amp, ang = 0.005, np.pi / 4
f = 1
if f == 5:
    ol = np.load(joindata("openloop", "ol_f_5_z_stamp_03_11_2021_14_02_00.npy")) * dmc2wf
elif f == 1:
    ol = np.load(joindata("openloop", "ol_f_1_z_stamp_03_11_2021_13_58_53.npy")) * dmc2wf

ol_spectra = [genpsd(ol[:,i], dt=dt) for i in range(2)]

ident = SystemIdentifier(ol, fs=fs)
klqg = ident.make_klqg_from_openloop()
def make_experiment(klqg):
    def record_lqg(schedule_maker, t=1, **kwargs):
        record_path = path.join("lqg", "lqg")
        for k in kwargs:
            record_path += f"_{k}_{kwargs.get(k)}"

        control_schedule = partial(control_schedule_from_law, control=kalman_lqg)
        dist_schedule = schedule_maker(t, **kwargs)
        return run_experiment(
            record_path,
            control_schedule,
            dist_schedule,
            t
        )

    record_lqgtrain = partial(record_lqg, make_train)
    record_lqgnone = partial(record_lqg, make_noise)
    record_lqgustep = partial(record_lqg, make_ustep)
    record_lqgsin = partial(record_lqg, make_sine)
    record_lqgatmvib = partial(record_lqg, make_atmvib)

    return record_lqgsin

def lqg(klqg, t=10):
    klqg.recompute()
    improvement = klqg.improvement()
    print(f"{improvement = }")
    assert improvement >= 1, f"Kalman-LQG setup does not improve in simulation"
    exp = make_experiment(klqg)
    klqg.x = np.zeros(klqg.state_size,)
    return exp(t=t, amp=amp, ang=ang, f=f)

def get_ol_cl_rms(zvals):
    data = []
    for mode in range(2):
        cl = zvals[:,mode]
        olc = ol[:len(cl),mode]
        rms_ratio = rms(cl) / rms(olc)
        rms_ratio = str(np.round(rms_ratio, 4))[:7]
        data.append([olc, cl, rms_ratio])
    return data

def plot_cl_rtf(data, timestamp=datetime.now().strftime("%d_%m_%Y_%H_%M_%S"), save=True):
    fig, axs = plt.subplots(2, figsize=(9,9))
    fig.tight_layout(pad=4.0)
    plt.suptitle("LQG rejection")
    for mode in range(2):
        olc, cl, rms_ratio = data[mode]
        f_ol, p_ol = genpsd(olc, dt=dt)
        f_cl, p_cl = genpsd(cl, dt=dt)
        axs[mode].loglog(f_ol, p_ol, label="Open-loop")
        axs[mode].loglog(f_cl, p_cl, label="Closed-loop")
        axs[mode].loglog(f_cl, p_cl / p_ol, label="Rejection TF")
        axs[mode].legend()
        axs[mode].set_xlabel("Frequency (Hz)")
        axs[mode].set_ylabel(r"Power (DM $units^2/Hz$)")
        axs[mode].set_title(f"Mode {mode}, CL/OL RMS {rms_ratio}")
        fname = f"../plots/cl_lqg_{timestamp}.pdf"
        if save:
            plt.savefig(joindata(fname))
    plt.show()

# start ad hoc modifications to the observe/control matrices
klqg.R *= 1000
# end modifications

if __name__ == "__main__":
    #times, zvals = record_olnone(t=10)
    times, zvals, timestamp = lqg(klqg, t=10)
    data = get_ol_cl_rms(zvals * dmc2wf)
    print(f"RMS ratios: {[float(x[2]) for x in data]}")
    if False and input("Plot? (y/n) ") == 'y':
        plot_cl_rtf(data, timestamp)
