"""
Control of vibrational modes with LQG.
"""
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from sealrtc import optics, sine_one, sine_five
from sealrtc.utils import fs, dt, joindata, genpsd, rms
from sealrtc.experiments import Experiment, make_sine
from sealrtc.controllers import make_lqg_from_ol

np.random.seed(5)

dmc2wf = np.load(joindata("bestflats", "lodmc2wfe.npy"))
amp, ang = 0.005, np.pi / 4
f = 1
if f == 5:
    ol = np.load(joindata("openloop", "ol_f_5_z_stamp_03_11_2021_14_02_00.npy")) * dmc2wf
    experiment = sine_five
elif f == 1:
    ol = np.load(joindata("openloop", "ol_f_1_z_stamp_03_11_2021_13_58_53.npy")) * dmc2wf
    experiment = sine_one

ol_spectra = [genpsd(ol[:,i], dt=dt) for i in range(2)]

lqg = make_lqg_from_ol(ol)

def get_ol_cl_rms(zvals):
    data = []
    for mode in range(2):
        cl = zvals[:,mode]
        olc = ol[:len(cl),mode]
        rms_ratio = rms(cl) / rms(olc)
        rms_ratio = str(np.round(rms_ratio, 4))[:7]
        data.append([olc, cl, rms_ratio])
    return data

def plot_cl_rtf(data, timestamp, save=False):
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
        fname = f"../plots/lqg_{timestamp}.pdf"
        if save:
            plt.savefig(joindata(fname))
    plt.show()

# start ad hoc modifications to the observe/control matrices
# end modifications
lqg.recompute()
   
if __name__ == "__main__":
    res = experiment.run(lqg)
    data = get_ol_cl_rms(res.measurements * dmc2wf)
    print(f"RMS ratios: {[float(x[2]) for x in data]}")
    if input("Plot? (y/n) ") == 'y':
        plot_cl_rtf(data, res.timestamp)
