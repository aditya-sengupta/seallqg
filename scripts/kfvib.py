"""
Control of vibrational modes with LQG.
"""
from datetime import datetime
from os import path

import numpy as np
from matplotlib import pyplot as plt

from sealrtc.launch import optics, sine_one, sine_five
from sealrtc.utils import fs, dt, joindata, genpsd, rms
from sealrtc.experiments import Experiment, make_sine, loadres
from sealrtc.controllers import make_lqg_from_ol

np.random.seed(5)

dmc2wf = np.load(joindata("bestflats", "lodmc2wfe.npy"))
f = 5
if f == 1:
    olres = loadres(path.join("openloop", "ol_amp_0.002_ang_0.7854_f_1_tstamp_2021_11_30_06_50_11.csv"))
elif f == 5:
    olres = loadres(path.join("openloop", "ol_amp_0.002_ang_0.7854_f_5_tstamp_2021_11_30_06_58_56.csv"))
experiment = Experiment(make_sine, 100, optics, amp=0.002, ang=np.pi/4, f=f)

ol = olres.measurements

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

def plot_timefreq(res):
    nsteps = len(res.measurements)
    nsteps_plot = min(1000, nsteps)
    _, axs = plt.subplots(1, 2, figsize=(10,6))
    plt.suptitle("Bench LQG control results")
    meastoplot = lambda meas: np.convolve(np.linalg.norm(meas, axis=1)[:nsteps_plot], np.ones(10) / 10, 'same')
    for (name, t, meas) in zip(["OL", "LQG"], [olres.texp, res.texp], [ol, res.measurements[:,:2]]):
        rmsval = rms(meas)
        axs[0].plot(t[:nsteps_plot], meastoplot(meas), label=f"{name}, rms = {round(rmsval, 3)}")
        freqs, psd = genpsd(meas[:,0], dt=dt)
        # adding in quadrature 
        axs[1].loglog(freqs, psd, label=f"{name} PSD")

    axs[0].set_title("Control residuals")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Time-averaged RMS error")
    axs[0].legend()
    axs[1].set_title("Residual PSD (component 0)")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Simulated power")
    axs[1].legend()
    plt.show()
   
if __name__ == "__main__":
    if optics.name == "Sim":
        res = lqg.simulate(nsteps=10000)
        plt.show()
    else:
        res = experiment.run(lqg)
        data = get_ol_cl_rms(res.measurements)
        print(f"RMS ratios: {[float(x[2]) for x in data]}")
        if input("Plot? (y/n) ") == 'y':
            plot_timefreq(res)
