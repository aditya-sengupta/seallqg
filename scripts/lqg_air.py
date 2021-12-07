"""
Control of Gaussian noise with LQG.
"""
from os import path

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn

from sealrtc.launch import optics
from sealrtc.utils import dt, get_timestamp, joindata, genpsd, rms
from sealrtc.experiments import Experiment, make_noise, loadres
from sealrtc.controllers import Openloop, LQG, add_delay

np.random.seed(5)

dur = 5
sigma = 0.0005
experiment = Experiment(make_noise, dur, optics, sd=sigma)
ol = Openloop()

olpath = path.join("lqg_calibration", "air.csv")

recalibrate = True
if recalibrate:
    print("Recalibration run")
    olres = experiment.run(ol)
    olres.to_csv(olpath)
else:
    olres = loadres(olpath)

olm = olres.measurements
ol_spectra = [genpsd(olm[:,i], dt=dt) for i in range(2)]
lqg = add_delay(LQG(
    np.eye(2),
    np.zeros((2,2)),
    np.eye(2),
    np.eye(2),
    sigma ** 2 * np.eye(2),
    np.diag([9.335e-5, 6.256e-5]) # a recent V
), d=0)

def get_ol_cl_rms(zvals):
    data = []
    for mode in range(2):
        cl = zvals[:,mode]
        olc = olm[:len(cl),mode]
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
    meastoplot = lambda meas: (meas*optics.dmc2wf)[:nsteps_plot,0]
    for (name, t, meas) in zip(["OL", "LQG"], [olres.texp, res.texp], [olm, res.measurements]):
        rmsval = rms(meas[:,:2] * optics.dmc2wf)
        axs[0].plot(t[:nsteps_plot], meastoplot(meas), label=f"{name}, rms = {round(rmsval, 3)}")
        freqs, psd = genpsd(meas[:,0] * optics.dmc2wf, dt=dt)
        axs[1].loglog(freqs, psd, label=f"{name} PSD")

    axs[0].set_title("Control residuals")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("RMS error (component 0)")
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
            plot_cl_rtf(data, get_timestamp())
