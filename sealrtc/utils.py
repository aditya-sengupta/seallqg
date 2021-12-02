"""
Module-wide utilities for SEAL real-time control.
"""

import time

from time import monotonic_ns as mns
from copy import deepcopy
from datetime import datetime
from os import path
from socket import gethostname
from math import ceil

import numpy as np
from scipy import signal, io
from scipy.signal import welch, windows
from tqdm import tqdm

host = gethostname()
if (host == "skaya.local") or ("cam.ac.uk" in host):
	ROOTDIR = "/Users/adityasengupta/research/ao/sealrtc/"
elif host == "SEAL":
	ROOTDIR = "/home/lab/asengupta/sealrtc"
else:
	ROOTDIR = path.dirname(path.abspath("__file__"))
	
DATADIR = path.join(ROOTDIR, "data")
PLOTDIR = path.join(ROOTDIR, "plots")

if host != "SEAL":
    dmdims = (320, 320)
    fs = 100.0 # Hz
else:
    dmdims = (11, 11)
    fs = 100.0 # Hz
    
imdims = (320, 320)
dt = 1 / fs
wav0 = 1.65e-6 #assumed wav0 for sine amplitude input in meters
beam_ratio = 5.361256544502618 #pixels/resel
tsleep = 0.01

def joindata(*args):
	return path.join(DATADIR, *args)

def joinsimdata(*args):
	return path.join(DATADIR, "scc_sim", *args)

def joinplot(*args):
	return path.join(PLOTDIR, *args)

def rms(data, places=8):
	"""
	Computes the root-mean-square of `data` to `places` places.
	"""
	return round(
		np.sqrt( # root
			np.mean( # mean
				(data - np.mean(data)) ** 2 # square
			)
		),
		places
	)

def meanstd(data, places=5):
	return f"{round(np.mean(data), places)} $\\pm$ {round(np.std(data), places)}"

def rmsz0(result, places=8):
	return rms(result.measurements[:,0], places)

def rmsz1(result, places=8):
	return rms(result.measurements[:,1], places)

def ratio(function, data1, data2, places=5, **kwargs):
	return round(function(data1, **kwargs) / function(data2, **kwargs), places)

def rms_ratios(res1, res2):
	return [ratio(r, res1, res2) for r in [rmsz0, rmsz1]]

def get_timestamp():
	return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def zeno(dur):
	"""
	Applies Zeno's paradox to "precisely" sleep for time 'dur'
	"""
	if dur > 0:
		t0 = time.time()
		while time.time() < t0 + dur:
			time.sleep(max(0, (time.time() - t0 - dur)/2))
		return time.time() - t0

def spinlock_till(t):
	"""
	Spin-locks to precisely sleep until mns() = t
	"""
	i = 0
	while mns() < t:
		i += 1

def spinlock(dur):
	"""
	Spin-locks to precisely sleep for time 'dur'
	"""
	spinlock_till(mns() + ceil(dur / 1e-9))

def spin(process, dt, dur, use_tqdm=True):
	"""
	Spin-locks around a process to do it every "dt" seconds for time "dur" seconds.
	"""
	start_time = mns()
	ticks_per_iter = ceil(dt / 1e-9)
	next_tick = start_time
	end_time = start_time + ceil(dur / 1e-9)

	if use_tqdm:
		pbar = tqdm(total=ceil(dur / dt))

	while mns() < end_time:
		process()
		next_tick += ticks_per_iter * ((mns() - next_tick) // ticks_per_iter + 1)
		spinlock_till(next_tick)
		if use_tqdm:
			pbar.update(1)

	if use_tqdm:
		pbar.close()

def scheduled_loop(self, action, dt, dur, t_start, progress=False):
	spinlock_till(t_start)
	spin(action, dt, dur, progress)

# keck TTs deleted 2021-10-14

def make_impulse_2(overshoot, rise_time, times=np.arange(0, 1, 0.001)):
	"""
	Makes the impulse response for a second-order system with a specified overshoot and rise time.
	"""
	damp = -np.log(overshoot) / np.sqrt(np.pi ** 2 + np.log(overshoot) ** 2)
	omega = (1/rise_time) * (1.76 * damp ** 3 - 0.417 * damp ** 2 + 1.039 * damp + 1)
	num = [omega**2]
	den = [1, 2 * omega * damp, omega**2]
	transfer_fn = signal.TransferFunction(num, den)
	t, y = signal.impulse(transfer_fn, T=times)
	return t[0], y[1] / sum(y[1])

def make_impulse_1(w, T=np.arange(0, 1, 0.001)):
	"""
	Makes the impulse response for a first-order system (with TF = w / (s + w) for input w).
	"""
	tf = signal.TransferFunction([w], [1, w])
	t, y = signal.impulse(tf, T=T)
	return t, y[1] / sum(y[1])

def genpsd(tseries, dt=dt, nseg=4, remove_dc=True):
	nperseg = 2**int(np.log2(tseries.shape[0]/nseg))
	# firstly ensures that nperseg is a power of 2
	# secondly ensures that there are at least nseg segments 
	# per total time series length for noise averaging
	window = windows.hann(nperseg)
	freq, psd = welch(
		tseries,
		fs=1./dt,
		window=window,
		noverlap=nperseg*0.25,
		nperseg=nperseg,
		detrend=False,
		scaling='density'
	)
	if remove_dc:
		freq, psd = freq[1:], psd[1:] #remove DC component (freq=0 Hz)
	return freq, psd

def save_pupil_image(optics):
	expt_init = optics.get_expt()
	optics.set_expt(1e-3)
	image = optics.stackim(100)
	timestamp = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
	pupil_path = joindata("pupils", f"pupil_{timestamp}")
	np.save(pupil_path, image)
	optics.set_expt(expt_init)
	return image
