# authored by Aditya Sengupta and Benjamin Gerard

import time
from copy import deepcopy
from datetime import datetime
from os import path
from socket import gethostname

import numpy as np
from scipy import signal, io
from scipy.signal import welch, windows

from .constants import dt

host = gethostname()
if (host == "skaya.local") or ("cam.ac.uk" in host):
	ROOTDIR = "/Users/adityasengupta/research/ao/sealrtc/"
elif host == "SEAL":
	ROOTDIR = "/home/lab/asengupta/sealrtc"
else:
	ROOTDIR = path.dirname(path.abspath("__file__"))
	
DATADIR = path.join(ROOTDIR, "data")
PLOTDIR = path.join(ROOTDIR, "plots")

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

def rmsz0(data, places=8):
	return rms(data[:,0], places)

def rmsz1(data, places=8):
	return rms(data[:,1], places)

def ratio(function, data1, data2, **kwargs):
	return function(data1, **kwargs) / function(data2, **kwargs)

def get_timestamp():
	return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

"""
Applies Zeno's paradox to "precisely" sleep for time 'duration'
"""
def zeno(duration):
	if duration > 0:
		t0 = time.time()
		while time.time() < t0 + duration:
			time.sleep(max(0, (time.time() - t0 - duration)/2))
		return time.time() - t0

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
