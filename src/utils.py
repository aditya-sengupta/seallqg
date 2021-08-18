# authored by Aditya Sengupta and Benjamin Gerard

import numpy as np
from scipy import signal, io
from scipy.signal import welch, windows
from copy import deepcopy
from os import path
from socket import gethostname

from .constants import dt

host = gethostname()

if host == "Adityas-MacBook-Air.local":
	DATADIR = "/Users/adityasengupta/research/ao/set-tt-control/data/"
elif host == "SEAL":
	DATADIR = "/home/lab/asengupta/data/"
else:
	DATADIR = path.join(path.dirname(path.abspath("__file__")), "data")
	
joindata = lambda f: path.join(DATADIR, f)

rms = lambda data: np.sqrt(np.mean((data - np.mean(data)) ** 2))

def get_keck_tts(num=128):
	# gets the right keck TTs that have the wrong powerlaw.
	# put in a number 128 through 132
	filename = '../telemetry/n0' + str(num) + '_LGS_trs.sav'
	telemetry = io.readsav(filename)['b']
	commands = deepcopy(telemetry['DTTCOMMANDS'])[0]
	commands = commands - np.mean(commands, axis=0)
	residuals = telemetry['DTTCENTROIDS'][0]
	pol = residuals[1:] + commands[:-1]
	return residuals[1:], commands[:-1], pol

def make_impulse_2(overshoot, rise_time, T=np.arange(0, 1, 0.001)):
	"""
	Makes the impulse response for a second-order system with a specified overshoot and rise time.
	"""
	z = -np.log(overshoot) / np.sqrt(np.pi ** 2 + np.log(overshoot) ** 2)
	w = (1/rise_time) * (1.76 * z ** 3 - 0.417 * z ** 2 + 1.039 * z + 1)
	num = [w**2]
	den = [1, 2 * w * z, w**2]
	tf = signal.TransferFunction(num, den)
	y, t, _ = signal.impulse(tf, T=T)
	return t[0], y[1] / sum(y[1])

def make_impulse_1(w, T=np.arange(0, 1, 0.001)):
	"""
	Makes the impulse response for a first-order system (with TF = w / (s + w) for input w).
	"""
	tf = signal.TransferFunction([w], [1, w])
	y, t, _ = signal.impulse(tf, T=T)
	return t, y[1] / sum(y[1])

def genpsd(tseries, dt=dt, nseg=4, remove_dc=True):
	nperseg = 2**int(np.log2(tseries.shape[0]/nseg)) 
	# firstly ensures that nperseg is a power of 2 
	# secondly ensures that there are at least nseg segments per total time series length for noise averaging
	window = windows.hann(nperseg)
	freq, psd = welch(tseries, fs=1./dt, window=window, noverlap=nperseg*0.25, nperseg=nperseg, detrend=False,scaling='density')
	if remove_dc:
		freq, psd = freq[1:], psd[1:] #remove DC component (freq=0 Hz)
	return freq, psd
