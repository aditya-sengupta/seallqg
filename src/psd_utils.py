import numpy as np
from scipy.signal import welch, windows
from matplotlib import pyplot as plt

def genpsd(tseries, dt, nseg=4):
	nperseg = 2**int(np.log2(tseries.shape[0]/nseg)) 
    # firstly ensures that nperseg is a power of 2 
    # secondly ensures that there are at least nseg segments per total time series length for noise averaging
	window = windows.hann(nperseg)
	freq, psd = welch(tseries, fs=1./dt, window=window, noverlap=nperseg*0.25, nperseg=nperseg, detrend=False,scaling='density')
	freq, psd = freq[1:],psd[1:] #remove DC component (freq=0 Hz)
	return freq, psd

def plot_rtf(f_ol, p_ol, f_cl, p_cl):
	fig, axs = plt.subplots(2)
	axs[0].loglog(f_ol, p_ol, label="OL")
	axs[0].loglog(f_cl, p_cl, label="CL")
	axs[0].set_xlabel("Frequency (Hz)")
	axs[0].set_ylabel("Power (DM units^2 / Hz)")
	axs[0].set_title("Open-loop and closed-loop PSDs")
	axs[0].legend()
	axs[1].loglog(f_ol, np.sqrt(p_cl / p_ol))
	axs[1].set_xlabel("Frequency (Hz)")
	axs[1].set_ylabel("Rejection (unitless)")
	axs[1].set_title("Rejection transfer function")
	
	