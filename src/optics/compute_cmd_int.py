# authored by Benjamin Gerard
# edited by Aditya Sengupta

"""
Compute command matrix and interaction matrix.
"""

import numpy as np
from numpy import float32
import time
import tqdm
# import pysao
from os import path
from matplotlib import pyplot as plt
from scipy.optimize import newton

from .image import optics
from .tt import rhoap, phiap, processim
from .ao import polar_grid, zernike
from ..utils import joindata

dmc2wf = np.load(joindata(path.join("bestflats", "lodmc2wfe.npy")))

def make_im_cm(verbose=True, rcond=1e-3):
	#make interaction matrix
	refvec = np.zeros((len(nmarr), ttmask[indttmask].shape[0]*2))
	zernarr = np.zeros((len(nmarr), aperture[indap].shape[0]))
	for i in range(len(nmarr)):
		n, m = nmarr[i]
		zern = funz(n, m, IMamp)
		time.sleep(tsleep)
		imzern = optics.stackim(10)
		optics.applydmc(bestflat)
		time.sleep(tsleep)
		imflat = optics.stackim(10)
		imdiff = imzern - imflat
		Im_diff = processim(imdiff)
		refvec[i] = np.array([np.real(Im_diff[indttmask]), np.imag(Im_diff[indttmask])]).flatten()
		zernarr[i] = zern[indap]

	IM = np.dot(refvec, refvec.T) #interaction matrix
	IMinv = np.linalg.pinv(IM, rcond=rcond)
	cmd_mtx = np.dot(IMinv, refvec).astype(np.float32)
	if verbose:
		print("Recomputed interaction matrix and command matrix")
	return IM, cmd_mtx

IM, cmd_mtx = make_im_cm()
optics.applydmc(bestflat)

def measure_tt(image, cmd_mtx=cmd_mtx):
	tar_ini = processim(image)
	tar = np.array([np.real(tar_ini[indttmask]), np.imag(tar_ini[indttmask])])
	tar = tar.reshape((tar.size, 1))
	coeffs = np.dot(cmd_mtx, tar)
	return coeffs * IMamp

def linearity(nlin=20, amp=IMamp, plot=True):
	def genzerncoeffs(i, zernamp):
		'''
		i: zernike mode
		zernamp: Zernike amplitude in DM units to apply
		'''
		n, m = nmarr[i]
		zern = funz(n,m,zernamp)
		time.sleep(tsleep)
		imzern = optics.stackim(10)
		imdiff = (imzern-imflat)
		tar_ini = processim(imdiff)
		tar = np.array([np.real(tar_ini[indttmask]), np.imag(tar_ini[indttmask])]).flatten()	
		coeffs = np.dot(cmd_mtx, tar)
		return coeffs*IMamp

	nlin = 20 #number of data points to scan through linearity measurements
	zernamparr = np.linspace(-1.5*amp, 1.5*amp, nlin)
	#try linearity measurement for Zernike mode 'mode'
	zernampout=np.zeros((len(nmarr), nlin))
	for i in tqdm.trange(nlin):
		zernamp=zernamparr[i]
		coeffsout = genzerncoeffs(i, zernamp)
		zernampout[:,i] = coeffsout
	
	optics.applydmc(bestflat)

	if plot:
		plt.figure()
		plt.plot(zernamparr,zernamparr,lw=1,color='k',ls='--',label='y=x')
		plt.plot(zernamparr,zernampout[0,:],lw=2,color='k',label='i=0')
		plt.plot(zernamparr,zernampout[1,:],lw=2,color='blue',label='i=1')
		plt.legend(loc='best')
		plt.xlabel("Applied command")
		plt.ylabel("System response")
		plt.show()

	return zernamparr, zernampout

def fit_polynomial(x, y, maxdeg=10, abstol=1e-4):
	"""
	Slight extension to np.polyfit that varies the degree.
	"""
	for deg in range(maxdeg):
		p = np.polyfit(x, y, deg=deg)
		errs = np.polyval(p, x) - y
		err = np.sum(errs ** 2)
		if err <= abstol:
			return p, err
	return p, err

def fit_linearity_curves():
	ps = []
	for mode in range(2):
		zin, zout = linearity(mode=mode, plot=False)
		ps.append(fit_polynomial(zin, zout[mode])[0])
	return ps

def command_for_actual(act, p):
	"""
	Uses Newton's method and a polynomial fit to the linearity curve 
	to calculate the command you should send to get an actual response of 'act'.
	"""
	d = len(p)
	obj = lambda x: np.dot(p, x ** np.arange(d-1, -1, -1)) - act
	dobj = lambda x: np.dot(np.arange(d-1, -1, -1), x ** np.arange(d-2, -2, -1))
	return newton(obj, act, fprime=dobj)

if __name__ == "__main__":
	linearity()
