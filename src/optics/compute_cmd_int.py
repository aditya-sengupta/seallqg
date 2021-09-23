# authored by Benjamin Gerard
# edited by Aditya Sengupta

"""
Compute command matrix and interaction matrix.
"""

import numpy as np
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

#initial setup: apply best flat, generate DM grid to apply future shapes
dmcini = optics.getdmc()
ydim, xdim = dmcini.shape
grid = np.mgrid[0:ydim, 0:xdim].astype(np.float32)
bestflat = np.load(joindata(path.join("bestflats", "bestflat_{0}_{1}.npy".format(optics.name, optics.dmdims[0]))))
#load bestflat, which should be an aligned FPM
optics.applydmc(bestflat)
imflat = optics.stack(100)

# expt(1e-3) #set exposure time
imini = optics.getim()
imydim, imxdim = imini.shape

tsleep = 0.01 #should be the same values from align_fpm.py and genDH.py

#DM aperture:
undersize = 29/32 #29 of the 32 actuators are illuminated
rho,phi = polar_grid(xdim,xdim*undersize)
aperture = np.zeros(rho.shape).astype(np.float32)
indap = np.where(rho > 0)
indnap = np.where(rho == 0)
aperture[indap] = 1

def remove_piston(dmc):  #function to remove piston from dm command to have zero mean (must be intermediate) 
	dmcout = dmc - np.median(dmc[indap]) + 0.5 #remove piston in the pupil
	dmcout[indnap] = bestflat[indnap] #set the non-illuminated regions to the best flat values
	return dmcout

#setup Zernike polynomials
nmarr = []
norder = 2 #how many radial Zernike orders to look at; just start with tip/tilt
for n in range(1, norder):
	for m in range(-n, n+1, 2):
		nmarr.append([n,m])

def funz(n, m, amp, bestflat=bestflat): #apply zernike to the DM
	z = zernike(n, m, rhoap, phiap)/2
	zdm = amp*(z.astype(np.float32))
	dmc = remove_piston(remove_piston(bestflat)+remove_piston(zdm))
	optics.applydmc(dmc)
	return dmc

#calibrated image center and beam ratio from genDH.py
imxcen, imycen = np.load(joindata("bestflats/imcen.npy"))
beam_ratio = np.load(joindata("bestflats/beam_ratio.npy"))
gridim = np.mgrid[0:imydim,0:imxdim]
rim = np.sqrt((gridim[0]-imycen)**2+(gridim[1]-imxcen)**2)

#algorithmic LOWFS mask (centered around the core, for light less than 6 lambda/D)
ttmask = np.zeros(imini.shape)
indttmask = np.where(rim/beam_ratio<6)
ttmask[indttmask] = 1

def vz(n, m, IMamp): #determine the minimum IMamp (interaction matrix amplitude) to be visible in differential images
	# ds9 = pysao.ds9()
	zern = funz(n, m, IMamp)
	time.sleep(tsleep)
	imzern = optics.stack(10)
	applydmc(bestflat)
	time.sleep(tsleep)
	imflat = optics.stack(10)
	return imflat
	# ds9.view((imzern-imflat)*ttmask)

IMamp = 0.1 #from above function

def make_im_cm(verbose=True):
	#make interaction matrix
	refvec = np.zeros((len(nmarr), ttmask[indttmask].shape[0]*2))
	zernarr = np.zeros((len(nmarr), aperture[indap].shape[0]))
	for i in range(len(nmarr)):
		n, m = nmarr[i]
		zern = funz(n, m, IMamp)
		time.sleep(tsleep)
		imzern = optics.stack(10)
		optics.applydmc(bestflat)
		time.sleep(tsleep)
		imflat = optics.stack(10)
		imdiff = imzern - imflat
		Im_diff = processim(imdiff)
		refvec[i] = np.array([np.real(Im_diff[indttmask]), np.imag(Im_diff[indttmask])]).flatten()
		zernarr[i] = zern[indap]

	IM = np.dot(refvec, refvec.T) #interaction matrix
	IMinv = np.linalg.pinv(IM, rcond=1e-6)
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

def linearity(mode=0, nlin=20, amp=IMamp, plot=True):
	def genzerncoeffs(i, zernamp):
		'''
		i: zernike mode
		zernamp: Zernike amplitude in DM units to apply
		'''
		n, m = nmarr[i]
		zern = funz(n,m,zernamp)
		time.sleep(tsleep)
		imzern = optics.stack(10)
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
		coeffsout = genzerncoeffs(mode, zernamp)
		zernampout[:,i] = coeffsout
	
	applydmc(bestflat)

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
