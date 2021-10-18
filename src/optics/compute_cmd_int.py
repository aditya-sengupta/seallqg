# authored by Benjamin Gerard
# edited by Aditya Sengupta

"""
Compute command matrix and interaction matrix.
"""

import time

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from .image import optics
from .tt import processim, funz
from .ao import polar_grid
from ..utils import joindata

dmc2wf = np.load(joindata("bestflats", "lodmc2wfe.npy"))
tsleep = 0.02

ydim, xdim = optics.dmdims
grid = np.mgrid[0:ydim, 0:xdim].astype(np.float32)
optics.set_expt(1e-3)
imydim, imxdim = optics.imdims

#DM aperture;
xy=np.sqrt((grid[0]-ydim/2+0.5)**2+(grid[1]-xdim/2+0.5)**2)
aperture=np.zeros(optics.dmdims).astype(np.float32)
aperture[np.where(xy<ydim/2)]=1
indap=np.where(aperture==1)
indnap=np.where(aperture==0)

remove_piston = lambda dmc: dmc-np.mean(dmc[indap]) #function to remove piston from dm command to have zero mean
tip = ((grid[0]-ydim/2+0.5)/ydim*2).astype(np.float32)
tilt = ((grid[1]-xdim/2+0.5)/ydim*2).astype(np.float32)

#calibrated image center and beam ratio from genDH.py
imxcen, imycen = np.load(joindata("bestflats", "imcen.npy"))
beam_ratio = np.load(joindata("bestflats", "beam_ratio.npy"))

#setup Zernike polynomials
nmarr = []
norder = 3 #how many radial Zernike orders to look at; just start with tip/tilt
for n in range(1, norder):
	for m in range(-n, n+1, 2):
		nmarr.append([n,m])

rho, phi = polar_grid(xdim, ydim)
rho[int((xdim-1)/2),int((ydim-1)/2)] = 0.00001 #avoid numerical divide by zero issues


gridim = np.mgrid[0:imydim,0:imxdim]
rim = np.sqrt((gridim[0]-imycen)**2+(gridim[1]-imxcen)**2)

ttmask = np.zeros(optics.imdims)
rmask = 10
indttmask = np.where(rim / beam_ratio < rmask)
ttmask[indttmask] = 1
IMamp = 0.001
sweep_amp = 5 * IMamp

def maskflat(z, mask=True):
	if mask:
		return z[indap]
	else:
		return z.flatten()

def make_zernarr(mask=True):
	if mask:
		s = aperture[indap].shape[0]
	else:
		s = aperture.size
	zernarr = np.zeros((len(nmarr), s)).astype(np.float32)
	bestflat, _ = optics.refresh()
	for (i, (n, m)) in enumerate(nmarr):
		zern = funz(n, m, IMamp, bestflat)
		zernarr[i] = maskflat(zern, mask)
	
	return zernarr

def make_im_cm(rcond=1e-3, verbose=True, mask=True):
	"""
	Make updated interaction and command matrices.
	"""
	if mask:
		s = aperture[indap].shape[0]
	else:
		s = aperture.size
	bestflat, imflat = optics.refresh()
	refvec = np.zeros((len(nmarr), ttmask[indttmask].shape[0]*2))
	zernarr = np.zeros((len(nmarr), s))
	for (i, (n, m)) in enumerate(nmarr):
		zern = funz(n, m, IMamp, bestflat)
		time.sleep(tsleep)
		imzern = optics.stackim(10)
		imdiff = imzern - imflat
		processed_imdiff = processim(imdiff)
		refvec[i] = np.array([
			np.real(processed_imdiff[indttmask]),
			np.imag(processed_imdiff[indttmask])
		]).flatten()
		zernarr[i] = maskflat(zern, mask)

	int_mtx = np.dot(refvec, refvec.T) #interaction matrix
	int_mtx_inv = np.linalg.pinv(int_mtx, rcond=rcond)
	cmd_mtx = np.dot(int_mtx_inv, refvec).astype(np.float32)
	if verbose:
		print("Recomputed interaction matrix and command matrix")
	return int_mtx, cmd_mtx

def measure_zcoeffs(image, cmd_mtx):
	"""
	Measures Zernike coefficient values from an image.
	"""
	tar_ini = processim(image)
	tar = np.array([np.real(tar_ini[indttmask]), np.imag(tar_ini[indttmask])])
	tar = tar.reshape((tar.size, 1))
	coeffs = np.dot(cmd_mtx, tar)
	return coeffs * IMamp

def genzerncoeffs(i, zernamp, cmd_mtx, bestflat, imflat):
	"""
	i: zernike mode
	zernamp: Zernike amplitude in DM units to apply
	"""
	n, m = nmarr[i]
	_ = funz(n, m, zernamp, bestflat)
	time.sleep(tsleep)
	imzern = optics.stackim(10)
	imdiff = imzern - imflat
	return measure_zcoeffs(imdiff, cmd_mtx)

def linearity(nlin=20, plot=True, rcond=1e-3):
	bestflat, imflat = optics.refresh()
	_, cmd_mtx = make_im_cm(rcond=rcond)
	zernamparr = sweep_amp * np.linspace(-1.5,1.5,nlin)
	zernampout = np.zeros((len(nmarr),len(nmarr),nlin))
	for nm in range(len(nmarr)):
		for i in range(nlin):
			zernamp = zernamparr[i]
			coeffsout = genzerncoeffs(nm, zernamp, cmd_mtx, bestflat, imflat)
			zernampout[nm,:,i] = coeffsout.flatten()

	optics.applybestflat()

	if plot:
		fig, axs=plt.subplots(ncols=3,nrows=2,figsize=(12,10),sharex=True,sharey=True)
		fig.suptitle(f"rcond = {rcond}")

		colors = mpl.cm.viridis(np.linspace(0,1,len(nmarr)))
		axarr=[axs[0,0],axs[0,1],axs[0,2],axs[1,0],axs[1,1],axs[1,2]]
		fig.delaxes(axarr[-1])
		for i in range(len(nmarr)):
			ax=axarr[i]
			ax.set_title('n,m='+str(nmarr[i][0])+','+str(nmarr[i][1]))
			if i==4:
				ax.plot(zernamparr*dmc2wf,zernamparr*dmc2wf,lw=1,color='k',ls='--',label='y=x')
				for j in range(len(nmarr)):
					if j==i:
						ax.plot(zernamparr*dmc2wf,zernampout[i,i,:]*dmc2wf,lw=2,color=colors[j],label='n,m='+str(nmarr[i][0])+','+str(nmarr[i][1]))
					else:
						ax.plot(zernamparr*dmc2wf,zernampout[i,j,:]*dmc2wf,lw=1,color=colors[j],label='n,m='+str(nmarr[j][0])+','+str(nmarr[j][1]))
			else:
				ax.plot(zernamparr*dmc2wf,zernamparr*dmc2wf,lw=1,color='k',ls='--')
				for j in range(len(nmarr)):
					if j==i:
						ax.plot(zernamparr*dmc2wf,zernampout[i,i,:]*dmc2wf,lw=2,color=colors[j])
					else:
						ax.plot(zernamparr*dmc2wf,zernampout[i,j,:]*dmc2wf,lw=1,color=colors[j])

		axarr[4].legend(bbox_to_anchor=(1.05,0.9))
		axarr[4].set_xlabel('input ($\\mu$m WFE, PV)')
		axarr[3].set_xlabel('input ($\\mu$m WFE, PV)')
		axarr[3].set_ylabel('reconstructed output ($\\mu$m WFE, PV)')
		axarr[0].set_ylabel('reconstructed output ($\\mu$m WFE, PV)')

	return zernamparr, zernampout

# fit_polynomial stuff removed on 2021-10-10, see git history before that to recover

mask = True
zernarr = make_zernarr(mask)

def zcoeffs_to_dmc(zcoeffs):
	"""
	Converts a measured coefficient value to an ideal DM command.
	
	Arguments
	---------
	zcoeffs : np.ndarray, (ncoeffs, 1)
	The tip and tilt values.

	Returns
	-------
	dmc : np.ndarray
	The corresponding DM command.
	"""
	if not mask:
		return np.dot(zernarr.T, -zcoeffs).reshape((ydim, xdim))
	else:
		dmc = np.copy(optics.dmzero)
		dmc[indap] = np.dot(zernarr.T, -zcoeffs)
		return dmc