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
from .process_zern import processim, funz
from .process_zern import aperture, indap, remove_piston, tip, tilt
from .ao import polar_grid
from ..utils import joindata
from ..constants import tsleep

dmc2wf = np.load(joindata("bestflats", "lodmc2wfe.npy"))

ydim, xdim = optics.dmdims
grid = np.mgrid[0:ydim, 0:xdim].astype(np.float32)
imydim, imxdim = optics.imdims
optics.set_expt(1e-3)

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

zernarr = np.zeros((len(nmarr), aperture[indap].shape[0])).astype(np.float32)
bestflat, _ = optics.refresh()
for (i, (n, m)) in enumerate(nmarr):
	zern = funz(n, m, IMamp, bestflat)
	zernarr[i] = zern[indap]
	
def make_im_cm(rcond=1e-3, verbose=True):
	"""
	Make updated interaction and command matrices.
	"""
	bestflat, imflat = optics.refresh()
	refvec = np.zeros((len(nmarr), ttmask[indttmask].shape[0]*2))
	for (i, (n, m)) in enumerate(nmarr):
		_ = funz(n, m, IMamp, bestflat)
		time.sleep(tsleep)
		imzern = optics.stackim(10)
		imdiff = imzern - imflat
		processed_imdiff = processim(imdiff)
		refvec[i] = np.array([
			np.real(processed_imdiff[indttmask]),
			np.imag(processed_imdiff[indttmask])
		]).flatten()

	int_mtx = np.dot(refvec, refvec.T) #interaction matrix
	int_mtx_inv = np.linalg.pinv(int_mtx, rcond=rcond)
	cmd_mtx = np.dot(int_mtx_inv, refvec).astype(np.float32)
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

def mz(cmd_mtx=None):
	"""
	A quick IPython shortcut.
	"""
	if cmd_mtx is None:
		_, cmd_mtx = make_im_cm()
	return measure_zcoeffs(optics.getim(), cmd_mtx)

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
			coeffsout = genzerncoeffs(nm, zernamp, cmd_mtx, bestflat, imflat).flatten()
			zernampout[nm,:,i] = coeffsout
		for j in range(2):
			print(f"Percentage deviation for {nm}, mode {j}: {100 * (zernampout[nm,j,:] - zernamp) / zernamp}")

	optics.applybestflat()

	if plot:
		wfe_in_microns = False
		if wfe_in_microns:
			conv = dmc2wf
			unit = "$\\mu$m"
		else:
			conv = 1
			unit = "DM units"
		fig, axs=plt.subplots(ncols=3,nrows=2,figsize=(12,10),sharex=True,sharey=True)
		fig.suptitle(f"rcond = {rcond}")

		colors = mpl.cm.viridis(np.linspace(0,1,len(nmarr)))
		axarr=[axs[0,0],axs[0,1],axs[0,2],axs[1,0],axs[1,1],axs[1,2]]
		fig.delaxes(axarr[-1])
		for i in range(len(nmarr)):
			ax=axarr[i]
			ax.set_title('n,m='+str(nmarr[i][0])+','+str(nmarr[i][1]))
			if i==4:
				ax.plot(zernamparr*conv,zernamparr*conv,lw=1,color='k',ls='--',label='y=x')
				for j in range(len(nmarr)):
					if j==i:
						ax.plot(zernamparr*conv,zernampout[i,i,:]*conv,lw=2,color=colors[j],label='n,m='+str(nmarr[i][0])+','+str(nmarr[i][1]))
					else:
						ax.plot(zernamparr*conv,zernampout[i,j,:]*conv,lw=1,color=colors[j],label='n,m='+str(nmarr[j][0])+','+str(nmarr[j][1]))
			else:
				ax.plot(zernamparr*conv,zernamparr*conv,lw=1,color='k',ls='--')
				for j in range(len(nmarr)):
					if j==i:
						ax.plot(zernamparr*conv,zernampout[i,i,:]*conv,lw=2,color=colors[j])
					else:
						ax.plot(zernamparr*conv,zernampout[i,j,:]*conv,lw=1,color=colors[j])

		axarr[4].legend(bbox_to_anchor=(1.05,0.9))
		axarr[4].set_xlabel(f'input ({unit} WFE, PV)')
		axarr[3].set_xlabel(f'input ({unit} WFE, PV)')
		axarr[3].set_ylabel(f'reconstructed output ({unit} WFE, PV)')
		axarr[0].set_ylabel(f'reconstructed output ({unit} WFE, PV)')

	return zernamparr, zernampout

# fit_polynomial stuff removed on 2021-10-10, see git history before that to recover

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
	dmc = np.copy(optics.dmzero)
	dmc[indap] = np.dot(zernarr.T, -zcoeffs)
	return dmc
