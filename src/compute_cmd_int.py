"""
Compute command matrix and interaction matrix.
"""

from tt import *
import numpy as np
from numpy import float32
from numba import njit, objmode
import time
import ao

#initial setup: apply best flat, generate DM grid to apply future shapes
dmcini = getdmc()
ydim, xdim = dmcini.shape
grid = np.mgrid[0:ydim, 0:xdim].astype(float32)
#bestflat=np.load('dmc_dh.npy') #dark hole
bestflat = np.load('/home/lab/asengupta/data/bestflats/bestflat.npy') #load bestflat, which should be an aligned FPM
applydmc(bestflat)

expt(1e-3) #set exposure time
imini = getim()
imydim, imxdim = imini.shape

tsleep = 0.01 #should be the same values from align_fpm.py and genDH.py

#DM aperture:
undersize = 29/32 #29 of the 32 actuators are illuminated
rho,phi = ao.polar_grid(xdim,xdim*undersize)
aperture = np.zeros(rho.shape).astype(float32)
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
	z = ao.zernike(n, m, rhoap, phiap)/2
	zdm = amp*(z.astype(float32))
	dmc = remove_piston(remove_piston(bestflat)+remove_piston(zdm))
	applydmc(dmc)
	return dmc

#calibrated image center and beam ratio from genDH.py
imxcen, imycen = np.load('/home/lab/blgerard/imcen.npy')
beam_ratio = np.load('/home/lab/blgerard/beam_ratio.npy')
gridim = np.mgrid[0:imydim,0:imxdim]
rim = np.sqrt((gridim[0]-imycen)**2+(gridim[1]-imxcen)**2)

#algorithmic LOWFS mask (centered around the core, for light less than 6 lambda/D)
ttmask = np.zeros(imini.shape)
indttmask = np.where(rim/beam_ratio<6)
ttmask[indttmask] = 1

def vz(n, m, IMamp): #determine the minimum IMamp (interaction matrix amplitude) to be visible in differential images
	zern = funz(n, m, IMamp)
	time.sleep(tsleep)
	imzern = stack(10)
	applydmc(bestflat)
	time.sleep(tsleep)
	imflat = stack(10)
	ds9.view((imzern-imflat)*ttmask)

IMamp = 0.1 #from above function

#make MTF side lobe mask
xsidemaskcen, ysidemaskcen=252.01, 159.4 #x and y location of the side lobe mask in the cropped image
sidemaskrad = 26.8 #radius of the side lobe mask
mtfgrid = np.mgrid[0:imini.shape[0], 0:imini.shape[1]]
sidemaskrho = np.sqrt((mtfgrid[0]-ysidemaskcen)**2 + (mtfgrid[1]-xsidemaskcen)**2)
sidemask = np.zeros(imini.shape)
sidemaskind = np.where(sidemaskrho < sidemaskrad)
sidemask[sidemaskind] = 1

def fftshift(arr):
	"""
	Just does np.fft.fftshift, but JIT-able
	"""
	s = int(np.ceil(len(arr) // 2))
	return np.concatenate((arr[s:], arr[:s]))

def processim(imin): #process SCC image, isolating the sidelobe in the FFT and IFFT back to the image
	otf = np.fft.fftshift(np.fft.fft2(imin, norm='ortho')) #(1) FFT the image
	otf_masked = otf * sidemask #(2) multiply by binary mask to isolate side lobe
	Iminus = np.fft.ifft2(otf_masked, norm='ortho') #(3) IFFT back to the image plane, now generating a complex-valued image
	return Iminus

#make interaction matrix
refvec = np.zeros((len(nmarr), ttmask[indttmask].shape[0]*2))
zernarr = np.zeros((len(nmarr), aperture[indap].shape[0]))
for i in range(len(nmarr)):
	n, m = nmarr[i]
	zern = funz(n, m, IMamp)
	time.sleep(tsleep)
	imzern = stack(10)
	applydmc(bestflat)
	time.sleep(tsleep)
	imflat = stack(10)
	imdiff = imzern - imflat
	Im_diff = processim(imdiff)
	refvec[i] = np.array([np.real(Im_diff[indttmask]), np.imag(Im_diff[indttmask])]).flatten()
	zernarr[i] = zern[indap]

IM = np.dot(refvec, refvec.T) #interaction matrix
IMinv = np.linalg.pinv(IM, rcond=1e-3)
cmd_mtx = np.dot(IMinv, refvec)
print("Recomputed interaction matrix and command matrix")
time_cmd_mtx = time.time()
cmd_mtx_age = lambda: time.time() - time_cmd_mtx

applydmc(bestflat)
time.sleep(tsleep)
imflat = stack(100)

def measure_tt(im):
    """
	can't JIT this, so will have to explicitly pass it in
	if im is None:
        im = getim()
	"""
    tar_ini = processim(im)
    tar = np.array([np.real(tar_ini[indttmask]), np.imag(tar_ini[indttmask])]).flatten()	
    coeffs = np.dot(cmd_mtx, tar)
    return coeffs * IMamp

def pc(rcond,i,sf):
	'''
	rcond: svd cutoff to be optimized
	i: which Zernike mode to apply
	sf: scale factor for amplitude to apply of given Zernike mode as a fraction of the input IM amplitude
	'''
	#rcond=1e-3
	IMinv=np.linalg.pinv(IM,rcond=rcond)
	cmd_mtx=np.dot(IMinv,refvec)

	n,m=nmarr[i]
	zern=funz(n,m,IMamp*sf)
	time.sleep(tsleep)
	imzern=stack(10)
	applydmc(bestflat)
	time.sleep(tsleep)
	imflat=stack(10)
	imdiff=(imzern-imflat)
	Im_diff=processim(imdiff)
	tar = np.array([np.real(Im_diff[indttmask]),np.imag(Im_diff[indttmask])]).flatten()
	
	coeffs=np.dot(cmd_mtx,tar)
	plt.plot(coeffs*IMamp)
	plt.axhline(IMamp*sf,0,len(coeffs),ls='--')


if __name__ == "__main__":
	from matplotlib import pyplot as plt

	def genzerncoeffs(i, zernamp):
		'''
		i: zernike mode
		zernamp: Zernike amplitude in DM units to apply
		'''
		n, m = nmarr[i]
		zern = funz(n,m,zernamp)
		time.sleep(tsleep)
		imzern = stack(10)
		imdiff = (imzern-imflat)
		tar_ini = processim(imdiff)
		tar = np.array([np.real(tar_ini[indttmask]), np.imag(tar_ini[indttmask])]).flatten()	
		coeffs = np.dot(cmd_mtx, tar)
		return coeffs*IMamp

	nlin = 20 #number of data points to scan through linearity measurements
	zernamparr = np.linspace(-1.5*IMamp, 1.5*IMamp, nlin)
	#try linearity measurement for Zernike mode 0
	zernampout=np.zeros((len(nmarr),nlin))
	for i in range(nlin):
		zernamp=zernamparr[i]
		coeffsout=genzerncoeffs(0,zernamp)
		zernampout[:,i]=coeffsout
	
	applydmc(bestflat)

	plt.figure()
	plt.plot(zernamparr,zernamparr,lw=1,color='k',ls='--',label='y=x')
	plt.plot(zernamparr,zernampout[0,:],lw=2,color='k',label='i=0')
	plt.plot(zernamparr,zernampout[1,:],lw=2,color='blue',label='i=1')
	plt.legend(loc='best')

