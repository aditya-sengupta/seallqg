import numpy as np
from numpy import float32
import time
import itertools
from scipy.ndimage.filters import median_filter

from image import *

dmcini = getdmc()
ydim, xdim = dmcini.shape
grid = np.mgrid[0:ydim, 0:xdim].astype(float32)
#bestflat=np.load('bestflat_zopt.npy') #if running code after running zern_opt.py (i.e., non-coronagraphic PSF)
#bestflat=np.load('bestflat.npy') #if running code to realign coronagraphic PSF
#bestflat=np.load('bestflat_shwfs.npy')
bestflat = np.load('/home/lab/blgerard/bestflat.npy') #zygo, best flat
applydmc(bestflat)

expt(1e-4) #set exposure time; for 0.25 mW
imini = getim() #Andor image just for referencing dimensions
imydim, imxdim = imini.shape

#DM aperture:
tsleep=0.01 #should be the same values from align_fpm.py and genDH.py

#DM aperture:
undersize=29/32 #29 of the 32 actuators are illuminated
rho,phi = ao.polar_grid(xdim,xdim*undersize)
aperture=np.zeros(rho.shape).astype(float32)
indap=np.where(rho>0)
indnap=np.where(rho==0)
aperture[indap]=1

ydim,xdim=dmcini.shape
grid=np.mgrid[0:ydim,0:xdim].astype(float32)
ygrid, xgrid = grid[0]-ydim/2, grid[1]-xdim/2
tip, tilt = (ygrid+ydim/2)/ydim, (xgrid+xdim/2)/xdim #min value is zero, max is one

#DM aperture:
undersize = 27/32 #assuming 27 of the 32 actuators are illuminated
rho,phi = ao.polar_grid(xdim, xdim*undersize)
cenaperture = np.zeros(rho.shape).astype(float32)
indapcen = np.where(rho>0)
cenaperture[indapcen] = 1

#aperture=np.load('DMmap.npy').astype(float32) #actual aperture, from close_SHWFS_loop.py

#from comparing cenaperture and aperture, the actual aperture is shifted down and to the right (in ds9) each by 1 pixel from the center
yapcen,xapcen = ydim/2.-0.5-1, xdim/2.-0.5-1
rap = np.sqrt((grid[0]-yapcen)**2.+(grid[1]-xapcen)**2.)
rap[np.where(rap>xdim/2.*undersize)] = 0.
rhoap = rap / np.max(rap)
phiap = np.arctan2(grid[1]-yapcen, grid[0]-xapcen)
indap = np.where(rhoap > 0)

#remove_piston = lambda dmc: dmc-np.mean(dmc[indap]) #function to remove piston from dm command to have zero mean (must be intermediate)
remove_piston = lambda dmc: dmc-np.median(dmc)

#applying tip/tilt recursively (use if steering back onto the FPM after running zern_opt)
def applytip(amp, verbose=True): #apply tip; amp is the P2V in DM units
	dmc = getdmc()
	dmctip = amp*tip
	dmc = remove_piston(dmc)+remove_piston(dmctip)+0.5
	return applydmc(dmc*aperture, verbose)

def applytilt(amp, verbose=True): #apply tilt; amp is the P2V in DM units
	dmc = getdmc()
	dmctilt = amp*tilt
	dmc = remove_piston(dmc)+remove_piston(dmctilt)+0.5
	return applydmc(dmc*aperture, verbose)

def applytiptilt(amptip, amptilt, bestflat=bestflat, verbose=True): #amp is the P2V in DM units
	dmctip = amptip*tip
	dmctilt = amptilt*tilt
	dmctiptilt = remove_piston(dmctip)+remove_piston(dmctilt)+remove_piston(bestflat)+0.5 #combining tip, tilt, and best flat, setting mean piston to 0.5
	return applydmc(aperture*dmctiptilt, verbose)

#setup Zernike polynomials
nmarr = []
norder = 2 #how many radial Zernike orders to look at; just start with tip/tilt
for n in range(norder):
	for m in range(-n, n+1, 2):
		nmarr.append([n, m])

def funz(n, m, amp, bestflat=bestflat): #apply zernike to the DM
	z = ao.zernike(n,m,rhoap,phiap)/2
	zdm = amp*(z.astype(float32))
	dmc = remove_piston(remove_piston(bestflat)+remove_piston(zdm))
	applydmc(dmc)
	return dmc

#calibrated image center and beam ratio from genDH.py
imxcen, imycen = np.load('/home/lab/blgerard/imcen.npy')
beam_ratio = np.load('/home/lab/blgerard/beam_ratio.npy')
gridim = np.mgrid[0:imydim, 0:imxdim]
rim = np.sqrt((gridim[0]-imycen)**2+(gridim[1]-imxcen)**2)

#algorithmic LOWFS mask (centered around the core, for light less than 6 lambda/D)
ttmask = np.zeros(imini.shape)
indttmask = np.where(rim/beam_ratio<6)
ttmask[indttmask] = 1

IMamp = 0.1

#make MTF side lobe mask
xsidemaskcen,ysidemaskcen = 252.01, 159.4 #x and y location of the side lobe mask in the cropped image
sidemaskrad = 26.8 #radius of the side lobe mask
mtfgrid = np.mgrid[0:imini.shape[0], 0:imini.shape[1]]
sidemaskrho = np.sqrt((mtfgrid[0]-ysidemaskcen)**2+(mtfgrid[1]-xsidemaskcen)**2)
sidemask = np.zeros(imini.shape)
sidemaskind = np.where(sidemaskrho < sidemaskrad)
sidemask[sidemaskind] = 1

def processim(imin): #process SCC image, isolating the sidelobe in the FFT and IFFT back to the image
	otf=np.fft.fftshift(np.fft.fft2(imin,norm='ortho')) #(1) FFT the image
	otf_masked=otf*sidemask #(2) multiply by binary mask to isolate side lobe
	Iminus=np.fft.ifft2(otf_masked,norm='ortho') #(3) IFFT back to the image plane, now generating a complex-valued image
	return Iminus

# make interaction matrix
"""refvec = np.zeros((len(nmarr), ttmask[indttmask].shape[0]*2))
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
IMinv = np.linalg.pinv(IM,rcond=1e-3)
cmd_mtx = np.dot(IMinv, refvec)"""
