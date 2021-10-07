from os import path
import numpy as np
from numpy import float32
from scipy import fft

from .ao import polar_grid, zernike, remove_piston
from .image import optics
from ..utils import joindata

# all of this should be part of an Optics instance

ydim, xdim = optics.dmdims
grid = np.mgrid[0:ydim, 0:xdim].astype(float32)
bestflat = np.load(joindata(path.join("bestflats", "bestflat_{0}_{1}.npy".format(optics.name, optics.dmdims[0]))))
optics.applydmc(bestflat)

optics.set_expt(1e-3) #set exposure time; for 0.15 mW
imydim, imxdim = optics.imdims

#DM aperture:
tsleep = 0.01 #should be the same values from align_fpm.py and genDH.py

#DM aperture:
undersize = 29/32 #29 of the 32 actuators are illuminated
rho,phi = polar_grid(xdim,xdim*undersize)
aperture = np.zeros(rho.shape).astype(float32)
indap = np.where(rho > 0)
indnap = np.where(rho == 0)
aperture[indap] = 1

ygrid, xgrid = grid[0] - ydim/2, grid[1] - xdim/2
tip, tilt = (ygrid + ydim/2)/ydim, (xgrid + xdim/2)/xdim #min value is zero, max is one

#DM aperture:
undersize = 27/32 #assuming 27 of the 32 actuators are illuminated
rho, phi = polar_grid(xdim, xdim*undersize)
cenaperture = np.zeros(rho.shape).astype(float32)
indapcen = np.where(rho > 0)
cenaperture[indapcen] = 1

#from comparing cenaperture and aperture, the actual aperture is shifted down and to the right (in ds9) each by 1 pixel from the center
yapcen,xapcen = ydim/2.-0.5-1, xdim/2.-0.5-1
rap = np.sqrt((grid[0]-yapcen)**2.+(grid[1]-xapcen)**2.)
rap[np.where(rap > xdim/2. * undersize)] = 0.
rhoap = rap / np.max(rap)
phiap = np.arctan2(grid[1]-yapcen, grid[0]-xapcen)
indap = np.where(rhoap > 0)

#applying tip/tilt recursively (use if steering back onto the FPM after running zern_opt)
def applytip(amp): #apply tip; amp is the P2V in DM units
	dmc = optics.getdmc()
	dmctip = amp*tip
	dmc = remove_piston(dmc) + remove_piston(dmctip)
	return optics.applydmc(dmc)

def applytilt(amp): #apply tilt; amp is the P2V in DM units
	dmc = optics.getdmc()
	dmctilt = amp*tilt
	dmc = remove_piston(dmc) + remove_piston(dmctilt)
	return optics.applydmc(dmc)

# add something to update best flat in here if needed
bestflat = optics.getdmc()

def applytiptilt(amptip, amptilt): #amp is the P2V in DM units
	dmctip = amptip*tip
	dmctilt = amptilt*tilt
	dmctiptilt = remove_piston(dmctip) + remove_piston(dmctilt) + remove_piston(bestflat) + 0.5 #combining tip, tilt, and best flat, setting mean piston to 0.5
	return optics.applydmc(dmctiptilt)

#setup Zernike polynomials
nmarr = []
norder = 2 #how many radial Zernike orders to look at; just start with tip/tilt
for n in range(norder):
	for m in range(-n, n+1, 2):
		nmarr.append([n, m])

def funz(n, m, amp, bestflat=bestflat): #apply zernike to the DM
	z = zernike(n,m,rhoap,phiap)/2
	zdm = amp*(z.astype(float32))
	dmc = remove_piston(remove_piston(bestflat)+remove_piston(zdm))
	optics.applydmc(dmc)
	return dmc

#calibrated image center and beam ratio from genDH.py
imxcen, imycen = np.load(joindata("bestflats/imcen.npy"))
beam_ratio = np.load(joindata("bestflats/beam_ratio.npy"))
gridim = np.mgrid[0:imydim, 0:imxdim]
rim = np.sqrt((gridim[0]-imycen)**2+(gridim[1]-imxcen)**2)

#algorithmic LOWFS mask (centered around the core, for light less than 6 lambda/D)
ttmask = np.zeros(optics.imdims)
indttmask = np.where(rim / beam_ratio<6)
ttmask[indttmask] = 1

IMamp = 0.1

#make MTF side lobe mask
xsidemaskcen,ysidemaskcen = 252.01, 159.4 #x and y location of the side lobe mask in the cropped image
sidemaskrad = 26.8 #radius of the side lobe mask
mtfgrid = np.mgrid[0:optics.imdims[0], 0:optics.imdims[1]].astype(float32)
sidemaskrho = np.sqrt((mtfgrid[0]-ysidemaskcen)**2+(mtfgrid[1]-xsidemaskcen)**2)
sidemask = np.zeros(optics.imdims, dtype=float32)
sidemaskind = np.where(sidemaskrho < sidemaskrad)
sidemask[sidemaskind] = 1

def processim(imin): #process SCC image, isolating the sidelobe in the FFT and IFFT back to the image
	otf = np.fft.fftshift(fft.fft2(imin, norm='ortho')) #(1) FFT the image
	otf_masked = otf * sidemask #(2) multiply by binary mask to isolate side lobe
	Iminus = fft.ifft2(otf_masked, norm='ortho') #(3) IFFT back to the image plane, now generating a complex-valued image
	return Iminus

tiptiltarr = np.array([tilt.flatten(), tip.flatten()]).T

def tt_to_dmc(tt):
    """
    Converts a measured tip-tilt value to an ideal DM command.
    
    Arguments
    ---------
    tt : np.ndarray, (2, 1)
    The tip and tilt values.

    Returns
    -------
    dmc : np.ndarray, (dm_x, dm_y)
    The corresponding DM command.
    """
    return np.matmul(tiptiltarr, -tt).reshape((ydim, xdim))
	