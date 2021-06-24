'''
optimize tip/tilt offset based on fringe visibility

so far, manual steering by eye seems better than the optimal determined value
'''
from ancillary_code import *
import numpy as np
from numpy import float32
import time
import ao_utils
import itertools
from scipy.ndimage.filters import median_filter

dmcini = getdmc()
ydim,xdim = dmcini.shape
grid = np.mgrid[0:ydim,0:xdim].astype(float32)
bestflat = np.load('bestflat_zopt.npy') #if running code after running zern_opt.py (i.e., non-coronagraphic PSF)
#bestflat = np.load('bestflat.npy') #if running code to realign coronagraphic PSF
#bestflat = np.load('bestflat_shwfs.npy')
applybestflat = lambda: applydmc(bestflat, False)
applybestflat()

expt(1e-4) #set exposure time; must be at 1e-4 to void saturating when steering off the FPM
imini = getim() #Andor image just for referencing dimensions

#DM aperture:
ydim, xdim = dmcini.shape
grid = np.mgrid[0:ydim, 0:xdim].astype(float32)
ygrid, xgrid = grid[0]-ydim/2, grid[1]-xdim/2
tip, tilt = (ygrid+ydim/2)/ydim, (xgrid+xdim/2)/xdim #min value is zero, max is one

#DM aperture:
undersize = 27/32 #assuming 27 of the 32 actuators are illuminated
rho, phi = ao_utils.polar_grid(xdim,xdim*undersize)
cenaperture = np.zeros(rho.shape).astype(float32)
indapcen = np.where(rho > 0)
cenaperture[indapcen] = 1

aperture = np.load('DMmap.npy').astype(float32) #actual aperture, from close_SHWFS_loop.py

#from comparing cenaperture and aperture, the actual aperture is shifted down and to the right (in ds9) each by 1 pixel from the center
yapcen,xapcen = ydim/2.-0.5-1,xdim/2.-0.5-1
rap = np.sqrt((grid[0]-yapcen)**2.+(grid[1]-xapcen)**2.)
rap[np.where(rap > xdim/2. *undersize)] = 0.
rhoap = rap/np.max(rap)
phiap = np.arctan2(grid[1]-yapcen,grid[0]-xapcen)
indap = np.where(rhoap>0)

remove_piston = lambda dmc: dmc-np.mean(dmc[indap]) #function to remove piston from dm command to have zero mean (must be intermediate)

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

#MANUALLY USE ABOVE FUNCTIONS TO STEER THE PSF BACK ONTO THE FPM AS NEEDED, then:
bestflat = getdmc()

#apply tip/tilt starting only from the bestflat point (start here if realigning the non-coronagraphic PSF) 
def applytiptilt(amptip, amptilt, bestflat = bestflat, verbose=True): #amp is the P2V in DM units
	dmctip = amptip*tip
	dmctilt = amptilt*tilt
	dmctiptilt = remove_piston(dmctip)+remove_piston(dmctilt)+remove_piston(bestflat)+0.5 #combining tip, tilt, and best flat, setting mean piston to 0.5
	return applydmc(aperture*dmctiptilt, verbose)
