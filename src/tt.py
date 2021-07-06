import numpy as np
from numpy import float32
import time
import itertools
from scipy.ndimage.filters import median_filter

from image import *

dmcini = getdmc()
ydim, xdim = dmcini.shape
grid=np.mgrid[0:ydim, 0:xdim].astype(float32)
#bestflat=np.load('bestflat_zopt.npy') #if running code after running zern_opt.py (i.e., non-coronagraphic PSF)
#bestflat=np.load('bestflat.npy') #if running code to realign coronagraphic PSF
#bestflat=np.load('bestflat_shwfs.npy')
bestflat = np.load('/home/lab/blgerard/bestflat.npy') #zygo, best flat
applydmc(bestflat)

expt(1e-4) #set exposure time; for 0.25 mW
imini=getim() #Andor image just for referencing dimensions

#DM aperture:
ydim,xdim=dmcini.shape
grid=np.mgrid[0:ydim,0:xdim].astype(float32)
ygrid,xgrid=grid[0]-ydim/2,grid[1]-xdim/2
tip,tilt=(ygrid+ydim/2)/ydim,(xgrid+xdim/2)/xdim #min value is zero, max is one

#DM aperture:
undersize=27/32 #assuming 27 of the 32 actuators are illuminated
rho,phi = ao.polar_grid(xdim,xdim*undersize)
cenaperture=np.zeros(rho.shape).astype(float32)
indapcen=np.where(rho>0)
cenaperture[indapcen]=1

#aperture=np.load('DMmap.npy').astype(float32) #actual aperture, from close_SHWFS_loop.py

#from comparing cenaperture and aperture, the actual aperture is shifted down and to the right (in ds9) each by 1 pixel from the center
yapcen,xapcen=ydim/2.-0.5-1,xdim/2.-0.5-1
rap=np.sqrt((grid[0]-yapcen)**2.+(grid[1]-xapcen)**2.)
rap[np.where(rap>xdim/2.*undersize)]=0.
rhoap=rap/np.max(rap)
phiap=np.arctan2(grid[1]-yapcen,grid[0]-xapcen)
indap=np.where(rhoap>0)

#remove_piston = lambda dmc: dmc-np.mean(dmc[indap]) #function to remove piston from dm command to have zero mean (must be intermediate)
remove_piston = lambda dmc: dmc-np.median(dmc)

#applying tip/tilt recursively (use if steering back onto the FPM after running zern_opt)
def applytip(amp): #apply tip; amp is the P2V in DM units
	dmc=getdmc()
	dmctip=amp*tip
	dmc=remove_piston(dmc)+remove_piston(dmctip)+0.5
	#applydmc(dmc*aperture)
	applydmc(dmc)

def applytilt(amp): #apply tilt; amp is the P2V in DM units
	dmc=getdmc()
	dmctilt=amp*tilt
	dmc=remove_piston(dmc)+remove_piston(dmctilt)+0.5
	#applydmc(dmc*aperture)
	applydmc(dmc)
