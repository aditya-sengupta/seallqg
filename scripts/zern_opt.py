'''
code to attempt Zernike optimization, aka "focal plane sharpening"
'''

from ancillary_code import *
import numpy as np
from numpy import float32
import time
import ao_utils

#setup
#bestflat = np.load('bestflat.npy')
#bestflat=np.load('bestflat_off_fpm.npy')
#bestflat = np.load('bestflat_shwfs.npy')
bestflat = np.load('bestflat_zopt.npy') #for bootstrapping
applybestflat = lambda: applydmc(bestflat, False)
applybestflat()
dmcini = getdmc()

#applytip/tilt, manually steering the PSF around until the PSF is sufficiently off the FPM and not saturating DM commands
ydim,xdim = dmcini.shape
grid = np.mgrid[0:ydim,0:xdim].astype(float32)
ygrid,xgrid = grid[0]-ydim/2,grid[1]-xdim/2
tip,tilt = (ygrid+ydim/2)/ydim,(xgrid+xdim/2)/xdim #min value is zero, max is one

#old DM aperture:
'''
undersize=27/32 #assuming 27 of the 32 actuators are illuminated
rho,phi=functions.polar_grid(xdim,xdim*undersize)
cenaperture=np.zeros(rho.shape).astype(float32)
indapcen=np.where(rho>0)
cenaperture[indapcen]=1

aperture = np.load('DMmap.npy').astype(float32) #actual aperture, from close_SHWFS_loop.py

#from comparing cenaperture and aperture, the actual aperture is shifted down and to the right (in ds9) each by 1 pixel from the center
yapcen,xapcen=ydim/2.-0.5-1,xdim/2.-0.5-1
rap=np.sqrt((grid[0]-yapcen)**2.+(grid[1]-xapcen)**2.)
rap[np.where(rap>xdim/2.*undersize)]=0.
rhoap=rap/np.max(rap)
phiap=np.arctan2(grid[1]-yapcen,grid[0]-xapcen)
indap=np.where(rhoap>0)
'''

#regular DM aperture:
undersize=29/32 #29 of the 32 actuators are illuminated
rho,phi=functions.polar_grid(xdim,xdim*undersize)
aperture=np.zeros(rho.shape).astype(float32)
indap=np.where(rho>0)
indnap=np.where(rho==0)
aperture[indap]=1

remove_piston  =  lambda dmc: dmc-np.mean(dmc[indap]) #function to remove piston from dm command to have zero mean (must be intermediate)

IMtt = np.array([(tip*aperture).flatten(),(tilt*aperture).flatten()])
CMtt = np.linalg.pinv(IMtt,rcond = 1e-5)
def rmtt(ph): #remove tip/tilt from aperture
	coeffs = np.dot(np.vstack((ph*aperture).flatten()).T,CMtt) 
	lsqtt = np.dot(IMtt.T,coeffs.T).reshape(tilt.shape).astype(float32)
	return ph*aperture-lsqtt

def applytip(amp): #apply tip; amp is the P2V in DM units
	dmc = getdmc()
	dmctip = amp*tip
	dmc = remove_piston(dmc)+remove_piston(dmctip)+0.5
	applydmc(dmc*aperture)

def applytilt(amp): #apply tilt; amp is the P2V in DM units
	dmc = getdmc()
	dmctilt = amp*tilt
	dmc = remove_piston(dmc)+remove_piston(dmctilt)+0.5
	applydmc(dmc*aperture)

ttdmc = getdmc()
#np.save('bestflat_offfpm.npy',ttdmc)

#Zernike polynomials
nmarr = []
norder = 15 #how many radial Zernike orders to look at
for n in range(2,norder):
	for m in range(-n,n+1,2):
		nmarr.append([n,m])

def funz(n,m,amp,bestflat=ttdmc): #apply zernike to the DM
	z=functions.zernike(n,m,rhoap,phiap)/2
	zdm=amp*(z.astype(float32))
	dmc=np.zeros(aperture.shape).astype(float32)
	dmc[indap]=(remove_piston(bestflat)+remove_piston(rmtt(zdm))+0.5)[indap]
	dmc[indnap]=bestflat[indnap]
	applydmc(dmc)
	return dmc

namp = 100 #how many grid points to walk through Zernike amplitude coefficients
amparr = np.linspace(-1,1,namp)

tsleep = 0.05 #time to sleep in between applying DM commands and grabbing images, determined in optt of genIM.py

#focus loop
n,m = 2,0
applyzero()
optarr = np.zeros(namp)
for a in range(namp):
	dmz = funz(n,m,amparr[a])
	time.sleep(tsleep)
	ims = stack(10)
	optarr[a] = np.max(ims)/np.sum(ims)
	print('focus amp = '+str(amparr[a]))
optamp = amparr[np.where(optarr == np.max(optarr))]
dmzo = funz(n,m,optamp[0])
bestflat = dmzo
plt.subplots(figsize = (17,5))
plt.plot(amparr,optarr,label = 'n,m = '+str(n)+','+str(m))
plt.ylabel('normalized SR (unitless)')
plt.xlabel('Zernike amplitude (DM units)')
for i in range(len(nmarr)):
	n,m = nmarr[i]
	if n == 2 and m == 0:
		continue #skip focus, which is already optimized
	print ('optimizing (n,m) = ('+str(n)+','+str(m)+')')
	optarr = np.zeros(namp)
	for a in range(namp):
		dmz = funz(n,m,amparr[a],bestflat = bestflat)
		time.sleep(tsleep)
		ims = stack(10)
		optarr[a] = np.max(ims)/np.sum(ims)
		print('amp = '+str(amparr[a]))
	optamp = amparr[np.where(optarr == np.max(optarr))]
	dmzopt = funz(n,m,optamp[0],bestflat = bestflat)
	bestflat = dmzopt
	applybestflat()
	time.sleep(tsleep)
	plt.plot(amparr,optarr,label = 'n,m = '+str(n)+','+str(m))
plt.legend(bbox_to_anchor = (1.05,1),loc = 'upper left',ncol = 5)
plt.tight_layout()

#np.save('bestflat_zopt_daren.npy',bestflat) #version for when Darren mooves the PSD off the FPM
np.save('bestflat_zopt.npy',bestflat)

#testing sine wave code:
ycen,xcen = ydim/2-0.5,xdim/2-0.5
rgrid = lambda pa:(grid[0]-ycen)/grid[0][-1,0]*np.cos(pa*np.pi/180)+(grid[1]-xcen)/grid[0][-1,0]*np.sin(pa*np.pi/180)
def dmsin(amp,freq,pa): 
	sin = amp*0.5*np.sin(2*np.pi*freq*rgrid(pa))
	sindm = sin.astype(float32)
	dmc = remove_piston(bestflat)+sindm+0.5
	applydmc(dmc*aperture)

def dmsin2(amp1,amp2,freq1,freq2,pa1,pa2): 
	sin1 = amp1*0.5*np.sin(2*np.pi*freq1*rgrid(pa1))
	sin2 = amp2*0.5*np.sin(2*np.pi*freq2*rgrid(pa2))
	sindm1 = sin1.astype(float32)
	sindm2 = sin2.astype(float32)
	dmc = remove_piston(bestflat)+sindm1+sindm2+0.5
	applydmc(dmc*aperture)
