'''
code to attempt Zernike optimization, aka "focal plane sharpening"
'''

from ancillary_code import *
import numpy as np
import time
import functions

#setup
bestflat=np.load('bestflat.npy')
applydmc(bestflat)
dmcini=getdmc()

#DM aperture:
rho,phi=functions.polar_grid(xdim,xdim*29/32) #assuming 29 of the 32 actuators are illuminated
aperture=np.zeros(rho.shape).astype(float32)
indap=np.where(rho>0)
aperture[indap]=1
remove_piston = lambda dmc: dmc-np.mean(dmc[indap]) #function to remove piston from dm command to have zero mean (must be intermediate)

#applytip/tilt, manually steering the PSF around until the PSF is sufficiently off the FPM and not saturating DM commands
ydim,xdim=dmcini.shape
grid=np.mgrid[0:ydim,0:xdim].astype(float32)
ygrid,xgrid=grid[0]-ydim/2,grid[1]-xdim/2
tip,tilt=(ygrid+ydim/2)/ydim,(xgrid+xdim/2)/xdim #min value is zero, max is one
def applytip(amp): #apply tip; amp is the P2V in DM units
	dmc=getdmc()
	dmctip=amp*tip
	dmc=remove_piston(dmc)+remove_piston(dmctip)+0.5
	applydmc(dmc)

def applytilt(amp): #apply tilt; amp is the P2V in DM units
	dmc=getdmc()
	dmctilt=amp*tilt
	dmc=remove_piston(dmc)+remove_piston(dmctilt)+0.5
	applydmc(dmc*aperture)

ttdmc=getdmc()

#Zernike polynomials
nmarr=[]
norder=6 #how many radial Zernike orders to look at
for n in range(2,norder):
	for m in range(-n,n+1,2):
		nmarr.append([n,m])

def funz(n,m,amp,bestflat=ttdmc): #apply zernike to the DM
	z=functions.zernike(n,m,rho,phi)/2
	zdm=amp*(z.astype(float32))
	dmc=remove_piston(bestflat)+remove_piston(zdm)+0.5
	applydmc(dmc*aperture)
	return dmc

namp=20 #how many grid points to walk through Zernike amplitude coefficients
amparr=np.linspace(-0.2,0.2,namp)

tsleep=0.05 #time to sleep in between applying DM commands and grabbing images, determined in optt of genIM.py

#focus loop
n,m=2,0
applyzero()
optarr=np.zeros(namp)
for a in range(namp):
	dmz=funz(n,m,amparr[a])
	time.sleep(tsleep)
	ims=stack(10)
	optarr[a]=np.max(ims)/np.sum(ims)
	print('focus amp='+str(amparr[a]))
optamp=amparr[np.where(optarr==np.max(optarr))]
dmzo=funz(n,m,optamp[0])
bestflat=dmzo
plt.figure()
plt.plot(amparr,optarr,label='n,m='+str(n)+','+str(m))
plt.ylabel('normalized SR (unitless)')
plt.xlabel('Zernike amplitude (DM units)')
for i in range(len(nmarr)):
	n,m=nmarr[i]
	if n==2 and m==0:
		continue #skip focus, which is already optimized
	print ('optimizing (n,m)=('+str(n)+','+str(m)+')')
	optarr=np.zeros(namp)
	for a in range(namp):
		dmz=funz(n,m,amparr[a],bestflat=bestflat)
		time.sleep(tsleep)
		ims=stack(10)
		optarr[a]=np.max(ims)/np.sum(ims)
		print('amp='+str(amparr[a]))
	optamp=amparr[np.where(optarr==np.max(optarr))]
	dmzopt=funz(n,m,optamp[0],bestflat=bestflat)
	bestflat=dmzopt
	applydmc(bestflat)
	time.sleep(tsleep)
	plt.plot(amparr,optarr,label='n,m='+str(n)+','+str(m))
plt.legend(bbox_to_anchor=(1.05,1),loc='upper left')
plt.tight_layout()

np.save('bestflat_zopt.npy',bestflat)

#testing sine wave code:
ycen,xcen=ydim/2-0.5,xdim/2-0.5
rgrid=lambda pa:(grid[0]-ycen)/grid[0][-1,0]*np.cos(pa*np.pi/180)+(grid[1]-xcen)/grid[0][-1,0]*np.sin(pa*np.pi/180)
def dmsin(amp,freq,pa): 
	sin=amp*0.5*np.sin(2*np.pi*freq*rgrid(pa))
	sindm=sin.astype(float32)
	dmc=remove_piston(bestflat)+sindm+0.5
	applydmc(dmc*aperture)

def dmsin2(amp1,amp2,freq1,freq2,pa1,pa2): 
	sin1=amp1*0.5*np.sin(2*np.pi*freq1*rgrid(pa1))
	sin2=amp2*0.5*np.sin(2*np.pi*freq2*rgrid(pa2))
	sindm1=sin1.astype(float32)
	sindm2=sin2.astype(float32)
	dmc=remove_piston(bestflat)+sindm1+sindm2+0.5
	applydmc(dmc*aperture)
