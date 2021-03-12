'''
code to attempt Zernike optimization, aka "focal plane sharpening"
'''

from ancillary_code import *
import numpy as np
import time

dmcini=getdmc()
ydim,xdim=dmcini.shape
grid=np.mgrid[0:ydim,0:xdim].astype(float32)
ygrid,xgrid=grid[0]-ydim/2,grid[1]-xdim/2
xy=np.sqrt(ygrid**2+xgrid**2)



tip,tilt=(ygrid+ydim/2)/ydim,(xgrid+xdim/2)/xdim #min value is zero, max is one
def applytip(amp): #apply tip; amp is the P2V in DM units
	dmc=getdmc()
	dmc=dmc+amp*tip
	applydmc(dmc)

def applytilt(amp): #apply tilt; amp is the P2V in DM units
	dmc=getdmc()
	dmc=dmc+amp*tilt
	applydmc(dmc)

dmzero=np.zeros(dmcini.shape).astype(float32)
applyzero = lambda : applydmc(dmzero)

#Zernike polynomials
rho,phi=functions.polar_grid(xdim,xdim)
aperture=np.zeros(rho.shape).astype(float32)
indap=np.where(rho>0)
aperture[indap]=1
nmarr=[]
norder=3 #how many radial Zernike orders to look at
for n in range(2,norder):
	for m in range(-n,n+1,2):
		nmarr.append([n,m])

remove_piston = lambda dmc: dmc-np.mean(dmc[indap]) #function to remove piston from dm command to have zero mean (must be intermediate)

def funz(n,m,amp,bestflat=dmzero): #apply zernike to the DM
	z=functions.zernike(n,m,rho,phi)/2
	zdm=amp*(z.astype(float32))
	dmcini=remove_piston(bestflat)+zdm
	dmc=dmcini+0.5 #ensure the mean piston is 0.5 to dm commands lie between 0 and 1
	applydmc(dmc*aperture)
	return dmc

namp=20 #how many grid points to walk through Zernike amplitude coefficients
amparr=np.linspace(-1,1,namp)

#focus loop
n,m=2,0
applyzero()
optarr=np.zeros(namp)
for a in range(namp):
	dmz=funz(n,m,amparr[a])
	time.sleep(5)
	ims=stack(10)
	optarr[a]=np.max(ims)/np.sum(ims)
	print('focus amp='+str(amparr[a]))
optamp=amparr[np.where(optarr==np.max(optarr))]
dmzo=funz(n,m,optamp[0])
bestflat=dmzo

for i in range(len(nmarr)):
	n,m=nmarr[i]
	if n==2 and m==0:
		continue #skip focus, which is already optimized
	print ('optimizing (n,m)=('+str(n)+','+str(m)+')')
	optarr=np.zeros(namp)
	for a in range(namp):
		dmz=funz(n,m,amparr[a],bestflat=bestflat)
		time.sleep(5)
		ims=stack(10)
		optarr[a]=np.max(ims)/np.sum(ims)
		print('amp='+str(amparr[a]))
	optamp=amparr[np.where(optarr==np.max(optarr))]
	dmzopt=funz(n,m,optamp[0],bestflat=bestflat)
	bestflat=dmzopt
	applydmc(bestflat)
	time.sleep(5)

np.save('bestflat.npy',bestflat)

#testing sine wave code:
ycen,xcen=ydim/2-0.5,xdim/2-0.5
rgrid=lambda pa:(grid[0]-ycen)/grid[0][-1,0]*np.cos(pa*np.pi/180)+(grid[1]-xcen)/grid[0][-1,0]*np.sin(pa*np.pi/180)
def dmsin(amp,freq,pa): 
	sin=amp*0.5*np.sin(2*np.pi*freq*rgrid(pa))
	sindm=sin.astype(float32)
	dmc=remove_piston(bestflat)+sindm+0.5
	applydmc(dmc*aperture)