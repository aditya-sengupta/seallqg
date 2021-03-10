'''
code to attempt Zernike optimization, aka "focal plane sharpening"
'''

from ancillary_code import *
import numpy as np
import time

dmcini=getdmc()
ydim,xdim=dmcini.shape
grid=np.mgrid[0:ydim,0:xdim]
ygrid,xgrid=grid[0]-ydim/2,grid[1]-xdim/2
xy=np.sqrt(ygrid**2+xgrid**2)

tip,tilt=ygrid/ydim,xgrid/xdim
def applytiptilt(amp,tip=True,tilt=True): #apply tip/tilt; amp is the P2V in DM units
	dmtip,dmtilt=np.zeros(dmcini.shape),np.zeros(dmcini.shape)
	if tip==True:
		dmtip=amp*tip
	if tip==False:
		dmtilt=amp*tilt
	dmc=getdmc()
	dmc=dmc+dmtip+dmtilt
	applydmc(dmc)

#Zernike polynomials
rho,phi=functions.polar_grid(xdim,xdim)
nmarr=[]
norder=5 #how many radial Zernike orders to look at
for n in range(2,norder):
	for m in range(-n,n+1,2):
		nmarr.append([n,m])

def funz(n,m,amp): #apply zernike to the DM
	z=amp*functions.zernike(n,m,rho,phi)/2
	cmd=getdmc()
	applydmc(cmd+z)
	time.sleep(0.2)

namp=20 #how many grid points to walk through Zernike amplitude coefficients
amparr=np.linspace(-1,1,namp)

for i in range(len(nmarr)):
	n,m=nmarr[i]
	print ('optimizing (n,m)=('+str(n)+','+str(m)+')')
	optarr=np.zeros(namp)
	for a in range(namp):
		funz(i,amparr[a])
		time.sleep(0.1)
		im=stack(10)
		optarr[a]=np.max(im)/np.sum(im)
	optamp=amparr[np.where(optarr==np.max(optarr))]
	funz(i,optamp)
