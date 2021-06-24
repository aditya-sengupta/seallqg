from krtc import *
import pysao
import numpy as np
import sys
import functions

#initialize; no need to load this more than once
#for full frame:

a=shmlib.shm('/tmp/ca01dit.im.shm') 
im=shmlib.shm('/tmp/ca01im.im.shm')

#for 150x150 subarray
'''
a=shmlib.shm('/tmp/ca03dit.im.shm') 
im=shmlib.shm('/tmp/ca03im.im.shm')
'''
def expt(t):
	'''
	change the exposure time

	for the large array the smallest exposure time is around 1e-5
	'''
	dit[0][0]=t; a.set_data(dit)

#To view images in pysao ds9:
def getim():
	return im.get_data()
def vim(): #view image
	ds9.view(getim())

def vmtf(): #view image MTF
	imm=getim()
	mtf=np.abs(np.fft.fftshift(np.fft.fft2(imm)))
	ds9.view(mtf)

def stack(n):
	ims=np.zeros(getim().shape)
	for i in range(n):
		ims=ims+getim()
	ims=ims/n
	return ims

mtf = lambda im: np.abs(np.fft.fftshift(np.fft.fft2(im)))


#DM commands

b=shmlib.shm('/tmp/dm02itfStatus.im.shm')
status=b.get_data()
status[0,0]=1
b.set_data(status)
dmChannel=shmlib.shm('/tmp/dm02disp01.im.shm')

def getdmc(): # read current command applied to the DM
	return dmChannel.get_data()
def applydmc(cmd): #apply command to the DM
	dmChannel.set_data(cmd)

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

def funz(i,amp):
	n,m=nmarr[i]
	z=amp*functions.zernike(n,m,rho,phi)/2
	applydmc(z)
	time.sleep(0.2)

sys.exit()

dmsin=lambda freq: np.sin(2*np.pi*freq*grid[0]/grid[0][-1,0])

# Push each actuator
for k in range(0,32):
    for l in range(0,32):
     cmd=cmd*0;
     cmd[k][l] = 1
     dmChannel.set_data(cmd)
     time.sleep(0.2)



##fig,axs=plt.subplots(ncols=1,nrows=2)

