from ancillary_code import *
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from numpy import float32
import time
import functions
import sys
from sealrtc import plot_linearity, optics

dmc2wf=np.load('recent_data/lodmc2wfe.npy') #alpao actuator dmc command units to microns WFE units
optics.applybestflat()

texp=1e-3
expt(texp) #set exposure time; for current light source config at 100 Hz
imydim,imxdim=optics.imdims
tsleep=0.02 #should be the same values from align_fpm_lodmc.py

#DM aperture;
aperture = optics.aperture
indap = optics.indap
tip, tilt = optics.tip, optics.tilt

#setup Zernike polynomials
nmarr=[]
norder=3 #how many radial Zernike orders to look at
for n in range(1,norder):
	for m in range(-n,n+1,2):
		nmarr.append([n,m])

rho, phi = optics.rho, optics.phi
imxcen,imycen=np.load('recent_data/imcen.npy')
beam_ratio=np.load('recent_data/beam_ratio.npy')

gridim=np.mgrid[0:imydim,0:imxdim]
rim=np.sqrt((gridim[0]-imycen)**2+(gridim[1]-imxcen)**2)

ttmask=np.zeros(imini.shape)
rmask=10
indttmask=np.where(rim/beam_ratio<rmask)
ttmask[indttmask]=1
IMamp=0.001

#make interaction matrix
def genIM():
	refvec=np.zeros((len(nmarr),ttmask[indttmask].shape[0]*2))
	zernarr=np.zeros((len(nmarr),aperture[indap].shape[0]))
	for i in range(len(nmarr)):
		n,m=nmarr[i]
		zern=optics.funz(n,m,IMamp)
		time.sleep(tsleep)
		imzern=stack(10)
		optics.applybestflat()
		time.sleep(tsleep)
		imflat=stack(10)
		imdiff=(imzern-imflat)
		Im_diff=processim(imdiff)
		refvec[i]=np.array([np.real(Im_diff[indttmask]),np.imag(Im_diff[indttmask])]).flatten()
		zernarr[i]=zern[indap]

	IM=np.dot(refvec,refvec.T) #interaction matrix
	return IM,refvec


rcond=1e-5 

def genCM(rcond=rcond):
	IM,refvec=genIM()
	IMinv=np.linalg.pinv(IM,rcond=rcond)
	cmd_mtx=np.dot(IMinv,refvec)
	return cmd_mtx

cmd_mtx=genCM()

optics.applybestflat()
time.sleep(tsleep)
imflat=stack(100)


#LINEARITY CHARACTERIZATION

def genzerncoeffs(i,zernamp):
	'''
	i: zernike mode
	zernamp: Zernike amplitude in DM units to apply
	'''
	n,m=nmarr[i]
	zern=optics.funz(n,m,zernamp)
	time.sleep(tsleep)
	imzern=stack(10)
	imdiff=(imzern-imflat)
	tar_ini=processim(imdiff)
	tar = np.array([np.real(tar_ini[indttmask]),np.imag(tar_ini[indttmask])]).flatten()	
	coeffs=np.dot(cmd_mtx, tar)
	return coeffs*IMamp

nlin = 20 #number of data points to scan through linearity measurements
zernamparr = np.linspace(-1.5*0.005,1.5*0.005,nlin)

#try linearity measurement for Zernike mode 0
zernampout=np.zeros((len(nmarr),len(nmarr),nlin))
for nm in range(len(nmarr)):
	for i in range(nlin):
		zernamp=zernamparr[i]
		coeffsout=genzerncoeffs(nm,zernamp)
		zernampout[nm,:,i]=coeffsout

optics.applybestflat()

plot_linearity(zernamparr, zernampout, dmc2wf, rcond=rcond)
plt.show()