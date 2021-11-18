'''
generate the IM for TTF with the SCC
'''

import os
import sys
import itertools
import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy.signal import welch
import matplotlib.pyplot as plt
import matplotlib as mpl
from glob import glob
import multiprocessing as mp
from functions import * #utility functions to use throughout the simulation

from par_functions import return_vars,propagate,scc,make_IM,make_cov,make_covinvrefj
imagepix,pupilpix,beam_ratio,e,no_phase_offset,xy_dh,grid,N_act,wav0,amp,aperture,indpup,ind_mask_dh,loopx,loopy,freq_loop,pa_loop,n,refdir,iter_arr=return_vars()
immask=np.zeros((aperture.shape))
indmask=np.where(xy_dh<(e+3)*beam_ratio)
immask[indmask]=1. #LOWFS algorithmic mask to improve linearity
phout_diff=np.load('ph_diff.npy')
sccref=scc(propagate(phout_diff))

if glob('covinvcor_lowfs.npy')==['covinvcor_lowfs.npy']:
	zrefarr=np.load('zrefarr.npy')
	covinvcor_lowfs=np.load('covinvcor_lowfs.npy')
else:

	rho,phi=polar_grid(imagepix,pupilpix)
	def zern(n,m,famp):
		'''
		Zernike polynomial whose peak to valley is defined by famp (in metres)
		'''
		z=zernike(n,m,rho,phi)
		pv=np.max(z)-np.min(z)
		return z/pv*famp/wav0*2.*np.pi
	zern_nm=[]
	for n in range(1,2): #tip/tilt, focus
		m=range(-1*n,n+2,2)
		for mm in m:
			zern_nm.append([n,mm])
	zern_nm.append([2,0])
	IMamp=1e-9 #interaction matrix mode amplitude
	def mkrfscc(i):
		'''
		generate images to be used as references for the least squares
		'''
		z=zern(zern_nm[i][0],zern_nm[i][1],IMamp)
		sccim=(scc(propagate(z+phout_diff))-sccref)[indmask]
		return sccim,z,i
	resultsscc=map(mkrfscc,range(len(zern_nm)))
	refarrscc=np.zeros((len(zern_nm),2*sccref[indmask].shape[0]))
	zrefarr=np.zeros((len(zern_nm),aperture[indpup].shape[0]))
	for result in resultsscc:
		sccim,z,i=result
		refarrscc[i]=np.ndarray.flatten(np.array([np.real(sccim),np.imag(sccim)]))
		zrefarr[i]=z[indpup]

	#covariance matrix:
	nl=len(zern_nm)
	covscc=np.zeros((nl,nl))
	for i in range(nl):
		for j in range(i+1):
			if covscc[j,i]==0.:
				covscc[i,j]=sum(refarrscc[i,:]*refarrscc[j,:])
				covscc[j,i]=covscc[i,j]
			
	covinv=np.linalg.pinv(covscc,rcond=1e-3) #svd analysis indicates that this non-linearities remain unchanged below 1e-3
	covinvcor_lowfs=np.dot(covinv,refarrscc) #projection fo reference image vector onto the covariance matrix
	np.save('covinvcor_lowfs.npy',covinvcor_lowfs)
	np.save('zrefarr.npy',zrefarr)