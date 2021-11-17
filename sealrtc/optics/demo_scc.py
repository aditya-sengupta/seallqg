"""
demo code, using SCC as both a HOWFS and LOWFS, to run closed-loop FAST control
"""
import os
from os import path
import sys
import itertools
import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy.signal import welch
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing as mp
import tqdm

from .ao import * #utility functions to use throughout the simulation
from .calibrate_scc import make_im_scc_howfs
from .par_functions import return_vars,propagate,scc,make_IM,make_cov,make_covinvrefj
from .calibrate_scc_lowfs import sccref,indmask,immask,zrefarr,covinvcor_lowfs
from ..utils import joinsimdata, joinplot

imagepix,pupilpix,beam_ratio,e,no_phase_offset,xy_dh,grid,N_act,wav0,amp,aperture,indpup,ind_mask_dh,loopx,loopy,freq_loop,pa_loop,n,refdir,iter_arr=return_vars()

#high order SCC command matrix
covinvcor_path = joinsimdata('covinvcor_' + refdir + '.npy')
if path.isfile(covinvcor_path):
	covinvcor = np.load(covinvcor_path)
else:
	covinvcor = make_im_scc_howfs()

#functions to generate coefficients and DM commands
i_arr = np.arange(n)
def lsq_get_coeffs(im_in):
	'''
	obtain the least squares coefficients
	'''
	Im_tar = scc(im_in)
	tar = np.ndarray.flatten(np.array([np.real(Im_tar[ind_mask_dh]), np.imag(Im_tar[ind_mask_dh])]))
	return np.matmul(covinvcor, tar)

fourierarr_path = joinsimdata('fourierarr_pupilpix_'+str(int(pupilpix))+'_N_act_'+str(int(N_act))+'_sin_amp_'+str(int(round(amp/1e-9*wav0/(2.*np.pi))))+'.npy')
fourierarr = np.load(fourierarr_path)

def corr(im_in,g=np.ones(len(freq_loop)*2)):
	'''
	apply DM correction
	'''
	coeffss=lsq_get_coeffs(im_in) #DM coefficients are computed on the recorded target image
	#return only DM commands:
	lsq_phase=np.zeros(aperture.shape)
	lsq_phase[indpup]=np.dot(fourierarr.T,-1.*coeffss*g.flatten())
	outt=propagate(lsq_phase)
	return lsq_phase, outt

#iterating on diffraction to improve the coronagraph's raw contrast
if path.isfile(joinsimdata("ph_diff.npy")):
	phout_diff = np.load(joinsimdata("ph_diff.npy"))
else:
	c10ld = lambda im: np.std(
		im[np.where(
			np.logical_and(
				np.logical_and(
					xy_dh < 10.5 * beam_ratio,
					xy_dh > 9.5 * beam_ratio
				),
				grid[1] < imagepix/2. - 4. * beam_ratio)
		)]
	)

	def corr_in(phin):
		"""
		correct diffraction-limited input phase, assuming an integrator unity gain
		"""
		imin = propagate(phin)
		phlsq, imoutlsq = corr(imin)
		phout = phin + phlsq
		return phout

	def iter_dl(niter):
		'''
		iterate on a diffraction-limited input, producing a half dark hole on diffraction
		'''
		phin = no_phase_offset
		cv = np.zeros(niter)
		for it in tqdm.trange(niter):
			phout = corr_in(phin)	
			phin = phout
			cv[it] = c10ld(propagate(phout, pin=False))

		return cv, phout

	cv, phout_diff = iter_dl(20)
	np.save(joinsimdata("ph_diff.npy"), phout_diff)

def lowfs_get_coeffs(imin):
	sccim = (scc(imin)-sccref)[indmask]
	imflat=np.ndarray.flatten(np.array([np.real(sccim),np.imag(sccim)]))
	coeffs=np.dot(covinvcor_lowfs,imflat)
	return coeffs

def lowfsout(imin,g=np.ones(zrefarr.shape[0])):
	coeffs = lowfs_get_coeffs(imin)
	phlsq=np.zeros(aperture.shape)
	phlsq[indpup]=np.dot(-(coeffs * g.flatten()),zrefarr)
	return phlsq

# close the loop on a residual AO phase screen
m_object = 5 # guide star magnitude at the given observing wavelength
t_int = 10e-3 # WFS integration time
t_stamp = 1e-3 # time stamp to use to integrate over t_int to simulate realization averaging over an exposure; should be smaller than and an integer multiple of t_int
iterim = int(round(t_int/t_stamp))
Tint = 5. #total integration time to use to measure the open loop SCC coefficients, in seconds; should also be an integer multiple of t_int; should be long enough properly sample the temporal statistics, covering both signal (low freq) and noise (high freq); but, for pure frozen flow it can't be more than beam_ratio pupil crossings, afterwhich the same phase screen just repeats again
niter = int(round(Tint/t_int)) #how many closed-loop iterations to run
t_lag = 0.1e-3 #assumed servo lag in between ending WFS exposure and sending DM commands; only for gain calculation purposes
Dtel = 10 #telescope diameter, m
vx = 10 #wind speed, m/s
nmrms = 100e-9 #level of AO residuals, normalized over 0 to 16 cycles/pupil, m rms

def genim(phatmin, phdm):
	'''
	simulate SCC exposure with translating atmospheric phase screen over the exposure
	'''
	imin=np.zeros((imagepix,imagepix))
	for iterint in range(iterim):
		phatmin = translate_atm(phatmin,beam_ratio,t_stamp,Dtel,vx=vx)
		imin = imin+propagate(phatmin+phdm,ph=True,m_object=m_object,t_int=t_stamp,Dtel=Dtel)

	imin=imin / iterim
	return imin,phatmin
