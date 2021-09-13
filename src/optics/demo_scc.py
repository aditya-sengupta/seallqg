'''
demo code, using SCC as both a HOWFS and LOWFS, to run closed-loop FAST control
'''

import os
import sys
import itertools
import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy.signal import welch
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing as mp

from .ao import * #utility functions to use throughout the simulation
from .par_functions import return_vars,propagate,scc,make_IM,make_cov,make_covinvrefj
from ..utils import joinsimdata, joinplot

imagepix,pupilpix,beam_ratio,e,no_phase_offset,xy_dh,grid,N_act,wav0,amp,aperture,indpup,ind_mask_dh,loopx,loopy,freq_loop,pa_loop,n,refdir,iter_arr=return_vars()

#high order SCC command matrix
import calibrate_scc
covinvcor = np.load(joinsimdata('covinvcor_'+refdir+'.npy'))

#functions to generate coefficients and DM commands
i_arr=list(range(n))
def lsq_get_coeffs(im_in):
	'''
	obtain the least squares coefficients
	'''
    
	Im_tar=scc(im_in)
	tar=np.ndarray.flatten(np.array([np.real(Im_tar[ind_mask_dh]),np.imag(Im_tar[ind_mask_dh])]))

	coeffs=np.zeros((n))
	for i in i_arr:
		coeffs[i]=sum(covinvcor[i]*tar)
	
	return coeffs

fourierarrfile='fourierarr_pupilpix_'+str(int(pupilpix))+'_N_act_'+str(int(N_act))+'_sin_amp_'+str(int(round(amp/1e-9*wav0/(2.*np.pi))))+'.npy'
fourierarr=np.load(fourierarrfile)

def corr(im_in,g=np.ones(len(freq_loop)*2)):
	'''
	apply DM correction
	'''
	coeffss=lsq_get_coeffs(im_in) #DM coefficients are computed on the recorded target image
	#return only DM commands:
	lsq_phase=np.zeros(aperture.shape)
	lsq_phase[indpup]=np.dot(fourierarr.T,-1.*coeffss*g.flatten())
	outt=propagate(lsq_phase)
	return lsq_phase,outt

#iterating on diffraction to improve the coronagraph's raw contrast
if path.isfile(joinsimdata("ph_diff.npy")):
	phout_diff=np.load('ph_diff.npy')
else:
	c10ld = lambda im: np.std(im[np.where(np.logical_and(np.logical_and(xy_dh<10.5*beam_ratio,xy_dh>9.5*beam_ratio),grid[1]<imagepix/2.-4.*beam_ratio))]) #indicies cover an annulus around 10 lambda/D and padded within the half DH

	def corr_in(phin):
		'''
		correct diffraction-limited input phase, assuming an integrator unity gain
		'''
		imin=propagate(phin)
		phlsq,imoutlsq=corr(imin)
		phout=phin+phlsq
		return phout

	def iter_dl(niter):
		'''
		iterate on a diffraction-limited input, producing a half dark hole on diffraction
		'''
		phin=no_phase_offset
		cv=np.zeros(niter)
		for it in range(niter):
			phout=corr_in(phin)	
			phin=phout
			cv[it]=c10ld(propagate(phout,pin=False))
			print(it)
		return cv,phout
	cv, phout_diff = iter_dl(20)
	np.save(joinsimdata('ph_diff.npy'),phout_diff)

#calibrate low order modes
from calibrate_scc_lowfs import sccref,indmask,immask,zrefarr,covinvcor_lowfs

def lowfs_get_coeffs(imin):
		sccim=(scc(imin)-sccref)[indmask]
		imflat=np.ndarray.flatten(np.array([np.real(sccim),np.imag(sccim)]))
		coeffs=np.dot(covinvcor_lowfs,imflat)
		return coeffs

def lowfsout(imin,g=np.ones(zrefarr.shape[0])):
		coeffs=lowfs_get_coeffs(imin)
		phlsq=np.zeros(aperture.shape)
		phlsq[indpup]=np.dot(-coeffs*g.flatten(),zrefarr)
		return phlsq

#close the loop on a residual AO phase screen
m_object=5 #guide star magnitude at the given observing wavelength
t_int=10e-3 #WFS integration time
t_stamp=1e-3 #time stamp to use to integrate over t_int to simulate realization averaging over an exposure; should be smaller than and an integer multiple of t_int
iterim=int(round(t_int/t_stamp))
Tint=5. #total integration time to use to measure the open loop SCC coefficients, in seconds; should also be an integer multiple of t_int; should be long enough properly sample the temporal statistics, covering both signal (low freq) and noise (high freq); but, for pure frozen flow it can't be more than beam_ratio pupil crossings, afterwhich the same phase screen just repeats again
niter=int(round(Tint/t_int)) #how many closed-loop iterations to run
t_lag=0.1e-3 #assumed servo lag in between ending WFS exposure and sending DM commands; only for gain calculation purposes
Dtel=10 #telescope diameter, m
vx=10 #wind speed, m/s
nmrms=100e-9 #level of AO residuals, normalized over 0 to 16 cycles/pupil, m rms

def genim(phatmin,phdm):
	'''
	simulate SCC exposure with translating atmospheric phase screen over the exposure
	'''
	imin=np.zeros((imagepix,imagepix))
	for iterint in list(range(iterim)):
		phatmin=translate_atm(phatmin,beam_ratio,t_stamp,Dtel,vx=vx)
		imin=imin+propagate(phatmin+phdm,ph=True,m_object=m_object,t_int=t_stamp,Dtel=Dtel)
	imin=imin/iterim
	return imin,phatmin

if path.isfile(joinsimdata("gopt.npy")):
	gopt = np.load('gopt.npy')
	gopt_lowfs=np.load('gopt_lowfs.npy')
else:
	phatm=make_noise_pl(nmrms,imagepix,pupilpix,wav0,-2)
	phatmin=phatm
	coeffsolarr=np.zeros((2*len(freq_loop),niter))
	coeffslowfsolarr=np.zeros((zrefarr.shape[0],niter))
	for i in list(range(niter)):

		imin,phatmin=genim(phatmin,phout_diff) #open loop: images are not used to generate DM commands, just to generate coefficients that are processed off-line to determine the optimal gains

		print('open-loop iteration: {0} of {1}'.format(i, niter))
		coeffsolarr[:,i]=lsq_get_coeffs(imin)
		coeffslowfsolarr[:,i]=lowfs_get_coeffs(imin)

	optgim=np.zeros((imagepix,imagepix))
	goptarr=np.zeros(len(freq_loop))
	numoptg=100
	for j in range(len(freq_loop)):
		#j=0
		coefftseries=detrend(np.sqrt(coeffsolarr[j,:]**2.+coeffsolarr[j+len(freq_loop),:]**2.))
		freqol,psdol=welch(coefftseries,fs=1./t_int)
		freqol,psdol=freqol[1:],psdol[1:] #remove DC component (freq=0 Hz)

		optg = lambda g: np.trapz(psdol*np.abs(Hrej(freqol,t_int,t_lag,g))**2.,freqol) #function to optimize the gain for the integral of the OL PSD times the square modulus of rejection transfer function as a function of the gain

		optgarr=np.zeros(numoptg)
		garr=np.linspace(0.,1.5,numoptg)
		for gg in range(numoptg):
			g=garr[gg]
			optgarr[gg]=optg(g)
		gopt=garr[np.where(optgarr==np.min(optgarr))[0]][0]
		optgim[p3i(loopy[j]-beam_ratio/2.):p3i(loopy[j]+beam_ratio/2.),p3i(loopx[j]-beam_ratio/2.):p3i(loopx[j]+beam_ratio/2.)]=gopt
		goptarr[j]=gopt
	goptlowfsarr=np.zeros(zrefarr.shape[0])
	for j in range(zrefarr.shape[0]):
		coefftseries_lowfs=detrend(coeffslowfsolarr[j,:])
		freqol_lowfs,psdol_lowfs=welch(coefftseries_lowfs,fs=1./t_int)
		freqol_lowfs,psdol_lowfs=freqol_lowfs[1:],psdol_lowfs[1:]
		optg_lowfs = lambda g: np.trapz(psdol_lowfs*np.abs(Hrej(freqol_lowfs,t_int,t_lag,g))**2.,freqol_lowfs)
		optglowfsarr=np.zeros(numoptg)
		garr=np.linspace(0.,1.5,numoptg)
		for gg in range(numoptg):
			g=garr[gg]
			optglowfsarr[gg]=optg_lowfs(g)
		gopt=garr[np.where(optglowfsarr==np.min(optglowfsarr))[0]][0]
		goptlowfsarr[j]=gopt

	gopt=np.vstack(np.append(goptarr,goptarr)) #assumes the same gain for the sine and cosine coefficients
	np.save(joinsimdata('ol_coeffs.npy'), coeffsolarr)
	np.save(joinsimdata('ol_coeffs_lowfs.npy'),coeffslowfsolarr)
	np.save(joinsimdata('gopt.npy'),gopt)
	np.save(joinsimdata('gopt_lowfs.npy'),goptlowfsarr)
	np.save(joinsimdata('goptim.npy'),optgim)

	gopt=np.load('gopt.npy')
	gopt_lowfs=np.load('gopt_lowfs.npy')

	'''
	from plot_opt_gain import plot_opt_gain
	plot_opt_gain(coeffsolarr,optgim,100,403,t_int,t_lag,freq_loop,imagepix,N_act,beam_ratio)
	from plot_opt_gain_ttf import plot_opt_gain_ttf
	plot_opt_gain_ttf(coeffslowfsolarr,t_int,t_lag,freq_loop,imagepix,N_act,beam_ratio)
	'''


#close the loop!
phatm=make_noise_pl(nmrms,imagepix,pupilpix,wav0,-2)
phatmin=phatm
phdm=phout_diff
#phdm=no_phase_offset
imin,phatmin=genim(phatmin, phout_diff) #first frame
np.save(joinsimdata('phdm.npy'), phdm)
np.save(joinsimdata('phatmin.npy'), phatmin)
np.save(joinsimdata('imin.npy'), imin)

#manually set DH corners equal to zero gain; not sure why there are a non-zero gain as I am setting these modes to zero in the IM...
ind_corner=np.where(freq_loop>N_act/2.)[0]
gopt[ind_corner,:],gopt[ind_corner+len(freq_loop),:]=0.,0.

vpup=lambda im:(aperture*im)[p3i(imagepix/2-pupilpix/2):p3i(imagepix/2+pupilpix/2),p3i(imagepix/2-pupilpix/2):p3i(imagepix/2+pupilpix/2)]
vim=lambda im: im[p3i(imagepix/2-N_act/2*beam_ratio):p3i(imagepix/2+N_act/2*beam_ratio),p3i(imagepix/2-N_act/2*beam_ratio):p3i(imagepix/2+N_act/2*beam_ratio)]

size=20
font = {'family' : 'Times New Roman',
        'size'   : size}

mpl.rc('font', **font)
mpl.rcParams['image.interpolation'] = 'nearest'

from matplotlib import animation

fig,axs=plt.subplots(ncols=2,nrows=2,figsize=(10,10))
[ax.axis('off') for ax in axs.flatten()]
[axs.flatten()[i].set_title(['OL phase','CL phase','OL SCC image','CL SCC image'][i],size=size) for i in list(range(4))]
im1=axs[0,0].imshow(vpup(phatm),vmin=-1,vmax=1)
im2=axs[0,1].imshow(vpup(phatm),vmin=-1,vmax=1)
im3=axs[1,0].imshow(vim(imin),vmin=0,vmax=1e-5)
im4=axs[1,1].imshow(vim(imin),vmin=0,vmax=1e-5)
fig.suptitle('t=0.000s, OL',y=0.1,x=0.51)

Tint_cl=1 #number of seconds to run the closed-loop simulation
num_time_steps=int(Tint_cl/t_int)
time_steps = np.arange(num_time_steps)

def animate(it):
	phdm=np.load('phdm.npy')
	phatmin=np.load('phatmin.npy')
	imin=np.load('imin.npy')
	
	#note that there is no servo lag simulated here; DM commands are applied to the atmospheric realization starting at the end of the previous exposure (so this is at least simulating half a frame lag). at 100 Hz, even a 1 ms lag is only one tenth of a frame delay, effectively negligible

	if it*t_int<=0.25: #open loop
		phlsq,imoutlsq=corr(imin,np.zeros(gopt.shape))
		phlowfs=lowfsout(imin,np.zeros(gopt_lowfs.shape))
		fig.suptitle('t='+str(round(t_int*it,4))+'s, OL',y=0.1,x=0.51)
	elif it*t_int<=0.5: #TTF loop closed
		phlsq,imoutlsq=corr(imin,np.zeros(gopt.shape))
		phlowfs=lowfsout(imin,gopt_lowfs)
		fig.suptitle('t='+str(round(t_int*it,4))+'s, CL TTF',y=0.1,x=0.51)
	else: #all loops closed
		phlsq,imoutlsq=corr(imin,gopt)
		phlowfs=lowfsout(imin,gopt_lowfs)
		fig.suptitle('t='+str(round(t_int*it,4))+'s, CL TTF+HO',y=0.1,x=0.51)
	phdm=phdm+phlsq+phlowfs

	imout_ol,phatmout=genim(phatmin,phout_diff)
	imout,phatmout=genim(phatmin,phdm)
	
	np.save(joinsimdata('phdm.npy'),phdm)
	np.save(joinsimdata('phatmin.npy'),phatmout)
	np.save(joinsimdata('imin.npy'),imout)

	im1.set_data(vpup(phatmout))
	im2.set_data(vpup(phatmout+phdm-phout_diff))
	im3.set_data(vim(imout_ol))
	im4.set_data(vim(imout))
	print('closed-loop: iteration {0} of {1}'.format(it, num_time_steps))
	return [im1,im2,im3,im4]

ani = animation.FuncAnimation(fig, animate, time_steps, interval=50, blit=True)
ani.save(joinplot('fast_cl_demo.gif'),writer='imagemagick')
plt.close(fig)