"""
demo code, using perfect SCC, to run closed-loop FAST control
"""

import os
from os import path
import sys
import itertools
import numpy as np
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing as mp
import warnings

from .ao import * #utility functions to use throughout the simulation
from ..utils import joindata, joinsimdata
from ..constants import dmdims, imdims, wav0, beam_ratio

p3i = lambda i: int(round(i)) #python2 to 3: change indicies that are floats to integers

#warnings.warn("DM command -> DM phase has not been implemented.")
N_act = imdims[0] #number of actuators across the pupil

imagepix = imdims[0]
pupilpix = int(round(imagepix/beam_ratio))

grid = np.mgrid[0:imagepix,0:imagepix]
xcen, ycen = imagepix/2, imagepix/2
xy_dh = np.sqrt((grid[1]-imagepix/2.)**2.+(grid[0]-imagepix/2.)**2.)

aperture = np.zeros((imagepix,imagepix))
aperture[np.where(xy_dh<pupilpix/2)] = 1. #unobscured aperture
indpup = np.where(aperture == 1.)

scc_diameter = 1.22*np.sqrt(2.)/32.*pupilpix
left_edge = 1.54*pupilpix-1.22*np.sqrt(2.)/32.*pupilpix/2.
xi_0 = left_edge+scc_diameter/2.
#make scc lyot stop hole in the GPI coronagraph
scc_xpos = imagepix/2. + xi_0
xy = np.sqrt((grid[0]-imagepix/2.)**2. + (grid[1]-scc_xpos)**2.) #radial grid centered on lyot stop hole
ind = np.where(xy < scc_diameter/2.)

lyot_stop = np.zeros(aperture.shape)
lyot_stop[np.where(xy_dh<pupilpix/2.*0.95)] = 1. #slightly undersized pupil
lyot_stop[ind] = 1.
normal_lyot = np.zeros(aperture.shape)
normal_lyot[np.where(xy_dh<pupilpix/2.*0.95)] = 1. #no pinhole

pinhole = np.zeros((imagepix,imagepix))
pinhole[ind] = 1.

#LLOWFS mask
lowfs_mask = np.zeros((imagepix,imagepix))
xcenmask,ycenmask = imagepix/2+xi_0,imagepix/2.
xypup = np.sqrt((grid[0]-ycenmask)**2.+(grid[1]-xcenmask)**2.)
rcut = pupilpix/2
lowfs_mask[np.where(xypup<rcut)] = 1.
lowfs_mask[ind] = 0.
lowfs_mask[np.where(normal_lyot==1.)] = 0.

#adding defocus to LLOWFS Lyot plane
xypup[np.where(xypup>rcut)] = 0.
xypup = xypup/np.max(xypup)
defocus = xypup**2.
defocus = defocus - np.mean(defocus[np.where(xypup > 0.)])
defocus = defocus/(np.max(defocus-np.min(defocus)))
focamp=0.

#make TG coronagraph mask:
e = 3 # IWA array in lambda/D
mfpm = np.zeros((imagepix, imagepix))
edge = e * beam_ratio
indfpm = np.where(xy_dh < edge)
mfpm[indfpm] = 1
tlt_unnorm= grid[1] - xcen
tlt_norm = tlt_unnorm / beam_ratio
ld_tlt_amp = -9.6 #tilt angle to steer off axis pupil into the SCC pinhole
tltfpm = tlt_norm * mfpm * ld_tlt_amp
gamp, sigma = 11.5, 2.03 #optimal TG paramegers to centrate the off-axis pupil light
gini = gamp * np.exp(-xy_dh**2./(2.*(sigma*beam_ratio)**2.))
g = gini - np.mean(gini) #subtract the mean so there is a zero piston offset
tg = tlt_norm * mfpm * ld_tlt_amp + g * mfpm

ind_mask_dh = np.where(
	np.logical_and(
		np.logical_and(
			grid[0]>imagepix/2.-N_act/2.*beam_ratio,
			grid[0]<imagepix/2.+N_act/2.*beam_ratio
		),
		np.logical_and(
			np.logical_and(
				grid[1]>imagepix/2.-N_act/2.*beam_ratio,
				grid[1]<imagepix/2.
			),
			xy_dh > (e+1) * beam_ratio
		)
	)
) #half DH ignoring Fourier modes close to the IWA
DH_mask = np.zeros((imagepix,imagepix))
DH_mask[ind_mask_dh] = 1.
contrast = lambda img: np.nanstd(img[ind_mask_dh])

no_phase_offset = np.zeros((imagepix,imagepix))
no_amp_err = np.ones((imagepix,imagepix))

indap = np.where(xy_dh<pupilpix/2)

def propagate(pupil_phase_dm,pin=True,ph=False,norm=True,m_object=5,t_int=10e-3,Dtel=10,llowfs=False):
	"""
	generate scc image
	"""
	#photon counting information; from http://www.astronomy.ohio-state.edu/~martini/usefuldata.html using f0 = f_lambda*Delta(lambda), accounting for units
	f0 = 93.3 * 0.29 * 10**8. * 0.03/(0.29/1.63) #photons/m**2/s, mag 0 star, 3 % bandpass centered on H band
	flux_object_ini = f0 * 10.**(-m_object/2.5)
	tr_atm, th, qe = 0.9, 0.2, 0.8 #assume transmission through the atmosphere, instrument throughput, quantum efficiency of CCD
	flux_object = flux_object_ini * tr_atm * th * qe
	Nphot = flux_object * t_int * np.pi * (Dtel/2.)**2. 

	pupil_wavefront_dm_unnorm=aperture*np.exp(1j*(pupil_phase_dm)) #initial 
	norm_phot=np.sum(intensity(np.ones((imagepix,imagepix))[np.where(xy_dh<pupilpix/2.)]))
	pupil_wavefront_dm=complex_amplitude(np.sqrt(intensity(pupil_wavefront_dm_unnorm)/norm_phot*Nphot),phase(pupil_wavefront_dm_unnorm))
	if norm: #normalize by summation; use for control
		norm=np.sum(intensity(np.fft.fftshift(pupil_to_image(pupil_wavefront_dm)))) #different definition of contrast, but this should better normalize diffraction limited to aberation-liited cases, where using only the PSF peak intensity would vary between the two but the sum should not
	else: #normalize by peak lyot value; use for plotting contrast
		norm=np.max(intensity(np.fft.fftshift(pupil_to_image(pupil_wavefront_dm*normal_lyot))))
	fpm_wavefront_ini=np.fft.fftshift(pupil_to_image(pupil_wavefront_dm)) #ft to image plane
	fpm_wavefront=complex_amplitude(np.abs(fpm_wavefront_ini),np.angle(fpm_wavefront_ini)+tg) #add phase mask
	lyot_pupil_wavefront=image_to_pupil(fpm_wavefront) #ift to pupil plane

	if not llowfs:
		if not pin:
			masked_lyot_pupil_wavefront = lyot_pupil_wavefront*normal_lyot #add scc lyot stop in pupil plane
		else:
			masked_lyot_pupil_wavefront = lyot_pupil_wavefront*lyot_stop #add scc lyot stop in pupil plane
		imnorm=intensity(pupil_to_image(masked_lyot_pupil_wavefront))
	else:
		imnorm=intensity(pupil_to_image(complex_amplitude(amplitude(lyot_pupil_wavefront*lowfs_mask),np.angle(lyot_pupil_wavefront)+defocus*focamp*1e-9/wav0*2.*np.pi)))
	if not ph:
		im = imnorm / norm
	else:
		im = np.random.poisson(imnorm) / norm	
	return im

rad_Im = round(pupilpix/2) #round(xi_0-rad_main)
Im_ind = np.where(xy<rad_Im) #initial side lobe mtf mask indicies
Im_mask = np.zeros((imagepix,imagepix))
Im_mask[Im_ind] = 1. #side lobe mtf mask

scc = lambda img: image_to_pupil(Im_mask*np.fft.fftshift(pupil_to_image(img+0j))) #isolate MTF side lobe, but don't shift to center, as calibrated in IM; apply grey scale (butterworth) DH mask to control region

amp = 1e-9 / wav0 * 2. * np.pi # amplitude for the sine and cosine reference images; if this is changed, a new array of Fourier modes should be generated when producing the interaction matrix

static_psf = propagate(no_phase_offset)

def funsin(freq,pa):
	'''
	input sine phase screen at a given frequency, pa,

	return I_minus, coronagraphic image, and that pure sine wave

	note that Im_out comes from a difference between the sine wave reference PSF and the target PSF (or stable PSF with no sine/cos during calibration if the target image will change)

	also note that Im_out is modulated by (1) a binary mask around the given sine spot, and (2) a binary mask defining the DH region. the width of (1) is yet to be optimized.
	'''
	sin=rotate(amp*np.sin(2.*np.pi*freq*imagepix/pupilpix*grid[0]/grid[0][-1,0]),pa,reshape=False,mode='wrap')
	im_out=(propagate(sin)-static_psf)
	im_spot_loc=np.abs(scc(im_out)) #this quantity appears to be the best way of finding the spot position, instead of propagate(sin,pin=False), (propagate(sin,pin=False)-propagate(no_phase_offset,pin=False)), or np.abs(scc(propagate(sin)))-np.abs(scc(propagate(no_phase_offset)))
	ygridcen,xgridcen = np.where(im_spot_loc*DH_mask == np.max(im_spot_loc*DH_mask))
	buttgrid=np.sqrt((grid[0]-ygridcen[0])**2.+(grid[1]-xgridcen[0])**2.)
	spot_mask=np.zeros(aperture.shape)
	spot_mask[np.where(buttgrid<beam_ratio/2.*1.5)]=1. #note the tunable factor of currently 1.5, setting how much overlap there is with the adjascent spots
	Im_out=scc(im_out)
	Im_out=complex_amplitude(np.abs(Im_out)*spot_mask,np.angle(Im_out)*spot_mask)
	return Im_out,im_out,sin

def funcos(freq,pa):
	'''
	the same for cosine waves
	'''
	cos=rotate(amp*np.cos(2.*np.pi*freq*imagepix/pupilpix*grid[0]/grid[0][-1,0]),pa,reshape=False,mode='wrap')
	im_out=(propagate(cos)-static_psf)
	im_spot_loc=np.abs(scc(im_out))
	ygridcen,xgridcen=np.where(im_spot_loc*DH_mask==np.max(im_spot_loc*DH_mask))
	buttgrid=np.sqrt((grid[0]-ygridcen[0])**2.+(grid[1]-xgridcen[0])**2.)
	spot_mask=np.zeros(aperture.shape)
	spot_mask[np.where(buttgrid<beam_ratio/2.*1.5)]=1.
	Im_out=scc(im_out)
	Im_out=complex_amplitude(np.abs(Im_out)*spot_mask,np.angle(Im_out)*spot_mask)

	return Im_out,im_out,cos


#CREATE IMAGE X,Y INDEXING THAT PLACES SINE, COSINE WAVES AT EACH LAMBDA/D REGION WITHIN THE NYQUIST REGION
#the nyquist limit from the on-axis psf is +/- N/2 lambda/D away, so the whole nyquist region is N lambda/D by N lambda/D, centered on the on-axis PSF
#to make things easier and symmetric, I want to place each PSF on a grid that is 1/2 lambda/D offset from the on-axis PSF position; this indexing should be PSF copy center placement location
allx,ally=list(zip(*itertools.product(np.linspace(p3i(imagepix/2-N_act/2.*beam_ratio+0.5*beam_ratio),p3i(imagepix/2.-0.5*beam_ratio),p3i(N_act/2.)),np.linspace(p3i(imagepix/2.-N_act/2.*beam_ratio+0.5*beam_ratio),p3i(imagepix/2.+N_act/2.*beam_ratio-0.5*beam_ratio),p3i(N_act)))))
loopx,loopy=np.array(list(allx)).astype(float),np.array(list(ally)).astype(float)

freq_loop=np.sqrt((loopy-imagepix/2.)**2.+(loopx-imagepix/2.)**2.)/beam_ratio #sine wave frequency for (lambda/D)**2 region w/in DH
pa_loop=90.-180./np.pi*np.arctan2(loopy-imagepix/2.,loopx-imagepix/2.) #position angle of sine wave for (lambda/D)**2 region w/in DH

n = 2*len(freq_loop)
iter_arr = list(range(len(pa_loop)))
refdir = "sin_amp_" + str(int(round(amp/1e-9*wav0/(2.*np.pi)))) + "_tg"
joinrefdir = lambda f: joinsimdata(path.join(refdir, f))

def return_vars():
	return imagepix,pupilpix,beam_ratio,e,no_phase_offset,xy_dh,grid,N_act,wav0,amp,aperture,indpup,ind_mask_dh,loopx,loopy,freq_loop,pa_loop,n,refdir,iter_arr

def make_IM(i):
	if freq_loop[i]<(e+1) or freq_loop[i]>N_act/2.*beam_ratio: #masking sine waves that fall within 1 lambda/D of the PSF core to prevent contamination/improve linearity of the SCC, and those that are in the outer corners of the square dark hole to minize noise amplification in closed-loop (fringe visibility is too low in this region)
		Im_cos,im_cos,cos=funcos(freq_loop[i],pa_loop[i])
		Im_sin,im_sin,sin=funsin(freq_loop[i],pa_loop[i])
		imdum=np.zeros((imagepix,imagepix))
		np.save(joinrefdir(str(i)), np.ndarray.flatten(np.array([np.real(imdum[ind_mask_dh]),np.imag(imdum[ind_mask_dh])])))
		np.save(joinrefdir(str(i+len(freq_loop))), np.ndarray.flatten(np.array([np.real(imdum[ind_mask_dh]),np.imag(imdum[ind_mask_dh])])))
	else:
		Im_cos,im_cos,cos = funcos(freq_loop[i],pa_loop[i])
		Im_sin,im_sin,sin = funsin(freq_loop[i],pa_loop[i])
		np.save(joinrefdir(str(i)), np.ndarray.flatten(np.array([np.real(Im_cos[ind_mask_dh]),np.imag(Im_cos[ind_mask_dh])])))
		np.save(joinrefdir(str(i+len(freq_loop))), np.ndarray.flatten(np.array([np.real(Im_sin[ind_mask_dh]),np.imag(Im_sin[ind_mask_dh])])))
		print(float(i)/(len(pa_loop)-1))
	return cos[indpup],sin[indpup],i #IMPORTANT: save fourier modes in pupil indicies! otherwise arrays take up ~5x more space...

def make_cov(i):
	refi = np.load(joinrefdir(str(i)+'.npy'))
	covi = []
	for j in list(range(i+1)): #these are the indicies needed for a symmetric matrix, the other half is the same: i,j = j,i
		refj =  np.load(joinrefdir(str(j)+'.npy'))
		covi.append(sum(refi*refj))
		print(float(i*(n-1)+j)/(n**2.)) #this would only be a full counter if I didn't use symmetry to fill in the matrix, so this is only a rough ~order of magnitude counter	
	return covi,i

def make_covinvrefj(i):
	covinv = np.load(joinsimdata('covinv_'+refdir+'.npy'))
	covinvcori = np.zeros(2*aperture[ind_mask_dh].shape[0])
	for j in list(range(n)):
		refj = np.load(joinrefdir(str(j)+'.npy'))
		covinvcori=covinvcori+covinv[i,j]*refj

	return i, covinvcori
