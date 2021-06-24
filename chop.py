'''
ancillary functions for the optical chopper
'''

import numpy as np
import time
import functions
import itertools
from ancillary_code import *

#initial setup: apply best flat, generate DM grid to apply future shapes
dmcini=getdmc()
ydim,xdim=dmcini.shape
grid=np.mgrid[0:ydim,0:xdim].astype(dmcini.dtype)
#bestflat=np.load('bestflat.npy') #for starting from flat DM with FPM aligned
#bestflat=np.load('dmc_dh.npy') #for bootstrapping
#applydmc(bestflat)

ygrid,xgrid=grid[0]-ydim/2,grid[1]-xdim/2
xy=np.sqrt(ygrid**2+xgrid**2)

texp=1e-4
expt(texp) #set exposure time; for source intensity of 0.24 mW; note that at a chopper frame rate of 100 Hz, 1e-4 means a duty cycle of 

imini=getim()

#make MTF side lobe mask
xsidemaskcen,ysidemaskcen=252.01,159.4 #x and y location of the side lobe mask in the cropped image
sidemaskrad=26.8 #radius of the side lobe mask
mtfgrid=np.mgrid[0:imini.shape[0],0:imini.shape[1]]
sidemaskrho=np.sqrt((mtfgrid[0]-ysidemaskcen)**2+(mtfgrid[1]-xsidemaskcen)**2)
sidemask=np.zeros(imini.shape)
sidemaskind=np.where(sidemaskrho<sidemaskrad)
sidemask[sidemaskind]=1

#central lobe MTF mask
yimcen,ximcen=imini.shape[0]/2,imini.shape[1]/2
cenmaskrad=49
cenmaskrho=np.sqrt((mtfgrid[0]-yimcen)**2+(mtfgrid[1]-ximcen)**2)
cenmask=np.zeros(imini.shape)
cenmaskind=np.where(cenmaskrho<cenmaskrad)
cenmask[cenmaskind]=1

#pinhole MTF mask
pinmaskrad=4
pinmask=np.zeros(imini.shape)
pinmaskind=np.where(cenmaskrho<pinmaskrad)
pinmask[pinmaskind]=1

def processim(imin,mask=sidemask): #process SCC image, isolating the sidelobe in the FFT and IFFT back to the image
	otf=np.fft.fftshift(np.fft.fft2(imin,norm='ortho')) #(1) FFT the image
	otf_masked=otf*mask #(2) multiply by binary mask to isolate side lobe
	Iminus=np.fft.ifft2(otf_masked,norm='ortho') #(3) IFFT back to the image plane, now generating a complex-valued image
	return Iminus

def getchop(nim): #get some number of frames of the chopper sequence, before checking for synchronization 
	imarr=np.zeros((nim+10,imini.shape[0],imini.shape[1]))
	for i in range(nim+10):
		imarr[i]=getim() #cannot use stack or add sleep)
	imarr=imarr[10:] #throw out the first 10 frames (not synchronized yet)
	return imarr

def gennorm(): #get normalization factor for SCC images; dependent on exposure time, frame rate, etc.
	nnorm=800 #number of frames to get for the normalization image; at the moment timed to cover the full sequence betore the chopper drift occurs (running Andor frames at 100 hz)
	normim=getchop(nnorm)
	normarr=np.array([])
	for i in range(0,nnorm,2):
		mtfim=mtf(normim[i]-normim[i+1])
		normarr=np.append(normarr,np.sum(mtfim[pinmaskind]))
	norm=np.median(normarr)
	return norm
'''
norm1em4=gennorm()
np.save('chopnorm1em4.npy',norm1em4)
'''
norm1em4=np.load('chopnorm1em4.npy')


def stackchop(nim=10,norm=norm1em4,stack=True,rf=False,ruf=False,rfmuf=False): 
	'''
	function to stack images with the chopper running

	inputs:
	- nim: number of images to stack, returning output fringed, unfringed, and fringed minus unfringed images
	- stack: boolean variable to indicate whether or not the sequence should be stacked. if True, median collapse, if flase, return the full unstacked sequence
	- rf, ruf, and rfmuf: bolean variables indicating if the function should only return fringed, unfringed, or fringed minus unfringed, respectively

	outputs:
	fringed, unfringed, and/or fringed minus unfringed images as configured by the inputs  

	'''

	imarr=getchop(nim)

	outimarr={'f':np.array([]),'uf':np.array([]),'fmuf':np.array([])} #empty arrays to append fringed (f), unfringed (uf), and fringed - unfringed (fmuf) images that are properly sorted in phase and not cropped by fringe visibility

	for i in range(0,imarr.shape[0]-1,2):
		im1=imarr[i]/norm
		im2=imarr[i+1]/norm
		diffmtf=mtf(im1)-mtf(im2)
		mtfdiff=mtf(im1-im2)

		#next, figure out if the usable frame is fringed or unfringed
		if np.sum(diffmtf[pinmaskind])/np.sum(mtfdiff[pinmaskind])>0: #fringed frame first
			outimarr['f']=np.append(outimarr['f'],im1)
			outimarr['uf']=np.append(outimarr['uf'],im2)
			outimarr['fmuf']=np.append(outimarr['fmuf'],im1-im2)
		if np.sum(diffmtf[pinmaskind])/np.sum(mtfdiff[pinmaskind])<0: #unfringed frame first
			outimarr['f']=np.append(outimarr['f'],im2)
			outimarr['uf']=np.append(outimarr['uf'],im1)
			outimarr['fmuf']=np.append(outimarr['fmuf'],im2-im1)

	cubesize=int(outimarr['f'].shape[0]/imini.shape[0]**2)
	for name in ['f','uf','fmuf']:
		outimarr[name]=outimarr[name].reshape(cubesize,im1.shape[0],im1.shape[1])
	if stack==True:
		imf,imuf,imfmuf=np.nanmedian(outimarr['f'],axis=0),np.nanmedian(outimarr['uf'],axis=0),np.nanmedian(outimarr['fmuf'],axis=0) 
	else:
		imf,imuf,imfmuf=outimarr['f'],outimarr['uf'],outimarr['fmuf']
	if rf==True: 
		return imf/norm
	elif ruf==True:
		return imuf/norm
	elif rfmuf==True:
		return imfmuf/norm
	else:
		return imf/norm,imuf/norm,imfmuf/norm



#old function to stack chopped images with the chopper slowly "drifting"

'''
def stackchop(nim=10,norm=norm1em4,stack=True,rf=False,ruf=False,rfmuf=False): 

	imarr=getchop(nim)

	outimarr={'f':np.array([]),'uf':np.array([]),'fmuf':np.array([])} #empty arrays to append fringed (f), unfringed (uf), and fringed - unfringed (fmuf) images that are properly sorted in phase and not cropped by fringe visibility
	pvsarr=np.array([])
	for i in range(0,imarr.shape[0]-1,2):
		im1=imarr[i]
		im2=imarr[i+1]
		diffmtf=mtf(im1)-mtf(im2)
		mtfdiff=mtf(im1-im2)
		#fvs=np.sum(mtfdiff[sidemaskind])/np.sum(mtf(im1)[cenmaskind]) #mtf-based fringe visibility
		pvs=np.sum(diffmtf[pinmaskind])/np.sum(mtfdiff[pinmaskind]) #mtf-based fringe visibility, using pinhole PSF
		#print(round(pvs,3))
		#fringe visibility cut: when less than this value, this is when the chopper partially blocking the pinhole and not usable; requires that the FPM remained sufficiently aligned to work;
		#if round(fvs,3)<0.089: #NOTE: THIS NUMBER CHANGES REGULARLY STILL, DEPENDING ON DAY TO DAY FPM MISALIGNMENT; MIGHT NEED A BETTER METRIC?
		pvsarr=np.append(pvsarr,pvs)
	#print((np.max(np.abs(pvsarr))-np.min(np.abs(pvsarr)))/np.max(np.abs(pvsarr)))
	if (np.max(np.abs(pvsarr))-np.min(np.abs(pvsarr)))/np.max(np.abs(pvsarr))>0.2: #if the fractional variation of pinhole PSF visibility over the sequence is too large (meaning images are acquired over a transition region), take another sequence (which won't be over a transition region)
		print('chopper transition, re-acquiring...')
		time.sleep(2) #the synchronization will happen after waiting ~2 seconds... might need to change this into a while loop to be more robust
		imarr=getchop(nim)

	for i in range(0,imarr.shape[0]-1,2):
		im1=imarr[i]/norm
		im2=imarr[i+1]/norm
		diffmtf=mtf(im1)-mtf(im2)
		mtfdiff=mtf(im1-im2)

		#next, figure out if the usable frame is fringed or unfringed
		if np.sum(diffmtf[pinmaskind])/np.sum(mtfdiff[pinmaskind])>0: #fringed frame first
			outimarr['f']=np.append(outimarr['f'],im1)
			outimarr['uf']=np.append(outimarr['uf'],im2)
			outimarr['fmuf']=np.append(outimarr['fmuf'],im1-im2)
		if np.sum(diffmtf[pinmaskind])/np.sum(mtfdiff[pinmaskind])<0: #unfringed frame first
			outimarr['f']=np.append(outimarr['f'],im2)
			outimarr['uf']=np.append(outimarr['uf'],im1)
			outimarr['fmuf']=np.append(outimarr['fmuf'],im2-im1)

	cubesize=int(outimarr['f'].shape[0]/imini.shape[0]**2)
	for name in ['f','uf','fmuf']:
		outimarr[name]=outimarr[name].reshape(cubesize,im1.shape[0],im1.shape[1])
	if stack==True:
		imf,imuf,imfmuf=np.nanmedian(outimarr['f'],axis=0),np.nanmedian(outimarr['uf'],axis=0),np.nanmedian(outimarr['fmuf'],axis=0) 
	else:
		imf,imuf,imfmuf=outimarr['f'],outimarr['uf'],outimarr['fmuf']
	if rf==True: 
		return imf/norm
	elif ruf==True:
		return imuf/norm
	elif rfmuf==True:
		return imfmuf/norm
	else:
		return imf/norm,imuf/norm,imfmuf/norm
'''