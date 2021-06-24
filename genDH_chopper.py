'''
generate SCC interaction matrix of Fourier modes and close the loop!

this version uses the optical chopper
'''


from ancillary_code import *
import numpy as np
import time
import functions
import itertools

#initial setup: apply best flat, generate DM grid to apply future shapes
dmcini=getdmc()
ydim,xdim=dmcini.shape
grid=np.mgrid[0:ydim,0:xdim].astype(float32)
bestflat=np.load('bestflat.npy') #for starting from flat DM with FPM aligned
#bestflat=np.load('dmc_dh.npy') #for bootstrapping
applydmc(bestflat)

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

ydim,xdim=dmcini.shape
grid=np.mgrid[0:ydim,0:xdim].astype(float32)
ygrid,xgrid=grid[0]-ydim/2,grid[1]-xdim/2
tip,tilt=(ygrid+ydim/2)/ydim,(xgrid+xdim/2)/xdim #min value is zero, max is one

#DM aperture:
undersize=29/32 #29 of the 32 actuators are illuminated
rho,phi=functions.polar_grid(xdim,xdim*undersize)
aperture=np.zeros(rho.shape).astype(float32)
indap=np.where(rho>0)
indnap=np.where(rho==0)
aperture[indap]=1

#aperture=np.load('DMmap.npy').astype(float32) #actual aperture, from close_SHWFS_loop.py; outdated

def remove_piston(dmc):  #function to remove piston from dm command to have zero mean (must be intermediate) 
	dmcout=dmc-np.median(dmc[indap])+0.5 #remove piston in the pupil
	dmcout[indnap]=bestflat[indnap] #set the non-illuminated regions to the best flat values
	return dmcout

IMtt=np.array([(tip).flatten(),(tilt).flatten()])
CMtt=np.linalg.pinv(IMtt,rcond=1e-5)
def rmtt(ph): #remove tip/tilt from DM commands
	coeffs=np.dot(np.vstack((ph).flatten()).T,CMtt) 
	lsqtt=np.dot(IMtt.T,coeffs.T).reshape(tilt.shape).astype(float32)
	return ph-lsqtt

#functions to apply DM Fourier modes 
ycen,xcen=ydim/2+0.5,xdim/2+0.5
indrho=np.where(rho>0)
gridnorm=np.max((grid[0]-ycen)[indrho])*2
rgrid=lambda pa:(grid[0]-ycen)/gridnorm*np.cos(pa*np.pi/180)+(grid[1]-xcen)/gridnorm*np.sin(pa*np.pi/180) #grid on which to define sine waves at a given position angle;
def dmsin(amp,freq,pa,bestflat=bestflat): #generate sine wave
	sin=amp*0.5*np.sin(2*np.pi*freq*rgrid(pa))
	sindm=sin.astype(float32)
	dmc=remove_piston(remove_piston(bestflat)+remove_piston(rmtt(sindm)))
	applydmc(dmc)
	return sindm

def dmcos(amp,freq,pa,bestflat=bestflat): #generate cosine wave
	cos=amp*0.5*np.cos(2*np.pi*freq*rgrid(pa))
	cosdm=cos.astype(float32)
	dmc=remove_piston(remove_piston(bestflat)+remove_piston(rmtt(cosdm)))
	applydmc(dmc)
	return cosdm

def dmf(amp,freq,pa,ph,bestflat=bestflat): #generate sin/cos wave (adjust phase)
	cos=amp*0.5*np.cos(2*np.pi*freq*rgrid(pa)-ph*np.pi/180)
	cosdm=cos.astype(float32)
	dmc=remove_piston(remove_piston(bestflat)+remove_piston(rmtt(cosdm)))
	applydmc(dmc)

def optt(tsleep): #function to optimize how long to wait in between applying DM command and recording image; note that this function is less meaningful at the moment, since 100 frames are acquired with the chopper running for each DM command in calibration
	dmsin(0.01,10,90)
	time.sleep(tsleep)
	im1=stackchop(ruf=True)
	applydmc(bestflat)
	time.sleep(tsleep)
	imf=stackchop(ruf=True)
	ds9.view(im1-imf)
#looks like I need to wait 3 seconds (full frame) or 0.05 seconds (320x320 ROI) in between each DM command to generate a stable image that isn't influenced by the previous command

tsleep=0.05 #on good days...
#tsleep=0.4 #on bad days...

#try to manuall null a few speckles in the remaining flat from Zygo; (only needed to run once)
'''
yind,xind=205,230 #left right speckle pair
nf,na,nph=10,10,10
farr=np.linspace(6,10,nf)
amparr=np.linspace(0.01,0.02,na)
pharr=np.linspace(0,360-360/nph,nph)
optarr=np.zeros((nf,na,nph))
for ff in range(nf):
	freq=farr[ff]
	for aa in range(na):
		amptune=amparr[aa]
		for pp in range(nph):
			ph=pharr[pp]
			dmf(amptune,freq,90,ph)
			time.sleep(tsleep)
			im=stack(100)
			optarr[ff,aa,pp]=im[yind,xind]
indopt=np.where(optarr==np.min(optarr))
dmf(amparr[indopt[1][0]],farr[indopt[0][0]],90,pharr[indopt[2][0]])
bestflat=getdmc()

yind,xind=155,175 #top bottom speckle pair
optarr=np.zeros((nf,na,nph))
for ff in range(nf):
	freq=farr[ff]
	for aa in range(na):
		amptune=amparr[aa]
		for pp in range(nph):
			ph=pharr[pp]
			dmf(amptune,freq,0,ph,bestflat=bestflat)
			time.sleep(tsleep)
			im=stack(100)
			optarr[ff,aa,pp]=im[yind,xind]
indopt=np.where(optarr==np.min(optarr))
dmf(amparr[indopt[1][0]],farr[indopt[0][0]],0,pharr[indopt[2][0]],bestflat=bestflat)
bestflat=getdmc()
np.save('bestflat.npy',bestflat)
'''

#apply four sine spots and fit Gaussians to find the image center; note that the pa is offset by 2 degrees from 0 and 90 degrees for the two sine spots with respects to the DM command plane; Daren says the DM clocked is likely just misaligned with respect to the Andor by this amount; this 2 deg shift compensates for this so the spots are vertical and horizonal with repsect to the Andor
amp,freq=0.01,6
dmsin(amp,freq,2)
time.sleep(tsleep)
imcen1=stackchop(ruf=True)
dmsin(amp,freq,92)
time.sleep(tsleep)
imcen2=stackchop(ruf=True)
applydmc(bestflat)
imcenf=stackchop(ruf=True)
#PAUSE, take a look at the image and guess the image center initially

beam_ratio=0.635*750/10.8/6.5 #theoretical number of pixels/resel: lambda (in microns)*focal length to camera (in mm)/coronagraphic beam diameter at the Lyot stop (in mm)/Andor pixel size (in microns)

#adjust the image center and beam ratio until happy
def vc(imxcenini,imycenini,fudge,cropi,rcrop=False): #syntax: vc(158,173,0.95,0), then vc(158,173,0.95,1), etc., continually iterating all numbers until happy that they are all centered 
	#imxcenini,imycenini=157.69,173
	#fudge=0.95 #tuning to center the side spots within each cropped region of the image

	freqfudge=freq*fudge #accound for the fact that the applied frequency could be wrong
	#fit the four sine spots with Gaussians to robustly set the image center
	ycropmin=np.round(np.array([imycenini-(freqfudge+1)*beam_ratio,imycenini-1*beam_ratio,imycenini+(freqfudge-1)*beam_ratio,imycenini-1*beam_ratio])).astype(int)
	ycropmax=np.round(np.array([imycenini-(freqfudge-1)*beam_ratio,imycenini+1*beam_ratio,imycenini+(freqfudge+1)*beam_ratio,imycenini+1*beam_ratio])).astype(int)
	xcropmin=np.round(np.array([imxcenini-1*beam_ratio,imxcenini-(freqfudge+1)*beam_ratio,imxcenini-1*beam_ratio,imxcenini+(freqfudge-1)*beam_ratio])).astype(int)
	xcropmax=np.round(np.array([imxcenini+1*beam_ratio,imxcenini-(freqfudge-1)*beam_ratio,imxcenini+1*beam_ratio,imxcenini+(freqfudge+1)*beam_ratio])).astype(int)

	crop1=(imcen1-imcenf)[ycropmin[0]:ycropmax[0],xcropmin[0]:xcropmax[0]]
	crop2=(imcen2-imcenf)[ycropmin[1]:ycropmax[1],xcropmin[1]:xcropmax[1]]
	crop3=(imcen1-imcenf)[ycropmin[2]:ycropmax[2],xcropmin[2]:xcropmax[2]]
	crop4=(imcen2-imcenf)[ycropmin[3]:ycropmax[3],xcropmin[3]:xcropmax[3]]
	ds9.view([crop1,crop2,crop3,crop4][cropi-1])
	if rcrop==True:
		return crop1,crop2,crop3,crop4,xcropmax,ycropmax

imxcenini,imycenini,freqfudge=180,191,1
crop1,crop2,crop3,crop4,xcropmax,ycropmax=vc(imxcenini,imycenini,freqfudge,0,rcrop=True)

from astropy.modeling import models,fitting
xpos,ypos=np.zeros(4),np.zeros(4)
for j in range(4):
	cropim=[crop1,crop2,crop3,crop4][j]
	fit_p=fitting.LevMarLSQFitter()
	indpos=np.where(cropim==np.max(cropim))
	pinit=models.Gaussian2D(np.max(cropim),indpos[1][0],indpos[0][0],1,1)
	xgrid,ygrid=np.mgrid[:cropim.shape[0],:cropim.shape[1]]
	p=fit_p(pinit,xgrid,ygrid,cropim)
	x0,y0=p.x_mean[0],p.y_mean[0]
	xpos[j],ypos[j]=xcropmax[j]-x0,ycropmax[j]-y0
imxcen,imycen=np.mean(xpos),np.mean(ypos)
np.save('imcen.npy',np.array([imxcen,imycen]))

#beam ratio calculation based on the the measured separation values; still discrepant with tune_beam_ratio below...
freqsep=(np.sqrt((xpos[2]-xpos[0])**2+(ypos[2]-ypos[0])**2)+np.sqrt((xpos[3]-xpos[1])**2+(ypos[3]-ypos[1])**2))/2
beam_ratio=freqsep/(2*freq*freqfudge)


#image cropping from full frame to ROI; not needed in subarray mode
'''
cropcen=[971,1119]
cropsize=[160,160]
cropim = lambda im: im[cropcen[0]-yimcen:cropcen[0]+yimcen,cropcen[1]-ximcen:cropcen[1]+ximcen]-np.median(im) #rudamentary bias subtraction is done here during the image cropping, assuming the image is large enough to not be affected by actual signal
'''

beam_ratio=6.5 #setting this as a hard value at the momnet as determined by the vf function below
np.save('beam_ratio.npy',beam_ratio)

Im_grid=np.mgrid[0:imini.shape[0],0:imini.shape[1]]
Im_rho=np.sqrt((Im_grid[0]-imycen)**2+(Im_grid[1]-imxcen)**2)
dh_mask=np.zeros(imini.shape)
maxld,minld=12,5 #maximum and minimum radius from which to dig a dark hole (sqrt(2) times larger for the maximum and the DH corners); both should be integers
#ylimld=11 #how much to go +/- in y for the DH
ind_dh=np.where(np.logical_and(np.logical_and(np.logical_and(Im_grid[1]-imxcen<(maxld+1)*beam_ratio,Im_grid[1]-imxcen>-(0+1)*beam_ratio),np.logical_and(Im_grid[0]-imycen<(maxld+1)*beam_ratio,Im_grid[0]-imycen>-(maxld+1)*beam_ratio)),Im_rho>(minld-1)*beam_ratio)) #full possible area for half DH, to thr right of the star in DS9
#ind_dh=np.where(np.logical_and(Im_rho<(maxld+2)*beam_ratio,Im_rho>(minld-2)*beam_ratio)) #full DH to flatten the DM
#ind_dh=np.where(dh_mask==0)
#ind_dh=np.where(np.logical_and(np.logical_and(Im_grid[1]-imxcen<-(minld)*beam_ratio,Im_grid[1]-imxcen>-(maxld)*beam_ratio),np.logical_and(Im_grid[0]-imycen<(maxld)*beam_ratio,Im_grid[0]-imycen>-(maxld)*beam_ratio))) #square DH
dh_mask[ind_dh]=1.

def scc_imin(imin,fmask=np.ones(imini.shape)): 
	'''
	function to generate images vectorized complex images within the DH after SCC processing 
	fmask adds an optional binary mask in I_minus space to linearize the IM
	'''
	Im_in=(processim(imin)*fmask)[ind_dh]
	return np.ndarray.flatten(np.array([np.real(Im_in),np.imag(Im_in)]))

imf=stackchop(ruf=True)
def tune_beam_ratio(beam_ratio,i,v=True): #function to determine the emperical beam ratio based on where the centering the binary mask on it's respective sine spots; the result should modify the beam_ratio variable back above the dh_mask code
	#CREATE IMAGE X,Y INDEXING THAT PLACES SINE, COSINE WAVES AT EACH LAMBDA/D REGION WITHIN THE NYQUIST REGION
	#the nyquist limit from the on-axis psf is +/- N/2 lambda/D away, so the whole nyquist region is N lambda/D by N lambda/D, centered on the on-axis PSF
	#to make things easier and symmetric, I want to place each PSF on a grid that is 1/2 lambda/D offset from the on-axis PSF position; this indexing should be PSF copy center placement location
	allx,ally=list(zip(*itertools.product(np.linspace(imxcen+0.5*beam_ratio,imxcen+maxld*beam_ratio-0.5*beam_ratio,maxld),np.linspace(imycen-maxld*beam_ratio+0.5*beam_ratio,imycen+maxld*beam_ratio-0.5*beam_ratio,2*maxld))))
	loopx,loopy=np.array(list(allx)).astype(float),np.array(list(ally)).astype(float)

	freq_loop=np.sqrt((loopy-imycen)**2+(loopx-imxcen)**2)/beam_ratio #sine wave frequency for (lambda/D)**2 region w/in DH
	pa_loop=90+180/np.pi*np.arctan2(loopy-imycen,loopx-imxcen) #position angle of sine wave for (lambda/D)**2 region w/in DH

	#only use desired spatial frequencies within the dark hole as input into the control matrix
	#indloop=np.where(np.logical_and(np.logical_and(loopy-imycen<maxld*beam_ratio,loopy-imycen>-maxld*beam_ratio),np.logical_and(loopx-imxcen<-minld*beam_ratio,loopy-imxcen>-maxld*beam_ratio))) #version for square dark hole; may not be updated!
	#indloop=np.where(np.logical_and(freq_loop>minld,freq_loop<maxld)) #version for annular dark hole
	indloop=np.where(np.logical_and(np.logical_and(np.logical_and(loopy-imycen<maxld*beam_ratio,loopy-imycen>-maxld*beam_ratio),np.logical_and(loopx-imxcen<maxld*beam_ratio,loopx-imxcen>0*beam_ratio)),freq_loop>minld)) #version for half dark hole
	loopx,loopy=loopx[indloop],loopy[indloop]
	freq_loop=freq_loop[indloop]
	pa_loop=pa_loop[indloop]

	#def vf(i): #function to look at binary mask around sine spot peaks in individual Fourier modes to see how well it is working in the IM further below; with the current setup, the theoretical mask location looks sufficient, removing the need to place bright sine spots to determine the mask for a given mode 
	applydmc(bestflat)
	time.sleep(tsleep)
	imf=stackchop(rfmuf=True)

	cos=dmcos(0.01,freq_loop[i],pa_loop[i])
	time.sleep(tsleep)
	imcb=stackchop(rfmuf=True)


	#diffimuf=np.abs(processim(imcb,mask=cenmask))-np.abs(processim(imf,mask=cenmask)) #unfringed differential image
	#indfmask=np.where(diffimuf==np.max(diffimuf[ind_dh]))
	#rmask=np.sqrt((mtfgrid[0]-indfmask[0][0])**2+(mtfgrid[1]-indfmask[1][0])**2)
	rmask=np.sqrt((Im_grid[0]-loopy[i])**2+(Im_grid[1]-loopx[i])**2)
	fmask=np.zeros(imini.shape)
	fmask[np.where(rmask<1.5*beam_ratio)]=1

	if v==True:
		ds9.view(np.abs(processim(imcb-imf))*fmask)
	else:
		return pa_loop,freq_loop,loopx,loopy


pa_loop,freq_loop,loopx,loopy=tune_beam_ratio(beam_ratio,0,v=False)
applydmc(bestflat)

fourierarr=np.zeros((len(freq_loop)*2,dmcini.flatten().shape[0]))
refvec=np.zeros((len(freq_loop)*2,len(dh_mask[ind_dh])*2))
for i in range(len(freq_loop)):

	#old method to define binary mask around sine spot manually based on where the peak is
	'''
	cos=dmcos(0.05,freq_loop[i],pa_loop[i])
	time.sleep(tsleep)
	imcb=stackchop(ruf=True)
	#diffimuf=np.abs(processim(imcb,mask=cenmask))-np.abs(processim(imf,mask=cenmask)) #unfringed differential image
	diffimuf=imcb-imfuf #unfringed differential image
	indfmask=np.where(diffimuf==np.max(diffimuf[ind_dh]))
	rmask=np.sqrt((mtfgrid[0]-indfmask[0][0])**2+(mtfgrid[1]-indfmask[1][0])**2)
	'''
	rmask=np.sqrt((Im_grid[0]-loopy[i])**2+(Im_grid[1]-loopx[i])**2)
	fmask=np.zeros(imini.shape)
	fmask[np.where(rmask<1.5*beam_ratio)]=1

	#add for full DH mask (to flatten DM)
	'''
	rmask2=np.sqrt((Im_grid[0]-(2*imycen-loopy[i]))**2+(Im_grid[1]-(2*imxcen-loopx[i]))**2)
	fmask[np.where(rmask2<1*beam_ratio)]=1	
	'''

	sin=dmsin(0.005,freq_loop[i],pa_loop[i])
	#sin=dmsin(0.1,freq_loop[i],pa_loop[i])
	time.sleep(tsleep)
	ims=stackchop(rfmuf=True)
	cos=dmcos(0.005,freq_loop[i],pa_loop[i])
	#cos=dmcos(0.1,freq_loop[i],pa_loop[i])
	time.sleep(tsleep)
	imc=stackchop(rfmuf=True)

	applydmc(bestflat)
	time.sleep(tsleep)
	imff,imfuf,imf=stackchop()

	fourierarr[i]=sin.flatten()
	fourierarr[i+len(freq_loop)]=cos.flatten()

	refvec[i]=scc_imin(ims-imf,fmask=fmask)
	refvec[i+len(freq_loop)]=scc_imin(imc-imf,fmask=fmask)
	print(i/len(freq_loop))

applydmc(bestflat)
IM=np.dot(refvec,refvec.T)

from datetime import datetime
np.save('IM/SCC/'+datetime.now().strftime("%d_%m_%Y_%H")+'.npy',IM)
np.save('SCC_IM_chopper.npy',IM)
#np.save('SCC_IM_flatdm.npy',IM)

#the following code will help determine the optimal SVD cutoff, looking at what SVD cutoff will best reconstruct individual modes

plt.figure()
def pc(rcond,i):
	#rcond=1e-3
	IMinv=np.linalg.pinv(IM,rcond=rcond)
	cmd_mtx=np.dot(IMinv,refvec)
	#i=100
	sin=dmsin(0.01,freq_loop[i],pa_loop[i])
	time.sleep(0.1)
	dum,dum,imin=stackchop()
	tar=scc_imin(imin-imf)
	coeffs=np.dot(cmd_mtx,tar)
	plt.plot(coeffs)


rcond=1e-5
#rcond=1e-1 #for attempted DM flattening
IMinv=np.linalg.pinv(IM,rcond=rcond)
cmd_mtx=np.dot(IMinv,refvec)

numiter=20
gain=0.3
leak=1
applydmc(bestflat)
imff,imfuf,imffmuf=stackchop()

time.sleep(tsleep)
for nit in range(numiter):
	imin=stackchop(rfmuf=True) #larger number of stacks increases the amount by which you can gain...
	tar=scc_imin(imin)
	coeffs=np.dot(cmd_mtx,tar)
	cmd=np.dot(fourierarr.T,-coeffs).reshape(dmcini.shape).astype(float32)
	applydmc((remove_piston(leak*getdmc()+cmd*gain)))
	time.sleep(tsleep)

dmc_dh=getdmc()

np.save('dmc_dh.npy',dmc_dh) #for half dark hole
#np.save('dmc_dh.npy',dmc_dh) #for flattening the DM (i.e., whole dark hole)

imdhf,imdhuf,imdhfmuf=stackchop()

np.save('images/SCC/flat_'+datetime.now().strftime("%d_%m_%Y_%H")+'.npy',np.array([imff,imfuf,imffmuf]))
np.save('images/SCC/flat_'+datetime.now().strftime("%d_%m_%Y_%H")+'.npy',np.array([imdhf,imdhuf,imdhfmuf]))

ind_cdi=np.where(Im_rho<(maxld)*beam_ratio) #index over which to normalize the reconstructed I minus
def cdi(imfmuf,imuf):
	I_R=np.abs(processim(imfmuf,mask=pinmask))
	I_minus=np.abs(processim(imfmuf))
	I_S_recon=I_minus**2/I_R

	imout=imuf-I_S_recon*np.std(imuf[ind_dh])/np.std(I_S_recon[ind_dh])
	return imout


from scipy.ndimage.filters import median_filter
hpf=lambda im: im-median_filter(im,21)

imin=hpf(imff)
imdhin=hpf(imdhf)
imdhout=hpf(cdi(imdhfmuf,imdhuf))

