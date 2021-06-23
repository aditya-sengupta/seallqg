'''
generate SCC interaction matrix of Fourier modes and close the loop!
'''

from ancillary_code import *
import numpy as np
import time
import ao_utils
import itertools

#initial setup: apply best flat, generate DM grid to apply future shapes
dmcini = getdmc()
ydim,xdim = dmcini.shape
grid = np.mgrid[0:ydim,0:xdim].astype(float32)
bestflat = np.load('bestflat.npy') #for starting from flat DM with FPM aligned
#bestflat = np.load('dmc_dh.npy') #for bootstrapping
applydmc(bestflat)

ygrid,xgrid = grid[0]-ydim/2,grid[1]-xdim/2
xy = np.sqrt(ygrid**2+xgrid**2)

expt(4e-3) #set exposure time
imini = getim()

ydim,xdim = dmcini.shape
grid = np.mgrid[0:ydim,0:xdim].astype(float32)
ygrid,xgrid = grid[0]-ydim/2,grid[1]-xdim/2
tip,tilt = (ygrid+ydim/2)/ydim,(xgrid+xdim/2)/xdim #min value is zero, max is one

#DM aperture:
undersize = 27/32 #assuming 27 of the 32 actuators are illuminated
rho,phi = ao_utils.polar_grid(xdim,xdim*undersize)
cenaperture = np.zeros(rho.shape).astype(float32)
indapcen = np.where(rho>0)
cenaperture[indapcen] = 1

aperture = np.load('DMmap.npy').astype(float32) #actual aperture, from close_SHWFS_loop.py

#from comparing cenaperture and aperture, the actual aperture is shifted down and to the right (in ds9) each by 1 pixel from the center
yapcen,xapcen = ydim/2.-0.5-1,xdim/2.-0.5-1
rap = np.sqrt((grid[0]-yapcen)**2.+(grid[1]-xapcen)**2.)
rap[np.where(rap>xdim/2.*undersize)] = 0.
rhoap = rap/np.max(rap)
phiap = np.arctan2(grid[1]-yapcen,grid[0]-xapcen)
indap = np.where(rhoap>0)

remove_piston  =  lambda dmc: dmc-np.mean(dmc[indap]) #function to remove piston from dm command to have zero mean (must be intermediate)

IMtt = np.array([(tip*aperture).flatten(),(tilt*aperture).flatten()])
CMtt = np.linalg.pinv(IMtt,rcond = 1e-5)
def rmtt(ph): #remove tip/tilt from aperture
	coeffs = np.dot(np.vstack((ph*aperture).flatten()).T,CMtt) 
	lsqtt = np.dot(IMtt.T,coeffs.T).reshape(tilt.shape).astype(float32)
	return ph*aperture-lsqtt

#functions to apply DM Fourier modes 
ycen,xcen = yapcen,xapcen
indrho = np.where(rhoap>0)
gridnorm = np.max((grid[0]-ycen)[indrho])*2
rgrid = lambda pa:(grid[0]-ycen)/gridnorm*np.cos(pa*np.pi/180)+(grid[1]-xcen)/gridnorm*np.sin(pa*np.pi/180) #grid on which to define sine waves at a given position angle;
def dmsin(amp,freq,pa,bestflat = bestflat): #generate sine wave
	sin = amp*0.5*np.sin(2*np.pi*freq*rgrid(pa))
	sindm = sin.astype(float32)
	dmc = remove_piston(bestflat)+remove_piston(rmtt(aperture*sindm))+0.5
	applydmc(dmc)
	return sindm

def dmcos(amp,freq,pa,bestflat = bestflat): #generate cosine wave
	cos = amp*0.5*np.cos(2*np.pi*freq*rgrid(pa))
	cosdm = cos.astype(float32)
	dmc = remove_piston(bestflat)+remove_piston(rmtt(aperture*cosdm))+0.5
	applydmc(dmc)
	return cosdm


def optt(tsleep): #function to optimize how long to wait in between applying DM command and recording image
	dmsin(0.02,10,90)
	time.sleep(tsleep)
	im1 = stack(10)
	applydmc(bestflat)
	time.sleep(tsleep)
	imf = stack(10)
	ds9.view(im1-imf)
#looks like I need to wait 3 seconds (full frame) or 0.05 seconds (320x320 ROI) in between each DM command to generate a stable image that isn't influenced by the previous command

#tsleep = 0.05 #on good days...
tsleep = 0.4 #on bad days...

#apply four sine spots and fit Gaussians to find the image center
amp,freq = 0.03,10
dmsin(amp,freq,0)
time.sleep(tsleep)
imcen1 = stack(1000)
dmsin(amp,freq,90)
time.sleep(tsleep)
imcen2 = stack(1000)
applydmc(bestflat)
#PAUSE, take a look at the image and guess the image center initially

beam_ratio = 0.635*750/10.8/6.5 #theoretical number of pixels/resel: lambda (in microns)*focal length to camera (in mm)/coronagraphic beam diameter at the Lyot stop (in mm)/Andor pixel size (in microns)

#adjust the image center and beam ratio until happy
def vc(imxcenini,imycenini,fudge,cropi,rcrop = False): #syntax: vc(158,173,0.95,0), then vc(158,173,0.95,1), etc., continually iterating all numbers until happy that they are all centered 
	#imxcenini,imycenini = 157.69,173
	#fudge = 0.95 #tuning to center the side spots within each cropped region of the image

	freqfudge = freq*fudge #accound for the fact that the applied frequency could be wrong
	#fit the four sine spots with Gaussians to robustly set the image center
	ycropmin = np.round(np.array([imycenini-(freqfudge+1)*beam_ratio,imycenini-1*beam_ratio,imycenini+(freqfudge-1)*beam_ratio,imycenini-1*beam_ratio])).astype(int)
	ycropmax = np.round(np.array([imycenini-(freqfudge-1)*beam_ratio,imycenini+1*beam_ratio,imycenini+(freqfudge+1)*beam_ratio,imycenini+1*beam_ratio])).astype(int)
	xcropmin = np.round(np.array([imxcenini-1*beam_ratio,imxcenini-(freqfudge+1)*beam_ratio,imxcenini-1*beam_ratio,imxcenini+(freqfudge-1)*beam_ratio])).astype(int)
	xcropmax = np.round(np.array([imxcenini+1*beam_ratio,imxcenini-(freqfudge-1)*beam_ratio,imxcenini+1*beam_ratio,imxcenini+(freqfudge+1)*beam_ratio])).astype(int)

	crop1 = imcen1[ycropmin[0]:ycropmax[0],xcropmin[0]:xcropmax[0]]
	crop2 = imcen2[ycropmin[1]:ycropmax[1],xcropmin[1]:xcropmax[1]]
	crop3 = imcen1[ycropmin[2]:ycropmax[2],xcropmin[2]:xcropmax[2]]
	crop4 = imcen2[ycropmin[3]:ycropmax[3],xcropmin[3]:xcropmax[3]]
	ds9.view([crop1,crop2,crop3,crop4][cropi-1])
	if rcrop == True:
		return crop1,crop2,crop3,crop4,xcropmax,ycropmax

imxcenini,imycenini,freqfudge = 166,138,0.95
crop1,crop2,crop3,crop4,xcropmax,ycropmax = vc(imxcenini,imycenini,freqfudge,0,rcrop = True)

from astropy.modeling import models,fitting
xpos,ypos = np.zeros(4),np.zeros(4)
for j in range(4):
	cropim = [crop1,crop2,crop3,crop4][j]
	fit_p = fitting.LevMarLSQFitter()
	indpos = np.where(cropim == np.max(cropim))
	pinit = models.Gaussian2D(np.max(cropim),indpos[1][0],indpos[0][0],1,1)
	xgrid,ygrid = np.mgrid[:cropim.shape[0],:cropim.shape[1]]
	p = fit_p(pinit,xgrid,ygrid,cropim)
	x0,y0 = p.x_mean[0],p.y_mean[0]
	xpos[j],ypos[j] = xcropmax[j]-x0,ycropmax[j]-y0
imxcen,imycen = np.mean(xpos),np.mean(ypos)
np.save('imcen.npy',np.array([imxcen,imycen]))

#trying to calculate the beam ratio based on the the measured separation values... doens't seem to be working well 
#freqsep = (np.sqrt((xpos[2]-xpos[0])**2+(ypos[2]-ypos[0])**2)+np.sqrt((xpos[3]-xpos[1])**2+(ypos[3]-ypos[1])**2))/2
#beam_ratio = freqsep/(2*freq*freqfudge)


#image cropping from full frame to ROI; not needed in subarray mode
'''
cropcen = [971,1119]
cropsize = [160,160]
cropim  =  lambda im: im[cropcen[0]-yimcen:cropcen[0]+yimcen,cropcen[1]-ximcen:cropcen[1]+ximcen]-np.median(im) #rudamentary bias subtraction is done here during the image cropping, assuming the image is large enough to not be affected by actual signal
'''

#make MTF side lobe mask
xsidemaskcen,ysidemaskcen = 252.01,159.4 #x and y location of the side lobe mask in the cropped image
sidemaskrad = 26.8 #radius of the side lobe mask
mtfgrid = np.mgrid[0:imini.shape[0],0:imini.shape[1]]
sidemaskrho = np.sqrt((mtfgrid[0]-ysidemaskcen)**2+(mtfgrid[1]-xsidemaskcen)**2)
sidemask = np.zeros(imini.shape)
sidemaskind = np.where(sidemaskrho<sidemaskrad)
sidemask[sidemaskind] = 1

#central lobe MTF mask
yimcen,ximcen = imini.shape[0]/2,imini.shape[1]/2
cenmaskrad = 49
cenmaskrho = np.sqrt((mtfgrid[0]-yimcen)**2+(mtfgrid[1]-ximcen)**2)
cenmask = np.zeros(imini.shape)
cenmaskind = np.where(cenmaskrho<cenmaskrad)
cenmask[cenmaskind] = 1

def processim(imin,mask = sidemask): #process SCC image, isolating the sidelobe in the FFT and IFFT back to the image
	otf = np.fft.fftshift(np.fft.fft2(imin,norm = 'ortho')) #(1) FFT the image
	otf_masked = otf*mask #(2) multiply by binary mask to isolate side lobe
	Iminus = np.fft.ifft2(otf_masked,norm = 'ortho') #(3) IFFT back to the image plane, now generating a complex-valued image
	return Iminus

beam_ratio = 6.4 #setting this as a hard value at the momnet as determined by the vf function below
np.save('beam_ratio.npy',beam_ratio)

Im_grid = np.mgrid[0:imini.shape[0],0:imini.shape[1]]
Im_rho = np.sqrt((Im_grid[0]-imycen)**2+(Im_grid[1]-imxcen)**2)
dh_mask = np.zeros(imini.shape)
maxld,minld = 10,5 #maximum and minimum radius from which to dig a dark hole (sqrt(2) times larger for the maximum and the DH corners); both should be integers
#ylimld = 11 #how much to go +/- in y for the DH
ind_dh = np.where(np.logical_and(np.logical_and(np.logical_and(Im_grid[1]-imxcen<(maxld+1)*beam_ratio,Im_grid[1]-imxcen>-(0+1)*beam_ratio),np.logical_and(Im_grid[0]-imycen<(maxld+1)*beam_ratio,Im_grid[0]-imycen>-(maxld+1)*beam_ratio)),Im_rho>(minld-1)*beam_ratio)) #full possible area for half DH, to thr right of the star in DS9
#ind_dh = np.where(np.logical_and(Im_rho<(maxld+2)*beam_ratio,Im_rho>(minld-2)*beam_ratio)) #full DH to flatten the DM
#ind_dh = np.where(dh_mask == 0)
#ind_dh = np.where(np.logical_and(np.logical_and(Im_grid[1]-imxcen<-(minld)*beam_ratio,Im_grid[1]-imxcen>-(maxld)*beam_ratio),np.logical_and(Im_grid[0]-imycen<(maxld)*beam_ratio,Im_grid[0]-imycen>-(maxld)*beam_ratio))) #square DH
dh_mask[ind_dh] = 1.

def scc_imin(imin,fmask = np.ones(imini.shape)): 
	'''
	function to generate images vectorized complex images within the DH after SCC processing 
	fmask adds an optional binary mask in I_minus space to linearize the IM
	'''
	Im_in = (processim(imin)*fmask)[ind_dh]
	return np.ndarray.flatten(np.array([np.real(Im_in),np.imag(Im_in)]))

def tune_beam_ratio(beam_ratio,i,v = True): #function to determine the emperical beam ratio based on where the centering the binary mask on it's respective sine spots; the result should modify the beam_ratio variable back above the dh_mask code
	#CREATE IMAGE X,Y INDEXING THAT PLACES SINE, COSINE WAVES AT EACH LAMBDA/D REGION WITHIN THE NYQUIST REGION
	#the nyquist limit from the on-axis psf is +/- N/2 lambda/D away, so the whole nyquist region is N lambda/D by N lambda/D, centered on the on-axis PSF
	#to make things easier and symmetric, I want to place each PSF on a grid that is 1/2 lambda/D offset from the on-axis PSF position; this indexing should be PSF copy center placement location
	allx,ally = list(zip(*itertools.product(np.linspace(imxcen-+0.5*beam_ratio,imxcen+maxld*beam_ratio-0.5*beam_ratio,maxld),np.linspace(imycen-maxld*beam_ratio+0.5*beam_ratio,imycen+maxld*beam_ratio-0.5*beam_ratio,2*maxld))))
	loopx,loopy = np.array(list(allx)).astype(float),np.array(list(ally)).astype(float)

	freq_loop = np.sqrt((loopy-imycen)**2+(loopx-imxcen)**2)/beam_ratio #sine wave frequency for (lambda/D)**2 region w/in DH
	pa_loop = 90+180/np.pi*np.arctan2(loopy-imycen,loopx-imxcen) #position angle of sine wave for (lambda/D)**2 region w/in DH

	#only use desired spatial frequencies within the dark hole as input into the control matrix
	#indloop = np.where(np.logical_and(np.logical_and(loopy-imycen<maxld*beam_ratio,loopy-imycen>-maxld*beam_ratio),np.logical_and(loopx-imxcen<-minld*beam_ratio,loopy-imxcen>-maxld*beam_ratio))) #version for square dark hole; may not be updated!
	#indloop = np.where(np.logical_and(freq_loop>minld,freq_loop<maxld)) #version for annular dark hole
	indloop = np.where(np.logical_and(np.logical_and(np.logical_and(loopy-imycen<maxld*beam_ratio,loopy-imycen>-maxld*beam_ratio),np.logical_and(loopx-imxcen<maxld*beam_ratio,loopx-imxcen>-0*beam_ratio)),freq_loop>minld)) #version for full dark hole
	loopx,loopy = loopx[indloop],loopy[indloop]
	freq_loop = freq_loop[indloop]
	pa_loop = pa_loop[indloop]

	#def vf(i): #function to look at binary mask around sine spot peaks in individual Fourier modes to see how well it is working in the IM further below; with the current setup, the theoretical mask location looks sufficient, removing the need to place bright sine spots to determine the mask for a given mode 
	applydmc(bestflat)
	time.sleep(tsleep)
	imf = stack(100)

	cos = dmcos(0.02,freq_loop[i],pa_loop[i])
	time.sleep(tsleep)
	imcb = stack(100)


	#diffimuf = np.abs(processim(imcb,mask = cenmask))-np.abs(processim(imf,mask = cenmask)) #unfringed differential image
	#indfmask = np.where(diffimuf == np.max(diffimuf[ind_dh]))
	#rmask = np.sqrt((mtfgrid[0]-indfmask[0][0])**2+(mtfgrid[1]-indfmask[1][0])**2)
	rmask = np.sqrt((Im_grid[0]-loopy[i])**2+(Im_grid[1]-loopx[i])**2)
	fmask = np.zeros(imini.shape)
	fmask[np.where(rmask<1*beam_ratio)] = 1

	if v == True:
		ds9.view(np.abs(processim(imcb-imf))*fmask)
	else:
		return pa_loop,freq_loop,loopx,loopy


pa_loop,freq_loop,loopx,loopy = tune_beam_ratio(beam_ratio,0,v = False)
applydmc(bestflat)

nstack = 1000 #number of frames to stack to reach sufficient fringe SNR
applydmc(bestflat)
time.sleep(tsleep)
imf = stack(nstack)

fourierarr = np.zeros((len(freq_loop)*2,dmcini.flatten().shape[0]))
refvec = np.zeros((len(freq_loop)*2,len(dh_mask[ind_dh])*2))
for i in range(len(freq_loop)):

	#old method to define binary mask around sine spot manually based on where the peak is
	'''
	cos = dmcos(0.2,freq_loop[i],pa_loop[i])
	time.sleep(tsleep)
	imcb = stack(100)
	diffimuf = np.abs(processim(imcb,mask = cenmask))-np.abs(processim(imf,mask = cenmask)) #unfringed differential image
	indfmask = np.where(diffimuf == np.max(diffimuf[ind_dh]))
	rmask = np.sqrt((mtfgrid[0]-indfmask[0][0])**2+(mtfgrid[1]-indfmask[1][0])**2)
	'''
	rmask = np.sqrt((Im_grid[0]-loopy[i])**2+(Im_grid[1]-loopx[i])**2)
	fmask = np.zeros(imini.shape)
	fmask[np.where(rmask<1*beam_ratio)] = 1

	#add for full DH mask (to flatten DM)
	'''
	rmask2 = np.sqrt((Im_grid[0]-(2*imycen-loopy[i]))**2+(Im_grid[1]-(2*imxcen-loopx[i]))**2)
	fmask[np.where(rmask2<1*beam_ratio)] = 1	
	'''

	sin = dmsin(0.01,freq_loop[i],pa_loop[i])
	#sin = dmsin(0.1,freq_loop[i],pa_loop[i])
	time.sleep(tsleep)
	ims = stack(nstack)
	cos = dmcos(0.01,freq_loop[i],pa_loop[i])
	#cos = dmcos(0.1,freq_loop[i],pa_loop[i])
	time.sleep(tsleep)
	imc = stack(nstack)

	fourierarr[i] = sin.flatten()
	fourierarr[i+len(freq_loop)] = cos.flatten()

	refvec[i] = scc_imin(ims-imf,fmask = fmask)
	refvec[i+len(freq_loop)] = scc_imin(imc-imf,fmask = fmask)
	print(i/len(freq_loop))

applydmc(bestflat)
IM = np.dot(refvec,refvec.T)

from datetime import datetime
np.save('IM/SCC/'+datetime.now().strftime("%d_%m_%Y_%H_%M")+'.npy',IM)
np.save('SCC_IM.npy',IM)
#np.save('SCC_IM_flatdm.npy',IM)

#the following code will help determine the optimal SVD cutoff, looking at what SVD cutoff will best reconstruct individual modes

plt.figure()
def pc(rcond,i):
	#rcond = 1e-3
	IMinv = np.linalg.pinv(IM,rcond = rcond)
	cmd_mtx = np.dot(IMinv,refvec)
	#i = 100
	sin = dmsin(0.01,freq_loop[i],pa_loop[i])
	time.sleep(0.1)
	imin = stack(100)
	tar = scc_imin(imin-imf)
	coeffs = np.dot(cmd_mtx,tar)
	plt.plot(coeffs)


rcond = 5e-4
#rcond = 1e-1 #for attempted DM flattening
IMinv = np.linalg.pinv(IM,rcond = rcond)
cmd_mtx = np.dot(IMinv,refvec)

numiter = 30
gain = 0.5
leak = 1
applydmc(bestflat)
time.sleep(tsleep)
for nit in range(numiter):
	imin = stack(100) #larger number of stacks increases the amount by which you can gain...
	tar = scc_imin(imin)
	coeffs = np.dot(cmd_mtx,tar)
	cmd = np.dot(fourierarr.T,-coeffs).reshape(dmcini.shape).astype(float32)
	applydmc(leak*getdmc()+cmd*gain)
	time.sleep(tsleep)

dmc_dh = getdmc()

np.save('dmc_dh.npy',dmc_dh) #for half dark hole
#np.save('dmc_dh.npy',dmc_dh) #for flattening the DM (i.e., whole dark hole)