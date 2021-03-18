'''
generate SCC interaction matrix of Fourier modes and close the loop!
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

expt(1e-3) #set exposure time
imini=getim()

beam_ratio=0.635*750/10.72/6.5 #theoretical number of pixels/resel: lambda (in microns)*focal length to camera (in mm)/Lyot stop beam diameter (in mm)/Andor pixel size (in microns)

#DM aperture:
rho,phi=functions.polar_grid(xdim,xdim*29/32) #assuming 29 of the 32 actuators are illuminated
aperture=np.zeros(rho.shape).astype(float32)
indap=np.where(rho>0)
aperture[indap]=1

#functions to apply DM Fourier modes 
ycen,xcen=ydim/2-0.5,xdim/2-0.5
indrho1=np.where(rho==1)
gridnorm=np.max(grid[0][indrho1])
rgrid=lambda pa:(grid[0]-ycen)/gridnorm*np.cos(pa*np.pi/180)+(grid[1]-xcen)/gridnorm*np.sin(pa*np.pi/180)
def dmsin(amp,freq,pa): #generate sine wave
	sin=amp*0.5*np.sin(2*np.pi*freq*rgrid(pa))
	sindm=sin.astype(float32)
	dmc=bestflat+sindm
	applydmc(dmc*aperture)
	return sindm

def dmcos(amp,freq,pa): #generate cosine wave
	cos=amp*0.5*np.cos(2*np.pi*freq*rgrid(pa))
	cosdm=cos.astype(float32)
	dmc=bestflat+cosdm
	applydmc(dmc*aperture)
	return cosdm

#apply four sine spots and fit Gaussians to find the image center
amp,freq=0.1,11
pa=0
sin1=amp*0.5*np.sin(2*np.pi*freq*rgrid(pa))
sindm1=sin1.astype(float32)
pa=90
sin2=amp*0.5*np.sin(2*np.pi*freq*rgrid(pa))
sindm2=sin2.astype(float32)
dmc=bestflat+sindm1+sindm2
applydmc(dmc*aperture)
#PAUSE, take a look at the image and guess the image center initially
imcen=stack(1000)
imxcen,imycen=158,168
# code is unfinished, not working yet, still working on it!
'''
crop1=imcen[int(imycen-(freq+2)*beam_ratio):int(imycen-(freq-2)*beam_ratio),int(imxcen-2*beam_ratio):int(imxcen+2*beam_ratio)]
crop2=imcen[int(imycen-2*beam_ratio):int(imycen+2*beam_ratio),int(imxcen-(freq+2)*beam_ratio):int(imxcen-(freq-2)*beam_ratio)]
crop3=imcen[int(imycen+(freq-2)*beam_ratio):int(imycen+(freq+2)*beam_ratio),int(imxcen-2*beam_ratio):int(imxcen+2*beam_ratio)]
crop4=imcen[int(imycen-2*beam_ratio):int(imycen+2*beam_ratio),int(imxcen+(freq-2)*beam_ratio):int(imxcen+(freq+2)*beam_ratio)]

from astropy.modeling import models,fitting
cropim=crop4
fit_p=fitting.LevMarLSQFitter()
indpos=np.where(cropim==np.max(cropim))
pinit=models.Gaussian2D(np.max(cropim),indpos[1][0],indpos[0][0],1,1)
xgrid,ygrid=np.mgrid[:cropim.shape[0],:cropim.shape[1]]
p=fit_p(pinit,xgrid,ygrid,cropim)
x0,y0=p.x_mean[0],p.y_mean[0]
'''
#image cropping from full frame to ROI; not needed in subarray mode
'''
cropcen=[971,1119]
cropsize=[160,160]
cropim = lambda im: im[cropcen[0]-yimcen:cropcen[0]+yimcen,cropcen[1]-ximcen:cropcen[1]+ximcen]-np.median(im) #rudamentary bias subtraction is done here during the image cropping, assuming the image is large enough to not be affected by actual signal
'''

#make MTF side lobe mask
xsidemaskcen,ysidemaskcen=161.66,252.22 #x and y location of the side lobe mask in the cropped image
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

def processim(imin,mask=sidemask): #process SCC image, isolating the sidelobe in the FFT and IFFT back to the image
	otf=np.fft.fftshift(np.fft.fft2(imin,norm='ortho')) #(1) FFT the image
	otf_masked=otf*mask #(2) multiply by binary mask to isolate side lobe
	Iminus=np.fft.ifft2(otf_masked,norm='ortho') #(3) IFFT back to the image plane, now generating a complex-valued image
	return Iminus

Im_rho=np.sqrt((mtfgrid[0]-imycen)**2+(mtfgrid[1]-imxcen)**2)
dh_mask=np.zeros(imini.shape)
maxld,minld=12,5 #maximum and minimum radius from which to dig a dark hole (sqrt(2) times larger for the maximum and the DH corners)
ylimld=7 #how much to go +/- in y for the DH
#ind_dh=np.where(np.logical_and(np.logical_and(Im_rho<(maxld+2)*beam_ratio,Im_rho>(minld-2)*beam_ratio),mtfgrid[1]-imxcen<0))
#ind_dh=np.where(np.logical_and(Im_rho<(maxld+2)*beam_ratio,Im_rho>(minld-2)*beam_ratio)) #full DH to flatten the DM
#ind_dh=np.where(dh_mask==0)
ind_dh=np.where(np.logical_and(np.logical_and(mtfgrid[1]-imxcen<-(minld-2)*beam_ratio,mtfgrid[1]-imxcen>-(maxld+2)*beam_ratio),np.logical_and(mtfgrid[0]-imycen<(ylimld+2)*beam_ratio,mtfgrid[0]-imycen>-(ylimld+2)*beam_ratio))) #note the +2 padding, which accounts for the unknown beam ratio in not cutting off the fmask later in the IM code at the edges of the DH
dh_mask[ind_dh]=1.


def scc_imin(imin,fmask=np.ones(imini.shape)): 
	'''
	function to generate images vectorized complex images within the DH after SCC processing 
	fmask adds an optional binary mask in I_minus space to linearize the IM
	'''
	Im_in=(processim(imin)*fmask)[ind_dh]
	return np.ndarray.flatten(np.array([np.real(Im_in),np.imag(Im_in)]))

#CREATE IMAGE X,Y INDEXING THAT PLACES SINE, COSINE WAVES AT EACH LAMBDA/D REGION WITHIN THE NYQUIST REGION
#the nyquist limit from the on-axis psf is +/- N/2 lambda/D away, so the whole nyquist region is N lambda/D by N lambda/D, centered on the on-axis PSF
#to make things easier and symmetric, I want to place each PSF on a grid that is 1/2 lambda/D offset from the on-axis PSF position; this indexing should be PSF copy center placement location
allx,ally=list(zip(*itertools.product(np.linspace(imxcen-maxld*beam_ratio+0.5*beam_ratio,imxcen-0.5*beam_ratio,maxld),np.linspace(imycen-maxld*beam_ratio+0.5*beam_ratio,imycen+maxld*beam_ratio-0.5*beam_ratio,2*maxld))))
loopx,loopy=np.array(list(allx)).astype(float),np.array(list(ally)).astype(float)

freq_loop=np.sqrt((loopy-imycen)**2+(loopx-imxcen)**2)/beam_ratio #sine wave frequency for (lambda/D)**2 region w/in DH
pa_loop=90-180/np.pi*np.arctan2(loopy-imycen,loopx-imxcen) #position angle of sine wave for (lambda/D)**2 region w/in DH

#only use desired spatial frequencies within the dark hole as input into the control matrix
indloop=np.where(np.logical_and(np.logical_and(loopy-imycen<ylimld*beam_ratio,loopy-imycen>-ylimld*beam_ratio),np.logical_and(loopx-imxcen<-minld*beam_ratio,loopy-imxcen>-maxld*beam_ratio))) #version for square dark hole
#indloop=np.where(np.logical_and(freq_loop>minld,freq_loop<maxld)) #version for annular dark hole
freq_loop=freq_loop[indloop]
pa_loop=pa_loop[indloop]

#function to look at binary mask around sine spot peaks in individual Fourier modes to see how well it is working in the IM further below
'''
def vf(i):
	cos=dmcos(0.2,freq_loop[i],pa_loop[i])
	time.sleep(tsleep)
	imcb=stack(1000)

	diffimuf=np.abs(processim(imcb,mask=cenmask))-np.abs(processim(imf,mask=cenmask)) #unfringed differential image
	indfmask=np.where(diffimuf==np.max(diffimuf[ind_dh]))
	rmask=np.sqrt((mtfgrid[0]-indfmask[0][0])**2+(mtfgrid[1]-indfmask[1][0])**2)
	fmask=np.zeros(diffimuf.shape)
	fmask[np.where(rmask<1*beam_ratio)]=1

	ds9.view(np.abs(processim(imcb-imf))*fmask)
'''

def optt(tsleep): #function to optimize how long to wait in between applying DM command and recording image
	i=0
	dmsin(0.2,freq_loop[i],pa_loop[i])
	time.sleep(tsleep)
	im1=stack(10)
	applydmc(bestflat)
	time.sleep(tsleep)
	imf=stack(10)
	ds9.view(im1-imf)
#looks like I need to wait 3 seconds (full frame) or 0.05 seconds (320x320 ROI) in between each DM command to generate a stable image that isn't influenced by the previous command


tsleep=0.05
nstack=1000 #number of frames to stack to reach sufficient fringe SNR
applydmc(bestflat)
time.sleep(tsleep)
imf=stack(nstack)

fourierarr=np.zeros((len(freq_loop)*2,dmcini.flatten().shape[0]))
refvec=np.zeros((len(freq_loop)*2,len(dh_mask[ind_dh])*2))
for i in range(len(freq_loop)):
	'''
	if freq_loop[i]<minld or freq_loop[i]>maxld: #leave spatial frequencies too close to the FPM IWA or outside the OWA set to zero
		continue
	else:
	'''
	
	cos=dmcos(0.2,freq_loop[i],pa_loop[i])
	time.sleep(tsleep)
	imcb=stack(100)

	diffimuf=np.abs(processim(imcb,mask=cenmask))-np.abs(processim(imf,mask=cenmask)) #unfringed differential image
	indfmask=np.where(diffimuf==np.max(diffimuf[ind_dh]))
	rmask=np.sqrt((mtfgrid[0]-indfmask[0][0])**2+(mtfgrid[1]-indfmask[1][0])**2)
	
	fmask=np.zeros(imini.shape)
	fmask[np.where(rmask<1*beam_ratio)]=1

	sin=dmsin(0.04,freq_loop[i],pa_loop[i])
	#sin=dmsin(0.1,freq_loop[i],pa_loop[i])
	time.sleep(tsleep)
	ims=stack(nstack)
	cos=dmcos(0.04,freq_loop[i],pa_loop[i])
	#cos=dmcos(0.1,freq_loop[i],pa_loop[i])
	time.sleep(tsleep)
	imc=stack(nstack)

	fourierarr[i]=sin.flatten()
	fourierarr[i+len(freq_loop)]=cos.flatten()

	refvec[i]=scc_imin(ims-imf,fmask=fmask)
	refvec[i+len(freq_loop)]=scc_imin(imc-imf,fmask=fmask)
	print(i/len(freq_loop))

applydmc(bestflat)
IM=np.dot(refvec,refvec.T)
#the following code will help determine the optimal SVD cutoff, looking at what SVD cutoff will best reconstruct individual modes
'''
plt.figure()
def pc(rcond,i):
	#rcond=1e-3
	IMinv=np.linalg.pinv(IM,rcond=rcond)
	cmd_mtx=np.dot(IMinv,refvec)
	#i=100
	sin=dmsin(0.04,freq_loop[i],pa_loop[i])
	time.sleep(0.1)
	imin=stack(100)
	tar=scc_imin(imin-imf)
	coeffs=np.dot(cmd_mtx,tar)
	plt.plot(coeffs)
'''

rcond=1e-4
IMinv=np.linalg.pinv(IM,rcond=rcond)
cmd_mtx=np.dot(IMinv,refvec)

numiter=40
gain=0.5
leak=1
applydmc(bestflat)
time.sleep(tsleep)
for nit in range(numiter):
	imin=stack(10000) #larger number of stacks increases the amount by which you can gain...
	tar=scc_imin(imin)
	coeffs=np.dot(cmd_mtx,tar)
	cmd=np.dot(fourierarr.T,-coeffs).reshape(dmcini.shape).astype(float32)
	applydmc(leak*getdmc()+cmd*gain)
	time.sleep(tsleep)

dmc_dh=getdmc()

np.save('dmc_dh.npy',dmc_dh)