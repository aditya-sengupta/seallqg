'''
generate SCC interaction matrix of Fourier modes
'''


from ancillary_code import *
import numpy as np
import time
import functions
import itertools

dmcini=getdmc()
ydim,xdim=dmcini.shape
grid=np.mgrid[0:ydim,0:xdim].astype(float32)
bestflat=np.load('bestflat.npy')
applydmc(bestflat)

ygrid,xgrid=grid[0]-ydim/2,grid[1]-xdim/2
xy=np.sqrt(ygrid**2+xgrid**2)

imini=getim()

beam_ratio=0.635*750/10.8/6.5 #theoretical number of pixels/resel: lambda (in microns)*focal length to camera/Lyot stop beam diameter/Andor pixel size (in microns) 

#DM aperture:
rho,phi=functions.polar_grid(xdim,xdim)
aperture=np.zeros(rho.shape).astype(float32)
indap=np.where(rho>0)
aperture[indap]=1

#apply DM Fourier modes
ycen,xcen=ydim/2-0.5,xdim/2-0.5
rgrid=lambda pa:(grid[0]-ycen)/grid[0][-1,0]*np.cos(pa*np.pi/180)+(grid[1]-xcen)/grid[0][-1,0]*np.sin(pa*np.pi/180)
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

#image cropping from full frame to ROI
#cropcen=[971,1119]
#cropsize=[160,160]
#cropim = lambda im: im[cropcen[0]-yimcen:cropcen[0]+yimcen,cropcen[1]-ximcen:cropcen[1]+ximcen]-np.median(im) #rudamentary bias subtraction is done here during the image cropping, assuming the image is large enough to not be affected by actual signal

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

Im_rho=np.sqrt((mtfgrid[0]-yimcen)**2+(mtfgrid[1]-ximcen)**2)
dh_mask=np.zeros(imini.shape)
maxld,minld=10,6 #maximum and minimum radius from which to dig a dark hole (sqrt(2) times larger for the maximum and the DH corners)
ylimld=5 #how much to go +/- in y for the DH
#ind_dh=np.where(np.logical_and(np.logical_and(Im_rho<maxld*beam_ratio,Im_rho>minld*beam_ratio),mtfgrid[1]-ximcen<0))
#ind_dh=np.where(dh_mask==0)
ind_dh=np.where(np.logical_and(np.logical_and(mtfgrid[1]-ximcen<-(minld-2)*beam_ratio,mtfgrid[1]-ximcen>-(maxld+2)*beam_ratio),np.logical_and(mtfgrid[0]-yimcen<(ylimld+2)*beam_ratio,mtfgrid[0]-yimcen>-(ylimld+2)*beam_ratio))) #note the +2 padding, which accounts for the unknown beam ratio in not cutting off the fmask later in the IM code at the edges of the DH
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
allx,ally=list(zip(*itertools.product(np.linspace(ximcen-maxld*beam_ratio+0.5*beam_ratio,yimcen-0.5*beam_ratio,maxld),np.linspace(yimcen-maxld*beam_ratio+0.5*beam_ratio,ximcen+maxld*beam_ratio-0.5*beam_ratio,2*maxld))))
loopx,loopy=np.array(list(allx)).astype(float),np.array(list(ally)).astype(float)

freq_loop=np.sqrt((loopy-yimcen)**2+(loopx-ximcen)**2)/beam_ratio #sine wave frequency for (lambda/D)**2 region w/in DH
pa_loop=90-180/np.pi*np.arctan2(loopy-yimcen,loopx-ximcen) #position angle of sine wave for (lambda/D)**2 region w/in DH

#only use desired spatial frequencies within the dark hole as input into the control matrix
indloop=np.where(np.logical_and(np.logical_and(loopy-yimcen<ylimld*beam_ratio,loopy-yimcen>-ylimld*beam_ratio),np.logical_and(loopx-ximcen<-minld*beam_ratio,loopy-ximcen>-maxld*beam_ratio)))
freq_loop=freq_loop[indloop]
pa_loop=pa_loop[indloop]

#looks like I need to wait 3 seconds (full frame) or 0.05 seconds (320x320 ROI) in between each DM command to generate a stable image that isn't influenced by the previous command
def optt(tsleep): #try to optimize how long to wait in between applying DM command and recording image
	i=0
	dmsin(0.2,freq_loop[i],pa_loop[i])
	time.sleep(tsleep)
	im1=stack(10)
	applydmc(bestflat)
	time.sleep(tsleep)
	imf=stack(10)
	ds9.view(im1-imf)

tsleep=0.05
applydmc(bestflat)
time.sleep(tsleep)
imf=stack(100)

fourierarr=np.zeros((len(freq_loop)*2,dmcini.flatten().shape[0]))
refvec=np.zeros((len(freq_loop)*2,len(dh_mask[ind_dh])*2))
for i in range(len(freq_loop)):
	'''
	if freq_loop[i]<minld or freq_loop[i]>maxld: #leave spatial frequencies too close to the FPM IWA or outside the OWA set to zero
		continue
	else:
	'''
	cos=dmcos(0.12,freq_loop[i],pa_loop[i])
	time.sleep(tsleep)
	imcb=stack(100)

	diffimuf=np.abs(processim(imcb,mask=cenmask))-np.abs(processim(imf,mask=cenmask)) #unfringed differential image
	indfmask=np.where(diffimuf==np.max(diffimuf[ind_dh]))
	rmask=np.sqrt((mtfgrid[0]-indfmask[0][0])**2+(mtfgrid[1]-indfmask[1][0])**2)
	fmask=np.zeros(diffimuf.shape)
	fmask[np.where(rmask<1*beam_ratio)]=1

	sin=dmsin(0.04,freq_loop[i],pa_loop[i])
	time.sleep(tsleep)
	ims=stack(100)
	cos=dmcos(0.04,freq_loop[i],pa_loop[i])
	time.sleep(tsleep)
	imc=stack(100)

	fourierarr[i]=sin.flatten()
	fourierarr[i+len(freq_loop)]=cos.flatten()

	refvec[i]=scc_imin(ims-imf,fmask=fmask)
	refvec[i+len(freq_loop)]=scc_imin(imc-imf,fmask=fmask)
	print(i/len(freq_loop))

applydmc(bestflat)
IM=np.dot(refvec,refvec.T)

rcond=1e-4
IMinv=np.linalg.pinv(IM,rcond=rcond)
cmd_mtx=np.dot(IMinv,refvec)

numiter=40
gain=0.5
applydmc(bestflat)
time.sleep(tsleep)
for nit in range(numiter):
	imin=stack(1000)
	tar=scc_imin(imin)
	coeffs=np.dot(cmd_mtx,tar)
	cmd=np.dot(fourierarr.T,-coeffs).reshape(dmcini.shape).astype(float32)
	applydmc(getdmc()+cmd*gain)
	time.sleep(tsleep)

np.save('dmc_dh.npy',getdmc())