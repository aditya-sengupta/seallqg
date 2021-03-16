'''
optimize tip/tilt offset based on fringe visibility

so far, manual steering by eye seems better than the optimal determined value
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

imini=getim()

ygrid,xgrid=grid[0]-ydim/2,grid[1]-xdim/2
xy=np.sqrt(ygrid**2+xgrid**2)

tip,tilt=(ygrid+ydim/2)/ydim,(xgrid+xdim/2)/xdim #min value is zero, max is one
ntip,ntilt=-1*(tip-0.5)+0.5,-1*(tilt-0.5)+0.5 #arrays for negative coefficients

def applytiptilt(amptip,amptilt,bestflat=bestflat): #apply tip; amp is the P2V in DM units
	if amptip<0:
		dmctip=amptip*ntip
	else:
		dmctip=amptip*tip
	if amptilt<0:
		dmctilt=amptilt*ntilt
	else: 
		dmctilt=amptilt*tilt
	applydmc(dmctip+dmctilt+bestflat)

#make MTF side lobe mask
xsidemaskcen,ysidemaskcen=161.66,252.22 #x and y location of the side lobe mask in the cropped image
sidemaskrad=26.8 #radius of the side lobe mask
mtfgrid=np.mgrid[0:imini.shape[0],0:imini.shape[1]]
sidemaskrho=np.sqrt((mtfgrid[0]-ysidemaskcen)**2+(mtfgrid[1]-xsidemaskcen)**2)
sidemask=np.zeros(imini.shape)
sidemaskind=np.where(sidemaskrho<sidemaskrad)
sidemask[sidemaskind]=1

#side lobe mask where there is no signal to measure SNR
xnoise,ynoise=54,252.22
sidemaskrhon=np.sqrt((mtfgrid[0]-ynoise)**2+(mtfgrid[1]-xnoise)**2)
sidemaskn=np.zeros(imini.shape)
sidemaskindn=np.where(sidemaskrhon<sidemaskrad)
sidemaskn[sidemaskindn]=1

def processim(imin,mask): #process SCC image, isolating the sidelobe in the FFT and IFFT back to the image
	otf=np.fft.fftshift(np.fft.fft2(imin,norm='ortho')) #(1) FFT the image
	otf_masked=otf*mask #(2) multiply by binary mask to isolate side lobe or noise mask
	Iminus=np.fft.ifft2(otf_masked,norm='ortho') #(3) IFFT back to the image plane, now generating a complex-valued image
	return np.abs(Iminus)

#grid tip/tilt search 
namp=20
amparr=np.linspace(-1,1,namp)
ttoptarr=np.zeros((namp,namp))
for i in range(namp):
	for j in range(namp):
		applytiptilt(amparr[i],amparr[j])
		#time.sleep(1)
		imopt=stack(100)
		imsig=processim(imopt,sidemask)
		imnoise=processim(imopt,sidemaskn)
		ttoptarr[i,j]=np.sum(imsig)/np.sum(imnoise)

indopttip,indopttilt=np.where(ttoptarr==np.max(ttoptarr))
indopttip,indopttilt=indopttip[0],indopttilt[0]
applytiptilt(amparr[indopttip],amparr[indopttilt])

ampdiff=amparr[2]-amparr[0] #how many discretized points to zoom in to from the previous iteration
tipamparr=np.linspace(amparr[indopttip]-ampdiff,amparr[indopttip]+ampdiff,namp)
tiltamparr=np.linspace(amparr[indopttilt]-ampdiff,amparr[indopttilt]+ampdiff,namp)
ttoptarr1=np.zeros((namp,namp))
for i in range(namp):
	for j in range(namp):
		applytiptilt(tipamparr[i],tiltamparr[j])
		#time.sleep(1)
		imopt=stack(100)
		imsig=processim(imopt,sidemask)
		imnoise=processim(imopt,sidemaskn)
		ttoptarr1[i,j]=np.sum(imsig)/np.sum(imnoise)

indopttip1,indopttilt1=np.where(ttoptarr1==np.max(ttoptarr1))
applytiptilt(tipamparr[indopttip1][0],tiltamparr[indopttilt1][0])


#apply DM Fourier modes
#DM aperture:
rho,phi=functions.polar_grid(xdim,xdim)
aperture=np.zeros(rho.shape).astype(float32)
indap=np.where(rho>0)
aperture[indap]=1
ycen,xcen=ydim/2-0.5,xdim/2-0.5
rgrid=lambda pa:(grid[0]-ycen)/grid[0][-1,0]*np.cos(pa*np.pi/180)+(grid[1]-xcen)/grid[0][-1,0]*np.sin(pa*np.pi/180)
def dmsin(amp,bestflat=bestflat): #generate sine wave
	freq,pa=2,0
	sin1=amp*0.5*np.sin(2*np.pi*freq*rgrid(pa))
	sindm1=sin1.astype(float32)
	freq,pa=2,90
	sin2=amp*0.5*np.sin(2*np.pi*freq*rgrid(pa))
	sindm2=sin2.astype(float32)
	freq,pa=2,45
	sin3=amp*0.5*np.sin(2*np.pi*freq*rgrid(pa))
	sindm3=sin3.astype(float32)
	freq,pa=2,135
	sin4=amp*0.5*np.sin(2*np.pi*freq*rgrid(pa))
	sindm4=sin4.astype(float32)

	dmc=bestflat+sindm1+sindm2+sindm3+sindm4
	applydmc(dmc*aperture)




def vim(tsleep): #if I want to close the tip/tilt loop, take a look at what 
	applydmc(bestflat)
	time.sleep(tsleep)
	imref=cropim(getim())
	applytiptilt(-0.1,0)
	time.sleep(tsleep)
	imtip=cropim(getim())
	#applytiptilt(0,0.1)
	#time.sleep(tsleep)
	#imtilt=cropim(getim())
	applydmc(bestflat)
	ds9.view(imtip-imref)