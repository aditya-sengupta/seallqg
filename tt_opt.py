'''
optimize tip/tilt offset based on fringe visibility

so far, manual steering by eye seems better than the optimal determined value
'''
from ancillary_code import *
import numpy as np
import time
import functions
import itertools
from scipy.ndimage.filters import median_filter

dmcini=getdmc()
ydim,xdim=dmcini.shape
grid=np.mgrid[0:ydim,0:xdim].astype(float32)
bestflat=np.load('bestflat_zopt.npy') #if running code after running zern_opt.py (i.e., non-coronagraphic PSF)
#bestflat=np.load('bestflat.npy') #if running code to realign coronagraphic PSF
applydmc(bestflat)

imini=getim() #Andor image just for referencing dimensions

#DM aperture:
rho,phi=functions.polar_grid(xdim,xdim*29/32) #assuming 29 of the 32 actuators are illuminated
aperture=np.zeros(rho.shape).astype(float32)
indap=np.where(rho>0)
aperture[indap]=1
remove_piston = lambda dmc: dmc-np.mean(dmc[indap]) #function to remove piston from dm command to have zero mean (must be intermediate since DM commands can't have negative values)

#tip/tilt grid:
ygrid,xgrid=grid[0]-ydim/2,grid[1]-xdim/2
xy=np.sqrt(ygrid**2+xgrid**2)
tip,tilt=(ygrid+ydim/2)/ydim,(xgrid+xdim/2)/xdim #min value is zero, max is one
def applytiptilt(amptip,amptilt,bestflat=bestflat): #apply tip; amp is the P2V in DM units
	dmctip=amptip*tip
	dmctilt=amptilt*tilt
	dmctiptilt=remove_piston(dmctip)+remove_piston(dmctilt)+remove_piston(bestflat)+0.5 #combining tip, tilt, and best flat, setting mean piston to 0.5
	applydmc(aperture*dmctiptilt)

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

def processimabs(imin,mask): #process SCC image, isolating the sidelobe in the FFT and IFFT back to the image
	otf=np.fft.fftshift(np.fft.fft2(imin,norm='ortho')) #(1) FFT the image
	otf_masked=otf*mask #(2) multiply by binary mask to isolate side lobe or noise mask
	Iminus=np.fft.ifft2(otf_masked,norm='ortho') #(3) IFFT back to the image plane, now generating a complex-valued image
	return np.abs(Iminus)

#grid tip/tilt search 
tsleep=0.05 #time to sleep in between applying DM commands and grabbing images, determined in optt of genIM.py

cenmaskrho=np.sqrt((mtfgrid[0]-mtfgrid[0].shape[0]/2)**2+(mtfgrid[1]-mtfgrid[0].shape[0]/2)**2) #radial grid for central MTF lobe
cenmask=np.zeros(imini.shape)
cenmaskradmax,cenmaskradmin=49,10 #mask radii for central lobe, ignoring central part where the pinhole PSF is (if not ignored, this would bias the alignment algorithm)   
cenmaskind=np.where(np.logical_and(cenmaskrho<cenmaskradmax,cenmaskrho>cenmaskradmin))
cenmask[cenmaskind]=1

namp=20
amparr=np.linspace(-1,1,namp)
ttoptarr=np.zeros((namp,namp))
for i in range(namp):
	for j in range(namp):
		applytiptilt(amparr[i],amparr[j])
		time.sleep(tsleep)
		imopt=stack(1000)
		mtfopt=mtf(imopt)
		sidefraction=np.sum(mtfopt[sidemaskind])/np.sum(mtfopt)
		cenfraction=np.sum(mtfopt[cenmaskind])/np.sum(mtfopt)
		ttoptarr[i,j]=1/cenfraction#sidefraction+0.01/cenfraction #the factor of 0.01 is a relative weight; because we only expect the fringe visibility to max out at 1%, this attempts to give equal weight to both terms 

medttoptarr=median_filter(ttoptarr,3) #smooth out hot pizels, attenuating noise issues
indopttip,indopttilt=np.where(medttoptarr==np.max(medttoptarr))
indopttip,indopttilt=indopttip[0],indopttilt[0]
applytiptilt(amparr[indopttip],amparr[indopttilt])

ampdiff=amparr[3]-amparr[0] #how many discretized points to zoom in to from the previous iteration
tipamparr=np.linspace(amparr[indopttip]-ampdiff,amparr[indopttip]+ampdiff,namp)
tiltamparr=np.linspace(amparr[indopttilt]-ampdiff,amparr[indopttilt]+ampdiff,namp)
ttoptarr1=np.zeros((namp,namp))
for i in range(namp):
	for j in range(namp):
		applytiptilt(tipamparr[i],tiltamparr[j])
		time.sleep(tsleep)
		imopt=stack(1000)
		mtfopt=mtf(imopt)
		sidefraction=np.sum(mtfopt[sidemaskind])/np.sum(mtfopt)
		cenfraction=np.sum(mtfopt[cenmaskind])/np.sum(mtfopt)
		ttoptarr1[i,j]=sidefraction+0.01/cenfraction 

medttoptarr1=median_filter(ttoptarr1,3) #smooth out hot pizels, attenuating noise issues
indopttip1,indopttilt1=np.where(medttoptarr1==np.max(medttoptarr1))
applytiptilt(tipamparr[indopttip1][0],tiltamparr[indopttilt1][0])

bestflat=getdmc()

#next: test bestflat by placing sine waves and adjusting by eye that the spot intensities look even; I have found that this approach can be misleading, potentially ofsetting the PSF on th FPM due to interference of the spots with the spekles making me think is it not centered when it is, and as a result lowering fringe visibility; for now, I would ignore this manual approach and go with the numerically optimized approach above  
'''
beam_ratio=0.635*750/10.72/6.5 #theoretical number of pixels/resel: lambda (in microns)*focal length to camera (in mm)/Lyot stop beam diameter (in mm)/Andor pixel size (in microns)

#functions to apply DM Fourier modes 
ycen,xcen=ydim/2-0.5,xdim/2-0.5
indrho1=np.where(rho==1)
gridnorm=np.max(grid[0][indrho1])
rgrid=lambda pa:(grid[0]-ycen)/gridnorm*np.cos(pa*np.pi/180)+(grid[1]-xcen)/gridnorm*np.sin(pa*np.pi/180)
def dmsin(amp,freq,pa,bestflat=bestflat): #generate sine wave
	sin=amp*0.5*np.sin(2*np.pi*freq*rgrid(pa))
	sindm=sin.astype(float32)
	dmc=bestflat+sindm
	applydmc(dmc*aperture)
	return sindm

tip,tilt=(ygrid+ydim/2)/ydim,(xgrid+xdim/2)/xdim #min value is zero, max is one
def applytip(amp): #apply tip; amp is the P2V in DM units
	dmc=getdmc()
	dmc=dmc+amp*tip
	applydmc(dmc)

def applytilt(amp): #apply tilt; amp is the P2V in DM units
	dmc=getdmc()
	dmc=dmc+amp*tilt
	applydmc(dmc)

sin1=dmsin(0.1,4,90,bestflat=bestflat)
#MANUALLY: APPLY applytilt(NUMBER) until satisfied
#applytilt(-0.1)
bestflat=getdmc()-sin1
sin2=dmsin(0.1,4,0,bestflat=bestflat)
#MANUALLY: APPLY applytilt(NUMBER) until satisfied
bestflat=getdmc()-sin2
'''
np.save('bestflat.npy',bestflat)

#unfinished code to close the tip/tilt loop:
def vim(tsleep): #if I want to close the tip/tilt loop, take a look at what timesteps in between images are needed 
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