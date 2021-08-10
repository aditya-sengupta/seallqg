# authored by Benjamin Gerard and Aditya Sengupta

import sys
sys.path.append("..")

import numpy as np
import time
from matplotlib import pyplot as plt
import pysao

from utils import joindata
from optics import get_expt, set_expt
from optics import mtfgrid, imini, sidemaskrad, sidemaskind, mtf, median_filter
from optics import applytiptilt, applydmc
from optics import stack, bestflat

expt_init = get_expt()
set_expt(1e-4)

#side lobe mask where there is no signal to measure SNR
xnoise,ynoise=161.66,252.22
sidemaskrhon=np.sqrt((mtfgrid[0]-ynoise)**2+(mtfgrid[1]-xnoise)**2)
sidemaskn=np.zeros(imini.shape)
sidemaskindn=np.where(sidemaskrhon<sidemaskrad)
sidemaskn[sidemaskindn]=1

def processimabs(imin,mask): #process SCC image, isolating the sidelobe in the FFT and IFFT back to the image
	otf = np.fft.fftshift(np.fft.fft2(imin,norm='ortho')) #(1) FFT the image
	otf_masked = otf * mask #(2) multiply by binary mask to isolate side lobe or noise mask
	Iminus = np.fft.ifft2(otf_masked,norm='ortho') #(3) IFFT back to the image plane, now generating a complex-valued image
	return np.abs(Iminus)

def optt(tsleep): #function to optimize how long to wait in between applying DM command and recording image
	ds9 = pysao.ds9()
	applytiptilt(-0.1,-0.1)
	time.sleep(tsleep)
	im1=stack(10)
	applydmc(bestflat)
	time.sleep(tsleep)
	imf=stack(10)
	ds9.view(im1-imf)
#tsleep=0.005 #on really good days
tsleep=0.01 #optimized from above function
#tsleep=0.4 #on bad days


cenmaskrho=np.sqrt((mtfgrid[0]-mtfgrid[0].shape[0]/2)**2+(mtfgrid[1]-mtfgrid[0].shape[0]/2)**2) #radial grid for central MTF lobe
cenmask=np.zeros(imini.shape)
cenmaskradmax,cenmaskradmin=49,10 #mask radii for central lobe, ignoring central part where the pinhole PSF is (if not ignored, this would bias the alignment algorithm)   
cenmaskind=np.where(np.logical_and(cenmaskrho<cenmaskradmax,cenmaskrho>cenmaskradmin))
cenmask[cenmaskind]=1

#grid tip/tilt search 
namp=10
amparr=np.linspace(-0.1,0.1,namp) #note the range of this grid search is can be small, assuming day to day drifts are minimal and so you don't need to search far from the previous day to find the new optimal alignment; for larger offsets the range may need to be increases (manimum search range is -1 to 1); but, without spanning the full -1 to 1 range this requires manual tuning of the limits to ensure that the minimum is not at the edge
ttoptarr=np.zeros((namp,namp))
for i in range(namp):
	for j in range(namp):
		applytiptilt(amparr[i],amparr[j])
		time.sleep(tsleep)
		imopt=stack(10)
		mtfopt=mtf(imopt)
		sidefraction=np.sum(mtfopt[sidemaskind])/np.sum(mtfopt)
		cenfraction=np.sum(mtfopt[cenmaskind])/np.sum(mtfopt)
		ttoptarr[i,j]=sidefraction+0.1/cenfraction #the factor of 0.01 is a relative weight; because we only expect the fringe visibility to max out at 1%, this attempts to give equal weight to both terms 

medttoptarr=median_filter(ttoptarr,3) #smooth out hot pizels, attenuating noise issues
indopttip,indopttilt=np.where(medttoptarr==np.max(medttoptarr))
indopttip,indopttilt=indopttip[0],indopttilt[0]
applytiptilt(amparr[indopttip],amparr[indopttilt])

def viewmed():
    plt.imshow(medttoptarr)
    plt.show()

#expt(1e-4)

ampdiff=amparr[2]-amparr[0] #how many discretized points to zoom in to from the previous iteration
tipamparr=np.linspace(amparr[indopttip]-ampdiff,amparr[indopttip]+ampdiff,namp)
tiltamparr=np.linspace(amparr[indopttilt]-ampdiff,amparr[indopttilt]+ampdiff,namp)
ttoptarr1=np.zeros((namp,namp))
for i in range(namp):
	for j in range(namp):
		applytiptilt(tipamparr[i],tiltamparr[j])
		time.sleep(tsleep)
		imopt=stack(10)
		mtfopt=mtf(imopt)
		sidefraction=np.sum(mtfopt[sidemaskind])/np.sum(mtfopt)
		cenfraction=np.sum(mtfopt[cenmaskind])/np.sum(mtfopt)
		ttoptarr1[i,j]=sidefraction+0.1/cenfraction 

medttoptarr1=median_filter(ttoptarr1,3) #smooth out hot pixels, attenuating noise issues
indopttip1,indopttilt1=np.where(medttoptarr1==np.max(medttoptarr1))
applytiptilt(tipamparr[indopttip1][0],tiltamparr[indopttilt1][0])

im_bestflat = stack(100)
set_expt(expt_init)
np.save(joindata("bestflats/bestflat.npy"), bestflat)
print("Saved best flat")
