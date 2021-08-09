# authored by Benjamin Gerard and Aditya Sengupta

import numpy as np
from matplotlib import pyplot as plt
import time

from ..utils import joindata
from optics.tt import get_expt, set_expt, stack
from optics.tt import getdmc, applydmc, tip, tilt, remove_piston
from optics.tt import imini, mtf, median_filter

expt_init = get_expt()
set_expt(1e-4)

bestflat = getdmc()

#apply tip/tilt starting only from the bestflat point (start here if realigning the non-coronagraphic PSF) 
def applytiptilt(amptip,amptilt,bestflat=bestflat): #amp is the P2V in DM units
	dmctip=amptip*tip
	dmctilt=amptilt*tilt
	dmctiptilt=remove_piston(dmctip)+remove_piston(dmctilt)+remove_piston(bestflat)+0.5 #combining tip, tilt, and best flat, setting mean piston to 0.5
	#applydmc(aperture*dmctiptilt)
	applydmc(dmctiptilt)

#make MTF side lobe mask
xsidemaskcen,ysidemaskcen=240.7,161.0 #x and y location of the side lobe mask in the cropped image
sidemaskrad=26.8 #radius of the side lobe mask
mtfgrid=np.mgrid[0:imini.shape[0],0:imini.shape[1]]
sidemaskrho=np.sqrt((mtfgrid[0]-ysidemaskcen)**2+(mtfgrid[1]-xsidemaskcen)**2)
sidemask=np.zeros(imini.shape)
sidemaskind=np.where(sidemaskrho<sidemaskrad)
sidemask[sidemaskind]=1

#side lobe mask where there is no signal to measure SNR
xnoise,ynoise=161.66,252.22
sidemaskrhon=np.sqrt((mtfgrid[0]-ynoise)**2+(mtfgrid[1]-xnoise)**2)
sidemaskn=np.zeros(imini.shape)
sidemaskindn=np.where(sidemaskrhon<sidemaskrad)
sidemaskn[sidemaskindn]=1

def processimabs(imin,mask): #process SCC image, isolating the sidelobe in the FFT and IFFT back to the image
	otf=np.fft.fftshift(np.fft.fft2(imin,norm='ortho')) #(1) FFT the image
	otf_masked=otf*mask #(2) multiply by binary mask to isolate side lobe or noise mask
	Iminus=np.fft.ifft2(otf_masked,norm='ortho') #(3) IFFT back to the image plane, now generating a complex-valued image
	return np.abs(Iminus)

#tsleep=0.005 #on really good days
tsleep=0.01 # optimized from tt_opt.optt
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

set_expt(expt_init)

bestflat = getdmc()
np.save(joindata("bestflats/bestflat.npy"), bestflat)
print("Saved best flat")
