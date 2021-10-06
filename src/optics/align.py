# authored by Benjamin Gerard and Aditya Sengupta

import numpy as np
import time
from datetime import datetime
from matplotlib import pyplot as plt
from os import path
# import pysao
from scipy.ndimage.filters import median_filter

from ..utils import joindata
from .image import optics
from .tt import mtfgrid, sidemaskrad, sidemaskind
from .ao import mtf, remove_piston
from .tt import applytiptilt, tip, tilt

def align_fast(view=True):
	expt_init = optics.get_expt()
	optics.set_expt(1e-4)

	bestflat = np.load(joindata(os.path.join("bestflats", "bestflat_{0}_{1}.npy".format(optics.name, optics.dmdims[0]))))

	#side lobe mask where there is no signal to measure SNR
	xnoise,ynoise=161.66,252.22
	sidemaskrhon=np.sqrt((mtfgrid[0]-ynoise)**2+(mtfgrid[1]-xnoise)**2)
	sidemaskn=np.zeros(optics.imdims)
	sidemaskindn=np.where(sidemaskrhon<sidemaskrad)
	sidemaskn[sidemaskindn]=1

	def optt(tsleep): #function to optimize how long to wait in between applying DM command and recording image
		applytiptilt(-0.001,-0.001)
		time.sleep(tsleep)
		im1=stack(10)
		applylodmc(bestflat)
		time.sleep(tsleep)
		imf=stack(10)
		ds9.view(im1-imf)
	#tsleep=0.005 #on really good days
	tsleep=0.02 #optimized from above function
	#tsleep=0.4 #on bad days


	cenmaskrho=np.sqrt((mtfgrid[0]-mtfgrid[0].shape[0]/2)**2+(mtfgrid[1]-mtfgrid[0].shape[0]/2)**2) #radial grid for central MTF lobe
	cenmask = np.zeros(optics.imdims)
	cenmaskradmax,cenmaskradmin=49,10 #mask radii for central lobe, ignoring central part where the pinhole PSF is (if not ignored, this would bias the alignment algorithm)   
	cenmaskind=np.where(np.logical_and(cenmaskrho<cenmaskradmax,cenmaskrho>cenmaskradmin))
	cenmask[cenmaskind]=1

	#grid tip/tilt search 
	namp = 10
	amparr = np.linspace(-0.005, 0.005, namp) 
	# note the range of this grid search is can be small, 
	# assuming day to day drifts are minimal and so you don't need to search far from the previous day 
	# to find the new optimal alignment; for larger offsets the range may need to be increases 
	# (manimum search range is -1 to 1); but, without spanning the full -1 to 1 range 
	# this requires manual tuning of the limits to ensure that the minimum is not at the edge
	ttoptarr = np.zeros((namp, namp))
	for i in range(namp):
		for j in range(namp):
			applytiptilt(amparr[i],amparr[j])
			time.sleep(tsleep)
			imopt = optics.stack(10)
			mtfopt=mtf(imopt)
			sidefraction=np.sum(mtfopt[sidemaskind])/np.sum(mtfopt)
			cenfraction=np.sum(mtfopt[cenmaskind])/np.sum(mtfopt)
			ttoptarr[i,j]=sidefraction+0.1/cenfraction 
			# the factor of 0.01 is a relative weight; because we only expect the fringe visibility to max out at 1%, 
			# this attempts to give equal weight to both terms 

	medttoptarr=median_filter(ttoptarr,3) #smooth out hot pixels, attenuating noise issues
	indopttip,indopttilt=np.where(medttoptarr==np.max(medttoptarr))
	indopttip,indopttilt=indopttip[0],indopttilt[0]
	applytiptilt(amparr[indopttip],amparr[indopttilt])

	if view:
		plt.imshow(medttoptarr)
		plt.show()

	ampdiff=amparr[2]-amparr[0] #how many discretized points to zoom in to from the previous iteration
	tipamparr=np.linspace(amparr[indopttip]-ampdiff,amparr[indopttip]+ampdiff,namp)
	tiltamparr=np.linspace(amparr[indopttilt]-ampdiff,amparr[indopttilt]+ampdiff,namp)
	ttoptarr1=np.zeros((namp,namp))
	for i in range(namp):
		for j in range(namp):
			applytiptilt(tipamparr[i],tiltamparr[j])
			time.sleep(tsleep)
			imopt = optics.stack(10)
			mtfopt=mtf(imopt)
			sidefraction=np.sum(mtfopt[sidemaskind])/np.sum(mtfopt)
			cenfraction=np.sum(mtfopt[cenmaskind])/np.sum(mtfopt)
			ttoptarr1[i,j]=sidefraction+0.1/cenfraction 

	medttoptarr1 = median_filter(ttoptarr1,3) #smooth out hot pixels, attenuating noise issues
	indopttip1,indopttilt1=np.where(medttoptarr1==np.max(medttoptarr1))
	applytiptilt(amparr[indopttip],amparr[indopttilt])

	

	optics.set_expt(expt_init)
	dt = datetime.now().strftime("%d_%m_%Y_%H")
	np.save(joindata(path.join("bestflats", "bestflat_{0}_{1}.npy".format(optics.name, optics.dmdims[0]))))
	np.save(joindata(path.join("bestflats", "bestflat_{0}_{1}_{2}.npy".format(optics.name, optics.dmdims[0], dt))))
	np.save(joindata(path.join("bestflats", "im_{0}_{1}_{2}.npy".format(optics.name, optics.dmdims[0], dt))))
	print("Saved best flat")

	expt(1e-5) #set exposure time to avoid saturation of non-coronagraphic PSF
time.sleep(5)

dmcs = []
ims = []
for (tip, tilt) in [(0.03, 0), (-0.03, 0), (0, 0.03), (0, -0.03)]:
	applytiptilt(tip, tilt, bestflat=bestflat)
	dmcs.append(optics.getdmc())
	time.sleep(tsleep)
	ims.append(optics.stackim(10))
	optics.applydmc(bestflat)

inds = [np.where(im == np.max(im)) for im in ims]

def genvy(arr): #return zeroth array element if multiple elements, or single element if not
	if len(arr[0])>1:
		return arr[0][0]
	else:
		return arr[0]

def genvx(arr): #return zeroth array element if multiple elements, or single element if not
	if len(arr[1])>1:
		return arr[1][0]
	else:
		return arr[1]

y1,y2,y3,y4 = (genvy(inds[i]) for i in range(4))
x1,x2,x3,x4 = (genvx(inds[i]) for i in range(4))
ymean=np.mean(np.array([y1,y2,y3,y4]))
xmean=np.mean(np.array([x1,x2,x3,x4]))
np.save(joindata(path.join("bestflats", "imcen.npy")), np.array([xmean,ymean]))

optics.set_expt(1e-3)

#calibrate DM units to lambda/D
beam_ratio=np.load(joindata(path.join("bestflats", "beam_ratio.npy"))) #beam ratio is from the old system, but it is not too consequential though if it is off...
cal1=np.sqrt((y1-y2)**2+(x1-x2)**2)/(np.max(aperture*(dm1-dm2))-np.min(aperture*(dm1-dm2)))/beam_ratio
cal2=np.sqrt((y3-y4)**2+(x3-x4)**2)/(np.max(aperture*(dm3-dm4))-np.min(aperture*(dm3-dm4)))/beam_ratio

dmc2wfe=np.mean(np.array([cal1,cal2])*0.633) #should be dm commands in volts to WFE in microns, but this is ~10x larger than the calibration I did by poking 1 actuator...?
np.save(joindata(path.join("bestflats", "lodmc2wfe.npy")), dmc2wfe[0])
