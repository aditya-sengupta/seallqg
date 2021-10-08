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
from .tt import applytip, applytilt, tip, tilt

def align_alpao_fast(manual=True, view=True):
	expt_init = optics.get_expt()
	optics.set_expt(1e-4)
	time.sleep(5)

	bestflat = np.load(joindata(path.join("bestflats", "bestflat_{0}_{1}.npy".format(optics.name, optics.dmdims[0]))))
	optics.applydmc(bestflat)
	ydim,xdim = optics.dmdims
	grid=np.mgrid[0:ydim,0:xdim].astype(np.float32)

	#DM aperture;
	xy=np.sqrt((grid[0]-ydim/2+0.5)**2+(grid[1]-xdim/2+0.5)**2)
	aperture=np.zeros(optics.dmdims).astype(np.float32)
	aperture[np.where(xy<ydim/2)]=1 
	indap=np.where(aperture==1)
	indnap=np.where(aperture==0)
	inddmuse=np.where(aperture.flatten()==1)[0]
	nact=len(inddmuse)

	remove_piston = lambda dmc: dmc-np.mean(dmc[indap]) #function to remove piston from dm command to have zero mean

	tip,tilt=((grid[0]-ydim/2+0.5)/ydim*2).astype(np.float32),((grid[1]-xdim/2+0.5)/ydim*2).astype(np.float32)# DM tip/tilt 

	if manual:
		steer = -1
		while steer != 0:
			steer = int(input("Input manual steering command: 0 to continue, 1 for tip, 2 for tilt \n"))
			if steer == 1 or steer == 2:
				if steer == 1:
					func = applytip
				else:
					func = applytilt
				amp = float(input("Input amplitude: "))
				func(amp)
		
	#MANUALLY USE ABOVE FUNCTIONS TO STEER THE PSF BACK ONTO THE FPM AS NEEDED, then:
	bestflat = optics.getdmc()

	#apply tip/tilt starting only from the bestflat point (start here if realigning the non-coronagraphic PSF) 
	def applytiptilt(amptip,amptilt,bestflat=bestflat): #amp is the P2V in DM units
		dmctip=amptip*tip
		dmctilt=amptilt*tilt
		dmctiptilt=remove_piston(dmctip)+remove_piston(dmctilt)+remove_piston(bestflat) #combining tip, tilt, and best flat
		#applydmc(aperture*dmctiptilt)
		optics.applydmc(dmctiptilt)

	def optt(tsleep): #function to optimize how long to wait in between applying DM command and recording image
		applytiptilt(-0.001,-0.001)
		time.sleep(tsleep)
		im1=optics.stackim(10)
		optics.applydmc(bestflat)
		time.sleep(tsleep)
		imf=optics.stackim(10)
		#ds9.view(im1-imf)
	#tsleep=0.005 #on really good days
	tsleep=0.02 #optimized from above function
	#tsleep=0.4 #on bad days

	cenmaskrho=np.sqrt((mtfgrid[0]-mtfgrid[0].shape[0]/2)**2+(mtfgrid[1]-mtfgrid[0].shape[0]/2)**2) #radial grid for central MTF lobe
	cenmask=np.zeros(optics.imdims)
	cenmaskradmax,cenmaskradmin=49,10 #mask radii for central lobe, ignoring central part where the pinhole PSF is (if not ignored, this would bias the alignment algorithm)   
	cenmaskind=np.where(np.logical_and(cenmaskrho<cenmaskradmax,cenmaskrho>cenmaskradmin))
	cenmask[cenmaskind]=1

	#grid tip/tilt search 
	namp=10
	amparr=np.linspace(-0.005,0.005,namp) #note the range of this grid search is can be small, assuming day to day drifts are minimal and so you don't need to search far from the previous day to find the new optimal alignment; for larger offsets the range may need to be increases (manimum search range is -1 to 1); but, without spanning the full -1 to 1 range this requires manual tuning of the limits to ensure that the minimum is not at the edge
	ttoptarr=np.zeros((namp,namp))
	for i in range(namp):
		for j in range(namp):
			applytiptilt(amparr[i],amparr[j])
			time.sleep(tsleep)
			imopt=optics.stackim(10)
			mtfopt=mtf(imopt)
			sidefraction=np.sum(mtfopt[sidemaskind])/np.sum(mtfopt)
			cenfraction=np.sum(mtfopt[cenmaskind])/np.sum(mtfopt)
			ttoptarr[i,j]=sidefraction+0.1/cenfraction #the factor of 0.01 is a relative weight; because we only expect the fringe visibility to max out at a few %, this attempts to give equal weight to both terms 

	medttoptarr=median_filter(ttoptarr,3) #smooth out hot pizels, attenuating noise issues
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
			imopt=optics.stackim(10)
			mtfopt=mtf(imopt)
			sidefraction=np.sum(mtfopt[sidemaskind])/np.sum(mtfopt)
			cenfraction=np.sum(mtfopt[cenmaskind])/np.sum(mtfopt)
			ttoptarr1[i,j]=sidefraction+0.1/cenfraction 

	medttoptarr1=median_filter(ttoptarr1,3) #smooth out hot pizels, attenuating noise issues
	indopttip1,indopttilt1=np.where(medttoptarr1==np.max(medttoptarr1))
	applytiptilt(tipamparr[indopttip1][0],tiltamparr[indopttilt1][0])

	if view:
		plt.imshow(medttoptarr1)
		plt.show()

	bestflat = optics.getdmc()
	im_bestflat = optics.stackim(100)

	dt = datetime.now().strftime("%d_%m_%Y_%H")
	np.save(joindata(path.join("bestflats", "bestflat_{0}_{1}.npy".format(optics.name, optics.dmdims[0]))), bestflat)
	np.save(joindata(path.join("bestflats", "bestflat_{0}_{1}_{2}.npy".format(optics.name, optics.dmdims[0], dt))), bestflat)
	np.save(joindata(path.join("bestflats", "im_{0}_{1}_{2}.npy".format(optics.name, optics.dmdims[0], dt))), im_bestflat)
	print("Saved best flat")

	optics.set_expt(1e-5) #set exposure time to avoid saturation of non-coronagraphic PSF
	time.sleep(5)

	dmcs = []
	ims = []
	for (amptip, amptilt) in [(0.03, 0), (-0.03, 0), (0, 0.03), (0, -0.03)]:
		applytiptilt(amptip, amptilt)
		dmcs.append(optics.getdmc())
		time.sleep(tsleep)
		ims.append(optics.stackim(10))
		optics.applydmc(bestflat)

	inds = [np.where(im == np.max(im)) for im in ims]

	def genv(arr, i): #return zeroth array element if multiple elements, or single element if not
		if len(arr[0]) > 1:
			return int(arr[i][0])
		return int(arr[i])

	y = [genv(inds[i], 0) for i in range(4)]
	x = [genv(inds[i], 1) for i in range(4)]
	ymean=np.mean(y)
	xmean=np.mean(x)

	np.save(joindata(path.join("bestflats", "imcen.npy")), np.array([xmean,ymean]))

	optics.set_expt(expt_init)

	#calibrate DM units to lambda/D
	beam_ratio=np.load(joindata(path.join("bestflats", "beam_ratio.npy"))) #beam ratio is from the old system, but it is not too consequential though if it is off...
	cal1=np.sqrt((y[0]-y[1])**2+(x[0]-x[1])**2)/(np.max(aperture*(dmcs[0]-dmcs[1]))-np.min(aperture*(dmcs[0]-dmcs[1])))/beam_ratio
	cal2=np.sqrt((y[2]-y[3])**2+(x[2]-x[3])**2)/(np.max(aperture*(dmcs[2]-dmcs[3]))-np.min(aperture*(dmcs[2]-dmcs[3])))/beam_ratio

	dmc2wfe = (cal1 + cal2) * 0.633 / 2 #should be dm commands in volts to WFE in microns, but this is ~10x larger than the calibration I did by poking 1 actuator...?
	np.save(joindata(path.join("bestflats", "lodmc2wfe.npy")), dmc2wfe)
