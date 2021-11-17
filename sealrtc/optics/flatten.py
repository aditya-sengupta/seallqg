# authored by Benjamin Gerard and Aditya Sengupta

import numpy as np
import time
from matplotlib import pyplot as plt
# import pysao
from scipy.ndimage.filters import median_filter
from os import path
import tqdm

from ..utils import joindata
from .utils import image_to_pupil, complex_amplitude, pupil_to_image

def flatten(optics):
	expt_init = optics.get_expt()
	optics.set_expt(1e-4)

	bestflat = optics.bestflat
	optics.applydmc(bestflat)
	
	dmcini = optics.getdmc()
	grid = np.mgrid[0:dmcini.shape[0],0:dmcini.shape[1]]
	xy = np.sqrt((grid[0]-dmcini.shape[0]/2+0.5)**2+(grid[1]-dmcini.shape[1]/2+0.5)**2)
	aperture = np.zeros(dmcini.shape).astype(np.dtype(np.float32))
	aperture[np.where(xy<dmcini.shape[0]/2-0)] = 1 
	#my initial thinking here was that the "-0" should be a "-2" 
	# because the two radial edge actuators were not fully illuminated, 
	# showing on the SHWFS that the peak of their illuminated influence functions 
	# was outside the illuminated aperture, so I would just leave them to zero; 
	# however, it was this difference that prevented me from getting a good best flat, 
	# so the full 97 actuator DM actuator should indeed be used!
	inddmuse = np.where(aperture.flatten()==1)[0]
	nact = len(inddmuse)
	xdm,ydm = (inddmuse/dmcini.shape[0]).astype(int), inddmuse % dmcini.shape[1]

	#remove tip/tilt from DM commands
	xdim, ydim = aperture.shape
	tip, tilt = (grid[0]-ydim/2+0.5)/ydim*2, (grid[1]-xdim/2+0.5)/ydim*2
	IMtt = np.array([(tip*aperture).flatten(),(tilt*aperture).flatten()])
	CMtt = np.linalg.pinv(IMtt,rcond=1e-5)
	def rmtt(ph): 
		coeffs=np.dot(np.vstack((ph*aperture).flatten()).T,CMtt) 
		lsqtt=np.dot(IMtt.T,coeffs.T).reshape(tilt.shape).astype(np.dtype(np.float32))
		return ph-lsqtt

	nstack = 100
	wf_ini = optics.getwf()
	wf_ini_arr = np.zeros((nstack,wf_ini.shape[0],wf_ini.shape[1])) #average a sequence of frames to determine the aperture mask, since some frames have fluctuating nan/non-nan values around the edges
	for i in range(nstack):
		wf_ini_arr[i] = optics.getwf()
	wf_ini=np.nanmedian(wf_ini_arr,axis=0)

	wf_ind=np.where(np.isnan(wf_ini)==False) #index for where the pupil is defined
	wfmask=np.zeros(wf_ini.shape)*np.nan
	wfmask[wf_ind]=1
	wf=wf_ini[wf_ind]

	rmpist=lambda wf: wf-np.nanmean(wf[wf_ind]) #remove piston from wfs wavefront

	def poke_act(i,amp=0.1): #poke an actuator
		dmc_poke=np.zeros(dmcini.shape).astype(np.dtype(np.float32))
		x,y=xdm[i],ydm[i]
		dmc_poke[y,x]=amp
		optics.applydmc(aperture*(bestflat+dmc_poke))

	def opttsleep(i,tsleep,amp=0.1): #find how long is needed to pause in between DM commands and WFS measurements, and see how to map which actuators are registering on the WFS pupil
		poke_act(i,amp=amp)
		time.sleep(tsleep)
		wf_push=rmpist(optics.stackwf(10))
		poke_act(i,amp=-amp) 
		time.sleep(tsleep)
		wf_pull=rmpist(optics.stackwf(10))
		out=wfmask*(wf_push-wf_pull)
		#ds9.view(out)

	tsleep=0.1

	def dmc2wf(im): #magnify dm commands to match WFS space
		padval=int(round((wf_ini.shape[0]-dmcini.shape[0])/2))
		fp= lambda im: np.pad(im,pad_width=padval,mode='constant',constant_values=0)
		otf=image_to_pupil(np.flipud(im))
		otfp=complex_amplitude(fp(np.abs(otf)),fp(np.angle(otf)))
		imout=np.abs(pupil_to_image(otfp))**2 #abs**2 is unphysical (should just be abs) but it better removes the ringing from an actuator pokemask
		return imout

	wfgrid=np.mgrid[0:wf_ini.shape[0],0:wf_ini.shape[1]]
	IMamp=0.1 #amplitude to poke each actuator in generating the IM (in DM units)

	slope_ini = optics.getslopes()
	# Push each actuator
	IM=np.zeros((nact,2*wf.shape[0])) #interaction matrix to fill
	for i in tqdm.trange(nact):
		actind=inddmuse[i]
		cmd=bestflat.flatten()
		cmd[actind] = cmd[actind]+IMamp
		optics.applydmc(bestflat+cmd.reshape(dmcini.shape))
		time.sleep(tsleep)
		spush = optics.stackslopes(10)
		cmd[actind] = cmd[actind] -2*IMamp
		optics.applydmc(bestflat+cmd.reshape(dmcini.shape))
		time.sleep(tsleep)
		spull = optics.stackslopes(10)

		out=np.array([rmpist(spush[0])-rmpist(spull[0]),rmpist(spush[1])-rmpist(spull[1])])

		dmimap=dmc2wf(optics.getdmc())
		indact=np.where(dmimap==np.max(dmimap))
		wfrgrid=np.sqrt((wfgrid[0]-indact[0][0]-0.5)**2+(wfgrid[1]-indact[1][0]-0.5)**2) #in reality it should be +0.5 instead of -0.5, but looking at the ds9.view(out*act_mask) images the latter looks better so I'm going with that.
		act_mask=np.zeros(wf_ini.shape)
		act_mask[np.where(np.logical_and(wfrgrid<9,wfmask==1))]=1

		#IM[i]=np.array([(act_mask*out[0])[wf_ind],(act_mask*out[1])[wf_ind]]).flatten()
		IM[i]=np.array([out[0][wf_ind],out[1][wf_ind]]).flatten()

	IM[np.where(np.isnan(IM)==True)]=0 #some supapertures may still be nans, set those to zero

	act_arr=np.vstack(np.zeros(nact).astype(np.dtype(np.float32))+IMamp*2) #array of DM commands
	def vcmd(rcond): #view the reconstructed DM commands, making sure that waffle mode is not propagated onto the DM
		cmd_mtx=np.linalg.pinv(IM,rcond=rcond)

		optics.applydmc(bestflat)
		time.sleep(tsleep)
		tar_ini = optics.stackslopes(30)
		tar=np.array([tar_ini[0][wf_ind],tar_ini[1][wf_ind]]).flatten().T
		tar[np.where(np.isnan(tar)==True)]=0 #shouldn't need this line, but again, incase of misregistrations...
		coeffs=np.dot(tar,cmd_mtx)
		cmd=np.zeros(dmcini.shape).astype(np.dtype(np.float32)).flatten()
		cmd[inddmuse]=(act_arr.T*-1*coeffs).flatten()
		#ds9.view(np.flipud(cmd.reshape(dmcini.shape)))

	#reference slopes from commented out code above preamble
	refslopes=np.load(joindata('refslopes/refSlopes4ALPAOflat.npy'))

	rcond=3e-1
	cmd_mtx=np.linalg.pinv(IM,rcond=rcond)
	numiter=20 #convergence seems around this many iterations for a gain of 0.5
	gain=0.5
	leak=1
	optics.applydmc(bestflat)
	time.sleep(tsleep)
	wfs = optics.stackwf(10)
	wfarr=np.zeros(numiter)
	for nit in range(numiter):
		tar_ini = optics.stackslopes(10)#-refslopes
		tar = np.array([tar_ini[0][wf_ind],tar_ini[1][wf_ind]]).flatten().T
		tar[np.where(np.isnan(tar)==True)]=0 #shouldn't need this line, but again, incase of misregistrations...
		coeffs = np.dot(tar,cmd_mtx)
		cmd = np.zeros(dmcini.shape).astype(np.dtype(np.float32)).flatten()
		cmd[inddmuse] = (act_arr.T*-1*coeffs).flatten()
		optics.applydmc(leak*optics.getdmc()+rmtt(cmd.reshape(dmcini.shape)*gain))
		time.sleep(tsleep)
		wfarr[nit]=np.std(optics.getwf()[wf_ind])

	wfe = optics.stackwf(10)
	print(np.nanstd(wfs[wf_ind])/np.nanstd(wfe[wf_ind]))
	optics.set_expt(expt_init)
	np.save(optics.bestflat_path, optics.getdmc())

