'''
close the AO loop with a SHWFS
'''

import numpy as np
import sys
import time
from ancillary_code import *


#4/6/21: Darren placed the dummy DM on the MEMs, during which I saved offset slopes and wavefront to close the loop to (slopes may be bogus; Sylvain hadn't yet fixed the slopes recording in real time problem).
'''
nstack=10000
refwf1=stackwf(nstack)
np.save('refwf1.npy',refwf1)
refwf2=stackwf(nstack)
np.save('refwf2.npy',refwf2)

refslopes=getSlopes()
np.save('refslopes.npy',refslopes)

expt(1e-3)
refscc1=stack(10000)
np.save('refscc1.npy',refscc1)
refscc2=stack(10000)
np.save('refscc2.npy',refscc2)

'''

#bestflat=np.load('bestflat.npy') #on FPM
#bestflat=np.load('bestflat_offfpm.npy') #off FPM
bestflat=dmzero+0.5
#bestflat=np.load('bestflat_shwfs.npy') #bootstrapping: previous SHWFS
applydmc(bestflat)
dmcini=getdmc()

wf_ini=getwf()
wf_ind=np.where(np.isnan(wf_ini)==False) #index for where the pupil is defined
wfmask=np.zeros(wf_ini.shape)*np.nan
wfmask[wf_ind]=1
wf=wf_ini[wf_ind]

'''
#generate a bad actuator mask for the WFS wavefront images; do this by applying a focus ramp and seeing which WFS pixels remain correlated  
#DM aperture to apply focus across the full DM:
ydim,xdim=dmcini.shape
rho,phi=functions.polar_grid(xdim,xdim) #here assuming 32 of the 32 actuators are illuminated, but this changes below
aperture=np.zeros(rho.shape).astype(float32)
indap=np.where(rho>0)
aperture[indap]=1
remove_piston = lambda dmc: dmc-np.mean(dmc[indap]) #function to remove piston from dm command to have zero mean (must be intermediate)

def funz(amp,bestflat=dmzero): #apply zernike to the DM
	z=functions.zernike(2,0,rho,phi)/2
	zdm=amp*(z.astype(float32))
	dmc=remove_piston(bestflat)+remove_piston(zdm)+0.5
	applydmc(dmc*aperture)
	return dmc


namp=30 #how many grid points to walk through Zernike amplitude coefficients
amparr=np.linspace(-0.5,0.5,namp)

tsleep=0.1 #time to sleep in between applying DM commands and grabbing WFS images images, see below for optimization

applyzero()
wfarr=np.zeros((namp,wf.shape[0]))
for a in range(namp):
	dmz=funz(amparr[a])
	time.sleep(tsleep)
	wfarr[a]=stackwf(10)[wf_ind]	
	print('focus amp='+str(amparr[a]))
applyzero()

wfcor=np.nansum(np.vstack(np.nanmedian(wfarr,axis=0)).T*wfarr*np.vstack(np.nanstd(wfarr,axis=1))/np.nanstd(np.nanmedian(wfarr,axis=0)),axis=0)/np.nansum(np.nanmedian(wfarr,axis=0)**2)
plt.figure()
plt.plot(wfcor)
plt.ylabel('wavefront correlation across focus ramp with median frame')
plt.xlabel('wavefront pixel index')

def vwfsmask(cormax,cormin):
	rmwfcor=np.where(np.logical_or(wfcor>cormax,wfcor<cormin))[0]
	wfsmask=np.zeros(wfmask.shape)
	wfsmask[wf_ind]=1
	wfsmask[wf_ind[0][rmwfcor],wf_ind[1][rmwfcor]]=0
	ds9.view(wfsmask)
#wfmask[wf_ind[rmwfcor]]=0
'''

def poke_act(x,y,amp=0.1): #poke an actuator
	dmc_poke=np.zeros(dmcini.shape).astype(float32)
	dmc_poke[y,x]=amp
	applydmc(bestflat+dmc_poke)

def opttsleep(x,y,tsleep): #find how long is needed to pause in between DM commands and WFS measurements, and see how to map which actuators are registering on the WFS pupil
	poke_act(x,y,amp=0.1)
	time.sleep(tsleep)
	wf_push=stackwf(10)
	poke_act(x,y,amp=-0.1)
	time.sleep(tsleep)
	wf_pull=stackwf(10)
	out=wfmask*(wf_push-wf_pull)
	ds9.view(out)
	if np.nanmin(out)<-0.01:
		print('use?')
		indmin=np.where(out==np.nanmin(out)) #where the actuator poke is
		subtpoke=out
		subtpoke[indmin[0][0]-2:indmin[0][0]+3,indmin[1][0]-2:indmin[1][0]+3]=np.nan #mask out the central actuator poke
		print('residual std='+str(round(np.nanstd(subtpoke),3)))



tsleep=0.1 #optimized from above function

#illuminated DM aperture after testing: centered at 15,15 (python indicies), radius of 13; so both x and y range between 2 and 28 (27 actuators across the pupil)
ydim,xdim=dmcini.shape
xdmcen,ydmcen=15,15
raddm=13
dmgrid=np.mgrid[0:ydim,0:xdim].astype(float32)
dmrgrid=np.sqrt((dmgrid[0]-ydmcen)**2+(dmgrid[1]-xdmcen)**2)
DMmap=np.zeros(dmcini.shape).astype(float32)
DMmap[np.where(dmrgrid<=raddm)]=1
inddmuse=np.where(DMmap.flatten()==1)[0]
nact=len(inddmuse)

'''
waffle=np.zeros(dmcini.shape).astype(float32)
waffleamp=0.1 #half the desired waffle peak-to valley
for j in range(0,32,2):
	for k in range(0,32,2):
		waffle[j,k]=2*waffleamp
waffle=waffle-waffleamp
'''

IMamp=0.1 #amplitude to poke each actuator in generating the IM (in DM units)

#animate IM
'''
import matplotlib.animation as animation

	fig,axs=plt.subplots(ncols=2,nrows=1,figsize=(8,4))
	ax1,ax2=axs[0],axs[1]
	plt.axis('off')
	ax1.axis('off')

	def map_acts(x,y): #map DM actuators to WFS
		poke_act(x,y,amp=0.2)
		time.sleep(tsleep)
		wf_push=getwf()
		dm_push=getdmc()
		poke_act(x,y,amp=-0.1)
		time.sleep(tsleep)
		wf_pull=getwf()
		dm_pull=getdmc()
		return dm_push-dm_pull,wf_push-wf_pull


	dmpp,wfpp=map_acts(15,15)

	im1=ax1.imshow(dmpp,vmin=0,vmax=0.2)
	im2=ax2.imshow(wfpp,vmin=-0.2,vmax=0.01)

	ax1.set_title('DM commands')
	ax2.set_title('WFS wavefront')

	tmpImage = shmlib.shm("/tmp/tempWF.im.shm",data = stackwf(10))
	def update_img(i):
		cmd=dmcini.flatten()*0;
		cmd[i] = IMamp
		applydmc(bestflat.flatten()+cmd)
		dm_push=getdmc()
		time.sleep(tsleep)
		wfpush=stackwf(10)
		cmd[i] = -IMamp
		applydmc(bestflat.flatten()+cmd)
		time.sleep(tsleep)
		wfpull=stackwf(10)
		dm_pull=getdmc()

		im1.set_data(dm_push-dm_pull)
		im2.set_data(wfpush-wfpull)
		tmpImage.set_data(wfpush-wfpull)

	#ani=animation.FuncAnimation(fig,update_img,frames=range(nact))
	ani=animation.FuncAnimation(fig,update_img,frames=range(32*18,32*19))
	#writer=animation.ImageMagickFileWriter(fps=10)
	writer=animation.ImageMagickFileWriter(fps=1)
	#ani.save('SHWFS_IM.gif',writer=writer)
	ani.save('SHWFS_line.gif',writer=writer)
'''

# Push each actuator
IM=np.zeros((nact,wf.shape[0])) #interaction matrix to fill
useact=np.zeros(nact) #boolean array to identify additional dead actuators within the illuminated WFS pupil 
for i in range(nact):
	actind=inddmuse[i]
	cmd=dmcini.flatten()*0;
	cmd[actind] = IMamp
	applydmc(bestflat+cmd.reshape(dmcini.shape))
	time.sleep(tsleep)
	wfpush=stackwf(10)
	cmd[actind] = -IMamp
	applydmc(bestflat+cmd.reshape(dmcini.shape))
	time.sleep(tsleep)
	wfpull=stackwf(10)

	out=wfpush-wfpull


	indmin=np.where(out==np.nanmin(out)) #where the actuator poke is
	pokemask=np.zeros(wfpush.shape)
	pokemask[indmin[0][0]-2:indmin[0][0]+3,indmin[1][0]-2:indmin[1][0]+3]=np.nan #mask out the central actuator poke
	
	subtpoke=np.zeros(wfpush.shape)
	indsubtpoke=np.where(np.isnan(pokemask*out)==False)
	subtpoke[indsubtpoke]=out[indsubtpoke]

	out[np.where(pokemask==0)]=0
	out[np.where(np.isnan(out)==True)]=0

	IM[i]=out[wf_ind]

	if np.nanmin(IM[i])<-0.01:

		if np.nanstd(subtpoke)<0.01: #this cut removes pinned actuators near a bad actuator that still poke but make a weird shape over the full DM surface
			useact[i]=1
	print('act'+str(i)+'of'+str(nact))

IM[np.where(np.isnan(IM)==True)]=0 #shouldn't need this line, but just in case there are pupil misregistrations, between the mask and when the IM is applied or any other reason that might generate nans

badacts=np.array([49,112,136,150,152,180,182,183,238,240,244,235,236,237,261,262,263,269,284,314,348,352,376,396,398,402,425,441,444,467,475,486,503,527]) #indicies of bad actuators, identified by eye in IM

#rmacts=np.where(useact==0)[0]
#IM_clean_ini=np.delete(IM,rmacts,axis=0)
IM_clean_ini=np.delete(IM,badacts,axis=0)

dead_acts=inddmuse[badacts]
dead_acts_xy=np.array(list(zip((dead_acts/32).astype(int),dead_acts%32))) #x,y format
np.save('dead_acts.npy',dead_acts)

#flatdmmap=DMmap.flatten()
#flatdmmap[dead_acts]=0
#DMmap=flatdmmap.reshape(dmcini.shape).astype(float32)
DMmap_deadact=np.ones(dmcini.shape).flatten()
DMmap_deadact[inddmuse[badacts]]=0
DMmap_deadact=DMmap_deadact.reshape(dmcini.shape)
DMmap=DMmap*DMmap_deadact
DMmapind=np.where(DMmap==1)
np.save('DMmap.npy',DMmap)

def vdeadacts(i):
	x,y=dead_acts_xy[i]
	poke_act(x,y,amp=0.1)
	time.sleep(tsleep)
	wf_push=stackwf(10)
	poke_act(x,y,amp=-0.1)
	time.sleep(tsleep)
	wf_pull=stackwf(10)
	out=wfmask*(wf_push-wf_pull)
	ds9.view(out)

#after manually looking through all the dead actuators in vdeadacts, there are only 2 that are clearly dead: indicied 13 and 23 of dead_acts
badwfsmask=np.ones(wf_ini.shape)
for i in [16]:
	x,y=dead_acts_xy[i]
	xr,yr=np.array([]).astype(int),np.array([]).astype(int)
	for j in range(4):
		poke_act(x+[1,0,-1,0][j],y+[0,-1,0,1][j],amp=0.1)
		time.sleep(tsleep)
		wf_push=stackwf(100)
		poke_act(x,y,amp=-0.1)
		time.sleep(tsleep)
		wf_pull=stackwf(100)
		out=wfmask*(wf_push-wf_pull)
		indmin=np.where(out==np.nanmin(out))
		xr,yr=np.append(xr,indmin[1][0]),np.append(yr,indmin[0][0])
	badwfsmask[np.min(yr):np.max(yr)+1,np.min(xr):np.max(xr)+1]=np.nan

wfrmind=np.where(np.isnan((wfmask*badwfsmask)[wf_ind])==True) #indicies to remove from IM
wfmask_bad=wfmask*badwfsmask
wfs_tar_ind=(np.delete(wf_ind[0],wfrmind[0]),np.delete(wf_ind[1],wfrmind[0]))
np.save('wfs_tar_ind.npy',wfs_tar_ind)

IM_clean=np.delete(IM_clean_ini,wfrmind,axis=1)
nact=IM_clean.shape[0]

from datetime import datetime
np.save('IM/SHWFS/'+datetime.now().strftime("%d_%m_%Y_%H_%M")+'.npy',IM_clean)
np.save('SHWFS_IM.npy',IM_clean)

#start code from here if you don't want to generate a new IM:
IM_clean=np.load('SHWFS_IM.npy')
nact=IM_clean.shape[0]
wfs_tar_ind=np.load('wfs_tar_ind.npy')

act_arr=np.vstack(np.zeros(nact).astype(float32)+IMamp*2) #array of DM commands

def gen_dmc(cmd_want): #input: voltages of all the actuators I want to control; output: square map of all voltages, including dead actuators
	cmd=np.zeros(dmcini.shape).astype(float32)
	cmd[DMmapind]=cmd_want
	return cmd

#the following code will help determine the optimal SVD cutoff, looking at what SVD cutoff will best reconstruct individual modes

plt.figure()
def pc(rcond,amp,i): #see if the coefficients can reconstruct the input
	#rcond=1e-3
	cmd_mtx=np.linalg.pinv(IM_clean,rcond=rcond)
	#i=100
	cmd=np.zeros(nact).astype(float32);
	cmd[i] = amp/2
	applydmc(bestflat+gen_dmc(cmd))
	time.sleep(tsleep)
	wfpush=stackwf(10)[wfs_tar_ind]
	cmd[i] = -amp/2
	applydmc(bestflat+gen_dmc(cmd))
	time.sleep(tsleep)
	wfpull=stackwf(10)[wfs_tar_ind]
	tar=np.vstack(wfpush-wfpull).T
	tar[np.where(np.isnan(tar)==True)]=0
	coeffs=np.dot(tar,cmd_mtx)

	plt.plot(coeffs.flatten())


def vcmd(rcond): #view the reconstructed DM commands, making sure that waffle mode is not propagated onto the DM
	cmd_mtx=np.linalg.pinv(IM_clean,rcond=rcond)

	applydmc(bestflat)
	time.sleep(tsleep)
	tar=np.vstack(stackwf(30)[wfs_tar_ind]).T
	tar[np.where(np.isnan(tar)==True)]=0 #shouldn't need this line, but again, incase of misregistrations...
	coeffs=np.dot(tar,cmd_mtx)
	cmd=np.zeros(dmcini.shape).astype(float32)
	cmd[DMmapind]=(act_arr.T*-1*coeffs)
	#cmd=(act_arr.T*-1*coeffs).reshape(dmcini.shape).astype(float32)
	ds9.view(cmd)

rcond=5e-2 #SVD cutoff; optimized from above code

cmd_mtx=np.linalg.pinv(IM_clean,rcond=rcond)
#np.save('CM/SHWFS/'+datetime.now().strftime("%d_%m_%Y_%H_%M")+'.npy',IM_clean)
#np.save('SHWFS_CM.npy',cmd_mtx)

numiter=10
gain=0.1
leak=1

refwf1=np.load('refwf1.npy') #reference slope measured with the flat DM in (see commented out code at the top of this script)
applydmc(np.load('bestflat_shwfs.npy'))
#applydmc(np.load('bestflat.npy'))
time.sleep(tsleep)
for nit in range(numiter):
	tar=np.vstack((stackwf(10))[wfs_tar_ind]).T #-refwf1
	tar[np.where(np.isnan(tar)==True)]=0 #shouldn't need this line, but again, incase of misregistrations...
	coeffs=np.dot(tar,cmd_mtx)
	cmd=np.zeros(dmcini.shape).astype(float32)
	cmd[DMmapind]=(act_arr.T*-1*coeffs)
	#cmd=(act_arr.T*-1*coeffs).reshape(dmcini.shape).astype(float32)
	applydmc(leak*getdmc()+cmd*gain)
	time.sleep(tsleep)

#adjust IMamp, rcond, and offset parameters until happy, then:

np.save('bestflat_shwfs.npy',getdmc())
