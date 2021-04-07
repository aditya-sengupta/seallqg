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

bestflat=np.load('bestflat.npy')
applydmc(bestflat)
dmcini=getdmc()
nact=dmcini.flatten().shape[0] #number of actuators

wf_ini=getwf()
wf_ind=np.where(np.isnan(wf_ini)==False) #index for where the pupil is defined
wfmask=np.zeros(wf_ini.shape)*np.nan
wfmask[wf_ind]=1
wf=wf_ini[wf_ind]



def poke_act(x,y,amp=0.1): #poke an actuatur
	dmc_poke=np.zeros(dmcini.shape).astype(float32)
	dmc_poke[y,x]=amp
	applydmc(bestflat+dmc_poke)

def opttsleep(x,y,tsleep): #find how long is needed to pause in between DM commands and WFS measurements, and see how to map which actuators are registering on the WFS pupil
	poke_act(x,y,amp=0.1)
	time.sleep(tsleep)
	wf_push=stackwf(100)
	poke_act(x,y,amp=-0.1)
	time.sleep(tsleep)
	wf_pull=stackwf(100)
	out=wfmask*(wf_push-wf_pull)
	ds9.view(out)
	if np.nanmin(out)<-0.01:
		print('use?')

tsleep=0.1 #optimized from above function

waffle=np.zeros(dmcini.shape).astype(float32)
waffleamp=0.1 #half the desired waffle peak-to valley
for j in range(0,32,2):
	for j in range(0,32,2):
		waffle[j,k]=2*waffleamp
waffle=waffle-waffleamp

IMamp=0.1 #amplitude to poke each actuator in generating the IM (in DM units)

#animate IM
'''
import matplotlib.animation as animation

fig,axs=plt.subplots(ncols=2,nrows=1,figsize=(8,4))
ax1,ax2=axs[0],axs[1]
plt.axis('off')
ax1.axis('off')

def map_acts(x,y): #map DM actuators to WFS
	poke_act(x,y,amp=0.1)
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

ani=animation.FuncAnimation(fig,update_img,frames=range(nact))
writer=animation.ImageMagickFileWriter(fps=10)
ani.save('SHWFS_IM.gif',writer=writer)
'''

# Push each actuator
IM=np.zeros((nact,wf.shape[0])) #interaction matrix to fill
useact=np.zeros(nact) #boolean array to indicate which DM actuators to use based on which are filling the WFS pupil 
for i in range(nact):
	cmd=dmcini.flatten()*0;
	cmd[i] = IMamp/2
	applydmc(bestflat.flatten()+cmd)
	time.sleep(tsleep)
	wfpush=stackwf(100)[wf_ind]
	cmd[i] = -IMamp/2
	applydmc(bestflat.flatten()+cmd)
	time.sleep(tsleep)
	wfpull=stackwf(100)[wf_ind]

	IM[i]=wfpush - wfpull
	if np.nanmin(IM[i])<-0.01:
		useact[i]=1

DMmap=useact.reshape(dmcini.shape) #2D aperture map of which actuators to use
DMmapind=np.where(DMmap==1)
np.save('DMmap.npy',DMmap)
induseact=np.where(useact==1)
IM=np.delete(IM,induseact)
from datetime import datetime
np.save('IM/SHWFS/'+datetime.now().strftime("%d_%m_%Y_%H_%M")+'.npy',IM)
np.save('SHWFS_IM.npy',IM)

act_arr=np.vstack(dmcini.flatten()[induseact]*0+IMamp) #array of DM commands

#the following code will help determine the optimal SVD cutoff, looking at what SVD cutoff will best reconstruct individual modes

plt.figure()
def pc(rcond,amp,i): #see if the coefficients can reconstruct the input
	#rcond=1e-3
	cmd_mtx=np.linalg.pinv(IM,rcond=rcond)
	#i=100
	cmd=dmcini.flatten()*0;
	cmd[i] = amp/2
	applydmc(bestflat.flatten()+cmd)
	time.sleep(tsleep)
	wfpush=stackwf(10)[wf_ind]
	cmd[i] = -amp/2
	applydmc(bestflat.flatten()+cmd)
	time.sleep(tsleep)
	wfpull=stackwf(10)[wf_ind]
	tar=np.vstack(wfpush-wfpull).T
	coeffs=np.dot(tar,cmd_mtx)

	plt.plot(coeffs.flatten())


def vcmd(rcond): #view the reconstructed DM commands, making sure that waffle mode is not propagated onto the DM
	cmd_mtx=np.linalg.pinv(IM,rcond=rcond)

	applydmc(bestflat)
	time.sleep(tsleep)
	tar=np.vstack(stackwf(100)[wf_ind]).T
	coeffs=np.dot(tar,cmd_mtx)
	cmd=np.zeros(dmcini.shape).astype(float32)
	cmd[DMmapind]=(act_arr.T*-1*coeffs)
	ds9.view(cmd)

rcond=1e-1 #SVD cutoff; optimized from above code

cmd_mtx=np.linalg.pinv(IM,rcond=rcond)
np.save('CM/SHWFS/'+datetime.now().strftime("%d_%m_%Y_%H_%M")+'.npy',IM)
np.save('SHWFS_CM.npy',cmd_mtx)

numiter=50
gain=0.5
leak=1

refwf1=np.load('refwf1.npy') #reference slope measured with the flat DM in (see commented out code at the top of this script)

applydmc(bestflat)
time.sleep(tsleep)
for nit in range(numiter):
	tar=np.vstack(stackwf(100)[wf_ind]).T #-refwf1
	coeffs=np.dot(tar,cmd_mtx)
	cmd=np.zeros(dmcini.shape).astype(float32)
	cmd[DMmapind]=(act_arr.T*-1*coeffs)
	applydmc(leak*getdmc()+cmd*gain)
	time.sleep(tsleep)

#adjust IMamp, rcond, and offset parameters until happy, then:

np.save('bestflat_shwfs.npy',getdmc())
