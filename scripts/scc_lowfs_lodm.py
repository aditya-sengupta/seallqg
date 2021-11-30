'''
this code is for low order Zernike with the LODM control, to be used after running align_fpm_lodmc.py. this code does the following

(1) build an interaction matrix for Zernike modes
(2) measure linearity for those modes
(3) close the loop on air in the testbed, ensuring loop stability
'''

from ancillary_code import *
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from numpy import float32
import time
from functions import *


dmc2wf=np.load('recent_data/lodmc2wfe.npy') #alpao actuator dmc command units to microns WFE units

#apply lowfs bestflat
dmcini=getlodmc()
ydim,xdim=dmcini.shape
grid=np.mgrid[0:ydim,0:xdim].astype(float32)
bestflat=np.load('recent_data/lodm_bestflat.npy')
applylodmc(bestflat)

#apply mems dark hole
#dmc_dh=np.load('dmc_dh.npy')
#applydmc(dmc_dh)
#im_dmc_dh=stack(100)


texp=1e-3
expt(texp) #set exposure time; for current light source config at 100 Hz
time.sleep(5)
imini=getim() #Andor image just for referencing dimensions
imydim,imxdim=imini.shape
tsleep=0.02 #should be the same values from align_fpm_lodmc.py

#DM aperture;
xy=np.sqrt((grid[0]-dmcini.shape[0]/2+0.5)**2+(grid[1]-dmcini.shape[1]/2+0.5)**2)
aperture=np.zeros(dmcini.shape).astype(float32)
aperture[np.where(xy<dmcini.shape[0]/2)]=1 
indap=np.where(aperture==1)
indnap=np.where(aperture==0)
inddmuse=np.where(aperture.flatten()==1)[0]
nact=len(inddmuse)

remove_piston = lambda dmc: dmc-np.mean(dmc[indap]) #function to remove piston from dm command to have zero mean

tip,tilt=((grid[0]-ydim/2+0.5)/ydim*2).astype(float32),((grid[1]-xdim/2+0.5)/ydim*2).astype(float32)# DM tip/tilt 

IMtt=np.array([(tip).flatten(),(tilt).flatten()])
CMtt=np.linalg.pinv(IMtt,rcond=1e-5)
def rmtt(ph): #remove tip/tilt from DM commands
	coeffs=np.dot(np.vstack((ph).flatten()).T,CMtt) 
	lsqtt=np.dot(IMtt.T,coeffs.T).reshape(tilt.shape).astype(float32)
	return ph-lsqtt

#setup Zernike polynomials
nmarr=[]
norder=3 #how many radial Zernike orders to look at
for n in range(1,norder):
	for m in range(-n,n+1,2):
		nmarr.append([n,m])

rho,phi=functions.polar_grid(xdim,xdim)
rho[int((xdim-1)/2),int((ydim-1)/2)]=0.00001 #avoid numerical divide by zero issues
def funz(n,m,amp,bestflat=bestflat): #apply zernike to the DM
	z=functions.zernike(n,m,rho,phi)/2
	zdm=amp*(z.astype(float32))
	dmc=remove_piston(remove_piston(bestflat)+remove_piston(zdm))
	applylodmc(dmc)
	return zdm #even though the Zernike is applied with a best flat, return only the pure Zernike; subsequent reconstructed Zernike mode coefficients should not be applied to best flat commands

imxcen,imycen=np.load('recent_data/imcen.npy')
beam_ratio=np.load('recent_data/beam_ratio.npy')

gridim=np.mgrid[0:imydim,0:imxdim]
rim=np.sqrt((gridim[0]-imycen)**2+(gridim[1]-imxcen)**2)

#algorithmic LOWFS mask (centered around the core, for light less than 6 lambda/D)

def vz(n,m,IMamp,rmask): #determine the minimum IMamp (interaction matrix amplitude) to be visible in differential images
	zern=funz(n,m,IMamp)
	time.sleep(tsleep)
	imzern=stack(10)
	applylodmc(bestflat)
	time.sleep(tsleep)
	imflat=stack(10)
	ttmask=np.zeros(imini.shape)
	ttmask[np.where(rim/beam_ratio<rmask)]=1
	ds9.view((np.abs(processim(imzern))-np.abs(processim(imflat)))*ttmask)

#from above function
ttmask=np.zeros(imini.shape)
rmask=10
indttmask=np.where(rim/beam_ratio<rmask)
ttmask[indttmask]=1
IMamp=0.001

#make interaction matrix
def genIM(rzernarr=False):
	refvec=np.zeros((len(nmarr),ttmask[indttmask].shape[0]*2))
	zernarr=np.zeros((len(nmarr),aperture[indap].shape[0]))
	for i in range(len(nmarr)):
		n,m=nmarr[i]
		zern=funz(n,m,IMamp)
		time.sleep(tsleep)
		imzern=stack(10)
		applylodmc(bestflat)
		time.sleep(tsleep)
		imflat=stack(10)
		imdiff=(imzern-imflat)
		Im_diff=processim(imdiff)
		refvec[i]=np.array([np.real(Im_diff[indttmask]),np.imag(Im_diff[indttmask])]).flatten()
		zernarr[i]=zern[indap]

	IM=np.dot(refvec,refvec.T) #interaction matrix
	if rzernarr==False:
		return IM,refvec
	else:
		return IM,zernarr

IM,zernarr=genIM(rzernarr=True)

#determine the optimal SVD cutoff, looking at what SVD cutoff will best reconstruct individual modes
'''
plt.figure()
plt.ylabel('DM units')
plt.xlabel('Zernike mode')

def pc(rcond,i,sf):
	
	#rcond: svd cutoff to be optimized
	#i: which Zernike mode to apply
	#sf: scale factor for amplitude to apply of given Zernike #mode as a fraction of the input IM amplitude
	
	#rcond=1e-3
	IMinv=np.linalg.pinv(IM,rcond=rcond)
	cmd_mtx=np.dot(IMinv,refvec)

	n,m=nmarr[i]
	zern=funz(n,m,IMamp*sf)
	time.sleep(tsleep)
	imzern=stack(10)
	applylodmc(bestflat)
	time.sleep(tsleep)
	imflat=stack(10)
	imdiff=(imzern-imflat)
	Im_diff=processim(imdiff)
	tar = np.array([np.real(Im_diff[indttmask]),np.imag(Im_diff[indttmask])]).flatten()
	
	coeffs=np.dot(cmd_mtx,tar)
	plt.plot(coeffs*IMamp)
	plt.axhline(IMamp*sf,0,len(coeffs),ls='--')
'''

rcond=1e-5 #not totally from above pc function; there seems to be tradeoffs. best to look at a range of rconds and linearity plots: results from looking at lots of plots 
#nrcond=5
#rcondarr=10**(-1*np.linspace(1,5,nrcond))
#for k in range(nrcond):
#rcond=rcondarr[k]

def genCM(rcond=rcond):
	IM,refvec=genIM()
	IMinv=np.linalg.pinv(IM,rcond=rcond)
	cmd_mtx=np.dot(IMinv,refvec)
	return cmd_mtx
cmd_mtx=genCM()

applylodmc(bestflat)
time.sleep(tsleep)
imflat=stack(100)


#LINEARITY CHARACTERIZATION

def genzerncoeffs(i,zernamp):
	'''
	i: zernike mode
	zernamp: Zernike amplitude in DM units to apply
	'''
	n,m=nmarr[i]
	zern=funz(n,m,zernamp)
	time.sleep(tsleep)
	imzern=stack(10)
	imdiff=(imzern-imflat)
	tar_ini=processim(imdiff)
	tar = np.array([np.real(tar_ini[indttmask]),np.imag(tar_ini[indttmask])]).flatten()	
	coeffs=np.dot(cmd_mtx, tar)
	return coeffs*IMamp

nlin = 20 #number of data points to scan through linearity measurements
zernamparr = np.linspace(-1.5*0.005,1.5*0.005,nlin)

#try linearity measurement for Zernike mode 0
zernampout=np.zeros((len(nmarr),len(nmarr),nlin))
for nm in range(len(nmarr)):
	for i in range(nlin):
		zernamp=zernamparr[i]
		coeffsout=genzerncoeffs(nm,zernamp)
		zernampout[nm,:,i]=coeffsout

applylodmc(bestflat)

fig,axs=plt.subplots(ncols=3,nrows=2,figsize=(12,10),sharex=True,sharey=True)
fig.suptitle('rcond='+str(rcond))

colors=mpl.cm.viridis(np.linspace(0,1,len(nmarr)))
axarr=[axs[0,0],axs[0,1],axs[0,2],axs[1,0],axs[1,1],axs[1,2]]
fig.delaxes(axarr[-1])
for i in range(len(nmarr)):
	ax=axarr[i]
	ax.set_title('n,m='+str(nmarr[i][0])+','+str(nmarr[i][1]))
	if i==4:
		ax.plot(zernamparr*dmc2wf,zernamparr*dmc2wf,lw=1,color='k',ls='--',label='y=x')
		for j in range(len(nmarr)):
			if j==i:
				ax.plot(zernamparr*dmc2wf,zernampout[i,i,:]*dmc2wf,lw=2,color=colors[j],label='n,m='+str(nmarr[i][0])+','+str(nmarr[i][1]))
			else:
				ax.plot(zernamparr*dmc2wf,zernampout[i,j,:]*dmc2wf,lw=1,color=colors[j],label='n,m='+str(nmarr[j][0])+','+str(nmarr[j][1]))
	else:
		ax.plot(zernamparr*dmc2wf,zernamparr*dmc2wf,lw=1,color='k',ls='--')
		for j in range(len(nmarr)):
			if j==i:
				ax.plot(zernamparr*dmc2wf,zernampout[i,i,:]*dmc2wf,lw=2,color=colors[j])
			else:
				ax.plot(zernamparr*dmc2wf,zernampout[i,j,:]*dmc2wf,lw=1,color=colors[j])

axarr[4].legend(bbox_to_anchor=(1.05,0.9))
axarr[4].set_xlabel('input ($\\mu$m WFE, PV)')
axarr[3].set_xlabel('input ($\\mu$m WFE, PV)')
axarr[3].set_ylabel('reconstructed output ($\\mu$m WFE, PV)')
axarr[0].set_ylabel('reconstructed output ($\\mu$m WFE, PV)')

np.save('lowfs_telemetry/linearity_'+datetime.now().strftime("%d_%m_%Y_%H")+'.npy',{'in':zernamparr,'out':zernampout,'rcond':rcond})

#NEXT: RECORD OPEN LOOP COEFFS FOR AIR AND MOVING TURBULENCE
cmd_mtx=genCM()
applylodmc(bestflat)
time.sleep(tsleep)
imflat=stack(100)

def genolcoeffs():
	imtar=getim()
	imdiff=imtar-imflat
	tar_ini=processim(imdiff)
	tar = np.array([np.real(tar_ini[indttmask]),np.imag(tar_ini[indttmask])]).flatten()	
	coeffs=np.dot(cmd_mtx, tar)
	#lodmc[indap]=np.dot(zernarr.T,-coeffs*IMamp)
	return coeffs*IMamp


nit=1000
coeff_arr_ol=np.zeros((nit,len(nmarr)))
for i in range(nit):
	coeff_arr_ol[i]=genolcoeffs()

colors=mpl.cm.viridis(np.linspace(0,1,len(nmarr)))
plt.figure()
plt.ylabel('OL coefficient (DM units)')
plt.xlabel('time (s)')
for j in range(len(nmarr)):
	plt.plot(np.arange(nit)*1e-2,coeff_arr_ol[:,j],color=colors[j],label='n,m='+str(nmarr[j][0])+','+str(nmarr[j][1]))
plt.legend(loc='best')


plt.figure()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('freq (Hz)')
plt.ylabel('PSD (DM units)')
for j in range(len(nmarr)):
	freq,psd=genpsd(coeff_arr_ol[:,j],1e-2,nseg=1)
	plt.plot(freq,psd,label='n,m='+str(nmarr[j][0])+','+str(nmarr[j][1]),color=colors[j])

plt.legend(loc='best')


#NEXT: CLOSE THE LOOP ON IN AIR TURBULENCE
cmd_mtx=genCM()
applylodmc(bestflat)
time.sleep(tsleep)
imflat=stack(100)

lodmc=np.zeros(aperture.shape).astype(float32)

#gain=0.1
leak=1
nit=1000
closeloopat=nit/2
coeff_arr_cl=np.zeros((nit,len(nmarr)))
for i in range(nit):
	if i<closeloopat:
		gain=0
	else:
		gain=0.1
	imtar=getim()
	imdiff=imtar-imflat
	tar_ini=processim(imdiff)
	tar = np.array([np.real(tar_ini[indttmask]),np.imag(tar_ini[indttmask])]).flatten()	
	coeffs=np.dot(cmd_mtx, tar)
	coeff_arr_cl[i]=coeffs*IMamp
	#Maaike's PWFC MVM goes here!
	lodmc[indap]=np.dot(zernarr.T,-coeffs)
	applylodmc(getlodmc()*leak+gain*lodmc)
	time.sleep(tsleep)

plt.figure()
plt.ylabel('CL coefficient (DM units)')
plt.xlabel('iteration')
for j in range(len(nmarr)):
	plt.plot(np.arange(nit),coeff_arr_cl[:,j],color=colors[j],label='n,m='+str(nmarr[j][0])+','+str(nmarr[j][1]))
plt.axvline(nit/2,color='r',lw=2,ls='--')
plt.legend(loc='best')
plt.tight_layout()

np.save('lowfs_telemetry/on_air_coeff_arr_cl'+datetime.now().strftime("%d_%m_%Y_%H")+'.npy',{'coeffs':coeff_arr_cl,'frame_rate':1/tsleep,'lag':tsleep,'gain':gain,'leak':leak,'closeloopat':closeloopat})

plt.figure()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('freq (Hz)')
plt.ylabel('PSD (DM units)')

for j in range(len(nmarr)-3):
	freq,psd_ol=genpsd(coeff_arr_cl[:int(closeloopat),j],tsleep,nseg=1)
	freq,psd_cl=genpsd(coeff_arr_cl[int(closeloopat):,j],tsleep,nseg=1)
	plt.plot(freq,psd_ol,ls='--',color=colors[j])
	plt.plot(freq,psd_cl,color=colors[j],label='n,m='+str(nmarr[j][0])+','+str(nmarr[j][1]))
plt.legend(loc='best')
plt.title('OL="--", CL="-"')


hrejarr=np.zeros(((len(nmarr),psd_ol.shape[0])))
for j in range(len(nmarr)):
	freq,psd_ol=genpsd(coeff_arr_cl[:int(closeloopat),j],tsleep,nseg=1)
	freq,psd_cl=genpsd(coeff_arr_cl[int(closeloopat):,j],tsleep,nseg=1)
	hrejarr[j]=np.sqrt(psd_cl/psd_ol)
hrej=np.median(hrejarr,axis=0)
plt.figure()
plt.xlabel('log$_10$(freq) (Hz)')
plt.ylabel('rejection transfer function (dB)')
plt.plot(np.log10(freq),20*np.log10(hrej))


#MOVING ATMOSPHERIC PHASE SCREENS
applylodmc(bestflat)
cmd_mtx=genCM()
applylodmc(bestflat)
time.sleep(tsleep)
imflat=stack(100)

#setup phase screens
Nact=11
wavelength=1.6e-6 #assume H band level AO residuals
wavefronterror=100e-9 #WFE in m rms
pupilpix=11 #number of illuminated actuators
npupcross=5 #number of pupil crossings before translation repeats itself
imagepix=pupilpix*npupcross
bmro=imagepix/pupilpix #also the number of pupil crossings
pl=-2 #power law

phcrop = lambda ph: ph[int((imagepix-Nact)/2):int((imagepix+Nact)/2),int((imagepix-Nact)/2):int((imagepix+Nact)/2)]

seal_conversion=dmc2wf*1e-6 #DM units (volts?) to m of wavefront

ph_al=make_noise_pl(wavefronterror,imagepix,pupilpix,wavelength,pl,Nact) #generate phase screen

ph_ini=antialias(ph_al,imagepix,bmro,pupilpix)/2/np.pi*wavelength/seal_conversion

ph=ph_ini-np.mean(ph_ini)

#setup phase screen translation:
Dtel=10
tlag=1e-3 #this is the assumed system latency between end of exposure and dm commands send
vx=1 #x wind speed in m/s
new_atm_lag = lambda phase_in: translate_atm(phase_in,bmro,tlag,Dtel,vx=vx,vy=0) # the atmospheric phase screen will be manually translated by this much after DM commands are generated
twfs=10e-3 #this is the assumed system frame rate
new_atm_frame= lambda phase_in: translate_atm(phase_in,bmro,twfs,Dtel,vx=vx,vy=0) #the atmospheric phase screen will be manually translated by this much for each frame

'''
#test code to view that AO residuals are indeed translating...
phin=ph
i=0
while i < 100:
	phout=new_atm_frame(phin)
	ds9.view(aperture*phcrop(phout)-remove_tt(phcrop(phout),pupilpix,pupilpix))
	phin=phout
	i=i+1
	#time.sleep(1)
'''
dmc_cmd_new=np.zeros(aperture.shape).astype(float32)
def loop_iter(diffim,phin,dmc_cmd_old,leak,gain):
	'''
	function representing one closed-loop iteration.

	diffim = input SCC image (differential from best flat image for atm turb)
	phin = input simulated atmospheric phase screen (npupcross x larger than the dm array size), in DM units, consistent as defined earlier in code

	dmc_cmd_old = dm command from the previous frame, but only the control component of it (i.e., removing the best flat and simulated turbulence components)
	leak = loop leak (scalar)
	gain = loop gain (scalar OR vector of the same lengths as coeffs for modal gains)

	set leak = 1, gain = 0 for open loop
	'''

	#dynamic control component
	tar_ini=processim(diffim)
	tar = np.array([np.real(tar_ini[indttmask]),np.imag(tar_ini[indttmask])]).flatten()	
	coeffs=np.dot(cmd_mtx, tar)
	#Maaike's PWFC MVM goes here!
	dmc_cmd_new[indap]=np.dot(zernarr.T,-coeffs*gain)
	
	dmc_cntl=remove_piston(leak*dmc_cmd_old+dmc_cmd_new) #scalar leak implemented in final DM commands (gain implemented earlier)

	#simulated atmospheric turbulence component
	phlag=new_atm_lag(phin) #add atmospheric lag by translating the DM commands a bit
	dmcph=phcrop(phlag).astype(np.float32)
	dmc_turb=(aperture*remove_piston(dmcph)).astype(float32)

	#DC best flat component
	dmc_flat=bestflat
	
	applylodmc(aperture*remove_piston(dmc_turb+dmc_flat+dmc_cntl))

	return dmc_cntl,dmc_turb,coeffs

#first play in open loop to save coefficients to determine optimal gains
#numiter=int(Dtel/vx*npupcross/twfs) #maximum number of iterations before the phase screen starts unphysically repeating itself (need a larger image)
numiter=5000
coeff_arr_ol=np.zeros((numiter,cmd_mtx.shape[0]))

phin=ph
dmc_cmd_old=bestflat-bestflat
applylodmc(remove_piston(remove_piston(bestflat)+remove_piston(phcrop(ph).astype(np.float32))))
time.sleep(tsleep)
for i in range(numiter):
	diffim=getim()-imflat

	dmc_cntl,dmc_turb,coeffs=loop_iter(diffim,phin,dmc_cmd_old,1,0.0)
	coeff_arr_ol[i]=coeffs

	#reset the loop
	dmc_cmd_old=dmc_cntl
	phout=new_atm_frame(phin)
	phin=phout

	time.sleep(tsleep)

plt.figure()
plt.ylabel('OL coefficient (IM units)')
plt.xlabel('time (s)')
for j in range(len(nmarr)):
	plt.plot(np.arange(numiter)*1e-2,coeff_arr_ol[:,j],color=colors[j],label='n,m='+str(nmarr[j][0])+','+str(nmarr[j][1]))
plt.legend(loc='best')
from datetime import datetime
np.save('lowfs_telemetry/coeff_arr_ol'+datetime.now().strftime("%d_%m_%Y_%H")+'.npy',{'coeffs':coeff_arr_ol,'frame_rate':1/twfs,'lag':tlag,'lo_wfe':wavefronterror,'vx':vx})


#next play in closed-loop
coeff_arr_cl=np.zeros((numiter,cmd_mtx.shape[0]))

gain=0.5
leak=0.99

phin=ph
dmc_cmd_old=bestflat-bestflat
applylodmc(remove_piston(remove_piston(bestflat)+remove_piston(phcrop(ph).astype(np.float32))))
time.sleep(tsleep)
for i in range(numiter):
	diffim=getim()-imflat

	#if i <numiter/4: #open loop: set gain to zero
		#dmc_cntl,dmc_turb,coeffs=loop_iter(diffim,phin,dmc_cmd_old,1,0.0)
	#else: #closed loop with optimal gains
	dmc_cntl,dmc_turb,coeffs=loop_iter(diffim,phin,dmc_cmd_old,gain,leak)
	coeff_arr_cl[i]=coeffs

	#reset the loop
	dmc_cmd_old=dmc_cntl
	phout=new_atm_frame(phin)
	phin=phout

	time.sleep(tsleep)

plt.figure()
plt.ylabel('coefficient (IM units)')
plt.xlabel('time (s)')
for j in range(len(nmarr)):
	plt.plot(np.arange(numiter)*1e-2,coeff_arr_cl[:,j],color=colors[j],label='n,m='+str(nmarr[j][0])+','+str(nmarr[j][1]))
plt.legend(loc='best')
plt.axvline(numiter/4*1e-2,color='r',lw=2,ls='--')

np.save('lowfs_telemetry/coeff_arr_cl'+datetime.now().strftime("%d_%m_%Y_%H")+'.npy',{'coeffs':coeff_arr_cl,'frame_rate':1/twfs,'lag':tlag,'lo_wfe':wavefronterror,'vx':vx,'gain':gain,'leak':leak})


plt.figure()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('freq (Hz)')
plt.ylabel('PSD (DM units)')

for j in range(len(nmarr)-3):
	freq,psd_ol=genpsd(coeff_arr_ol[100:,j],1e-2,nseg=5)
	freq,psd_cl=genpsd(coeff_arr_cl[100:,j],1e-2,nseg=5)
	plt.plot(freq,psd_ol,ls='--',color=colors[j])
	plt.plot(freq,psd_cl,color=colors[j],label='n,m='+str(nmarr[j][0])+','+str(nmarr[j][1]))
plt.legend(loc='best')
plt.title('OL="--", CL="-"')


hrejarr=np.zeros(((len(nmarr),psd_ol.shape[0])))
for j in range(len(nmarr)):
	freq,psd_ol=genpsd(coeff_arr_ol[100:,j],1e-2,nseg=5)
	freq,psd_cl=genpsd(coeff_arr_cl[100:,j],1e-2,nseg=5)
	hrejarr[j]=np.sqrt(psd_cl/psd_ol)
hrej=np.median(hrejarr,axis=0)
plt.figure()
plt.xlabel('log$_10$(freq) (Hz)')
plt.ylabel('rejection transfer function (dB)')
plt.plot(np.log10(freq),20*np.log10(hrej))
