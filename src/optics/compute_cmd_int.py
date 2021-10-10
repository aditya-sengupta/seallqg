# authored by Benjamin Gerard
# edited by Aditya Sengupta

"""
Compute command matrix and interaction matrix.
"""

import numpy as np
from numpy import float32
import time
import tqdm
# import pysao
from os import path
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.optimize import newton

from .image import optics
from .tt import processim
from .ao import polar_grid, zernike
from ..utils import joindata

dmc2wf = np.load(joindata(path.join("bestflats", "lodmc2wfe.npy")))

#initial setup: apply best flat, generate DM grid to apply future shapes
dmcini = optics.getdmc()
ydim, xdim = optics.dmdims
grid = np.mgrid[0:ydim, 0:xdim].astype(np.float32)
bestflat = np.load(joindata(path.join("bestflats", "bestflat_{0}_{1}.npy".format(optics.name, optics.dmdims[0]))))
#load bestflat, which should be an aligned FPM
optics.applydmc(bestflat)
time.sleep(1)
imflat = optics.stackim(100)

optics.set_expt(1e-3) #set exposure time
imydim, imxdim = optics.imdims

tsleep = 0.02 
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
nmarr = []
norder = 3 #how many radial Zernike orders to look at; just start with tip/tilt
for n in range(1, norder):
	for m in range(-n, n+1, 2):
		nmarr.append([n,m])

rho, phi = polar_grid(xdim, ydim)
rho[int((xdim-1)/2),int((ydim-1)/2)] = 0.00001 #avoid numerical divide by zero issues

def funz(n, m, amp, bestflat=bestflat): #apply zernike to the DM
	z = zernike(n, m, rho, phi)/2
	zdm = amp*(z.astype(np.float32))
	dmc = remove_piston(remove_piston(bestflat)+remove_piston(zdm))
	optics.applydmc(dmc)
	return zdm

#calibrated image center and beam ratio from genDH.py
imxcen, imycen = np.load(joindata(path.join("bestflats", "imcen.npy")))
beam_ratio = np.load(joindata(path.join("bestflats", "beam_ratio.npy")))

gridim = np.mgrid[0:imydim,0:imxdim]
rim = np.sqrt((gridim[0]-imycen)**2+(gridim[1]-imxcen)**2)

def vz(n, m, IMamp): #determine the minimum IMamp (interaction matrix amplitude) to be visible in differential images
	# ds9 = pysao.ds9()
	zern = funz(n, m, IMamp)
	time.sleep(tsleep)
	imzern = optics.stackim(10)
	optics.applydmc(bestflat)
	time.sleep(tsleep)
	imflat = optics.stackim(10)
	return imflat
	# ds9.view((imzern-imflat)*ttmask)

#from above function
ttmask=np.zeros(optics.imdims)
rmask=10
indttmask=np.where(rim/beam_ratio<rmask)
ttmask[indttmask]=1
IMamp=0.001

def make_im_cm(verbose=True, rcond=1e-3):
	#make interaction matrix
	refvec = np.zeros((len(nmarr), ttmask[indttmask].shape[0]*2))
	zernarr = np.zeros((len(nmarr), aperture[indap].shape[0]))
	for i in range(len(nmarr)):
		n, m = nmarr[i]
		zern = funz(n, m, IMamp)
		time.sleep(tsleep)
		imzern = optics.stackim(10)
		optics.applydmc(bestflat)
		time.sleep(tsleep)
		imflat = optics.stackim(10)
		imdiff = imzern - imflat
		Im_diff = processim(imdiff)
		refvec[i] = np.array([np.real(Im_diff[indttmask]), np.imag(Im_diff[indttmask])]).flatten()
		zernarr[i] = zern[indap]

	IM = np.dot(refvec, refvec.T) #interaction matrix
	IMinv = np.linalg.pinv(IM, rcond=rcond)
	cmd_mtx = np.dot(IMinv, refvec).astype(np.float32)
	if verbose:
		print("Recomputed interaction matrix and command matrix")
	return IM, cmd_mtx

IM, cmd_mtx = make_im_cm()
optics.applydmc(bestflat)

def measure_tt(image, cmd_mtx=cmd_mtx):
	tar_ini = processim(image)
	tar = np.array([np.real(tar_ini[indttmask]), np.imag(tar_ini[indttmask])])
	tar = tar.reshape((tar.size, 1))
	coeffs = np.dot(cmd_mtx, tar)
	return coeffs * IMamp

def linearity(mode=0, nlin=20, amp=IMamp, plot=True):
	def genzerncoeffs(i,zernamp):
		'''
		i: zernike mode
		zernamp: Zernike amplitude in DM units to apply
		'''
		n,m=nmarr[i]
		zern=funz(n,m,zernamp)
		time.sleep(tsleep)
		imzern=optics.stackim(10)
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

	optics.applydmc(bestflat)

	fig,axs=plt.subplots(ncols=3,nrows=2,figsize=(12,10),sharex=True,sharey=True)
	#fig.suptitle('rcond='+str(rcond))

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

	return zernamparr, zernampout

def fit_polynomial(x, y, maxdeg=10, abstol=1e-4):
	"""
	Slight extension to np.polyfit that varies the degree.
	"""
	for deg in range(maxdeg):
		p = np.polyfit(x, y, deg=deg)
		errs = np.polyval(p, x) - y
		err = np.sum(errs ** 2)
		if err <= abstol:
			return p, err
	return p, err

def fit_linearity_curves():
	ps = []
	for mode in range(2):
		zin, zout = linearity(mode=mode, plot=False)
		ps.append(fit_polynomial(zin, zout[mode])[0])
	return ps

def command_for_actual(act, p):
	"""
	Uses Newton's method and a polynomial fit to the linearity curve 
	to calculate the command you should send to get an actual response of 'act'.
	"""
	d = len(p)
	obj = lambda x: np.dot(p, x ** np.arange(d-1, -1, -1)) - act
	dobj = lambda x: np.dot(np.arange(d-1, -1, -1), x ** np.arange(d-2, -2, -1))
	return newton(obj, act, fprime=dobj)

if __name__ == "__main__":
	linearity()
