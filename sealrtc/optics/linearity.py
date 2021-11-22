"""
Linearity plotting.
"""

import time
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from .utils import nmarr
from ..utils import joindata, get_timestamp

def linearity(optics, nlin=20, plot=True, save=True, rcond=1e-3):
	bestflat, imflat = optics.refresh()
	_, cmd_mtx = make_im_cm(rcond=rcond)
	zernamparr = sweep_amp * np.linspace(-1.5,1.5,nlin)
	zernampout = np.zeros((len(nmarr),len(nmarr),nlin))
	for nm in range(len(nmarr)):
		for i in range(nlin):
			zernamp = zernamparr[i]
			coeffsout = optics.genzerncoeffs(nm, zernamp).flatten()
			zernampout[nm,:,i] = coeffsout

	optics.applybestflat()

	if save:
		tstamp = get_timestamp()
		path_out = joindata("linearity", f"lout_{tstamp}.npy")
		np.save(joindata("linearity", f"lin_{tstamp}.npy"), zernamparr)
		np.save(path_out, zernampout)
		print(f"Saved output to {path_out}")

	if plot:
		plot_linearity(zernamparr, zernampout, rcond)

	return zernamparr, zernampout

def plot_linearity(zernamparr, zernampout, rcond=None):
	wfe_in_microns = False
	if wfe_in_microns:
		conv = dmc2wf
		unit = "$\\mu$m"
	else:
		conv = 1
		unit = "DM units"
	fig, axs = plt.subplots(ncols=3,nrows=2,figsize=(12,10),sharex=True,sharey=True)
	if rcond is not None:
		fig.suptitle(f"rcond = {rcond}")
	else:
		fig.suptitle("Linearity in the first five Zernike modes on FAST")

	colors = mpl.cm.viridis(np.linspace(0,1,len(nmarr)))
	axarr=[axs[0,0],axs[0,1],axs[0,2],axs[1,0],axs[1,1],axs[1,2]]
	fig.delaxes(axarr[-1])
	for i in range(len(nmarr)):
		ax=axarr[i]
		ax.set_title('n,m='+str(nmarr[i][0])+','+str(nmarr[i][1]))
		if i==4:
			ax.plot(zernamparr*conv,zernamparr*conv,lw=1,color='k',ls='--',label='y=x')
			for j in range(len(nmarr)):
				if j==i:
					ax.plot(zernamparr*conv,zernampout[i,i,:]*conv,lw=2,color=colors[j],label='n,m='+str(nmarr[i][0])+','+str(nmarr[i][1]))
				else:
					ax.plot(zernamparr*conv,zernampout[i,j,:]*conv,lw=1,color=colors[j],label='n,m='+str(nmarr[j][0])+','+str(nmarr[j][1]))
		else:
			ax.plot(zernamparr*conv,zernamparr*conv,lw=1,color='k',ls='--')
			for j in range(len(nmarr)):
				if j==i:
					ax.plot(zernamparr*conv,zernampout[i,i,:]*conv,lw=2,color=colors[j])
				else:
					ax.plot(zernamparr*conv,zernampout[i,j,:]*conv,lw=1,color=colors[j])

	axarr[4].legend(bbox_to_anchor=(1.05,0.9))
	axarr[4].set_xlabel(f'input ({unit} WFE, PV)')
	axarr[3].set_xlabel(f'input ({unit} WFE, PV)')
	axarr[3].set_ylabel(f'reconstructed output ({unit} WFE, PV)')
	axarr[0].set_ylabel(f'reconstructed output ({unit} WFE, PV)')

# fit_polynomial stuff removed on 2021-10-10, see git history before that to recover
