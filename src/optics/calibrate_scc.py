'''
generate the high order SCC IM
'''

import os
from os import path
import sys
import itertools
import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy.signal import welch
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
# from toolz import pipe

from .ao import * 
from .par_functions import return_vars, propagate, scc, make_IM, make_cov, make_covinvrefj
from ..utils import joindata

imagepix, pupilpix, beam_ratio, e, no_phase_offset,xy_dh,grid,N_act,wav0,amp,aperture,indpup,ind_mask_dh,loopx,loopy,freq_loop,pa_loop,n,refdir,iter_arr=return_vars()

def make_im_scc_howfs():
	covinvcor_path = joindata(path.join("scc_sim", "covinvcor_{0}.npy".format(refdir)))
	if path.isfile(covinvcor_path):
		covinvcor = np.load(covinvcor_path)
	else:
		print('Making sine and cosine references')
		try: 
			os.makedirs(joinsimdata(refdir))
		except Exception:
			dum = 1

		results = process_map(make_IM, iter_arr, max_workers=mp.cpu_count())

		fourierarr_path = joinsimdata('fourierarr_pupilpix_'+str(int(pupilpix))+'_N_act_'+str(int(N_act))+'_sin_amp_'+str(int(round(amp/1e-9*wav0/(2.*np.pi))))+'.npy')
		if not path.isfile(fourierarr_path):
			fourierarr=np.zeros((n,aperture[indpup].shape[0]))
			for result in results:
				cos,sin,i = result
				fourierarr[i] = cos
				fourierarr[i+len(freq_loop)] = sin
			np.save(fourierarr_path, fourierarr)
			results = False
			fourierarr = False

		tpool.close()
		tpool.join()

		#make covariance matrix:
		print('Making covariance matrix')

		tpool = mp.Pool(processes=numcores)

		i_arr = list(range(n))
		results = tpool.map(make_cov, i_arr)
		tpool.close()
		tpool.join()

		cov = np.zeros((2*len(freq_loop),2*len(freq_loop)))
		for result in results:
			covi, i = result
			for j in range(i+1):
				cov[i,j] = covi[j]
				cov[j,i] = covi[j] #symmetric matrix

		np.save(joinsimdata('cov_'+refdir),cov)

		#invert covariance matrix:
		rcond=  1e-5 #for this setup, anything below rcond = 1e-2 makes no difference becuase of the implemented binary masks; this will likely not be true and need to be optimized in real life
		covinv = np.linalg.pinv(cov,rcond=rcond)
		np.save(joinsimdata('covinv_'+refdir),covinv)
		cov = False

		#dot product by reference vector to not have to save reference images
		refi=np.load(joinsimdata(refdir+'/'+str(i)+'.npy'))
		
		tpool=mp.Pool(processes=numcores)
		results=tpool.map(make_covinvrefj,i_arr)
		tpool.close()
		tpool.join()
		
		covinvcor=np.zeros((n,refi.shape[0]))
		for result in results:
			i,covinvcori=result
			covinvcor[i]=covinvcori
		np.save(joinsimdata('covinvcor_'+refdir), covinvcor)
		covinv=False
		results=False
		return covinvcor

		#delete files used to create covariance matrix
		# import shutil
		# shutil.rmtree(refdir)
