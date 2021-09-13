'''
generate the high order SCC IM
'''

import os
import sys
import itertools
import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy.signal import welch
import matplotlib.pyplot as plt
import matplotlib as mpl
from glob import glob
import multiprocessing as mp
from functions import * #utility functions to use throughout the simulation
from tqdm.contrib.concurrent import process_map
from par_functions import return_vars,propagate,scc,make_IM,make_cov,make_covinvrefj
imagepix,pupilpix,beam_ratio,e,no_phase_offset,xy_dh,grid,N_act,wav0,amp,aperture,indpup,ind_mask_dh,loopx,loopy,freq_loop,pa_loop,n,refdir,iter_arr=return_vars()

if glob('covinvcor_'+refdir+'.npy') == ['covinvcor_'+refdir+'.npy']:
    covinvcor=np.load('covinvcor_'+refdir+'.npy')
else:
    print('making sine and cosine references...')
    try: 
        os.makedirs(refdir)
    except Exception:
        dum=1

    numcores = mp.cpu_count()
    results = process_map(make_IM, iter_arr, max_workers=numcores)

    if glob('fourierarr_pupilpix_'+str(int(pupilpix))+'_N_act_'+str(int(N_act))+'_sin_amp_'+str(int(round(amp/1e-9*wav0/(2.*np.pi))))+'.npy')==[]:
        fourierarr=np.zeros((n,aperture[indpup].shape[0]))
        for result in results:
            cos,sin,i=result
            fourierarr[i]=cos
            fourierarr[i+len(freq_loop)]=sin
        np.save('fourierarr_pupilpix_'+str(int(pupilpix))+'_N_act_'+str(int(N_act))+'_sin_amp_'+str(int(round(amp/1e-9*wav0/(2.*np.pi))))+'.npy',fourierarr)
        results=False
        fourierarr=False

    #make covariance matrix:
    print('making covariance matrix...')

    tpool=mp.Pool(processes=numcores)

    i_arr=list(range(n))
    results=tpool.map(make_cov,i_arr)
    tpool.close()
    tpool.join()

    cov=np.zeros((2*len(freq_loop),2*len(freq_loop)))
    for result in results:
        covi,i=result
        for j in range(i+1):
            cov[i,j]=covi[j]
            cov[j,i]=covi[j] #symmetric matrix

    np.save('cov_'+refdir,cov)

    #invert covariance matrix:
    rcond=1e-5 #for this setup, anything below rcond = 1e-2 makes no difference becuase of the implemented binary masks; this will likely not be true and need to be optimized in real life
    covinv=np.linalg.pinv(cov,rcond=rcond)
    np.save('covinv_'+refdir,covinv)
    cov=False

#dot product by reference vector to not have to save reference images
    refi=np.load(refdir+'/'+str(i)+'.npy')

    tpool=mp.Pool(processes=numcores)
    results=tpool.map(make_covinvrefj,i_arr)
    tpool.close()
    tpool.join()

    covinvcor=np.zeros((n,refi.shape[0]))
    for result in results:
        i,covinvcori=result
        covinvcor[i]=covinvcori
    np.save('covinvcor_'+refdir,covinvcor)
    covinv=False
    results=False

#delete files used to create covariance matrix
    import shutil
    shutil.rmtree(refdir)
