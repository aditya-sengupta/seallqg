# authored by Aditya Sengupta

import numpy as np
from datetime import datetime
import tqdm

from tt import *
from compute_cmd_int import *
from image import *

# find limits and poisson ll are in git somewhere

def get_noise(delay=1e-2):
    im1 = getim()
    time.sleep(delay)
    im2 = getim()
    imdiff = im2 - im1
    return imdiff

def measurement_noise_diff_image():
    """
    Takes a series of differential images at different TT locations and with varying delays.
    """
    delays = [1e-4, 1e-3, 1e-2, 1e-2, 1e-1, 1]
    niters = 10
    applied_tts = [[0.0, 0.0]]
    tt_vals = np.zeros((len(applied_tts), len(delays), 2))

    for (i, applied_tt) in enumerate(applied_tts): # maybe change this later
        print("Applying TT {}".format(applied_tt))
        applytiptilt(applied_tt[0], applied_tt[1])
        for (j, d) in enumerate(tqdm.tqdm(delays)):
            for _ in range(niters):
                im1 = getim()
                time.sleep(d)
                im2 = getim()
                imdiff = im2 - im1
                tt_vals[i][j] += measure_tt(imdiff)

    ttvals = tt_vals / niters
    applydmc(bestflat)
    fname = "/home/lab/asengupta/data/measurenoise_ttvals_{}".format(datetime.now().strftime("%d_%m_%Y_%H"))
    np.save(fname, ttvals)
    return ttvals
    
def tt_center_noise(nsteps=1000, delay=1e-2):
    ttvals = np.zeros((0,2))
    for _ in tqdm.trange(nsteps):
        time.sleep(delay)
        im = getim()
        ttvals = np.vstack((ttvals, measure_tt(im - imflat)))

    np.save("/home/lab/asengupta/data/tt_center_noise/tt_center_noise_nsteps_{0}_delay_{1}_dt_{2}".format(str(nsteps), str(delay), datetime.now().strftime("%d_%m_%Y_%H")), ttvals)
    return ttvals

def uconvert_ratio(amp=1.0):
    expt_init = get_expt()
    set_expt(1e-5)
    uconvert_matrix = np.zeros((2,2))
    for (mode, dmcmd) in enumerate([applytip, applytilt]):
        applydmc(bestflat)
        dmcmd(amp)
        dm2 = getdmc()
        cm2x = []
        while len(cm2x) != 1:
            im2 = stack(100)
            cm2x, cm2y = np.where(im2 == np.max(im2))

        applydmc(bestflat)
        dmcmd(-amp)
        dm1 = getdmc()
        cm1x = []
        while len(cm1x) != 1:
            im1 = stack(100)
            cm1x, cm1y = np.where(im1 == np.max(im1))

        dmdiff = aperture * (dm2 - dm1)
        
        dmdrange = np.max(dmdiff) - np.min(dmdiff)
        uconvert_matrix[mode] = [dmdrange /  (cm2y - cm1y), dmdrange / (cm2x - cm1x)]

    set_expt(expt_init)
    applydmc(bestflat)
    return uconvert_matrix

def noise_floor(niters=100):
    delays = np.array([5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e-0])
    noises = np.zeros((len(delays), 2))
    for (i, delay) in enumerate(delays):
        for _ in tqdm.trange(niters):
            noises[i] += np.abs(measure_tt(getim() - (time.sleep(delay) or getim())))

    noises /= niters
    return noises
