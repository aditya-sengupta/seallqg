# my tip-tilt experiments
import numpy as np
from datetime import datetime
import tqdm

from tt import *
from compute_cm_im import *

def find_limits(min_amp=1e-4, max_amp=1e-0):
    """
    Applies tips and tilts from the best flat to find the feasible range of DM TT commands.

    Arguments: none

    Returns: (list, list); the first is min_tip, max_tip; the second is min_tilt, max_tilt
    """
    limits = []
    for dmfn in [applytip, applytilt]:
        for sgn in [-1, +1]:
            applydmc(bestflat)
            applied_cmd = 0.0
            for step_size in 10 ** np.arange(np.log10(max_amp), np.log10(min_amp)-1, -1):
                time.sleep(tsleep)
                in_range = True
                while in_range:
                    applied_cmd += sgn * step_size
                    in_range = all(dmfn(sgn * step_size))
                applied_cmd -= sgn * step_size
                dmfn(-sgn * step_size, False) # move it back within range, and retry with the smaller step size
            limits.append(applied_cmd)
                
    applydmc(bestflat)
    return limits[:2], limits[2:] # the first is min_tip, max_tip; the second is min_tilt, max_tilt

def poisson_ll(data, lam):
    from scipy.special import gammaln
    
    n = len(data)
    return -n * lam + np.sum(data) * np.log(lam) - np.sum(np.nan_to_num(gammaln(imdiff), posinf=0))

def get_noise(delay=1e-2):
    im1 = getim()
    time.sleep(delay)
    im2 = getim()
    imdiff = np.abs(im2 - im1)
    return imdiff

def noise_fishing(delay=1e-2):
    """
    Take a differential image and fit a Poisson distribution to it.
    """
    imdiff = get_noise(delay).ravel()
    lam = np.mean(imdiff)
    ll = poisson_ll(imdiff, lam)
    return lam, ll

def masked_noise_fishing(delay=1e-2):
    im = getim()
    mean, sd = np.mean(im), np.std(im)
    imdiff = get_noise(delay)[im < mean + sd].ravel()
    lam = np.mean(imdiff)
    ll = poisson_ll(imdiff, lam)
    return lam, ll

def random_removal_noise_fishing(delay=1e-2):
    random_mask = np.ones((320, 320), dtype=bool)
    for _ in range(2150):
        r, c = np.random.randint(0, 319, (2,))
        random_mask[r][c] = False
    imdiff = get_noise(delay)[random_mask].ravel()
    lam = np.mean(imdiff)
    ll = poisson_ll(imdiff, lam)
    return lam, ll

def measurement_noise_diff_image():
    """
    Takes a series of differential images at different TT locations and with varying delays.
    """
    delays = [1e-4, 1e-3, 1e-2, 1e-2, 1e-1, 1]
    niters = 10
    applied_tts = [[0.0, 0.0]]
    tt_vals = np.zeros((len(applied_tts), len(delays), 2))

    for (i, applied_tt) in enumerate(applied_tts): # maybe change this later
        applytiptilt(applied_tt[0], applied_tt[1])
        for (j, d) in enumerate(delays):
            for _ in range(niters):
                im1 = getim()
                time.sleep(d)
                im2 = getim()
                imdiff = im2 - im1
                tar_ini = processim(imdiff)
                tar = np.array([np.real(tar_ini[indttmask]), np.imag(tar_ini[indttmask])]).flatten()	
                coeffs = np.dot(cmd_mtx, tar)
                tt_vals[i][j] += coeffs * IMamp

    ttvals = tt_vals / niters
    applydmc(bestflat)
    fname = "/home/lab/asengupta/data/measurenoise_ttvals_{}".format(datetime.now().strftime("%d_%m_%Y_%H"))
    np.save(fname, ttvals)
    return ttvals
    
def command_linearity():
    pass

def uniformity():
    pass

def amplitude_linearity():
    pass

def unit_steps(min_amp, max_amp, steps_amp, steps_ang=12, tsleep=tsleep, nframes=50):
    angles = np.linspace(0.0, 2 * np.pi, steps_ang)
    amplitudes = np.linspace(min_amp, max_amp, steps_amp)
    for amp in tqdm.tqdm(amplitudes):
        for ang in angles:
            applydmc(bestflat)
            time.sleep(tsleep) # vary this later: for now I'm after steady-state error
            imflat = stack(nframes)
            applytiptilt(amp * np.cos(ang), amp * np.sin(ang))
            time.sleep(tsleep)
            imtt = stack(nframes)
            fname = "/home/lab/asengupta/data/unitstep_amp_{0}_ang_{1}_dt_{2}".format(round(amp, 3), round(ang, 3), datetime.now().strftime("%d_%m_%Y_%H"))
            fname += ".npy"
            np.save(fname, imtt-imflat)
    applydmc(bestflat)

def sinusoids():
    pass
