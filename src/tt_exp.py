# my tip-tilt experiments
import numpy as np
from datetime import datetime
import tqdm
import threading

from ao import amplitude
from tt import *
from compute_cmd_int import *

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
    return -n * lam + np.sum(data) * np.log(lam) - np.sum(np.nan_to_num(gammaln(data), posinf=0))

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

def apply_usteps(min_amp, max_amp, steps_amp, steps_ang=12, tsleep=tsleep, nframes=50):
    angles = np.linspace(0.0, 2 * np.pi, steps_ang)
    amplitudes = np.linspace(min_amp, max_amp, steps_amp)
    for amp in tqdm.tqdm(amplitudes):
        for ang in angles:
            applydmc(bestflat)
            time.sleep(tsleep) # vary this later: for now I'm after steady-state error
            applytiptilt(amp * np.cos(ang), amp * np.sin(ang))
            time.sleep(tsleep)
    applydmc(bestflat)

def apply_sinusoids(delay=1e-2):
    nsteps_per_osc = 50
    nosc = 50
    times = np.arange(0, nsteps_per_osc * nosc * delay, delay)
    f = 1
    lims = [[-0.05, 0.15], [-0.05, 0.15]]
    for (j, (dmcmd, lim)) in enumerate([("tip", lims[0]), ("tilt", lims[1])]):
        print("Applying " + dmcmd)
        dmfn = eval("apply" + dmcmd)
        amplitude = 0.5 * min(np.abs(lim))
        control_commands = amplitude * np.diff(np.sin(2 * np.pi * times * f))
        ttresponse = np.zeros_like(control_commands)
        for (i, cmd) in enumerate(control_commands):
            dmfn(cmd)
            time.sleep(delay)
            ttresponse[i] = measure_tt()[j]
        fname = "/home/lab/asengupta/data/sinusoid_amp_{0}_nsteps_{1}_nosc_{2}_f_{3}_delay_{4}_mode_{5}_dt_{6}".format(round(amplitude, 3), nsteps_per_osc, nosc, f, delay, dmcmd, datetime.now().strftime("%d_%m_%Y_%H"))
        np.save(fname, ttresponse)
    
def uconvert_ratio(amp=0.1):
    applydmc(bestflat)
    applytilt(amp)
    dm2 = getdmc()
    cm2 = measure_tt(stack(1000))[0] # stack? does that accurately represent what I'm after?
    applytip(-2*amp)
    dm1 = getdmc()
    cm1 = measure_tt(stack(1000))[0]
    applydmc(bestflat)
    dmdiff = aperture * (dm2 - dm1)
    return (np.max(dmdiff) - np.min(dmdiff)) /  (cm2 - cm1)

def noise_floor(niters=100):
    delays = np.array([5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e-0])
    noises = np.zeros((len(delays), 2))
    for (i, delay) in enumerate(delays):
        for _ in tqdm.trange(niters):
            noises[i] += np.abs(measure_tt(getim() - (time.sleep(delay) or getim())))

    noises /= niters
    return noises

def record_im(t=0.5):
    nimages = int(np.ceil(t / 1e-3)) * 3 # the 3 is a safety factor and 1e-3 is a fake delay
    imvals = np.empty((nimages, 320, 320))
    i = 0
    t1 = time.time()
    times = np.empty((nimages,))

    while time.time() < t1 + t:
        imvals[i] = im.get_data(check=False) - imflat
        times[i] = time.time()
        i += 1
        #time.sleep(max(0, delay - (time.time() - tl)))
    fname = "/home/lab/asengupta/data/recordings/recim_dt_{1}_delay_{0}".format(delay, datetime.now().strftime("%d_%m_%Y_%H_%M"))
    np.save(fname, imvals)
    print("np.load('{0}')".format(fname))
    print(i)
    return imvals

def record_tt(t=0.5, delay=1e-3):
    ttvals = np.zeros((0, 2))
    t1 = time.time()
    while time.time() < t1 + t:
        tl = time.time()
        ttvals = np.vstack((ttvals, measure_tt(im.get_data(check=False) - imflat)))
        time.sleep(max(0, delay - (time.time() - tl)))
    fname = "/home/lab/asengupta/data/recordings/rectt_dt_{1}_delay_{0}".format(delay, datetime.now().strftime("%d_%m_%Y_%H_%M"))
    np.save(fname, ttvals)
    print("np.load('{0}')".format(fname))
    return ttvals

def record_usteps():
    applydmc(bestflat)
    threading.Thread(target=record_tt).start()
    threading.Thread(target=lambda: time.sleep(10) or applytip(0.1)).start()