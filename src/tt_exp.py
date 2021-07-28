# my tip-tilt experiments
import numpy as np
from datetime import datetime
import tqdm
from threading import Thread
from queue import Queue


from ao import amplitude
from tt import *
from compute_cmd_int import *
from image import *


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
        uconvert_matrix[mode] = [dmdrange /  (cm2x - cm1x), dmdrange / (cm2y - cm1y)]

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

def record_im(t=1, dt=datetime.now().strftime("%d_%m_%Y_%H_%M")):
    nimages = int(np.ceil(t / 1e-3)) * 5 # the 5 is a safety factor and 1e-3 is a fake delay
    imvals = np.empty((nimages, 320, 320))
    i = 0
    t1 = time.time()
    times = np.zeros((nimages,))

    while time.time() < t1 + t:
        imvals[i] = im.get_data(check=True) # False doesn't wait for a new frame
        times[i] = time.time()
        i += 1
    
    times = times - t1
    times = times[:i]
    imvals = imvals[:i]
    fname_im = "/home/lab/asengupta/data/recordings/recim_dt_{0}.npy".format(dt)
    fname_t = "/home/lab/asengupta/data/recordings/rectime_dt_{0}.npy".format(dt)
    np.save(fname_im, imvals)
    np.save(fname_t, times)
    print("np.load('{0}')".format(fname_im))
    return times, imvals, fname_im

def tt_from_im(fname):
    applydmc(bestflat)
    imflat = stack(100)
    ims = np.load(fname)
    ttvals = np.zeros((ims.shape[0], 2))
    for (i, im) in enumerate(ims):
        ttvals[i] = measure_tt(im - imflat)
    
    fname_tt = fname.replace("im", "tt")
    np.save(fname_tt, ttvals)
    return ttvals

def record_experiment(command_schedule, path, t=1, verbose=True):
    applydmc(bestflat)
    dt = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    path = path + "_dt_" + dt + ".npy"

    record_thread = threading.Thread(target=lambda: record_im(t=t, dt=dt))
    command_thread = threading.Thread(target=command_schedule)

    if verbose:
        print("Starting recording and commands...")

    record_thread.start()
    command_thread.start()
    record_thread.join()
    command_thread.join()

    if verbose:
        print("Done with experiment.")

    applydmc(bestflat)

    times = np.load("/home/lab/asengupta/data/recordings/rectime_dt_{0}.npy".format(dt))
    ttvals = tt_from_im("/home/lab/asengupta/data/recordings/recim_dt_{0}.npy".format(dt))
    np.save(path, times)
    path = path.replace("time", "tt")
    np.save(path, ttvals)
    return times, ttvals 

def record_usteps(tip_amp=0.1, tilt_amp=0.0):
    path = "../data/usteps/ustep_time_amps_{1}_{2}".format(tip_amp, tilt_amp)
    def command_schedule(tip_amp, tilt_amp):
        time.sleep(0.5)
        funz(1, -1, tip_amp, bestflat=bestflat)
        funz(1, 1, tilt_amp, bestflat=bestflat)

    return record_experiment(command_schedule, path)

def record_usteps_in_circle(niters=10, amp=0.1, nangles=12):
    for _ in tqdm.trange(niters):
        for ang in np.arange(0, 2 * np.pi, np.pi / 6):
            record_usteps(amp * np.cos(ang), amp * np.sin(ang), verbose=False)

def record_sinusoids(delay=1e-2):
    for mode in [0, 1]:
        nsteps_per_osc = 50
        nosc = 50
        times = np.arange(0, nsteps_per_osc * nosc * delay, delay)
        f = 1
        amplitude = 1.0
        dmfn = lambda cmd: funz(1, 2*mode-1, cmd, bestflat=bestflat)
        path = "../data/sinusoid/sinusoid_amp_{0}_nsteps_{1}_nosc_{2}_f_{3}_delay_{4}_mode_{5}".format(round(amplitude, 3), nsteps_per_osc, nosc, f, delay, mode)

        def command_schedule():
            control_commands = amplitude * np.diff(np.sin(2 * np.pi * times * f))
            for (i, cmd) in enumerate(control_commands):
                dmfn(cmd)
                time.sleep(delay)
            
        record_experiment(command_schedule, path)
    
def record_atm_vib(atm=0, vib=2, delay=1e-2, scaledown=10):
    """
    Plays vibrations/turbulence on the DM - scaled down in amplitude and in time by a factor of 10.
    """
    fname = "../data/sims/ol_atm_{0}_vib_{1}.npy".format(atm, vib)
    path = "../data/atmvib/atm_{0}_vib_{1}.npy".format(atm, vib)
    
    def command_schedule():
        control_commands = np.diff(np.load(fname), axis=0) / scaledown
        for (i, cmd) in enumerate(control_commands):
            applytiptilt(cmd[0], cmd[1], verbose=False)
            time.sleep(delay)

    record_experiment(command_schedule, path)

def clear_images():
    """
    Clears ../data/recordings/recim_*.
    These files take up a lot of space and often aren't necessary or useful after TTs have been extracted, 
    so this just deletes them after prompting the user to make sure they know what they're doing.
    """
    verify = input("WARNING: this function deletes all the saved timeseries of images. Are you sure you want to do this? [y/n] ")
    if verify == 'y':
        for file in os.listdir("/home/lab/asengupta/data/recordings"):
            if file.startswith("recim"):
                print("Deleting " + file)
                os.remove("/home/lab/asengupta/data/recordings/" + file)
    print("Files deleted.")

def tt_to_dmc(tt):
    """
    Converts a measured tip-tilt value to an ideal DM command, using the amplitude of the linearity matrix.
    
    Arguments
    ---------
    tt : np.ndarray, (2,)
    The tip and tilt values.

    Returns
    -------
    dmc : np.ndarray, (dm_x, dm_y)
    The corresponding DM command.
    """
    pass

def integrator_control(gain=0.1, leak=1, niters=1000):
    """
    Runs closed-loop integrator control with a fixed gain.
    
    Arguments
    ---------
    gain : float
    The gain value to use for the integrator.

    Returns
    -------
    times : np.ndarray, (niters,)
    The times at which new tip-tilt values are measured.

    ttvals : np.ndarray, (niters, 2)
    The tip-tilt values in closed loop.
    """
    for i in tqdm.trange(niters):
        frame = getim()
        tt = measure_tt(frame)
        dmcn = tt_to_dmc(tt)
        applydmc(leak * getdmc() + gain * dmcn) 
