import time, os
import numpy as np
from datetime import datetime
from threading import Thread

from tt import *
from compute_cmd_int import measure_tt

tiptiltarr = np.array([tilt.flatten(), tip.flatten()]).T
bestflat = np.load("../data/bestflats/bestflat.npy")

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
        ttvals[i] = measure_tt(im - imflat).flatten()
    
    fname_tt = fname.replace("im", "tt")
    np.save(fname_tt, ttvals)
    return ttvals

def record_experiment(command_schedule, path, t=1, verbose=True):
    bestflat = np.load("../data/bestflats/bestflat.npy")
    # applydmc(bestflat)
    # imflat = stack(100)
    dt = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    path = path + "_time_dt_" + dt + ".npy"

    record_thread = Thread(target=lambda: record_im(t=t, dt=dt))
    command_thread = Thread(target=command_schedule)

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

def tt_to_dmc(tt):
    """
    Converts a measured tip-tilt value to an ideal DM command.
    
    Arguments
    ---------
    tt : np.ndarray, (2, 1)
    The tip and tilt values.

    Returns
    -------
    dmc : np.ndarray, (dm_x, dm_y)
    The corresponding DM command.
    """
    return np.matmul(tiptiltarr, -tt).reshape((ydim,xdim))

def clear_images():
    """
    Clears ../data/recordings/recim_*.
    These files take up a lot of space and often aren't necessary or useful after TTs have been extracted, 
    so this just deletes them after prompting the user to make sure they know what they're doing.

    No parameters, no return values.
    """
    verify = input("WARNING: this function deletes all the saved timeseries of images. Are you sure you want to do this? [y/n] ")
    if verify == 'y':
        for file in os.listdir("/home/lab/asengupta/data/recordings"):
            if file.startswith("recim"):
                print("Deleting " + file)
                os.remove("/home/lab/asengupta/data/recordings/" + file)
    print("Files deleted.")

def refresh_imflat():
    dmc = getdmc()
    applydmc(bestflat)
    imflat = stack(100)
    applydmc(dmc)