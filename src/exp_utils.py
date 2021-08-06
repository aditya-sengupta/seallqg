import time, os
import numpy as np
from datetime import datetime
from threading import Thread
from queue import Queue

from tt import *
from compute_cmd_int import measure_tt, make_im_cm

tiptiltarr = np.array([tilt.flatten(), tip.flatten()]).T
bestflat = np.load("../data/bestflats/bestflat.npy")

def record_im(out_q, t=1, dt=datetime.now().strftime("%d_%m_%Y_%H_%M_%S")):
    t1 = time.time()
    times = [] 
    # there is no way this over the preallocation will be the bottleneck, it takes nanoseconds to append
    # this would be much easier to feel good about in a lower level language

    while time.time() < t1 + t:
        imval = im.get_data(check=True) # False doesn't wait for a new frame
        times.append(time.time())
        out_q.put(imval)

    out_q.put(None) # this is a placeholder to tell the queue that there's no more images coming
    
    times = np.array(times) - t1
    fname = "/home/lab/asengupta/data/recordings/rectime_dt_{0}.npy".format(dt)
    np.save(fname, times)
    return times
    
def tt_from_queued_image(in_q, dt=datetime.now().strftime("%d_%m_%Y_%H_%M_%S")):
    imflat = np.load("../data/bestflats/imflat.npy")
    fname = "/home/lab/asengupta/data/recordings/rectt_dt_{0}.npy".format(dt)
    ttvals = [] # I have become the victim of premature optimization
    while True:
        # if you don't have any work, take a nap!
        if in_q.empty():
            time.sleep(0.01)
        else:
            v = in_q.get()
            in_q.task_done()
            if v is None:
                ttvals = np.array(ttvals)
                np.save(fname, ttvals)
                return ttvals
            else:
                ttvals.append(measure_tt(v - imflat).flatten())

def record_experiment(command_schedules, path, t=1, verbose=True):
    if not isinstance(command_schedules, list):
        command_schedules = [command_schedules]
    from refresh_imflat import bestflat, imflat
    # bestflat = np.load("../data/bestflats/bestflat.npy")
    # imflat = np.load("../data/bestflats/imflat.npy")
    applydmc(bestflat)
    baseline_ttvals = measure_tt(getim() - imflat)
    if np.any(np.abs(baseline_ttvals) > 0.03):
        warnings.warn("The system may not be aligned: baseline TT is {}".format(baseline_ttvals.flatten()))

    dt = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    path = path + "_time_dt_" + dt + ".npy"

    q = Queue()
    record_thread = Thread(target=lambda q: record_im(q, t=t, dt=dt), args=(q,))
    compute_thread = Thread(target=lambda q: tt_from_queued_image(q, dt=dt), args=(q,))
    command_threads = [Thread(target=c) for c in command_schedules]
    # idk if this is efficient but I don't think it matters when there won't be more than, like, 3 threads here

    if verbose:
        print("Starting recording and commands...")

    record_thread.start()
    compute_thread.start()
    for c in command_threads:
        c.start()

    q.join()
    record_thread.join()
    compute_thread.join()
    for c in command_threads:
        c.join()

    if verbose:
        print("Done with experiment.")

    applydmc(bestflat)

    times = np.load("/home/lab/asengupta/data/recordings/rectime_dt_{0}.npy".format(dt))
    ttvals = np.load("/home/lab/asengupta/data/recordings/rectt_dt_{0}.npy".format(dt))
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
