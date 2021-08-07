import time
import numpy as np
from datetime import datetime
from threading import Thread
from queue import Queue
from functools import partial

from tt import *
from compute_cmd_int import measure_tt

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

    out_q.put(None) 
    # this is a placeholder to tell the queue that there's no more images coming
    
    times = np.array(times) - t1
    fname = "/home/lab/asengupta/data/recordings/rectime_dt_{0}.npy".format(dt)
    np.save(fname, times)
    return times
    
def tt_from_queued_image(in_q, out_q, cmd_mtx, dt=datetime.now().strftime("%d_%m_%Y_%H_%M_%S")):
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
                ttval = measure_tt(v - imflat, cmd_mtx=cmd_mtx).flatten()
                out_q.put(ttval)
                ttvals.append(ttval)

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

def integrator_schedule(q, t=1, delay=0.01, gain=0.1, leak=1.0):
    t1 = time.time()
    while time.time() < t1 + t:
        ti = time.time()
        tt = q.get()
        q.task_done()
        dmcn = tt_to_dmc(tt)
        applydmc(leak * getdmc() + gain * dmcn) 
        time.sleep(max(0, delay - (time.time() - ti)))

def kalman_schedule(q, kf, t=1, delay=0.01):
    pass

def record_experiment(path, control_schedule=lambda: None, dist_schedule=lambda: None, t=1, verbose=True):
    from refresh_imflat import bestflat, imflat
    from compute_cmd_int import make_im_cm
    # bestflat = np.load("../data/bestflats/bestflat.npy")
    # imflat = np.load("../data/bestflats/imflat.npy")
    applydmc(bestflat)
    _, cmd_mtx = make_im_cm()
    baseline_ttvals = measure_tt(getim() - imflat)
    if np.any(np.abs(baseline_ttvals) > 0.03):
        warnings.warn("The system may not be aligned: baseline TT is {}".format(baseline_ttvals.flatten()))

    dt = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    path = path + "_time_dt_" + dt + ".npy"

    q_compute = Queue()
    q_control = Queue()
    record_thread = Thread(target=partial(record_im, t=t, dt=dt), args=(q_compute,))
    compute_thread = Thread(target=partial(tt_from_queued_image, dt=dt), args=(q_compute, q_control, cmd_mtx,))
    control_thread = Thread(target=control_schedule, args=(q_control,))
    command_thread = Thread(target=dist_schedule)

    if verbose:
        print("Starting recording and commands...")

    record_thread.start()
    compute_thread.start()
    control_thread.start()
    command_thread.start()

    q_compute.join()
    q_control.join()
    record_thread.join()
    compute_thread.join()
    control_thread.join()
    command_thread.join()

    if verbose:
        print("Done with experiment.")

    applydmc(bestflat)

    times = np.load("/home/lab/asengupta/data/recordings/rectime_dt_{0}.npy".format(dt))
    ttvals = np.load("/home/lab/asengupta/data/recordings/rectt_dt_{0}.npy".format(dt))
    np.save(path, times)
    path = path.replace("time", "tt")
    np.save(path, ttvals)
    return times, ttvals 
