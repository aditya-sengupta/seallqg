# authored by Aditya Sengupta

import time
import os
import numpy as np
import warnings
from datetime import datetime
from threading import Thread
from queue import Queue
from functools import partial

from ..utils import joindata
from ..optics import getim, applydmc
from ..optics import measure_tt, make_im_cm
from ..optics import refresh

bestflat = np.load(joindata("bestflats/bestflat.npy"))

def record_im(out_q, t=1, dt=datetime.now().strftime("%d_%m_%Y_%H_%M_%S")):
    t1 = time.time()
    times = [] 
    # there is no way this over the preallocation will be the bottleneck, it takes nanoseconds to append
    # this would be much easier to feel good about in a lower level language

    while time.time() < t1 + t:
        imval = getim()
        times.append(time.time())
        out_q.put(imval)

    out_q.put(None) 
    # this is a placeholder to tell the queue that there's no more images coming
    
    times = np.array(times) - t1
    fname = joindata("recordings/rectime_dt_{0}.npy".format(dt))
    np.save(fname, times)
    return times
    
def tt_from_queued_image(in_q, out_q, cmd_mtx, dt=datetime.now().strftime("%d_%m_%Y_%H_%M_%S")):
    imflat = np.load(joindata("bestflats/imflat.npy"))
    fname = joindata("recordings/rectt_dt_{0}.npy".format(dt))
    ttvals = [] # I have become the victim of premature optimization
    while True:
        # if you don't have any work, take a nap!
        if in_q.empty():
            time.sleep(0.01)
        else:
            v = in_q.get()
            in_q.task_done()
            if v is not None:
                ttval = measure_tt(v - imflat, cmd_mtx=cmd_mtx).flatten()
                out_q.put(ttval)
                ttvals.append(ttval)
            else:
                ttvals = np.array(ttvals)
                np.save(fname, ttvals)
                return ttvals

def control_schedule(q, control, t=1, delay=0.01):
    """
    The SEAL schedule for a controller.

    Arguments
    ---------
    q : Queue
    The queue to poll for new tip-tilt values.

    controller : callable
    The function to execute control.
    """
    t1 = time.time()
    while time.time() < t1 + t:
        # ti = time.time()
        if q.empty():
            time.sleep(delay/2)
        else:
            tt = q.get()
            q.task_done()
            applydmc(control(tt))
            # time.sleep(max(0, delay - (time.time() - ti)))

def record_experiment(path, control_schedule, dist_schedule, t=1, verbose=True):
    bestflat, imflat = refresh()
    applydmc(bestflat)
    _, cmd_mtx = make_im_cm()
    baseline_ttvals = measure_tt(getim() - imflat, cmd_mtx=cmd_mtx)
    if np.any(np.abs(baseline_ttvals) > 0.03):
        warnings.warn("The system may not be aligned: baseline TT is {}".format(baseline_ttvals.flatten()))

    dt = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    path = joindata(path) + "_time_dt_" + dt + ".npy"

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

    timepath = joindata("recordings/rectime_dt_{0}.npy".format(dt))
    ttpath = joindata("recordings/rectt_dt_{0}.npy".format(dt))
    times = np.load(timepath)
    ttvals = np.load(ttpath)
    np.save(path, times)
    path = path.replace("time", "tt")
    np.save(path, ttvals)
    os.remove(timepath)
    os.remove(ttpath)
    return times, ttvals 
