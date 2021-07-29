import numpy as np
from datetime import datetime
import tqdm
from threading import Thread
from queue import Queue

from tt import *
from compute_cmd_int import *
from image import *
from tt_exp import record_im, tt_from_im

tiptiltarr = np.array([tilt.flatten(), tip.flatten()]).T

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

def measure(out_q):
    t1 = time.time()

    ttval = measure_tt(getim() - imflat)
    out_q.put(ttval)
    
def record_closedloop(controller, path, t=1, verbose=True):
    applydmc(bestflat)
    dt = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    path = path + "_dt_" + dt + ".npy"

    q = Queue()
    record_thread = Thread(target=lambda: record_im(t=t, dt=dt))
    control_thread = Thread(target=controller, args=(q,))
    measure_thread = Thread(target=measure, args=(q,))

    if verbose:
        print("Starting recording and commands...")

    record_thread.start()
    control_thread.start()
    measure_thread.start()
    record_thread.join()
    q.join()

    if verbose:
        print("Done with experiment.")

    applydmc(bestflat)

    times = np.load("/home/lab/asengupta/data/recordings/rectime_dt_{0}.npy".format(dt))
    ttvals = tt_from_im("/home/lab/asengupta/data/recordings/recim_dt_{0}.npy".format(dt))
    np.save(path, times)
    path = path.replace("time", "tt")
    np.save(path, ttvals)
    return times, ttvals 

def nocontrol(measurement):
    return getdmc()

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
