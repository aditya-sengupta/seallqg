import numpy as np
from datetime import datetime
import tqdm
from threading import Thread
from queue import Queue

from tt import *
from compute_cmd_int import *
from image import *

tiptiltarr = np.array([tilt.flatten(), tip.flatten()])

def tt_to_dmc(tt):
    """
    Converts a measured tip-tilt value to an ideal DM command, using the amplitude of the linearity matrix.
    
    Arguments
    ---------
    tt : np.ndarray, (2, 1)
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
