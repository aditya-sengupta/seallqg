# authored by Aditya Sengupta

import numpy as np
from ..optics import tt_to_dmc, getdmc

def openloop(measurement):
    """
    Arguments
    ---------
    measurement : np.ndarray, (2,)
    The (tilt, tip) measurement to work off.

    Returns
    -------
    command : np.ndarray, (ydim, xdim)
    The command to be put on the DM.
    """
    return getdmc()

def integrate(measurement, gain=0.1, leak=1.0):
    """
    Simple integrator control.

    Arguments
    ---------
    measurement : np.ndarray, (2,)
    The (tilt, tip) measurement to work off.

    gain : float
    The integrator gain (scaling factor for the ideal DMC)

    leak : float
    The integrator leak ("forgetting factor" for the existing DMC)

    Returns
    -------
    command : np.ndarray, (ydim, xdim)
    The command to be put on the DM.
    """
    dmcn = tt_to_dmc(measurement)
    return gain * dmcn + leak * getdmc()

def kalman_integrate(measurement, x, kf, gain=0.1, leak=1.0):
    x = kf.predict(kf.update(x, measurement))
    dmcn = tt_to_dmc(kf.measure(x))
    return gain * dmcn + leak * getdmc()

def lqg(measurement):
    """
    Linear-quadratic-Gaussian control.
    """
    pass
