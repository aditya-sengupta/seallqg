# authored by Aditya Sengupta

import numpy as np
from functools import partial

from .observer import identity, make_kf_observer
from ..optics import optics, zcoeffs_to_dmc

def control(measurement, observer, controller, logger, **kwargs):
    """
    Arguments
    ---------
    measurement : np.ndarray, (2,)
    The (tilt, tip) measurement to work off.

    observer : callable
    The filter to translate from a measurement to a state.

    controller : callable
    The function to return the optimal control command from the state estimate.

    kwargs : any
    Any parameters to be passed into the observer and controller

    Returns
    -------
    command : np.ndarray, (ydim, xdim)
    The command to be put on the DM.
    """
    state = observer(measurement[:2], **kwargs) # biryani
    u = controller(state, **kwargs)
    return u

# Controllers

def ol_controller(state, **kwargs):
    """
    Do nothing.
    """
    return np.array([0, 0]).astype(np.float32), optics.getdmc()

def integrator(state, **kwargs):
    """
    Simple integrator control.

    Arguments
    ---------
    state : np.ndarray, (2,)
    The (tilt, tip) state to integrate with.

    gain : float
    The integrator gain (scaling factor for the ideal DMC)

    leak : float
    The integrator leak ("forgetting factor" for the existing DMC)

    Returns
    -------
    command : np.ndarray, (ydim, xdim)
    The command to be put on the DM.
    """
    dmcn = zcoeffs_to_dmc(np.pad(state, (0,3)))
    return state, gain * dmcn + leak * optics.getdmc()
    
def lqg_controller(state, **kwargs):
    """
    Linear-quadratic-Gaussian control.
    """
    u = kwargs.get("klqg").control()
    # correct for KLQG built for only tip-tilt
    u = np.pad(u, (0,3))
    return u, optics.getdmc() + zcoeffs_to_dmc(u)

# Control laws: combination of an observer and controller
openloop = partial(control, observer=identity, controller=ol_controller)
integrate = partial(control, observer=identity, controller=integrator)
kalman_lqg = partial(control, observer=kfilter, controller=lqg_controller)
