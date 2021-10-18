# authored by Aditya Sengupta

import numpy as np
from functools import partial

from .observer import identity, make_kf_observer
from ..optics import optics, zcoeffs_to_dmc

def control(measurement, observer, controller, **kwargs):
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
    state = observer(measurement, **kwargs)
    u = controller(state, **kwargs)
    return u

# Controllers

def ol_controller(state, **kwargs):
    """
    Do nothing.
    """
    return np.array([0, 0]).astype(np.float32), optics.getdmc()

def integrator(state, gain=0.1, leak=1.0, **kwargs):
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
    dmcn = zcoeffs_to_dmc(state)
    return state, gain * dmcn + leak * optics.getdmc()
    
def make_lqg_controller(klqg):
    def lqg_controller(state, **kwargs):
        """
        Linear-quadratic-Gaussian control.
        """
        u = klqg.control()
        # correct for KLQG built for only tip-tilt
        u = np.pad(u, (0,3))
        return u, optics.getdmc() + zcoeffs_to_dmc(u)

    return lqg_controller

# Control laws: combination of an observer and controller
openloop = partial(control, observer=identity, controller=ol_controller)
integrate = partial(control, observer=identity, controller=integrator)

def make_kalman_controllers(klqg):
    kfilter = make_kf_observer(klqg)
    lqg = make_lqg_controller(klqg)
    kalman_integrate = partial(control, observer=kfilter, controller=integrator)
    kalman_lqg = partial(control, observer=kfilter, controller=lqg)
    return kalman_integrate, kalman_lqg
