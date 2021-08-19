# authored by Aditya Sengupta

import numpy as np
from functools import partial

from .observer import identity, make_kf_observer
from ..optics import optics, tt_to_dmc

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
    return controller(state, **kwargs)

# Controllers

def ol_controller(state, **kwargs):
    """
    Do nothing.
    """
    return optics.getdmc()

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
    dmcn = tt_to_dmc(state)
    return gain * dmcn + leak * optics.getdmc()

def lqg_controller(klqg, **kwargs):
    """
    Linear-quadratic-Gaussian control.
    """
    return optics.getdmc() + tt_to_dmc(klqg.control())

# Control laws: combination of an observer and controller
openloop = partial(control, observer=identity, controller=ol_controller)
integrate = partial(control, observer=identity, controller=integrator)

def make_kalman_controllers(klqg):
    kfilter = make_kf_observer(klqg)
    kalman_integrate = partial(control, observer=kfilter, controller=integrator)
    kalman_lqg = partial(control, observer=kfilter, controller=lqg_controller)
    return kalman_integrate, kalman_lqg
