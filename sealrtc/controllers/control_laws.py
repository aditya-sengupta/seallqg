"""
Control laws. 
These are all functions with a state as their first argument, 
and further arguments that are filled in by the Controller constructor.
"""

import numpy as np
from ..optics import optics

def nothing(state):
    """
    Do nothing.
    """
    return np.array([0, 0]), optics.getdmc()

def integrate(state, gain, leak):
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
    dmcn = optics.zcoeffs_to_dmc(np.pad(state, (0,3)))
    return state, gain * dmcn + leak * optics.getdmc()
    
def lqr(state, klqg):
    """
    Linear-quadratic-Gaussian control.
    """
    u = klqg.control()
    # TODO generalize
    u = np.pad(u, (0,3))
    return u, optics.getdmc() + optics.zcoeffs_to_dmc(u)