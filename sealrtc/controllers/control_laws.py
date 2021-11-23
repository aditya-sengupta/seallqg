"""
Control laws. 
These are all functions with a state as their first argument, 
and further arguments that are filled in by the Controller constructor.

All of these functions return (tip, tilt), leak:
(tip, tilt) gets converted by the optics to an ideal DM command, 
and leak is the scalar multiple of the existing command we want to put on.
"""
import numpy as np

def nothing(state):
    """
    Do nothing.
    """
    return np.array([0, 0]), 1

def integrate(state, integ):
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
    return integ.control(state), integ.leak
    
def lqr(state, klqg):
    """
    Linear-quadratic-Gaussian control.
    """
    u = klqg.control()
    return u, 1