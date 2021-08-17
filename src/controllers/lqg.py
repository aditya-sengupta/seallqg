# authored by Aditya Sengupta

import numpy as np
from copy import copy
from scipy import linalg

from .dare import solve_dare

class LQGController:
    """
    Sufficient information to build an LQG controller:

    x - the state
    A - the time-evolution matrix
    B - the input-to-state matrix
    Q - the state penalty (cost x'Qx)
    R - the input penalty (cost u'Ru)

    This could be integrated with the KFilter 
    into one big "LQG state space description" class
    but for now it's separate to test observer-controller separation
    """

def compute_lqg_gain(A, B, Q, R, verbose=True):
    P = solve_dare(A, B, Q, R, verbose)
    return -np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
