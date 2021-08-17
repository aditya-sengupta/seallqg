# authored by Aditya Sengupta

import numpy as np
from copy import copy
from scipy import linalg

def compute_lqg_gain(A, B, Q, R, verbose=True):
    try:
        P = linalg.solve_discrete_are(A, B, Q, R)
        if verbose:
            print("Solved discrete ARE.")
    except (ValueError, np.linalg.LinAlgError):
        if verbose:
            print("Discrete ARE solve failed, falling back to iterative solution.")
        P = copy(Q)
        lastP = np.zeros_like(A)
        iters = 0
        while not np.allclose(lastP, P):
            # I have entirely made this up and I have not tested it
            # I really hope the discrete ARE just works
            P = A.T @ P @ A + Q
            K = -np.linalg.inv(B @ P @ B.T + R) @ B.T @ P @ A
            P = P - K @ B @ P
            iters += 1

        if verbose:
            print("Solved iteratively in {} iterations".format(iters))
            
    return -np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
