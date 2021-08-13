import numpy as np
import scipy.linalg as la
from copy import copy

def dare_iterative_update(A, B, Q, R, P):
    return A.T @ P @ A - (A.T @ P @ B) @ np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A) + Q

def check_dare(A, B, Q, R, P):
    return np.allclose(
        P,
        dare_iterative_update(A, B, Q, R, P)
    )

def solve_dare(A, B, Q, R, verbose=True, max_iters=1000):
    try:
        P = la.solve_discrete_are(A, B, Q, R)
        if verbose:
            print("Solved discrete ARE.")
    except (ValueError, np.linalg.LinAlgError):
        if verbose:
            print("Discrete ARE solve failed, falling back to iterative solution.")
        newP = copy(Q)
        P = np.zeros_like(A)
        iters = 0
        while (not np.allclose(P, newP)) and iters < max_iters:
            P = newP
            newP = dare_iterative_update(A, B, Q, R, P)
            iters += 1
        if verbose:
            if check_dare(A, B, Q, R, P):
                print("Solved iteratively in {} iterations.".format(iters))
            else:
                print("Iterative solve failed in {} iterations.".format(iters))
    return P