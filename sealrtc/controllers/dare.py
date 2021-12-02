import numpy as np
import scipy.linalg as la
from slycot.synthesis import sg02ad
from slycot.exceptions import SlycotArithmeticError
from copy import copy

def dare_iterative_update(A, B, Q, R, P):
    return A.T @ P @ A - (A.T @ P @ B) @ np.linalg.pinv(B.T @ P @ B + R) @ (B.T @ P @ A) + Q

def check_dare(A, B, Q, R, P):
    return np.allclose(
        P,
        dare_iterative_update(A, B, Q, R, P)
    )

def solve_dare(A, B, Q, R, verbose=True, max_iters=1000):
    try:
        try:
            n = A.shape[0]
            m = B.shape[1]
            E = np.eye(n)
            L = np.zeros_like(B)
            P = sg02ad('D', 'B', 'N', 'U', 'Z', 'N', 'S', 'R', n, m, 1, A, E, B, Q, R, L)[1]
            if verbose:
                print("Solved DARE with slycot.")
        except (ValueError, SlycotArithmeticError) as e:
            if verbose:
                print("slycot error", e)
            P = la.solve_discrete_are(A, B, Q, R)
            if verbose:
                print("Solved DARE with scipy.")
    except (ValueError, np.linalg.LinAlgError) as e:
        if verbose:
            print("scipy error", e)
            print("Discrete ARE solve failed, falling back to iterative solution.")
        P, _ = solve_dare_iter(A, B, Q, R, verbose, max_iters)
    return P
    
def solve_dare_iter(A, B, Q, R, verbose=True, max_iters=1000):
    newP = copy(Q)
    P = np.zeros_like(A)
    iters = 0
    while (not np.allclose(P, newP)) and iters < max_iters:
        P = newP
        newP = dare_iterative_update(A, B, Q, R, P)
        iters += 1
    if verbose:
        if check_dare(A, B, Q, R, P):
            if verbose:
                print(f"Solved iteratively in {iters} iterations.")
        else:
            raise RuntimeError(f"Iterative solve failed in {iters} iterations.")
    return P, iters
    