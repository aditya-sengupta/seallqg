import numpy as np
from functools import partial

from ..utils import joindata
from .integrator import Integrator
from .observe_laws import identity, kfilter
from .control_laws import nothing, integrate, lqr

def controller(measurement, observe_law, control_law):
    state = observe_law(measurement[:2]) # TODO generalize
    return control_law(state)

def make_openloop():
    return joindata("openloop", "ol"), partial(
        controller, 
        observe_law=identity, 
        control_law=nothing
    )

def make_integrator(integ=Integrator()):
    return joindata("integrator", f"int_gain_{integ.gain}_leak_{integ.leak}"), partial(
        controller, 
        observe_law=identity, 
        control_law=partial(integrate, integ=integ)
    )

def make_lqg(klqg):
    return joindata("lqg", f"klqg_nstate_{klqg.state_size}"), partial(
        controller, 
        observe_law=partial(kfilter, klqg=klqg), 
        control_law=partial(lqr, klqg=klqg)
    )
