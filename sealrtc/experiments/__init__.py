"""
Module containing all the possible experiments, and how to run them.
"""

from .exp_result import *
from .experiment import *
from .schedules import *

__all__ = [
    "Experiment",
    "ExperimentResult",
    "loadres",
    "make_air",
    "make_ustep",
    "make_train",
    "make_sine",
    "make_atmvib"
]