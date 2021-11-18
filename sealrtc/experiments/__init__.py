"""
Module containing all the possible experiments, and how to run them.
"""

from .exp_result import *
from .experiment import *

__all__ = [
    "Experiment",
    "ExperimentResult",
    "loadres",
    "short_wait",
    "med_wait",
    "long_wait",
    "sine_one",
    "sine_five",
    "ol",
    "integ",
    "olnone",
    "olsin1",
    "olsin5",
    "intnone",
    "intsin1",
    "intsin5"
]