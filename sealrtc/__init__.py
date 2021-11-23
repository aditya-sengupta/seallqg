from .controllers import *
from .experiments import *
from .optics import *
from .utils import *

def make_experiments():
    optics = make_optics
    # Some predefined experiments
    short_wait = Experiment(make_air, 1, optics)
    med_wait = Experiment(make_air, 10, optics)
    long_wait = Experiment(make_air, 100, optics)

    ustep_tilt = Experiment(make_ustep, 1, optics, tilt_amp=0.005, tip_amp=0.0)
    ustep_tip = Experiment(make_ustep, 1, optics, tilt_amp=0.0, tip_amp=0.005)

    sine_one = Experiment(make_sine, 10, optics, amp=0.003, ang=np.pi/4, f=1)
    sine_five = Experiment(make_sine, 10, optics, amp=0.003, ang=np.pi/4, f=5)

    # and some controllers from the controller submodule
    ol = make_openloop()
    integ = make_integrator()

    # Some pairings of experiments with controllers (zero-argument runnable functions from the Python interpreter)
    olnone = lambda: med_wait.run(ol)
    olsin1 = lambda: sine_one.run(ol)
    olsin5 = lambda: sine_five.run(ol)
    intnone = lambda: med_wait.run(integ)
    intsin1 = lambda: sine_one.run(integ)
    intsin5 = lambda: sine_five.run(integ)

    globals().update(locals()) # not ideal, but I'll figure out a better way to do lazy execution if it matters

    # uconvert_ratio removed 2021-11-17 at commit just after 6a207e3

experiment_mode = False
if experiment_mode:
    make_experiments()