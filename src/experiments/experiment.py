# authored by Aditya Sengupta
# my tip-tilt experiments

import numpy as np
from os import path
from functools import partial

from .schedules import make_noise, make_ustep, make_train, make_sine, make_atmvib
from .exp_runner import run_experiment, control_schedule_from_law
from ..optics import optics
from ..optics import applytip, applytilt, aperture
from ..controllers import openloop, integrate

def uconvert_ratio(amp=1.0):
	bestflat, _ = optics.refresh()
	expt_init = optics.get_expt()
	optics.set_expt(1e-5)
	uconvert_matrix = np.zeros((2,2))
	for (mode, dmcmd) in enumerate([applytip, applytilt]):
		optics.applydmc(bestflat)
		dmcmd(amp)
		dm2 = optics.getdmc()
		cm2x = []
		while len(cm2x) != 1:
			im2 = optics.stackim(100)
			cm2x, cm2y = np.where(im2 == np.max(im2))

		optics.applydmc(bestflat)
		dmcmd(-amp)
		dm1 = optics.getdmc()
		cm1x = []
		while len(cm1x) != 1:
			im1 = optics.stackim(100)
			cm1x, cm1y = np.where(im1 == np.max(im1))

		dmdiff = aperture * (dm2 - dm1)
		
		dmdrange = np.max(dmdiff) - np.min(dmdiff)
		uconvert_matrix[mode] = [dmdrange /  (cm2y - cm1y), dmdrange / (cm2x - cm1x)]

	optics.set_expt(expt_init)
	optics.applydmc(bestflat)
	return uconvert_matrix

# "record" functions: a bunch of combinations of a control_schedule and dist_schedule
# with specific handling for parameters like amplitudes

def record_openloop(schedule_maker, duration=10, **kwargs):
	record_path = path.join("openloop", "ol")
	v = True
	for k in kwargs:
		if k == "verbose":
			v = v and kwargs.get(k)
		else:
			record_path += f"_{k}_{kwargs.get(k)}"

	control_schedule= partial(control_schedule_from_law, control=openloop)
	dist_schedule = schedule_maker(duration=duration, **kwargs)
	return run_experiment(
		record_path,
		control_schedule,
		dist_schedule,
		duration,
		verbose=v
	)

record_oltrain = partial(record_openloop, make_train)
record_olnone = partial(record_openloop, make_noise)
record_olustep = partial(record_openloop, make_ustep)
record_olsin = partial(record_openloop, make_sine)
record_olatmvib = partial(record_openloop, make_atmvib)
# deleted record OL usteps in circle as it didn't seem too useful
# can add it back in to schedules.py later if desired

def record_integrator(schedule_maker, duration=1, gain=0.1, leak=1.0, **kwargs):
	"""
	Record experiments with an integrator.
	"""
	record_path = path.join("integrator", f"int_gain_{gain}_leak_{leak}")
	v = True
	hc = False
	for k in kwargs:
		if k == "verbose":
			v = v and kwargs.get(k)
		else:
			record_path += f"_{k}_{kwargs.get(k)}"
			if k == "hc":
				hc = kwargs.get(k)
				if hc:
					print("Closing the loop halfway into the experiment.")

	control_schedule = partial(control_schedule_from_law, control=partial(integrate, gain=gain, leak=leak), half_close=hc)
	dist_schedule = schedule_maker(duration, **kwargs)
	return run_experiment(
		record_path, 
		control_schedule,
		dist_schedule,
		duration,
		verbose=v
	)

record_inttrain = partial(record_integrator, make_train)
record_intnone = partial(record_integrator, make_noise)
record_intustep = partial(record_integrator, make_ustep)
record_intsin = partial(record_integrator, make_sine)
record_intatmvib = partial(record_integrator, make_atmvib)
