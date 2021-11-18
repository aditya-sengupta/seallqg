"""
Core runner for SEAL experiments.
"""

import sys
import logging
import warnings

from threading import Thread
from queue import Queue
from abc import ABC
import numpy as np
from copy import copy

from time import monotonic_ns as mns

from .utils import LogRecord_ns, Formatter_ns
from .exp_result import ExperimentResult, result_from_log
from .schedules import make_noise, make_ustep, make_train, make_sine, make_atmvib

from ..controllers import make_openloop, make_integrator

from ..constants import dt
from ..utils import get_timestamp, spin, spinlock, joindata
from ..optics import optics
from ..optics import measure_zcoeffs, make_im_cm
from ..optics import align
from ..optics import zcoeffs_to_dmc, applytip, applytilt

class Experiment:
	"""
	The basic specification for an experiment plan.

	Experiments know:
	- their duration
	- kwargs like verbose and half-closing
	- their logger
	- their optics system

	and also dynamically set their timestep.

	They accept the controller as an argument to `run`.
	"""
	def __init__(self, dist_maker, dur, optics=optics, dt=dt, half_close=False, rcond=1e-4, verbose=True, **kwargs):
		self.dur = dur
		self.optics = optics
		self.dt = dt
		self.half_close = half_close
		self.timestamp = None
		self.rcond = rcond
		self.verbose = verbose
		self.logger = None # if you ever run into this, you're trying to analyze a log of a run that hasn't happened yet
		self.params = dict(kwargs)
		nsteps = int(np.ceil(dur / dt))
		self.disturbance = dist_maker(dur, **kwargs)
		self.iters = 0

	def update_logger(self):
		logging.setLogRecordFactory(LogRecord_ns)
		self.logger = logging.getLogger()
		self.logger.handlers.clear()
		self.logger.setLevel(logging.INFO)
		formatter = Formatter_ns('%(asctime)s | %(levelname)s | %(message)s')
		stdout_handler = logging.StreamHandler(sys.stdout)
		stdout_handler.setLevel(logging.ERROR)
		stdout_handler.setFormatter(formatter)
		self.log_path = joindata("log", f"log_{self.timestamp}.log")
		file_handler = logging.FileHandler(self.log_path)
		file_handler.setLevel(logging.DEBUG)
		file_handler.setFormatter(formatter)
		self.logger.addHandler(file_handler)
		self.logger.addHandler(stdout_handler)

	def iterate(self, controller):
		applytilt(optics, self.disturbance[self.iters, 0])
		applytip(optics, self.disturbance[self.iters, 1])
		self.logger.info(f"Disturbance {self.iters}: {self.disturbance[self.iters, :]}")
		imval = self.optics.getim()
		self.iters += 1
		self.logger.info(f"Exposure    {self.iters}: {[mns()]}")
		imdiff = imval - self.imflat
		z = measure_zcoeffs(imdiff, self.cmd_mtx).flatten()
		self.logger.info(f"Measurement {self.iters}: {z}")
		u, dmc = controller(z)
		self.optics.applydmc(dmc)
		self.logger.info(f"DMC         {self.iters}: {u}")

	def check_alignment(self):
		_, self.cmd_mtx = make_im_cm(rcond=self.rcond, verbose=self.verbose)
		bestflat = np.load(self.optics.bestflat_path)
		self.imflat = np.load(self.optics.imflat_path)
		baseline_zvals = measure_zcoeffs(self.optics.getim() - self.imflat, cmd_mtx=self.cmd_mtx)

		i = 0
		while np.any(np.abs(baseline_zvals) > 1e-3):
			self.logger.info(f"The system may not be aligned: baseline Zernikes is {baseline_zvals.flatten()}.")
			align(self.optics, manual=False, view=False)
			_, self.cmd_mtx = make_im_cm(rcond=self.rcond)
			bestflat, self.imflat = self.optics.refresh(self.verbose)
			baseline_zvals = measure_zcoeffs(self.optics.getim() - self.imflat, cmd_mtx=self.cmd_mtx)
			i += 1
			if i > 10: # arbitrary
				err_message = "Cannot align system: realign manually and try experiment again."
				self.logger.error(err_message)
				raise RuntimeError(err_message)
				
		self.logger.info("System aligned and command matrix updated.")
		return bestflat

	def record_path(self, p):
		for k in self.params:
			p += f"_{k}_{round(self.params[k], 4)}"

		p += f"_tstamp_{self.timestamp}.csv"
		return p

	def run(self, con):
		if self.verbose:
			print("Starting experiment.")
		root_path, controller = con
		self.timestamp = get_timestamp()
		self.update_logger()
		bestflat = self.check_alignment()
		if self.half_close:
			self.logger.info("Closing the loop halfway into the experiment.")

		self.logger.info("Starting recording and commands.")

		spin(lambda: self.iterate(controller), self.dt, self.dur)

		self.logger.info("Done with experiment.")
		self.optics.applydmc(bestflat)
		self.iters = 0
		print(f"Experiment finished, log written to {self.log_path}")

		result = result_from_log(self.timestamp, self.log_path)

		if True or self.optics.name != "Sim":
			result.to_csv(self.record_path(root_path), self.params)

		return result

# Some predefined experiments
short_wait = Experiment(make_noise, 1)
med_wait = Experiment(make_noise, 10)
long_wait = Experiment(make_noise, 100)

ustep_tilt = Experiment(make_ustep, 1, tilt_amp=0.005, tip_amp=0.0)
ustep_tip = Experiment(make_ustep, 1, tilt_amp=0.0, tip_amp=0.005)

sine_one = Experiment(make_sine, 10, amp=0.003, ang=np.pi/4, f=1)
sine_five = Experiment(make_sine, 10, amp=0.003, ang=np.pi/4, f=5)

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

# uconvert_ratio removed 2021-11-17 at commit just after 6a207e3
