"""
Core runner for SEAL experiments.
"""

import logging
import sys

from copy import copy
from functools import partial
from math import ceil
from multiprocessing import Process
from time import monotonic_ns as mns

import numpy as np
from tqdm import trange

from .utils import LogRecord_ns, Formatter_ns
from .exp_result import ExperimentResult, result_from_log
from .schedules import make_air, make_ustep, make_train, make_sine, make_atmvib

from ..utils import dt
from ..utils import get_timestamp, scheduled_loop, joindata
from ..optics import align # TODO remove cross submodule dependency

class Experiment:
	"""
	The basic specification for an experiment plan.

	Experiments know:
	- their duration
	- kwargs like half-closing
	- their logger
	- their optics system

	and also dynamically set their timestep.

	They accept the controller as an argument to `run`.
	"""
	def __init__(self, dist_maker, dur, optics, dt=dt, half_close=False, **kwargs):
		self.dur = dur
		self.optics = optics
		self.dt = dt
		self.half_close = half_close
		self.timestamp = None
		self.logger = None # if you ever run into this, you're trying to analyze a log of a run that hasn't happened yet
		self.params = dict(kwargs)
		self.disturbance = dist_maker(dur, **kwargs)
		self.iters = 0
		self.dist_iters = 0

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

	def disturb_iter(self):
		self.optics.applytilt(self.disturbance[self.dist_iters, 0])
		self.optics.applytip(self.disturbance[self.dist_iters, 1])
		self.logger.info(f"Disturbance {self.dist_iters}: {self.disturbance[self.dist_iters, :]}")
		self.dist_iters += 1
		
	def loop_iter(self, controller):
		dmc = self.optics.zcoeffs_to_dmc(self.u) + self.optics.bestflat # + controller.leak * self.optics.getdmc()
		self.optics.applydmc(dmc)
		self.logger.info(f"DMC         {self.iters}: {self.u}")
		imval = self.optics.getim(check=False)
		self.iters += 1
		self.logger.info(f"Exposure    {self.iters}: {[mns()]}")
		measurement = self.optics.measure(imval)
		self.logger.info(f"Measurement {self.iters}: {measurement}")
		self.u = controller(measurement)

	def check_alignment(self):
		baseline_zvals = self.optics.measure()

		i = 0
		while np.any(np.abs(baseline_zvals) > 1e-3):
			self.logger.info(f"The system may not be aligned: baseline Zernikes is {baseline_zvals.flatten()}.")
			align(self.optics, manual=False, view=False)
			self.optics.make_im_cm()
			baseline_zvals = self.optics.measure()
			i += 1
			if i > 10: # arbitrary
				err_message = "Cannot align system: realign manually and try experiment again."
				self.logger.error(err_message)
				raise RuntimeError(err_message)
				
		self.logger.info("System aligned and command matrix updated.")

	def record_path(self, p):
		for k in self.params:
			p += f"_{k}_{round(self.params[k], 4)}"

		p += f"_tstamp_{self.timestamp}.csv"
		return p

	def simulate(self, controller, measure_std=0.001):
		states = np.zeros((len(self.disturbance), 2))
		states[0] = self.disturbance[0]
		for i in trange(1, len(self.disturbance)):
			measurement = states[i-1] + np.random.normal(0, measure_std, (2,))
			states[i] = states[i-1] + self.disturbance[i] + controller(measurement)[0]

		return states

	def run(self, controller):
		self.u = np.array([0.0, 0.0])
		print("Starting experiment.")
		self.timestamp = get_timestamp()
		self.update_logger()
		self.check_alignment()
		if self.half_close:
			self.logger.info("Closing the loop halfway into the experiment.")

		self.logger.info("Starting recording and commands.")
		t_start = mns()

		processes = [
			Process(target=scheduled_loop, args=(partial(self.loop_iter, controller), self.dt, self.dur, t_start, False), name="loop"),
			Process(target=scheduled_loop, args=(self.disturb_iter, self.dt, self.dur, t_start, True), name="disturbances")
		]

		for p in processes:
			p.start()

		for p in processes:
			p.join(self.dur * 1.1)

		for p in processes:
			if p.is_alive():
				self.logger.info(f"Terminating process {p.name}.")
				p.terminate()	

		self.logger.info("Done with experiment.")
		self.optics.applybestflat()
		self.iters = 0
		self.dist_iters = 0
		print(f"Experiment finished, log written to {self.log_path}")

		result = result_from_log(self.timestamp, self.log_path)

		if self.optics.name != "Sim":
			result.to_csv(self.record_path(controller.root_path), self.params)

		return result

