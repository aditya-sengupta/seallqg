"""
Core runner for SEAL experiments.
"""

import sys
import time
import logging
import warnings
from threading import Thread
from queue import Queue
from abc import ABC
import numpy as np
from copy import copy

from .utils import LogRecord_ns, Formatter_ns
from .exp_result import ExperimentResult, result_from_log
from .schedules import make_noise, make_ustep, make_train, make_sine, make_atmvib

from ..controllers import make_openloop, make_integrator

from ..constants import dt
from ..utils import get_timestamp, zeno, joindata
from ..optics import optics
from ..optics import measure_zcoeffs, make_im_cm
from ..optics import align

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
		self.disturbance = dist_maker(dur, **kwargs)

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
		
	def record(self, q):
		"""
		Get images from the optics system, put them in the output queue,
		and record the times at which images are received.
		"""
		t_start = time.time()
		num_exposures = 0
		self.logger.info("Recording initialized.")

		while time.time() < t_start + self.dur:
			imval = self.optics.getim()
			t = time.time()
			q.put((num_exposures, t, imval))
			self.logger.info(f"Exposure    {num_exposures}: {[t]}")
			num_exposures += 1
			zeno(dt - (time.time() - t))

		q.put((0, 0, None))
		# this is a placeholder to tell the queue that there's no more images coming

	def compute(self, in_q, out_q):
		"""
		Takes in images from the queue `in_q`,
		and converts them to Zernike coefficient values 
		which get sent to the queue `out_q`.
		"""
		img = 0 # non-None start value
		while img is not None:
			if not in_q.empty():
				i, t, img = in_q.get()
				in_q.task_done()
				if img is not None:
					imdiff = img - self.imflat
					zval = measure_zcoeffs(imdiff, self.cmd_mtx).flatten()
					self.logger.info(f"Measurement {i}: {zval}")
					out_q.put((i, t, zval))

	def control(self, q, controller):
		u = None # the most recently applied control command
		t1 = time.time()
		t = t1
		# frame_delay = 2

		while t < t1 + self.dur:
			i, t_exp, z = q.get()
			q.task_done()
			u, dmc = controller(z)
			if (not self.half_close) or (t >= t1 + t / 2):
				#if t - t_exp < (frame_delay * dt):
				#	zeno((frame_delay * dt) - (time.time() - t_exp))
				self.optics.applydmc(dmc)
				self.logger.info(f"DMC         {i}: {u}")
			t = time.time()
	
	def check_alignment(self):
		_, self.cmd_mtx = make_im_cm(rcond=self.rcond, verbose=self.verbose)
		bestflat = np.load(self.optics.bestflat_path)
		self.imflat = np.load(self.optics.imflat_path)
		baseline_zvals = measure_zcoeffs(self.optics.getim() - self.imflat, cmd_mtx=self.cmd_mtx)

		i = 0
		while np.any(np.abs(baseline_zvals) > 1e-3):
			self.logger.info(f"The system may not be aligned: baseline Zernikes is {baseline_zvals.flatten()}.")
			self.optics.align(manual=False, view=False)
			_, self.cmd_mtx = make_im_cm(rcond=rcond)
			bestflat, self.imflat = self.optics.refresh(verbose)
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

		q_compute = Queue()
		q_control = Queue()
		record_thread = Thread(target=self.record, args=(q_compute,))
		compute_thread = Thread(target=self.compute, args=(q_compute, q_control,))
		control_thread = Thread(target=self.control, args=(q_control, controller,))
		disturb_thread = Thread(target=self.disturbance, args=(self.logger,))

		self.logger.info("Starting recording and commands.")

		record_thread.start()
		compute_thread.start()
		control_thread.start()
		disturb_thread.start()

		q_compute.join()
		q_control.join()
		record_thread.join(self.dur)
		compute_thread.join(self.dur)
		control_thread.join(self.dur)
		disturb_thread.join(self.dur)

		self.logger.info("Done with experiment.")
		self.optics.applydmc(bestflat)
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

sine_one = Experiment(make_sine, 10, amp=0.005, ang=np.pi/4, f=1)
sine_five = Experiment(make_sine, 10, amp=0.005, ang=np.pi/4, f=5)

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
