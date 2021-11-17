"""
Core runner for SEAL experiments.
"""

import time
import logging
import warnings
from threading import Thread
from queue import Queue
from abc import ABC
import numpy as np
from copy import copy

from .exp_logger import experiment_logger
from .exp_result import ExperimentResult, result_from_log
from .schedules import make_noise, make_ustep, make_train, make_sine, make_atmvib
from ..constants import dt
from ..utils import get_timestamp, zeno
from ..optics import optics
from ..optics import measure_zcoeffs, make_im_cm
from ..optics import align

class Experiment(ABC):
	"""
	The basic specification for an experiment plan.

	Experiments know:
	- their duration
	- kwargs like verbose and half-closing
	- their logger
	- (ideally later, their optics system and their dt, but I'm okay leaving those global for now)

	and also dynamically set their timestep.

	They accept the controller as an argument to `run`.
	"""
	def set_default_params(self):
		self.params = {"verbose": True, "rcond" : 1e-4, "hc": False}
		# really Optics should be responsible for rcond

	def record(self, q):
		"""
		Get images from the optics system, put them in the output queue,
		and record the times at which images are received.
		"""
		t_start = time.time()
		num_exposures = 0

		while time.time() < t_start + self.dur:
			imval = optics.getim()
			t = time.time()
			q.put((num_exposures, t, imval))
			self.logger.info(f"Exposure    {num_exposures}: {[t]}")
			num_exposures += 1
			zeno(dt - (time.time() - t))

		out_q.put((0, 0, None))
	# this is a placeholder to tell the queue that there's no more images coming

	def compute(in_q, out_q):
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
			if (not half_close) or (t >= t1 + t / 2):
				#if t - t_exp < (frame_delay * dt):
				#	zeno((frame_delay * dt) - (time.time() - t_exp))
				optics.applydmc(dmc)
				self.logger.info(f"DMC         {i}: {u}")
			t = time.time()
	
	def check_alignment(self):
		rcond = self.params.get("rcond")
		verbose = self.params.get("verbose")
		_, cmd_mtx = make_im_cm(rcond=rcond, verbose=verbose)
		bestflat = np.load(optics.bestflat_path)
		imflat = np.load(optics.imflat_path)
		baseline_zvals = measure_zcoeffs(optics.getim() - imflat, cmd_mtx=cmd_mtx)

		i = 0
		while np.any(np.abs(baseline_zvals) > 1e-3):
			self.logger.info(f"The system may not be aligned: baseline Zernikes is {baseline_zvals.flatten()}.")
			optics.align(manual=False, view=False)
			_, cmd_mtx = make_im_cm(rcond=rcond)
			bestflat, imflat = optics.refresh(verbose)
			baseline_zvals = measure_zcoeffs(optics.getim() - imflat, cmd_mtx=cmd_mtx)
			i += 1
			if i > 10: # arbitrary
				err_message = "Cannot align system: realign manually and try experiment again."
				self.logger.error(err_message)
				raise RuntimeError(err_message)
				
		logger.info("System aligned and command matrix updated.")
		return bestflat, imflat, cmd_mtx

	@property
	def record_path(self):
		p = copy(self.record_root)
		v = True
		hc = False
		for k in self.params:
			if k == "verbose":
				v = v and kwargs.get(k)
			else:
				p += f"_{k}_{kwargs.get(k)}"
				if k == "hc":
					hc = kwargs.get(k)
					self.logging.info("Closing the loop halfway into the experiment.")

		p += f"_tstamp_{self.timestamp}.csv"
		return p

	def run(self, controller):
		self.timestamp = get_timestamp()
		self.logger, self.log_path = experiment_logger(self.timestamp)
		bestflat, self.imflat, self.cmd_mtx = self.check_alignment()

		q_compute = Queue()
		q_control = Queue()
		record_thread = Thread(target=self.record, args=(q_compute,))
		compute_thread = Thread(target=self.compute, args=(q_compute, q_control,))
		control_thread = Thread(target=self.control, args=(q_control, controller,))
		disturb_thread = Thread(target=self.disturbance)

		self.logger.info("Starting recording and commands.")

		record_thread.start()
		compute_thread.start()
		control_thread.start()
		command_thread.start()

		q_compute.join()
		q_control.join()
		record_thread.join(self.dur)
		compute_thread.join(self.dur)
		control_thread.join(self.dur)
		command_thread.join(self.dur)

		logger.info("Done with experiment.")
		optics.applydmc(bestflat)

		result = result_from_log(log_path, self.params)

		if optics.name != "Sim":
			result.to_csv(self.record_path)

		return result

"""
# uconvert_ratio removed 2021-11-17 at commit just after 6a207e3

# "record" functions: a bunch of combinations of a control_schedule and dist_schedule
# with specific handling for parameters like amplitudes

# TODO make sure that the kwargs are getting passed into the controller!
record_openloop = partial(run_experiment, path.join("openloop", "ol"), openloop)

record_oltrain = partial(record_openloop, make_train)
record_olnone = partial(record_openloop, make_noise)
record_olustep = partial(record_openloop, make_ustep)
record_olsin = partial(record_openloop, make_sine)
record_olatmvib = partial(record_openloop, make_atmvib)
# deleted record OL usteps in circle as it didn't seem too useful
# can add it back in to schedules.py later if desired

record_integrator = partial(run_experiment, path.join("integrator", "int"), integrate)

record_inttrain = partial(record_integrator, make_train)
record_intnone = partial(record_integrator, make_noise)
record_intustep = partial(record_integrator, make_ustep)
record_intsin = partial(record_integrator, make_sine)
record_intatmvib = partial(record_integrator, make_atmvib)

make_record_lqg = lambda lqg: partial(run_experiment, path.join("lqg", "lqg"), lqg)"""
