"""
Core runner for the experiments in experiment.py.
Written by Aditya Sengupta
"""

import time
import os
import logging
import sys
import warnings
from threading import Thread
from queue import Queue
from functools import partial
import numpy as np

from .exp_logger import experiment_logger
from .exp_result import ExperimentResult, result_from_log
from ..constants import dt
from ..utils import get_timestamp, zeno
from ..optics import optics
from ..optics import measure_zcoeffs, make_im_cm
from ..optics import align

def record_im(out_q, duration, timestamp, logger):
	"""
	Get images from the optics system, put them in the output queue,
	and record the times at which images are received.
	"""
	t_start = time.time()
	num_exposures = 0

	while time.time() < t_start + duration:
		imval = optics.getim()
		t = time.time()
		out_q.put((num_exposures, t, imval))
		logger.info(f"Exposure    {num_exposures}: {[t]}")
		num_exposures += 1
		zeno(dt - (time.time() - t))

	out_q.put((0, 0, None))
	# this is a placeholder to tell the queue that there's no more images coming
	
def zcoeffs_from_queued_image(in_q, out_q, imflat, cmd_mtx, timestamp, logger):
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
				imdiff = img - imflat
				zval = measure_zcoeffs(imdiff, cmd_mtx).flatten()
				logger.info(f"Measurement {i}: {zval}")
				out_q.put((i, t, zval))

def control_schedule_from_law(q, control, timestamp, logger, duration=1, half_close=False):
	"""
	The SEAL schedule for a controller.

	Arguments
	---------
	q : Queue
	The queue to poll for new Zernike coefficient values.

	control : callable
	The function to execute control.
	"""
	u = None # the most recently applied control command
	t1 = time.time()
	t = t1
	# frame_delay = 2

	while t < t1 + duration:
		i, t_exp, z = q.get()
		q.task_done()
		u, dmc = control(z, logger=logger, u=u)
		if (not half_close) or (t >= t1 + t / 2):
			#if t - t_exp < (frame_delay * dt):
			#	zeno((frame_delay * dt) - (time.time() - t_exp))
			optics.applydmc(dmc)
			logger.info(f"DMC         {i}: {u}")
		t = time.time()

def check_alignment(logger, rcond, verbose, imax=10):
	_, cmd_mtx = make_im_cm(rcond=rcond, verbose=verbose)
	bestflat = np.load(optics.bestflat_path)
	imflat = np.load(optics.imflat_path)
	baseline_zvals = measure_zcoeffs(optics.getim() - imflat, cmd_mtx=cmd_mtx)

	i = 0
	while np.any(np.abs(baseline_zvals) > 1e-3):
		logger.info(f"The system may not be aligned: baseline Zernikes is {baseline_zvals.flatten()}.")
		align(manual=False, view=False)
		_, cmd_mtx = make_im_cm(rcond=rcond)
		bestflat, imflat = optics.refresh(verbose)
		baseline_zvals = measure_zcoeffs(optics.getim() - imflat, cmd_mtx=cmd_mtx)
		i += 1
		if i > imax:
			err_message = "Cannot align system: realign manually and try experiment again."
			logger.error(err_message)
			raise RuntimeError(err_message)
			
	logger.info("System aligned and command matrix updated.")
	return bestflat, imflat, cmd_mtx

def run_experiment(record_path, control_schedule, dist_schedule, duration, rcond=1e-4, verbose=True):
	timestamp = get_timestamp()
	logger, log_path = experiment_logger(timestamp)
	record_path += f"_tstamp_{timestamp}.csv"

	bestflat, imflat, cmd_mtx = check_alignment(logger, rcond, verbose)
	
	q_compute = Queue()
	q_control = Queue()
	record = partial(record_im, duration=duration, timestamp=timestamp, logger=logger)
	compute = partial(zcoeffs_from_queued_image, imflat=imflat, cmd_mtx=cmd_mtx, timestamp=timestamp, logger=logger)
	control = partial(control_schedule, duration=duration, timestamp=timestamp, logger=logger)
	command = partial(dist_schedule, logger=logger)
	
	record_thread = Thread(target=record, args=(q_compute,))
	compute_thread = Thread(target=compute, args=(q_compute, q_control,))
	control_thread = Thread(target=control, args=(q_control,))
	command_thread = Thread(target=command)

	logger.info("Starting recording and commands...")

	record_thread.start()
	compute_thread.start()
	control_thread.start()
	command_thread.start()

	q_compute.join()
	q_control.join()
	record_thread.join(duration)
	compute_thread.join(duration)
	control_thread.join(duration)
	command_thread.join(duration)

	logger.info("Done with experiment.")
	optics.applydmc(bestflat)

	result = result_from_log(log_path)

	if True or optics.name != "Sim":
		result.to_csv(record_path)
	warnings.warn("save on for sim")

	return result
