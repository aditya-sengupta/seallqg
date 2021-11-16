"""
Utilities for the experiments in experiment.py.
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

from ..constants import dt
from ..utils import joindata, get_timestamp
from ..optics import optics
from ..optics import measure_zcoeffs, make_im_cm
from ..optics import align

def record_im(out_q, duration, timestamp, logger):
	"""
	Get images from the optics system, put them in the output queue,
	and record the times at which images are received.
	"""
	t_start = time.time()
	times = []

	num_exposures = 0
	while time.time() < t_start + duration:
		t0 = time.time()
		imval = optics.getim()
		t = time.time()
		out_q.put((num_exposures, t, imval))
		logger.info(f"Exposure    {num_exposures}")
		times.append(t)
		num_exposures += 1
		#time.sleep(max(0, dt - (time.time() - t0)))

	out_q.put((0, 0, None))
	# this is a placeholder to tell the queue that there's no more images coming
	
	times = np.array(times) - t_start
	fname = joindata("recordings", f"rectime_stamp_{timestamp}.npy")
	np.save(fname, times)
	return times
	
def zcoeffs_from_queued_image(in_q, out_q, imflat, cmd_mtx, timestamp, logger):
	"""
	Takes in images from the queue `in_q`,
	and converts them to Zernike coefficient values 
	which get sent to the queue `out_q`.
	"""
	fname = joindata("recordings", f"recz_stamp_{timestamp}.npy")
	zvals = []
	img = 0 # non-None start value
	while img is not None:
		if not in_q.empty():
			i, t, img = in_q.get()
			in_q.task_done()
			if img is not None:
				imdiff = img - imflat
				zval = measure_zcoeffs(imdiff, cmd_mtx).flatten()
				logger.info(f"Measurement {i}")
				out_q.put((i, t, zval))
				zvals.append(zval)
			else:
				time.sleep(dt * 0)
	zvals = np.array(zvals)
	np.save(fname, zvals)
	return zvals

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
	fname = joindata("recordings", f"recc_stamp_{timestamp}.npy")
	last_z = None # the most recently applied control command
	t1 = time.time()
	t = t1
	cvals = []

	while t < t1 + duration:
		i, t_exp, z = q.get()
		q.task_done()
		last_z, dmc = control(z, logger=logger, u=last_z)
		t = time.time()
		if (not half_close) or (t >= t1 + t / 2):
			optics.applydmc(dmc)
			logger.info(f"DMC         {i}")
		else:
			i, z = q.get()
			q.task_done()
			last_z, dmc = control(z, logger=logger, u=last_z)
			t = time.time()
			if (not half_close) or (t >= t1 + t / 2):
				optics.applydmc(dmc)
				logger.info(f"DMC         {i}")
			else:
				last_z *= 0
			cvals.append(last_z)

	np.save(fname, np.array(cvals))

def record_experiment(record_path, control_schedule, dist_schedule, t=1, rcond=1e-4, half_close=False, verbose=True):
	_, cmd_mtx = make_im_cm(rcond=rcond, verbose=verbose)
	#bestflat, imflat = optics.refresh(verbose=False)
	bestflat = np.load(optics.bestflat_path)
	imflat = np.load(optics.imflat_path)
	baseline_zvals = measure_zcoeffs(optics.getim() - imflat, cmd_mtx=cmd_mtx)

	i = 0
	imax = 10
	while np.any(np.abs(baseline_zvals) > 1e-3):
		warnings.warn(f"The system may not be aligned: baseline Zernikes is {baseline_zvals.flatten()}.")
		align(manual=False, view=False)
		_, cmd_mtx = make_im_cm(rcond=rcond)
		bestflat, imflat = optics.refresh(verbose)
		baseline_zvals = 0 * measure_zcoeffs(optics.getim() - imflat, cmd_mtx=cmd_mtx)
		i += 1
		if i > imax:
			print("Cannot align system: realign manually and try experiment again.")
			return [], []
	
	timestamp = get_timestamp()
	record_path = joindata(record_path) + f"_time_stamp_{timestamp}.npy"

	# TODO split the logging handling into a separate function or file
	# from https://stackoverflow.com/questions/31328300/python-logging-module-logging-timestamp-to-include-microsecond
	class LogRecord_ns(logging.LogRecord):
		def __init__(self, *args, **kwargs):
			self.created_ns = time.time_ns() # Fetch precise timestamp
			super().__init__(*args, **kwargs)

	class Formatter_ns(logging.Formatter):
		default_nsec_format = '%s,%09d'
		def formatTime(self, record, datefmt=None):
			if datefmt is not None: # Do not handle custom formats here ...
				return super().formatTime(record, datefmt) # ... leave to original implementation
			ct = self.converter(record.created_ns / 1e9)
			t = time.strftime(self.default_time_format, ct)
			s = self.default_nsec_format % (t, record.created_ns - (record.created_ns // 10**9) * 10**9)
			return s

	logging.setLogRecordFactory(LogRecord_ns)

	logger = logging.getLogger()
	logger.handlers.clear()
	logger.setLevel(logging.INFO)
	formatter = Formatter_ns('%(asctime)s | %(levelname)s | %(message)s')
	stdout_handler = logging.StreamHandler(sys.stdout)
	stdout_handler.setLevel(logging.DEBUG)
	stdout_handler.setFormatter(formatter)

	file_handler = logging.FileHandler(joindata("log", f"log_{timestamp}.log"))
	file_handler.setLevel(logging.DEBUG)
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	logger.addHandler(stdout_handler)
	
	q_compute = Queue()
	q_control = Queue()
	record_thread = Thread(target=partial(record_im, duration=t, timestamp=timestamp, logger=logger), args=(q_compute,))
	compute_thread = Thread(target=partial(zcoeffs_from_queued_image, timestamp=timestamp, logger=logger), args=(q_compute, q_control, imflat, cmd_mtx,))
	control_thread = Thread(target=partial(control_schedule, duration=t, timestamp=timestamp, logger=logger), args=(q_control,))
	command_thread = Thread(target=dist_schedule)

	if verbose:
		logger.info("Starting recording and commands...")

	record_thread.start()
	compute_thread.start()
	control_thread.start()
	command_thread.start()

	q_compute.join()
	q_control.join()
	record_thread.join(t*1.1)
	compute_thread.join(t*1.1)
	control_thread.join(t*1.1)
	command_thread.join(t*1.1)

	if control_thread.is_alive():
		warnings.warn("Control not terminating, waiting an additional 10 seconds...")
		time.sleep(10)

	if verbose:
		logger.info("Done with experiment.")

	optics.applydmc(bestflat)

	timepath = joindata("recordings", f"rectime_stamp_{timestamp}.npy")
	zpath = joindata("recordings", f"recz_stamp_{timestamp}.npy")
	cpath = joindata("recordings", f"recc_stamp_{timestamp}.npy")
	times = np.load(timepath)
	zvals = np.load(zpath)
	try:
		cvals = np.load(cpath)
	except FileNotFoundError:
		logger.error("Control loop schedule too far out of sync and did not complete.")
		raise
	if optics.name != "Sim":
		np.save(record_path, times)
		if verbose:
			logger.info(f"Times    saved to {record_path}")
		record_path = record_path.replace("time", "cmd")
		np.save(record_path, cvals)
		if verbose:
			logger.info(f"Commands saved to {record_path}")
		record_path = record_path.replace("cmd", "z")
		np.save(record_path, zvals)
		if verbose:
			logger.info(f"Coeffs   saved to {record_path}")
	os.remove(timepath)
	os.remove(zpath)
	os.remove(cpath)
	return times, zvals
