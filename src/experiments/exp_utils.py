"""
Utilities for the experiments in experiment.py.
Written by Aditya Sengupta
"""

import time
import os
import warnings
from threading import Thread
from queue import Queue
from functools import partial
import numpy as np

from ..constants import dt
from ..utils import joindata, get_timestamp
from ..optics import optics
from ..optics import measure_tt, make_im_cm
from ..optics import align_alpao_fast

def record_im(out_q, duration=1, timestamp=get_timestamp()):
	"""
	Get images from the optics system, put them in the output queue,
	and record the times at which images are received.
	"""
	t_start = time.time()
	times = []

	while time.time() < t_start + duration:
		imval = optics.getim()
		times.append(time.time())
		out_q.put(imval)

	out_q.put(None)
	# this is a placeholder to tell the queue that there's no more images coming
	
	times = np.array(times) - t_start
	fname = joindata("recordings", f"rectime_stamp_{timestamp}.npy")
	np.save(fname, times)
	return times
	
def tt_from_queued_image(in_q, out_q, cmd_mtx, timestamp=get_timestamp()):
	"""
	Takes in images from the queue `in_q`,
	and converts them to tip-tilt values which get sent to the queue `out_q`.
	"""
	imflat = np.load(optics.imflat_path)
	fname = joindata("recordings", f"rectt_stamp_{timestamp}.npy")
	ttvals = []
	while True:
		# if you don't have any work, take a nap!
		if in_q.empty():
			time.sleep(dt)
		else:
			img = in_q.get()
			in_q.task_done()
			if img is not None:
				assert not np.allclose(img - imflat, 0), "not seeing any pixel noise"
				ttval = measure_tt(img - imflat, cmd_mtx).flatten()
				out_q.put(ttval)
				ttvals.append(ttval)
			else:
				ttvals = np.array(ttvals)
				np.save(fname, ttvals)
				return ttvals

def control_schedule_from_law(q, control, t=1):
	"""
	The SEAL schedule for a controller.

	Arguments
	---------
	q : Queue
	The queue to poll for new tip-tilt values.

	control : callable
	The function to execute control.
	"""
	last_tt = None # the most recently applied control command
	t1 = time.time()
	while time.time() < t1 + t:
		if q.empty():
			time.sleep(dt/2)
		else:
			tt = q.get()
			q.task_done()
			last_tt, dmc = control(tt, u=last_tt)
			optics.applydmc(dmc)

def record_experiment(record_path, control_schedule, dist_schedule, t=1, verbose=True):
	_, cmd_mtx = make_im_cm(verbose)
	bestflat, imflat = optics.refresh(verbose)
	baseline_ttvals = measure_tt(optics.getim() - imflat, cmd_mtx=cmd_mtx)

	i = 0
	imax = 10
	while np.any(np.abs(baseline_ttvals) > 0.01):
		warnings.warn(f"The system may not be aligned: baseline TT is {baseline_ttvals.flatten()}.")
		align_alpao_fast(manual=False, view=False)
		_, cmd_mtx = make_im_cm()
		bestflat, imflat = optics.refresh(verbose)
		baseline_ttvals = measure_tt(optics.getim() - imflat, cmd_mtx=cmd_mtx)
		i += 1
		if i > imax:
			print("Cannot align system: realign manually and try experiment again.")
			return [], []
	

	timestamp = get_timestamp()
	record_path = joindata(record_path) + f"_time_stamp_{timestamp}.npy"

	q_compute = Queue()
	q_control = Queue()
	record_thread = Thread(target=partial(record_im, duration=t, timestamp=timestamp), args=(q_compute,))
	compute_thread = Thread(target=partial(tt_from_queued_image, timestamp=timestamp), args=(q_compute, q_control, cmd_mtx,))
	control_thread = Thread(target=partial(control_schedule, t=t), args=(q_control,))
	command_thread = Thread(target=dist_schedule)

	if verbose:
		print("Starting recording and commands...")

	record_thread.start()
	compute_thread.start()
	control_thread.start()
	command_thread.start()

	q_compute.join()
	q_control.join()
	record_thread.join(t)
	compute_thread.join(t)
	control_thread.join(t)
	command_thread.join(t)

	if verbose:
		print("Done with experiment.")

	optics.applydmc(bestflat)

	timepath = joindata("recordings", f"rectime_stamp_{timestamp}.npy")
	ttpath = joindata("recordings", f"rectt_stamp_{timestamp}.npy")
	times = np.load(timepath)
	ttvals = np.load(ttpath)
	np.save(record_path, times)
	record_path = record_path.replace("time", "tt")
	np.save(record_path, ttvals)
	os.remove(timepath)
	os.remove(ttpath)
	return times, ttvals
