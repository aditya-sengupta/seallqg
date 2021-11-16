# authored by Benjamin Gerard and Aditya Sengupta (and Sylvain Cetre?)

import warnings
import time
from abc import ABC, abstractmethod, abstractproperty
from copy import copy
from os import path
import numpy as np
from socket import gethostname

from .par_functions import propagate
from ..utils import joindata
from ..constants import dmdims, imdims, dt

optics = None

class Optics(ABC):
	"""
	Driver class for the DM-WFS-image loop.
	Supports updating the DM command, viewing the image, getting slopes, images, and saving data.
	"""
	@property
	def dmzero(self):
		"""
		The zero state for the DM.
		"""
		return np.zeros(self.dmdims, dtype=np.float32)

	@property
	def bestflat_path(self):
		"""
		The path to load/save the best flat.
		"""
		return joindata("bestflats", f"bestflat_{self.name}_{self.dmdims[0]}.npy")

	@property
	def imflat_path(self):
		"""
		The path to load/save the image generated from the best flat.
		"""
		return joindata("bestflats", f"imflat_{self.name}_{self.imdims[0]}.npy")

	def applybestflat(self):
		self.applydmc(self.bestflat)

	def applyzero(self):
		self.applydmc(self.dmzero)

	@property
	def bestflat(self):
		"""
		The best-flat position from file.
		"""
		return np.load(self.bestflat_path)

	def refresh(self, verbose=True):
		self.applybestflat()
		time.sleep(1)
		imflat = self.stackim(100)
		np.save(self.imflat_path, imflat)
		if verbose:
			print("Updated the flat image.")
		return self.getdmc(), imflat

	def stack(self, func, num_frames):
		"""
		Average a measurement of some function over `num_frames` frames.
		"""
		ims = func()
		for _ in range(num_frames - 1):
			ims = ims + func()
		
		ims = np.nan_to_num(ims)
		return ims / num_frames

	def stackwf(self, num_frames):
		return self.stack(self.getwf, num_frames)

	def stackim(self, num_frames):
		return self.stack(self.getim, num_frames)

	def stackslopes(self, num_frames):
		return self.stack(self.getslopes, num_frames)

	@abstractmethod
	def getim(self):
		pass

	@abstractmethod
	def getdmc(self):
		pass

	@abstractmethod
	def applydmc(self, dmc, **kwargs):
		pass

	@abstractmethod
	def set_expt(self, t):
		pass

	@abstractmethod
	def get_expt(self):
		pass

	@abstractmethod
	def getwf(self):
		pass

	@abstractmethod
	def getslopes(self):
		pass

class FAST(Optics):
	# updated 6 Oct 2021 for the new ALPAO DM
	def __init__(self):
		from krtc import shmlib
		import zmq
		self.a = shmlib.shm('/tmp/ca03dit.im.shm') 
		self.im = shmlib.shm('/tmp/ca03im.im.shm')
		self.b = shmlib.shm('/tmp/dm02itfStatus.im.shm')
		status = self.b.get_data()
		status[0,0] = 1
		self.b.set_data(status)
		self.dmChannel = shmlib.shm('/tmp/dm03disp01.im.shm')
		self.dmdims = self.getdmc().shape
		self.imdims = self.getim().shape
		self.name = "FAST_LODM"
		port = "5556"
		context = zmq.Context()
		self.socket = context.socket(zmq.REQ)
		self.socket.connect(f"tcp://128.114.22.20:{port}")
		self.socket.send_string("pupSize")
		self.pup_size = np.frombuffer(self.socket.recv(), dtype=np.int32)[0]

	def set_expt(self, t):
		'''
		change the exposure time

		for the large array the smallest exposure time is around 1e-5
		'''
		dit = self.a.get_data()
		dit[0][0] = t; self.a.set_data(dit)
	
	def get_expt(self):
		return self.a.get_data()[0][0]

	def getim(self):
		return self.im.get_data(check=False)

	def getdmc(self): # read current command applied to the DM
		return self.dmChannel.get_data()

	def applydmc(self, dmc, min_cmd=-0.2, max_cmd=0.2): #apply command to the DM
		"""
		Applies the DM command `dmc`, with safeguards
		"""
		if np.any(dmc < min_cmd):
			warnings.warn("saturating DM zeros!")
		if np.any(dmc > max_cmd):
			warnings.warn("saturating DM ones!")
		dmc = np.maximum(min_cmd, np.minimum(max_cmd, dmc))
		dmc = np.nan_to_num(dmc)
		self.dmChannel.set_data(dmc.astype(np.float32))

	def getwf(self):
		self.socket.send_string("wavefront")
		data = self.socket.recv()
		return np.frombuffer(data, dtype=np.float32).reshape(self.pup_size, self.pup_size)

	def getslopes(self):
		self.socket.send_string("slopes")
		data = self.socket.recv()
		slopes = np.frombuffer(data, dtype=np.float32).reshape(self.pup_size, 2*self.pup_size)
		slopes_x = slopes[:,:self.pup_size]
		slopes_y = slopes[:,self.pup_size:]
		return np.array([slopes_x, slopes_y])

class Sim(Optics):
	"""
	A simulated adaptive optics loop.
	"""
	def __init__(self):
		self.dmdims = dmdims
		self.imdims = imdims
		self.expt = 1e-3
		self.dt = dt
		self.dmc = copy(self.dmzero)
		self.name = "Sim"
		self.wait = 0
		self.set_wait()
	
	def set_wait(self):
		t0 = time.time()
		for _ in range(10):
			self.getim()
		t1 = time.time()
		self.wait = max(0, self.dt - (t1 - t0)/10)

	def set_expt(self, t):
		self.expt = t

	def get_expt(self):
		return self.expt

	def getim(self):
		time.sleep(self.wait)
		return propagate(self.dmc, ph=True, t_int=self.expt)

	def stackim(self, n):
		return self.getim()

	def getdmc(self):
		return self.dmc

	def applydmc(self, dmc):
		assert self.dmc.shape == dmc.shape
		self.dmc = np.maximum(0, np.minimum(1, dmc))

	def getslopes(self):
		raise NotImplementedError()

	def getwf(self):
		raise NotImplementedError()

sim_mode = False
if gethostname() == "SEAL" and not sim_mode:
	optics = FAST()
else:
	print("Running in simulation mode.")
	optics = Sim()
