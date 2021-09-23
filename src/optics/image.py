# authored by Benjamin Gerard and Aditya Sengupta (and Sylvain Cetre?)

from abc import ABC, abstractmethod
import numpy as np
from copy import copy
import time
import warnings
from os import path

from .par_functions import propagate
from ..utils import joindata

optics = None

class Optics(ABC): 
	@property
	def dmzero(self):
		return np.zeros(self.dmdims, dtype=np.float32)

	def applyzero(self):
		self.applydmc(self.dmzero)

	def refresh(self, verbose=True):
		bestflat = np.load(joindata(path.join("bestflats", "bestflat_{0}_{1}.npy".format(self.name, self.dmdims[0]))))
		dmc = self.getdmc()
		self.applydmc(bestflat)
		imflat = self.stack(100)
		np.save(joindata(path.join("bestflats", "imflat_{0}_{1}.npy".format(self.name, self.imdims[0])), imflat), imflat)
		if verbose:
			print("Updated the flat image.")
		self.applydmc(dmc)
		return bestflat, imflat

	def stack(self, n):
		ims = self.getim()
		for _ in range(n - 1):
			ims = ims + self.getim()
		ims = ims / n
		return ims

	@abstractmethod
	def getim(self):
		pass

	@abstractmethod
	def getdmc(self):
		pass

	@abstractmethod
	def applydmc(self, cmd):
		pass

	@abstractmethod
	def set_expt(self, t):
		pass

	@abstractmethod
	def get_expt(self):
		pass

class FAST(Optics):
	def __init__(self):
		from krtc import shmlib
		self.a = shmlib.shm('/tmp/ca03dit.im.shm') 
		self.im = shmlib.shm('/tmp/ca03im.im.shm')
		self.b = shmlib.shm('/tmp/dm02itfStatus.im.shm')
		status = self.b.get_data()
		status[0,0] = 1
		self.b.set_data(status)
		self.dmChannel = shmlib.shm('/tmp/dm02disp01.im.shm')
		self.dmdims = self.getdmc().shape
		self.imdims = self.getim().shape
		self.name = "FAST"

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
		return self.im.get_data(check=True)

	def getdmc(self): # read current command applied to the DM
		return self.dmChannel.get_data()

	def applydmc(self, dmc, min_cmd=0.05, max_cmd=0.95):
		"""
		Applies the DM command `dmc`, with safeguards
		"""
		if np.any(dmc < min_cmd):
			warnings.warn("saturating DM zeros!")
		if np.any(dmc > max_cmd):
			warnings.warn("saturating DM ones!")
		dmc = np.maximum(min_cmd, np.minimum(max_cmd, dmc))
		self.dmChannel.set_data(dmc.astype(np.float32))

class Sim(Optics):
	def __init__(self):
		from ..constants import dmdims, imdims, dt
		self.dmdims = imdims # dmdims
		self.imdims = imdims
		self.expt = 1e-3
		self.dt = dt
		self.dmc = copy(self.dmzero)
		self.name = "Sim"

	def set_expt(self, t):
		self.expt = t

	def get_expt(self):
		return self.expt

	def getim(self):
		return propagate(self.dmc, ph=True, t_int=self.expt)

	def getdmc(self):
		return self.dmc

	def applydmc(self, dmc):
		assert self.dmc.shape == dmc.shape
		self.dmc = np.maximum(0, np.minimum(1, dmc))
	
try:
	from krtc import shmlib
	optics = FAST()
except (ModuleNotFoundError, OSError):
	print("Running in simulation mode.")
	optics = Sim()
