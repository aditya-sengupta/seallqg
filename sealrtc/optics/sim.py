import time
import numpy as np

from .optics import Optics
from .par_functions import propagate
from ..utils import dmdims, imdims, dt
from copy import copy

class Sim(Optics):
	"""
	A simulated adaptive optics loop.
	"""
	def __init__(self):
		self.dmdims = dmdims
		self.imdims = imdims
		self.dummy_image = np.zeros(self.imdims)
		self.expt = 1e-3
		self.dt = dt
		self.name = "Sim"
		self.dmc = copy(np.zeros(self.dmdims))
		self.set_process_vars()

	def set_expt(self, t):
		self.expt = t

	def get_expt(self):
		return self.expt

	def getim(self, check=True):
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