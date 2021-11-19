import time
import warnings
import numpy as np

from .optics import Optics
from ..constants import dt

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
		self.set_process_vars()
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
