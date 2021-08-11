# authored by Benjamin Gerard and Aditya Sengupta (and Sylvain Cetre?)

from abc import ABC, abstractmethod
import numpy as np
import warnings

class Optics(ABC): # todo
	pass
	
hardware_mode = True

try:
	from krtc import shmlib
except ModuleNotFoundError:
	print("Running in simulation mode.")
	hardware_mode = False

#initialize; no need to load this more than once
	#for full frame:

if hardware_mode:
	import pysao
	"""try:
		ds9 = pysao.ds9()
	except OSError:
		pass"""

	a = shmlib.shm('/tmp/ca03dit.im.shm') 
	im = shmlib.shm('/tmp/ca03im.im.shm')

def expt(t):
	'''
	change the exposure time

	for the large array the smallest exposure time is around 1e-5
	'''
	dit = a.get_data()
	dit[0][0] = t; a.set_data(dit)

def get_expt():
	return a.get_data()[0][0]

def set_expt(t):
	expt(t)

#To view images in pysao ds9:
def getim():
	return im.get_data(check=True)

def stack(n):
	ims = getim()
	for _ in range(n-1):
		ims = ims+getim()
	ims = ims/n
	return ims

mtf = lambda image: np.abs(np.fft.fftshift(np.fft.fft2(image)))

#kilo DM commands
if hardware_mode:
	b = shmlib.shm('/tmp/dm02itfStatus.im.shm')
	status = b.get_data()
	status[0,0] = 1
	b.set_data(status)
	dmChannel = shmlib.shm('/tmp/dm02disp01.im.shm')

	def getdmc(): # read current command applied to the DM
		return dmChannel.get_data()

	def applydmc(cmd, verbose=True): #apply command to the DM
		"""
		Applies the DM command `cmd`.
		Returns two booleans: whether the command is in range below (everything is >=0), and above (everything is <=1),
		unless verbose=False, in which case nothing is returned.
		"""
		indneg = np.where(cmd<0)
		if len(indneg[0])>0:
			cmd[indneg] = 0 #minimum value is zero
			warnings.warn('saturating DM zeros!')
		indpos = np.where(cmd>1)
		if len(indpos[0])>0:
			cmd[indpos] = 1 #maximum value is 1
			warnings.warn('saturating DM ones!')
		dmChannel.set_data(cmd)
		if verbose:
			return (len(indneg[0]) <= 0, len(indpos[0]) <= 0)

dmzero = None
if hardware_mode:
	dmcini = getdmc()
	dmzero = np.zeros(dmcini.shape, dtype=np.float32)
	applyzero  =  lambda : applydmc(dmzero, False)
else:
	from ..constants import dmdims
	dmzero = np.zeros(dmdims, dtype=np.float32)