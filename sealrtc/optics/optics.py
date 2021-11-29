import time
from abc import ABC, abstractmethod, abstractproperty
from os import path
import numpy as np

from .utils import nmarr, polar_grid, zernike
from ..utils import joindata
from ..utils import tsleep

class Optics(ABC):
	"""
	Driver class for the DM-WFS-image loop.
	Supports updating the DM command, viewing the image, getting slopes, images, and saving data.
	"""
	def set_process_vars(self):
		self.bestflat = np.load(self.bestflat_path)
		self.imflat = np.load(self.imflat_path)
		self.dmzero = np.zeros(self.dmdims)
		imydim, imxdim = self.imdims
		ydim, xdim = self.dmdims
		self.rho, self.phi = polar_grid(xdim, ydim)
		self.rho[int((xdim-1)/2), int((ydim-1)/2)] = 0.00001 # avoid numerical divide by zero issues
		grid = np.mgrid[0:ydim, 0:xdim]
		self.grid = grid
		self.tip = ((grid[0]-ydim/2+0.5)/ydim*2)
		self.tilt = ((grid[1]-xdim/2+0.5)/ydim*2)

		#make MTF side lobe mask
		xsidemaskcen, ysidemaskcen = 240.7,161.0 #x and y location of the side lobe mask in the cropped image
		sidemaskrad= 26.8 #radius of the side lobe mask
		mtfgrid = np.mgrid[0:imydim, 0:imxdim]
		sidemaskrho = np.sqrt((mtfgrid[0]-ysidemaskcen)**2+(mtfgrid[1]-xsidemaskcen)**2)
		sidemask = np.zeros(self.imdims)
		self.sidemaskind = np.where(sidemaskrho < sidemaskrad)
		sidemask[self.sidemaskind] = 1
		self.mtfgrid = mtfgrid
		self.sidemask = sidemask

		#central lobe MTF mask
		yimcen,ximcen = imydim/2,imxdim/2
		cenmaskrad = 49
		cenmaskrho = np.sqrt((mtfgrid[0]-yimcen)**2+(mtfgrid[1]-ximcen)**2)
		cenmask = np.zeros(self.imdims)
		self.cenmaskind = np.where(cenmaskrho < cenmaskrad)
		cenmask[self.cenmaskind] = 1

		#pinhole MTF mask
		pinmaskrad = 4
		pinmask = np.zeros(self.imdims)
		pinmaskind=np.where(cenmaskrho < pinmaskrad)
		pinmask[pinmaskind] = 1

		#DM aperture;
		xy = np.sqrt((grid[0]-ydim/2+0.5)**2+(grid[1]-xdim/2+0.5)**2)
		aperture = np.zeros(self.dmdims)
		aperture[np.where(xy<ydim/2)] = 1 
		self.aperture = aperture
		self.indap = np.where(aperture == 1)
		self.dmc2wf = np.load(joindata("bestflats", "lodmc2wfe.npy"))
		#calibrated image center and beam ratio from genDH.py
		imycen, imxcen = np.load(joindata("bestflats", "imcen.npy"))
		beam_ratio = np.load(joindata("bestflats", "beam_ratio.npy"))

		gridim = np.mgrid[0:imydim,0:imxdim]
		rim = np.sqrt((gridim[0]-imxcen)**2+(gridim[1]-imycen)**2) # TODO double check this
		self.ttmask = np.zeros(self.imdims)
		rmask = 10
		self.indttmask = np.where(rim / beam_ratio < rmask)
		self.ttmask[self.indttmask] = 1
		self.IMamp = 0.001
		self.zernarr = np.zeros((len(nmarr), aperture[self.indap].shape[0]))
		for (i, (n, m)) in enumerate(nmarr):
			self.zernarr[i] = self.funz(n, m)[self.indap]
	
		self.make_im_cm()
		
	def processim(self, imin): #process SCC image, isolating the sidelobe in the FFT and IFFT back to the image
		otf = np.fft.fftshift(np.fft.fft2(imin, norm='ortho')) #(1) FFT the image
		otf_masked = otf * self.sidemask #(2) multiply by binary mask to isolate side lobe
		Iminus = np.fft.ifft2(otf_masked, norm='ortho') #(3) IFFT back to the image plane, now generating a complex-valued image
		return Iminus

	def remove_piston(self, dmc):
		return dmc - np.mean(dmc[self.indap])

	def applytip(self, amp):
		dmc = self.getdmc()
		dmctip = amp * self.tip
		dmc = self.remove_piston(dmc) + self.remove_piston(dmctip)
		self.applydmc(dmc)

	def applytilt(self, amp):
		dmc = self.getdmc()
		dmctilt = amp * self.tilt
		dmc = self.remove_piston(dmc) + self.remove_piston(dmctilt)
		self.applydmc(dmc)

	def applytiptilt(self, amptip, amptilt): #amp is the P2V in DM units
		dmctip = amptip * self.tip
		dmctilt = amptilt * self.tilt
		dmctiptilt = self.remove_piston(dmctip) + self.remove_piston(dmctilt) + self.remove_piston(self.bestflat) + 0.5 #combining tip, tilt, and best flat, setting mean piston to 0.5
		return self.applydmc(dmctiptilt)

	def funz(self, n, m, amp=None): #apply zernike to the DM
		if amp is None:
			amp = self.IMamp
		z = zernike(n, m, self.rho, self.phi)/2
		zdm = amp * z.astype(np.float32)
		dmc = self.remove_piston(self.remove_piston(self.bestflat) + self.remove_piston(zdm))
		self.applydmc(dmc)
		return zdm #even though the Zernike is applied with a best flat, return only the pure Zernike; subsequent reconstructed Zernike mode coefficients should not be applied to best flat commands

	def make_im_cm(self, rcond=1e-3, attempts=0):
		"""
		Make updated interaction and command matrices.
		"""
		self.set_expt(1e-3)
		self.refresh()
		refvec = np.zeros((len(nmarr), self.ttmask[self.indttmask].shape[0]*2))
		for (i, (n, m)) in enumerate(nmarr):
			_ = self.funz(n, m)
			time.sleep(tsleep)
			imzern = self.stackim(10)
			try:
				imdiff = imzern - self.imflat
				assert not np.allclose(imdiff, 0), "didn't move"
			except AssertionError:
				if attempts < 5:
					self.make_im_cm(rcond, attempts = attempts + 1)
				else:
					raise
			processed_imdiff = self.processim(imdiff)
			refvec[i] = np.array([
				np.real(processed_imdiff[self.indttmask]),
				np.imag(processed_imdiff[self.indttmask])
			]).flatten()


		assert not np.allclose(refvec, 0), "Zero measurements in all components"
		self.int_mtx = np.dot(refvec, refvec.T) #interaction matrix
		int_mtx_inv = np.linalg.pinv(self.int_mtx, rcond=rcond)
		self.cmd_mtx = np.dot(int_mtx_inv, refvec)#.astype(np.float32)

	def measure(self, image=None):
		"""
		Measures Zernike coefficient values from an image relative to the flat image.
		"""
		if image is None:
			image = self.getim()
		tar_ini = self.processim(image - self.imflat)
		tar = np.array([np.real(tar_ini[self.indttmask]), np.imag(tar_ini[self.indttmask])])
		tar = tar.reshape((tar.size, 1))
		coeffs = np.dot(self.cmd_mtx, tar).flatten()
		return coeffs * self.IMamp

	def zcoeffs_to_dmc(self, zcoeffs):
		"""
		Converts a measured coefficient value to an ideal DM command.
		
		Arguments
		---------
		zcoeffs : np.ndarray, (ncoeffs, 1)
		The tip and tilt values.

		Returns
		-------
		dmc : np.ndarray
		The corresponding DM command.
		"""
		dmc = np.copy(self.dmzero)
		dmc[self.indap] = np.dot(self.zernarr.T, -np.pad(zcoeffs, (0, 3)))
		return dmc

	def genzerncoeffs(self, i, zernamp):
		"""
		i: zernike mode
		zernamp: Zernike amplitude in DM units to apply

		this is a bit redundant, but i need it to track down errors
		"""
		n, m = nmarr[i]
		_ = self.funz(n, m, zernamp)
		time.sleep(tsleep)
		imzern = self.stackim(10)
		imdiff = imzern - self.imflat
		tar_ini = self.processim(imdiff)
		tar = np.array([np.real(tar_ini[self.indttmask]),np.imag(tar_ini[self.indttmask])]).flatten()
		return np.dot(self.cmd_mtx, tar) * self.IMamp

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

	def refresh(self):
		self.applybestflat()
		time.sleep(1)
		self.imflat = self.stackim(100)
		np.save(self.imflat_path, self.imflat)

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
