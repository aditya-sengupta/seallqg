import numpy as np
from numpy import float32
from scipy import fft

from .utils import polar_grid, zernike
from .optics import optics
from ..utils import joindata

# all of this should be part of an Optics instance

ydim, xdim = optics.dmdims
grid = np.mgrid[0:ydim, 0:xdim].astype(float32)
bestflat = optics.bestflat
imydim, imxdim = optics.imdims

#make MTF side lobe mask
xsidemaskcen,ysidemaskcen=240.7,161.0 #x and y location of the side lobe mask in the cropped image
sidemaskrad=26.8 #radius of the side lobe mask
mtfgrid=np.mgrid[0:imydim,0:imxdim]
sidemaskrho=np.sqrt((mtfgrid[0]-ysidemaskcen)**2+(mtfgrid[1]-xsidemaskcen)**2)
sidemask=np.zeros(optics.imdims)
sidemaskind=np.where(sidemaskrho<sidemaskrad)
sidemask[sidemaskind]=1

#central lobe MTF mask
yimcen,ximcen=imydim/2,imxdim/2
cenmaskrad=49
cenmaskrho=np.sqrt((mtfgrid[0]-yimcen)**2+(mtfgrid[1]-ximcen)**2)
cenmask=np.zeros(optics.imdims)
cenmaskind=np.where(cenmaskrho<cenmaskrad)
cenmask[cenmaskind]=1

#pinhole MTF mask
pinmaskrad=4
pinmask=np.zeros(optics.imdims)
pinmaskind=np.where(cenmaskrho<pinmaskrad)
pinmask[pinmaskind]=1

#DM aperture;
xy=np.sqrt((grid[0]-ydim/2+0.5)**2+(grid[1]-xdim/2+0.5)**2)
aperture=np.zeros(optics.dmdims).astype(np.float32)
aperture[np.where(xy<ydim/2)]=1 
indap=np.where(aperture==1)
#indnap=np.where(aperture==0)
#inddmuse=np.where(aperture.flatten()==1)[0]
#nact=len(inddmuse)

tip,tilt=((grid[0]-ydim/2+0.5)/ydim*2).astype(np.float32),((grid[1]-xdim/2+0.5)/ydim*2).astype(np.float32)# DM tip/tilt 

remove_piston = lambda dmc: dmc-np.mean(dmc[indap]) #function to remove piston from dm command to have zero mean

def processim(imin): #process SCC image, isolating the sidelobe in the FFT and IFFT back to the image
	otf = np.fft.fftshift(fft.fft2(imin, norm='ortho')) #(1) FFT the image
	otf_masked = otf * sidemask #(2) multiply by binary mask to isolate side lobe
	Iminus = fft.ifft2(otf_masked, norm='ortho') #(3) IFFT back to the image plane, now generating a complex-valued image
	return Iminus

def applytiptilt(amptip, amptilt): #amp is the P2V in DM units
	dmctip = amptip*tip
	dmctilt = amptilt*tilt
	dmctiptilt = remove_piston(dmctip) + remove_piston(dmctilt) + remove_piston(bestflat) + 0.5 #combining tip, tilt, and best flat, setting mean piston to 0.5
	return optics.applydmc(dmctiptilt)

def applytip(optics, amp):
	dmc = optics.getdmc()
	dmctip = amp * tip
	dmc = remove_piston(dmc) + remove_piston(dmctip)
	optics.applydmc(dmc)

def applytilt(optics, amp):
	dmc = optics.getdmc()
	dmctilt = amp * tilt
	dmc = remove_piston(dmc) + remove_piston(dmctilt)
	optics.applydmc(dmc)
	
rho, phi = polar_grid(xdim, ydim)
rho[int((xdim-1)/2),int((ydim-1)/2)]=0.00001 #avoid numerical divide by zero issues

def funz(n, m, amp, bestflat): #apply zernike to the DM
	z = zernike(n, m, rho, phi)/2
	zdm = amp*(z.astype(float32))
	dmc = remove_piston(remove_piston(bestflat)+remove_piston(zdm))
	optics.applydmc(dmc)
	return zdm #even though the Zernike is applied with a best flat, return only the pure Zernike; subsequent reconstructed Zernike mode coefficients should not be applied to best flat commands
