'''
process the Zygo data to flatten the MEMs kilo DM
'''

import numpy as np
#import pysao
from scipy.io import loadmat
#from scipy.ndimage import median_filter
#from scipy.ndimage.filters import generic_filter
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.ndimage.interpolation import rotate

fnames=['zygo_data/0volts_phase_nm.mat','zygo_data/0.5volts_phase_nm.mat']

data={}
for i in range(len(fnames)):
	data_uncrop=loadmat(fnames[i])['phase']
	#first, just looking at the images, defining a cropping sequence
	cropcen=(487,465)
	cropsize=610 #this size is carefully adjusted to that the end image shape is close to an integer multiple of 32
	ri=lambda inp: int(round(inp))
	cropim = lambda im: im[cropcen[1]-ri(cropsize/2):cropcen[1]+ri(cropsize/2),cropcen[0]-ri(cropsize/2):cropcen[0]+ri(cropsize/2)]
	data_crop=cropim(data_uncrop) #crop image
	data_conv=interpolate_replace_nans(data_crop,Gaussian2DKernel(x_stddev=data_crop.shape[0]/32/2.355)) #interpolate nans and convolve with 1 actuator pitch FWHM Gaussian)
	data_rot=rotate(data_conv,0.7) #correct for clocking, by eye
	crop_again=ri((data_rot.shape[0]-data_crop.shape[0])/2)
	data_rot_crop=data_rot[crop_again:-crop_again,crop_again:-crop_again] #crop again after rotation
	#print(data_rot_crop.shape[0]/32)
	data[str(i)]=data_rot_crop[:-1,:-1] #taking a row and column off the edge, this is needed to make the image size an integer multiple of 32

zygo0,zygo1=data['0'],data['1']

grid=np.mgrid[0:32,0:32]
tip,tilt=grid[0]-16+0.5,grid[1]-16+0.5
dmmap=np.zeros((32,32))
inddmap=np.where(np.sqrt(tip**2+tilt**2)<15) #only try to flatten actuators not in the corners and padding one from the edge
dmmap[inddmap]=1

wf_want=np.median(dmwf1[inddmap]) #the wavefront we want applied on the DM on top of 0.5v to flatten it

def binim(im): #bin into 32x32 DM space
	imreshape=im.reshape((32,im.shape[0]//32,32,im.shape[1]//32))
	imbin=np.median(np.median(imreshape,axis=-1),axis=1)
	return imbin

#remove tip/tilt
ref=np.zeros((2,1024))
ref[0]=(tip*dmmap).flatten()
ref[1]=(tilt*dmmap).flatten()
ttiminv=np.linalg.pinv(np.dot(ref,ref.T),rcond=1e-5)
ttcm=np.dot(ttiminv,ref).T
def rmtt(dmwf):
	coeffs=np.dot(np.vstack((dmwf*dmmap).flatten()).T,ttcm)
	return dmwf-np.dot(coeffs,ref).reshape(32,32)

dmwf0,dmwf1=rmtt(binim(zygo0)),rmtt(binim(zygo1)) #tip/tilt removed and binned

#goal: get a flat DM at mean of 0.5V, i.e., apply the negative wavefront of what is measured at 0.5 V to try and flatten it. To do this, we need to convert wavefront to volts for each actuator, which can be done by the difference between 0 and 0.5V frames being what wavefront 0.5V does to every actuator (assuming a linear fit)

wf2dm=(0-0.5)/(dmwf0-dmwf1) #converts wavefront units to DM units: 0.5v was applied for wf1, 0v for wf0; the wavefront decrease from wf0 to wf1 (meaning applying positive voltage of 0.5 decreases the wavefront by dmwf0-dmwf1); this is the slope of the relationship dm_command=wavefront*slope+intercept
intercept=-dmwf0*wf2dm #this is the intercept of the equation dm_command=wavefront*slope+intercept

dm_command = lambda wavefront: wavefront*wf2dm+intercept

dm_apply=dm_command(wf_want)
dm_apply[np.where(dmmap==0)]=0.5 #set the edges and corners equal to 0.5

