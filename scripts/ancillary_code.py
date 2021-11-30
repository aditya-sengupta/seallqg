'''
code for functions to read out images and apply and readout DM commands
'''

from krtc import *
import zmq
import numpy as np
import time

#initialize; no need to load this more than once
#for full frame:


# ANDOR CAMERA COMMANDS
#for large format
#a=shmlib.shm('/tmp/ca01dit.im.shm') 
#im=shmlib.shm('/tmp/ca01im.im.shm')

#for 320x320 subarray
a=shmlib.shm('/tmp/ca03dit.im.shm') 
im=shmlib.shm('/tmp/ca03im.im.shm')

def expt(t):
	'''
	change the exposure time

	for the large array the smallest exposure time is around 1e-5
	'''
	dit=a.get_data()
	dit[0][0]=t; a.set_data(dit)


#dark acquisition (frames with planet and start light source off; at the moment only the star can be controlled remotely...)
#imdark=stack(10000)
#np.save('imdark.npy',imdark)
#imdark=np.load('imdark.npy')
def getim():
	return im.get_data(check=True)#-imdark #check=True ensures in chopper mode it only gets a new image when the chopper triggers a new frame

def stack(n):
	ims=np.zeros(getim().shape)
	for i in range(n):
		ims=ims+getim()
	ims=ims/n
	return ims

mtf = lambda im: np.abs(np.fft.fftshift(np.fft.fft2(im)))


#ALPAO DM COMMANDS

lodmChannel1 = shmlib.shm('/tmp/dm03disp01.im.shm')
# To read what it currently in this DM channels
def getlodmc():
	return lodmChannel1.get_data()
def applylodmc(cmd): #apply command to the DM
	indneg=np.where(cmd<-0.2)
	if len(indneg[0])>0:
		cmd[indneg]=-0.2 #minimum value is -1, setting to -0.2 for safety
		print('saturating DM minus ones!')
	indneg=None
	indpos=np.where(cmd>0.2)
	if len(indpos[0])>0:
		cmd[indpos]=0.2 #maximum value is 1, setting to 0.2 for safety
		print('saturating DM ones!')
	indpos=None
	lodmChannel1.set_data(cmd)
lodmcini=getlodmc()
lodmzero=np.zeros(lodmcini.shape,dtype=np.float32)
applylozero = lambda : applylodmc(lodmzero)

imini=getim()
#make MTF side lobe mask
xsidemaskcen,ysidemaskcen=240.7,161.0 #x and y location of the side lobe mask in the cropped image
sidemaskrad=26.8 #radius of the side lobe mask
mtfgrid=np.mgrid[0:imini.shape[0],0:imini.shape[1]]
sidemaskrho=np.sqrt((mtfgrid[0]-ysidemaskcen)**2+(mtfgrid[1]-xsidemaskcen)**2)
sidemask=np.zeros(imini.shape)
sidemaskind=np.where(sidemaskrho<sidemaskrad)
sidemask[sidemaskind]=1

#central lobe MTF mask
yimcen,ximcen=imini.shape[0]/2,imini.shape[1]/2
cenmaskrad=49
cenmaskrho=np.sqrt((mtfgrid[0]-yimcen)**2+(mtfgrid[1]-ximcen)**2)
cenmask=np.zeros(imini.shape)
cenmaskind=np.where(cenmaskrho<cenmaskrad)
cenmask[cenmaskind]=1

#pinhole MTF mask
pinmaskrad=4
pinmask=np.zeros(imini.shape)
pinmaskind=np.where(cenmaskrho<pinmaskrad)
pinmask[pinmaskind]=1

def processim(imin,mask=sidemask): #process SCC image, isolating the sidelobe (or another binary mask for other cases) in the FFT and IFFT back to the image
	otf=np.fft.fftshift(np.fft.fft2(imin,norm='ortho')) #(1) FFT the image
	otf_masked=otf*mask #(2) multiply by binary mask to isolate side lobe
	Iminus=np.fft.ifft2(otf_masked,norm='ortho') #(3) IFFT back to the image plane, now generating a complex-valued image
	return Iminus

#WFS slopes
port="5556"
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://128.114.22.20:%s" % port)

def getPupilSize(sock):
    socket.send_string("pupSize");
    data=socket.recv()
    pupSize = np.frombuffer(data, dtype=np.int32)
    return pupSize

pupSize = getPupilSize(socket)[0]

def getWavefront():
    socket.send_string("wavefront");
    data=socket.recv()
    wf = np.frombuffer(data, dtype=np.float32).reshape(pupSize, pupSize)
    return wf

def stackWavefront(n): #average some number of frames of wavefront
	imw=np.zeros((n,pupSize,pupSize))
	for i in range(n):
		imw[i]=getWavefront()
	return np.nanmean(imw,axis=0)

def getSlopes():
    socket.send_string("slopes");
    data=socket.recv()
    slopes = np.frombuffer(data, dtype=np.float32).reshape(pupSize, 2*pupSize)
    sx = slopes[:,:pupSize]
    sy = slopes[:,pupSize:]
    return np.array([sx, sy])

def stackSlopes(n): #average some number of frames of slopes
	ims=np.zeros(getSlopes().shape)
	for i in range(n):
		ims=ims+getSlopes()
	ims=ims/n
	return ims





#wf = getWavefront()
#sx,sy = getSlopes()


'''
# Push each actuator
for k in range(0,32):
    for l in range(0,32):
     cmd=cmd*0;
     cmd[k][l] = 1
     dmChannel.set_data(cmd)
     time.sleep(0.2)
'''


##fig,axs=plt.subplots(ncols=1,nrows=2)

