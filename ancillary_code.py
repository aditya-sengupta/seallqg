'''
code for functions to read out images and apply and readout DM commands
'''

from krtc import *
import zmq
import pysao
import numpy as np
import sys
import time
import functions

#initialize; no need to load this more than once
#for full frame:

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

def getim():
	return im.get_data()
def vim(): #view image in ds9
	ds9.view(getim())

def vmtf(): #view image MTF
	imm=getim()
	mtf=np.abs(np.fft.fftshift(np.fft.fft2(imm)))
	ds9.view(mtf)

def stack(n):
	ims=np.zeros(getim().shape)
	for i in range(n):
		ims=ims+getim()
	ims=ims/n
	return ims

mtf = lambda im: np.abs(np.fft.fftshift(np.fft.fft2(im)))


#DM commands

b=shmlib.shm('/tmp/dm02itfStatus.im.shm')
status=b.get_data()
status[0,0]=1
b.set_data(status)
dmChannel=shmlib.shm('/tmp/dm02disp01.im.shm')

def getdmc(): # read current command applied to the DM
	return dmChannel.get_data()
def applydmc(cmd): #apply command to the DM
	indneg=np.where(cmd<0)
	if len(indneg[0])>0:
		cmd[indneg]=0 #minimum value is zero
		print('saturating DM zeros!')
	indneg=None
	indpos=np.where(cmd>1)
	if len(indpos[0])>0:
		cmd[indpos]=1 #maximum value is 1
		print('saturating DM ones!')
	indpos=None
	dmChannel.set_data(cmd)

dmcini=getdmc()
dmzero=np.zeros(dmcini.shape,dtype=np.float32)
applyzero = lambda : applydmc(dmzero)


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

def getwf():
    socket.send_string("wavefront");
    data=socket.recv()
    wf = np.frombuffer(data, dtype=np.float32).reshape(pupSize, pupSize)
    return wf

def stackwf(n): #average some number of frames of wavefront
	imw=np.zeros(getwf().shape)
	for i in range(n):
		imw=imw+getwf()
	imw=imw/n
	return imw

def getSlopes(): #something still wrong with slopes...
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

