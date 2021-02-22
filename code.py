from krtc import *
import pysao
a=shmlib.shm('/tmp/ca02dit.im.shm')
dit=a.get_data()
dit[0][0]=0.001; a.set_data(dit)

#To view images in pysao ds9:
im=shmlib.shm('/tmp/ca02im.im.shm')
#ds9.view(im.get_data())
