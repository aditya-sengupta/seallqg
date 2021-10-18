# authored by Aditya Sengupta

"""
These are constants for simulation mode; in hardware mode, these will be overridden by actual data.
"""
dmdims = (320, 320)
imdims = (320, 320)
fs = 100.0 # Hz
dt = 1 / fs
wav0 = 1.65e-6 #assumed wav0 for sine amplitude input in meters
beam_ratio = 5.361256544502618 #pixels/resel
tsleep = 0.02