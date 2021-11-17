"""
FAST System Identification Data Collection Mega-Schedule
"""
from sealrtc import *
import numpy as np

optics.set_expt(1e-3)

print("FAST System Identification Data Collection Mega-Schedule")
print("--------------------------------------------------------")

# open loops

print("Open loops")
print("No disturbance")
record_olnone(t=100, verbose=False)

"""print("Unit steps")
for amp in [0.05, 0.10, 0.15]:
    record_olustep(t=10, tip_amp=amp, tilt_amp=0.0, verbose=False)
    record_olustep(t=10, tip_amp=0.0, tilt_amp=amp, verbose=False)"""

print("Sine waves")
for amp in [0.01, 0.05, 0.10]:
    for ang in (np.pi/4) * np.arange(8):
        record_olsin(t=20, amp=amp, ang=ang, f=0.2, verbose=False)

print("atm-vib")
for (atm, vib) in zip([0, 0, 0, 1, 1, 1, 2, 6], [2, 3, 10, 0, 2, 3, 0, 0]):
    record_olatmvib(t=100, atm=atm, vib=vib, scaledown=5, verbose=False)

# integrators

for gain in [0.1, 0.2]:
    print(f"Integrator, gain = {gain}")
    print("No disturbance")
    record_intnone(t=100, verbose=False)

    print("Unit steps")
    for amp in [0.05, 0.10, 0.15]:
        record_intustep(gain=gain, t=10, tip_amp=amp, tilt_amp=0.0, verbose=False)
        record_intustep(gain=gain, t=10, tip_amp=0.0, tilt_amp=amp, verbose=False)

    print("Unit step trains")
    for amp in [0.01, 0.05, 0.10, 0.15]:
        record_inttrain(gain=gain, t=20, n=40, tip_amp=amp, tilt_amp=0.0, verbose=False)
        record_inttrain(gain=gain, t=20, n=40, tip_amp=0.0, tilt_amp=amp, verbose=False)

    print("Sine waves")
    for amp in [0.01, 0.05, 0.10]:
        for ang in (np.pi/4) * np.arange(8):
            record_intsin(gain=gain, t=20, amp=amp, ang=ang, f=0.2, verbose=False)

    print("atm-vib")
    for (atm, vib) in zip([0, 0, 0, 1, 1, 1, 2, 6], [2, 3, 10, 0, 2, 3, 0, 0]):
        record_intatmvib(gain=gain, t=100, atm=atm, vib=vib, scaledown=5, verbose=False)
    