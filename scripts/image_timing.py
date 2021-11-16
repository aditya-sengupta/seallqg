import sys
sys.path.append("..")

import time 
import numpy as np
from matplotlib import pyplot as plt

from queue import Queue
import logging

from src import *
from src.experiments.exp_runner import record_im

q = Queue()

times = record_im(q, 1, "", logging)

"""times = []
t_start = time.time()

optics.getim()
for i in range(1000):
    t0 = time.time()
    im = optics.getim()
    print(f"Got image {i}")
    times.append(t0)
"""
dtimes = np.diff(times[10:])
mean, std = round(np.mean(dtimes), 4), round(np.std(dtimes), 4)
print(f"Exposure delay: {mean} +/- {std}")
plt.hist(dtimes)
plt.show()