import sys
sys.path.append("..")

import time
import numpy as np
from matplotlib import pyplot as plt

from src import *

times = []
t_start = time.time()

optics.getim()
for i in range(1000):
    t0 = time.time()
    im = optics.getim()
    print(f"Got image {i}")
    times.append(t0)

dtimes = np.diff(times)
mean, std = round(np.mean(dtimes), 4), round(np.std(dtimes), 4)
print(f"Exposure delay: {mean} +/- {std}")