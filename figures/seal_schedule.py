import numpy as np
from matplotlib import pyplot as plt
import sys
import re
from os.path import join

fs = 100

def stamp_to_seconds(t):
    h, m, s, ms = [int(x) for x in re.search("(\d+):(\d+):(\d+),(\d+)", t).groups()]
    return 3600 * h + 60 * m + s + 0.001 * ms

good_runs = ["07_02_14", "07_04_45", "08_11_19", "08_11_46"]

exposures = []
measures = []
dmcs = []

total_nframes = 0
for fname in good_runs:
    with open(join("..", "data", "log", f"log_13_11_2021_{fname}.log")) as file:
        final_frame = np.inf
        for line in file:
            time = re.search("\d+:\d+:\d+,\d+", line)[0]
            seconds = stamp_to_seconds(time)
            event = re.search("INFO \| (.+)", line)[1]
            if not any([event.startswith(x) for x in ["Exposure", "Measurement", "DMC"]]):
                continue
            frame_num = re.search("\d+", event)
            if frame_num:
                frame_num = int(frame_num[0])
            if event.startswith("Exposure"):
                exposures.append(seconds)
            elif event.startswith("Measurement"):
                measures.append(seconds)
            elif event.startswith("DMC"):
                dmcs.append(seconds)
                final_frame = frame_num
        total_nframes += final_frame
        exposures = exposures[:total_nframes]
        measures = measures[:total_nframes]
        dmcs = dmcs[:total_nframes]

t0 = exposures[0]
exposures = np.array(exposures) - t0
measures = np.array(measures) - t0
dmcs = np.array(dmcs) - t0

nstart = 1200
npoints = 30
tstart = exposures[nstart]
plt.figure(figsize=(8,6))
plt.scatter(exposures[nstart:nstart+npoints]-tstart, 1.5*np.ones_like(exposures[nstart:nstart+npoints]), label="exposures")
plt.scatter(measures[nstart:nstart+npoints]-tstart, np.ones_like(measures[nstart:nstart+npoints]), label="measurements")
plt.scatter(dmcs[nstart:nstart+npoints]-tstart, 0.5*np.ones_like(dmcs[nstart:nstart+npoints]), label="dmcs")
min_time = exposures[nstart]
max_time = dmcs[nstart+npoints-1]
for v in np.arange(exposures[nstart], exposures[nstart+npoints]+0.01, 0.01):
    plt.axvline(v-tstart, color='k')
plt.title("SEAL schedule observations")
plt.xlabel("Time (seconds)")
plt.yticks([])
plt.legend(bbox_to_anchor=(0.27, 0.2))
plt.savefig("seal_schedule.pdf", bbox_inches="tight")