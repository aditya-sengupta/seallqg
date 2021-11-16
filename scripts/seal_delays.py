import numpy as np
from matplotlib import pyplot as plt
import sys
import re
from os.path import join

fs = 100

def stamp_to_seconds(t):
    h, m, s, ns = [int(x) for x in re.search("(\d+):(\d+):(\d+),(\d+)", t).groups()]
    return 3600 * h + 60 * m + s + 1e-9 * ns

good_runs = ["03_55_23"]

exposures = []
measures = []
dmcs = []

total_nframes = 0
for fname in good_runs:
    with open(join("data", "log", f"log_16_11_2021_{fname}.log")) as file:
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

get_meanstd = lambda data: f"{round(np.mean(data), 3)} $\pm$ {round(np.std(data), 3)}"
measure_delays = (measures - exposures) * fs
control_delays = (dmcs - measures) * fs
total_delays = (dmcs - exposures) * fs

fig, axs = plt.subplots(1,3, figsize=(12,8))
def plot_delay_hist(data, i, xlabel, title):
    bins = int(max(data) * 1000 / fs) + 1
    axs[i].hist(data, bins=min(100, bins), range = (0, min(5, max(total_delays))))
    axs[i].set_xlabel(xlabel)
    axs[i].set_ylabel("Count")
    axs[i].set_title(f"{title}: {get_meanstd(data)} frames")

for (i, (data, xlabel, title)) in enumerate([
    (total_delays, "Exposure to DM command", "Overall AO loop"),
    (measure_delays, "Measurement to DM command", "Measurement"),
    (control_delays, "Exposure to measurement", "Controller")
]):
    plot_delay_hist(data, i, xlabel, title)
fig.suptitle("Delays in the AO loop on SEAL, in #frames")
plt.show()
#plt.savefig("../plots/seal_delay.pdf")