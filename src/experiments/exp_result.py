import time
import re
import numpy as np
import tqdm
import pandas as pd

def stamp_to_seconds(t):
    h, m, s, ns = [int(x) for x in re.search(r"(\d+):(\d+):(\d+),(\d+)", t).groups()]
    return 3600 * h + 60 * m + s + 1e-9 * ns

def string_to_numpy(s):
    return np.array(s.split(" "), dtype=np.float64)

class ExperimentResult:
    def __init__(self, log_path):
        texp = [] # time when exposure is logged
        tmeas = [] # time when measurement is logged
        tdmc = [] # time when DMC is logged
        texp_loop = [] # internal time at exposure: what the AO loop thinks is the start of the iteration
        measurements = [] # Zernike coefficient measurement
        commands = [] # Tip-tilt command applied

        with open(log_path) as file:
            final_frame = np.inf
            for line in tqdm.tqdm(file):
                tstamp = re.search(r"\d+:\d+:\d+,\d+", line)
                if tstamp:
                    tstamp = tstamp[0]
                    seconds = stamp_to_seconds(tstamp)
                    event = re.search(r"INFO \| (.+)", line)[1]
                    if any([event.startswith(x) for x in ["Exposure", "Measurement", "DMC"]]):
                        frame_num = re.search(r"(\d+):", event)
                        if frame_num:
                            frame_num = int(frame_num[1])
                        data = re.search(r"\[(.*)\]", event)
                        if data:
                            data = string_to_numpy(data)
                        if event.startswith("Exposure"):
                            texp.append(seconds)
                            texp_loop.append(data)
                        elif event.startswith("Measurement"):
                            tmeas.append(seconds)
                            measurements.append(data)
                        elif event.startswith("DMC"):
                            tdmc.append(seconds)
                            commands.append(data)
                            final_frame = frame_num
        
        t0 = texp[0]
        self.texp = np.array(texp) - t0
        self.tmeas = np.array(tmeas) - t0
        self.tdmc = np.array(tdmc) - t0
        self.texp_loop = np.array(texp_loop).flatten()
        self.measurements = np.array(measurements)
        self.commands = np.array(commands)  



    def delay_hist(self):
        raise RuntimeError("move this over from the scripts")