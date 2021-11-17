import time
import re
import numpy as np
import tqdm
import pandas as pd

from .utils import stamp_to_seconds, string_to_numpy
from ..utils import joindata

"""
Note that this is not a great implementation when you get significantly more results than this, 
and the whole concept is kind of janky, but I want to cut down on latency in the actual loop,
so this is what we're doing I guess.
"""
def result_from_log(log_path):
    texp = [] # time when exposure is logged
    tmeas = [] # time when measurement is logged
    tdmc = [] # time when DMC is logged
    texp_loop = [] # internal time at exposure: what the AO loop thinks is the start of the iteration
    measurements = [] # Zernike coefficient measurement
    commands = [] # Tip-tilt command applied

    with open(log_path) as file:
        last_line = None
        overrun_corrected = False
        final_frame = np.inf
        for line in tqdm.tqdm(file):
            if last_line is not None:
                assert "]" in line and "[" not in line, f"Overrun at {last_line} corrected with {line}"
                line = last_line + line
                line = line.replace('\n', ' ')
                last_line = None
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
                    if data is None:
                        last_line = line
                    else:
                        data = string_to_numpy(data[1])
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
    return ExperimentResult([
        np.array(texp) - t0,
        np.array(tmeas) - t0,
        np.array(tdmc) - t0,
        np.array(texp_loop).flatten() - texp_loop[0],
        np.array(measurements),
        np.array(commands)
    ])

"""
Data container for all the results of an experiment. 
Interoperates with pd.DataFrame and contains result analysis functionality.
"""
class ExperimentResult:
    def __init__(self, data):
        l = len(min(data, key=len))
        data = [d[:l] for d in data]
        self.texp, self.tmeas, self.tdmc, self.texp_loop, self.measurements, self.commands = data

    def to_pandas(self):
        d = {
            "texp" : self.texp,
            "tmeas" : self.tmeas,
            "tdmc" : self.tdmc,
            "texp_loop" : self.texp_loop
        }
        d.update({
            f"meas{i}" : self.measurements[:,i] for i in range(self.measurements.shape[1])
        })
        d.update({
            f"cmd{i}" : self.commands[:,i] for i in range(self.commands.shape[1])
        })
        return pd.DataFrame(data=d)

    def to_csv(self, record_path, params={}):
        f = open(joindata(record_path), 'a')
        for p in params:
            f.write(f'# {p}: {params[p]} \n')
        self.to_pandas().to_csv(f)
        f.close()

    def delay_hist(self):
        raise RuntimeError("move this over from the scripts")

def loadres(record_path):
    df = pd.read_csv(joindata(record_path))
    data = list(map(
            lambda name: df[name].to_numpy(),
            ["texp", "tmeas", "tdmc", "texp_loop"]
        )
    )
    measure_cols = filter(lambda x: x.startswith("meas"), df.columns)
    data.append(df[measure_cols].to_numpy())
    control_cols = filter(lambda x: x.startswith("cmd"), df.columns)
    data.append(df[control_cols].to_numpy())

    return ExperimentResult(data)
