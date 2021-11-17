import logging
import time
import re
import numpy as np

def stamp_to_seconds(t):
    h, m, s, ns = [int(x) for x in re.search(r"(\d+):(\d+):(\d+),(\d+)", t).groups()]
    return 3600 * h + 60 * m + s + 1e-9 * ns

def string_to_numpy(s):
    return np.array(
        list(filter(
            lambda x: bool(x) and not x.isspace(),
            s.split(" ")
        )),
        dtype=np.float64
    )

# from https://stackoverflow.com/questions/31328300/python-logging-module-logging-timestamp-to-include-microsecond
class LogRecord_ns(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        self.created_ns = time.time_ns() # Fetch precise timestamp
        super().__init__(*args, **kwargs)

class Formatter_ns(logging.Formatter):
    default_nsec_format = '%s,%09d'
    def formatTime(self, record, datefmt=None):
        if datefmt is not None: # Do not handle custom formats here ...
            return super().formatTime(record, datefmt) # ... leave to original implementation
        ct = self.converter(record.created_ns / 1e9)
        t = time.strftime(self.default_time_format, ct)
        s = self.default_nsec_format % (t, record.created_ns - (record.created_ns // 10**9) * 10**9)
        return s
