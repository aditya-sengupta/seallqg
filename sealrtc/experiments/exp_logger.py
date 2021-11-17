import sys
import time
import logging
import re

from ..utils import joindata

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

def experiment_logger(timestamp):
    logging.setLogRecordFactory(LogRecord_ns)

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = Formatter_ns('%(asctime)s | %(levelname)s | %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    log_path = joindata("log", f"log_{timestamp}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger, log_path
