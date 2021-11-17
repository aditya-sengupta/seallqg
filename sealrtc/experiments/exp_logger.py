import sys
import time
import logging
import re

from .utils import LogRecord_ns, Formatter_ns
from ..utils import joindata

def experiment_logger(timestamp):
    logging.setLogRecordFactory(LogRecord_ns)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = Formatter_ns('%(asctime)s | %(levelname)s | %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.ERROR)
    stdout_handler.setFormatter(formatter)
    log_path = joindata("log", f"log_{timestamp}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger, log_path