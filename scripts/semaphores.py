from threading import Thread, Semaphore, Lock
from time import sleep
from time import monotonic_ns as mns
import numpy as np
from functools import partial

def spin(process, dt, dur):
	"""
	Spin-locks around a process to do it every "dt" seconds for time "dur" seconds.
	"""
	t0 = mns()
	t1 = t0
	ticks_loop = int(np.ceil(dur / 1e-9))
	ticks_inner = int(np.ceil(dt / 1e-9))
	sleep(dt / 2)
	while mns() - t0 <= ticks_loop:
		process()
		i = 0
		sleep(dt / 2)
		while mns() - t1 <= ticks_inner:
			i += 1
		t1 += ticks_inner

mutex = Lock()

def function(word):
    with mutex:
        print(word)

spinner = lambda word, delay: sleep(delay) or spin(partial(function, word), 0.1, 0.51)

thread_one = Thread(target=partial(spinner, "one", 0))
thread_two = Thread(target=partial(spinner, "two", 0.001))

thread_one.start()
thread_two.start()

thread_one.join()
thread_two.join()
