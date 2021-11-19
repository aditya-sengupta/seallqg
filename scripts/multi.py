from multiprocessing import Process, Lock
from time import monotonic_ns as mns
from math import ceil
from functools import partial

from sealrtc import spinlock_till, spinlock, spin

dt = 0.01
dur = 0.2

def act(lock, message, delay, t0):
    spinlock_till(t0)
    def iteration():
        lock.acquire()
        try:
            print(f"{message} {(mns() - t0) / 1e9}")
            spinlock(delay)
        finally:
            lock.release()
    
    spin(iteration, dt, dur)

if __name__ == '__main__':
    lock = Lock()

    processes = []
    t0 = mns()
    processes.append(Process(target=act, args=(lock, "Dist", 0.001, t0)))
    processes.append(Process(target=act, args=(lock, "Loop", 0.005, t0+1e6)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()
 