from multiprocessing import Process
from time import monotonic_ns as mns

from sealrtc import spinlock_till, spinlock, spin

dt = 0.01
dur = 0.2

def act(message, delay, t0):
    spinlock_till(t0)
    t1 = mns()
    def iteration():
        print(f"{message} {(mns() - t1) / 1e9}")
        spinlock(delay)

    spin(iteration, dt, dur)

if __name__ == '__main__':
    t0 = mns()
    with Pool() as pool:
        pool.starmap(act, [("Dist", 0.001, t0), ("Loop", 0.005, t0)])

    """processes.append(Process(target=act, args=("Dist", 0.001, t0)))
    processes.append(Process(target=act, args=("Loop", 0.005, t0)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()"""
 
