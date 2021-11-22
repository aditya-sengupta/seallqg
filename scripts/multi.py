from multiprocessing import Process
from time import monotonic_ns as mns

from sealrtc import spinlock_till, spinlock, spin

dt = 0.01
dur = 0.2

def act(message, delay, t0, go_first):
    # if I put the tick handling thing in here, it simulated a perfect one frame delay

    if go_first:    
        spinlock_till(t0)
    else:
        spinlock_till(t0+1e3)

    def iteration():
        print(f"{message} {(mns() - t0) / 1e9}")
        spinlock(delay)

    spin(iteration, dt, dur)

if __name__ == '__main__':
    t0 = mns()
    with Pool() as pool:
        pool.starmap(act, [("Dist", 0.001, t0), ("Loop", 0.005, t0)])

    """processes.append(Process(target=act, args=("Dist", 0.006, t0, True)))
    processes.append(Process(target=act, args=("Loop", 0.005, t0, False)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()"""
 
