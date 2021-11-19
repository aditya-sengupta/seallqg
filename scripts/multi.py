from multiprocessing import Process
from time import monotonic_ns as mns

from sealrtc import spinlock_till, spinlock, spin

dt = 0.01
dur = 0.2

def act(message, delay, t0):
    spinlock_till(t0)
    def iteration():
        print(f"{message} {(mns() - t0) / 1e9}")
        spinlock(delay)

    
    spin(iteration, dt, dur)

if __name__ == '__main__':
    processes = []
    t0 = mns()

    processes.append(Process(target=act, args=("Dist", 0.006, t0)))
    processes.append(Process(target=act, args=("Loop", 0.005, t0+1e6)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()
 
