from queue import Queue
from threading import Thread
from time import sleep

nmax = 10

def measure(out_q):
    for i in range(nmax):
        measurement = i
        print("Measurement " + str(i))
        out_q.put(measurement)

def control(in_q):
    for i in range(nmax):
        measurement = in_q.get()  
        print("Input " + str(i) + ": computing from measurement " +
        str(measurement))
        in_q.task_done()

def record():
    for i in range(nmax):
        print("Recorded frame " + str(i))

q = Queue()
t1 = Thread(target = measure, args=(q,))
t2 = Thread(target = control, args=(q,))
t3 = Thread(target = record)
t1.start()
t2.start()
t3.start()

q.join()
t3.join()

