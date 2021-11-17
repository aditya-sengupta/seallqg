from sealrtc import spin
from threading import Thread
from time import sleep

fn_one = spin(lambda: print("thread one"), 0.1, 0.51)
fn_two = sleep(0.001) or spin(lambda: print("thread two"), 0.1, 0.51)

thread_one = Thread(target=fn_one)
thread_two = Thread(target=fn_two)

thread_one.start()
thread_two.start()

thread_one.join()
thread_two.join()
