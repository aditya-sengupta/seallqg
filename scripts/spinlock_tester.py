from sealrtc import spin
from threading import Thread
from time import sleep

fn_one = lambda: spin(lambda: print("thread one"), 0.1, 0.51)
fn_two = lambda: spin(lambda: print("thread two"), 0.1, 0.51)

thread_one = Thread(target=fn_one)
thread_two = Thread(target=fn_two)

thread_one.start()
thread_two.start()

thread_one.join()
thread_two.join()
