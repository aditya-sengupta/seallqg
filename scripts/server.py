# Echo server program
import socket
from time import monotonic_ns as mns
import numpy as np

def spin(process, dt, dur):
	"""
	Spin-locks around a process to do it every "dt" seconds for time "dur" seconds.
	"""
	t0 = mns()
	t1 = t0
	ticks_loop = int(np.ceil(dur / 1e-9))
	ticks_inner = int(np.ceil(dt / 1e-9))
	while mns() - t0 <= ticks_loop:
		t1 += ticks_inner
		process()
		i = 0
		while mns() - t1 <= ticks_inner:
			i += 1

HOST = ''                 # Symbolic name meaning all available interfaces
PORT = 50008              # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(1)
conn, addr = s.accept()
with conn:
    def send_time():
        conn.sendall(mns().to_bytes(8, 'big'))
    spin(send_time, 0.01, 1.00)
    conn.sendall((0).to_bytes(8, 'big'))
s.close()