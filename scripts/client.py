# Echo client program
import socket
from time import monotonic_ns as mns

HOST = 'localhost'   # The remote host
PORT = 50008       # The same port as used by the server
times = []
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
data = 1
while data != 0:
	times.append(data)
	data = int.from_bytes(s.recv(1024), 'big')	
s.close()	

dtimes = np.diff(times[1:]) / 1e9
print(round(np.mean(dtimes), 4), "+/-", round(np.std(dtimes), 4))
