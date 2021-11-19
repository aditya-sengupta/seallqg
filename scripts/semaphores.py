from asyncio import Task, Semaphore
import asyncio
from time import monotonic_ns as mns
from time import sleep
from math import ceil

dt = 0.1
dur = 1.0
t0 = mns()
get_time = lambda: (mns() - t0) / 1e9

def spinlock_till(t):
	"""
	Spin-locks to precisely sleep until mns is t
	"""
	i = 0
	while mns() < t:
		i += 1

async def spinlock(dur):
	"""
	Spin-locks to precisely sleep for time 'dur'
	"""
	spinlock_till(mns() + ceil(dur * 1e9))

def spin(process, dt, dur):
	"""
	Spin-locks around a process to do it every "dt" seconds for time "dur" seconds.
	"""
	t0 = mns()
	t1 = t0
	ticks_loop = ceil(dur * 1e9)
	ticks_inner = ceil(dt * 1e9)
	while mns() - t0 <= ticks_loop:
		process()
		spinlock_till(t1 + ticks_inner)
		t1 += ticks_inner

async def zeno(t):
	"""
	Drop-in replacement for asyncio.sleep
	"""
	t0 = mns()
	diff = t
	while diff > 0:
		asyncio.sleep(diff / 2)
		diff = ((t0 + t) - mns()) / 1e9

async def disturbance(sem: Semaphore):
	while True:
		async with sem:
			print("disturbance", get_time())
		await asyncio.sleep(dt)

async def loop(sem: Semaphore):
	while True:
		async with sem:
			print("loop       ", get_time())
		await asyncio.sleep(dt)
	
async def main():
	sem = Semaphore(value=1)
	for _coroutine in (disturbance, loop):
		asyncio.create_task(_coroutine(sem))
	
	await zeno(dur)

asyncio.run(main())
