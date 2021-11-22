def make_optics():
	from socket import gethostname

	sim_mode = False
	if gethostname() == "SEAL" and not sim_mode:
		from .fast import FAST
		optics = FAST()
		mode = "hardware"
	else:
		from .sim import Sim
		optics = Sim()
		mode = "simulation"

	print(f"SEAL Real-Time Controller running in {mode} mode.")
	return optics

from .flatten import flatten
from .align import align
from .linearity import linearity, plot_linearity

__all__ = [
	"make_optics",
	"flatten",
	"align",
	"linearity",
	"plot_linearity",
]