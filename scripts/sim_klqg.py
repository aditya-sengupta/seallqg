from matplotlib import pyplot as plt
from src import *

ol = np.load(joindata("openloop", "ol_z_stamp_18_10_2021_08_40_07.npy"))
fs = 100
nsteps = 10000
ident = SystemIdentifier(ol[:, :2], fs=fs)
klqg = ident.make_klqg_from_openloop()
klqg.W[:2,:2] *= 1e3
klqg.Q[:4,:4] *= 1e4
x0 = np.array([1.0, 0.0] * (klqg.state_size // 2))
print(f"Improvement: {klqg.improvement(x0=x0)}")
states_un = klqg.sim_process(nsteps=nsteps, x0=x0)
f_un, p_un = genpsd(states_un[:,0], dt=1/fs)
states = klqg.sim_control(nsteps=nsteps, x0=x0)
f_cl, p_cl = genpsd(states[:,0], dt=1/fs)
plt.loglog(f_un, p_un, label="Open-loop PSD")
plt.loglog(f_cl, p_cl, label="Closed-loop PSD")
plt.loglog(f_un, p_cl / p_un, label="Rejection TF")
plt.legend()
plt.show()
