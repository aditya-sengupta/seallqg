from matplotlib import pyplot as plt

from src import *

ol = np.load("data/tt_center_noise/tt_center_noise_nsteps_10000_delay_0.01_dt_21_07_2021_12.npy")
fs = 100
nsteps = 10000
ident = SystemIdentifier(ol, fs=fs)
klqg = ident.make_klqg_from_openloop()
x0 = 0.1 * np.hstack((np.array([0.0, 0.0]), np.array([1.0, 0.0, 1.0, 0.0])))
print("Improvement: ", klqg.improvement(x0=x0))
states_un = klqg.sim_process(nsteps=nsteps, x0=x0)
f_un, p_un = genpsd(states_un[:,0], dt=1/fs)
states = klqg.sim_control(nsteps=nsteps, x0=x0)
f_cl, p_cl = genpsd(states[:,0], dt=1/fs)
plt.loglog(f_un, p_un, label="Open-loop PSD")
plt.loglog(f_cl, p_cl, label="Closed-loop PSD")
plt.loglog(f_un, p_cl / p_un, label="Rejection TF")
plt.legend()
plt.show()
