import numpy as np
from matplotlib import pyplot as plt
from sealrtc import joindata, SystemIdentifier, genpsd, ol, integ

np.random.seed(120)

ol_values = np.load(joindata("openloop", "ol_f_1_z_stamp_03_11_2021_13_58_53.npy"))
dmc2wf = np.load(joindata("bestflats", "lodmc2wfe.npy"))
ol_values *= dmc2wf
fs = 100
nsteps = 10000
ident = SystemIdentifier(ol_values, fs=fs, N_vib_max=1)
klqg = ident.make_klqg_from_openloop()
improvement = klqg.improvement(ol, integ)
print(f"{improvement = }")
assert improvement[1] > 1.0, "LQG loses to the integrator."
states_un = klqg.sim_process(nsteps=nsteps, x0=x0)
f_un, p_un = genpsd(states_un[:,0], dt=1/fs)
states = klqg.sim_control(nsteps=nsteps, x0=x0)
f_cl, p_cl = genpsd(states[:,0], dt=1/fs)
plt.loglog(f_un, p_un, label="Open-loop PSD")
plt.loglog(f_cl, p_cl, label="Closed-loop PSD")
plt.loglog(f_un, p_cl / p_un, label="Rejection TF")
plt.title("Simulated results only!")
plt.legend()
plt.show()
