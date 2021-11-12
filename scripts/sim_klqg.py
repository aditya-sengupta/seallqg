import numpy as np
from matplotlib import pyplot as plt
from src import joindata, SystemIdentifier, genpsd

np.random.seed(120)

ol = np.load(joindata("openloop", "ol_f_1_z_stamp_03_11_2021_13_58_53.npy"))
dmc2wf = np.load(joindata("bestflats", "lodmc2wfe.npy"))
ol *= dmc2wf
fs = 100
nsteps = 10000
ident = SystemIdentifier(ol, fs=fs, N_vib_max=1)
klqg = ident.make_klqg_from_openloop()
x0 = None #np.array([1.0, 0.0] * (klqg.state_size // 2))
improvement = klqg.improvement(x0=x0)
print(f"{improvement = }")
if improvement > 1.0 and False:
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
