import numpy as np
from matplotlib import pyplot as plt
from src import joindata, SystemIdentifier, genpsd
from src.controllers.dare import solve_dare

dmc2wf = np.load(joindata("bestflats", "lodmc2wfe.npy"))
ol = np.load(joindata("openloop", "ol_z_stamp_18_10_2021_08_40_07.npy"))[:10000,:2] * dmc2wf
#ol = np.load(joindata("openloop", "ol_tt_stamp_21_08_2021_08_56_31.npy"))

fs = 100
nsteps = 10000
ident = SystemIdentifier(ol, fs=fs)
klqg = ident.make_klqg_from_openloop()
x0 = None #np.array([1.0, 0.0] * (klqg.state_size // 2))
Ppr = solve_dare(A.T, C.T * 0, W, V)
print(f"Uncontrolled process covariance: {C @ Ppr @ C.T}")
print(f"Controlled process covariance: {C @ klqg.Pcon @ C.T}")
improvement = klqg.improvement(x0=x0)
print(f"Improvement: {improvement}")
if improvement > 1.0 and False:
    states_un = klqg.sim_process(nsteps=nsteps, x0=x0)
    f_un, p_un = genpsd(states_un[:,0], dt=1/fs)
    states = klqg.sim_control(nsteps=nsteps, x0=x0)
    f_cl, p_cl = genpsd(states[:,0], dt=1/fs)
    plt.loglog(f_un, p_un, label="Open-loop PSD")
    plt.loglog(f_cl, p_cl, label="Closed-loop PSD")
    plt.loglog(f_un, p_cl / p_un, label="Rejection TF")
    plt.legend()
    plt.show()
