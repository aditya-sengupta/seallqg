import numpy as np
from matplotlib import pyplot as plt
from sealrtc import joindata, SystemIdentifier, genpsd, ol, integ, rms

ol_values = np.load(joindata("openloop", "ol_f_1_z_stamp_03_11_2021_13_58_53.npy"))
dmc2wf = np.load(joindata("bestflats", "lodmc2wfe.npy"))
ol_values *= dmc2wf
fs = 100
nsteps = 10000
ident = SystemIdentifier(ol_values, fs=fs, N_vib_max=1)
klqg = ident.make_klqg_from_openloop()

states, states_un, states_in = klqg.simulate(ol, integ, nsteps=nsteps)
improvements = [rms(states_un) / rms(states), rms(states_in) / rms(states)]
print(f"{improvements = }")
assert improvements[1] > 1.0, "LQG loses to the integrator."
f_un, p_un = genpsd(states_un[:,0], dt=1/fs)
f_cl, p_cl = genpsd(states[:,0], dt=1/fs)

fig, axs = plt.subplots(1, 2, figsize=(10,6))
fig.suptitle("Simulated LQG control")
axs[0].plot(np.linalg.norm(states_un, axis=1), label="Open-loop, ")
axs[0].plot(np.linalg.norm(states_in, axis=1), label="Integrator")
axs[0].plot(np.linalg.norm(states, axis=1), label="LQG controller")
axs[0].set_xlabel("Timesteps")
axs[0].set_ylabel("RMS error (unknown units to be fixed)")
axs[0].legend()
axs[1].loglog(f_un, p_un, label="Open-loop PSD")
axs[1].loglog(f_cl, p_cl, label="Closed-loop PSD")
axs[1].loglog(f_un, p_cl / p_un, label="Rejection TF")
axs[1].set_title("Simulated LQG transfer functions")
axs[1].legend()
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel(r"Power (unknown $units^2/Hz$)")
plt.show()
