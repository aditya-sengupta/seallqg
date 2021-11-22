import numpy as np
from matplotlib import pyplot as plt
from sealrtc import loadres
from sealrtc import joindata

fs = 100

good_run = "2021_11_19_08_44_34"

res = loadres("lqg/klqg_nstate_18_amp_0.005_ang_0.7854_f_1_tstamp_2021_11_19_08_44_34.csv")
exposures = res.texp
measures = res.tmeas
dmcs = res.tdmc

nstart = 600
npoints = 30
tstart = exposures[nstart]
plt.figure(figsize=(8,6))
plt.scatter(exposures[nstart:nstart+npoints]-tstart, 1.5*np.ones_like(exposures[nstart:nstart+npoints]), label="exposures")
plt.scatter(measures[nstart:nstart+npoints]-tstart, np.ones_like(measures[nstart:nstart+npoints]), label="measurements")
plt.scatter(dmcs[nstart:nstart+npoints]-tstart, 0.5*np.ones_like(dmcs[nstart:nstart+npoints]), label="dmcs")
min_time = exposures[nstart]
max_time = dmcs[nstart+npoints-1]
for v in np.arange(exposures[nstart], exposures[nstart+npoints]+0.01, 0.01):
    plt.axvline(v-tstart, color='k')
plt.title("SEAL schedule observations")
plt.xlabel("Time (seconds)")
plt.yticks([])
plt.legend(bbox_to_anchor=(0.27, 0.2))
plt.savefig("seal_schedule.pdf", bbox_inches="tight")