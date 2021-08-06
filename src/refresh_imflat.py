from tt import getdmc, applydmc, stack, np, getim
from compute_cmd_int import measure_tt

bestflat = np.load("../data/bestflats/bestflat.npy")
dmc = getdmc()
applydmc(bestflat)
imflat = stack(100)
np.save("../data/bestflats/imflat.npy", imflat)
print(measure_tt(getim() - imflat))
applydmc(dmc)
