# authored by Aditya Sengupta

from tt import getdmc, applydmc, stack, np
# from compute_cmd_int import measure_tt

def refresh():
    bestflat = np.load("../data/bestflats/bestflat.npy")
    dmc = getdmc()
    applydmc(bestflat)
    imflat = stack(100)
    np.save("../data/bestflats/imflat.npy", imflat)
    print("Updated the flat image.")
    applydmc(dmc)
    return bestflat, imflat

if __name__ == "__main__":
    refresh()
