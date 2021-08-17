import numpy as np
from scipy import linalg as la
from scipy import stats as st
from matplotlib import pyplot as plt

import sys
sys.path.append("..")
from src import *

inv = np.linalg.inv
mvn = st.multivariate_normal
rms = lambda data: round(np.sqrt(np.mean((data - np.mean(data)) ** 2)), 4)

ol = np.load("../data/sims/ol_atm_0_vib_2.npy")
ident = SystemIdentifier()
ident