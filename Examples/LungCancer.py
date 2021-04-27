import numpy as np
import pandas as pd
from SSVD.SSVD_numba import SSVD_numba
from SSVD.get_plot import get_plot

LungData = pd.read_csv('LungCancerData.txt', sep=' ')
LungData = np.array(LungData.T) # 56 subjects
clusters = [[1]*20,[2]*(33-20),[3]*(50-33),[4]*(56-50)]
clusters = np.array(sum(clusters, []))

u, s, v, niter = SSVD_numba(LungData, tol=1e-6, gamma1=2, gamma2=2, max_it = 100)

get_plot(u, s, v, clusters,8000)
