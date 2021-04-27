import numpy as np
import pandas as pd
from SSVD.SSVD_numba import SSVD_numba
from SSVD.get_plot import get_plot


df=pd.read_excel('BrainCancerData.xlsx')
labels=df.iloc[:,-1]
labels=np.array(labels)

df.drop('No.', inplace=True, axis=1)
df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)

df = np.array(df.iloc[0:350,:])
labels = labels[0:350]

u, s, v, niter = SSVD_numba(df, tol=1e-6, gamma1=2, gamma2=2, max_it = 100)

get_plot(u, s, v, labels,1300)
