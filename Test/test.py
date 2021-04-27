import numpy as np
from SSVD.SSVD_numba import SSVD_numba
from SSVD.get_plot import get_plot

u_tilde = [[10],[9],[8],[7],[6],[5],[4],[3], [2]*17, [0]*75]
u_tilde = np.array(sum(u_tilde, [])).reshape((-1,1))
u_true = u_tilde / np.linalg.norm(u_tilde)

v_tilde = [[10],[-10],[8],[-8],[5],[-5],[3]*5,[-3]*5,[0]*34]
v_tilde = np.array(sum(v_tilde, [])).reshape((-1,1))
v_true = v_tilde / np.linalg.norm(v_tilde)

s_true = 50

X_star = s_true*u_true@v_true.T

#construct X with noise added
err = np.random.normal(0, 1, size=X_star.shape)

X = X_star + err

u, s, v, niter = SSVD_numba(X, tol=1e-6, gamma1=2, gamma2=2, max_it = 10)

clusters = [[1]*10,[2]*15,[3]*75]
clusters = np.array(sum(clusters, []))

get_plot(u, s, v, clusters)
