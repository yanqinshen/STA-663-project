#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import linalg as la
import numba
from numba import jit
import warnings
warnings.filterwarnings('ignore')

@jit (nopython = True, parallel = True)

def SSVD_numba(X, tol=1e-6, gamma1=2, gamma2=2, max_it = 100):
    
    """
    Numba version of SSVD to obtain the first layer of matrix X
    
    X = matrix to be decomposed
    tol = stopping criteria for convergence
    gamma = weight parameters
    max_it = maximum # of iterations
    
    Return S,V,D and number of iterations
    """
    
    U, S, Vt = np.linalg.svd(X)
    n = X.shape[0]
    d = X.shape[1]
                    
    S = S[0]
    U = U[:,0].reshape((n,1))
    V = Vt[:,0].reshape((d,1))
                    
    SST = np.sum(X**2)
    
    num_it = 0
    V_delta, U_delta = 1000, 1000
    
    while (V_delta > tol) or (U_delta > tol) or (num_it > max_it):
        
        num_it += 1
        
        ### UPDATE V ###

        XTu = np.dot(X.T, U)
        '''Since zero cannot be raised to power, we will replace the zeros in XTu with very small numbers'''
        replaced_XTu = np.ravel(np.abs(XTu))
        nonzero_ind = (replaced_XTu > 1e-4)
        zero_ind = (replaced_XTu <= 1e-4)

        
        w2 = np.zeros(d)
        w2[nonzero_ind] = replaced_XTu[nonzero_ind] ** (-gamma2)
        w2[zero_ind] = np.Inf
        sig_sq = np.trace((X-S*np.outer(U,V)) @ (X-S*np.outer(U,V)).T)/(n*d-d)

        # calculate BIC and find best lambda for current V

        lambda2s = np.linspace(0, 12, 5000)[1:]
        BICs = np.zeros(len(lambda2s))
        X_hat = S * np.outer(U,V)

        for i in numba.prange(len(lambda2s)):
            lamb2 = lambda2s[i]

             #given current lambda, count df & calculate BIC
            dfV_hat = np.sum(np.abs(XTu) > lamb2 * w2/2) # number of rows that satify abs(XTu) > lamb2 * w2/2
            BICs[i] = np.sum((X - X_hat)**2) /(n*d * sig_sq) + np.log(n*d)/(n*d) * dfV_hat
        opt_lamb2 = lambda2s[np.argmin(BICs)]

         #for best lambda, calculate v_tilde

        V_tilde = np.zeros((d, 1))
        for j in numba.prange(d):
            XTu_j = XTu[j]
            V_tilde[j] = np.sign(XTu_j) * np.maximum(np.abs(XTu_j) - opt_lamb2 * w2[j]/2, 0)
    
         #new V

        V_new = V_tilde / np.sqrt(np.sum(V_tilde**2)) #max(np.sum(abs(V_tilde), axis=0))
        V_delta = np.sqrt(np.sum((V - V_new)**2))
        V = V_new

        
        ### UPDATE U ###

        Xv = np.dot(X, V)
        '''Since zero cannot be raised to power, we will replace the zeros in XTu with very small numbers'''
        replaced_Xv = np.ravel(np.abs(Xv))
        nonzero_ind = (replaced_Xv > 1e-4)
        zero_ind = (replaced_Xv <= 1e-4)
        
        w1 = np.zeros(n)
        w1[nonzero_ind] = replaced_Xv[nonzero_ind] ** (-gamma1)
        w1[zero_ind] = np.Inf

        # calculate BIC and find best lambda for current U

        lambda1s = np.linspace(0, 12, 5000)[1:]
        BICs = np.zeros(len(lambda1s))
        X_hat = S * np.outer(U,V)

        for i in numba.prange(len(lambda1s)):
            lamb1 = lambda1s[i]
    
            # given current lambda, count df & calculate BIC
            dfU_hat =  np.sum(np.abs(Xv) > lamb1 * w1/2)# number of rows that satify abs(Xv) > lamb1 * w1/2
            BICs[i] = np.sum((X - X_hat)**2) /(n*d * sig_sq) + np.log(n*d)/(n*d) * dfU_hat
        opt_lamb1 = lambda1s[np.argmin(BICs)]

        # for best lambda, calculate v_tilde

        U_tilde = np.zeros((n, 1))
        for j in numba.prange(n):
            Xv_j = Xv[j]
            U_tilde[j] = np.sign(Xv_j) * np.maximum(np.abs(Xv_j) - opt_lamb1 * w1[j]/2, 0)

        #new V

        U_new = U_tilde / np.sqrt(np.sum(U_tilde**2)) #max(np.sum(abs(U_tilde), axis=0))
        U_delta = np.sqrt(np.sum((U - U_new)**2))
        U = U_new
        
        
        
    S = U.T @ X @ V
    
    # return non-zero entries of the sparse matrices
    return (np.ravel(U),
            S,
            np.ravel(V),
            num_it)
